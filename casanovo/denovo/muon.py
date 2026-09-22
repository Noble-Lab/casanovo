"""Muon optimizer with an auxiliary AdamW for non-matrix parameters.

Single-device implementation following Keller Jordan's reference
(https://github.com/KellerJordan/Muon; MIT-licensed) and Liu et al.
2025, "Muon is Scalable for LLM Training" (arXiv:2502.16982).
"""

from typing import Any, Callable, Dict, List, Optional

import torch


def _zeropower_via_newtonschulz5(
    G: torch.Tensor, steps: int = 5
) -> torch.Tensor:
    """
    Approximately orthogonalize a matrix via Newton-Schulz iteration.

    Uses the quintic iteration and coefficients from the reference Muon
    implementation, tuned for convergence speed rather than exact
    orthogonality: the singular values of the result land roughly in
    [0.7, 1.2], which empirically does not hurt model quality.

    Parameters
    ----------
    G : torch.Tensor of shape (m, n)
        The matrix to orthogonalize.
    steps : int
        The number of Newton-Schulz iterations.

    Returns
    -------
    torch.Tensor of shape (m, n)
        The orthogonalized matrix, in ``G``'s dtype.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    # Ensure the spectral norm is at most 1.
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton-Schulz) plus AdamW.

    Parameter groups with ``use_muon=True`` are updated with Muon:
    Nesterov-momentum SGD whose per-matrix update is orthogonalized with
    Newton-Schulz and rescaled by ``sqrt(max(1, m / n))``. These groups
    must contain only parameters with >= 2 dimensions (hidden weight
    matrices). Groups with ``use_muon=False`` are updated with standard
    decoupled-weight-decay AdamW; embeddings, the output head, biases,
    and normalization parameters belong there, per the Muon usage
    guidance.

    Both algorithms live in one optimizer object so that a single LR
    scheduler (e.g. ``CosineWarmupScheduler``) drives the whole model:
    the scheduler rescales each group's base learning rate by the same
    factor.

    Parameters
    ----------
    param_groups : List[Dict[str, Any]]
        Parameter groups, each with a ``use_muon`` key. Muon groups
        accept ``lr`` (default 0.02), ``momentum`` (default 0.95),
        ``weight_decay`` (default 0.0), and ``ns_steps`` (default 5).
        AdamW groups accept ``lr`` (default 5e-4), ``betas`` (default
        ``(0.9, 0.999)``), ``eps`` (default 1e-8), and ``weight_decay``
        (default 0.0).
    """

    def __init__(self, param_groups: List[Dict[str, Any]]):
        for group in param_groups:
            group["use_muon"] = bool(group.get("use_muon", False))
            if group["use_muon"]:
                for p in group["params"]:
                    if p.ndim < 2:
                        raise ValueError(
                            "Muon parameter groups must contain only "
                            f"matrices; got a parameter with {p.ndim} "
                            "dimension(s)."
                        )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
            else:
                group.setdefault("lr", 5e-4)
                group.setdefault("betas", (0.9, 0.999))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: Dict[str, Any]) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.ndim > 2:
                # E.g. conv filters: flatten the trailing dimensions.
                g = g.view(g.size(0), -1)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.lerp_(g, 1 - group["momentum"])
            g = g.lerp(buf, group["momentum"])  # Nesterov-style blend.
            update = _zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
            # Match the update RMS across matrix shapes.
            update *= max(1.0, g.size(0) / g.size(1)) ** 0.5
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update.view_as(p), alpha=-group["lr"])

    def _adamw_step(self, group: Dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "exp_avg" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(p.grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            denom = (exp_avg_sq / (1 - beta2**t)).sqrt().add_(group["eps"])
            update = (exp_avg / (1 - beta1**t)) / denom
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update, alpha=-group["lr"])
