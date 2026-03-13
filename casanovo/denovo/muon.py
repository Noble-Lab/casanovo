"""Muon optimizer: MomentUm Orthogonalized by Newton-Schulz.

Reference: https://kellerjordan.github.io/posts/muon/
"""

import torch


def newtonschulz5(G, steps=5, eps=1e-7):
    """Approximately orthogonalize G via Newton-Schulz iteration.

    Computes the nearest semi-orthogonal matrix to G by iterating
    the quintic polynomial X <- aX + bAX + cAAX, where A = XX^T.
    Runs in bfloat16 for numerical stability, then casts back.

    Parameters
    ----------
    G : torch.Tensor of shape (m, n)
        The 2-D gradient matrix to orthogonalize.
    steps : int, optional
        Number of Newton-Schulz iterations (default: 5).
    eps : float, optional
        Small constant added to the norm to avoid division by zero.

    Returns
    -------
    torch.Tensor of shape (m, n)
        Approximately orthogonalized matrix in the original dtype.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2-D hidden-layer weight matrices.

    Applies SGD-momentum updates followed by Newton-Schulz
    orthogonalization. Embeddings, biases, and classifier heads
    should use a standard optimizer (e.g. Adam) instead.

    Parameters
    ----------
    params : iterable
        Iterable of 2-D ``torch.Tensor`` parameters to optimize.
    lr : float, optional
        Learning rate (default: 0.02).
    momentum : float, optional
        SGD momentum coefficient (default: 0.95).
    nesterov : bool, optional
        Use Nesterov-style momentum (default: True).
    ns_steps : int, optional
        Number of Newton-Schulz iterations (default: 5).
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    ):
        """Initialize Muon."""
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Perform a single Muon optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon requires 2-D parameters; got shape {p.shape}"
                    )
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf
                p.add_(newtonschulz5(g, steps=ns_steps), alpha=-lr)
