"""Unit tests for the Muon optimizer (casanovo.denovo.muon)."""

import pytest
import torch

from casanovo.denovo.muon import Muon, newtonschulz5


# ---------------------------------------------------------------------------
# newtonschulz5
# ---------------------------------------------------------------------------


def test_newtonschulz5_square_shape():
    """Output shape matches input for a square matrix."""
    G = torch.randn(8, 8)
    assert newtonschulz5(G).shape == G.shape


def test_newtonschulz5_tall_shape():
    """Output shape matches input for a tall (rows > cols) matrix."""
    G = torch.randn(16, 8)
    assert newtonschulz5(G).shape == G.shape


def test_newtonschulz5_wide_shape():
    """Output shape matches input for a wide (cols > rows) matrix."""
    G = torch.randn(8, 16)
    assert newtonschulz5(G).shape == G.shape


def test_newtonschulz5_approximate_orthogonality():
    """Output should be more orthogonal than the normalized input.

    Newton-Schulz in bfloat16 is approximate; we check that X^T X is
    closer to the identity than the raw (column-normalized) input is.
    """
    torch.manual_seed(0)
    G = torch.randn(16, 8)
    X = newtonschulz5(G)
    eye = torch.eye(8)
    # raw column-normalized input
    G_norm = G / G.norm(dim=0, keepdim=True)
    input_err = (G_norm.T @ G_norm - eye).abs().max().item()
    output_err = (X.T @ X - eye).abs().max().item()
    assert output_err < input_err, (
        f"output_err={output_err:.4f} not less than input_err={input_err:.4f}"
    )


def test_newtonschulz5_dtype_preserved():
    """Output dtype matches input dtype."""
    G = torch.randn(8, 8, dtype=torch.float32)
    assert newtonschulz5(G).dtype == torch.float32


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_param():
    """A single 2-D parameter with a fixed gradient."""
    p = torch.nn.Parameter(torch.ones(4, 4))
    p.grad = torch.ones(4, 4)
    return p


def test_muon_step_updates_param(simple_param):
    """A Muon step should change the parameter values."""
    original = simple_param.data.clone()
    opt = Muon([simple_param], lr=0.02, momentum=0.95)
    opt.step()
    assert not torch.equal(simple_param.data, original)


def test_muon_momentum_buffer_initialized(simple_param):
    """After the first step the momentum buffer should exist."""
    opt = Muon([simple_param], lr=0.02, momentum=0.95)
    opt.step()
    assert "momentum_buffer" in opt.state[simple_param]


def test_muon_momentum_buffer_accumulates(simple_param):
    """The momentum buffer should change between steps."""
    opt = Muon([simple_param], lr=0.02, momentum=0.95)
    opt.step()
    buf_after_1 = opt.state[simple_param]["momentum_buffer"].clone()
    simple_param.grad = torch.ones_like(simple_param)
    opt.step()
    buf_after_2 = opt.state[simple_param]["momentum_buffer"].clone()
    assert not torch.equal(buf_after_1, buf_after_2)


def test_muon_rejects_1d_param():
    """Muon should raise ValueError for non-2-D parameters."""
    p = torch.nn.Parameter(torch.ones(4))
    p.grad = torch.ones(4)
    opt = Muon([p], lr=0.02)
    with pytest.raises(ValueError, match="2-D"):
        opt.step()


def test_muon_skips_none_grad():
    """Parameters without gradients should be silently skipped."""
    p = torch.nn.Parameter(torch.ones(4, 4))
    # p.grad is None by default
    opt = Muon([p], lr=0.02)
    original = p.data.clone()
    opt.step()
    assert torch.equal(p.data, original)


# ---------------------------------------------------------------------------
# configure_optimizers parameter splitting
# ---------------------------------------------------------------------------


def _make_tiny_model(use_muon=True):
    """Return a minimal Spec2Pep instance suitable for optimizer tests."""
    from casanovo.denovo.model import Spec2Pep

    return Spec2Pep(
        dim_model=32,
        n_head=2,
        dim_feedforward=16,
        n_layers=1,
        use_muon=use_muon,
        muon_lr=0.02,
        muon_momentum=0.95,
        warmup_iters=1,
        cosine_schedule_period_iters=1,
        lr=5e-4,
        weight_decay=1e-5,
    )


def test_configure_optimizers_muon_off_returns_one_optimizer():
    """With use_muon=False, a single Adam optimizer is returned."""
    model = _make_tiny_model(use_muon=False)
    result = model.configure_optimizers()
    optimizers, _ = result
    assert len(optimizers) == 1
    assert isinstance(optimizers[0], torch.optim.Adam)


def test_configure_optimizers_muon_on_returns_two_optimizers():
    """With use_muon=True, both Muon and Adam optimizers are returned."""
    model = _make_tiny_model(use_muon=True)
    optimizers, _ = model.configure_optimizers()
    assert len(optimizers) == 2
    assert isinstance(optimizers[0], Muon)
    assert isinstance(optimizers[1], torch.optim.Adam)


def test_muon_params_are_all_2d():
    """Every parameter assigned to Muon must be 2-D."""
    model = _make_tiny_model(use_muon=True)
    optimizers, _ = model.configure_optimizers()
    muon_opt = optimizers[0]
    for group in muon_opt.param_groups:
        for p in group["params"]:
            assert p.ndim == 2, f"Non-2D param in Muon group: shape {p.shape}"


def test_param_groups_cover_all_parameters():
    """Muon + Adam param groups should cover every trainable parameter."""
    model = _make_tiny_model(use_muon=True)
    optimizers, _ = model.configure_optimizers()
    opt_param_ids = set()
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                opt_param_ids.add(id(p))
    all_param_ids = {
        id(p) for p in model.parameters() if p.requires_grad
    }
    assert opt_param_ids == all_param_ids


def test_param_groups_have_no_overlap():
    """No parameter should appear in both the Muon and Adam groups."""
    model = _make_tiny_model(use_muon=True)
    optimizers, _ = model.configure_optimizers()
    muon_ids = {
        id(p)
        for group in optimizers[0].param_groups
        for p in group["params"]
    }
    adam_ids = {
        id(p)
        for group in optimizers[1].param_groups
        for p in group["params"]
    }
    assert muon_ids.isdisjoint(adam_ids)
