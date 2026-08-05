import torch

from src.cpas_mlp import PerAttributeMLP

DEFAULT_PARAMS = 502_690


def _batch(b=3, k=2, d=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    v_ref = torch.nn.functional.normalize(torch.randn(b, d, generator=gen), dim=-1)
    dirs = torch.nn.functional.normalize(torch.randn(b, k, d, generator=gen), dim=-1)
    signs = torch.tensor([[1.0, -1.0]] * b)[:, :k]
    mask = torch.ones(b, k, dtype=torch.bool)
    return v_ref, dirs, signs, mask


def _trained_like(model):
    """Undo the baseline-equivalent zero init so the heads actually respond."""
    for layer in (model.head_gamma, model.head_alpha, model.delta_up):
        torch.nn.init.normal_(layer.weight, std=0.05)
    return model.eval()


def _pair(**kwargs):
    """A cross and a no-cross model sharing identical weights."""
    cross = _trained_like(PerAttributeMLP(cross_attributes=True, **kwargs))
    nocross = PerAttributeMLP(cross_attributes=False, **kwargs)
    nocross.load_state_dict(cross.state_dict())
    return cross, nocross.eval()


def test_default_parameter_count():
    """The headline claim of the design: 0.50M against CPAS's 2.4M."""
    model = PerAttributeMLP()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == DEFAULT_PARAMS


def test_disabling_the_context_does_not_change_the_parameter_count():
    """--no-cross zeroes c_a; it does not delete MLP2. Equal parameters are
    what make it an exact ablation rather than a different model."""
    live = sum(p.numel() for p in PerAttributeMLP().parameters())
    off = sum(p.numel() for p in PerAttributeMLP(cross_attributes=False).parameters())
    assert live == off


def test_single_attribute_queries_ignore_the_context():
    """Exclude-self pooling: with K=1 there is no other attribute, so cross and
    no-cross are identical by construction on the 8 single-attribute benchmark
    queries. Any measured difference must come from the multi-attribute ones."""
    v_ref, dirs, signs, mask = _batch(k=1, d=16)
    cross, nocross = _pair(d=16, hidden=8, sign_dim=4, rank=4)
    with torch.no_grad():
        assert torch.allclose(
            cross(v_ref, dirs, signs, mask),
            nocross(v_ref, dirs, signs, mask),
            atol=1e-6,
        )


def test_multi_attribute_queries_use_the_context():
    v_ref, dirs, signs, mask = _batch(k=2, d=16)
    cross, nocross = _pair(d=16, hidden=8, sign_dim=4, rank=4)
    with torch.no_grad():
        assert not torch.allclose(
            cross(v_ref, dirs, signs, mask),
            nocross(v_ref, dirs, signs, mask),
            atol=1e-6,
        )


def test_delta_is_low_rank():
    """The bend is confined to a rank-r subspace - the regularizer on the one
    component docs/method-proposal-cpas.md flags as over-free (cos = 0.49)."""
    model = _trained_like(PerAttributeMLP(d=16, hidden=8, sign_dim=4, rank=4))
    assert model.delta_up.weight.shape == (16, 4)
    assert torch.linalg.matrix_rank(model.delta_up.weight) <= 4


def test_context_excludes_padded_slots():
    """A padded slot must not leak into any real attribute's context."""
    v_ref, dirs, signs, mask = _batch(k=2, d=16)
    model = _trained_like(PerAttributeMLP(d=16, hidden=8, sign_dim=4, rank=4))
    pad_dirs = torch.cat([dirs, torch.randn(dirs.shape[0], 1, 16)], dim=1)
    pad_signs = torch.cat([signs, torch.ones(signs.shape[0], 1)], dim=1)
    pad_mask = torch.cat([mask, torch.zeros(mask.shape[0], 1, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        _, alpha, delta = model.steer(v_ref, dirs, signs, mask)
        _, alpha_pad, delta_pad = model.steer(v_ref, pad_dirs, pad_signs, pad_mask)
    assert torch.allclose(alpha, alpha_pad[:, :2], atol=1e-6)
    assert torch.allclose(delta, delta_pad[:, :2], atol=1e-6)
