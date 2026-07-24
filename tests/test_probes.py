import torch

from src.probes import compose_probe, fit_linear_probes


def _separable_data(n=200, d=8, a=3, seed=0):
    # Each attribute is linearly separable along its own coordinate axis.
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=gen)
    labels = (x[:, :a] > 0).long()
    return x / x.norm(dim=-1, keepdim=True), labels


def test_fit_linear_probes_shapes_and_types():
    x, y = _separable_data()
    w, b = fit_linear_probes(x, y, epochs=50)
    assert w.shape == (3, 8)
    assert b.shape == (3,)
    assert not w.requires_grad


def test_fit_linear_probes_separates_separable_data():
    x, y = _separable_data()
    w, b = fit_linear_probes(x, y, epochs=300)
    preds = (x @ w.T + b) > 0
    acc = (preds == y.bool()).float().mean()
    assert acc > 0.95


def test_fit_linear_probes_is_deterministic():
    x, y = _separable_data()
    w1, _ = fit_linear_probes(x, y, epochs=20)
    w2, _ = fit_linear_probes(x, y, epochs=20)
    assert torch.equal(w1, w2)


def test_compose_probe_moves_query_toward_positive_direction():
    v_ref = torch.tensor([1.0, 0.0, 0.0, 0.0])
    directions = torch.eye(4)
    q = compose_probe(v_ref, directions[1:2], directions[2:3], gamma=1.0)
    assert torch.allclose(q.norm(), torch.tensor(1.0), atol=1e-5)
    assert q[1] > 0    # positive constraint direction added
    assert q[2] < 0    # negative constraint direction subtracted
    assert q[0] > 0    # reference identity kept


def test_compose_probe_gamma_scales_identity():
    v_ref = torch.tensor([1.0, 0.0])
    pos = torch.tensor([[0.0, 1.0]])
    none = torch.zeros((0, 2))
    q_lo = compose_probe(v_ref, pos, none, gamma=0.1)
    q_hi = compose_probe(v_ref, pos, none, gamma=10.0)
    assert q_hi[0] > q_lo[0]


def test_compose_probe_no_constraints_returns_reference():
    v_ref = torch.tensor([3.0, 4.0])
    none = torch.zeros((0, 2))
    q = compose_probe(v_ref, none, none, gamma=1.0)
    assert torch.allclose(q, v_ref / v_ref.norm())
