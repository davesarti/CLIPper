import pytest
import torch

from src.cpas import CPAS
from src.cpas_mlp import PerAttributeMLP
from src.probes import compose_probe
from src.steering import compose, pad_queries

ARCHITECTURES = [
    pytest.param(lambda **kw: CPAS(d=16, heads=2, ffn=32, **kw), id="cpas"),
    pytest.param(
        lambda **kw: PerAttributeMLP(d=16, hidden=8, sign_dim=4, rank=4, **kw),
        id="mlp",
    ),
]


def _batch(b=3, k=2, d=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    v_ref = torch.nn.functional.normalize(torch.randn(b, d, generator=gen), dim=-1)
    dirs = torch.nn.functional.normalize(torch.randn(b, k, d, generator=gen), dim=-1)
    signs = torch.tensor([[1.0, -1.0]] * b)
    mask = torch.ones(b, k, dtype=torch.bool)
    return v_ref, dirs, signs, mask


def test_compose_at_baseline_values_matches_compose_probe():
    """gamma=0.6, alpha=1, delta=0 is exactly the fixed probe rule."""
    v_ref, dirs, signs, _ = _batch()
    b, k, _ = dirs.shape
    q = compose(
        v_ref, dirs, signs,
        gamma=torch.full((b,), 0.6),
        alpha=torch.ones(b, k),
        delta=torch.zeros_like(dirs),
    )
    expected = torch.stack([
        compose_probe(v_ref[i], dirs[i, :1], dirs[i, 1:], gamma=0.6)
        for i in range(b)
    ])
    assert torch.allclose(q, expected, atol=1e-5)


def test_compose_output_is_normalized():
    v_ref, dirs, signs, _ = _batch()
    b, k, _ = dirs.shape
    q = compose(
        v_ref, dirs, signs,
        gamma=torch.rand(b),
        alpha=torch.rand(b, k),
        delta=0.1 * torch.randn_like(dirs),
    )
    assert torch.allclose(q.norm(dim=-1), torch.ones(b), atol=1e-5)


@pytest.mark.parametrize("build", ARCHITECTURES)
def test_untrained_model_reproduces_probe_composition(build):
    v_ref, dirs, signs, mask = _batch()
    model = build(gamma_init=0.6).eval()
    with torch.no_grad():
        q = model(v_ref, dirs, signs, mask)
    expected = torch.stack([
        compose_probe(v_ref[i], dirs[i, :1], dirs[i, 1:], gamma=0.6)
        for i in range(v_ref.shape[0])
    ])
    assert torch.allclose(q, expected, atol=1e-5)


@pytest.mark.parametrize("build", ARCHITECTURES)
def test_output_is_normalized(build):
    v_ref, dirs, signs, mask = _batch()
    q = build()(v_ref, dirs, signs, mask)
    assert torch.allclose(q.norm(dim=-1), torch.ones(q.shape[0]), atol=1e-5)


@pytest.mark.parametrize("build", ARCHITECTURES)
def test_padded_slots_do_not_change_the_query(build):
    v_ref, dirs, signs, mask = _batch(k=2)
    model = build().eval()
    pad_dirs = torch.cat([dirs, torch.randn(dirs.shape[0], 1, dirs.shape[2])], dim=1)
    pad_signs = torch.cat([signs, torch.ones(signs.shape[0], 1)], dim=1)
    pad_mask = torch.cat([mask, torch.zeros(mask.shape[0], 1, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        assert torch.allclose(
            model(v_ref, dirs, signs, mask),
            model(v_ref, pad_dirs, pad_signs, pad_mask),
            atol=1e-5,
        )


@pytest.mark.parametrize("build", ARCHITECTURES)
def test_attribute_order_does_not_change_the_query(build):
    v_ref, dirs, signs, mask = _batch(k=2)
    model = build().eval()
    flip = [1, 0]
    with torch.no_grad():
        a = model(v_ref, dirs, signs, mask)
        b = model(v_ref, dirs[:, flip], signs[:, flip], mask[:, flip])
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("build", ARCHITECTURES)
def test_steer_masks_padded_predictions(build):
    v_ref, dirs, signs, mask = _batch(k=2)
    mask[:, 1] = False
    _, alpha, delta = build().steer(v_ref, dirs, signs, mask)
    assert torch.all(alpha[:, 1] == 0)
    assert torch.all(delta[:, 1] == 0)


@pytest.mark.parametrize("build", ARCHITECTURES)
def test_delta_max_zero_keeps_the_probe_directions(build):
    v_ref, dirs, signs, mask = _batch()
    _, _, delta = build(delta_max=0.0).steer(v_ref, dirs, signs, mask)
    assert torch.all(delta == 0)


@pytest.mark.parametrize("build", ARCHITECTURES)
def test_training_step_changes_the_query(build):
    v_ref, dirs, signs, mask = _batch()
    model = build()
    target = torch.nn.functional.normalize(torch.randn_like(v_ref), dim=-1)
    before = model(v_ref, dirs, signs, mask).detach()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(10):
        opt.zero_grad()
        loss = (1 - (model(v_ref, dirs, signs, mask) * target).sum(-1)).mean()
        loss.backward()
        opt.step()
    after = model(v_ref, dirs, signs, mask).detach()
    assert not torch.allclose(before, after, atol=1e-4)
    assert (after * target).sum(-1).mean() > (before * target).sum(-1).mean()


def test_pad_queries_builds_signs_and_mask():
    directions = torch.eye(5)
    dirs, signs, mask = pad_queries([([0], [1, 2]), ([3], [])], directions)
    assert dirs.shape == (2, 3, 5)
    assert torch.equal(signs[0], torch.tensor([1.0, -1.0, -1.0]))
    assert torch.equal(mask[1], torch.tensor([True, False, False]))
    assert torch.equal(dirs[1, 0], directions[3])
