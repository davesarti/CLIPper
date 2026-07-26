import torch

from src.cpas import CPAS, pad_queries
from src.probes import compose_probe


def _batch(b=3, k=2, d=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    v_ref = torch.nn.functional.normalize(torch.randn(b, d, generator=gen), dim=-1)
    dirs = torch.nn.functional.normalize(torch.randn(b, k, d, generator=gen), dim=-1)
    signs = torch.tensor([[1.0, -1.0]] * b)
    mask = torch.ones(b, k, dtype=torch.bool)
    return v_ref, dirs, signs, mask


def test_untrained_model_reproduces_probe_composition():
    v_ref, dirs, signs, mask = _batch()
    model = CPAS(d=16, heads=2, ffn=32, gamma_init=0.6).eval()
    with torch.no_grad():
        q = model(v_ref, dirs, signs, mask)
    expected = torch.stack([
        compose_probe(v_ref[i], dirs[i, :1], dirs[i, 1:], gamma=0.6)
        for i in range(v_ref.shape[0])
    ])
    assert torch.allclose(q, expected, atol=1e-5)


def test_output_is_normalized():
    v_ref, dirs, signs, mask = _batch()
    model = CPAS(d=16, heads=2, ffn=32)
    q = model(v_ref, dirs, signs, mask)
    assert torch.allclose(q.norm(dim=-1), torch.ones(q.shape[0]), atol=1e-5)


def test_padded_slots_do_not_change_the_query():
    v_ref, dirs, signs, mask = _batch(k=2)
    model = CPAS(d=16, heads=2, ffn=32).eval()
    pad_dirs = torch.cat([dirs, torch.randn(dirs.shape[0], 1, dirs.shape[2])], dim=1)
    pad_signs = torch.cat([signs, torch.ones(signs.shape[0], 1)], dim=1)
    pad_mask = torch.cat([mask, torch.zeros(mask.shape[0], 1, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        assert torch.allclose(
            model(v_ref, dirs, signs, mask),
            model(v_ref, pad_dirs, pad_signs, pad_mask),
            atol=1e-5,
        )


def test_attribute_order_does_not_change_the_query():
    v_ref, dirs, signs, mask = _batch(k=2)
    model = CPAS(d=16, heads=2, ffn=32).eval()
    flip = [1, 0]
    with torch.no_grad():
        a = model(v_ref, dirs, signs, mask)
        b = model(v_ref, dirs[:, flip], signs[:, flip], mask[:, flip])
    assert torch.allclose(a, b, atol=1e-5)


def test_steer_masks_padded_predictions():
    v_ref, dirs, signs, mask = _batch(k=2)
    mask[:, 1] = False
    model = CPAS(d=16, heads=2, ffn=32)
    _, alpha, delta = model.steer(v_ref, dirs, signs, mask)
    assert torch.all(alpha[:, 1] == 0)
    assert torch.all(delta[:, 1] == 0)


def test_training_step_changes_the_query():
    v_ref, dirs, signs, mask = _batch()
    model = CPAS(d=16, heads=2, ffn=32)
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


def _trained_like(model):
    """Undo the baseline-equivalent zero init so the heads actually respond."""
    for head in (model.head_gamma, model.head_alpha, model.head_delta):
        torch.nn.init.normal_(head.weight, std=0.05)
    return model.eval()


def test_cross_attention_lets_one_attribute_change_another_step():
    v_ref, dirs, signs, mask = _batch(k=2)
    model = _trained_like(CPAS(d=16, heads=2, ffn=32))
    swapped = dirs.clone()
    swapped[:, 1] = torch.nn.functional.normalize(torch.randn(dirs.shape[0], 16), dim=-1)
    with torch.no_grad():
        _, alpha, _ = model.steer(v_ref, dirs, signs, mask)
        _, alpha_other, _ = model.steer(v_ref, swapped, signs, mask)
    # Slot 0 is unchanged; its step still reacts to what is queried beside it.
    assert not torch.allclose(alpha[:, 0], alpha_other[:, 0], atol=1e-6)


def test_without_cross_attention_steps_ignore_the_other_attributes():
    v_ref, dirs, signs, mask = _batch(k=2)
    model = _trained_like(CPAS(d=16, heads=2, ffn=32, cross_attributes=False))
    swapped = dirs.clone()
    swapped[:, 1] = torch.nn.functional.normalize(torch.randn(dirs.shape[0], 16), dim=-1)
    with torch.no_grad():
        _, alpha, delta = model.steer(v_ref, dirs, signs, mask)
        _, alpha_other, delta_other = model.steer(v_ref, swapped, signs, mask)
    assert torch.allclose(alpha[:, 0], alpha_other[:, 0], atol=1e-6)
    assert torch.allclose(delta[:, 0], delta_other[:, 0], atol=1e-6)


def test_delta_max_zero_keeps_the_probe_directions():
    v_ref, dirs, signs, mask = _batch()
    model = CPAS(d=16, heads=2, ffn=32, delta_max=0.0)
    _, _, delta = model.steer(v_ref, dirs, signs, mask)
    assert torch.all(delta == 0)


def test_pad_queries_builds_signs_and_mask():
    directions = torch.eye(5)
    dirs, signs, mask = pad_queries([([0], [1, 2]), ([3], [])], directions)
    assert dirs.shape == (2, 3, 5)
    assert torch.equal(signs[0], torch.tensor([1.0, -1.0, -1.0]))
    assert torch.equal(mask[1], torch.tensor([True, False, False]))
    assert torch.equal(dirs[1, 0], directions[3])
