"""CPAS-specific behaviour.

The contract shared with the other combiners - baseline-equivalent init,
normalization, padding and order invariance - is exercised for both
architectures in tests/test_steering.py. What is left here is the one thing
only CPAS has: attention between attribute tokens, and the mask that ablates it.
"""

import torch

from src.cpas import CPAS


def _batch(b=3, k=2, d=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    v_ref = torch.nn.functional.normalize(torch.randn(b, d, generator=gen), dim=-1)
    dirs = torch.nn.functional.normalize(torch.randn(b, k, d, generator=gen), dim=-1)
    signs = torch.tensor([[1.0, -1.0]] * b)
    mask = torch.ones(b, k, dtype=torch.bool)
    return v_ref, dirs, signs, mask


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
