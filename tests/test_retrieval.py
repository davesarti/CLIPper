import pytest
import torch

from src.data import get_paths, load_annotations
from src.retrieval import PROMPTS, parse_query, compose, rank


def test_parse_single_positive():
    assert parse_query("+Smiling") == (["Smiling"], [])


def test_parse_single_negative():
    assert parse_query("-Heavy_Makeup") == ([], ["Heavy_Makeup"])


def test_parse_compound():
    pos, neg = parse_query("+Wearing_Lipstick, -Heavy_Makeup, +Smiling")
    assert pos == ["Wearing_Lipstick", "Smiling"]
    assert neg == ["Heavy_Makeup"]


def test_parse_rejects_missing_sign():
    with pytest.raises(ValueError):
        parse_query("Smiling")


def test_prompts_cover_benchmark_attributes():
    annotations = load_annotations(get_paths())
    for entry in annotations:
        pos, neg = parse_query(entry["query"])
        for attr in pos + neg:
            assert attr in PROMPTS, f"missing prompt for {attr}"


def test_prompts_are_natural_language():
    for attr, prompt in PROMPTS.items():
        assert "_" not in prompt, f"raw underscore leaked into prompt for {attr}"
        assert prompt.startswith("a photo of")


def _unit(*vals):
    v = torch.tensor(vals, dtype=torch.float32)
    return v / v.norm()


def test_compose_is_normalized():
    v_ref = _unit(1.0, 0.0, 0.0)
    pos = torch.stack([_unit(0.0, 1.0, 0.0)])
    neg = torch.stack([_unit(0.0, 0.0, 1.0)])
    q = compose(v_ref, pos, neg)
    assert torch.isclose(q.norm(), torch.tensor(1.0), atol=1e-6)


def test_compose_signs():
    v_ref = _unit(1.0, 0.0, 0.0)
    pos = torch.stack([_unit(0.0, 1.0, 0.0)])
    neg = torch.stack([_unit(0.0, 0.0, 1.0)])
    q = compose(v_ref, pos, neg)
    expected = _unit(1.0, 1.0, -1.0)
    assert torch.allclose(q, expected, atol=1e-6)


def test_compose_empty_constraints():
    v_ref = _unit(3.0, 4.0, 0.0)
    empty = torch.zeros((0, 3))
    q = compose(v_ref, empty, empty)
    assert torch.allclose(q, v_ref, atol=1e-6)


def test_compose_gamma_scales_identity():
    v_ref = _unit(1.0, 0.0, 0.0)
    pos = torch.stack([_unit(0.0, 1.0, 0.0)])
    empty = torch.zeros((0, 3))
    q_lo = compose(v_ref, pos, empty, gamma=0.1)
    q_hi = compose(v_ref, pos, empty, gamma=10.0)
    assert (q_hi @ v_ref) > (q_lo @ v_ref)
    assert torch.allclose(q_hi, _unit(10.0, 1.0, 0.0), atol=1e-6)


def test_rank_orders_by_similarity():
    features = torch.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(1, 1, 0)])
    query = _unit(1, 0.1, 0).unsqueeze(0)  # closest to 0, then 2, then 1
    order = rank(query, features)
    assert order[0].tolist() == [0, 2, 1]


def test_rank_excludes_source():
    features = torch.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(1, 1, 0)])
    query = _unit(1, 0.1, 0).unsqueeze(0)
    order = rank(query, features, exclude=[0])
    assert order[0][0].item() != 0
    assert order[0][-1].item() == 0  # excluded index sinks to the bottom


def test_rank_rejects_mismatched_exclude_length():
    features = torch.stack([_unit(1, 0, 0), _unit(0, 1, 0)])
    query = _unit(1, 0, 0).unsqueeze(0)
    with pytest.raises(ValueError):
        rank(query, features, exclude=[0, 1])
