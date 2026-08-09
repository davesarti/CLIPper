import pytest
import torch

from src.attribute_head import (
    AttributeMLP,
    bit_accuracy,
    fit_attribute_head,
    tune_thresholds,
)
from src.attribute_retrieval import (
    attribute_scores,
    constraint_violation,
    expected_hamming,
    rank_by_attributes,
    target_code,
)


def test_target_code_forces_only_the_queried_bits():
    ref = torch.tensor([[True, True, False, False]])
    out = target_code(ref, pos_rows=[2], neg_rows=[1])
    assert out.tolist() == [[True, False, True, False]]
    assert ref.tolist() == [[True, True, False, False]]  # input untouched


def test_expected_hamming_is_exact_for_confident_probabilities():
    # Probabilities at 0/1 make the expectation a plain bit count.
    db = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    ref = torch.tensor([[True, False, False]])
    out = expected_hamming(db, ref, rows=[0, 1, 2])
    assert out.shape == (2, 1)
    assert out[0, 0] == pytest.approx(1.0)   # differs on attribute 2 only
    assert out[1, 0] == pytest.approx(1.0)   # differs on attribute 0 only


def test_expected_hamming_is_maximally_uncertain_at_one_half():
    db = torch.full((1, 4), 0.5)
    ref = torch.zeros(1, 4, dtype=torch.bool)
    assert expected_hamming(db, ref, rows=[0, 1, 2, 3])[0, 0] == pytest.approx(2.0)


def test_expected_hamming_ignores_attributes_outside_rows():
    db = torch.tensor([[1.0, 1.0]])
    ref = torch.zeros(1, 2, dtype=torch.bool)
    assert expected_hamming(db, ref, rows=[0])[0, 0] == pytest.approx(1.0)
    assert expected_hamming(db, ref, rows=[])[0, 0] == pytest.approx(0.0)


def test_constraint_violation_flags_missing_positives_and_present_negatives():
    code = torch.tensor([[True, False], [False, False], [True, True]])
    v = constraint_violation(code, pos_rows=[0], neg_rows=[1])
    assert v.tolist() == [0.0, 1.0, 1.0]


def test_a_violating_candidate_is_pushed_below_a_compliant_one():
    # Image 0 is a perfect attribute match but breaks the negative constraint.
    probs = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    code = probs > 0.5
    ref = torch.tensor([[True, True]])
    scores = attribute_scores(probs, code, ref, pos_rows=[], neg_rows=[1],
                              lam_constraint=100.0)
    assert scores[0, 1] > scores[0, 0]


def test_ranking_prefers_the_smaller_hamming_distance():
    # Attribute 0 is queried; images differ from the reference on the rest.
    probs = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],   # matches the reference on all non-queried
        [1.0, 0.0, 1.0, 1.0],   # differs on one
        [1.0, 0.0, 0.0, 0.0],   # differs on three
    ])
    code = probs > 0.5
    ref = torch.tensor([[True, True, True, True]])
    order = rank_by_attributes(probs, code, ref, pos_rows=[0], neg_rows=[])
    assert order[0].tolist() == [0, 1, 2]


def test_cosine_only_breaks_ties_at_a_small_weight():
    # Two candidates identical in attribute space; cosine decides.
    probs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    code = probs > 0.5
    ref = torch.tensor([[True, True]])
    cos = torch.tensor([[0.1], [0.9]])
    tied = rank_by_attributes(probs, code, ref, [0], [])
    broken = rank_by_attributes(probs, code, ref, [0], [], cosine=cos, w_cos=1.0)
    assert tied[0, 0] == 0        # stable order without the tiebreak
    assert broken[0, 0] == 1      # higher cosine wins


def test_rank_excludes_the_reference_and_validates_its_length():
    probs = torch.rand(5, 3)
    code = probs > 0.5
    ref = torch.zeros(2, 3, dtype=torch.bool)
    order = rank_by_attributes(probs, code, ref, [0], [], exclude=[1, 3])
    assert order[0, -1] == 1 and order[1, -1] == 3
    with pytest.raises(ValueError):
        rank_by_attributes(probs, code, ref, [0], [], exclude=[1])


def test_bit_accuracy_and_thresholds():
    logits = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    labels = torch.tensor([[1, 0], [1, 1]])
    assert bit_accuracy(logits, labels) == pytest.approx(0.75)


def test_tune_thresholds_finds_a_split_that_beats_one_half():
    # Attribute 0's positives sit at p ~ 0.3, so 0.5 misclassifies all of them.
    logits = torch.tensor([[-0.85], [-0.85], [-3.0], [-3.0]])
    labels = torch.tensor([[1], [1], [0], [0]])
    th = tune_thresholds(logits, labels)
    assert bit_accuracy(logits, labels, th) > bit_accuracy(logits, labels)


def test_attribute_head_learns_a_separable_toy_problem():
    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(400, 16), dim=-1)
    y = torch.stack([x[:, 0] > 0, x[:, 1] > 0, (x[:, 0] + x[:, 1]) > 0], dim=1)
    model, acc = fit_attribute_head(x[:300], y[:300], x[300:], y[300:],
                                    hidden=32, epochs=60, batch_size=32,
                                    dropout=0.0, log=False)
    assert isinstance(model, AttributeMLP)
    assert acc > 0.85


def test_head_config_round_trips_through_the_saved_dict():
    model = AttributeMLP(d=8, hidden=16, attributes=5, dropout=0.1)
    assert AttributeMLP(**model.config)(torch.randn(2, 8)).shape == (2, 5)
