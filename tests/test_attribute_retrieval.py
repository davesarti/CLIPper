import pytest
import torch

from src.attribute_head import (
    AttributeMLP,
    attribute_reliability,
    bit_accuracy,
    fit_attribute_head,
    reliability_weights,
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


def test_unit_weights_match_the_unweighted_distance():
    db = torch.rand(4, 5)
    ref = torch.rand(2, 5) > 0.5
    rows = [0, 2, 4]
    plain = expected_hamming(db, ref, rows)
    weighted = expected_hamming(db, ref, rows, weights=torch.ones(5))
    assert torch.allclose(plain, weighted)


def test_a_zero_weight_removes_the_attribute_from_the_distance():
    ref = torch.zeros(1, 3, dtype=torch.bool)
    w = torch.tensor([1.0, 0.0, 1.0])
    quiet = expected_hamming(torch.tensor([[0.2, 0.1, 0.3]]), ref, [0, 1, 2], weights=w)
    loud = expected_hamming(torch.tensor([[0.2, 0.9, 0.3]]), ref, [0, 1, 2], weights=w)
    assert float(quiet[0, 0]) == pytest.approx(float(loud[0, 0]))


def test_doubling_a_weight_doubles_that_attributes_contribution():
    db = torch.tensor([[1.0]])
    ref = torch.zeros(1, 1, dtype=torch.bool)
    single = expected_hamming(db, ref, [0], weights=torch.tensor([1.0]))
    double = expected_hamming(db, ref, [0], weights=torch.tensor([2.0]))
    assert float(double[0, 0]) == pytest.approx(2 * float(single[0, 0]))


def test_attribute_scores_forwards_the_weights():
    # Attribute 1 is zeroed, so the two candidates - which differ only there -
    # must score identically.
    probs = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    code = probs > 0.5
    ref = torch.tensor([[True, True]])
    scores = attribute_scores(probs, code, ref, [], [], lam_constraint=0.0,
                              weights=torch.tensor([1.0, 0.0]))
    assert float(scores[0, 0]) == pytest.approx(float(scores[0, 1]))


def test_a_float_reference_of_hard_bits_reproduces_the_thresholded_code():
    # The soft reference is a strict generalisation: fed 0/1 it must be exact.
    db = torch.rand(4, 3)
    hard = torch.tensor([[True, False, True]])
    assert torch.allclose(expected_hamming(db, hard, [0, 1, 2]),
                          expected_hamming(db, hard.float(), [0, 1, 2]))


def test_target_code_keeps_probabilities_and_forces_the_queried_bits():
    soft = torch.tensor([[0.90, 0.51, 0.20, 0.80]])
    out = target_code(soft, pos_rows=[2], neg_rows=[0])
    assert out[0].tolist() == pytest.approx([0.0, 0.51, 1.0, 0.80])
    assert float(soft[0, 0]) == pytest.approx(0.90)   # input untouched


def test_an_uncertain_reference_bit_cannot_reorder_candidates():
    # At p = 0.5 the attribute contributes the same amount to every candidate,
    # so it drops out of the ranking instead of deciding it on noise.
    probs = torch.tensor([[0.0], [1.0]])
    ref = torch.tensor([[0.5]])
    d = expected_hamming(probs, ref, [0])
    assert float(d[0, 0]) == pytest.approx(float(d[1, 0]))


def test_reliability_is_one_for_a_perfect_predictor():
    labels = torch.tensor([[True, False], [False, True], [True, True]])
    assert attribute_reliability(labels, labels).tolist() == pytest.approx([1.0, 1.0])


def test_reliability_is_zero_for_a_majority_class_predictor():
    # Attribute 0 is 20% positive, so answering "no" every time scores 80%
    # accuracy. It detects nothing and must be worth nothing - this is the
    # trap that rules accuracy out as a weight on CelebA.
    labels = torch.zeros(10, 1, dtype=torch.bool)
    labels[:2, 0] = True
    pred = torch.zeros(10, 1, dtype=torch.bool)
    assert float(attribute_reliability(pred, labels)[0]) == pytest.approx(0.0)


def test_reliability_weights_are_non_negative_and_average_to_one():
    labels = torch.tensor([[True, False], [False, True], [True, True], [False, False]])
    pred = torch.tensor([[True, True], [False, False], [True, False], [False, True]])
    w = reliability_weights(pred, labels)
    assert float(w.mean()) == pytest.approx(1.0)
    assert bool((w >= 0).all())   # an anti-correlated attribute is clamped, not negated
