import pytest
import torch

from src.evaluation import run_probe_benchmark, violation_rate
from src.rerank import (
    Rerank,
    database_probe_probs,
    exclusion_penalty,
    rank_with_exclusion,
)
from src.retrieval import rank


def _probs(n=6, a=3):
    torch.manual_seed(0)
    return torch.rand(n, a)


def test_database_probe_probs_matches_the_logistic_model():
    features = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    weights, biases = torch.randn(3, 4), torch.randn(3)
    p = database_probe_probs(features, weights, biases)
    assert p.shape == (5, 3)
    assert torch.allclose(p[2, 1], torch.sigmoid(features[2] @ weights[1] + biases[1]))


def test_compliant_candidates_are_never_charged():
    # Attribute 0 absent (p < 0.5) and attribute 1 present: satisfies -0, +1.
    probs = torch.tensor([[0.1, 0.9, 0.5], [0.8, 0.2, 0.5]])
    penalty = exclusion_penalty(probs, pos_rows=[1], neg_rows=[0],
                                lam_neg=1.0, lam_pos=1.0)
    assert penalty[0] == 0.0
    assert penalty[1] > 0.0


def test_penalty_grows_with_lambda_and_is_zero_when_off():
    probs = _probs()
    assert torch.equal(exclusion_penalty(probs, [0], [1]), torch.zeros(6))
    small = exclusion_penalty(probs, [0], [1], lam_neg=0.5).sum()
    large = exclusion_penalty(probs, [0], [1], lam_neg=1.0).sum()
    assert large == 2 * small


def test_linear_penalty_charges_compliant_candidates_too():
    """Ablation row 4: without the hinge the penalty is compensatory again."""
    probs = torch.tensor([[0.1, 0.9, 0.5]])
    hinged = exclusion_penalty(probs, [1], [0], lam_neg=1.0, hinge=True)
    linear = exclusion_penalty(probs, [1], [0], lam_neg=1.0, hinge=False)
    assert hinged[0] == 0.0
    assert linear[0] > 0.0


def test_per_attribute_thresholds_are_respected():
    probs = torch.tensor([[0.4, 0.0, 0.0]])
    tau_low = torch.tensor([0.3, 0.5, 0.5])
    tau_high = torch.tensor([0.9, 0.5, 0.5])
    assert exclusion_penalty(probs, [], [0], lam_neg=1.0, thresholds=tau_low)[0] > 0
    assert exclusion_penalty(probs, [], [0], lam_neg=1.0, thresholds=tau_high)[0] == 0


def test_inactive_rerank_reproduces_plain_cosine_ranking():
    q = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    db = torch.nn.functional.normalize(torch.randn(9, 4), dim=-1)
    exclude = [0, 1, 2]
    plain = rank(q, db, exclude=exclude)
    for rerank in (None, Rerank(_probs(9), lam_neg=0.0, lam_pos=0.0)):
        out = rank_with_exclusion(q, db, [0], [1], rerank=rerank, exclude=exclude)
        assert torch.equal(out, plain)


def test_rerank_demotes_a_violating_top_hit():
    # Image 0 is the cosine winner but carries the forbidden attribute.
    db = torch.eye(3)
    q = torch.tensor([[0.9, 0.8, 0.0]])
    probs = torch.tensor([[0.99, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.0, 0.0]])
    assert rank_with_exclusion(q, db, [], [0], Rerank(probs))[0, 0] == 0
    penalized = rank_with_exclusion(q, db, [], [0], Rerank(probs, lam_neg=1.0))
    assert penalized[0, 0] == 1


def test_top_m_leaves_candidates_outside_the_shortlist_uncharged():
    db = torch.eye(3)
    q = torch.tensor([[0.9, 0.8, 0.7]])
    probs = torch.ones(3, 1)  # every image carries the forbidden attribute
    full = Rerank(probs, lam_neg=1.0).scores(q, db, [], [0])
    two = Rerank(probs, lam_neg=1.0, top_m=2).scores(q, db, [], [0])
    assert torch.allclose(full[0, 2] + 0.5, two[0, 2])  # image 2 is off the shortlist
    assert torch.allclose(full[0, 0], two[0, 0])


def test_violation_rate_counts_broken_constraints_in_the_top_k():
    labels = torch.tensor([[True, False], [True, True], [False, False]])
    order = torch.tensor([[0, 1, 2]])
    # Query +0, -1: image 0 complies, image 1 has the forbidden attribute,
    # image 2 misses the required one.
    assert violation_rate(order, labels, [0], [1], k=3) == pytest.approx(2 / 3)
    assert violation_rate(order, labels, [0], [1], k=1) == 0.0


def test_benchmark_reports_violation_rate_only_when_labels_are_given():
    image_features = torch.eye(4)
    directions = torch.eye(4)[:2]
    annotations = [{"query": "+Smiling", "ground_truth": {"0": [1]}}]
    attr_index = {"Smiling": 1}
    labels = torch.zeros(4, 2, dtype=torch.bool)

    plain = run_probe_benchmark(annotations, image_features, directions, attr_index)
    assert "V@10" not in plain.columns
    with_labels = run_probe_benchmark(annotations, image_features, directions,
                                      attr_index, labels=labels)
    # No database image has Smiling, so every returned image violates the query.
    assert with_labels.loc[0, "V@10"] == 1.0
