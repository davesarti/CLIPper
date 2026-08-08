import torch

from src.probes import (
    average_precision,
    compose_probe,
    fit_linear_probes,
    score_attributes,
)


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


def test_fit_linear_probes_defaults_are_unchanged():
    # Every number in results/ was produced at these defaults; a change here
    # silently invalidates them.
    import inspect

    sig = inspect.signature(fit_linear_probes)
    assert sig.parameters["epochs"].default == 2000
    assert sig.parameters["lr"].default == 0.05
    assert sig.parameters["weight_decay"].default == 0.0
    assert sig.parameters["class_balanced"].default is False


def test_fit_linear_probes_weight_decay_shrinks_weights():
    x, y = _separable_data()
    w_plain, _ = fit_linear_probes(x, y, epochs=200, weight_decay=0.0)
    w_reg, _ = fit_linear_probes(x, y, epochs=200, weight_decay=0.5)
    assert w_reg.norm() < w_plain.norm()


def test_fit_linear_probes_class_balancing_lifts_rare_positive_recall():
    # 5% positives along one axis; unweighted BCE under-predicts the class.
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(600, 6, generator=gen)
    x = x / x.norm(dim=-1, keepdim=True)
    thresh = x[:, 0].quantile(0.95)
    y = (x[:, 0] > thresh).long().unsqueeze(1)
    w_plain, b_plain = fit_linear_probes(x, y, epochs=400)
    w_bal, b_bal = fit_linear_probes(x, y, epochs=400, class_balanced=True)
    pos = y.squeeze(1).bool()
    recall_plain = ((x @ w_plain.T + b_plain).squeeze(1)[pos] > 0).float().mean()
    recall_bal = ((x @ w_bal.T + b_bal).squeeze(1)[pos] > 0).float().mean()
    assert recall_bal > recall_plain


def test_fit_linear_probes_class_balancing_survives_an_empty_attribute():
    # An attribute with zero positives must not produce nan and must not
    # corrupt its neighbours' gradients.
    x, y = _separable_data()
    y[:, 1] = 0
    w, b = fit_linear_probes(x, y, epochs=50, class_balanced=True)
    assert torch.isfinite(w).all() and torch.isfinite(b).all()


def test_average_precision_perfect_ranking_is_one():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([1, 1, 0, 0])
    assert average_precision(scores, labels) == 1.0


def test_average_precision_single_positive_ranked_last():
    # Only positive sits at rank n, so precision at that rank is 1/n.
    scores = torch.tensor([0.9, 0.8, 0.7, 0.1])
    labels = torch.tensor([0, 0, 0, 1])
    assert abs(average_precision(scores, labels) - 0.25) < 1e-6


def test_average_precision_is_at_least_prevalence_for_perfect_ranking():
    scores = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
    labels = torch.tensor([1, 0, 1, 0, 0])
    # ranks 1 and 3 -> (1/1 + 2/3) / 2
    expected = (1.0 + 2.0 / 3.0) / 2
    assert abs(average_precision(scores, labels) - expected) < 1e-6


def test_average_precision_no_positives_is_nan():
    scores = torch.tensor([0.5, 0.4])
    labels = torch.tensor([0, 0])
    assert average_precision(scores, labels) != average_precision(scores, labels)


def test_score_attributes_returns_one_metric_per_attribute():
    x, y = _separable_data()
    w, b = fit_linear_probes(x, y, epochs=200)
    aucs, aps = score_attributes(x @ w.T + b, y)
    assert len(aucs) == 3 and len(aps) == 3
    assert all(v > 0.9 for v in aucs)
    assert all(v > 0.9 for v in aps)
