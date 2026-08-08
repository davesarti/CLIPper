import pytest
import torch

from src.cpas_mlp import PerAttributeMLP
from src.mining import Triplet
from src.training import (
    build_batch,
    infonce_loss,
    recall_at_1,
    run_epoch,
    violation_loss,
)


def _pool(n=40, d=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    features = torch.nn.functional.normalize(torch.randn(n, d, generator=gen), dim=-1)
    directions = torch.nn.functional.normalize(torch.randn(5, d, generator=gen), dim=-1)
    return features, directions


def _triplets(count=4, violations=1):
    return [
        Triplet(ref=i, positives=[0], negatives=[1], target=count + i,
                violations=[2 * count + i + j for j in range(violations)],
                distractor=3 * count + i)
        for i in range(count)
    ]


def test_build_batch_gathers_features_and_negatives():
    features, directions = _pool()
    triplets = _triplets()
    batch = build_batch(triplets, features, directions)
    assert batch.v_ref.shape == (4, 8)
    assert batch.dirs.shape == (4, 2, 8)
    assert batch.violations.shape == (4, 1, 8)
    assert batch.negatives.shape == (4, 2, 8)
    assert torch.equal(batch.targets[0], features[triplets[0].target])
    # Candidates keep the original [violation, distractor, lazy] layout, the
    # last slot being the lazy negative: the reference itself.
    assert batch.candidates.shape == (4, 3, 8)
    assert torch.equal(batch.candidates[0, 0], features[triplets[0].violations[0]])
    assert torch.equal(batch.candidates[0, 2], features[triplets[0].ref])


def test_build_batch_pads_short_violation_lists_by_cycling():
    features, directions = _pool()
    triplets = _triplets(count=2, violations=1)
    triplets[0] = Triplet(**{**vars(triplets[0]), "violations": [4, 5, 6]})
    batch = build_batch(triplets, features, directions)
    assert batch.violations.shape == (2, 3, 8)
    # The one-violation triplet repeats its own near miss rather than borrowing.
    assert torch.equal(batch.violations[1, 0], batch.violations[1, 2])


def test_violation_loss_falls_when_the_target_outranks_its_violations():
    d = 8
    targets = torch.nn.functional.normalize(torch.randn(4, d), dim=-1)
    violations = torch.nn.functional.normalize(torch.randn(4, 8, d), dim=-1)
    good = violation_loss(targets, targets, violations)
    bad = violation_loss(violations[:, 0], targets, violations)
    assert good < bad


def test_infonce_is_lower_when_the_target_is_ranked_first():
    d = 8
    targets = torch.nn.functional.normalize(torch.randn(4, d), dim=-1)
    negatives = torch.nn.functional.normalize(torch.randn(4, 3, d), dim=-1)
    good = infonce_loss(targets, targets, negatives)
    bad = infonce_loss(targets[[1, 2, 3, 0]], targets, negatives)
    assert good < bad


def test_recall_at_1_is_one_when_query_equals_target():
    features, directions = _pool()
    batch = build_batch(_triplets(), features, directions)
    assert recall_at_1(batch.targets, batch) == 1.0


def test_run_epoch_without_optimizer_leaves_weights_unchanged():
    features, directions = _pool()
    model = PerAttributeMLP(d=8, hidden=8, sign_dim=4, rank=4)
    before = model.head_alpha.bias.clone()
    loss, recall = run_epoch(model, _triplets(), features, directions, batch_size=2)
    assert torch.equal(model.head_alpha.bias, before)
    assert loss > 0 and 0.0 <= recall <= 1.0


def test_default_run_epoch_keeps_violations_inside_the_main_softmax():
    """lambda_violation = 0 must reproduce the original single-softmax loss."""
    features, directions = _pool()
    model = PerAttributeMLP(d=8, hidden=8, sign_dim=4, rank=4)
    triplets = _triplets()
    batch = build_batch(triplets, features, directions)
    with torch.no_grad():
        q = model(batch.v_ref, batch.dirs, batch.signs, batch.mask)
        expected = float(infonce_loss(q, batch.targets, batch.candidates))
    loss, _ = run_epoch(model, triplets, features, directions, batch_size=4)
    assert loss == pytest.approx(expected, abs=1e-6)


def test_lambda_violation_adds_the_separate_term():
    features, directions = _pool()
    model = PerAttributeMLP(d=8, hidden=8, sign_dim=4, rank=4)
    triplets = _triplets()
    base, _ = run_epoch(model, triplets, features, directions, batch_size=4)
    weighted, _ = run_epoch(model, triplets, features, directions, batch_size=4,
                            lambda_violation=1.0)
    assert weighted != base


def test_run_epoch_with_optimizer_reduces_loss():
    features, directions = _pool()
    model = PerAttributeMLP(d=8, hidden=8, sign_dim=4, rank=4)
    triplets = _triplets(8)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    first, _ = run_epoch(model, triplets, features, directions, optimizer=opt, batch_size=8)
    for _ in range(15):
        last, _ = run_epoch(model, triplets, features, directions, optimizer=opt, batch_size=8)
    assert last < first
