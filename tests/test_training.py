import torch

from src.cpas import CPAS
from src.mining import Triplet
from src.training import build_batch, infonce_loss, recall_at_1, run_epoch


def _pool(n=40, d=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    features = torch.nn.functional.normalize(torch.randn(n, d, generator=gen), dim=-1)
    directions = torch.nn.functional.normalize(torch.randn(5, d, generator=gen), dim=-1)
    return features, directions


def _triplets(count=4):
    return [
        Triplet(ref=i, positives=[0], negatives=[1],
                target=count + i, violation=2 * count + i, distractor=3 * count + i)
        for i in range(count)
    ]


def test_build_batch_gathers_features_and_negatives():
    features, directions = _pool()
    triplets = _triplets()
    batch = build_batch(triplets, features, directions)
    assert batch.v_ref.shape == (4, 8)
    assert batch.dirs.shape == (4, 2, 8)
    assert batch.negatives.shape == (4, 3, 8)
    assert torch.equal(batch.targets[0], features[triplets[0].target])
    # Third negative slot is the lazy negative: the reference itself.
    assert torch.equal(batch.negatives[0, 2], features[triplets[0].ref])


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
    model = CPAS(d=8, heads=2, ffn=16)
    before = model.head_alpha.bias.clone()
    loss, recall = run_epoch(model, _triplets(), features, directions, batch_size=2)
    assert torch.equal(model.head_alpha.bias, before)
    assert loss > 0 and 0.0 <= recall <= 1.0


def test_run_epoch_with_optimizer_reduces_loss():
    features, directions = _pool()
    model = CPAS(d=8, heads=2, ffn=16)
    triplets = _triplets(8)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    first, _ = run_epoch(model, triplets, features, directions, optimizer=opt, batch_size=8)
    for _ in range(15):
        last, _ = run_epoch(model, triplets, features, directions, optimizer=opt, batch_size=8)
    assert last < first
