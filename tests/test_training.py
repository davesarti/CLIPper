import torch

from src.cpas_mlp import PerAttributeMLP
from src.criterion import valid_mask
from src.mining import MinedQuery, Miner
from src.training import build_batch, infonce_loss, recall_at_1, run_epoch


def _pool(n=40, a=6, d=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    features = torch.nn.functional.normalize(torch.randn(n, d, generator=gen), dim=-1)
    directions = torch.nn.functional.normalize(torch.randn(a, d, generator=gen), dim=-1)
    labels = torch.rand(n, a, generator=gen) > 0.5
    return features, directions, labels


def _queries(count=4, n_neg=2):
    return [
        MinedQuery(ref=i, add=[0], remove=[1], target=count + i,
                   violators=[2 * count + i] * n_neg,
                   drifters=[3 * count + i] * n_neg)
        for i in range(count)
    ]


def test_build_batch_gathers_features_and_both_masks():
    features, directions, labels = _pool()
    queries = _queries()
    batch = build_batch(queries, features, directions, labels)
    assert batch.v_ref.shape == (4, 8)
    assert batch.dirs.shape == (4, 2, 8)
    assert batch.negatives.shape == (4, 5, 8)   # 2 violators + 2 drifters + lazy
    assert batch.neg_mask.shape == (4, 5)
    assert batch.false_neg.shape == (4, 4)
    assert torch.equal(batch.targets[0], features[queries[0].target])
    assert torch.equal(batch.negatives[0, 0], features[queries[0].violators[0]])
    assert torch.equal(batch.negatives[0, 2], features[queries[0].drifters[0]])


def test_the_reference_always_owns_the_last_negative_slot():
    """Anti-collapse protection must be structural, not a matter of the draw."""
    features, directions, labels = _pool()
    queries = _queries()
    batch = build_batch(queries, features, directions, labels)
    for i, q in enumerate(queries):
        assert torch.equal(batch.negatives[i, -1], features[q.ref])
        assert bool(batch.neg_mask[i, -1])


def test_short_families_are_masked_not_repeated():
    """A repeated negative would carry double weight in the softmax."""
    features, directions, labels = _pool()
    queries = _queries(count=2, n_neg=3)
    queries[1] = MinedQuery(ref=1, add=[0], remove=[1], target=5,
                            violators=[6], drifters=[7, 8])
    batch = build_batch(queries, features, directions, labels)
    assert batch.negatives.shape[1] == 3 + 3 + 1
    assert batch.neg_mask[0].all()
    assert batch.neg_mask[1].tolist() == [True, False, False,   # 1 of 3 violators
                                          True, True, False,   # 2 of 3 drifters
                                          True]                # lazy


def test_the_false_negative_mask_never_hides_a_rows_own_target():
    features, directions, labels = _pool()
    batch = build_batch(_queries(), features, directions, labels)
    assert not batch.false_neg.diagonal().any()


def test_the_false_negative_mask_flags_another_rows_valid_target():
    """Two rows with the same query: each row's target is valid for the other."""
    features, directions, labels = _pool(n=200, a=8, seed=3)
    miner = Miner(labels, min_targets=3, seed=1)
    q = miner.sample(k=1)
    assert q is not None
    valid = valid_mask(labels, labels[q.ref], q.add, q.remove).nonzero(as_tuple=True)[0]
    other = int(valid[valid != q.target][0])
    twin = MinedQuery(q.ref, q.add, q.remove, other, q.violators, q.drifters)
    batch = build_batch([q, twin], features, directions, labels)
    assert bool(batch.false_neg[0, 1]) and bool(batch.false_neg[1, 0])


def test_masked_candidates_are_removed_from_the_softmax():
    features, directions, labels = _pool()
    batch = build_batch(_queries(), features, directions, labels)
    q = batch.targets
    free = infonce_loss(q, batch.targets, batch.negatives)
    masked = infonce_loss(q, batch.targets, batch.negatives,
                          batch.neg_mask, batch.false_neg)
    # Dropping candidates can only leave the positive a larger share.
    assert masked <= free


def test_unmasked_loss_matches_the_plain_formula_bit_for_bit():
    """The regression guard: masks off must reproduce the previous behaviour."""
    gen = torch.Generator().manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(4, 8, generator=gen), dim=-1)
    targets = torch.nn.functional.normalize(torch.randn(4, 8, generator=gen), dim=-1)
    negatives = torch.nn.functional.normalize(torch.randn(4, 3, 8, generator=gen), dim=-1)
    logits = torch.cat([q @ targets.T,
                        torch.einsum("bd,bmd->bm", q, negatives)], dim=1) / 0.05
    expected = torch.nn.functional.cross_entropy(logits, torch.arange(4))
    assert torch.equal(infonce_loss(q, targets, negatives), expected)


def test_infonce_is_lower_when_the_target_is_ranked_first():
    gen = torch.Generator().manual_seed(1)
    targets = torch.nn.functional.normalize(torch.randn(4, 8, generator=gen), dim=-1)
    negatives = torch.nn.functional.normalize(torch.randn(4, 3, 8, generator=gen), dim=-1)
    assert infonce_loss(targets, targets, negatives) < \
        infonce_loss(targets[[1, 2, 3, 0]], targets, negatives)


def test_recall_at_1_is_one_when_query_equals_target():
    features, directions, labels = _pool()
    batch = build_batch(_queries(), features, directions, labels)
    assert recall_at_1(batch.targets, batch) == 1.0


def test_run_epoch_without_optimizer_leaves_weights_unchanged():
    features, directions, labels = _pool()
    model = PerAttributeMLP(d=8, hidden=8, sign_dim=4, rank=4)
    before = model.head_alpha.bias.clone()
    loss, recall = run_epoch(model, _queries(), features, directions, labels,
                             batch_size=2)
    assert torch.equal(model.head_alpha.bias, before)
    assert loss > 0 and 0.0 <= recall <= 1.0


def test_run_epoch_with_optimizer_reduces_loss():
    features, directions, labels = _pool()
    model = PerAttributeMLP(d=8, hidden=8, sign_dim=4, rank=4)
    queries = _queries(8)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    first, _ = run_epoch(model, queries, features, directions, labels,
                         optimizer=opt, batch_size=8)
    for _ in range(15):
        last, _ = run_epoch(model, queries, features, directions, labels,
                            optimizer=opt, batch_size=8)
    assert last < first
