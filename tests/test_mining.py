import pytest
import torch

from src.mining import (
    IDENTITY_PROXY,
    TripletMiner,
    correlated_pairs,
    proxy_rows,
)


def _pool(n=400, a=12, d=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    labels = torch.rand(n, a, generator=gen) > 0.5
    features = torch.nn.functional.normalize(torch.randn(n, d, generator=gen), dim=-1)
    return labels, features


def _miner(**kwargs):
    labels, features = _pool()
    return TripletMiner(labels, features, proxy_rows=[0, 1, 2], **kwargs)


def test_target_satisfies_the_flipped_constraints():
    miner = _miner()
    for _ in range(20):
        t = miner.sample(k=2)
        if t is None:
            continue
        assert miner.labels[t.target, t.positives].all()
        assert not miner.labels[t.target, t.negatives].any()
        assert t.target != t.ref


def test_flips_are_taken_against_the_reference_state():
    miner = _miner()
    t = miner.sample(k=3)
    assert t is not None
    # Positives are attributes the reference lacks, negatives ones it has.
    assert not miner.labels[t.ref, t.positives].any()
    assert miner.labels[t.ref, t.negatives].all()


def test_violation_negatives_break_a_constraint():
    miner = _miner()
    for _ in range(20):
        t = miner.sample(k=2)
        if t is None or not t.negatives:
            continue
        for v in t.violations:
            assert miner.labels[v, t.negatives].any()
            assert miner.labels[v, t.positives].all()
        return
    pytest.skip("no query with negative constraints was sampled")


def test_mines_one_violation_by_default_and_n_when_asked():
    assert all(len(t.violations) == 1 for t in _miner().sample_batch(10))
    many = _miner(n_violations=8).sample_batch(10)
    assert all(1 <= len(t.violations) <= 8 for t in many)
    assert any(len(t.violations) > 1 for t in many)


def test_violations_are_ordered_by_similarity_to_the_reference():
    miner = _miner(n_violations=5)
    t = miner.sample(k=2)
    assert t is not None
    sims = [float(miner.features[v] @ miner.features[t.ref]) for v in t.violations]
    assert sims == sorted(sims, reverse=True)


def test_neg_fraction_shifts_the_realized_negation_share():
    # Prevalence 0.25, close to CelebA's 0.226: uniform flips are mostly
    # additions because a flip is a negation only when the reference has the
    # attribute.
    gen = torch.Generator().manual_seed(1)
    labels = torch.rand(600, 8, generator=gen) < 0.25
    features = torch.nn.functional.normalize(torch.randn(600, 4, generator=gen), dim=-1)
    kwargs = dict(proxy_rows=[6, 7], min_candidates=5)

    uniform = TripletMiner(labels, features, **kwargs)
    uniform.sample_batch(150, ks=(2,))
    forced = TripletMiner(labels, features, neg_fraction=0.5, **kwargs)
    forced.sample_batch(150, ks=(2,))

    assert uniform.summary()["negation_share"] < 0.4
    assert forced.summary()["negation_share"] > uniform.summary()["negation_share"]
    assert forced.summary()["negation_share"] == pytest.approx(0.5, abs=0.15)


def test_summary_reports_the_rejection_rate():
    miner = _miner(min_candidates=10_000)
    miner.sample(k=1)
    assert miner.summary()["rejection_rate"] == 1.0
    assert miner.summary()["too_few_candidates"] == 1.0


def test_reference_with_too_few_on_attributes_degrades_gracefully():
    # One reference, one ON attribute: asking for two negations must not raise
    # and must not silently emit an all-positive query.
    labels = torch.zeros(50, 6, dtype=torch.bool)
    labels[:, 0] = True
    features = torch.nn.functional.normalize(torch.randn(50, 4), dim=-1)
    miner = TripletMiner(labels, features, proxy_rows=[1, 2], min_candidates=1,
                         neg_fraction=1.0)
    for _ in range(10):
        positives, negatives = miner._sample_flips(0, k=3)
        assert negatives == [0]
        assert len(positives) == 2  # the rest falls back to additions


def test_correlated_pairs_finds_the_correlated_columns_only():
    labels = torch.zeros(100, 3, dtype=torch.bool)
    labels[:50, 0] = True
    labels[:45, 1] = True          # nearly the same column as 0
    labels[::2, 2] = True          # independent
    assert correlated_pairs(labels, threshold=0.3) == [(0, 1)]


def test_correlated_pair_sampling_puts_the_pair_in_tension():
    labels = torch.zeros(200, 4, dtype=torch.bool)
    labels[:, 0] = True            # every reference has attribute 0
    labels[:100, 2] = True
    features = torch.nn.functional.normalize(torch.randn(200, 4), dim=-1)
    miner = TripletMiner(labels, features, proxy_rows=[2, 3], min_candidates=1,
                         pairs=[(0, 1)], pair_prob=1.0)
    positives, negatives = miner._sample_flips(0, k=2)
    # Attribute 0 is ON and 1 is OFF, so the pair can only be split this way.
    assert negatives == [0] and positives == [1]
    assert miner._used_pair


def test_pair_prob_zero_leaves_sampling_untouched():
    plain = _miner(seed=3).sample_batch(8)
    with_table = _miner(seed=3, pairs=[(0, 1)], pair_prob=0.0).sample_batch(8)
    assert plain == with_table


def test_distractor_satisfies_constraints_but_differs_from_reference():
    miner = _miner()
    t = miner.sample(k=2)
    assert t is not None
    assert miner.labels[t.distractor, t.positives].all()
    assert not miner.labels[t.distractor, t.negatives].any()
    agree = lambda i: (
        miner.labels[i, miner.proxy_rows] == miner.labels[t.ref, miner.proxy_rows]
    ).sum()
    assert agree(t.distractor) <= agree(t.target)


def test_rejects_flip_sets_with_too_few_candidates():
    miner = _miner(min_candidates=10_000)
    assert miner.sample(k=1) is None


def test_sampling_is_deterministic_given_a_seed():
    a, b = _miner(seed=7).sample_batch(5), _miner(seed=7).sample_batch(5)
    assert a == b


def test_sample_batch_respects_the_flip_curriculum():
    miner = _miner()
    batch = miner.sample_batch(10, ks=(1,))
    assert len(batch) == 10
    assert all(len(t.positives) + len(t.negatives) == 1 for t in batch)


def test_proxy_rows_maps_names_to_indices():
    names = ["Pad", *IDENTITY_PROXY]
    assert proxy_rows(names) == list(range(1, len(IDENTITY_PROXY) + 1))
    with pytest.raises(KeyError):
        proxy_rows(["Smiling"])
