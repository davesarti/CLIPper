import pytest
import torch

from src.mining import IDENTITY_PROXY, TripletMiner, proxy_rows


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


def test_violation_negative_breaks_a_constraint():
    miner = _miner()
    for _ in range(20):
        t = miner.sample(k=2)
        if t is None or not t.negatives:
            continue
        assert miner.labels[t.violation, t.negatives].any()
        assert miner.labels[t.violation, t.positives].all()
        return
    pytest.skip("no query with negative constraints was sampled")


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
