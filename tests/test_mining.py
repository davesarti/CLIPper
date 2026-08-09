import pytest
import torch

from src.criterion import hamming_to, satisfies
from src.mining import Miner, retention_weights


def _labels(n=400, a=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, a, generator=gen) > 0.5


def _miner(**kwargs):
    kwargs.setdefault("min_targets", 3)
    return Miner(_labels(), **kwargs)


def _parts(miner, q):
    """(satisfies, hamming) for the whole pool under q's query."""
    return (satisfies(miner.labels, q.add, q.remove),
            hamming_to(miner.labels, miner.labels[q.ref], q.add + q.remove))


def test_the_miner_needs_no_image_features():
    """The rule is label logic; depending on features is what made it slow."""
    Miner(_labels())   # constructs with labels alone


def test_flips_are_taken_against_the_reference_state():
    q = _miner().sample(k=3)
    assert q is not None
    assert not _miner().labels[q.ref, q.add].any()      # added: reference lacks them
    assert Miner(_labels()).labels[q.ref, q.remove].all()  # removed: reference has them


def test_target_is_a_valid_answer_under_the_assignment_rule():
    miner = _miner()
    for _ in range(20):
        q = miner.sample(k=2)
        if q is None:
            continue
        s, h = _parts(miner, q)
        assert s[q.target] and h[q.target] <= miner.max_hamming
        assert q.target != q.ref


def test_violators_are_inside_the_ball_and_break_a_constraint():
    miner = _miner()
    for _ in range(20):
        q = miner.sample(k=2)
        if q is None:
            continue
        s, h = _parts(miner, q)
        for v in q.violators:
            assert not s[v]
            assert h[v] <= miner.max_hamming


def test_drifters_satisfy_the_constraints_and_sit_outside_the_ball():
    miner = _miner()
    for _ in range(20):
        q = miner.sample(k=2)
        if q is None:
            continue
        s, h = _parts(miner, q)
        for d in q.drifters:
            assert s[d]
            assert h[d] > miner.max_hamming


def test_drifters_come_from_the_innermost_non_empty_shell():
    """The negative that misses the ball by one is the one that teaches its edge."""
    miner = _miner()
    for _ in range(20):
        q = miner.sample(k=2)
        if q is None or not q.drifters:
            continue
        s, h = _parts(miner, q)
        outside = h[s & (h > miner.max_hamming)]
        assert len(set(int(h[d]) for d in q.drifters)) == 1
        assert int(h[q.drifters[0]]) == int(outside.min())


def test_the_reference_is_never_drawn_as_a_violator():
    """It owns a dedicated batch slot; drawing it too would count it twice."""
    miner = _miner()
    for _ in range(20):
        q = miner.sample(k=2)
        if q is not None:
            assert q.ref not in q.violators


def test_families_hold_at_most_n_negatives():
    miner = _miner(n_negatives=3)
    q = miner.sample(k=1)
    assert q is not None
    assert len(q.violators) <= 3 and len(q.drifters) <= 3


def test_rejects_queries_with_too_few_valid_targets():
    assert _miner(min_targets=10_000).sample(k=1) is None


def test_sampling_is_deterministic_given_a_seed():
    assert _miner(seed=7).sample_batch(5) == _miner(seed=7).sample_batch(5)


def test_sample_batch_respects_the_flip_curriculum():
    batch = _miner().sample_batch(10, ks=(1,))
    assert len(batch) == 10
    assert all(len(q.add) + len(q.remove) == 1 for q in batch)


def test_weights_steer_which_attributes_get_flipped():
    weights = torch.zeros(8)
    weights[3] = 1.0
    batch = _miner(weights=weights).sample_batch(8, ks=(1,))
    assert all((q.add + q.remove) == [3] for q in batch)


def test_weights_must_match_the_attribute_count():
    with pytest.raises(ValueError):
        _miner(weights=torch.ones(3))


def test_retention_weights_invert_the_acceptance_rate():
    weights = retention_weights(torch.tensor([0.5, 0.25]))
    assert weights[1] == pytest.approx(2 * weights[0])


def test_retention_weights_survive_an_attribute_that_never_survived():
    """Zero retention would divide by zero and swallow the sampling budget."""
    assert torch.isfinite(retention_weights(torch.tensor([0.0, 0.5]))).all()
