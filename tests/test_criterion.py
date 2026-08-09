import torch

from src.criterion import MAX_HAMMING, hamming_to, satisfies, valid_mask


def _labels(n=200, a=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, a, generator=gen) > 0.5


def test_satisfies_requires_all_positives_and_no_negatives():
    labels = _labels()
    ok = satisfies(labels, [0, 1], [2])
    assert labels[ok, 0].all() and labels[ok, 1].all()
    assert not labels[ok, 2].any()
    # Complete, not just sound: nothing meeting the rule may be left out.
    expected = labels[:, 0] & labels[:, 1] & ~labels[:, 2]
    assert torch.equal(ok, expected)


def test_an_empty_query_is_satisfied_by_everything():
    labels = _labels()
    assert satisfies(labels, [], []).all()


def test_hamming_ignores_the_queried_columns():
    labels = _labels()
    ref = labels[3]
    queried = [0, 5]
    others = [a for a in range(labels.shape[1]) if a not in queried]
    expected = (labels[:, others] != ref[others]).sum(dim=1)
    assert torch.equal(hamming_to(labels, ref, queried), expected)


def test_hamming_over_an_empty_query_counts_every_column():
    labels = _labels()
    ref = labels[7]
    assert torch.equal(hamming_to(labels, ref, []), (labels != ref).sum(dim=1))


def test_hamming_to_itself_is_zero():
    labels = _labels()
    assert int(hamming_to(labels, labels[11], [0])[11]) == 0


def test_valid_mask_is_the_conjunction_of_both_conditions():
    labels = _labels()
    ref, add, remove = labels[5], [0], [1]
    got = valid_mask(labels, ref, add, remove)
    expected = satisfies(labels, add, remove) & (
        hamming_to(labels, ref, add + remove) <= MAX_HAMMING
    )
    assert torch.equal(got, expected)


def test_valid_mask_tightens_monotonically_with_the_radius():
    labels = _labels()
    ref = labels[2]
    loose = valid_mask(labels, ref, [0], [], max_hamming=4)
    tight = valid_mask(labels, ref, [0], [], max_hamming=1)
    assert int(loose.sum()) > int(tight.sum())
    assert (tight <= loose).all()
