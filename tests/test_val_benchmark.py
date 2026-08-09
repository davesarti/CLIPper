import torch

from src.evaluation import build_val_benchmark, score_val_benchmark


def _labels(n=300, a=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(n, a, generator=gen) > 0.5)


def test_ground_truth_follows_the_assignment_hamming_rule():
    """S3.1.1: constraints satisfied AND Hamming <= 2 on the non-queried attrs."""
    labels = _labels()
    directions = torch.eye(8)
    others = [a for a in range(8) if a not in (0, 1)]
    tasks = build_val_benchmark(labels, [([0], [1])], directions,
                                per_query=20, min_gt=1, seed=1)
    task = tasks[0]
    for row, ref in enumerate(task["refs"].tolist()):
        gt = task["gt_mask"][row].nonzero(as_tuple=True)[0]
        assert labels[gt, 0].all()          # T+ satisfied
        assert not labels[gt, 1].any()      # T- satisfied
        assert ref not in gt.tolist()       # reference excluded
        distance = (labels[gt][:, others] != labels[ref, others]).sum(dim=1)
        assert int(distance.max()) <= 2


def test_ground_truth_is_complete_not_just_sound():
    """Every image meeting the rule must be present, or recall is overstated."""
    labels = _labels()
    others = [a for a in range(8) if a != 0]
    tasks = build_val_benchmark(labels, [([0], [])], torch.eye(8),
                                per_query=5, min_gt=1, seed=7)
    task = tasks[0]
    for row, ref in enumerate(task["refs"].tolist()):
        expected = (labels[:, 0]
                    & ((labels[:, others] != labels[ref, others]).sum(dim=1) <= 2))
        expected[ref] = False
        assert torch.equal(task["gt_mask"][row], expected)


def test_max_hamming_is_configurable_and_monotone():
    labels = _labels()
    loose = build_val_benchmark(labels, [([0], [])], torch.eye(8), per_query=5,
                                min_gt=1, seed=3, max_hamming=3)
    tight = build_val_benchmark(labels, [([0], [])], torch.eye(8), per_query=5,
                                min_gt=1, seed=3, max_hamming=1)
    assert int(loose[0]["gt_mask"].sum()) > int(tight[0]["gt_mask"].sum())


def test_score_is_one_when_the_model_points_at_ground_truth():
    labels = _labels()
    directions = torch.eye(8)
    tasks = build_val_benchmark(labels, [([0], [1])], directions,
                                per_query=10, min_gt=1, seed=3)
    db = torch.nn.functional.normalize(torch.randn(labels.shape[0], 8), dim=-1)

    class PerfectModel:
        """Returns, for each reference, one of its ground-truth image vectors."""
        def eval(self):
            return self

        def __call__(self, v_ref, dirs, signs, mask):
            gt_mask = tasks[0]["gt_mask"]
            first_gt = gt_mask.float().argmax(dim=1)
            return db[first_gt]

    assert score_val_benchmark(PerfectModel(), db, tasks, k=10) == 1.0


def test_score_is_zero_when_the_model_points_away_from_ground_truth():
    labels = _labels()
    directions = torch.eye(8)
    tasks = build_val_benchmark(labels, [([0], [1])], directions,
                                per_query=15, min_gt=1, seed=4)
    db = torch.nn.functional.normalize(torch.randn(labels.shape[0], 8), dim=-1)

    class WrongModel:
        """Points every query at a fixed non-ground-truth image."""
        def eval(self):
            return self

        def __call__(self, v_ref, dirs, signs, mask):
            # Row 0 is never ground truth for these seeds (it lacks attr0 / satisfies).
            never_gt = (~tasks[0]["gt_mask"].any(dim=0)).nonzero(as_tuple=True)[0][0]
            return db[never_gt].expand(v_ref.shape[0], -1)

    assert score_val_benchmark(WrongModel(), db, tasks, k=1) == 0.0
