import torch

from src.evaluation import build_val_benchmark, score_val_benchmark


def _labels(n=300, a=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(n, a, generator=gen) > 0.5)


def test_ground_truth_satisfies_constraints_and_matches_proxy():
    labels = _labels()
    directions = torch.eye(8)
    proxy = [4, 5, 6]
    # query: +attr0, -attr1 (proxy excludes neither, so both are enforced separately)
    tasks = build_val_benchmark(labels, [([0], [1])], proxy, directions,
                                per_query=20, min_gt=1, seed=1)
    task = tasks[0]
    for row, ref in enumerate(task["refs"].tolist()):
        gt = task["gt_mask"][row].nonzero(as_tuple=True)[0]
        assert labels[gt, 0].all()          # T+ satisfied
        assert not labels[gt, 1].any()      # T- satisfied
        assert (labels[gt][:, proxy] == labels[ref, proxy]).all()  # identity match
        assert ref not in gt.tolist()       # reference excluded


def test_queried_attribute_is_dropped_from_the_proxy_match():
    labels = _labels()
    directions = torch.eye(8)
    proxy = [0, 4, 5]  # attr0 is both a proxy AND the queried attribute
    tasks = build_val_benchmark(labels, [([0], [])], proxy, directions,
                                per_query=20, min_gt=1, seed=2)
    # +attr0 forces the target to differ from a reference lacking attr0; if attr0
    # were kept in the proxy, no reference lacking it could ever have ground truth.
    refs_without_attr0 = [r for r in tasks[0]["refs"].tolist() if not labels[r, 0]]
    assert refs_without_attr0, "expected some references that lack the queried attr"


def test_score_is_one_when_the_model_points_at_ground_truth():
    labels = _labels()
    directions = torch.eye(8)
    tasks = build_val_benchmark(labels, [([0], [1])], [4, 5], directions,
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
    tasks = build_val_benchmark(labels, [([0], [1])], [4, 5], directions,
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
