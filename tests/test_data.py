from pathlib import Path

from src.data import get_paths, load_annotations, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_get_paths_local():
    paths = get_paths()
    # Not running on Colab, so paths must point into the repo
    assert paths.data_root == REPO_ROOT
    assert paths.annotations_path == REPO_ROOT / "celeba_evaluation.json"
    assert paths.features_dir == REPO_ROOT / "features"


def test_load_annotations():
    annotations = load_annotations(get_paths())
    assert len(annotations) == 14
    first = annotations[0]
    assert set(first.keys()) == {"query", "ground_truth"}
    assert first["query"] == "+Smiling"
    # keys are dataset-index strings, values are lists of ints
    idx, targets = next(iter(first["ground_truth"].items()))
    assert isinstance(idx, str) and idx.isdigit()
    assert all(isinstance(t, int) for t in targets)


def test_load_dataset():
    celeba = load_dataset(get_paths())
    assert len(celeba) == 19962  # official CelebA test split
