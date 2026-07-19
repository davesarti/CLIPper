"""Runtime-aware paths and data loading for the CelebA retrieval project."""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from torchvision.datasets import CelebA

REPO_ROOT = Path(__file__).resolve().parent.parent


def _on_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


@dataclass(frozen=True)
class Paths:
    data_root: Path
    annotations_path: Path
    features_dir: Path


def get_paths() -> Paths:
    if _on_colab():
        return Paths(
            data_root=Path("/content/datasets"),
            annotations_path=Path("/content/drive/MyDrive/datasets/celeba_evaluation.json"),
            features_dir=Path("/content/drive/MyDrive/datasets/features"),
        )
    return Paths(
        data_root=REPO_ROOT,
        annotations_path=REPO_ROOT / "celeba_evaluation.json",
        features_dir=REPO_ROOT / "features",
    )


def load_dataset(paths: Paths, split: str = "test") -> CelebA:
    # Note: CelebA appends "celeba/" to the root itself.
    if not (paths.data_root / "celeba").is_dir():
        raise FileNotFoundError(
            f"CelebA folder not found under {paths.data_root}. "
            "Expected <data_root>/celeba/img_align_celeba/..."
        )
    return CelebA(root=paths.data_root, split=split, download=False)


def load_annotations(paths: Paths) -> list[dict]:
    if not paths.annotations_path.is_file():
        raise FileNotFoundError(f"Evaluation JSON not found: {paths.annotations_path}")
    with open(paths.annotations_path) as f:
        return json.load(f)
