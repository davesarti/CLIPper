"""Fit the 40 linear attribute probes and save them to results/probe_weights.pt.

One logistic regression per attribute on the cached 30k-image train-split
CLIP features (features/clip-vit-base-patch32_train30k.pt, which stores the
features together with the sampled train-split indices used to align the
labels). Fitting is deterministic (zero init, full-batch Adam), so re-runs
reproduce the same weights.

Run with: conda run -n clipper python scripts/fit_probes.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_paths, load_dataset
from src.features import ClipEncoder
from src.probes import fit_linear_probes

paths = get_paths()
train = load_dataset(paths, split="train")

cache = paths.features_dir / f"{ClipEncoder.MODEL_NAME.split('/')[-1]}_train30k.pt"
if not cache.is_file():
    sys.exit(f"Missing {cache}: extract the train-split sample features first.")
saved = torch.load(cache, weights_only=True)
features, indices = saved["features"], saved["indices"]
labels = train.attr[indices]
# torchvision's attr_names carries a trailing empty entry; drop it.
attr_names = [n for n in train.attr_names if n]
assert features.shape[0] == labels.shape[0], "feature/label size mismatch"
assert len(attr_names) == labels.shape[1], "attribute name/label count mismatch"

print(f"Fitting {labels.shape[1]} probes on {features.shape[0]} images...")
# The fit_linear_probes defaults (2000 epochs, no weight decay) match the
# recipe of the shipped results/probe_weights.pt: refit directions align at
# cosine > 0.96 and reproduce the benchmark numbers within noise.
w, b = fit_linear_probes(features, labels)

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)
torch.save(
    {"weights": w, "biases": b, "attributes": attr_names},
    out_dir / "probe_weights.pt",
)
print(f"Saved probe weights {tuple(w.shape)} to {out_dir / 'probe_weights.pt'}")
