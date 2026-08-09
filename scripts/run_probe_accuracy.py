"""Measure the linear attribute probes: fit on train, score on valid.

Writes results/probe_accuracy.csv with per-attribute ROC AUC and average
precision on the held-out valid split, plus the positive rate needed to read
them. AP is reported next to AUC because AUC flatters a rare attribute, while
AP is measured against a baseline of that attribute's own positive rate.

Current result: macro-mean valid AUC 0.929 (0.731 Oval_Face to 0.999 Male),
macro-mean AP 0.746. The attributes appearing in the evaluation benchmark all
score AUC 0.909 or better.

Probe quality lives in this file. The similarly named attribute_accuracy.csv
under results/archive/ belongs to the abandoned zero-shot text-prompt method and
measures prompt classification instead; do not read probe quality out of it.

Reads only cached features; the test split is never loaded. Takes a couple of
minutes on CPU, nearly all of it the 2000-step probe fit.

Run with: conda run -n clipper python scripts/run_probe_accuracy.py
"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_paths, load_dataset
from src.features import ClipEncoder, load_pool, resolve_pool
from src.probes import fit_linear_probes, score_attributes

paths = get_paths()
slug = ClipEncoder.MODEL_NAME.split("/")[-1]

train = load_dataset(paths, split="train")
train_cache = resolve_pool(paths.features_dir)
if not train_cache.is_file():
    sys.exit(f"Missing {train_cache}: run scripts/extract_train_features.py first.")
train_features, indices = load_pool(train_cache)
train_labels = train.attr[indices]

valid = load_dataset(paths, split="valid")
valid_cache = paths.features_dir / f"{slug}_valid.pt"
if not valid_cache.is_file():
    sys.exit(f"Missing {valid_cache}: run scripts/extract_train_features.py "
             "--split valid --all first.")
valid_features = torch.load(valid_cache, weights_only=True)
if isinstance(valid_features, dict):
    valid_features = valid_features["features"]
valid_labels = valid.attr

# torchvision's attr_names carries a trailing empty entry; drop it.
attribute_names = [n for n in train.attr_names if n]
assert train_features.shape[0] == train_labels.shape[0], "train size mismatch"
assert valid_features.shape[0] == valid_labels.shape[0], "valid size mismatch"
assert len(attribute_names) == train_labels.shape[1], "attribute count mismatch"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"fitting on {tuple(train_features.shape)} from {train_cache.name} "
      f"({device}), scoring {tuple(valid_features.shape)}")
w, b = fit_linear_probes(train_features.to(device), train_labels.to(device))
# Scoring is a per-attribute Python loop, so bring the probes back to CPU.
w, b = w.cpu(), b.cpu()
aucs, aps = score_attributes(valid_features @ w.T + b, valid_labels)

rows = [
    {
        "attribute": name,
        "auc_valid": aucs[j],
        "ap_valid": aps[j],
        "pos_rate_valid": float(valid_labels[:, j].float().mean()),
    }
    for j, name in enumerate(attribute_names)
]
df = pd.DataFrame(rows).sort_values("auc_valid")
mean_row = {
    "attribute": "MACRO_MEAN",
    "auc_valid": df["auc_valid"].mean(),
    "ap_valid": df["ap_valid"].mean(),
    "pos_rate_valid": df["pos_rate_valid"].mean(),
}
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "probe_accuracy.csv", index=False)
print(df.to_string(index=False))
print(f"\nWrote {out_dir / 'probe_accuracy.csv'}")
