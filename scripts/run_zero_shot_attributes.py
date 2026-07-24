"""Zero-shot prompt-based attribute classification: calibrate thresholds on valid, report on test.

Run with: conda run -n clipper python scripts/run_zero_shot_attributes.py
First run encodes the ~19.9k-image valid split (~30-60 min on CPU); later
runs load cached features and finish in seconds.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attributes import (
    ATTRIBUTE_PROMPTS,
    accuracy,
    attribute_scores,
    balanced_accuracy,
    build_text_embeddings,
    calibrate_threshold_precision,
    roc_auc,
)

MIN_PRECISION = 0.7   # recall is maximized subject to this precision floor
MIN_AUC = 0.6         # valid-split AUC below this -> attribute never rendered
from src.data import get_paths, load_dataset
from src.features import ClipEncoder, load_or_extract

paths = get_paths()
encoder = ClipEncoder()

datasets = {split: load_dataset(paths, split=split) for split in ("valid", "test")}
features = {
    split: load_or_extract(encoder, ds, paths.features_dir, split=split)
    for split, ds in datasets.items()
}
for split, ds in datasets.items():
    assert features[split].shape[0] == len(ds), f"{split}: feature/dataset size mismatch"

pos_emb, neg_emb = build_text_embeddings(encoder.encode_texts)
scores = {s: attribute_scores(features[s], pos_emb, neg_emb) for s in features}

# Column j of `scores` follows ATTRIBUTE_PROMPTS order; align labels to it.
attr_names = list(ATTRIBUTE_PROMPTS)
col = [datasets["test"].attr_names.index(a) for a in attr_names]
labels = {s: datasets[s].attr[:, col] for s in datasets}

rows = []
thresholds = {}
unreliable = []
for j, attr in enumerate(attr_names):
    sv, yv = scores["valid"][:, j], labels["valid"][:, j]
    st, yt = scores["test"][:, j], labels["test"][:, j]
    thr = calibrate_threshold_precision(sv, yv, min_precision=MIN_PRECISION)
    thresholds[attr] = thr
    auc_valid = roc_auc(sv, yv)
    if auc_valid < MIN_AUC:
        unreliable.append(attr)
    preds, pos = st > thr, yt.bool()
    rows.append({
        "attribute": attr,
        "auc": roc_auc(st, yt),
        "auc_valid": auc_valid,
        "bal_acc": balanced_accuracy(st, yt, thr),
        "acc": accuracy(st, yt, thr),
        "precision": (preds & pos).sum().item() / max(preds.sum().item(), 1),
        "threshold": thr,
        "pos_rate_true": yt.float().mean().item(),
        "pos_rate_pred": preds.float().mean().item(),
    })

df = pd.DataFrame(rows).sort_values("bal_acc", ascending=False)
metric_cols = [c for c in df.columns if c != "attribute"]
mean_row = {"attribute": "MACRO_MEAN"} | df[metric_cols].mean().to_dict()
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "attribute_accuracy.csv", index=False)
with open(out_dir / "attribute_thresholds.json", "w") as f:
    json.dump({
        "min_precision": MIN_PRECISION,
        "min_auc": MIN_AUC,
        "thresholds": thresholds,
        "unreliable": unreliable,
    }, f, indent=2)

print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nunreliable (valid AUC < {MIN_AUC}, never rendered): {unreliable}")
