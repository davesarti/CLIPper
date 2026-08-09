"""Measure the linear attribute probes on the held-out valid split.

Writes results/probe_accuracy.csv with, per attribute:

| column | meaning |
|---|---|
| `auc_valid` | ROC AUC - threshold-free ranking quality |
| `ap_valid` | average precision, against a baseline of `pos_rate_valid` |
| `pos_rate_valid` | the attribute's positive rate, needed to read AP |
| `threshold` | decision threshold maximizing bit accuracy on valid |
| `acc_valid_at_half` | per-bit accuracy at the default 0.5 cut |
| `acc_valid_tuned` | per-bit accuracy at `threshold` |

AP is reported next to AUC because AUC flatters a rare attribute, while AP is
measured against that attribute's own positive rate. Accuracy is reported next
to both because the retrieval score consumes a *thresholded* code (docs/method.md
S4.1): AUC measures ranking within an attribute, and the score never sees a
ranking. The MACRO_MEAN row's accuracy columns are the overall bit accuracy.

The thresholds are also written to results/probe_thresholds.json. They are
measured here, not applied: `run_attribute_retrieval.py` cuts the linear probe
at 0.5, and a code produced with different thresholds is a different code.

Scores the probes saved by scripts/fit_probes.py, so the numbers describe the
weights the rest of the pipeline actually loads; with none saved it fits them
here on the same recipe. Reads only cached features - the test split is never
loaded.

Run with: conda run -n clipper python scripts/run_probe_accuracy.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attribute_head import bit_accuracy, tune_thresholds
from src.data import get_paths, load_dataset
from src.features import ClipEncoder, load_pool, resolve_pool
from src.probes import fit_linear_probes, load_raw_probes, score_attributes

REPO_ROOT = Path(__file__).resolve().parent.parent
paths = get_paths()
slug = ClipEncoder.MODEL_NAME.split("/")[-1]

valid = load_dataset(paths, split="valid")
valid_cache = paths.features_dir / f"{slug}_valid.pt"
if not valid_cache.is_file():
    sys.exit(f"Missing {valid_cache}:\n"
             f"  uv run scripts/extract_train_features.py --split valid --all")
valid_features = torch.load(valid_cache, weights_only=True)
if isinstance(valid_features, dict):
    valid_features = valid_features["features"]
valid_labels = valid.attr
assert valid_features.shape[0] == valid_labels.shape[0], "valid size mismatch"

try:
    w, b, attribute_names = load_raw_probes(REPO_ROOT)
    source = "results/probe_weights.pt"
except FileNotFoundError:
    train = load_dataset(paths, split="train")
    train_cache = resolve_pool(paths.features_dir)
    if not train_cache.is_file():
        sys.exit(f"Missing {train_cache}:\n"
                 f"  uv run scripts/extract_train_features.py --all")
    train_features, indices = load_pool(train_cache)
    train_labels = train.attr[indices]
    # torchvision's attr_names carries a trailing empty entry; drop it.
    attribute_names = [n for n in train.attr_names if n]
    assert train_features.shape[0] == train_labels.shape[0], "train size mismatch"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"no saved probes; fitting on {tuple(train_features.shape)} from "
          f"{train_cache.name} ({device})")
    w, b = fit_linear_probes(train_features.to(device), train_labels.to(device))
    w, b = w.cpu(), b.cpu()
    source = f"fitted here on {train_cache.name}"

assert len(attribute_names) == valid_labels.shape[1], "attribute count mismatch"
print(f"scoring {source} on {tuple(valid_features.shape)} valid images")

logits = valid_features @ w.T + b
# Scoring is a per-attribute Python loop, so everything here stays on CPU.
aucs, aps = score_attributes(logits, valid_labels)
thresholds = tune_thresholds(logits, valid_labels)
probs = torch.sigmoid(logits)
correct_half = (probs > 0.5) == valid_labels.bool()
correct_tuned = (probs > thresholds) == valid_labels.bool()

rows = [
    {
        "attribute": name,
        "auc_valid": aucs[j],
        "ap_valid": aps[j],
        "pos_rate_valid": float(valid_labels[:, j].float().mean()),
        "threshold": float(thresholds[j]),
        "acc_valid_at_half": float(correct_half[:, j].float().mean()),
        "acc_valid_tuned": float(correct_tuned[:, j].float().mean()),
    }
    for j, name in enumerate(attribute_names)
]
df = pd.DataFrame(rows).sort_values("auc_valid")
metric_cols = [c for c in df.columns if c != "attribute"]
mean_row = {"attribute": "MACRO_MEAN"} | df[metric_cols].mean().to_dict()
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

out_dir = REPO_ROOT / "results"
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "probe_accuracy.csv", index=False)
with open(out_dir / "probe_thresholds.json", "w") as f:
    json.dump({
        "source": source,
        "tuned_on": f"{slug}_valid ({valid_features.shape[0]} images)",
        "bit_accuracy_at_half": bit_accuracy(logits, valid_labels),
        "bit_accuracy_tuned": bit_accuracy(logits, valid_labels, thresholds),
        "thresholds": {name: float(thresholds[j])
                       for j, name in enumerate(attribute_names)},
    }, f, indent=2)

print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nvalid bit accuracy  {bit_accuracy(logits, valid_labels):.4f} at 0.5"
      f"  ->  {bit_accuracy(logits, valid_labels, thresholds):.4f} tuned")
print(f"Wrote {out_dir / 'probe_accuracy.csv'} and "
      f"{out_dir / 'probe_thresholds.json'}")
