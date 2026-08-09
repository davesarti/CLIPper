"""Train the nonlinear attribute head on frozen CLIP features (docs/method.md S3).

Retrieval quality is bounded by attribute-code accuracy, and the measured
exchange rate is steep: +0.003 bit accuracy bought +0.022 R@10. This trains a
small MLP on the train split, selects the epoch with the best *held-out bit
accuracy* on the valid split, and tunes per-attribute decision thresholds there
too. The saved file carries the thresholds with the weights, since a code
produced with different thresholds is a different code.

The linear probe is scored in the same run for comparison, so the delta is
within-run and not across probe refits.

Run:  conda run -n clipper python scripts/fit_attribute_head.py
      ... scripts/fit_attribute_head.py --hidden 2048 --epochs 100
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attribute_head import bit_accuracy, fit_attribute_head, tune_thresholds
from src.data import get_paths, load_dataset
from src.features import ClipEncoder
from src.probes import load_raw_probes

REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hidden", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pool-features", type=Path, default=None,
                    help="training features; default: full train split if "
                         "present, else the 30k sample")
parser.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "attribute_head.pt")
args = parser.parse_args()
torch.manual_seed(args.seed)

paths = get_paths()
device = "cuda" if torch.cuda.is_available() else "cpu"
slug = ClipEncoder.MODEL_NAME.split("/")[-1]

pool_path = args.pool_features
if pool_path is None:
    full = paths.features_dir / f"{slug}_train.pt"
    pool_path = full if full.is_file() else paths.features_dir / f"{slug}_train30k.pt"
if not pool_path.is_file():
    sys.exit(f"Missing {pool_path}: run scripts/extract_train_features.py --all")

saved = torch.load(pool_path, weights_only=True)
if isinstance(saved, dict):
    features, indices = saved["features"], saved["indices"]
else:
    features, indices = saved, torch.arange(saved.shape[0])
labels = load_dataset(paths, split="train").attr[indices].bool()

valid_path = paths.features_dir / f"{slug}_valid.pt"
if not valid_path.is_file():
    sys.exit(f"Missing {valid_path}: the valid split is the selection surface.")
v_saved = torch.load(valid_path, weights_only=True)
if isinstance(v_saved, dict):
    val_features = v_saved["features"]
    val_labels = load_dataset(paths, split="valid").attr[v_saved["indices"]].bool()
else:
    val_features = v_saved
    val_labels = load_dataset(paths, split="valid").attr.bool()

print(f"train {tuple(features.shape)} from {pool_path.name}  "
      f"valid {tuple(val_features.shape)}  on {device}")

# Reference point, recomputed here so the comparison is within this run.
W, B, attributes = load_raw_probes(REPO_ROOT)
probe_logits = val_features @ W.T + B
probe_acc = bit_accuracy(probe_logits, val_labels)
probe_th = tune_thresholds(probe_logits, val_labels)
print(f"linear probe:  valid bit accuracy {probe_acc:.4f}  "
      f"(tuned thresholds {bit_accuracy(probe_logits, val_labels, probe_th):.4f})")

print(f"MLP head hidden={args.hidden} dropout={args.dropout}:")
model, acc = fit_attribute_head(
    features, labels, val_features, val_labels,
    hidden=args.hidden, dropout=args.dropout, epochs=args.epochs, lr=args.lr,
    weight_decay=args.weight_decay, batch_size=args.batch_size, device=device,
)
with torch.no_grad():
    val_logits = model(val_features.to(device)).cpu()
thresholds = tune_thresholds(val_logits, val_labels)
acc_tuned = bit_accuracy(val_logits, val_labels, thresholds)

print(f"\nvalid bit accuracy  linear {probe_acc:.4f} -> MLP {acc:.4f} "
      f"({acc - probe_acc:+.4f})")
print(f"with tuned thresholds: {acc_tuned:.4f}")
print("Expect roughly 7x this delta in R@10; confirm with "
      "scripts/run_attribute_retrieval.py")

args.out.parent.mkdir(exist_ok=True)
torch.save({
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "config": model.config,
    "thresholds": thresholds,
    "attributes": attributes,
    "val_bit_accuracy": acc,
    "val_bit_accuracy_tuned": acc_tuned,
    "linear_probe_bit_accuracy": probe_acc,
    "pool": pool_path.name,
    "seed": args.seed,
}, args.out)
print(f"Wrote {args.out}")
