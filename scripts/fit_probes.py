"""Fit the 40 linear attribute probes and save them to results/probe_weights.pt.

One logistic regression per attribute on cached train-split CLIP features. The
full split (features/clip-vit-base-patch32_train.pt) is used when present,
otherwise the 30k sample; `--pool-features` overrides. Fitting is deterministic
(zero init, full-batch Adam), so re-runs reproduce the same weights.

The probes serve two roles downstream: the normalized rows are the edit
directions used by the composition, and the raw weights with their biases are
the classifier behind the attribute-space score (docs/method.md S4).

**Refitting invalidates every number in results/.** The saved weights are the
recipe behind the reported benchmarks, and both the probe pool and the fitting
defaults change the directions. Recompute the baseline rows in the same run as
anything you compare against them.

Run with: conda run -n clipper python scripts/fit_probes.py
          ... scripts/fit_probes.py --pool-features features/..._train.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_paths, load_dataset
from src.features import load_pool, resolve_pool
from src.probes import fit_linear_probes

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--pool-features", type=Path, default=None,
                    help="feature cache to fit on; default: the full train "
                         "split if extracted, else the 30k sample")
parser.add_argument("--epochs", type=int, default=2000,
                    help="full-batch Adam steps; the default is the recipe "
                         "behind the shipped weights")
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--device", default=None,
                    help="cuda/cpu; defaults to cuda when available. The full "
                         "split at 2000 full-batch steps is slow on CPU.")
args = parser.parse_args()

paths = get_paths()
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
cache = resolve_pool(paths.features_dir, args.pool_features)
if not cache.is_file():
    sys.exit(f"Missing {cache}: run scripts/extract_train_features.py "
             f"(add --all for the full split).")

features, indices = load_pool(cache)
train = load_dataset(paths, split="train")
labels = train.attr[indices]
# torchvision's attr_names carries a trailing empty entry; drop it.
attr_names = [n for n in train.attr_names if n]
assert features.shape[0] == labels.shape[0], "feature/label size mismatch"
assert len(attr_names) == labels.shape[1], "attribute name/label count mismatch"

print(f"Fitting {labels.shape[1]} probes on {features.shape[0]} images "
      f"from {cache.name} ({device})...")
w, b = fit_linear_probes(features.to(device), labels.to(device),
                         epochs=args.epochs, lr=args.lr)

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)
torch.save(
    {"weights": w.cpu(), "biases": b.cpu(), "attributes": attr_names,
     "pool": cache.name, "images": int(features.shape[0]),
     "epochs": args.epochs, "lr": args.lr},
    out_dir / "probe_weights.pt",
)
print(f"Saved probe weights {tuple(w.shape)} to {out_dir / 'probe_weights.pt'}")
print("Refit done: recompute results/probe_accuracy.csv and any benchmark row "
      "you intend to compare against.")
