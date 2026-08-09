"""Tune and evaluate the exclusion re-rank (docs/method.md S7).

Three stages, in this order:

1. **Sweep on the held-out val benchmark.** lambda- first with lambda+ = 0, then
   lambda+ at the winning lambda-. The 14 test queries and their ground truth
   are never involved in the choice.
2. **2x2 test table.** {fixed probe rule, CPAS-MLP} x {re-rank off, on}, with the
   fixed-rule row recomputed in this same run against the same probe file -
   absolute numbers here are not comparable across probe refits, only deltas
   within a run are.
3. **Ablations.** negative-only / positive-only / both / linear penalty instead
   of the hinge / hinge on the top-200 shortlist only.

Every cell reports R@{1,5,10}, P@{1,5,10} and the top-10 violation rate. If R@10
rises while the violation rate does not fall, the gain is not coming from
exclusion.

Run:  conda run -n clipper python scripts/run_exclusion_rerank.py \
          --checkpoint results/cpas_model.pt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cpas_mlp import PerAttributeMLP
from src.data import get_paths, load_annotations, load_dataset
from src.evaluation import (
    build_val_benchmark,
    run_cpas_benchmark,
    run_probe_benchmark,
    score_val_benchmark,
)
from src.features import ClipEncoder, load_or_extract
from src.probes import load_probes, load_raw_probes, roc_auc
from src.rerank import Rerank, database_probe_probs
from src.retrieval import parse_query
from src.steering import FixedRule

BASELINE_GAMMA = 0.6
VAL_FRACTION = 0.1
# Extended past the proposal's {0, .1, .25, .5, 1, 2, 4}: on the first run val
# R@10 was still rising at 4.0, and a monotone curve means the sweep has not
# reached the point where the penalty starts fighting the cosine term.
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
REPO_ROOT = Path(__file__).resolve().parent.parent
METRIC_COLS = ["R@1", "R@5", "R@10", "P@1", "P@5", "P@10", "V@10"]

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "results" / "cpas_model.pt",
                    help="CPAS-MLP checkpoint; skipped if absent")
parser.add_argument("--val-per-query", type=int, default=200)
parser.add_argument("--top-m", type=int, default=200,
                    help="shortlist size for the two-stage ablation")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "exclusion_rerank.csv")
args = parser.parse_args()

paths = get_paths()
device = "cuda" if torch.cuda.is_available() else "cpu"
celeba = load_dataset(paths)
annotations = load_annotations(paths)
features = load_or_extract(ClipEncoder(), celeba, paths.features_dir)
test_labels = celeba.attr.bool()
directions, _, attributes = load_probes(REPO_ROOT)
raw_weights, raw_biases, raw_attributes = load_raw_probes(REPO_ROOT)
assert raw_attributes == attributes
attr_index = {name: i for i, name in enumerate(attributes)}

# The saved biases belong to the raw weights, so pairing them with the
# normalized directions yields meaningless probabilities. Confirm the raw pair
# separates a known attribute before trusting db_probs. Preferred surface is
# the valid split, so the number is comparable to results/probe_accuracy.csv;
# the test split is also held out from probe fitting, so it is a sound fallback
# on a machine that never extracted the valid features.
valid_path = paths.features_dir / f"{ClipEncoder.MODEL_NAME.split('/')[-1]}_valid.pt"
if valid_path.is_file():
    valid = torch.load(valid_path, weights_only=True)
    check_features = valid["features"] if isinstance(valid, dict) else valid
    check_labels = load_dataset(paths, split="valid").attr.bool()
    if isinstance(valid, dict):
        check_labels = check_labels[valid["indices"]]
    check_split = "valid"
else:
    check_features, check_labels, check_split = features, test_labels, "test"
    print(f"{valid_path.name} not found; running the probe check on the test "
          f"split instead (also held out from probe fitting)")
check_row = attr_index["Male"]
check_auc = roc_auc(check_features @ raw_weights[check_row] + raw_biases[check_row],
                    check_labels[:, check_row])
print(f"Sanity check - Male probe {check_split} AUC from the raw weights: {check_auc:.4f}")
if check_auc < 0.9:
    sys.exit("raw probe weights do not reproduce the reported AUC; wrong tensors loaded")

db_probs = database_probe_probs(features, raw_weights, raw_biases)
print(f"db_probs: {tuple(db_probs.shape)}, "
      f"{db_probs.numel() * 4 / 1e6:.1f} MB")

# ---------------------------------------------------------------- combiners
combiners: dict[str, object] = {f"probe rule (gamma={BASELINE_GAMMA})": FixedRule(BASELINE_GAMMA)}
if args.checkpoint.is_file():
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    config = dict(checkpoint.get("config", {}))
    config.pop("arch", None)
    model = PerAttributeMLP(**config)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as err:  # e.g. a checkpoint from the transformer variant
        print(f"{args.checkpoint} is not a PerAttributeMLP checkpoint, skipping it:\n"
              f"  {str(err).splitlines()[0]}")
    else:
        model.eval()
        combiners["CPAS-MLP"] = model
else:
    print(f"no checkpoint at {args.checkpoint}: evaluating the fixed rule only")
if "CPAS-MLP" not in combiners:
    print("Only the fixed probe rule will be evaluated; the 2x2 table needs a "
          "CPAS-MLP checkpoint (train one with scripts/train_cpas.py).")

# --------------------------------------------- stage 1: sweep on validation
# The val pool is the held-out slice of the train split used during training,
# rebuilt with the same seed and fraction so a validation reference is never a
# training candidate (mirrors scripts/train_cpas.py).
slug = ClipEncoder.MODEL_NAME.split("/")[-1]
pool_path = paths.features_dir / f"{slug}_train.pt"
if not pool_path.is_file():
    pool_path = paths.features_dir / f"{slug}_train30k.pt"
saved = torch.load(pool_path, weights_only=True)
if isinstance(saved, dict):
    pool_features, pool_indices = saved["features"], saved["indices"]
else:
    pool_features, pool_indices = saved, torch.arange(saved.shape[0])
pool_labels = load_dataset(paths, split="train").attr[pool_indices].bool()

perm = torch.randperm(pool_features.shape[0],
                      generator=torch.Generator().manual_seed(0))
val_rows = perm[int(pool_features.shape[0] * (1 - VAL_FRACTION)):]
val_pool = pool_features[val_rows]
val_labels = pool_labels[val_rows]
query_specs = [
    ([attr_index[a] for a in parse_query(e["query"])[0]],
     [attr_index[a] for a in parse_query(e["query"])[1]])
    for e in annotations
]
val_tasks = build_val_benchmark(
    val_labels, query_specs, directions,
    per_query=args.val_per_query, seed=args.seed,
)
val_probs = database_probe_probs(val_pool, raw_weights, raw_biases)
print(f"Val benchmark: {len(val_tasks)} queries, "
      f"{sum(t['refs'].shape[0] for t in val_tasks)} held-out references")

sweep_rows = []
best: dict[str, tuple[float, float]] = {}
for name, combiner in combiners.items():
    scores = {}
    for lam_neg in LAMBDA_GRID:
        r10 = score_val_benchmark(
            combiner, val_pool, val_tasks,
            rerank=Rerank(val_probs, lam_neg=lam_neg),
        )
        scores[(lam_neg, 0.0)] = r10
        sweep_rows.append({"combiner": name, "stage": "lambda_neg",
                           "lam_neg": lam_neg, "lam_pos": 0.0, "val_R@10": r10})
        print(f"[val] {name:26s} lam- {lam_neg:<5} lam+ 0     R@10 {r10:.4f}")
    best_neg = max(LAMBDA_GRID, key=lambda l: scores[(l, 0.0)])
    for lam_pos in LAMBDA_GRID[1:]:
        r10 = score_val_benchmark(
            combiner, val_pool, val_tasks,
            rerank=Rerank(val_probs, lam_neg=best_neg, lam_pos=lam_pos),
        )
        scores[(best_neg, lam_pos)] = r10
        sweep_rows.append({"combiner": name, "stage": "lambda_pos",
                           "lam_neg": best_neg, "lam_pos": lam_pos, "val_R@10": r10})
        print(f"[val] {name:26s} lam- {best_neg:<5} lam+ {lam_pos:<5} R@10 {r10:.4f}")
    best[name] = max(scores, key=scores.get)
    print(f"  -> best on val: lam- {best[name][0]}, lam+ {best[name][1]} "
          f"(R@10 {scores[best[name]]:.4f})\n")

pd.DataFrame(sweep_rows).to_csv(
    args.out.with_name(args.out.stem + "_sweep.csv"), index=False)


# ------------------------------------------------- stages 2 and 3: test set
rows, per_query = [], []


def evaluate(name: str, variant: str, combiner, rerank: Rerank | None) -> dict:
    if isinstance(combiner, FixedRule):
        df = run_probe_benchmark(annotations, features, directions, attr_index,
                                 gamma=combiner.gamma, rerank=rerank,
                                 labels=test_labels)
    else:
        df = run_cpas_benchmark(annotations, features, combiner, directions,
                                attr_index, rerank=rerank, labels=test_labels)
    mean = df[df["query"] == "MEAN"].iloc[0]
    per_query.append(df.assign(combiner=name, variant=variant))
    return {"combiner": name, "variant": variant} | mean[METRIC_COLS].to_dict()



for name, combiner in combiners.items():
    lam_neg, lam_pos = best[name]
    variants = {
        "0 off": None,
        "1 negative only": Rerank(db_probs, lam_neg=lam_neg or LAMBDA_GRID[1]),
        "2 positive only": Rerank(db_probs, lam_pos=lam_pos or LAMBDA_GRID[1]),
        "3 both (tuned)": Rerank(db_probs, lam_neg=lam_neg, lam_pos=lam_pos),
        "4 linear, no hinge": Rerank(db_probs, lam_neg=lam_neg, lam_pos=lam_pos,
                                     hinge=False),
        "5 hinge on top-200": Rerank(db_probs, lam_neg=lam_neg, lam_pos=lam_pos,
                                     top_m=args.top_m),
    }
    for variant, rerank in variants.items():
        rows.append(evaluate(name, variant, combiner, rerank))
        r = rows[-1]
        print(f"{name:26s} {variant:20s} R@10 {r['R@10']:.4f}  V@10 {r['V@10']:.4f}")

table = pd.DataFrame(rows)
args.out.parent.mkdir(exist_ok=True)
table.to_csv(args.out, index=False)
pd.concat(per_query, ignore_index=True).to_csv(
    args.out.with_name(args.out.stem + "_per_query.csv"), index=False)

print("\nMEAN over the 14 test queries "
      "(V@10 = fraction of top-10 breaking a constraint):")
print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nWrote {args.out}, {args.out.stem}_sweep.csv, {args.out.stem}_per_query.csv")
print("Differences below 0.02 R@10 are within seed-to-seed spread - unresolved.")
