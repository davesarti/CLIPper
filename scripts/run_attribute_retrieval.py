"""Evaluate attribute-space retrieval on the 14-query benchmark (docs/method.md S7).

Ranks by the ground-truth rule itself - expected Hamming distance on the
non-queried attributes plus a constraint penalty - rather than by cosine
similarity to a composed query vector. Both attribute predictors are evaluated
in the same run, and the cosine baseline is recomputed alongside them, so every
comparison is within-run.

lam_constraint and w_cos are swept on the held-out validation benchmark, never
on the 14 test queries.

Run:  conda run -n clipper python scripts/run_attribute_retrieval.py
      ... --head results/attribute_head.pt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attribute_head import load_attribute_head
from src.attribute_retrieval import rank_by_attributes, target_code
from src.data import get_paths, load_annotations, load_dataset
from src.evaluation import MAX_HAMMING, _query_row, _with_mean_row, negation_subset
from src.features import ClipEncoder, load_or_extract, load_pool, resolve_pool
from src.probes import load_probes, load_raw_probes
from src.retrieval import parse_query

torch.set_grad_enabled(False)
REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_FRACTION = 0.1
LAM_GRID = (1.0, 4.0, 16.0, 100.0)
COS_GRID = (0.0, 1.0, 3.0, 10.0)
METRIC_COLS = ["R@1", "R@5", "R@10", "P@1", "P@5", "P@10", "V@10"]

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--head", type=Path,
                    default=REPO_ROOT / "results" / "attribute_head.pt",
                    help="MLP attribute head; skipped if absent")
parser.add_argument("--val-refs", type=int, default=200,
                    help="held-out references per query for the sweep")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=None,
                    help="default: results/attribute_retrieval.csv, or "
                         "..._no_cosine.csv under --no-cosine")
parser.add_argument("--no-cosine", action="store_true",
                    help="score by attributes alone: pin w_cos = 0 instead of "
                         "sweeping it, dropping every embedding term from the "
                         "ranking (docs/method.md S5, third term)")
args = parser.parse_args()
cos_grid = (0.0,) if args.no_cosine else COS_GRID
if args.out is None:
    stem = "attribute_retrieval_no_cosine" if args.no_cosine \
        else "attribute_retrieval"
    args.out = REPO_ROOT / "results" / f"{stem}.csv"

paths = get_paths()
celeba = load_dataset(paths)
annotations = load_annotations(paths)
features = load_or_extract(ClipEncoder(), celeba, paths.features_dir)
labels = celeba.attr.bool()
directions, _, attributes = load_probes(REPO_ROOT)
W, B, _ = load_raw_probes(REPO_ROOT)
attr_index = {name: i for i, name in enumerate(attributes)}
queries = [parse_query(e["query"]) for e in annotations]

# ------------------------------------------------------------ predictors
predictors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
probe_probs = torch.sigmoid(features @ W.T + B)
predictors["linear probe"] = (probe_probs, probe_probs > 0.5)
if args.head.is_file():
    head, saved = load_attribute_head(args.head)
    probs = torch.sigmoid(head(features))
    th = saved.get("thresholds")
    predictors["MLP head"] = (probs, probs > (0.5 if th is None else th))
    print(f"loaded {args.head.name}: val bit accuracy "
          f"{saved.get('val_bit_accuracy', float('nan')):.4f}, "
          f"trained on {saved.get('pool', '?')}")
else:
    print(f"no attribute head at {args.head}: linear probe only "
          f"(train one with scripts/fit_attribute_head.py)")


# ------------------------------------------------ sweep on held-out val data
# The val pool is a held-out slice of the train split; its ground truth uses the
# same S3.1.1 rule as the test benchmark, so a gain here is meaningful.
pool_path = resolve_pool(paths.features_dir)
pool_features, pool_indices = load_pool(pool_path)
pool_labels = load_dataset(paths, split="train").attr[pool_indices].bool()
perm = torch.randperm(pool_features.shape[0],
                      generator=torch.Generator().manual_seed(0))
val_rows = perm[int(pool_features.shape[0] * (1 - VAL_FRACTION)):]
val_features, val_labels = pool_features[val_rows], pool_labels[val_rows]
gen = torch.Generator().manual_seed(args.seed)
val_refs = torch.randperm(val_features.shape[0], generator=gen)[: args.val_refs]
print(f"val sweep pool: {val_features.shape[0]} images, {len(val_refs)} references")


def val_recall(probs, code, lam, w_cos) -> float:
    """Mean R@10 over the val pool under the S3.1.1 ground-truth rule."""
    hits = total = 0
    for pos, neg in queries:
        pr = [attr_index[a] for a in pos]; nr = [attr_index[a] for a in neg]
        others = [a for a in range(len(attributes)) if a not in set(pr) | set(nr)]
        sat = torch.ones(val_features.shape[0], dtype=torch.bool)
        if pr: sat &= val_labels[:, pr].all(dim=1)
        if nr: sat &= ~val_labels[:, nr].any(dim=1)
        rest = val_labels[:, others]
        cos = val_features @ val_features[val_refs].T if w_cos else None
        order = rank_by_attributes(
            probs, code, target_code(code[val_refs], pr, nr), pr, nr,
            exclude=val_refs.tolist(), lam_constraint=lam,
            cosine=cos, w_cos=w_cos,
        )
        for row, r in enumerate(val_refs.tolist()):
            gt = sat & ((rest != rest[r]).sum(dim=1) <= MAX_HAMMING)
            gt[r] = False
            if int(gt.sum()) < 3:
                continue
            hits += bool(gt[order[row, :10]].any()); total += 1
    return hits / max(total, 1)


best: dict[str, tuple[float, float]] = {}
sweep = []
for name, (probs, code) in predictors.items():
    val_probs = torch.sigmoid(val_features @ W.T + B) if name == "linear probe" \
        else torch.sigmoid(head(val_features))
    val_code = val_probs > 0.5
    scores = {}
    for lam in LAM_GRID:
        for w in cos_grid:
            r10 = val_recall(val_probs, val_code, lam, w)
            scores[(lam, w)] = r10
            sweep.append({"predictor": name, "lam_constraint": lam,
                          "w_cos": w, "val_R@10": r10})
            print(f"[val] {name:14s} lam {lam:<6} w_cos {w:<5} R@10 {r10:.4f}")
    best[name] = max(scores, key=scores.get)
    print(f"  -> best on val: lam {best[name][0]}, w_cos {best[name][1]} "
          f"(R@10 {scores[best[name]]:.4f})\n")
pd.DataFrame(sweep).to_csv(args.out.with_name(args.out.stem + "_sweep.csv"),
                           index=False)

# ----------------------------------------------------- test benchmark
rows, per_query = [], []
for name, (probs, code) in predictors.items():
    lam, w_cos = best[name]
    metrics = []
    for entry, (pos, neg) in zip(annotations, queries):
        pr = [attr_index[a] for a in pos]; nr = [attr_index[a] for a in neg]
        sources = [int(k) for k in entry["ground_truth"].keys()]
        src = torch.tensor(sources)
        cos = features @ features[src].T if w_cos else None
        order = rank_by_attributes(
            probs, code, target_code(code[src], pr, nr), pr, nr,
            exclude=sources, lam_constraint=lam, cosine=cos, w_cos=w_cos,
        )
        metrics.append(_query_row(entry, order, sources, labels, pr, nr))
    df = _with_mean_row(metrics)
    per_query.append(df.assign(predictor=name))
    rows.append({"method": f"attribute space ({name})",
                 "lam_constraint": lam, "w_cos": w_cos}
                | df[df["query"] == "MEAN"].iloc[0][METRIC_COLS].to_dict()
                | {"neg_R@10": negation_subset(df)})
    print(f"{name:14s} R@10 {rows[-1]['R@10']:.4f}  V@10 {rows[-1]['V@10']:.4f}")

table = pd.DataFrame(rows)
args.out.parent.mkdir(exist_ok=True)
table.to_csv(args.out, index=False)
pd.concat(per_query, ignore_index=True).to_csv(
    args.out.with_name(args.out.stem + "_per_query.csv"), index=False)
print("\nMEAN over the 14 test queries:")
print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nWrote {args.out} (+ _sweep, _per_query)")
print("Compare against the cosine rows from scripts/run_cpas_ablation.py; "
      "differences below 0.02 R@10 are unresolved.")
