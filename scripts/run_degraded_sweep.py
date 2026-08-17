"""Does the cosine term earn its weight only when the attribute code is unreliable?

Section 8 of the report argues that the third term of the score can only help
insofar as it is a second, coarser estimate of the attribute code - so its
optimal weight should grow as the predictor degrades and vanish as it improves.
The right-hand end of that curve is already measured: with the MLP head the
sweep picks w = 1 for the composed query and effectively 0 for the raw
reference. This script builds the rest of the curve by corrupting the predictor
at increasing rates and re-selecting w each time.

Everything happens on the held-out validation slice of the training pool, never
on the 14 test queries: the claim is about *what the sweep selects*, which is a
validation-side statement by construction.

Two query vectors are swept, because they are different claims:

    v_ref  - the raw reference embedding. No fusion, no parameters, and it does
             not know the query, so on a queried attribute it points at exactly
             the wrong value.
    cpas   - the composed query from CPAS-MLP, i.e. the reference moved toward
             the requested attributes.

Both are reported at every corruption level. If the two curves separate, the
fusion module carries something the raw reference does not; if they coincide,
it does not, and that is the honest result.

The full (corruption, seed, source, w) grid is written out rather than just the
argmax: with a validation SE around 0.01, argmax over a flat curve is a noise
generator, and the shape is what carries the evidence.

Run:  conda run -n clipper python scripts/run_degraded_sweep.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attribute_head import load_attribute_head
from src.attribute_retrieval import constraint_violation, expected_hamming
from src.cpas_mlp import PerAttributeMLP
from src.criterion import MAX_HAMMING
from src.data import get_paths, load_annotations, load_dataset
from src.features import load_pool, resolve_pool
from src.probes import load_probes, load_raw_probes
from src.retrieval import parse_query
from src.steering import pad_queries

torch.set_grad_enabled(False)
REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_FRACTION = 0.1          # must match scripts/run_attribute_retrieval.py
MIN_GT = 3                  # a reference with fewer valid targets is skipped

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--head", type=Path,
                    default=REPO_ROOT / "results" / "attribute_head.pt")
parser.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "results" / "mlp_final_s0.pt")
parser.add_argument("--corruption", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
                    help="per-entry probability of replacing a predicted "
                         "probability with a uniform draw")
parser.add_argument("--w-grid", type=float, nargs="+",
                    default=[0.0, 1.0, 3.0, 10.0, 30.0],
                    help="30 is included because w* is expected to grow; if the "
                         "curve saturates there, widen it")
parser.add_argument("--lam", type=float, default=4.0,
                    help="held fixed: the report shows the constraint penalty is "
                         "saturated, and sweeping both knobs would confound them")
parser.add_argument("--seeds", type=int, default=3,
                    help="corruption draws per level; one draw makes both axes "
                         "noisy, and this is cheap")
parser.add_argument("--val-refs", type=int, default=200)
parser.add_argument("--hard-reference", action="store_true",
                    help="use thresholded reference bits. Default is the soft "
                         "reference, which is the delivered configuration")
parser.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "degraded_predictor_sweep.csv")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
paths = get_paths()
directions, _, attributes = load_probes(REPO_ROOT)
W, B, _ = load_raw_probes(REPO_ROOT)
attr_index = {name: i for i, name in enumerate(attributes)}
A = len(attributes)

# ---------------------------------------------------- the validation slice
# Rebuilt exactly as run_attribute_retrieval.py does, so a number here is
# comparable with the sweep that produced the reported configuration.
pool_path = resolve_pool(paths.features_dir)
pool_features, pool_indices = load_pool(pool_path)
pool_labels = load_dataset(paths, split="train").attr[pool_indices].bool()
perm = torch.randperm(pool_features.shape[0],
                      generator=torch.Generator().manual_seed(0))
val_rows = perm[int(pool_features.shape[0] * (1 - VAL_FRACTION)):]
val_features = pool_features[val_rows].to(device)
val_labels = pool_labels[val_rows].to(device)
gen = torch.Generator().manual_seed(0)
val_refs = torch.randperm(val_features.shape[0], generator=gen)[: args.val_refs]
val_refs = val_refs.to(device)
print(f"val pool {val_features.shape[0]} images, {len(val_refs)} references, {device}")

# ------------------------------------------------------------- predictors
head, saved = load_attribute_head(args.head)
head = head.to(device).eval()
head_thresholds = saved.get("thresholds")
head_thresholds = (0.5 if head_thresholds is None
                   else head_thresholds.to(device))
clean_probs = torch.sigmoid(head(val_features))
probe_probs = torch.sigmoid(val_features @ W.to(device).T + B.to(device))
print(f"head {args.head.name}: val bit accuracy "
      f"{saved.get('val_bit_accuracy', float('nan')):.4f}")

checkpoint = torch.load(args.checkpoint, weights_only=True)
config = dict(checkpoint.get("config", {}))
config.pop("arch", None)
combiner = PerAttributeMLP(**config)
combiner.load_state_dict(checkpoint["state_dict"])
combiner = combiner.to(device).eval()
print(f"combiner {args.checkpoint.name}: epoch {checkpoint.get('epoch')}")

# --------------------------------------------- per-query constants, once
# Ground truth, the queried rows and both cosine matrices depend only on the
# labels and the features, never on the corruption, so they are hoisted out of
# the grid. This is what keeps the whole sweep to minutes.
tasks = []
for entry in load_annotations(paths):
    pos, neg = parse_query(entry["query"])
    pr = [attr_index[a] for a in pos]
    nr = [attr_index[a] for a in neg]
    others = [a for a in range(A) if a not in set(pr) | set(nr)]

    sat = torch.ones(val_features.shape[0], dtype=torch.bool, device=device)
    if pr:
        sat &= val_labels[:, pr].all(dim=1)
    if nr:
        sat &= ~val_labels[:, nr].any(dim=1)
    rest = val_labels[:, others]

    gt, keep = [], []
    for row, r in enumerate(val_refs.tolist()):
        mask = sat & ((rest != rest[r]).sum(dim=1) <= MAX_HAMMING)
        mask[r] = False
        if int(mask.sum()) >= MIN_GT:
            gt.append(mask)
            keep.append(row)
    if not keep:
        continue
    refs = val_refs[torch.tensor(keep, device=device)]

    dirs, signs, mask_k = pad_queries([(pr, nr)] * len(refs), directions)
    q = combiner(val_features[refs], dirs.to(device), signs.to(device),
                 mask_k.to(device))
    tasks.append({
        "pr": pr, "nr": nr, "others": others, "refs": refs,
        "gt": torch.stack(gt),                             # (R, N)
        "v_ref": val_features @ val_features[refs].T,      # (N, R)
        "cpas": val_features @ q.T,                        # (N, R)
    })
print(f"{len(tasks)} queries, "
      f"{sum(t['refs'].shape[0] for t in tasks)} graded (query, reference) pairs")


def recall_at_10(probs, code, source, w_cos) -> float:
    """Mean R@10 over the val benchmark for one predictor and one weight."""
    hits = total = 0
    for t in tasks:
        ref_code = (code if args.hard_reference else probs)[t["refs"]].clone()
        ref_code = ref_code.to(probs.dtype)
        if t["pr"]:
            ref_code[:, t["pr"]] = 1.0
        if t["nr"]:
            ref_code[:, t["nr"]] = 0.0

        score = -expected_hamming(probs, ref_code, t["others"])          # (N, R)
        score -= args.lam * constraint_violation(code, t["pr"], t["nr"]).unsqueeze(1)
        if w_cos:
            score = score + w_cos * t[source]
        # Only the top 10 matter, so never sort the whole database.
        score = score.T                                                  # (R, N)
        rows = torch.arange(score.shape[0], device=device)
        score[rows, t["refs"]] = float("-inf")
        top = score.topk(10, dim=1).indices
        hits += int(t["gt"].gather(1, top).any(dim=1).sum())
        total += score.shape[0]
    return hits / max(total, 1)


def bit_accuracy(code) -> float:
    return float((code == val_labels).float().mean())


rows = []


def sweep(tag: str, probs, code, corruption, seed):
    acc = bit_accuracy(code)
    for w in args.w_grid:
        # With w = 0 the query vector never enters the score, so the two
        # sources would produce identical numbers; record it once, unlabelled.
        for source in (["none"] if w == 0 else ["v_ref", "cpas"]):
            r10 = recall_at_10(probs, code, source, w)
            rows.append({"predictor": tag, "corruption": corruption,
                         "seed": seed, "bit_accuracy": acc,
                         "q_source": source, "w_cos": w, "val_R10": r10})
            # The anchor carries no corruption level, so it cannot be formatted
            # as a number.
            c_txt = "-" if corruption is None else f"{corruption:.2f}"
            print(f"  {tag:12s} c={c_txt:<5} s={seed:<2} {source:6s} "
                  f"w={w:<5} acc={acc:.4f} R@10={r10:.4f}", flush=True)


# Anchor: a genuinely weaker predictor, not a corrupted one. It is a different
# architecture trained on a different pool, so it belongs on the plot as a
# reference point and not as a sample of the corruption curve.
print("\n== anchor: linear probe (uncorrupted) ==")
sweep("linear probe", probe_probs, probe_probs > 0.5, corruption=None, seed=-1)

print("\n== MLP head, corrupted ==")
for corruption in args.corruption:
    for seed in range(args.seeds if corruption > 0 else 1):
        if corruption == 0:
            probs = clean_probs
        else:
            g = torch.Generator(device="cpu").manual_seed(1000 + seed)
            draw = torch.rand(clean_probs.shape, generator=g).to(device)
            hit = torch.rand(clean_probs.shape, generator=g).to(device) < corruption
            probs = torch.where(hit, draw, clean_probs)
        sweep("MLP head", probs, probs > head_thresholds, corruption, seed)

table = pd.DataFrame(rows)
args.out.parent.mkdir(exist_ok=True)
table.to_csv(args.out, index=False)
print(f"\nWrote {args.out}  ({len(table)} rows)")

best = (table[table.q_source != "none"]
        .sort_values("val_R10", ascending=False)
        .groupby(["predictor", "corruption", "seed", "q_source"], dropna=False)
        .first().reset_index())
print("\nBest w per condition (read the CSV for the full curves):")
print(best[["predictor", "corruption", "seed", "bit_accuracy",
            "q_source", "w_cos", "val_R10"]].to_string(index=False))
