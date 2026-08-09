"""Is the S3.1.1 mining rule feasible? Measure before implementing it.

Step 0 of docs/method-proposal-mining-alignment.md, and it is blocking. The
proposed target rule is much stricter than the current one: today a sampled
query is kept when >= 20 pool images satisfy its constraints, under the new rule
it is kept only when >= 5 also land within Hamming 2 of the reference across the
~38 non-queried attributes.

If that rejects most high-k queries, the effective training distribution
silently narrows to frequent attributes - the same class of defect the proposal
exists to fix - and the design has to change rather than the code.

Nothing here trains or writes model state. It reads the same mining pool
train_cpas.py would use, applies the same 90/10 train split, and reports:

  * acceptance rate under the old rule and the new one, per flip count k
  * how many valid targets / violators / drifters a kept query actually has
  * the innermost non-empty drifter shell, which S6 of the proposal samples from
  * the share of kept queries carrying at least one negation
  * per-attribute frequency among kept queries, old rule vs new

Run with:
    conda run -n clipper python scripts/measure_mining_rule.py
    ... scripts/measure_mining_rule.py --samples 2000 --device cuda
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_paths, load_annotations, load_dataset
from src.features import load_pool, resolve_pool
from src.retrieval import parse_query

MAX_HAMMING = 2      # assignment S3.1.1 (2)
MIN_TARGETS = 5      # assignment S3.1.1 inclusion rule
OLD_MIN_CANDIDATES = 20   # src/mining.py today
VAL_FRACTION = 0.1        # must match scripts/train_cpas.py

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--samples", type=int, default=1000,
                    help="sampled references per flip count k")
parser.add_argument("--pool-features", type=Path, default=None,
                    help="feature cache identifying the pool; default: the full "
                         "train split if extracted, else the 30k sample")
parser.add_argument("--device", default=None, help="cuda/cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

paths = get_paths()
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

# The pool is only used for its *indices*: the rule is pure label logic, so the
# features themselves are never touched. Acceptance rates depend on pool size,
# though, so it has to be the pool training would actually mine from.
cache = resolve_pool(paths.features_dir, args.pool_features)
if not cache.is_file():
    sys.exit(f"Missing {cache}: run scripts/extract_train_features.py first.")
_, indices = load_pool(cache)

train = load_dataset(paths, split="train")
attributes = [n for n in train.attr_names if n]
labels = train.attr[indices].bool()

# Same disjoint split as train_cpas.py, so the measured pool is the one the
# miner would sample from and not the whole cache.
perm = torch.randperm(labels.shape[0], generator=torch.Generator().manual_seed(0))
labels = labels[perm[: int(labels.shape[0] * (1 - VAL_FRACTION))]].to(device)
n, a = labels.shape
print(f"Pool: {cache.name} -> {n} images, {a} attributes, on {device}\n")

gen = torch.Generator().manual_seed(args.seed)

# The attributes the 14 graded queries are made of. Retention on *these* is what
# decides whether the stricter rule can be used with uniform flip sampling: they
# are the entangled ones (Male drags facial hair, makeup, lipstick), so they are
# exactly the ones a Hamming-2 budget rejects most often.
index = {name: i for i, name in enumerate(attributes)}
bench_rows = {index[name]
              for entry in load_annotations(paths)
              for side in parse_query(entry["query"])
              for name in side}
print(f"Benchmark attributes ({len(bench_rows)}): "
      f"{', '.join(sorted(attributes[r] for r in bench_rows))}\n")


def sample_flips(ref: int, k: int) -> tuple[list[int], list[int]]:
    """Same rule as TripletMiner._sample_flips: k random attributes, split by state."""
    state = labels[ref]
    rows = torch.randperm(a, generator=gen)[:k]
    add = [int(r) for r in rows if not state[r]]
    remove = [int(r) for r in rows if state[r]]
    return add, remove


def satisfies(add: list[int], remove: list[int]) -> torch.Tensor:
    ok = torch.ones(n, dtype=torch.bool, device=device)
    if add:
        ok &= labels[:, add].all(dim=1)
    if remove:
        ok &= ~labels[:, remove].any(dim=1)
    return ok


def hamming_to(ref: int, queried: set[int]) -> torch.Tensor:
    """Disagreements with `ref` over the NON-queried attributes."""
    others = [i for i in range(a) if i not in queried]
    rest = labels[:, others]
    return (rest != rest[ref]).sum(dim=1)


def quantiles(values: list[int]) -> str:
    if not values:
        return "n/a"
    t = torch.tensor(values, dtype=torch.float)
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75]))
    return f"{int(q[0])} / {int(q[1])} / {int(q[2])}"


rows_out = []
freq_old: dict[int, Counter] = {}
freq_new: dict[int, Counter] = {}

for k in (1, 2, 3):
    kept_old = kept_new = with_negation = 0
    kept_old_b = kept_new_b = 0     # queries touching >= 1 benchmark attribute
    n_valid, n_viol, n_drift, shells = [], [], [], []
    freq_old[k], freq_new[k] = Counter(), Counter()

    for _ in range(args.samples):
        ref = int(torch.randint(n, (1,), generator=gen))
        add, remove = sample_flips(ref, k)
        if not add and not remove:
            continue

        s = satisfies(add, remove)
        s[ref] = False
        queried = set(add) | set(remove)

        touches_bench = bool(queried & bench_rows)
        old_ok = int(s.sum()) >= OLD_MIN_CANDIDATES
        if old_ok:
            kept_old += 1
            kept_old_b += touches_bench
            for r in queried:
                freq_old[k][r] += 1

        h = hamming_to(ref, queried)
        inside = h <= MAX_HAMMING
        valid = s & inside
        valid[ref] = False
        if int(valid.sum()) < MIN_TARGETS:
            continue

        # Kept under the new rule.
        kept_new += 1
        kept_new_b += touches_bench
        if remove:
            with_negation += 1
        for r in queried:
            freq_new[k][r] += 1

        violators = (~s) & inside
        violators[ref] = False
        drifters = s & ~inside

        n_valid.append(int(valid.sum()))
        n_viol.append(int(violators.sum()))
        n_drift.append(int(drifters.sum()))
        if int(drifters.sum()):
            shells.append(int(h[drifters].min()))

    rows_out.append({
        "k": k,
        "old %": f"{100 * kept_old / args.samples:.1f}",
        "new %": f"{100 * kept_new / args.samples:.1f}",
        # Retention = share of today's accepted queries that survive the new
        # rule, split by whether the flip set touches a graded attribute.
        "keep% bench": f"{100 * kept_new_b / max(kept_old_b, 1):.1f}",
        "keep% other": f"{100 * (kept_new - kept_new_b) / max(kept_old - kept_old_b, 1):.1f}",
        "negation %": f"{100 * with_negation / max(kept_new, 1):.1f}",
        "valid q1/med/q3": quantiles(n_valid),
        "violators": quantiles(n_viol),
        "drifters": quantiles(n_drift),
        "shell med": quantiles(shells).split(" / ")[1] if shells else "n/a",
    })

print(f"{args.samples} sampled references per k. 'old %' = kept by today's rule "
      f"(>= {OLD_MIN_CANDIDATES} satisfy the constraints); 'new %' = kept by "
      f"S3.1.1 (>= {MIN_TARGETS} also within Hamming {MAX_HAMMING}).\n")
header = list(rows_out[0].keys())
print(" | ".join(f"{c:>16}" for c in header))
print("-" * (19 * len(header)))
for row in rows_out:
    print(" | ".join(f"{str(row[c]):>16}" for c in header))

pooled_old, pooled_new = Counter(), Counter()
for k in (1, 2, 3):
    pooled_old += freq_old[k]
    pooled_new += freq_new[k]
total_old = max(sum(pooled_old.values()), 1)
total_new = max(sum(pooled_new.values()), 1)


def frequency_table(rows: list[int], title: str) -> None:
    """Share of kept queries touching each attribute, old rule vs new."""
    print(f"\n{title}")
    print(f"{'attribute':>22} | {'old %':>7} | {'new %':>7} | {'kept %':>7}")
    print("-" * 54)
    for r in rows:
        kept = 100 * pooled_new[r] / max(pooled_old[r], 1)
        print(f"{attributes[r]:>22} | {100 * pooled_old[r] / total_old:7.2f} | "
              f"{100 * pooled_new[r] / total_new:7.2f} | {kept:7.1f}")


print("\n" + "=" * 54)
print("'kept %' is the retention: of the queries touching this attribute that\n"
      "today's rule accepts, how many survive the S3.1.1 rule. This is the\n"
      "number that sets how hard the flip sampler has to be steered.")

frequency_table(sorted(bench_rows, key=lambda r: -pooled_new[r]),
                "GRADED attributes - the 14 benchmark queries are made of these")

others = [r for r, _ in pooled_new.most_common() if r not in bench_rows][:10]
frequency_table(others, "Top 10 non-graded attributes by new-rule frequency")

bench_kept = sum(pooled_new[r] for r in bench_rows)
bench_old = sum(pooled_old[r] for r in bench_rows)
other_kept = total_new - bench_kept
other_old = total_old - bench_old
print(f"\nOverall retention: graded {100 * bench_kept / max(bench_old, 1):.1f}% "
      f"vs non-graded {100 * other_kept / max(other_old, 1):.1f}%.")
print("A large gap means uniform flip sampling lets the rejection step choose\n"
      "the training distribution for us, and choose it away from what is graded.")

print("\nRead this against S7 of docs/method-proposal-mining-alignment.md before "
      "writing any training code.")
