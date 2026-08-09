"""Train the CPAS combiner on mined attribute-flip triplets (GPU-oriented).

Differences from the first CPU run, all aimed at the actual objective -
Recall@10 on the 14 evaluation queries:

* **Checkpoint selection is R@10, not the mining proxy.** Every epoch the model
  is scored on a held-out val benchmark that mirrors the eval ground-truth rule
  (constraint satisfaction + identity-proxy match) over val-split references and
  the same 14 query shapes. The saved checkpoint is the epoch with the best val
  R@10. The old val-triplet recall@1 is kept only as a diagnostic - it tracks
  the true metric poorly, which is why it is no longer the selection signal.
* **Heavier mining.** Triplets are re-mined every epoch (`--remine-every`) from
  the full train pool if its features are available, so the model never sees the
  same synthetic edit twice and targets are drawn from many more candidates.
* **GPU defaults.** Larger batch (more in-batch InfoNCE negatives), features and
  mining kept on device.

Extract full-pool features first for the heaviest setting (falls back to the
30k sample if absent):
    features/clip-vit-base-patch32_train.pt   # {"features","indices"} or a tensor

Run (GPU):  conda run -n clipper python scripts/train_cpas.py
Smoke:      ... scripts/train_cpas.py --warmup 1 --epochs 1 --triplets 400 \
                --val-per-query 20 --batch-size 128
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cpas_mlp import PerAttributeMLP
from src.data import get_paths, load_annotations, load_dataset
from src.evaluation import build_val_benchmark, score_val_benchmark
from src.features import ClipEncoder, load_pool, resolve_pool
from src.mining import TripletMiner, proxy_rows
from src.probes import load_probes
from src.retrieval import parse_query
from src.training import run_epoch

VAL_FRACTION = 0.1

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--warmup", type=int, default=5, help="epochs on k=1 only")
parser.add_argument("--epochs", type=int, default=45, help="epochs on k in {1,2,3}")
parser.add_argument("--triplets", type=int, default=40_000,
                    help="training triplets mined per (re-)mining")
parser.add_argument("--remine-every", type=int, default=1,
                    help="re-mine fresh triplets every N epochs (1 = every epoch)")
parser.add_argument("--batch-size", type=int, default=1024,
                    help="larger batch = more in-batch InfoNCE negatives")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--val-per-query", type=int, default=200,
                    help="held-out references per query in the val R@10 benchmark")
parser.add_argument("--delta-max", type=float, default=0.3,
                    help="bound on the direction bend; 0 leaves only rescaling")
parser.add_argument("--rank", type=int, default=32,
                    help="rank of the delta factorization")
parser.add_argument("--no-cross-attributes", dest="no_cross_attributes",
                    action="store_true",
                    help="ablation: attributes cannot condition on each other "
                         "(zeroes the pooled context)")
parser.add_argument("--patience", type=int, default=15,
                    help="stop after this many main-phase epochs without a val "
                         "R@10 improvement; 0 disables early stopping")
parser.add_argument("--seed", type=int, default=0, help="model init seed")
parser.add_argument("--pool-features", type=Path, default=None,
                    help="feature file for the mining pool; default: full train "
                         "split if present, else the 30k sample")
parser.add_argument("--out", type=Path, default=None, help="checkpoint path")
args = parser.parse_args()
torch.manual_seed(args.seed)

REPO_ROOT = Path(__file__).resolve().parent.parent
paths = get_paths()
device = "cuda" if torch.cuda.is_available() else "cpu"
slug = ClipEncoder.MODEL_NAME.split("/")[-1]


pool_path = resolve_pool(paths.features_dir, args.pool_features)
if not pool_path.is_file():
    sys.exit(f"Missing {pool_path}: extract the train-split features first.")
features, indices = load_pool(pool_path)
print(f"Mining pool: {pool_path.name} ({features.shape[0]} images) on {device}")

train_split = load_dataset(paths, split="train")
labels = train_split.attr[indices].bool()
attributes = [n for n in train_split.attr_names if n]
directions, _, probe_attributes = load_probes(REPO_ROOT)
assert probe_attributes == attributes, "probe/label attribute order mismatch"
attr_index = {name: i for i, name in enumerate(attributes)}

# Disjoint image pools: a validation reference is never a training candidate.
perm = torch.randperm(features.shape[0], generator=torch.Generator().manual_seed(0))
split = int(features.shape[0] * (1 - VAL_FRACTION))
pools = {"train": perm[:split], "val": perm[split:]}
rows = proxy_rows(attributes)
train_pool = features[pools["train"]].to(device)
val_pool = features[pools["val"]].to(device)
train_miner = TripletMiner(labels[pools["train"]], train_pool, rows, seed=args.seed)

# Held-out val benchmark, built once (ground truth does not depend on the model).
annotations = load_annotations(paths)
query_specs = [
    ([attr_index[a] for a in parse_query(e["query"])[0]],
     [attr_index[a] for a in parse_query(e["query"])[1]])
    for e in annotations
]
val_tasks = build_val_benchmark(
    labels[pools["val"]], query_specs, directions,
    per_query=args.val_per_query, seed=args.seed,
)
print(f"Val R@10 benchmark: {len(val_tasks)} queries, "
      f"{sum(t['refs'].shape[0] for t in val_tasks)} held-out references")

config = {"delta_max": args.delta_max, "rank": args.rank,
          "cross_attributes": not args.no_cross_attributes}
model = PerAttributeMLP(**config).to(device)
print(f"CPAS-MLP: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
      f"trainable parameters")
directions = directions.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

out_path = args.out or REPO_ROOT / "results" / "cpas_model.pt"
out_path.parent.mkdir(exist_ok=True)


def save_checkpoint(best: dict) -> None:
    """Write the best-so-far weights, atomically: a crash mid-write keeps the
    previous checkpoint intact rather than truncating it."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(
        {"state_dict": best["state"], "attributes": attributes,
         "val_r10": best["r10"], "epoch": best["epoch"],
         "config": config,
         "seed": args.seed},
        tmp,
    )
    tmp.replace(out_path)


best = {"r10": -1.0, "state": None, "epoch": -1}
triplets = None
# Checkpointing on every improvement means an interrupted run (Ctrl-C, dropped
# ssh, OOM) still leaves the best epoch on disk. Note this keeps the weights
# only: no optimizer or RNG state, so a run can be kept but not resumed.
try:
    for epoch in range(args.warmup + args.epochs):
        phase, ks = ("warmup", (1,)) if epoch < args.warmup else ("main", (1, 2, 3))
        if triplets is None or epoch % args.remine_every == 0:
            triplets = train_miner.sample_batch(args.triplets, ks=ks)

        loss, proxy_r1 = run_epoch(
            model, triplets, train_pool, directions,
            optimizer=optimizer, batch_size=args.batch_size, device=device,
        )
        val_r10 = score_val_benchmark(model, val_pool, val_tasks, k=10)
        print(f"epoch {epoch:3d} [{phase}]  loss {loss:.4f}  proxy r@1 {proxy_r1:.3f}  "
              f"| val R@10 {val_r10:.4f}", flush=True)
        if val_r10 > best["r10"]:
            best = {
                "r10": val_r10,
                "state": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()},
                "epoch": epoch,
            }
            save_checkpoint(best)
        # Warmup trains on a different (k=1 only) distribution, so its epochs
        # never count against patience: the clock starts with the main phase.
        elif args.patience and phase == "main":
            stale = epoch - max(best["epoch"], args.warmup - 1)
            if stale >= args.patience:
                print(f"early stop: no val R@10 improvement in {stale} epochs "
                      f"(best epoch {best['epoch']})")
                break
except KeyboardInterrupt:
    print("\ninterrupted; keeping the best checkpoint so far")

if best["state"] is None:
    sys.exit("no epoch completed: nothing to save")
save_checkpoint(best)
print(f"\nBest epoch {best['epoch']} (val R@10 {best['r10']:.4f}) saved to {out_path}")
