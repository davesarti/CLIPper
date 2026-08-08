"""Extract and cache CLIP features for a non-test split.

Everything downstream of the baseline needs these files and nothing else in the
repo produces them: scripts/fit_probes.py reads
features/clip-vit-base-patch32_train30k.pt to fit the 40 probes,
scripts/train_cpas.py mines its triplets from either that sample or the
full-split features/clip-vit-base-patch32_train.pt, and
scripts/run_probe_accuracy.py scores the probes on
features/clip-vit-base-patch32_valid.pt.

The saved dict is {"features", "indices"}: the indices are rows of the *split*
(not dataset indices), which is how fit_probes, train_cpas and run_probe_accuracy
realign the CelebA attribute labels with the cached features.

Run with:
    conda run -n clipper python scripts/extract_train_features.py            # 30k train sample
    conda run -n clipper python scripts/extract_train_features.py --all      # full train split
    conda run -n clipper python scripts/extract_train_features.py --split valid --all
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_paths, load_dataset
from src.features import ClipEncoder

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--split", choices=("train", "valid"), default="train",
                    help="split to encode; the test split is the retrieval "
                         "database and is cached by the benchmark scripts")
parser.add_argument("-n", "--num-samples", type=int, default=30_000,
                    help="images to sample from the train split (default 30000, "
                         "the size fit_probes.py expects)")
parser.add_argument("--all", action="store_true",
                    help="use the whole train split instead of a sample; heavier "
                         "mining pool for train_cpas.py")
parser.add_argument("--batch-size", type=int, default=256,
                    help="encoder batch size; lower it if the GPU runs out of memory")
parser.add_argument("--seed", type=int, default=0,
                    help="sampling seed; fixed so re-runs cache the same subset")
parser.add_argument("--out", type=Path, default=None,
                    help="output path (default: features/<model>_train[30k].pt)")
args = parser.parse_args()

paths = get_paths()
split = load_dataset(paths, split=args.split)

if args.all:
    indices = torch.arange(len(split))
else:
    if not 0 < args.num_samples <= len(split):
        sys.exit(f"--num-samples must be in (0, {len(split)}], got {args.num_samples}")
    gen = torch.Generator().manual_seed(args.seed)
    # Sorted so the subset is read in file order: sequential JPEG reads are
    # markedly faster than random ones, and sorting cannot bias the sample.
    indices = torch.randperm(len(split), generator=gen)[: args.num_samples].sort().values

slug = ClipEncoder.MODEL_NAME.split("/")[-1]
suffix = args.split if args.all else f"{args.split}{len(indices) // 1000}k"
out_path = args.out or paths.features_dir / f"{slug}_{suffix}.pt"
if out_path.is_file():
    sys.exit(f"{out_path} already exists; delete it to re-extract.")

encoder = ClipEncoder()
print(f"Encoding {len(indices)} {args.split}-split images on {encoder.device}...")
features = encoder.encode_images(Subset(split, indices.tolist()),
                                 batch_size=args.batch_size)
assert features.shape[0] == len(indices), "feature/index count mismatch"

out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({"features": features, "indices": indices}, out_path)
print(f"Saved {tuple(features.shape)} features to {out_path}")
