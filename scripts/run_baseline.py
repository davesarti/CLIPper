"""Extract full test-split features (cached) and run the 14-query benchmark.

Run with: conda run -n clipper python scripts/run_baseline.py
First run takes ~30-60 min on CPU for feature extraction; later runs load
the cache and finish in seconds.

Pass --num-samples/-n to restrict the image pool to the first N images of
the test split; omit it to use the full split.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_paths, load_annotations, load_dataset
from src.evaluation import negation_subset, run_benchmark
from src.features import ClipEncoder, load_or_extract

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "-n", "--num-samples", type=int, default=None,
    help="Restrict the image pool to the first N images of the test split.",
)
args = parser.parse_args()

paths = get_paths()
celeba = load_dataset(paths)
annotations = load_annotations(paths)
encoder = ClipEncoder()

features = load_or_extract(encoder, celeba, paths.features_dir)
assert features.shape[0] == len(celeba), "feature/dataset size mismatch"

# True test-split labels, used only for scoring (V@10) - never for ranking.
# torchvision's attr_names carries a trailing empty entry; drop it.
labels = celeba.attr.bool()
attr_index = {name: i for i, name in enumerate(n for n in celeba.attr_names if n)}

out_name = "baseline_results.csv"
if args.num_samples is not None:
    n = args.num_samples
    if not 0 < n <= features.shape[0]:
        sys.exit(f"--num-samples must be in (0, {features.shape[0]}], got {n}")
    features = features[:n]
    labels = labels[:n]
    annotations = [
        {**entry, "ground_truth": {
            src: [t for t in targets if t < n]
            for src, targets in entry["ground_truth"].items()
            if int(src) < n
        }}
        for entry in annotations
    ]
    annotations = [entry for entry in annotations if entry["ground_truth"]]
    out_name = f"baseline_results_n{n}.csv"

df = run_benchmark(annotations, features, encoder.encode_texts,
                   labels=labels, attr_index=attr_index)

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / out_name, index=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
mean = df[df["query"] == "MEAN"].iloc[0]
print(f"\nMEAN  R@10 {mean['R@10']:.4f}  P@10 {mean['P@10']:.4f}  "
      f"V@10 {mean['V@10']:.4f}  neg_R@10 {negation_subset(df):.4f}")
print(f"Wrote {out_dir / out_name}")
