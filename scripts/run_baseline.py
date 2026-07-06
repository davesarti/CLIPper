"""Extract full test-split features (cached) and run the 14-query benchmark.

Run with: conda run -n clipper python scripts/run_baseline.py
First run takes ~30-60 min on CPU for feature extraction; later runs load
the cache and finish in seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_paths, load_annotations, load_dataset
from src.evaluation import run_benchmark
from src.features import ClipEncoder, load_or_extract

paths = get_paths()
celeba = load_dataset(paths)
annotations = load_annotations(paths)
encoder = ClipEncoder()

features = load_or_extract(encoder, celeba, paths.features_dir)
assert features.shape[0] == len(celeba), "feature/dataset size mismatch"

df = run_benchmark(annotations, features, encoder.encode_texts)

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "baseline_results.csv", index=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
