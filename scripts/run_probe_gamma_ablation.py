"""Gamma ablation on probe-direction composition, query attributes only.

Runs the 14-query benchmark with q = normalize(gamma * v_ref + sum(pos
probe dirs) - sum(neg probe dirs)) for several gamma values, using the
saved linear-probe weights (results/probe_weights.pt) as attribute
directions. Writes the per-gamma MEAN metrics to
results/probe_gamma_ablation.csv and the per-query table at the best
gamma; compare against the other methods' result CSVs manually.

Run with: conda run -n clipper python scripts/run_probe_gamma_ablation.py
"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_paths, load_annotations, load_dataset
from src.evaluation import run_probe_benchmark
from src.features import ClipEncoder, load_or_extract

GAMMAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]

paths = get_paths()
celeba = load_dataset(paths)
annotations = load_annotations(paths)
encoder = ClipEncoder()

features = load_or_extract(encoder, celeba, paths.features_dir)
assert features.shape[0] == len(celeba), "feature/dataset size mismatch"

out_dir = Path(__file__).resolve().parent.parent / "results"
saved = torch.load(out_dir / "probe_weights.pt", map_location="cpu", weights_only=False)
w = saved["weights"]
directions = w / w.norm(dim=1, keepdim=True)
attr_index = {name: i for i, name in enumerate(saved["attributes"])}

summary_rows = []
best = None
for gamma in GAMMAS:
    df = run_probe_benchmark(annotations, features, directions, attr_index, gamma=gamma)
    mean = df[df["query"] == "MEAN"].iloc[0]
    summary_rows.append({"method": f"probe gamma={gamma:g}"} | {
        c: mean[c] for c in df.columns if c not in ("query", "sources")
    })
    print(f"gamma={gamma:g}  R@10={mean['R@10']:.4f}")
    if best is None or mean["R@10"] > best[1]:
        best = (gamma, mean["R@10"], df)

summary = pd.DataFrame(summary_rows)
summary.to_csv(out_dir / "probe_gamma_ablation.csv", index=False)
best[2].to_csv(out_dir / f"probe_gamma{best[0]:g}_results.csv", index=False)

print(f"\nBest gamma by R@10: {best[0]:g}")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
