"""Benchmark the trained CPAS combiner against fixed probe composition.

Runs the 14 evaluation queries over the full test split for the trained model
and for the gamma=0.6 probe-direction rule it was initialized from, and reports
the probe-drift diagnostic (signed logit shift on the queried attributes vs.
leakage into the other 38) for both.

Writes results/cpas_results.csv and results/cpas_probe_drift.csv.

Run with: conda run -n clipper python scripts/run_cpas_benchmark.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cpas import CPAS
from src.steering import pad_queries
from src.data import get_paths, load_annotations, load_dataset
from src.evaluation import probe_drift, run_cpas_benchmark, run_probe_benchmark
from src.features import ClipEncoder, load_or_extract
from src.probes import compose_probe, load_probes
from src.retrieval import parse_query

BASELINE_GAMMA = 0.6

REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, default=REPO_ROOT / "results" / "cpas_model.pt")
args = parser.parse_args()

paths = get_paths()
celeba = load_dataset(paths)
annotations = load_annotations(paths)
features = load_or_extract(ClipEncoder(), celeba, paths.features_dir)
assert features.shape[0] == len(celeba), "feature/dataset size mismatch"

directions, biases, attributes = load_probes(REPO_ROOT)
attr_index = {name: i for i, name in enumerate(attributes)}
weights = directions  # drift is measured along the same normalized directions

checkpoint = torch.load(args.model, weights_only=True)
model = CPAS(**checkpoint.get("config", {}))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
val_score = checkpoint.get("val_r10", checkpoint.get("val_recall"))
print(f"Loaded CPAS from epoch {checkpoint['epoch']} (val score {val_score:.4f})")

cpas_df = run_cpas_benchmark(annotations, features, model, directions, attr_index)
probe_df = run_probe_benchmark(
    annotations, features, directions, attr_index, gamma=BASELINE_GAMMA
)

metric_cols = [c for c in cpas_df.columns if c not in ("query", "sources")]
comparison = pd.DataFrame([
    {"method": f"probe composition (gamma={BASELINE_GAMMA})"}
    | probe_df[probe_df["query"] == "MEAN"].iloc[0][metric_cols].to_dict(),
    {"method": "CPAS"}
    | cpas_df[cpas_df["query"] == "MEAN"].iloc[0][metric_cols].to_dict(),
])

drift_rows = []
with torch.no_grad():
    for entry in annotations:
        positives, negatives = parse_query(entry["query"])
        pos_rows = [attr_index[a] for a in positives]
        neg_rows = [attr_index[a] for a in negatives]
        sources = [int(k) for k in entry["ground_truth"].keys()]
        v_ref = features[sources]

        dirs, signs, mask = pad_queries([(pos_rows, neg_rows)] * len(sources), directions)
        queries = {
            "CPAS": model(v_ref, dirs, signs, mask),
            "probe": torch.stack([
                compose_probe(v, directions[pos_rows], directions[neg_rows],
                              gamma=BASELINE_GAMMA)
                for v in v_ref
            ]),
        }
        for method, q in queries.items():
            drift = probe_drift(q, v_ref, weights, biases, pos_rows, neg_rows)
            drift_rows.append({"query": entry["query"], "method": method} | drift)

drift_df = pd.DataFrame(drift_rows)
drift_mean = drift_df.groupby("method")[["queried", "non_queried_abs"]].mean()

out_dir = REPO_ROOT / "results"
cpas_df.to_csv(out_dir / "cpas_results.csv", index=False)
drift_df.to_csv(out_dir / "cpas_probe_drift.csv", index=False)

print("\nRetrieval (MEAN over the 14 queries):")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nProbe drift (higher `queried`, lower `non_queried_abs` is better):")
print(drift_mean.to_string(float_format=lambda x: f"{x:.4f}"))
