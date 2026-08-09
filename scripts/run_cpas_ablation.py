"""Benchmark a set of trained CPAS checkpoints against the fixed probe rule.

Each checkpoint is evaluated on the 14 queries over the full test split and on
the probe-drift diagnostic; rows are averaged over the seeds sharing a variant
name. Checkpoints are passed as name=path pairs, e.g.

    conda run -n clipper python scripts/run_cpas_ablation.py \
        "CPAS (delta=0.3)=runs/d0.3_s0.pt" "CPAS (delta=0.3)=runs/d0.3_s1.pt"

Writes results/cpas_ablation.csv (per checkpoint) and prints the variant means.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cpas_mlp import PerAttributeMLP
from src.steering import pad_queries
from src.data import get_paths, load_annotations, load_dataset
from src.evaluation import (
    negation_subset,
    probe_drift,
    run_cpas_benchmark,
    run_probe_benchmark,
)
from src.features import ClipEncoder, load_or_extract
from src.probes import compose_probe, load_probes
from src.retrieval import parse_query

BASELINE_GAMMA = 0.6
REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", nargs="+", metavar="NAME=PATH",
                    help="variant name and checkpoint path")
args = parser.parse_args()

paths = get_paths()
celeba = load_dataset(paths)
annotations = load_annotations(paths)
features = load_or_extract(ClipEncoder(), celeba, paths.features_dir)
directions, biases, attributes = load_probes(REPO_ROOT)
attr_index = {name: i for i, name in enumerate(attributes)}
metric_cols = ["R@1", "R@5", "R@10", "P@1", "P@5", "P@10", "V@10"]
labels = celeba.attr.bool()  # test-split labels, for the violation rate


@torch.no_grad()
def drift_of(query_fn) -> dict[str, float]:
    """Mean probe drift over the benchmark queries for a query builder."""
    totals = {"queried": 0.0, "non_queried_abs": 0.0}
    for entry in annotations:
        positives, negatives = parse_query(entry["query"])
        pos_rows = [attr_index[a] for a in positives]
        neg_rows = [attr_index[a] for a in negatives]
        sources = [int(k) for k in entry["ground_truth"].keys()]
        v_ref = features[sources]
        drift = probe_drift(
            query_fn(v_ref, pos_rows, neg_rows), v_ref,
            directions, biases, pos_rows, neg_rows,
        )
        for key in totals:
            totals[key] += drift[key] / len(annotations)
    return totals


def probe_queries(v_ref, pos_rows, neg_rows):
    return torch.stack([
        compose_probe(v, directions[pos_rows], directions[neg_rows],
                      gamma=BASELINE_GAMMA)
        for v in v_ref
    ])


rows = []
probe_df = run_probe_benchmark(
    annotations, features, directions, attr_index, gamma=BASELINE_GAMMA,
    labels=labels,
)
rows.append(
    {"variant": f"probe composition (gamma={BASELINE_GAMMA})", "checkpoint": "-", "seed": -1}
    | probe_df[probe_df["query"] == "MEAN"].iloc[0][metric_cols].to_dict()
    | {"neg_R@10": negation_subset(probe_df)}
    | drift_of(probe_queries)
)

for spec in args.runs:
    name, _, path = spec.rpartition("=")  # variant names may contain "="
    checkpoint = torch.load(path, weights_only=True)
    config = dict(checkpoint.get("config", {}))
    config.pop("arch", None)  # older checkpoints tag the architecture
    model = PerAttributeMLP(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    df = run_cpas_benchmark(annotations, features, model, directions, attr_index,
                            labels=labels)
    with torch.no_grad():
        drift = drift_of(lambda v, p, n: model(v, *pad_queries([(p, n)] * len(v), directions)))
    rows.append(
        {"variant": name, "checkpoint": Path(path).name,
         "seed": checkpoint.get("seed", -1),
         "val_recall": checkpoint.get("val_r10", checkpoint.get("val_recall")),
         "epoch": checkpoint.get("epoch")}
        | df[df["query"] == "MEAN"].iloc[0][metric_cols].to_dict()
        | {"neg_R@10": negation_subset(df)}
        | drift
    )
    print(f"{name:34s} seed {rows[-1]['seed']}  R@10 {rows[-1]['R@10']:.4f}")

table = pd.DataFrame(rows)
out_dir = REPO_ROOT / "results"
out_dir.mkdir(exist_ok=True)
table.to_csv(out_dir / "cpas_ablation.csv", index=False)

summary = table.groupby("variant", sort=False)[
    [*metric_cols, "neg_R@10", "queried", "non_queried_abs"]
].mean()
print("\nMEAN over the 14 queries, averaged over seeds:")
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\nWrote {out_dir / 'cpas_ablation.csv'}")
