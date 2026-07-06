"""Retrieval metrics and the benchmark loop over the evaluation queries."""

from collections.abc import Callable

import pandas as pd
import torch

from src.retrieval import PROMPTS, compose, parse_query, rank

KS = (1, 5, 10)


def evaluate_retrieval(
    retrieved_indices: list[int],
    ground_truth_indices: list[int],
    k: int,
) -> dict:
    """Recall@K (binary hit rate) and Precision@K for a single source image.

    Same semantics as the function provided in the course skeleton.
    """
    top_k = retrieved_indices[:k]
    num_hits = len(set(top_k) & set(ground_truth_indices))
    return {
        f"Recall@{k}": 1 if num_hits > 0 else 0,
        f"Precision@{k}": num_hits / k,
    }


def run_benchmark(
    annotations: list[dict],
    image_features: torch.Tensor,
    encode_texts: Callable[[list[str]], torch.Tensor],
) -> pd.DataFrame:
    """Evaluate the compose+rank baseline on every query in `annotations`.

    image_features: (N, D) L2-normalized features for the whole test split,
        row i = dataset index i.
    encode_texts: maps a list of prompt strings to (M, D) L2-normalized
        embeddings (injected so tests can fake it and the model loads once).
    """
    rows = []
    for entry in annotations:
        positives, negatives = parse_query(entry["query"])
        pos_texts = encode_texts([PROMPTS[a] for a in positives]) if positives \
            else torch.zeros((0, image_features.shape[1]))
        neg_texts = encode_texts([PROMPTS[a] for a in negatives]) if negatives \
            else torch.zeros((0, image_features.shape[1]))

        source_indices = [int(k) for k in entry["ground_truth"].keys()]
        query_vecs = torch.stack([
            compose(image_features[i], pos_texts, neg_texts) for i in source_indices
        ])
        order = rank(query_vecs, image_features, exclude=source_indices)

        sums = {f"R@{k}": 0.0 for k in KS} | {f"P@{k}": 0.0 for k in KS}
        for row_idx, src in enumerate(source_indices):
            retrieved = order[row_idx, : max(KS)].tolist()
            targets = entry["ground_truth"][str(src)]
            for k in KS:
                m = evaluate_retrieval(retrieved, targets, k)
                sums[f"R@{k}"] += m[f"Recall@{k}"]
                sums[f"P@{k}"] += m[f"Precision@{k}"]

        n = len(source_indices)
        rows.append(
            {"query": entry["query"], "sources": n}
            | {key: val / n for key, val in sums.items()}
        )

    df = pd.DataFrame(rows)
    metric_cols = [f"R@{k}" for k in KS] + [f"P@{k}" for k in KS]
    mean_row = {"query": "MEAN", "sources": df["sources"].sum()}
    mean_row |= df[metric_cols].mean().to_dict()
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df[["query", "sources", *metric_cols]]
