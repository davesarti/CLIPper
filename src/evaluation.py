"""Retrieval metrics and the benchmark loop over the evaluation queries."""

from collections.abc import Callable

import pandas as pd
import torch

from src.caption import encode_caption, flip_state, render_caption
from src.cpas import CPAS, pad_queries
from src.probes import compose_probe
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


def _query_row(entry: dict, order: torch.Tensor, source_indices: list[int]) -> dict:
    """Per-query metrics row: R@K / P@K averaged over the source images."""
    sums = {f"R@{k}": 0.0 for k in KS} | {f"P@{k}": 0.0 for k in KS}
    for row_idx, src in enumerate(source_indices):
        retrieved = order[row_idx, : max(KS)].tolist()
        targets = entry["ground_truth"][str(src)]
        for k in KS:
            m = evaluate_retrieval(retrieved, targets, k)
            sums[f"R@{k}"] += m[f"Recall@{k}"]
            sums[f"P@{k}"] += m[f"Precision@{k}"]
    n = len(source_indices)
    return {"query": entry["query"], "sources": n} | {
        key: val / n for key, val in sums.items()
    }


def _with_mean_row(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [f"R@{k}" for k in KS] + [f"P@{k}" for k in KS]
    mean_row = {"query": "MEAN", "sources": df["sources"].sum()}
    mean_row |= df[metric_cols].mean().to_dict()
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df[["query", "sources", *metric_cols]]


def run_benchmark(
    annotations: list[dict],
    image_features: torch.Tensor,
    encode_texts: Callable[[list[str]], torch.Tensor],
    gamma: float = 1.0,
) -> pd.DataFrame:
    """Evaluate the compose+rank baseline on every query in `annotations`.

    image_features: (N, D) L2-normalized features for the whole test split,
        row i = dataset index i.
    encode_texts: maps a list of prompt strings to (M, D) L2-normalized
        embeddings (injected so tests can fake it and the model loads once).
    gamma: reference-image weight passed through to compose.
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
            compose(image_features[i], pos_texts, neg_texts, gamma=gamma)
            for i in source_indices
        ])
        order = rank(query_vecs, image_features, exclude=source_indices)
        rows.append(_query_row(entry, order, source_indices))

    return _with_mean_row(rows)


def run_probe_benchmark(
    annotations: list[dict],
    image_features: torch.Tensor,
    directions: torch.Tensor,
    attr_index: dict[str, int],
    gamma: float = 1.0,
) -> pd.DataFrame:
    """Evaluate compose_probe+rank on every query in `annotations`.

    directions: (A, D) L2-normalized probe weight directions;
    attr_index: attribute name -> row of `directions`.
    Only the attributes named in each query are used.
    """
    rows = []
    for entry in annotations:
        positives, negatives = parse_query(entry["query"])
        pos_dirs = directions[[attr_index[a] for a in positives]]
        neg_dirs = directions[[attr_index[a] for a in negatives]]

        source_indices = [int(k) for k in entry["ground_truth"].keys()]
        query_vecs = torch.stack([
            compose_probe(image_features[i], pos_dirs, neg_dirs, gamma=gamma)
            for i in source_indices
        ])
        order = rank(query_vecs, image_features, exclude=source_indices)
        rows.append(_query_row(entry, order, source_indices))

    return _with_mean_row(rows)


def build_val_benchmark(
    labels: torch.Tensor,
    query_specs: list[tuple[list[int], list[int]]],
    proxy_rows: list[int],
    directions: torch.Tensor,
    per_query: int = 200,
    min_gt: int = 3,
    seed: int = 0,
) -> list[dict]:
    """Held-out replica of the eval benchmark for checkpoint selection.

    Mirrors how the real celeba_evaluation.json ground truth is built - images
    that satisfy the query constraints and match the reference on the identity-
    proxy attributes - but over a held-out image pool (the val split), so it
    tracks the true R@10 without ever touching the test references or the test
    ground truth. Only the query *shapes* are shared with the benchmark, which
    is exactly the distribution we are graded on.

    labels: (N, A) bool for the val pool (references and database are this pool);
    query_specs: (positive rows, negative rows) per query, indexing `labels`;
    proxy_rows: identity-proxy attribute rows; directions: (A, D) probe dirs.
    Returns one task dict per query with precomputed reference indices, a
    (R, N) ground-truth mask, and the padded (dirs, signs, mask) model inputs.
    """
    gen = torch.Generator().manual_seed(seed)
    n = labels.shape[0]
    tasks = []
    for pos_rows, neg_rows in query_specs:
        satisfies = torch.ones(n, dtype=torch.bool)
        if pos_rows:
            satisfies &= labels[:, pos_rows].all(dim=1)
        if neg_rows:
            satisfies &= ~labels[:, neg_rows].any(dim=1)
        # A queried attribute must differ from the reference by construction, so
        # it cannot be part of the identity match (mirrors mining target choice).
        queried = set(pos_rows) | set(neg_rows)
        prox = [r for r in proxy_rows if r not in queried]

        refs, masks = [], []
        for r in torch.randperm(n, generator=gen).tolist():
            if prox:
                agree = (labels[:, prox] == labels[r, prox]).all(dim=1)
            else:
                agree = torch.ones(n, dtype=torch.bool)
            gt = satisfies & agree
            gt[r] = False
            if int(gt.sum()) >= min_gt:
                refs.append(r)
                masks.append(gt)
            if len(refs) >= per_query:
                break
        if not refs:
            continue
        refs_t = torch.tensor(refs)
        dirs, signs, mask = pad_queries([(pos_rows, neg_rows)] * len(refs), directions)
        tasks.append({
            "refs": refs_t,
            "gt_mask": torch.stack(masks),
            "dirs": dirs, "signs": signs, "mask": mask,
        })
    return tasks


@torch.no_grad()
def score_val_benchmark(model: CPAS, db: torch.Tensor, tasks: list[dict], k: int = 10) -> float:
    """Mean Recall@k of `model` over a val benchmark from `build_val_benchmark`.

    db: (N, D) L2-normalized features of the val pool (the ranking database);
    a query counts as a hit when any ground-truth image is in its top-k.
    """
    model.eval()
    device = db.device
    hits, total = 0, 0
    for task in tasks:
        refs = task["refs"].to(device)
        q = model(
            db[refs], task["dirs"].to(device),
            task["signs"].to(device), task["mask"].to(device),
        )
        sims = q @ db.T
        sims[torch.arange(refs.shape[0], device=device), refs] = float("-inf")
        top = sims.topk(k, dim=1).indices
        hit = task["gt_mask"].to(device).gather(1, top).any(dim=1)
        hits += int(hit.sum())
        total += refs.shape[0]
    return hits / max(total, 1)


@torch.no_grad()
def run_cpas_benchmark(
    annotations: list[dict],
    image_features: torch.Tensor,
    model: CPAS,
    directions: torch.Tensor,
    attr_index: dict[str, int],
) -> pd.DataFrame:
    """Evaluate the trained CPAS combiner on every query in `annotations`.

    Same protocol as run_probe_benchmark; the fixed composition is replaced by
    the model, which predicts a reference weight, per-attribute step sizes and
    direction bends for each (source image, query) pair.
    """
    model.eval()
    rows = []
    for entry in annotations:
        positives, negatives = parse_query(entry["query"])
        pos_rows = [attr_index[a] for a in positives]
        neg_rows = [attr_index[a] for a in negatives]

        source_indices = [int(k) for k in entry["ground_truth"].keys()]
        dirs, signs, mask = pad_queries(
            [(pos_rows, neg_rows)] * len(source_indices), directions
        )
        query_vecs = model(image_features[source_indices], dirs, signs, mask)
        order = rank(query_vecs, image_features, exclude=source_indices)
        rows.append(_query_row(entry, order, source_indices))

    return _with_mean_row(rows)


@torch.no_grad()
def probe_drift(
    query_vecs: torch.Tensor,
    v_ref: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
    pos_rows: list[int],
    neg_rows: list[int],
) -> dict[str, float]:
    """Mean probe-logit shift from reference to query, queried vs. the rest.

    `queried` is signed so that positive means "moved the right way" (+ for T+
    attributes, - for T-); `non_queried_abs` is leakage into the attributes the
    query never mentioned, which non-orthogonal directions cause and a
    conditioned composition should shrink (docs/method-proposal-cpas.md S4).
    """
    shift = (query_vecs - v_ref) @ weights.T  # biases cancel
    queried = pos_rows + neg_rows
    others = [a for a in range(weights.shape[0]) if a not in set(queried)]
    wanted = torch.cat(
        [shift[:, pos_rows], -shift[:, neg_rows]], dim=1
    ) if queried else shift[:, :0]
    return {
        "queried": float(wanted.mean()) if queried else 0.0,
        "non_queried_abs": float(shift[:, others].abs().mean()),
    }


def run_caption_benchmark(
    annotations: list[dict],
    image_features: torch.Tensor,
    states: torch.Tensor,
    encode_texts: Callable[[list[str]], torch.Tensor],
) -> pd.DataFrame:
    """Evaluate the predict->flip->caption pipeline on every query.

    states: (N, A) bool predicted attribute states, row i = dataset index i
        (from caption.predict_state on the whole retrieval split).
    Captions repeat across sources with identical flipped states, so each
    distinct caption is encoded once.
    """
    rows = []
    cache: dict[str, torch.Tensor] = {}
    for entry in annotations:
        positives, negatives = parse_query(entry["query"])
        source_indices = [int(k) for k in entry["ground_truth"].keys()]

        query_vecs = []
        for i in source_indices:
            flipped, forced_off = flip_state(states[i], positives, negatives)
            caption = render_caption(flipped, forced_off)
            if caption not in cache:
                cache[caption] = encode_caption(caption, encode_texts)
            query_vecs.append(cache[caption])

        order = rank(torch.stack(query_vecs), image_features, exclude=source_indices)
        rows.append(_query_row(entry, order, source_indices))

    return _with_mean_row(rows)
