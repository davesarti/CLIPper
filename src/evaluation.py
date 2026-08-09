"""Retrieval metrics and the benchmark loop over the evaluation queries."""

from collections.abc import Callable

import pandas as pd
import torch

from src.criterion import MAX_HAMMING, valid_mask
from src.steering import Steerer, pad_queries
from src.probes import compose_probe
from src.rerank import Rerank, rank_with_exclusion
from src.retrieval import PROMPTS, compose, parse_query, rank

KS = (1, 5, 10)
VIOLATION_K = 10

# MAX_HAMMING and the rule it belongs to now live in src/criterion.py, shared
# with the miner so the two cannot drift apart again. It is re-exported here
# because the benchmark scripts have always imported it from this module.


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


def violation_rate(
    order: torch.Tensor,
    labels: torch.Tensor,
    pos_rows: list[int],
    neg_rows: list[int],
    k: int = VIOLATION_K,
) -> float:
    """Fraction of returned top-k images that break at least one constraint.

    This is what the exclusion penalty directly targets, and R@k cannot stand in
    for it: R@k can improve for reasons unrelated to exclusion, so a rise in
    R@k without a fall here leaves the mechanism claim unsupported
    (docs/method.md S8).

    order: (S, N) retrieved indices, best first; labels: (N, A) bool.
    """
    top = order[:, :k]
    got = labels.to(order.device)[top]  # (S, k, A)
    bad = torch.zeros(top.shape, dtype=torch.bool, device=order.device)
    if pos_rows:
        bad |= ~got[:, :, pos_rows].all(dim=-1)
    if neg_rows:
        bad |= got[:, :, neg_rows].any(dim=-1)
    return float(bad.float().mean())


def negation_subset(df: pd.DataFrame, column: str = "R@10") -> float:
    """Mean of `column` over the benchmark queries carrying a negation.

    The overall mean is diluted by the positive-only queries that the
    negation-aware changes are not meant to help - 6 of the 14 - so the subset
    is reported separately (docs/method.md S8).
    """
    rows = df[(df["query"] != "MEAN") & df["query"].str.contains("-")]
    return float(rows[column].mean()) if len(rows) else float("nan")


def _query_row(
    entry: dict,
    order: torch.Tensor,
    source_indices: list[int],
    labels: torch.Tensor | None = None,
    pos_rows: list[int] | None = None,
    neg_rows: list[int] | None = None,
) -> dict:
    """Per-query metrics row: R@K / P@K averaged over the source images.

    With `labels` given, the row also carries the top-10 violation rate; the
    column is absent otherwise, so callers that never pass labels produce the
    same table they always did.
    """
    sums = {f"R@{k}": 0.0 for k in KS} | {f"P@{k}": 0.0 for k in KS}
    for row_idx, src in enumerate(source_indices):
        retrieved = order[row_idx, : max(KS)].tolist()
        targets = entry["ground_truth"][str(src)]
        for k in KS:
            m = evaluate_retrieval(retrieved, targets, k)
            sums[f"R@{k}"] += m[f"Recall@{k}"]
            sums[f"P@{k}"] += m[f"Precision@{k}"]
    n = len(source_indices)
    row = {"query": entry["query"], "sources": n} | {
        key: val / n for key, val in sums.items()
    }
    if labels is not None:
        row[f"V@{VIOLATION_K}"] = violation_rate(
            order, labels, pos_rows or [], neg_rows or [], k=VIOLATION_K
        )
    return row


def _with_mean_row(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [f"R@{k}" for k in KS] + [f"P@{k}" for k in KS]
    if f"V@{VIOLATION_K}" in df.columns:
        metric_cols.append(f"V@{VIOLATION_K}")
    mean_row = {"query": "MEAN", "sources": df["sources"].sum()}
    mean_row |= df[metric_cols].mean().to_dict()
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df[["query", "sources", *metric_cols]]


def run_benchmark(
    annotations: list[dict],
    image_features: torch.Tensor,
    encode_texts: Callable[[list[str]], torch.Tensor],
    gamma: float = 1.0,
    labels: torch.Tensor | None = None,
    attr_index: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Evaluate the compose+rank baseline on every query in `annotations`.

    image_features: (N, D) L2-normalized features for the whole test split,
        row i = dataset index i.
    encode_texts: maps a list of prompt strings to (M, D) L2-normalized
        embeddings (injected so tests can fake it and the model loads once).
    gamma: reference-image weight passed through to compose.
    labels / attr_index: optional (N, A) bool test-split labels and the
        attribute-name -> column map, which add the top-10 violation rate to
        the table. Both or neither; with neither the table is what it always
        was. The baseline is scored exactly like every other method
        (docs/method.md S8 rule 1), so its row is comparable column for column.
    """
    if (labels is None) != (attr_index is None):
        raise ValueError("labels and attr_index must be given together")
    rows = []
    for entry in annotations:
        positives, negatives = parse_query(entry["query"])
        pos_texts = encode_texts([PROMPTS[a] for a in positives]) if positives \
            else torch.zeros((0, image_features.shape[1]))
        neg_texts = encode_texts([PROMPTS[a] for a in negatives]) if negatives \
            else torch.zeros((0, image_features.shape[1]))
        pos_rows = [attr_index[a] for a in positives] if attr_index else None
        neg_rows = [attr_index[a] for a in negatives] if attr_index else None

        source_indices = [int(k) for k in entry["ground_truth"].keys()]
        query_vecs = torch.stack([
            compose(image_features[i], pos_texts, neg_texts, gamma=gamma)
            for i in source_indices
        ])
        order = rank(query_vecs, image_features, exclude=source_indices)
        rows.append(_query_row(entry, order, source_indices, labels,
                               pos_rows, neg_rows))

    return _with_mean_row(rows)


def run_probe_benchmark(
    annotations: list[dict],
    image_features: torch.Tensor,
    directions: torch.Tensor,
    attr_index: dict[str, int],
    gamma: float = 1.0,
    rerank: Rerank | None = None,
    labels: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Evaluate compose_probe+rank on every query in `annotations`.

    directions: (A, D) L2-normalized probe weight directions;
    attr_index: attribute name -> row of `directions`.
    Only the attributes named in each query are used.

    rerank: optional exclusion penalty (src/rerank.py); with None the ranking
        is plain cosine, bit-identical to before the re-rank existed.
    labels: optional (N, A) bool test-split labels, which add the top-10
        violation rate to the table.
    """
    rows = []
    for entry in annotations:
        positives, negatives = parse_query(entry["query"])
        pos_rows = [attr_index[a] for a in positives]
        neg_rows = [attr_index[a] for a in negatives]

        source_indices = [int(k) for k in entry["ground_truth"].keys()]
        query_vecs = torch.stack([
            compose_probe(image_features[i], directions[pos_rows],
                          directions[neg_rows], gamma=gamma)
            for i in source_indices
        ])
        order = rank_with_exclusion(
            query_vecs, image_features, pos_rows, neg_rows,
            rerank=rerank, exclude=source_indices,
        )
        rows.append(_query_row(entry, order, source_indices, labels, pos_rows, neg_rows))

    return _with_mean_row(rows)


def build_val_benchmark(
    labels: torch.Tensor,
    query_specs: list[tuple[list[int], list[int]]],
    directions: torch.Tensor,
    per_query: int = 200,
    min_gt: int = 3,
    seed: int = 0,
    max_hamming: int = MAX_HAMMING,
) -> list[dict]:
    """Held-out replica of the eval benchmark for checkpoint selection.

    Reproduces the assignment's ground-truth rule exactly (S3.1.1: constraints
    satisfied, plus Hamming distance <= 2 to the reference over the non-queried
    attributes) but over a held-out image pool, so it tracks the true R@10
    without ever touching the test references or the test ground truth. Only the
    query *shapes* are shared with the benchmark, which is the distribution we
    are graded on.

    An earlier version required exact agreement on ten identity-proxy
    attributes. That is a different task: it selected checkpoints and tuned
    hyperparameters against ground truth the benchmark does not use, which is
    why validation gains did not transfer to test.

    The rule itself is `src/criterion.valid_mask`, shared with the miner: the
    two used to carry separate implementations, and only one of them was right.

    labels: (N, A) bool for the val pool (references and database are this pool);
    query_specs: (positive rows, negative rows) per query, indexing `labels`;
    directions: (A, D) probe dirs.
    Returns one task dict per query with precomputed reference indices, a
    (R, N) ground-truth mask, and the padded (dirs, signs, mask) model inputs.
    """
    gen = torch.Generator().manual_seed(seed)
    n = labels.shape[0]
    tasks = []
    for pos_rows, neg_rows in query_specs:
        refs, masks = [], []
        for r in torch.randperm(n, generator=gen).tolist():
            gt = valid_mask(labels, labels[r], pos_rows, neg_rows, max_hamming)
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
            "pos_rows": list(pos_rows), "neg_rows": list(neg_rows),
        })
    return tasks


@torch.no_grad()
def score_val_benchmark(
    model: Steerer,
    db: torch.Tensor,
    tasks: list[dict],
    k: int = 10,
    rerank: Rerank | None = None,
) -> float:
    """Mean Recall@k of `model` over a val benchmark from `build_val_benchmark`.

    db: (N, D) L2-normalized features of the val pool (the ranking database);
    a query counts as a hit when any ground-truth image is in its top-k.

    `rerank` applies the exclusion penalty to the val ranking; its `db_probs`
    must be the val pool's, not the test split's. This is the surface the
    lambda sweep tunes on - the 14 test queries are never touched.
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
        if rerank is None:
            sims = q @ db.T
        else:
            sims = rerank.scores(q, db, task["pos_rows"], task["neg_rows"])
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
    model: Steerer,
    directions: torch.Tensor,
    attr_index: dict[str, int],
    rerank: Rerank | None = None,
    labels: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Evaluate the trained CPAS combiner on every query in `annotations`.

    Same protocol as run_probe_benchmark; the fixed composition is replaced by
    the model, which predicts a reference weight, per-attribute step sizes and
    direction bends for each (source image, query) pair. `rerank` and `labels`
    mean what they do there - the exclusion penalty is orthogonal to the
    combiner, which is why both call sites take it.
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
        order = rank_with_exclusion(
            query_vecs, image_features, pos_rows, neg_rows,
            rerank=rerank, exclude=source_indices,
        )
        rows.append(_query_row(entry, order, source_indices, labels, pos_rows, neg_rows))

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
    conditioned composition should shrink (docs/method.md S8).
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
