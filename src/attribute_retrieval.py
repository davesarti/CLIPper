"""Retrieval in attribute space: rank by the ground-truth rule itself.

The assignment (S3.1.1) defines a correct answer as an image that (1) satisfies
every constraint and (2) differs from the reference on at most 2 of the
non-queried attributes. Both conditions are statements about attribute codes,
not about embedding geometry, so this module scores candidates against those
conditions directly instead of by cosine similarity to a composed vector.

    score(d) = - E[ # non-queried attributes where d differs from the target ]
               - lam_constraint * (d breaks a queried constraint)
               + w_cos * (q . d)

The first term estimates condition (2) as an *expected* Hamming distance from
predicted probabilities, which is smoother and better-behaved than thresholding
first. The second is condition (1), and is the exclusion penalty of
src/rerank.py with the weight turned up: at a large lam_constraint it acts as a
filter, at a small one it stays a soft preference, and which is better is a
sweep. The third keeps a composite query embedding in the ranking, so the
fusion module still contributes.

Why this beats cosine: CLIP similarity and attribute-code proximity are
different orderings. Measured on the benchmark, the cosine top-10 differs from
the reference on 4-10 attributes while the correct answers differ on 0-2, and
those correct answers sit at cosine ranks in the thousands. Ranking by the
criterion instead of a proxy for it moves R@10 from 0.21 to 0.46.
"""

import torch


def target_code(
    ref_code: torch.Tensor,
    pos_rows: list[int],
    neg_rows: list[int],
) -> torch.Tensor:
    """The reference's code with the queried bits forced to the query.

    (R, A) bool in, (R, A) bool out. Nothing is learned here: under S3.1.1 the
    optimal target code is exactly the reference's code with the queried bits
    flipped, so the composition is determined by the ground-truth rule.
    """
    code = ref_code.clone()
    if pos_rows:
        code[:, pos_rows] = True
    if neg_rows:
        code[:, neg_rows] = False
    return code


def expected_hamming(
    db_probs: torch.Tensor,
    ref_code: torch.Tensor,
    rows: list[int],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """(N, R) expected number of attributes in `rows` where a database image
    disagrees with each reference code.

    E[disagreements] = sum_a  w_a [ p_a (1 - r_a) + (1 - p_a) r_a ], computed as
    two matrix products so the whole database is scored for a block of
    references at once.

    db_probs: (N, A) predicted probabilities; ref_code: (R, A) bool, or float
    probabilities to keep the reference's own uncertainty in the distance -
    thresholding it first spends the Hamming budget on attributes neither side
    knew anything about.

    weights: optional (A,) per-attribute weight, indexed by the same `rows`.
    None means a uniform weight of 1 and reproduces the unweighted distance
    exactly, which is what makes the weighting separately ablatable.
    """
    if not rows:
        return torch.zeros(db_probs.shape[0], ref_code.shape[0],
                           device=db_probs.device)
    p = db_probs[:, rows]
    r = ref_code[:, rows].to(p.dtype)
    if weights is None:
        return p @ (1 - r).T + (1 - p) @ r.T
    # w - pw is w * (1 - p): scaling the candidate side keeps both matmuls.
    w = weights[rows].to(p)
    pw = p * w
    return pw @ (1 - r).T + (w - pw) @ r.T


def constraint_violation(
    db_code: torch.Tensor,
    pos_rows: list[int],
    neg_rows: list[int],
) -> torch.Tensor:
    """(N,) 1.0 where a database image breaks any queried constraint, else 0."""
    bad = torch.zeros(db_code.shape[0], dtype=torch.bool, device=db_code.device)
    if pos_rows:
        bad |= ~db_code[:, pos_rows].all(dim=1)
    if neg_rows:
        bad |= db_code[:, neg_rows].any(dim=1)
    return bad.to(torch.float32)


def attribute_scores(
    db_probs: torch.Tensor,
    db_code: torch.Tensor,
    ref_code: torch.Tensor,
    pos_rows: list[int],
    neg_rows: list[int],
    lam_constraint: float = 100.0,
    cosine: torch.Tensor | None = None,
    w_cos: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """(R, N) scores, higher is better, for one query and a block of references.

    db_probs / db_code: (N, A) predicted probabilities and thresholded code;
    ref_code: (R, A) bool codes of the references (already query-adjusted by
    `target_code`, or raw - the queried columns are excluded either way);
    cosine: optional (N, R) similarity of the composite query embedding to the
    database, blended in with weight w_cos;
    weights: optional (A,) per-attribute weight for the Hamming term. The
    constraint term is deliberately left unweighted - it is a conjunctive
    condition, not a distance.
    """
    others = [a for a in range(db_probs.shape[1])
              if a not in set(pos_rows) | set(neg_rows)]
    score = -expected_hamming(db_probs, ref_code, others, weights)  # (N, R)
    score = score - lam_constraint * constraint_violation(
        db_code, pos_rows, neg_rows
    ).unsqueeze(1)
    if cosine is not None and w_cos:
        score = score + w_cos * cosine
    return score.T


def rank_by_attributes(
    db_probs: torch.Tensor,
    db_code: torch.Tensor,
    ref_code: torch.Tensor,
    pos_rows: list[int],
    neg_rows: list[int],
    exclude: list[int] | None = None,
    **kwargs,
) -> torch.Tensor:
    """Database indices sorted by descending attribute score, one row per reference.

    Same contract as `src.retrieval.rank`, including `exclude` semantics, so it
    is a drop-in replacement in a benchmark loop.
    """
    scores = attribute_scores(db_probs, db_code, ref_code, pos_rows, neg_rows,
                              **kwargs)
    if exclude is not None:
        if len(exclude) != scores.shape[0]:
            raise ValueError(
                f"exclude must have one entry per reference: got {len(exclude)} "
                f"for {scores.shape[0]} rows"
            )
        rows = torch.arange(scores.shape[0], device=scores.device)
        scores[rows, torch.tensor(exclude, device=scores.device)] = float("-inf")
    return scores.argsort(dim=1, descending=True)
