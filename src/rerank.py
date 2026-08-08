"""Exclusion re-rank: a non-compensatory penalty on constraint violations.

A single query vector scores every candidate with one linear functional, and a
linear functional is compensatory: `q.d` is a weighted sum of attribute
evidence, so a surplus on one attribute pays for a violation on another. The
ground truth is conjunctive. This module adds a hinge penalty on top of the
cosine term (docs/method.md S7):

    s(d) = q.d
           - lam_neg * sum_{a in T-} relu( p_a(d) - tau_a )
           - lam_pos * sum_{a in T+} relu( tau_a - p_a(d) )

`p_a(d)` is the probe's predicted probability, so one lambda is meaningful
across all 40 attributes (raw logits have per-attribute scale). Below the
threshold the penalty is exactly zero, so a compliant candidate is never
charged; above it the cost cannot be bought back by a better cosine elsewhere.

The whole thing is default-off: `Rerank(lam_neg=0, lam_pos=0)` reproduces plain
cosine ranking bit-for-bit, which is what lets the ablation attribute any change
to the re-rank alone.
"""

from dataclasses import dataclass

import torch


def database_probe_probs(
    features: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    """(N, A) probability that each database image has each attribute.

    features: (N, D) L2-normalized image features.
    weights:  (A, D) RAW probe weights - not the normalized directions that the
        composition uses. `src.probes.load_probes` normalizes its weights while
        the saved biases belong to the unnormalized ones, so pairing the two
        gives plausible-looking but meaningless probabilities; use
        `src.probes.load_raw_probes` here.
    biases:   (A,)
    """
    if weights.shape[0] != biases.shape[0]:
        raise ValueError("weights and biases must describe the same attributes")
    return torch.sigmoid(features @ weights.T + biases)


def exclusion_penalty(
    db_probs: torch.Tensor,
    pos_rows: list[int],
    neg_rows: list[int],
    lam_neg: float = 0.0,
    lam_pos: float = 0.0,
    thresholds: torch.Tensor | None = None,
    hinge: bool = True,
) -> torch.Tensor:
    """(N,) non-negative penalty per database image for one query's constraints.

    thresholds: (A,) per-attribute tau, or None for a shared 0.5.
    hinge=False replaces relu(p - tau) with p (and relu(tau - p) with 1 - p),
    which is ablation row 4: a linear penalty is still compensatory - it is just
    another term in a weighted sum, expressible by moving `q` itself.
    """
    n, a = db_probs.shape
    tau = (
        torch.full((a,), 0.5, dtype=db_probs.dtype, device=db_probs.device)
        if thresholds is None
        else thresholds.to(db_probs)
    )
    penalty = torch.zeros(n, dtype=db_probs.dtype, device=db_probs.device)
    if lam_neg and neg_rows:
        excess = db_probs[:, neg_rows] - (tau[neg_rows] if hinge else 0.0)
        penalty += lam_neg * (excess.relu() if hinge else excess).sum(dim=1)
    if lam_pos and pos_rows:
        deficit = (tau[pos_rows] if hinge else 1.0) - db_probs[:, pos_rows]
        penalty += lam_pos * (deficit.relu() if hinge else deficit).sum(dim=1)
    return penalty


@dataclass(frozen=True)
class Rerank:
    """Scoring configuration shared by every call site.

    db_probs: (N, A) from `database_probe_probs`, computed once for the whole
        database (19,962 x 40 floats, ~3 MB) - scoring a query is then one
        gather and one hinge.
    top_m: apply the penalty to the top-m candidates by cosine only (ablation
        row 5, the cheap two-stage version); None penalizes the whole database.
    """

    db_probs: torch.Tensor
    lam_neg: float = 0.0
    lam_pos: float = 0.0
    thresholds: torch.Tensor | None = None
    hinge: bool = True
    top_m: int | None = None

    @property
    def active(self) -> bool:
        return bool(self.lam_neg) or bool(self.lam_pos)

    def scores(
        self,
        query_vecs: torch.Tensor,
        image_features: torch.Tensor,
        pos_rows: list[int],
        neg_rows: list[int],
    ) -> torch.Tensor:
        """(S, N) re-ranked scores, higher is better."""
        sims = query_vecs @ image_features.T
        if not self.active:
            return sims
        penalty = exclusion_penalty(
            self.db_probs.to(sims.device), pos_rows, neg_rows,
            self.lam_neg, self.lam_pos, self.thresholds, self.hinge,
        )
        if self.top_m is None:
            return sims - penalty
        # Two-stage: only the shortlist is charged, so candidates below it can
        # no longer be displaced by a penalized shortlist member.
        m = min(self.top_m, sims.shape[1])
        shortlist = sims.topk(m, dim=1).indices
        charged = sims.clone()
        charged.scatter_add_(1, shortlist, -penalty[shortlist])
        return charged


def rank_with_exclusion(
    query_vecs: torch.Tensor,
    image_features: torch.Tensor,
    pos_rows: list[int],
    neg_rows: list[int],
    rerank: Rerank | None = None,
    exclude: list[int] | None = None,
) -> torch.Tensor:
    """Image indices sorted by descending penalized score.

    Same contract as `src.retrieval.rank` (including `exclude` semantics), so a
    benchmark loop switches with one line. With `rerank=None` or an inactive
    configuration this is exactly `rank`.
    """
    if exclude is not None and len(exclude) != query_vecs.shape[0]:
        raise ValueError(
            f"exclude must have one entry per query row: "
            f"got {len(exclude)} for {query_vecs.shape[0]} rows"
        )
    if rerank is None:
        scores = query_vecs @ image_features.T
    else:
        scores = rerank.scores(query_vecs, image_features, pos_rows, neg_rows)
    if exclude is not None:
        rows = torch.arange(scores.shape[0], device=scores.device)
        scores[rows, torch.tensor(exclude, device=scores.device)] = float("-inf")
    return scores.argsort(dim=1, descending=True)
