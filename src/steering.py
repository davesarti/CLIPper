"""Pieces shared by the steered-composition combiner and its call sites.

The combiner (PerAttributeMLP, src/cpas_mlp.py) predicts a reference weight,
per-attribute step sizes and direction bends, then builds the query from them.
That formula lives here once, apart from the model: the ablations in
docs/method.md only mean something if the variants differ solely
in how (gamma, alpha, delta) are produced, which is worth nothing if the
composition can silently drift between them.
"""

from typing import Protocol

import torch
from torch import nn


def compose(
    v_ref: torch.Tensor,
    dirs: torch.Tensor,
    signs: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Build query embeddings from predicted steering quantities.

    q = normalize( gamma * v_ref + sum_a sign_a * alpha_a * normalize(w_a + delta_a) )

    v_ref: (B, D); dirs/delta: (B, K, D); signs/alpha: (B, K); gamma: (B,).
    alpha and delta are expected to be zeroed on padded slots, which is what
    keeps padding out of the sum. Returns (B, D), L2-normalized.
    """
    bent = nn.functional.normalize(dirs + delta, dim=-1)
    steps = (signs * alpha).unsqueeze(-1) * bent  # (B, K, D)
    q = gamma.unsqueeze(-1) * v_ref + steps.sum(dim=1)
    return nn.functional.normalize(q, dim=-1)


class Steerer(Protocol):
    """What training and evaluation actually require of a combiner.

    Typing the call sites against this rather than against the concrete model
    is what let the transformer variant be swapped out for cpas_mlp with
    nothing downstream to change.
    """

    def steer(
        self,
        v_ref: torch.Tensor,
        dirs: torch.Tensor,
        signs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def __call__(
        self,
        v_ref: torch.Tensor,
        dirs: torch.Tensor,
        signs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        ...


class FixedRule:
    """The gamma-weighted probe composition, wearing the Steerer interface.

    q = normalize(gamma * v_ref + sum(pos_dirs) - sum(neg_dirs)), i.e.
    `probes.compose_probe` with alpha = 1 and delta = 0. Having it satisfy the
    same protocol as the trained combiner is what lets the val benchmark and
    the exclusion-rerank sweep run over both without a second code path.
    """

    def __init__(self, gamma: float = 0.6) -> None:
        self.gamma = gamma

    def eval(self) -> "FixedRule":
        return self

    def train(self, mode: bool = True) -> "FixedRule":
        return self

    def steer(
        self,
        v_ref: torch.Tensor,
        dirs: torch.Tensor,
        signs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = torch.full((v_ref.shape[0],), self.gamma, device=v_ref.device)
        alpha = mask.to(dirs.dtype)
        return gamma, alpha, torch.zeros_like(dirs)

    def __call__(
        self,
        v_ref: torch.Tensor,
        dirs: torch.Tensor,
        signs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        gamma, alpha, delta = self.steer(v_ref, dirs, signs, mask)
        return compose(v_ref, dirs, signs, gamma, alpha, delta)


def pad_queries(
    queries: list[tuple[list[int], list[int]]],
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack per-query attribute rows into padded (dirs, signs, mask) tensors.

    queries: list of (positive rows, negative rows) indexing `directions`;
    directions: (A, D) L2-normalized probe directions.
    """
    d = directions.shape[1]
    k = max((len(p) + len(n) for p, n in queries), default=0)
    dirs = torch.zeros(len(queries), k, d)
    signs = torch.zeros(len(queries), k)
    mask = torch.zeros(len(queries), k, dtype=torch.bool)
    for i, (pos, neg) in enumerate(queries):
        rows = pos + neg
        if not rows:
            continue
        dirs[i, : len(rows)] = directions[rows]
        signs[i, : len(pos)] = 1.0
        signs[i, len(pos) : len(rows)] = -1.0
        mask[i, : len(rows)] = True
    return dirs, signs, mask
