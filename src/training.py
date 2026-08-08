"""Contrastive training of the CPAS combiner on mined attribute-flip triplets.

The query built from (reference, flips) must rank its mined target above every
other image in the batch and above the three hard negatives mined with it
(violation / identity distractor / lazy), which is the InfoNCE objective of
docs/method.md S5.
"""

from dataclasses import dataclass

import torch
from torch import nn

from src.steering import Steerer, pad_queries
from src.mining import Triplet


@dataclass(frozen=True)
class Batch:
    """Model inputs and ranking candidates for one batch of triplets."""

    v_ref: torch.Tensor       # (B, D)
    dirs: torch.Tensor        # (B, K, D)
    signs: torch.Tensor       # (B, K)
    mask: torch.Tensor        # (B, K) bool
    targets: torch.Tensor     # (B, D)
    violations: torch.Tensor  # (B, M, D) near misses breaking a constraint
    negatives: torch.Tensor   # (B, 2, D) distractor, lazy

    def to(self, device: str) -> "Batch":
        return Batch(*(t.to(device) for t in vars(self).values()))

    @property
    def candidates(self) -> torch.Tensor:
        """(B, M + 2, D) every mined negative, violations first.

        Concatenating in this order keeps the single-violation case identical
        to the original [violation, distractor, lazy] layout.
        """
        return torch.cat([self.violations, self.negatives], dim=1)


def build_batch(
    triplets: list[Triplet],
    features: torch.Tensor,
    directions: torch.Tensor,
) -> Batch:
    """Gather features and padded attribute tensors for a list of triplets.

    Triplets may carry different numbers of violations (a query with few near
    misses yields fewer), so the short ones are padded by cycling their own
    violations up to the batch maximum: a repeated near miss is weighted twice
    in the loss, which is milder than padding with an unrelated image.
    """
    dirs, signs, mask = pad_queries(
        [(t.positives, t.negatives) for t in triplets], directions
    )
    refs = [t.ref for t in triplets]
    m = max(len(t.violations) for t in triplets)
    violation_rows = [
        [t.violations[i % len(t.violations)] for i in range(m)] for t in triplets
    ]
    negatives = torch.stack(
        [
            features[[t.distractor for t in triplets]],
            features[refs],
        ],
        dim=1,
    )
    return Batch(
        v_ref=features[refs],
        dirs=dirs,
        signs=signs,
        mask=mask,
        targets=features[[t.target for t in triplets]],
        violations=features[torch.tensor(violation_rows, device=features.device)],
        negatives=negatives,
    )


def infonce_loss(
    q: torch.Tensor,
    targets: torch.Tensor,
    negatives: torch.Tensor,
    tau: float = 0.05,
) -> torch.Tensor:
    """InfoNCE over in-batch targets plus each query's own hard negatives.

    q, targets: (B, D) L2-normalized; negatives: (B, M, D). Row i's positive is
    targets[i]; its negatives are every other in-batch target and its M mined
    ones.
    """
    in_batch = q @ targets.T                                  # (B, B)
    mined = torch.einsum("bd,bmd->bm", q, negatives)          # (B, M)
    logits = torch.cat([in_batch, mined], dim=1) / tau
    labels = torch.arange(q.shape[0], device=q.device)
    return nn.functional.cross_entropy(logits, labels)


def violation_loss(
    q: torch.Tensor,
    targets: torch.Tensor,
    violations: torch.Tensor,
    tau: float = 0.05,
) -> torch.Tensor:
    """The target must outrank its *own* violations, and nothing else.

    Inside the big InfoNCE softmax a single violation competes with 1023
    in-batch targets and two other mined negatives, so at tau = 0.05 it carries
    gradient only when it already ranks near the top: the term meant to teach
    "negation is a constraint" is roughly one thousandth of the loss mass. Here
    the sum runs over M items, so each violation carries real gradient
    (docs/method.md S6.2).

    q, targets: (B, D) L2-normalized; violations: (B, M, D).
    """
    positive = (q * targets).sum(dim=1, keepdim=True)              # (B, 1)
    mined = torch.einsum("bd,bmd->bm", q, violations)              # (B, M)
    logits = torch.cat([positive, mined], dim=1) / tau
    labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    return nn.functional.cross_entropy(logits, labels)


def recall_at_1(q: torch.Tensor, batch: Batch) -> float:
    """Fraction of queries ranking their target above every batch candidate."""
    in_batch = q @ batch.targets.T
    mined = torch.einsum("bd,bmd->bm", q, batch.candidates)
    scores = torch.cat([in_batch, mined], dim=1)
    hit = scores.argmax(dim=1) == torch.arange(q.shape[0], device=q.device)
    return float(hit.float().mean())


def run_epoch(
    model: Steerer,
    triplets: list[Triplet],
    features: torch.Tensor,
    directions: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    batch_size: int = 256,
    tau: float = 0.05,
    device: str = "cpu",
    lambda_violation: float = 0.0,
) -> tuple[float, float]:
    """One pass over `triplets`; trains when an optimizer is given.

    lambda_violation > 0 pulls the violations out of the main softmax and gives
    them their own weighted term; at 0 (the default) they stay inside it exactly
    as before, so the original recipe is reproduced bit-for-bit.

    Returns (mean loss, mean batch recall@1).
    """
    train = optimizer is not None
    model.train(train)
    losses, recalls = [], []
    for start in range(0, len(triplets) - 1, batch_size):
        chunk = triplets[start : start + batch_size]
        if len(chunk) < 2:  # InfoNCE needs at least one in-batch negative
            continue
        batch = build_batch(chunk, features, directions).to(device)
        with torch.set_grad_enabled(train):
            q = model(batch.v_ref, batch.dirs, batch.signs, batch.mask)
            if lambda_violation:
                loss = infonce_loss(q, batch.targets, batch.negatives, tau=tau)
                loss = loss + lambda_violation * violation_loss(
                    q, batch.targets, batch.violations, tau=tau
                )
            else:
                loss = infonce_loss(q, batch.targets, batch.candidates, tau=tau)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
        recalls.append(recall_at_1(q.detach(), batch))
    n = max(len(losses), 1)
    return sum(losses) / n, sum(recalls) / n
