"""Contrastive training of the CPAS-MLP combiner on mined attribute-flip queries.

The query built from (reference, flips) must rank its mined target above every
other candidate in the batch: the other rows' targets, plus its own mined
negatives (violators / drifters / the reference itself). That is the InfoNCE
objective of docs/method.md S5.

Two masks make the objective say what it means, and both default off so the
unmasked loss is bit-identical to the version that came before
(docs/method-proposal-mining-alignment.md S4):

`neg_mask` - a family can hold fewer than the requested number of negatives, and
padding by repetition would silently give a repeated negative double weight in
the softmax. Padded slots go to -inf instead, the same way pad_queries handles
short attribute lists.

`false_neg` - the in-batch negatives are the other rows' targets, and nothing
stops one of those from being a *correct answer* to this row's query. Left
alone, the loss pushes the query away from an image that would count as a hit at
evaluation. Roughly two rows in a thousand-wide batch, but they are the
semantically closest candidates, so at tau = 0.05 they carry gradient out of all
proportion to their number.
"""

from dataclasses import dataclass

import torch
from torch import nn

from src.criterion import MAX_HAMMING, valid_mask
from src.mining import MinedQuery
from src.steering import Steerer, pad_queries


@dataclass(frozen=True)
class Batch:
    """Model inputs and ranking candidates for one batch of mined queries."""

    v_ref: torch.Tensor      # (B, D)
    dirs: torch.Tensor       # (B, K, D)
    signs: torch.Tensor      # (B, K)
    mask: torch.Tensor       # (B, K) bool
    targets: torch.Tensor    # (B, D)
    negatives: torch.Tensor  # (B, 2M+1, D) violators | drifters | lazy
    neg_mask: torch.Tensor   # (B, 2M+1) bool, False on padded slots
    false_neg: torch.Tensor  # (B, B) bool, True where an in-batch target is valid here

    def to(self, device: str) -> "Batch":
        return Batch(*(t.to(device) for t in vars(self).values()))


def build_batch(
    queries: list[MinedQuery],
    features: torch.Tensor,
    directions: torch.Tensor,
    labels: torch.Tensor,
    max_hamming: int = MAX_HAMMING,
) -> Batch:
    """Gather features, padded attribute tensors and both masks.

    features / labels: (N, D) and (N, A) for the *mining pool*, aligned row for
    row - every index in a MinedQuery is a row of both.
    """
    dirs, signs, mask = pad_queries([(q.add, q.remove) for q in queries], directions)
    refs = [q.ref for q in queries]
    targets = [q.target for q in queries]

    b, d = len(queries), features.shape[1]
    n_viol = max((len(q.violators) for q in queries), default=0)
    n_drift = max((len(q.drifters) for q in queries), default=0)
    negatives = torch.zeros(b, n_viol + n_drift + 1, d,
                            dtype=features.dtype, device=features.device)
    neg_mask = torch.zeros(negatives.shape[:2], dtype=torch.bool,
                           device=features.device)
    for i, q in enumerate(queries):
        if q.violators:
            negatives[i, : len(q.violators)] = features[q.violators]
            neg_mask[i, : len(q.violators)] = True
        if q.drifters:
            end = n_viol + len(q.drifters)
            negatives[i, n_viol:end] = features[q.drifters]
            neg_mask[i, n_viol:end] = True
    # The reference owns the last slot unconditionally. It is formally a
    # violator - Hamming 0, every queried constraint broken - but it is also the
    # single highest-scoring image in the database whenever q collapses onto
    # v_ref, so its presence must not depend on a draw.
    negatives[:, -1] = features[refs]
    neg_mask[:, -1] = True

    # Validity is per row: row i has its own reference code and its own queried
    # columns, so there is no shared column set to vectorise over. B is ~1e3 and
    # each step is (B, A), so the loop is cheap and correct.
    target_labels, ref_labels = labels[targets], labels[refs]
    false_neg = torch.zeros(b, b, dtype=torch.bool, device=labels.device)
    for i, q in enumerate(queries):
        false_neg[i] = valid_mask(target_labels, ref_labels[i], q.add, q.remove,
                                  max_hamming)
    false_neg.fill_diagonal_(False)   # a row's own target is its positive

    return Batch(
        v_ref=features[refs],
        dirs=dirs,
        signs=signs,
        mask=mask,
        targets=features[targets],
        negatives=negatives,
        neg_mask=neg_mask,
        false_neg=false_neg,
    )


def _logits(
    q: torch.Tensor,
    targets: torch.Tensor,
    negatives: torch.Tensor,
    neg_mask: torch.Tensor | None,
    false_neg: torch.Tensor | None,
) -> torch.Tensor:
    """(B, B + M) candidate scores, masked slots sent to -inf."""
    in_batch = q @ targets.T
    if false_neg is not None:
        in_batch = in_batch.masked_fill(false_neg, float("-inf"))
    mined = torch.einsum("bd,bmd->bm", q, negatives)
    if neg_mask is not None:
        mined = mined.masked_fill(~neg_mask, float("-inf"))
    return torch.cat([in_batch, mined], dim=1)


def infonce_loss(
    q: torch.Tensor,
    targets: torch.Tensor,
    negatives: torch.Tensor,
    neg_mask: torch.Tensor | None = None,
    false_neg: torch.Tensor | None = None,
    tau: float = 0.05,
) -> torch.Tensor:
    """InfoNCE over in-batch targets plus each query's own mined negatives.

    q, targets: (B, D) L2-normalized; negatives: (B, M, D). Row i's positive is
    targets[i]. With both masks left at None this is the plain unmasked loss.
    """
    logits = _logits(q, targets, negatives, neg_mask, false_neg) / tau
    labels = torch.arange(q.shape[0], device=q.device)
    return nn.functional.cross_entropy(logits, labels)


def recall_at_1(q: torch.Tensor, batch: Batch) -> float:
    """Fraction of queries ranking their target above every batch candidate."""
    scores = _logits(q, batch.targets, batch.negatives, batch.neg_mask,
                     batch.false_neg)
    hit = scores.argmax(dim=1) == torch.arange(q.shape[0], device=q.device)
    return float(hit.float().mean())


def run_epoch(
    model: Steerer,
    queries: list[MinedQuery],
    features: torch.Tensor,
    directions: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    batch_size: int = 256,
    tau: float = 0.05,
    device: str = "cpu",
) -> tuple[float, float]:
    """One pass over `queries`; trains when an optimizer is given.

    Returns (mean loss, mean batch recall@1).
    """
    train = optimizer is not None
    model.train(train)
    losses, recalls = [], []
    for start in range(0, len(queries) - 1, batch_size):
        chunk = queries[start : start + batch_size]
        if len(chunk) < 2:  # InfoNCE needs at least one in-batch negative
            continue
        batch = build_batch(chunk, features, directions, labels).to(device)
        with torch.set_grad_enabled(train):
            q = model(batch.v_ref, batch.dirs, batch.signs, batch.mask)
            loss = infonce_loss(q, batch.targets, batch.negatives,
                                batch.neg_mask, batch.false_neg, tau=tau)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
        recalls.append(recall_at_1(q.detach(), batch))
    n = max(len(losses), 1)
    return sum(losses) / n, sum(recalls) / n
