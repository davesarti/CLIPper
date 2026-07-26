"""Attribute-flip triplet mining for CPAS training.

Training examples are synthesized from CelebA train-split labels, never from
retrieval annotations: sample a reference, flip k of its attributes (0->1 gives
T+, 1->0 gives T-), then look up a real image that satisfies the flipped
constraints and still looks like the same kind of person.

Each example also carries the three hard negatives of the proposal
(docs/method-proposal-cpas.md, S3), each aimed at one shortcut:

    violation  - satisfies T+ but breaks a T- (or misses a T+ when the query
                 has no negatives): negation is a constraint, not a direction
    distractor - satisfies the constraints but is a different kind of person:
                 do not ignore the reference
    lazy       - the reference itself: do not return it unchanged
"""

from dataclasses import dataclass

import torch

# Attributes used as an identity stand-in: CelebA has identity labels, but the
# flipped-attribute target is a different person by construction, so "same
# person" is approximated by agreement on stable, non-editable traits.
IDENTITY_PROXY = (
    "Male",
    "Young",
    "Chubby",
    "Double_Chin",
    "Oval_Face",
    "Narrow_Eyes",
    "Big_Nose",
    "Big_Lips",
    "Pointy_Nose",
    "High_Cheekbones",
)


@dataclass(frozen=True)
class Triplet:
    """One mined training example; all fields index the mining pool."""

    ref: int
    positives: list[int]  # attribute rows to add
    negatives: list[int]  # attribute rows to remove
    target: int
    violation: int
    distractor: int


class TripletMiner:
    """Samples attribute-flip triplets from a labelled feature pool.

    labels: (N, A) bool attribute matrix; features: (N, D) L2-normalized CLIP
    features aligned with it; proxy_rows: attribute rows forming the identity
    proxy; min_candidates: reject a sampled flip set that fewer than this many
    images satisfy (rare combinations such as bald+female have no usable
    targets, so training never sees them).
    """

    def __init__(
        self,
        labels: torch.Tensor,
        features: torch.Tensor,
        proxy_rows: list[int],
        min_candidates: int = 20,
        seed: int = 0,
    ) -> None:
        if labels.shape[0] != features.shape[0]:
            raise ValueError("labels and features must describe the same images")
        self.labels = labels.bool()
        self.features = features
        self.proxy_rows = proxy_rows
        self.min_candidates = min_candidates
        self.gen = torch.Generator().manual_seed(seed)

    def _satisfies(self, positives: list[int], negatives: list[int]) -> torch.Tensor:
        """Bool mask over the pool: has every T+ and lacks every T-."""
        ok = torch.ones(self.labels.shape[0], dtype=torch.bool)
        if positives:
            ok &= self.labels[:, positives].all(dim=1)
        if negatives:
            ok &= ~self.labels[:, negatives].any(dim=1)
        return ok

    def _identity_agreement(self, ref: int) -> torch.Tensor:
        """Per-image count of identity-proxy attributes matching the reference."""
        proxy = self.labels[:, self.proxy_rows]
        return (proxy == proxy[ref]).sum(dim=1)

    def _sample_flips(self, ref: int, k: int) -> tuple[list[int], list[int]]:
        """Pick k attributes of the reference to flip, split by flip direction."""
        state = self.labels[ref]
        rows = torch.randperm(state.shape[0], generator=self.gen)[:k]
        positives = [int(a) for a in rows if not state[a]]
        negatives = [int(a) for a in rows if state[a]]
        return positives, negatives

    def _pick_violation(
        self, ref: int, positives: list[int], negatives: list[int]
    ) -> int:
        """A near miss: keeps the rest of the query but breaks one constraint.

        With negatives present this is the violation negative of the proposal
        (all T+, at least one T-). An all-positive query has no T- to break, so
        the near miss instead drops one of the requested positives.
        """
        if negatives:
            bad = self.labels[:, negatives].any(dim=1)
            if positives:
                bad &= self.labels[:, positives].all(dim=1)
        else:
            bad = ~self.labels[:, positives].all(dim=1)
        bad[ref] = False
        return self._closest(ref, bad)

    def _closest(self, ref: int, candidates: torch.Tensor) -> int:
        """Index of the candidate most similar to the reference, or -1."""
        rows = candidates.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return -1
        sims = self.features[rows] @ self.features[ref]
        return int(rows[sims.argmax()])

    def sample(self, k: int, max_tries: int = 20) -> Triplet | None:
        """Mine one triplet with k flipped attributes, or None if sampling failed."""
        n = self.labels.shape[0]
        for _ in range(max_tries):
            ref = int(torch.randint(n, (1,), generator=self.gen))
            positives, negatives = self._sample_flips(ref, k)
            if not positives and not negatives:
                continue

            candidates = self._satisfies(positives, negatives)
            candidates[ref] = False
            if int(candidates.sum()) < self.min_candidates:
                continue

            agreement = self._identity_agreement(ref)
            sims = self.features @ self.features[ref]
            # Agreement dominates; similarity only breaks ties within a level.
            score = agreement.float() + 0.5 * sims
            score[~candidates] = float("-inf")
            target = int(score.argmax())

            # Same constraints, least like the reference: forces v_ref to matter.
            drift = agreement.float() - 0.5 * sims
            drift[~candidates] = float("inf")
            drift[target] = float("inf")
            distractor = int(drift.argmin())

            violation = self._pick_violation(ref, positives, negatives)
            if violation < 0 or distractor == target:
                continue
            return Triplet(ref, positives, negatives, target, violation, distractor)
        return None

    def sample_batch(self, size: int, ks: tuple[int, ...] = (1,)) -> list[Triplet]:
        """Mine `size` triplets, drawing each example's flip count from `ks`."""
        out: list[Triplet] = []
        while len(out) < size:
            k = int(ks[int(torch.randint(len(ks), (1,), generator=self.gen))])
            triplet = self.sample(k)
            if triplet is not None:
                out.append(triplet)
        return out


def proxy_rows(attributes: list[str]) -> list[int]:
    """Rows of the identity-proxy attributes in `attributes` (order preserved)."""
    index = {name: i for i, name in enumerate(attributes)}
    missing = [a for a in IDENTITY_PROXY if a not in index]
    if missing:
        raise KeyError(f"identity-proxy attributes absent from labels: {missing}")
    return [index[a] for a in IDENTITY_PROXY]
