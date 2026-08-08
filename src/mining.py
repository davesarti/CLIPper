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
    violations: list[int]  # near misses that break a constraint, closest first
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
        neg_fraction: float | None = None,
        n_violations: int = 1,
        pairs: list[tuple[int, int]] | None = None,
        pair_prob: float = 0.0,
    ) -> None:
        """The last four arguments are the negation-aware knobs; their defaults
        reproduce the original uniform mining exactly, down to the RNG draws.

        neg_fraction: target share of flips that are negations. Uniform
            sampling makes a flip a negation only when the reference already
            has the attribute, so at CelebA's mean prevalence of 0.226 roughly
            77% of trained edits are additions
            (docs/method-proposal-negation-mining.md S2.1). None keeps that.
        n_violations: near misses mined per triplet (S2.2).
        pairs / pair_prob: with probability `pair_prob`, draw the flip set from
            `pairs` - correlated attributes put in tension, one added and one
            removed, which is the shape of the queries that fail (S2.4).
        """
        if labels.shape[0] != features.shape[0]:
            raise ValueError("labels and features must describe the same images")
        if n_violations < 1:
            raise ValueError("n_violations must be at least 1")
        self.labels = labels.bool().to(features.device)
        self.features = features
        self.proxy_rows = proxy_rows
        self.min_candidates = min_candidates
        self.neg_fraction = neg_fraction
        self.n_violations = n_violations
        self.pairs = pairs or []
        self.pair_prob = pair_prob
        self.gen = torch.Generator().manual_seed(seed)
        self._used_pair = False
        # Forcing negations makes constraint sets harder to satisfy, so the
        # rejection rate rises and the realized training distribution is no
        # longer the requested one. Counted here so a run can report it.
        self.stats = {"tries": 0, "accepted": 0, "empty_flips": 0,
                      "too_few_candidates": 0, "no_violation": 0,
                      "flips": 0, "negated_flips": 0, "from_pairs": 0}

    def _satisfies(self, positives: list[int], negatives: list[int]) -> torch.Tensor:
        """Bool mask over the pool: has every T+ and lacks every T-."""
        ok = torch.ones(
            self.labels.shape[0], dtype=torch.bool, device=self.labels.device
        )
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
        positives: list[int] = []
        negatives: list[int] = []
        self._used_pair = False

        if self.pair_prob and self.pairs and k >= 2:
            draw = float(torch.rand(1, generator=self.gen))
            if draw < self.pair_prob:
                pair = self.pairs[int(torch.randint(len(self.pairs), (1,),
                                                    generator=self.gen))]
                a, b = pair
                # In tension means one added and one removed, which requires
                # the reference to disagree on the pair; if it does not, fall
                # through to ordinary sampling rather than forcing it.
                if bool(state[a]) != bool(state[b]):
                    on, off = (a, b) if state[a] else (b, a)
                    negatives, positives = [on], [off]
                    k -= 2
                    self._used_pair = True

        if k > 0:
            taken = set(positives) | set(negatives)
            if self.neg_fraction is None:
                rows = torch.randperm(state.shape[0], generator=self.gen)
                rows = [int(a) for a in rows if int(a) not in taken][:k]
                positives += [a for a in rows if not state[a]]
                negatives += [a for a in rows if state[a]]
            else:
                positives, negatives = self._split_flips(state, k, positives,
                                                         negatives, taken)

        return positives, negatives

    def _split_flips(
        self,
        state: torch.Tensor,
        k: int,
        positives: list[int],
        negatives: list[int],
        taken: set[int],
    ) -> tuple[list[int], list[int]]:
        """Draw k flips with a requested share of them negations.

        The number of negations is binomial(k, neg_fraction), then clamped to
        what the reference can actually supply: a reference with fewer ON
        attributes than the draw asks for gets fewer negations rather than
        being rejected, or rare-attribute references vanish from training.
        """
        on = [a for a in range(state.shape[0]) if state[a] and a not in taken]
        off = [a for a in range(state.shape[0]) if not state[a] and a not in taken]
        k_neg = int((torch.rand(k, generator=self.gen) < self.neg_fraction).sum())
        k_neg = min(k_neg, len(on))
        k_pos = min(k - k_neg, len(off))
        # Whatever the OFF set could not supply goes back to the ON set, so the
        # realized flip count stays k whenever the reference allows it at all.
        k_neg = min(k_neg + (k - k_neg - k_pos), len(on))

        def draw(pool: list[int], count: int) -> list[int]:
            if count <= 0:
                return []
            order = torch.randperm(len(pool), generator=self.gen)[:count]
            return [pool[int(i)] for i in order]

        return positives + draw(off, k_pos), negatives + draw(on, k_neg)

    def _pick_violations(
        self, ref: int, positives: list[int], negatives: list[int]
    ) -> list[int]:
        """Near misses: keep the rest of the query but break one constraint.

        With negatives present these are the violation negatives of the CPAS
        proposal (all T+, at least one T-). An all-positive query has no T- to
        break, so the near miss instead drops one of the requested positives.

        Returns up to `n_violations` indices, most similar to the reference
        first. One arbitrary near miss gives the exclusion signal nothing to
        generalize from (docs/method-proposal-negation-mining.md S2.2).
        """
        if negatives:
            bad = self.labels[:, negatives].any(dim=1)
            if positives:
                bad &= self.labels[:, positives].all(dim=1)
        else:
            bad = ~self.labels[:, positives].all(dim=1)
        bad[ref] = False
        return self._closest(ref, bad, self.n_violations)

    def _closest(
        self, ref: int, candidates: torch.Tensor, count: int = 1
    ) -> list[int]:
        """Indices of the `count` candidates most similar to the reference.

        Returns fewer than `count` when the candidate set is smaller, and an
        empty list when it is empty.
        """
        rows = candidates.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return []
        sims = self.features[rows] @ self.features[ref]
        top = sims.topk(min(count, rows.numel())).indices
        return [int(rows[i]) for i in top]

    def sample(self, k: int, max_tries: int = 20) -> Triplet | None:
        """Mine one triplet with k flipped attributes, or None if sampling failed."""
        n = self.labels.shape[0]
        for _ in range(max_tries):
            self.stats["tries"] += 1
            ref = int(torch.randint(n, (1,), generator=self.gen))
            positives, negatives = self._sample_flips(ref, k)
            if not positives and not negatives:
                self.stats["empty_flips"] += 1
                continue

            candidates = self._satisfies(positives, negatives)
            candidates[ref] = False
            if int(candidates.sum()) < self.min_candidates:
                self.stats["too_few_candidates"] += 1
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

            violations = self._pick_violations(ref, positives, negatives)
            if not violations or distractor == target:
                self.stats["no_violation"] += 1
                continue
            # Counted here, not at sampling time: a rejected constraint set
            # never reaches training, so only accepted flips describe the
            # distribution the model actually sees.
            self.stats["accepted"] += 1
            self.stats["flips"] += len(positives) + len(negatives)
            self.stats["negated_flips"] += len(negatives)
            self.stats["from_pairs"] += int(self._used_pair)
            return Triplet(ref, positives, negatives, target, violations, distractor)
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


    def summary(self) -> dict[str, float]:
        """Realized mining distribution, for the before/after comparison.

        `rejection_rate` is the share of sampling attempts thrown away, and
        `negation_share` the share of accepted flips that are negations - the
        two numbers that say whether the requested training distribution is the
        one the model actually saw.
        """
        tries = max(self.stats["tries"], 1)
        flips = max(self.stats["flips"], 1)
        accepted = max(self.stats["accepted"], 1)
        return {
            "tries": self.stats["tries"],
            "accepted": self.stats["accepted"],
            "rejection_rate": 1 - self.stats["accepted"] / tries,
            "negation_share": self.stats["negated_flips"] / flips,
            "empty_flips": self.stats["empty_flips"] / tries,
            "too_few_candidates": self.stats["too_few_candidates"] / tries,
            "no_violation": self.stats["no_violation"] / tries,
            "from_pairs": self.stats["from_pairs"] / accepted,
        }


def correlated_pairs(
    labels: torch.Tensor, threshold: float = 0.3
) -> list[tuple[int, int]]:
    """Attribute pairs whose absolute label correlation exceeds `threshold`.

    Uniform attribute pairs are mostly easy because most CelebA attributes are
    nearly independent. The queries that break the model pair correlated ones -
    Wearing_Lipstick and Heavy_Makeup correlate at +0.80, so adding one while
    removing the other asks for a region that is both small and hard to
    separate (docs/method-proposal-negation-mining.md S2.4).

    labels: (N, A) bool; computed once from the train split.
    """
    x = labels.float()
    x = x - x.mean(dim=0)
    sd = x.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)
    corr = (x.T @ x) / (sd[:, None] * sd[None, :])
    rows, cols = (corr.abs() > threshold).triu(diagonal=1).nonzero(as_tuple=True)
    return [(int(i), int(j)) for i, j in zip(rows, cols)]


def proxy_rows(attributes: list[str]) -> list[int]:
    """Rows of the identity-proxy attributes in `attributes` (order preserved)."""
    index = {name: i for i, name in enumerate(attributes)}
    missing = [a for a in IDENTITY_PROXY if a not in index]
    if missing:
        raise KeyError(f"identity-proxy attributes absent from labels: {missing}")
    return [index[a] for a in IDENTITY_PROXY]
