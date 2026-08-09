"""Attribute-flip query mining, aimed at the criterion the task is graded by.

There are no retrieval annotations for the train split, so training examples are
synthesised from CelebA labels: sample a reference, flip k of its attributes
(0->1 gives T+, 1->0 gives T-), then look up real images that are correct
answers to the query that produces.

"Correct answer" is the assignment's own rule (src/criterion.py): constraints
satisfied, and Hamming distance <= 2 from the reference over the non-queried
attributes. An earlier version elected a single target by agreement on ten
hand-picked identity-proxy attributes with a CLIP-similarity tiebreak - a
different rule, which taught the model to change roughly ten things while the
benchmark tolerates two (docs/method-proposal-mining-alignment.md).

Two consequences of using the real rule are worth knowing:

* Every member of the valid set is *equally* correct - S3.1.1 defines no
  ordering inside it - so the target is **drawn uniformly**, not elected by an
  argmax. Taking an argmax would teach a preference the criterion does not have.
* The two conditions partition the wrong answers into exactly two families, and
  the informative member of each sits at the boundary, not at the extreme:

      violators - Hamming <= 2 but a constraint is broken. The reference itself
                  is the h = 0 member; it keeps a dedicated slot in the batch
                  (src/training.py) rather than being drawn.
      drifters  - constraints satisfied but Hamming > 2, drawn from the
                  innermost non-empty shell. The old distractor picked the
                  *least* similar candidate, i.e. the easiest negative of its
                  family; one that misses the ball by a single attribute is the
                  one that teaches the decision boundary.

Nothing here touches image features: the rule is pure label logic, which makes
mining ~10x cheaper than the CLIP-similarity version and lets the tests run on
label fixtures alone.
"""

from dataclasses import dataclass

import torch

from src.criterion import MAX_HAMMING, MIN_TARGETS, hamming_to, satisfies

N_NEGATIVES = 8   # per family; measured to be always available (S7 of the proposal)


@dataclass(frozen=True)
class MinedQuery:
    """One mined training example; all fields index the mining pool.

    Every field is required and positional on purpose: this replaced a
    three-field `Triplet`, and a missed call site should fail loudly rather than
    default to something plausible.
    """

    ref: int
    add: list[int]         # attribute rows to add    -> T+
    remove: list[int]      # attribute rows to remove -> T-
    target: int            # one valid answer, drawn uniformly
    violators: list[int]   # inside the ball, break a constraint
    drifters: list[int]    # satisfy the constraints, outside the ball


class Miner:
    """Samples attribute-flip queries from a labelled pool.

    labels: (N, A) bool attribute matrix for the mining pool.
    min_targets: reject a sampled flip set with fewer valid answers than this.
        The default mirrors the benchmark's own inclusion rule, so training
        queries are as hard as graded ones instead of systematically easier.
    n_negatives: how many to draw per family.
    weights: (A,) optional sampling weights over attributes. Rejection is
        attribute-dependent - `Mustache` survives it at 6.6%, `Black_Hair` at
        36.3% - so uniform sampling lets the filter choose the training
        distribution. Weights of 1/retention pre-compensate for it; None keeps
        uniform sampling and today's behaviour (proposal S7.1).
    """

    def __init__(
        self,
        labels: torch.Tensor,
        min_targets: int = MIN_TARGETS,
        n_negatives: int = N_NEGATIVES,
        max_hamming: int = MAX_HAMMING,
        weights: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        self.labels = labels.bool()
        self.min_targets = min_targets
        self.n_negatives = n_negatives
        self.max_hamming = max_hamming
        if weights is not None:
            if weights.shape != (labels.shape[1],):
                raise ValueError(
                    f"weights must have one entry per attribute: got "
                    f"{tuple(weights.shape)} for {labels.shape[1]} attributes"
                )
            weights = weights.double().clamp(min=0)
            if float(weights.sum()) <= 0:
                raise ValueError("weights must have a positive sum")
        self.weights = weights
        self.gen = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------ steps

    def _sample_flips(self, ref: int, k: int) -> tuple[list[int], list[int]]:
        """Pick k attributes of the reference to flip, split by flip direction."""
        a = self.labels.shape[1]
        if self.weights is None:
            rows = torch.randperm(a, generator=self.gen)[:k]
        else:
            rows = torch.multinomial(self.weights, k, replacement=False,
                                     generator=self.gen)
        state = self.labels[ref]
        add = [int(r) for r in rows if not state[r]]
        remove = [int(r) for r in rows if state[r]]
        return add, remove

    def _draw(self, pool: torch.Tensor, count: int) -> list[int]:
        """`count` uniform draws without replacement from a boolean mask."""
        rows = pool.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return []
        take = min(count, rows.numel())
        picked = torch.randperm(rows.numel(), generator=self.gen)[:take]
        return rows[picked].tolist()

    # ----------------------------------------------------------------- public

    def sample(self, k: int, max_tries: int = 50) -> MinedQuery | None:
        """Mine one query with k flipped attributes, or None if sampling failed.

        max_tries is deliberately generous. Acceptance is ~10.6% at k = 3, so a
        small budget would fail often, and `sample_batch` draws a fresh k on
        failure - which would quietly under-represent the hardest flip count.
        At 50 tries that leakage is under half a percent.
        """
        n = self.labels.shape[0]
        for _ in range(max_tries):
            ref = int(torch.randint(n, (1,), generator=self.gen))
            add, remove = self._sample_flips(ref, k)
            if not add and not remove:
                continue

            queried = add + remove
            s = satisfies(self.labels, add, remove)
            h = hamming_to(self.labels, self.labels[ref], queried)
            inside = h <= self.max_hamming

            valid = s & inside
            valid[ref] = False
            if int(valid.sum()) < self.min_targets:
                continue

            violators = (~s) & inside
            violators[ref] = False   # it owns a dedicated batch slot; not drawn twice
            if not bool(violators.any()):
                continue

            drifters = s & ~inside
            if not bool(drifters.any()):
                continue
            # The innermost non-empty shell: the candidates that miss the ball by
            # as little as possible are the ones that teach where its edge is.
            shell = int(h[drifters].min())
            drifters &= h == shell

            return MinedQuery(
                ref=ref,
                add=add,
                remove=remove,
                target=self._draw(valid, 1)[0],
                violators=self._draw(violators, self.n_negatives),
                drifters=self._draw(drifters, self.n_negatives),
            )
        return None

    def sample_batch(self, size: int, ks: tuple[int, ...] = (1,)) -> list[MinedQuery]:
        """Mine `size` queries, drawing each example's flip count from `ks`."""
        out: list[MinedQuery] = []
        while len(out) < size:
            k = int(ks[int(torch.randint(len(ks), (1,), generator=self.gen))])
            query = self.sample(k)
            if query is not None:
                out.append(query)
        return out


def retention_weights(retention: torch.Tensor, floor: float = 0.01) -> torch.Tensor:
    """Sampling weights that undo an attribute-dependent rejection rate.

    Rejection is a filter, so P_realised(a) is proportional to
    P_sampled(a) * retention(a). Sampling at 1/retention(a) therefore lands on a
    uniform realised distribution - inverse-propensity correction for a filter
    that cannot be removed.

    `retention` is the per-attribute column of scripts/measure_mining_rule.py,
    as a fraction. `floor` guards attributes that were never accepted in the
    measurement, which would otherwise divide by zero and swallow the whole
    sampling budget.

    This corrects the marginals only. `Male` and `Mustache` together are far
    worse than either alone, because correlated attributes drag each other out
    of the ball, and a per-attribute weight cannot see that.
    """
    return 1.0 / retention.double().clamp(min=floor)
