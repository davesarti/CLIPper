# Negation-aware mining: training the combiner on the case it fails

**Method proposal — compositional image retrieval on CelebA with frozen CLIP ViT-B/32.**

| | |
|---|---|
| Dataset | CelebA (40 binary attributes); test split as fixed retrieval database (19,962 images) |
| Encoder | Frozen CLIP ViT-B/32, d = 512; features pre-computed once |
| New parameters | **Zero.** This changes the training data distribution and the loss, not the model |
| Files changed | `src/mining.py`, `src/training.py`, `scripts/train_cpas.py` |
| Requires | A full retrain of the combiner (~1 GPU-hour per seed) to evaluate |

## 0. Context for a reader starting here

The task: given a reference image `v_ref`, positive attribute constraints `T+`
and negative constraints `T−`, retrieve test-split images that keep the
reference's identity, have every attribute in `T+`, and none in `T−`.

The current combiner is CPAS-MLP (`src/cpas_mlp.py`). It predicts a reference
weight `γ`, per-attribute step sizes `α_a` and bounded direction bends `Δ_a`,
then composes a single query vector

```
q = normalize( γ·v_ref + Σ_a s_a · α_a · normalize(ŵ_a + Δ_a) ),   s_a = ±1
```

and ranks the database by cosine similarity. It is trained contrastively on
synthetic triplets mined from CelebA **train-split labels** — the test split is
never touched — with an InfoNCE loss at τ = 0.05.

## 1. The problem this fixes

**The model is barely trained on negation.**

`TripletMiner._sample_flips` in `src/mining.py` picks `k` attributes uniformly
at random from all 40 and flips them relative to the reference:

```python
rows = torch.randperm(state.shape[0], generator=self.gen)[:k]
positives = [int(a) for a in rows if not state[a]]   # 0 -> 1  : T+
negatives = [int(a) for a in rows if state[a]]       # 1 -> 0  : T-
```

A flip becomes a negative constraint **only if the reference already has that
attribute**. Mean attribute prevalence across the 40 CelebA attributes is
**0.226**, so in expectation only about 23% of flips are negations and roughly
77% of all trained edits are additions. Queries carrying two or more negations
are rarer still.

**And the negation signal that does exist is numerically drowned.** Each triplet
mines exactly one violation image — a near miss that satisfies `T+` but breaks a
`T−` — and it enters the same softmax as 1023 in-batch targets plus two other
mined negatives (`infonce_loss` in `src/training.py`). At τ = 0.05 it
contributes gradient only when it already ranks near the top. The term meant to
teach "negation is a constraint" is roughly one thousandth of the loss mass.

**Third, the target selection pushes the other way.** The mined target is chosen
for maximal agreement with the reference on ten identity-proxy attributes, with
CLIP similarity as the tie-break. The dominant learning pressure is therefore
"stay near `v_ref`" — the same direction that trades exclusion away.

**Evidence that this is where the failures are.** In
`results/cpas_results.csv`, `corr(R@10, number of negated attributes) = −0.39`
across the 14 benchmark queries, and the worst two are
`+Wearing_Lipstick, -Heavy_Makeup, +Smiling` (0.059) and `+Chubby, -Young`
(0.086).

**What this is not.** It is not a probe-quality problem.
`results/probe_accuracy.csv` gives a macro-mean valid AUC of 0.929, and every
attribute in the benchmark scores 0.909 or better — including all of the ones in
the failing queries. The directions are good; the training distribution is
skewed.

## 2. The method

Four changes, none of which adds a parameter.

### 2.1 Force negations into the sampling distribution

Replace uniform flip sampling with an explicit split. For a triplet with `k`
flips, draw the number of negations `k⁻` from a curriculum-controlled
distribution, then sample `k⁻` attributes from the reference's **ON** set and
`k − k⁻` from its **OFF** set.

Target: roughly **half** of mined triplets carry at least one negation, against
the current ~23%.

Two cases need explicit handling, and getting them wrong biases the pool
silently:
- a reference with fewer than `k⁻` ON attributes — reduce `k⁻` for that sample
  rather than rejecting, or rare-attribute references disappear from training;
- the resulting constraint set must still pass the existing
  `min_candidates = 20` check. **Log the rejection rate before and after this
  change.** Negation-heavy constraint sets are harder to satisfy, so rejection
  will rise; if it rises sharply the effective training distribution is not the
  one intended.

### 2.2 Mine several violations, not one

`_pick_violation` currently returns a single index — the violating image most
similar to the reference. Return the top **8** instead. Widening the set of
near misses per query gives the exclusion signal something to generalize from
rather than one arbitrary point.

`Triplet.violation: int` becomes `Triplet.violations: list[int]`. This is a
breaking change to the dataclass; `build_batch` in `src/training.py` and
`tests/test_mining.py` follow.

### 2.3 Give violations their own loss term

Take the violations out of the big softmax and give them a dedicated objective
with its own weight:

```
L = InfoNCE( q, target ; in-batch targets + distractor + lazy )
  + λ_v · L_violation

L_violation = − log  exp(q·t / τ) / ( exp(q·t / τ) + Σ_i exp(q·v_i / τ) )
```

`L_violation` says exactly one thing: the target must outrank **its own**
violations. Because the sum is over 8 items rather than 1026, each violation
carries real gradient. `λ_v` is a hyperparameter to sweep — start at
`{0.25, 0.5, 1.0, 2.0}`.

Keep the identity distractor and the lazy negative where they are; they solve
different failure modes (ignoring `v_ref`, and returning it unchanged) and are
not the subject of this proposal.

### 2.4 Sample the hard combinations on purpose

Uniform attribute pairs are mostly easy because most attribute pairs are nearly
independent. The queries that break the model pair **correlated** attributes:
`Wearing_Lipstick` and `Heavy_Makeup` have a label correlation of **+0.80**, so
adding one while removing the other asks for a region of the space that is both
small and hard to separate.

With probability ρ (start at 0.3), draw the flip set from a precomputed list of
attribute pairs whose absolute label correlation exceeds 0.3, choosing signs so
the pair is in tension (one added, one removed). Compute the correlation matrix
once from `celeba/list_attr_celeba.txt` restricted to the train split.

This is the change most likely to matter and the one most likely to be
misattributed, because it alters what "average difficulty" means. Ablate it
separately (§4).

## 3. Implementation notes

- `src/mining.py`: `_sample_flips`, `_pick_violation`, the `Triplet` dataclass,
  and a new correlated-pair table. `sample()` gains the curriculum knobs.
- `src/training.py`: `Batch` carries `(B, M, D)` violations separately from the
  two other negatives; `infonce_loss` splits into the base term plus
  `violation_loss`; `run_epoch` sums them.
- `scripts/train_cpas.py`: new flags `--neg-fraction`, `--n-violations`,
  `--lambda-violation`, `--correlated-pair-prob`. Defaults must reproduce
  today's behavior exactly, so the change is opt-in and the baseline stays
  runnable from the same script.
- The existing tests in `tests/test_mining.py` and `tests/test_training.py`
  encode the current contract. Update them deliberately, and add: a test that
  the realized negation fraction matches the requested one within tolerance, and
  a test that a reference with too few ON attributes degrades gracefully instead
  of raising or silently emitting an all-positive query.

## 4. Evaluation

Retrain CPAS-MLP with the new mining and loss, **3 seeds**, and compare against
3 seeds of the current recipe trained in the same campaign with the same probe
file. Absolute numbers in this project are not comparable across probe refits;
the comparable quantity is the delta within a campaign.

Report on the 14-query benchmark: R@{1,5,10}, P@{1,5,10}, per query and mean.

Add two metrics, because R@10 alone cannot tell you whether the mechanism
worked:

> **Violation rate @10** — fraction of returned top-10 images breaking at least
> one constraint of their query, from CelebA labels. This is the direct target.
>
> **Negation-subset R@10** — mean R@10 restricted to the benchmark queries that
> contain at least one negation. The overall mean is diluted by positive-only
> queries this change is not meant to help.

**Resolution limit.** Seed spread on this benchmark reaches 0.019 R@10 for some
variants. Treat any difference below **0.02 R@10 as unresolved**, and say so
rather than reporting a 0.01 gain as a result.

## 5. Ablations

Each row adds one component to the row above, so the deltas attribute the gain:

| # | Variant | Isolates |
|---|---|---|
| 0 | Current mining and loss | reference point |
| 1 | + forced negation fraction (§2.1) | does simply seeing more negations help? |
| 2 | + 8 violations (§2.2) | does a richer near-miss set help? |
| 3 | + separate weighted violation loss (§2.3) | was the signal there but drowned? |
| 4 | + correlated-pair sampling (§2.4) | does training on the hard cases transfer? |

Row 3 is the cheapest possible test of the hypothesis that the training signal,
not the architecture, is the limiting factor: it changes no data at all, only
how the already-mined violations enter the loss.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Forcing negations shifts the mining pool toward frequent attributes, because rare combinations fail `min_candidates` | log rejection rate and the realized per-attribute frequency before and after; report both |
| `λ_v` too high makes the model optimize violation ranking at the expense of retrieval | sweep `λ_v` and report the curve, not just the winner |
| Correlated-pair sampling makes training look worse on the average query while helping the hard ones | this is why negation-subset R@10 is reported separately from the mean |
| The gain is real but unresolvable at 14 queries, 6 of which have no negation | report per-query results; consider enlarging the benchmark with held-out queries of controlled `k` before concluding |
| `Triplet` dataclass change silently breaks a call site | it is a required positional field; keep it required rather than defaulting it, so a missed call site fails loudly |

## 7. Relationship to the exclusion re-rank proposal

`docs/method-proposal-exclusion-rerank.md` attacks the same failure from the
scoring side and needs no retraining. The two are independent: one changes what
the model learns, the other changes how candidates are scored at query time.

Measure them **separately first**. If both are applied at once and the number
moves, neither is attributable — and they could equally be redundant, since a
model trained to respect exclusion may leave nothing for a re-rank to fix. Run
the re-rank first because it is cheaper, then this, then the combination as a
third cell.
