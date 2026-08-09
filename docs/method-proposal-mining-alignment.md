# Mining alignment: train the combiner on the criterion it is graded by

**Design spec — CelebA compositional retrieval, frozen CLIP ViT-B/32.**

| | |
|---|---|
| Changes | `src/mining.py`, `src/training.py`, `scripts/train_cpas.py`, new `src/criterion.py` |
| New parameters | **Zero.** This changes what counts as a correct answer during training, not the model |
| Cost | Mining gets ~10× *cheaper*; a retrain is ~1 GPU-hour per seed, optional |
| Status | Design agreed, feasibility **measured** (§7) — no blockers left. Implementation not started |

---

## 0. Why

`docs/method.md` §1 establishes the criterion the task is defined by. Assignment
§3.1.1: an image is a correct answer **iff** (1) it satisfies the query's
constraints **and** (2) its non-queried attributes are within **Hamming distance
2** of the reference's. Verified by exact set reconstruction on all 33,052
(query, reference) pairs of `celeba_evaluation.json`.

`build_val_benchmark` was corrected to this rule. **`src/mining.py` was not.** It
still selects the training target by agreement on ten hand-picked identity-proxy
attributes with a CLIP-similarity tiebreak:

```python
score = agreement.float() + 0.5 * sims     # agreement over IDENTITY_PROXY (10 of 40)
target = int(score.argmax())
```

The two definitions disagree in both directions:

- a candidate matching all 10 proxy attributes but differing on 20 of the other
  28 is the miner's ideal target, and **is not a valid answer** under §3.1.1;
- a candidate at Hamming 2 whose two mismatches fall on proxy attributes **is a
  valid answer**, and the miner ranks it poorly.

In one sentence: **we are teaching the model to change too many things.** We show
it an exemplar that differs from the reference in ten places, then grade it on a
rule that tolerates two.

This is `docs/method.md` §9 item 3, and the same defect the val benchmark had —
where it was measured to cost validation-to-test transfer.

### Expected effect, stated up front

The miner trains **CPAS-MLP**, which enters the final score only through
`w·(q·d)` and contributes **~0.004 R@10** (§7). Realistically this change moves
the **cosine-space row** of the results table (0.267), not the headline 0.465.
That row is the honest ceiling of cosine-space composition and belongs in the
report correct — but nobody should expect the top-line number to move. Budget
GPU hours accordingly: `docs/method.md` §9 item 1 (attribute head on the full
split, 7× amplification) has a far better return.

---

## 1. Extract the criterion — new `src/criterion.py`

The rule currently lives inline inside `build_val_benchmark`. Re-implementing it
in the miner is how the two copies drift apart again, which is the whole reason
this document exists.

```python
MAX_HAMMING = 2   # moves here from evaluation.py

def satisfies(labels, add, remove) -> torch.Tensor:
    """(N,) bool: has every attribute in `add`, none in `remove`."""

def hamming_to(labels, ref, queried) -> torch.Tensor:
    """(N,) int: disagreements with `labels[ref]` over the NON-queried columns.

    `queried` is the set of attribute rows named by the query; they differ from
    the reference by construction and are excluded, exactly as in §3.1.1.
    """
```

Everything composes from these two:

```
valid     =  satisfies & (hamming <= MAX_HAMMING)
violators = ~satisfies & (hamming <= MAX_HAMMING)
drifters  =  satisfies & (hamming >  MAX_HAMMING)
```

`src/mining.py` and `src/evaluation.py` both import from here. `criterion.py`
imports only `torch`, so there is no cycle.

`build_val_benchmark` is refactored to call these instead of its inline version.
Its existing test — the val benchmark reproduces §3.1.1 soundly *and* completely
— is the regression guard for that refactor and must keep passing unchanged.

---

## 2. Data structure

```python
@dataclass(frozen=True)
class MinedQuery:
    ref: int
    add: list[int]         # attribute rows to add       (was: positives)
    remove: list[int]      # attribute rows to remove    (was: negatives)
    target: int            # one valid answer, sampled uniformly
    violators: list[int]   # inside the ball, break a constraint
    drifters: list[int]    # satisfy the constraints, outside the ball
```

Renames, and why:

- `Triplet` → **`MinedQuery`**. It has six fields, not three.
- `positives`/`negatives` → **`add`/`remove`**. `negatives` currently means
  *attribute rows to remove*, while everywhere else in contrastive learning it
  means *wrong examples*. Both senses live in the same file today.
- `violation`/`distractor` → **`violators`/`drifters`**, now lists, and named
  after which half of §3.1.1 they fail.

This is a breaking change to the dataclass. Keep every field **required and
positional** so a missed call site fails loudly rather than defaulting to
something plausible.

---

## 3. The new `sample()`

```
0.  (optional) attribute sampling weights, see §7.1 — default off, uniform
1.  ref at random; flip k attributes → add / remove
2.  h = hamming_to(labels, ref, queried)          # computed ONCE
    s = satisfies(labels, add, remove)
3.  valid = s & (h <= 2);  valid[ref] = False
    ── REJECT and resample if |valid| < MIN_TARGETS
4.  target = uniform draw from valid
5.  violators = ~s & (h <= 2);  violators[ref] = False
    → M uniform draws without replacement
6.  drifters = s & (h > 2)
    → M uniform draws without replacement from the innermost non-empty
      shell (smallest h ≥ 3)
    ── REJECT if either family is empty; if a family holds 1..M-1 members,
       take all of them and pad (§4.2) rather than rejecting
```

Three things this gets right that the current version does not:

**The target is drawn, not elected.** Under §3.1.1 every member of `valid` is
equally correct; there is no "more correct" one. Taking an argmax teaches a
preference the criterion does not have. Drawing uniformly makes the distribution
of targets the model is asked to hit *be* the distribution of valid answers.

**The drifters are at the boundary, not at the extreme.** Today's distractor
minimises `agreement − 0.5·sims`, i.e. it picks the candidate **furthest** from
the ball. A candidate at Hamming 15 is rejected by any model and teaches
nothing; a candidate at Hamming 3 is the decision boundary. We have been mining
the *easiest* negative of its family.

**The rejection rule is the benchmark's own.** `MIN_TARGETS = 5` mirrors the
assignment's inclusion rule ("only references with ≥ 5 valid targets"), so the
difficulty of training queries matches the difficulty of graded queries instead
of being systematically easier.

`ref` is excluded from the violator pool: it occupies a dedicated slot (§4) and
would otherwise be able to appear twice in the same row, counting double in the
softmax.

---

## 4. Batching, negatives, and the false-negative mask

### 4.1 The negatives tensor

```python
negatives = torch.cat([
    features[violators],           # (B, M, D)
    features[drifters],            # (B, M, D)
    features[refs].unsqueeze(1),   # (B, 1, D)   ← lazy, always present
], dim=1)                          # (B, 2M+1, D)
```

**`lazy` keeps a dedicated slot.** It is formally a violator — Hamming 0,
breaks every queried constraint — but it must not be folded into the sampled
`violators`, because then it would only reach the batch on the epochs it happens
to be drawn. The reference is the strongest attractor in the database: if `q`
collapses onto `v_ref` that image scores 1.0, the maximum attainable. A row that
did not draw it is a row whose gradient has nothing pushing against the
collapse. One column out of 2M+1 buys a **structural** guarantee instead of a
probabilistic one.

(What this protects is the collapse, not the literal case: the source image is
excluded from the ranking at evaluation time. But if `q ≈ v_ref` its *neighbours*
score high too, and those are not excluded.)

### 4.2 Padding

A family can hold fewer than M candidates. Pad with a **mask**, not by
repetition: a repeated negative silently carries double weight in the softmax,
which is a weighting nobody chose and nobody will remember. `Batch` gains
`neg_mask: (B, 2M+1)` and `infonce_loss` sends masked slots to `-inf`. This
mirrors `pad_queries`, which already solves the same problem for attribute
slots.

### 4.3 The in-batch false-negative mask

The mined negatives are clean by construction — a violator breaks a constraint,
a drifter is outside the ball, neither is a valid answer. The problem is
confined to the **in-batch** negatives: row `i`'s denominator contains every
other row's target, and nothing stops one of those from being a valid answer to
query `i`. When that happens the loss actively pushes the query away from an
image that would count as correct at evaluation.

Rough size: a valid set of ~50 in a 27k pool, batch 1024 → about **2 false
negatives per row**. Few, but they are the semantically closest candidates, so at
τ = 0.05 they carry disproportionate gradient.

```python
false_neg[i, j] = (j != i) and target_j is valid for query_i     # (B, B) bool
```

Built in `build_batch` from the label matrix already in memory: `B×B×40`
comparisons, ~42 M per batch, negligible. `infonce_loss` sends those logits to
`-inf`. **The diagonal is never masked** — dedicated test.

**Implementation trap.** Validity is evaluated per *row*: row `i` has its own
reference and its own queried columns, so the Hamming distance in `false_neg[i,
j]` is taken between `target_j` and `ref_i` over the columns **not queried by row
`i`**. There is no single shared column set across the batch, and vectorising as
if there were is the mistake that will produce a mask that looks reasonable and
is wrong. Either loop over rows (B = 1024, cheap) or build a per-row column mask
explicitly.

### 4.4 Signature changes

| function | change |
|---|---|
| `build_batch(queries, features, directions)` | gains `labels`; returns `neg_mask` and `false_neg` in `Batch` |
| `infonce_loss(q, targets, negatives, tau)` | gains `neg_mask`, `false_neg` |
| `recall_at_1(q, batch)` | must respect both masks |
| `run_epoch(...)` | passes `labels` through |
| `scripts/train_cpas.py` | passes `labels`; drops `proxy_rows` |

---

## 5. Deletions

The miner no longer uses CLIP similarity for anything, so:

| deleted | consequence |
|---|---|
| `_identity_agreement`, `IDENTITY_PROXY`, `proxy_rows` | the miner was their last consumer (the val benchmark already dropped them) — delete outright |
| `_closest`, and the `features` constructor argument | **the miner becomes pure label logic** |
| `features @ features[ref]` (~75 M ops/example) | mining runs **~10× faster** |

Two consequences worth having: miner tests no longer need fake feature tensors,
and a full-pool retrain becomes practical again (it costs 2h15 per seed today,
almost all of it in mining).

---

## 6. Parameters

| name | value | rationale |
|---|---|---|
| `MAX_HAMMING` | **2, unchanged** | assignment §3.1.1. §7 confirmed we do not need to relax it |
| `MIN_TARGETS` | **5, unchanged** | assignment §3.1.1 inclusion rule. §7 confirmed |
| flip counts `k` | **{1, 2, 3}, unchanged** | k = 3 survives at 10.6%, which costs ~9.4 attempts per accepted example — still cheaper than today because the CLIP tiebreak is gone (§7) |
| attribute sampling weights | **off by default**, `1/retention` when on | §7.1 |
| `M` (per family) | 8 | the count measured at +0.02 R@10 in the negation-mining work, before the scoring change. §7 confirms both families always hold far more than 8 |
| drifter shell | smallest `h ≥ 3` that is non-empty | §7 measured the innermost shell at **h = 3 for every k**, so in practice this is always 3 |
| violator draw | uniform inside the ball | `ref` already covers the `h = 0` extreme |
| within-shell draw | uniform | keeps the miner label-only. Picking the CLIP-nearest would be harder negatives but reintroduces the `features` dependency and its cost — a one-line upgrade if it is ever wanted |

Nothing about the assignment's rule is relaxed. That was a live option before §7
and is now closed: no second criterion has to be explained in the report.

---

## 7. Feasibility — measured

`scripts/measure_mining_rule.py`, 5000 sampled references per k, on the full
train pool (146,493 images) with the same 90/10 split training uses. Read-only:
no model, no training, no writes.

### 7.0 Results

| k | old % | new % | valid (q1/med/q3) | violators | drifters | innermost shell | negations |
|---|---|---|---|---|---|---|---|
| 1 | 100.0 | 53.7 | 15 / **43** / 148 | 72 / **232** / 630 | 21k / **35k** / 76k | **3** | 31.5% |
| 2 | 97.1 | 23.4 | 9 / **24** / 70 | 222 / **567** / 1406 | 5.6k / **13k** / 27k | **3** | 57.9% |
| 3 | 85.6 | 10.6 | 8 / **17** / 43 | 515 / **1124** / 2303 | 1.7k / **4.3k** / 12k | **3** | 79.2% |

Everything the design assumed holds:

- **The valid set is not degenerate** (median 43/24/17), so drawing the target
  uniformly from it is a real choice and not an argmax in disguise.
- **Both negative families always exceed M = 8** by two to three orders of
  magnitude. Padding (§4.2) will be a rare edge case, not the norm.
- **The innermost drifter shell is h = 3 at every k.** The boundary is populated,
  so "the negative that misses the ball by one" exists and the family is the
  hard negative it was designed to be — not a restatement of today's distractor.
- **Negations become more frequent, for free**: 31.5 / 57.9 / 79.2% against the
  ~23% baseline. Removing a rare attribute keeps you inside a Hamming-2 ball far
  more easily than adding one does, so the stricter rule *favours* negation. Part
  of what the removed negation-mining work was built to fix solves itself here.
- **k = 3 stays.** 10.6% acceptance means ~9.4 attempts per accepted example, but
  each attempt now costs ~5.5 M ops instead of ~80 M (the CLIP tiebreak is gone),
  so k = 3 mining is *cheaper* than today. With `max_tries = 20` about 10% of k=3
  requests return `None`; `sample_batch` already retries, so batches still fill.

### 7.1 What the measurement changed: sampling weights

One assumption did **not** hold, and it produced the only design change.

**The hypothesis that was wrong.** We expected the stricter rule to penalise the
graded attributes as a class, because they are the entangled ones. It does not:
retention is **22.7% for graded vs 24.5% for non-graded**, and the split is flat
at every k (50.2/55.2, 22.9/25.3, 12.0/13.0). There is no category-level bias.

**The real problem is the spread *within* the graded set** — a factor of five:

| survive best | | starved | |
|---|---|---|---|
| Black_Hair | 36.3% | Wearing_Lipstick | 18.5% |
| Wavy_Hair | 35.4% | Eyeglasses | 17.5% |
| Young | 32.0% | Wearing_Hat | 17.0% |
| Smiling | 30.6% | Male | 16.2% |
| Blond_Hair | 30.3% | **Chubby** | **8.5%** |
| Heavy_Makeup | 24.1% | **Mustache** | **6.6%** |

The rule does not discriminate graded from non-graded; it discriminates
**entangled and rare** from **separable and common**. Flipping `Male` drags
facial hair, makeup and lipstick with it, so the Hamming-2 budget breaks. And the
coincidence is bad: the starved attributes are **exactly the ones the benchmark
already fails on** — `+Wearing_Lipstick, −Heavy_Makeup, +Smiling` (R@10 0.059)
and `+Chubby, −Young` (0.086). Left alone, the new miner would train *least* on
the cases that are already worst.

**Why this happens.** Rejection is a filter, and the filter is
attribute-dependent:

```
P_realised(a)  ∝  P_sampled(a) × retention(a)
```

With uniform sampling, `P_realised(a) ∝ retention(a)` — `Black_Hair` reaches
training 5.5× more often than `Mustache`, purely as a side effect. **Nobody chose
that weighting.** The rejection step is silently picking the training
distribution.

**The correction.** To land on a target distribution `T`, sample with
`P_sampled(a) ∝ T(a) / retention(a)`. With `T` uniform the weights are
`1/retention` — inverse-propensity correction, pre-compensating a filter that
cannot be removed:

| | retention | weight | attempts | accepted |
|---|---|---|---|---|
| Mustache | 6.6% | 15.2 | 152 | ~10 |
| Black_Hair | 36.3% | 2.75 | 28 | ~10 |

Cost: mean acceptance moves from the arithmetic to the **harmonic** mean of the
retentions. With only three or four attributes below 20% the tail is short —
expect **10–20% more attempts**, still well under today's mining cost.

**`T` is today's distribution, not the benchmark's.** Tilting the sampler toward
graded attributes is tempting and probably productive, and it is deliberately
**not** done here, for two reasons:

1. *Attributability.* This change exists to fix **what counts as a correct
   answer**. Changing **which queries get trained** in the same commit means a
   moving number cannot be assigned to either. It is the same discipline as
   initialising CPAS exactly at the fixed rule, defaulting new flags off, and
   recomputing the baseline row inside every run.
2. *It is a different question.* "Should we train more on the attributes we are
   graded on?" is legitimate — the query shapes are given in the assignment, and
   `build_val_benchmark` already uses them. But it is an optimisation **toward
   the benchmark**, and it has to be run on purpose, measured alone, and stated
   in the report — not smuggled inside a change that claims to be about
   something else.

Two knobs, two experiments. Restore neutrality first; tilt afterwards, if at all.

**Implementation.** `scripts/measure_mining_rule.py` writes the retention column
to `results/mining_retention.pt`; `Miner(weights=...)` draws flips with
`torch.multinomial` instead of `torch.randperm`, and
`train_cpas.py --sampling-weights results/mining_retention.pt` wires the two
together. `mining.retention_weights` does the inversion, with a floor so an
attribute that never survived the measurement cannot divide by zero and swallow
the whole sampling budget. **Default is uniform**, so the weighted run is an
ablation against the unweighted one rather than a silent change.

The table is tied to the pool it was measured on. Refit on a different pool and
it corrects for a filter that is no longer the one running — the saved file
records `pool`, `images` and `samples` so this is checkable.

---

## 8. Testing

New and updated contracts, all model-free like the existing 97:

- `satisfies` / `hamming_to` reproduce the mask `build_val_benchmark` builds
  today — **sound and complete**, on a hand-built fixture. This is the guard for
  the §1 refactor.
- A mined `target` is always in `valid`.
- Every `violator` breaks at least one constraint **and** has `h ≤ 2`.
- Every `drifter` satisfies all constraints **and** has `h > 2`.
- `ref` never appears in `violators`.
- `lazy` is present in the negatives tensor of **every** row.
- The false-negative mask never masks the diagonal.
- With `false_neg` all-false and `neg_mask` all-true, `infonce_loss` is
  bit-identical to the current implementation — the regression guard that lets
  the ablation attribute any change to this work alone.
- A query whose valid set is smaller than `MIN_TARGETS` is rejected, and
  `sample()` degrades to `None` rather than raising after `max_tries`.

---

## 9. Work order

| # | step | status | effort |
|---|---|---|---|
| 0 | Feasibility measurement (§7) | ✅ done — `scripts/measure_mining_rule.py` | — |
| 1 | `src/criterion.py` + tests; refactor `build_val_benchmark` onto it | ✅ done | — |
| 2 | Rewrite `sample()` and `MinedQuery` + tests | ✅ done | — |
| 3 | Negatives tensor, padding mask, false-negative mask + tests | ✅ done | — |
| 4 | Attribute sampling weights (§7.1), default off | ✅ done | — |
| 5 | Renames across call sites | ✅ done | — |
| 6 | Retrain CPAS-MLP, 3 seeds | **todo** | ~2h15/seed on the full pool |
| 7 | Evaluate: `run_cpas_ablation.py`, then `run_exclusion_rerank.py` | **todo** | minutes |

Steps 1–5 land as one change: they are a single rewrite of what counts as a
correct answer, and splitting them would leave the repo in states where the
miner and the val benchmark disagree — the exact condition this fixes. Tests:
**113 passing**, plus 4 pre-existing failures that need CelebA on disk and are
unrelated to this change (verified by re-running them on a clean tree).

Step 6 is what the retrain is *for*, and it is worth being explicit: it will not
make cosine-space composition competitive with the attribute-space score. The
0.267 → 0.465 gap is structural (`docs/method.md` §1: correct answers sit at
cosine ranks in the thousands), not a training-quality gap. What it buys is the
right to say **"we trained the fusion module against the correct criterion and
its ceiling is still here"** — which closes the obvious objection that the
combiner was simply trained badly. Without step 6 that claim is not available.

---

## 10. Risks

| risk | mitigation |
|---|---|
| ~~Rejection rate explodes~~ | **measured** (§7): 53.7 / 23.4 / 10.6%, workable at every k |
| The rejection step reshapes the training distribution | **measured and real** (§7.1): a 5× spread within the graded attributes, worst on the queries that already fail. Corrected by inverse-retention sampling weights; the correction and its limits are stated, not hidden |
| Weights only fix the marginals | stated in §12. `Male + Mustache` together is far worse than either alone, and per-attribute weights cannot see that |
| Breaking dataclass change silently missed at a call site | every field required and positional, so a miss is an exception rather than a default |
| The masks are subtly wrong and quietly weaken the loss | the all-off regression test pins the current behaviour bit-for-bit |
| Effort spent here displaces §9 item 1 of `method.md` | §0 states the expected effect; item 1 has 7× amplification and should go first if hours are scarce |

---

## 11. Out of scope

Deliberately **not** part of this change, to keep the ablation attributable:

- Forcing a negation fraction in the flip sampler, multiple violations per
  example as a *separate weighted loss term*, and correlated-pair sampling.
  These were implemented, measured at **+0.02 R@10**, and removed as subsumed by
  the scoring change (`docs/method.md` §9). Do not quietly reintroduce them here.
- Multi-positive InfoNCE. Considered and deferred: uniform sampling covers the
  valid set in distribution, and the false-negative mask removes the actual
  contradiction. If it is revisited, the mask is a prerequisite for it anyway.
- Anything about the attribute head, the scoring function, or the re-rank.

---

## 12. Limitations, recorded on purpose

Things this design does **not** solve. None is a reason not to ship it; all are
reasons not to overclaim in the report.

**The sampling weights are a first-order correction.** They fix the marginal
distribution over attributes, not the joint. `Male` alone retains at 16.2%, but
`Male + Mustache` together is far worse than either — they are correlated, so
they drag each other out of the ball, and a per-attribute weight cannot see it.
The second-order fix is pair weighting, which is the correlated-pair sampling
already removed from the repo; do not reintroduce it here (§11).

**The weights are pooled over k.** Retention drops sharply with k (53.7 → 23.4 →
10.6 overall), and almost certainly does so unevenly across attributes. One
table for all three flip counts is a simplification.

**The retention table is tied to one pool.** It was measured on the full train
split. Fitting probes or mining on the 30k sample changes the numbers, so the
table has to be re-measured if the pool changes — otherwise the correction
corrects for a filter that is no longer the one running.

**`sample_batch` under-represents high k, before any weighting.** It draws `k`,
calls `sample(k)`, and on failure loops and draws a **new** `k`. Since `sample()`
fails ~10% of the time at k = 3 (§7) and ~0% at k = 1, the realised mix is about
34.6 / 34.3 / 30.9 instead of a third each. Small, but it is a bias nobody chose,
and the fix is one line: retry with the same `k` rather than redrawing. Measure
the realised mix after implementing.

**All of this improves chain A only.** The combiner reaches the final score
through `w·(q·d)` and contributes ~0.004 R@10. This work moves the cosine-space
row (0.267), which belongs in the report correct — but the headline number is
governed by attribute-prediction accuracy, not by any of this (§0).

---

## 13. Open questions

| question | why it is open | cost to answer |
|---|---|---|
| Should the sampler be tilted **toward** the graded attributes? | Legitimate — query shapes are given in the assignment and `build_val_benchmark` already uses them — but it is optimisation toward the benchmark and must be its own declared experiment (§7.1) | one more retrain |
| Multi-positive InfoNCE instead of one sampled target | Deferred: uniform sampling covers the valid set in distribution and the false-negative mask removes the actual contradiction. If revisited, the mask is a prerequisite anyway | medium |
| Per-**pair** retention weighting | The honest second-order fix, but it overlaps with removed work | high; needs its own justification |
| Does the realised k mix need fixing? | See §12; measure first | one line + a measurement |

---

## 14. Decisions taken in discussion, living outside this document

Recorded here because they change what the report has to say, and because
nothing else in the repo currently records them.

**CPAS-MLP will not be wired into the attribute-space score.** The option was
considered and rejected: it adds no expressive power (a query vector contributes
`q·d`, a linear functional of `d`; the criterion needs a per-candidate
classifier, a threshold and a conjunction — none expressible that way), and there
is a plausible mechanism by which it would *hurt*, since the cosine term's only
remaining job is carrying identity and `q = v_ref` is the purest identity signal
available.

**Consequence, and it needs fixing.** `docs/method.md` §5 states that "the cosine
term keeps a composite query embedding in the ranking, so the fusion module still
contributes and the assignment's Φ requirement is met by a component that is
actually in the score." As implemented, `run_attribute_retrieval.py` computes the
cosine term as `features @ features[src].T` — the **raw reference**, no combiner.
That sentence is false today. Either the wiring lands (~25 lines, one file) or
§5 is rewritten to state plainly that Φ is built, evaluated as its own method,
and deliberately excluded from the final score — with the measured argument for
why. The second is defensible and arguably a better story; it is not defensible
to leave the documentation claiming the first.

**`docs/method.md` §9 item 4 stays open by decision, not omission.** Whether
CPAS-MLP as `q` beats raw reference similarity is untested because we chose not
to test it, not because nobody got to it.

**Priority.** `docs/method.md` §9 item 1 — training the attribute head on the
full train split, at 7× amplification — outranks everything in this document.
This work is correctness; that one is the remaining performance lever.
