# Compositional image retrieval on CelebA — the full method

**Frozen CLIP ViT-B/32, linear attribute probes, a learned combiner, and a
non-compensatory scoring rule.**

This is the single reference for the current pipeline — it folds in the separate
CPAS, exclusion-rerank and negation-mining proposals. **Anything not described
in this document is not part of the method.**

It describes only what is current. How the method was arrived at, and the
approaches that were tried and dropped along the way — prompt arithmetic, the
attribute-caption method, the transformer combiner, the SCAC proposal — are in
`docs/method-history.md`. Read that one for *why* the design is what it is; read
this one for *what* it is.

| | |
|---|---|
| Task | Given a reference image and attribute constraints `T+` / `T−`, retrieve images that keep the reference's identity, have every attribute in `T+` and none in `T−` |
| Dataset | CelebA, 40 binary attributes; the test split is the fixed retrieval database (19,962 images) |
| Encoder | Frozen CLIP ViT-B/32, d = 512, features pre-computed once |
| Benchmark | 14 fixed queries with provided ground truth (`celeba_evaluation.json`), R@{1,5,10} and P@{1,5,10} |
| Trained parts | 40 linear probes; one combiner MLP (~500k parameters); 2 scalars for the re-rank, tuned not learned |

---

## 1. Pipeline

Five stages. Each one is independently testable, and each later stage can be
switched off to recover the earlier one exactly.

```
CelebA image ──► [1] frozen CLIP ──► v_ref (512-d, L2-normalized)
                                        │
CelebA labels ──► [2] linear probes ──► ŵ_a  directions  ├──► [3/4] combiner ──► q
                                    └─► w_a, b_a  raw    │
                                                         ▼
                                    [5] score(d) = q·d − hinge penalty(d)
                                                         │
                                                         ▼
                                                   ranked database
```

1. **Features** — encode the database once, cache to disk.
2. **Probes** — one logistic regression per attribute on frozen features, giving
   both an *edit direction* and a *classifier*.
3. **Composition** — the fixed rule: a γ-weighted sum of the reference and the
   signed attribute directions.
4. **CPAS-MLP** — a learned combiner that replaces the fixed rule's constants
   with per-query predictions. Trained contrastively on mined triplets.
5. **Exclusion re-rank** — a hinge penalty on constraint violations, applied to
   the score rather than to the query vector.

Stages 3 and 4 are alternatives; 5 stacks on either.

---

## 2. Features

`src/features.py`. CLIP ViT-B/32 image encoder, L2-normalized outputs, cached as
`features/clip-vit-base-patch32_{split}.pt`. Nothing here is trained. The test
split is the retrieval database; the train split is the mining pool; the valid
split exists for probe scoring.

Everything downstream reads the cache, so a feature refit invalidates every
number in `results/`.

---

## 3. Attribute probes

`src/probes.py`, fit by `scripts/fit_probes.py`, scored by
`scripts/run_probe_accuracy.py`.

One logistic regression per attribute, full-batch Adam, 2000 steps, lr 0.05.
**These defaults are the recipe behind `results/probe_weights.pt` and every
number downstream of it; changing them silently invalidates the benchmarks.**

The probes serve two different roles, and the distinction causes a specific bug
if missed:

| use | tensor | loaded by |
|---|---|---|
| edit **direction** in the composition | `w_a / ‖w_a‖` | `load_probes` |
| **classifier** `p_a(d) = σ(w_a·d + b_a)` for the re-rank | raw `w_a`, `b_a` | `load_raw_probes` |

The saved biases belong to the *unnormalized* weights. Pairing them with the
normalized directions produces plausible-looking but meaningless probabilities.
`scripts/run_exclusion_rerank.py` asserts a known attribute reproduces its
reported AUC before using the probabilities at all.

**Quality** (`results/probe_accuracy.csv`): macro-mean valid AUC **0.929**, from
0.731 (Oval_Face) to 0.999 (Male); macro-mean AP 0.746. Every attribute in the
benchmark scores 0.909 AUC or better.

That number matters because it rules out an explanation: **the failures below
are not a probe-quality problem.** The directions are good; the way constraints
are combined is the problem.

---

## 4. Composition — the fixed rule

`compose_probe` in `src/probes.py`:

```
q = normalize( γ·v_ref + Σ_{a∈T+} ŵ_a − Σ_{a∈T−} ŵ_a )
```

γ trades identity preservation against the attribute edits. Swept by
`scripts/run_probe_gamma_ablation.py`; **γ = 0.6** is the operating point and the
baseline every later method is measured against.

Using probe directions rather than CLIP text embeddings is what makes this work:
the directions live in the visual embedding space where the database lives, so
there is no modality gap to cross. Prompt arithmetic (`src/retrieval.py`,
`scripts/run_baseline.py`) is retained only as the reference baseline row.

`FixedRule` in `src/steering.py` wraps this same formula behind the combiner
interface, so the fixed rule can be fed to any call site that takes a model.

---

## 5. CPAS-MLP — the learned combiner

`src/cpas_mlp.py`. The fixed rule uses one global γ and an implicit step size of
1 for every attribute and every reference. CPAS-MLP predicts those instead,
conditioned on the (reference, query) pair:

```
q = normalize( γ·v_ref + Σ_a s_a · α_a · normalize(ŵ_a + Δ_a) ),   s_a = ±1
```

- **γ** — per-query reference weight
- **α_a** — per-attribute step size (how far to move for *this* reference)
- **Δ_a** — bounded direction bend, low-rank, `‖Δ‖ ≤ delta_max` (default 0.3)

The attributes see each other through a pooled context vector, so a step can
depend on what else is being asked for. `--no-cross-attributes` zeroes that
pooling as an ablation.

The composition formula itself lives once in `src/steering.py::compose`, apart
from the model. Ablations only mean something if the variants differ solely in
how `(γ, α, Δ)` are produced.

### 5.1 Training data — mined triplets

`src/mining.py`. Training examples are synthesized from **train-split labels**;
the test split is never touched. Sample a reference, flip k of its attributes
(0→1 gives `T+`, 1→0 gives `T−`), then find a real image satisfying the flipped
constraints that still looks like the same kind of person — "same person" being
approximated by agreement on ten stable, non-editable identity-proxy attributes,
with CLIP similarity as the tie-break.

Each example carries three hard negatives, each aimed at one shortcut:

| negative | what it prevents |
|---|---|
| violation | satisfies `T+` but breaks a `T−` — *negation is a constraint, not a direction* |
| distractor | satisfies the constraints but is a different kind of person — *do not ignore the reference* |
| lazy (the reference itself) | *do not return it unchanged* |

Flip sets that fewer than `min_candidates = 20` images satisfy are rejected, so
training never sees combinations with no usable target.

### 5.2 Loss

`src/training.py`. InfoNCE at τ = 0.05: the query built from (reference, flips)
must rank its mined target above every other target in the batch and above its
own three mined negatives.

### 5.3 Checkpoint selection

`scripts/train_cpas.py`. Selection is on **val R@10**, not the mining proxy.
Every epoch the model is scored on a held-out benchmark built by
`build_val_benchmark`, which mirrors the real ground-truth rule (constraint
satisfaction + identity-proxy match) over a 10% held-out slice of the mining pool
and the same 14 query shapes. The old val-triplet recall@1 is kept only as a
diagnostic — it tracks the true metric poorly, which is why it is not the
selection signal.

Triplets are re-mined every epoch, so the model never sees the same synthetic
edit twice.

---

## 6. Negation-aware mining

Four changes to the training distribution and the loss. **No new parameters.**
All four are off by default, so the baseline recipe stays runnable from the same
script.

### 6.1 The problem

**The model is barely trained on negation.** Uniform flip sampling makes a flip a
negative constraint only if the reference *already has* the attribute. Mean
CelebA attribute prevalence is 0.226, so only ~23% of flips become negations and
~77% of trained edits are additions. Measured on the real pool: realized
negation share **0.195**.

**And the negation signal that exists is numerically drowned.** One violation per
triplet enters the same softmax as 1023 in-batch targets plus two other mined
negatives. At τ = 0.05 it contributes gradient only when it already ranks near
the top — the term meant to teach exclusion is roughly one thousandth of the
loss mass.

**Third, target selection pushes the other way.** The target is chosen for
maximal identity agreement with the reference, so the dominant learning pressure
is "stay near `v_ref`" — the same direction that trades exclusion away.

### 6.2 The changes

| flag | change | rationale |
|---|---|---|
| `--neg-fraction 0.5` | draw the number of negations from a binomial, sample them from the reference's ON set and the rest from its OFF set | force negations into the distribution; realized share 0.195 → **0.512** |
| `--n-violations 8` | mine the 8 closest near misses instead of 1 | one arbitrary near miss gives nothing to generalize from |
| `--lambda-violation 0.5` | pull violations out of the main softmax into `λ_v · L_violation`, where the target must outrank only its own violations | the signal was there but drowned; over 8 items instead of 1026 each violation carries real gradient |
| `--correlated-pair-prob 0.3` | with probability ρ, draw the flip set from attribute pairs with \|corr\| > 0.3, signs in tension (one added, one removed) | random pairs are easy because most attributes are near-independent; the failing queries pair correlated ones (Wearing_Lipstick/Heavy_Makeup correlate at **+0.80**) |

Two edge cases are handled explicitly, because getting them wrong biases the
pool silently:

- a reference with fewer ON attributes than the draw asks for gets **fewer
  negations**, not a rejection — otherwise rare-attribute references vanish from
  training;
- forced negations make constraint sets harder to satisfy, so **rejection rises**.
  The miner counts this and `train_cpas.py` prints it every mining round:

  ```
  mining: rejection 0.074 (too few candidates 0.069, no violation 0.005)  negation share 0.543
  ```

  Report both numbers. If rejection climbs past ~0.3 the effective training
  distribution is not the one intended.

### 6.3 Verified invariant

With all four flags off, the miner reproduces the pre-change implementation
**triplet for triplet, RNG draw for RNG draw** (checked over 300 triplets at
k ∈ {1,2,3}), and `run_epoch` computes the identical single-softmax loss. The
baseline is therefore a true reference point, not an approximation of one.

---

## 7. Exclusion re-rank

### 7.1 The problem

**A single query vector scores every candidate with one linear functional, and a
linear functional is compensatory.** `q·d` is a weighted sum of attribute
evidence, so a large surplus on one attribute pays for a violation on another.
The ground truth is conjunctive: a target must satisfy *every* constraint.

Negation is the acute case. It enters the composition only as `−α_a·ŵ_a`, a
subtraction whose single magnitude does two jobs: raising `α_a` to make the
exclusion bite also dilutes `v_ref` and the positive terms, because the sum is
renormalized. One knob, two objectives.

This is a property of the **output form**, not of the trunk that predicts
(γ, α, Δ). No architecture change removes it — which is why this changes the
scoring function instead.

**Evidence** (fixed rule, per-query, `results/exclusion_rerank_per_query.csv`):
`corr(R@10, number of negated attributes) = −0.54` across the 13 distinct
benchmark queries. The worst are `-Male, -Mustache` (0.000),
`+Chubby, -Young` (0.027) and `+Wearing_Lipstick, -Heavy_Makeup, +Smiling`
(0.059), against 0.39–0.43 for well-populated positive queries.

### 7.2 The method

`src/rerank.py`. Keep `q` and its cosine term; add a penalty that is **not**
compensatory:

```
s(d) = q·d
       − λ⁻ · Σ_{a∈T−} relu( p_a(d) − τ_a )      # forbidden attribute present
       − λ⁺ · Σ_{a∈T+} relu( τ_a − p_a(d) )      # required attribute absent
```

Three properties matter:

- **The hinge makes it a constraint, not a discount.** Below the threshold the
  penalty is exactly zero, so a compliant candidate is never charged. Above it,
  the cost cannot be bought back by a better cosine elsewhere.
- **Probabilities, not logits.** Raw logits have per-attribute scale, so a shared
  λ would silently weight attributes by their logit magnitude. Squashing to [0,1]
  makes one λ meaningful across all 40.
- **It is nearly free.** `P = σ(features @ w.T + b)` is one (19962, 40) matrix,
  3.2 MB, computed once. Scoring a query is a gather and a hinge.

`λ⁻ = λ⁺ = 0` reproduces plain cosine ranking bit-for-bit. That default-off
guarantee is what lets the ablation attribute any change to the re-rank alone.

### 7.3 Fitting λ

No gradient training. Swept on the **held-out validation benchmark**, never on
the 14 test queries: λ⁻ first with λ⁺ = 0, then λ⁺ at the winning λ⁻, thresholds
fixed at τ = 0.5.

The curve **plateaus rather than collapsing** — val R@10 rises 0.3815 → 0.4068 by
λ⁻ = 4 and is then flat out to λ⁻ = 32. The grid was extended past the original
{0…4} precisely because a monotone curve means the sweep stopped too early. Best
on validation: **λ⁻ = 4, λ⁺ = 2** (val R@10 0.4222).

Per-attribute thresholds are supported (`thresholds=`) but untested; τ = 0.5
shared is the current setting.

---

## 8. Evaluation protocol

The 14-query benchmark against the full test split. Four metric families:

| metric | what it says |
|---|---|
| **R@{1,5,10}, P@{1,5,10}** | the graded objective |
| **V@10** — violation rate | fraction of returned top-10 images breaking at least one constraint of their query, from CelebA labels. The direct target of the re-rank. |
| **neg_R@10** | mean R@10 over the queries carrying a negation. The overall mean is diluted by the 6 positive-only queries the negation work is not meant to help. |
| **probe drift** | mean probe-logit shift from reference to query, signed for queried attributes and absolute for the rest. Leakage into unmentioned attributes is what non-orthogonal directions cause. |

Three rules that make the numbers comparable:

1. **Recompute the fixed-rule row inside every run.** Absolute numbers shift with
   a probe refit; only deltas within a run are comparable.
2. **Resolution limit: differences below 0.02 R@10 are unresolved.** Seed-to-seed
   spread reaches 0.019. Do not report a 0.01 gain as a result.
3. **V@10 is the mechanism check.** If R@10 rises but V@10 does not fall, the gain
   is not coming from exclusion and the mechanism claim is unsupported.

The validation benchmark (`build_val_benchmark`) is built from held-out
references and never touches the 14 test queries or their ground truth. Caveat
to state when reporting: it reuses the same 14 *query shapes*, so it is held-out
data but not a held-out query distribution.

---

## 9. Current results

### 9.1 Combiners, cosine scoring, 14 queries

| method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Prompt arithmetic (γ = 1) | 0.023 | 0.071 | 0.106 |
| Probe composition (γ = 0.6) | 0.051 | 0.144 | 0.210 |
| **CPAS-MLP** | 0.066 | 0.182 | **0.267** |

### 9.2 Exclusion re-rank — the 2×2 and the ablations

One seed, λ tuned on validation per combiner.

| combiner | variant | R@1 | R@5 | R@10 | V@10 |
|---|---|---|---|---|---|
| probe rule γ=0.6 | 0 off | 0.052 | 0.140 | 0.206 | 0.351 |
| probe rule γ=0.6 | 1 negative only | 0.052 | 0.143 | 0.209 | 0.266 |
| probe rule γ=0.6 | 2 positive only | 0.056 | 0.155 | 0.216 | 0.258 |
| probe rule γ=0.6 | **3 both (tuned)** | 0.056 | 0.158 | **0.216** | **0.171** |
| probe rule γ=0.6 | 4 linear, no hinge | 0.028 | 0.088 | 0.127 | 0.049 |
| probe rule γ=0.6 | 5 hinge on top-200 | 0.056 | 0.158 | 0.216 | 0.172 |
| CPAS-MLP | 0 off | 0.062 | 0.187 | 0.267 | 0.401 |
| CPAS-MLP | 1 negative only | 0.062 | 0.185 | 0.270 | 0.336 |
| CPAS-MLP | 2 positive only | 0.061 | 0.185 | 0.265 | 0.331 |
| CPAS-MLP | **3 both (tuned)** | 0.061 | 0.183 | **0.279** | **0.260** |
| CPAS-MLP | 4 linear, no hinge | 0.036 | 0.126 | 0.187 | 0.081 |
| CPAS-MLP | 5 hinge on top-200 | 0.061 | 0.183 | 0.277 | 0.263 |

**What is established:**

- **The mechanism works on both combiners.** V@10 falls 0.351 → 0.171 on the
  fixed rule and 0.401 → 0.260 on CPAS-MLP. The penalty does what it is designed
  to do, and it does it regardless of which combiner produced `q`, which is what
  "orthogonal to the combiner" predicts.
- **Row 4 is the decisive ablation and it confirms the §7.1 argument.** A linear
  penalty — still compensatory, expressible by moving `q` itself — buys
  compliance by destroying retrieval: R@10 collapses to 0.127 / 0.187 while V@10
  goes to 0.049 / 0.081. The non-compensatory hinge is the active ingredient, not
  the extra probe signal.
- **Row 5 matches row 3 to three decimals.** The cheap two-stage version
  (penalty on the top-200 by cosine only) is sufficient.

**What is not established:**

- The R@10 gains (+0.009 fixed rule, +0.011 CPAS-MLP) are **below the 0.02
  resolution limit**. They are the right sign on both combiners but cannot be
  claimed from one seed.
- On CPAS-MLP, R@1 and R@5 move slightly *down* while R@10 moves up. A real
  effect does not usually split that way; treat +0.011 as suggestive.

**One finding worth carrying forward:** CPAS-MLP has a *higher* violation rate
than the fixed rule (0.401 vs 0.351) despite far better R@10. The learned
combiner buys retrieval quality partly by being sloppier about constraints —
direct support for the §6 premise that the training distribution does not teach
exclusion, and the clearest link between the two extensions.

### 9.3 Negation-aware mining

**Implemented and tested; not yet run.** Requires a full retrain (~1 GPU-hour
per seed). The mining-distribution effect is verified (negation share
0.195 → 0.512, rejection 0.074), but no retrained checkpoint has been
benchmarked. The ablation ladder in §10 is the outstanding experiment.

---

## 10. Ablations

**Exclusion re-rank** — `scripts/run_exclusion_rerank.py`, all six rows in one
run, no training. Results in §9.2.

**Negation mining** — `scripts/train_cpas.py`, one flag added per row, then
`scripts/run_cpas_ablation.py`. Each row adds to the row above, so the deltas
attribute the gain:

| # | Variant | Isolates |
|---|---|---|
| 0 | current mining and loss | reference point |
| 1 | `+ --neg-fraction 0.5` | does simply seeing more negations help? |
| 2 | `+ --n-violations 8` | does a richer near-miss set help? |
| 3 | `+ --lambda-violation 0.5` | was the signal there but drowned? |
| 4 | `+ --correlated-pair-prob 0.3` | does training on the hard cases transfer? |

Row 3 is the cheapest test of the hypothesis that the training *signal*, not the
architecture, is the limiting factor: it changes no data at all, only how
already-mined violations enter the loss.

**CPAS-MLP architecture** — `--delta-max 0` (rescaling only, no bend),
`--no-cross-attributes` (attributes cannot condition on each other), `--rank`.

**Measure the two extensions separately before combining them.** If both are
applied at once and the number moves, neither is attributable — and they could
be redundant, since a model trained to respect exclusion may leave nothing for a
re-rank to fix. `run_cpas_ablation.py` deliberately scores with plain cosine;
`run_exclusion_rerank.py --checkpoint <negation-mined model>` is the combination
cell, and it reports that model's own re-rank-off baseline alongside it.

---

## 11. Repository map

Every module below is reachable from the current method and has a test file.

### `src/`

| module | role |
|---|---|
| `data.py` | paths, CelebA loading, annotations |
| `features.py` | CLIP wrapper and the feature cache |
| `probes.py` | probe fitting, both loaders, `compose_probe`, AUC/AP scoring |
| `steering.py` | the composition formula (once), `pad_queries`, the `Steerer` protocol, `FixedRule` |
| `cpas_mlp.py` | the combiner: predicts (γ, α, Δ) |
| `mining.py` | triplet mining, negation controls, correlated-pair table, mining stats |
| `training.py` | batching, InfoNCE, the separate violation loss, the epoch loop |
| `rerank.py` | probe probabilities, the hinge penalty, `rank_with_exclusion` |
| `evaluation.py` | metrics, the three benchmark loops, the val benchmark, probe drift |
| `retrieval.py` | query parsing, prompt templates, plain cosine `rank` — the prompt-arithmetic baseline only |

### `scripts/`

| script | produces |
|---|---|
| `extract_train_features.py` | the mining pool (`--all` for the full train split) |
| `fit_probes.py` | `results/probe_weights.pt` |
| `run_probe_accuracy.py` | `results/probe_accuracy.csv` |
| `run_baseline.py` | `results/baseline_results.csv` (prompt arithmetic) |
| `run_probe_gamma_ablation.py` | `results/probe_gamma_ablation.csv` — picks γ = 0.6 |
| `train_cpas.py` | a CPAS-MLP checkpoint; all negation-mining flags live here |
| `run_cpas_ablation.py` | `results/cpas_ablation.csv` — combiners under cosine scoring, with V@10, neg_R@10 and probe drift |
| `run_exclusion_rerank.py` | `results/exclusion_rerank{,_sweep,_per_query}.csv` — the λ sweep, the 2×2 and the ablations |
| `smoke_test.py` | a 100-image end-to-end sanity run; development only |

### `tests/` — 95 tests, model-free

One file per `src` module. They run in ~30 s on CPU without CelebA or the CLIP
weights, using toy tensors. The contracts they pin that are easy to break:

- the default mining path and the default loss reproduce the pre-extension
  behavior exactly;
- an inactive `Rerank` gives ranking identical to plain cosine;
- a compliant candidate is never charged by the hinge, and a linear penalty
  *does* charge it;
- V@10 columns appear only when labels are passed;
- a reference with too few ON attributes degrades gracefully rather than raising
  or silently emitting an all-positive query.

### `results/`

Live files are the ones the current method produces. `results/archive/` holds
output from the abandoned approaches described in `docs/method-history.md` —
zero-shot prompt classification, the caption method, the transformer combiner.
Nothing in the code reads them; they are kept so past numbers are not lost, and
`method-history.md` cites them. Two are worth knowing about:

- `archive/cpas_model.pt` is a transformer checkpoint that **will not load** as a
  `PerAttributeMLP`.
- `archive/cpas_ablation_transformer.csv` is the δ_max ablation behind
  `method-history.md` §4. It was moved out of `results/` because
  `run_cpas_ablation.py` writes to that filename and would overwrite it.

---

## 12. Known gaps and next steps

Ordered by what would change a conclusion.

1. **Seeds.** Every re-rank number is one seed. The re-rank is deterministic
   given a combiner, so the noise is inherited from the combiner's training
   seed. Three seeds are needed to resolve the +0.01 R@10 movements.
2. **Run the negation-mining ablation ladder** (§10). It is implemented, tested
   and unrun.
3. **The combination cell** — negation-mined model *plus* re-rank — after the two
   are measured separately.
4. **Why the validation gain does not transfer.** Val R@10 rose +0.041 with the
   re-rank; test R@10 moved +0.009. Both benchmarks use the same ground-truth
   rule, so the gap is unexplained and worth understanding before trusting val
   for anything but λ selection.
5. **Per-attribute thresholds** for the hinge. Supported in code, never swept;
   τ = 0.5 shared is arbitrary.
6. **Benchmark size.** 14 queries, 13 distinct, 7 with a negation, one with only
   27 source images and R@10 = 0 everywhere. Several conclusions are limited by
   this rather than by the methods. Enlarging it with held-out queries of
   controlled k would raise the resolution of everything above.

Minor cleanups, none blocking: `compose_probe` and `FixedRule` express the same
formula in two places; `smoke_test.py` predates the current pipeline and only
exercises the prompt baseline.

---

## 13. Risks

| Risk | Mitigation |
|---|---|
| λ large enough to dominate the cosine turns retrieval into probe classification | sweep λ and report the whole curve; the signature is V@10 falling while R@10 also falls — visible in ablation row 4 |
| The penalty helps only because probes add information, not because of the hinge | ablation row 4 detects exactly this, and does |
| Raw-vs-normalized probe weights confusion | the run asserts a known attribute's AUC before using the probabilities |
| Tuning λ on validation looks like fitting the benchmark | the val benchmark is built from held-out references and never touches the 14 queries; state this, and state that the query shapes are shared |
| Forcing negations shifts the pool toward frequent attributes | the miner logs rejection rate and realized negation share every round; report both |
| `λ_v` too high optimizes violation ranking at the expense of retrieval | sweep it and report the curve |
| Correlated-pair sampling helps the hard queries but looks worse on the mean | this is why neg_R@10 is reported separately |
| Reporting a gain that is really seed noise | the 0.02 R@10 resolution limit, stated with every result |

---

## 14. Assignment compliance

The assignment asks for a fusion module Φ yielding a composite query embedding,
and separately states that *"the retrieval function must score images highest if
they share the latent identity of the reference, explicitly contain glasses, and
explicitly do not contain red hair... You must define how these positive and
negative constraints interact in the embedding space,"* with the stated
objective being a *dynamic similarity metric*.

The composite embedding remains the primary retrieval mechanism: CPAS-MLP is Φ,
and it is evaluated on its own throughout. The exclusion penalty is presented as
part of the similarity metric, and **every number is reported with and without
it**. That turns a possible objection into an ablation, which the assignment
grades under methodological thoroughness.
