# Compositional image retrieval on CelebA — the full method

**Frozen CLIP ViT-B/32, a learned attribute predictor, and retrieval scored
directly against the assignment's ground-truth criterion.**

This is the single reference for the current pipeline. **Anything not described
in this document is not part of the method.**

It describes only what is current. How the method was arrived at, and what was
tried and dropped — prompt arithmetic, the attribute-caption method, the
transformer combiner, the SCAC proposal — are in `docs/method-history.md`. Read
that one for *why* the design is what it is; this one for *what* it is.

| | |
|---|---|
| Task | Given a reference image and attribute constraints `T+` / `T−`, retrieve images that preserve the reference's identity, have every attribute in `T+` and none in `T−` |
| Dataset | CelebA, 40 binary attributes; the test split is the fixed retrieval database (19,962 images) |
| Encoder | Frozen CLIP ViT-B/32, d = 512, features pre-computed once |
| Benchmark | 14 fixed queries, `celeba_evaluation.json`, R@{1,5,10} and P@{1,5,10} |
| Trained parts | one attribute head (~1.1 M parameters) on frozen features; optionally the CPAS-MLP combiner; 2 scalars tuned on validation |

---

## 1. The criterion the task is actually defined by

Everything below follows from one fact, so it comes first.

**Assignment §3.1.1** defines a retrieved image as correct if and only if:

1. it strictly satisfies the query's positive / negative constraints, **and**
2. its remaining attributes are within **Hamming distance 2** of the
   reference's.

Both conditions are statements about 40-bit **attribute codes** — not about
embedding geometry. Verified against `celeba_evaluation.json` by exact set
reconstruction on **all 33,052 (query, reference) pairs**. (The benchmark also
only includes references with ≥ 5 valid targets, also per §3.1.1; the observed
minimum ground-truth set size is exactly 5.)

Three consequences drive the design:

- **The composition is not a learning problem.** The optimal target code is the
  reference's code with the queried bits flipped. §3.1.1 defines it that way.
  There is nothing for a fusion module to discover about *what* to aim at.
- **Retrieval quality is bounded by attribute-prediction accuracy.** With true
  labels the benchmark is solved exactly (R@10 = 1.000, measured). Every point
  of the remaining gap is code error.
- **Ranking by cosine similarity is a proxy, and a weak one.** CLIP similarity
  and attribute-code proximity are different orderings. Measured: the cosine
  top-10 differs from the reference on 4–10 attributes while the correct answers
  differ on 0–2, and those correct answers sit at cosine ranks in the thousands.

The task is nearest-neighbour retrieval in attribute space. The method scores
that criterion directly.

---

## 2. Pipeline

```
CelebA image ──► [1] frozen CLIP ──► features (512-d, L2-normalized)
                                          │
                                          ├──► [2] attribute head ──► p(d) ∈ [0,1]^40
                                          │                           for every image
                                          │                                 │
      T+ / T− ─────────────────────────────────────────────► [3] target code
                                          │                                 │
                                          └──► [4] optional combiner ──► q  │
                                                                   │        │
                                            [5] score = −E[Hamming] − λ·violation + w·(q·d)
                                                                            │
                                                                            ▼
                                                                     ranked database
```

1. **Features** — encode once, cache to disk. Frozen throughout.
2. **Attribute head** — predict all 40 attribute probabilities per image.
3. **Target code** — the reference's predicted code with queried bits forced.
4. **Combiner** (optional) — CPAS-MLP, producing a composite query embedding.
5. **Score** — expected Hamming on non-queried attributes, plus a constraint
   penalty, plus an optional cosine term.

Setting `w = 0` drops the combiner; setting the Hamming term aside and `w = 1`
recovers the old cosine pipeline exactly. Both are reported.

---

## 3. Features

`src/features.py`. CLIP ViT-B/32 image encoder, L2-normalized, cached as
`features/clip-vit-base-patch32_{split}.pt`. Nothing here is trained. Test split
= retrieval database; train split = training pool; valid split = the selection
surface for the attribute head and its thresholds.

A feature refit invalidates every number in `results/`.

---

## 4. Attribute prediction — where the performance is

This is the component that matters, so it gets the most care.

### 4.1 Why accuracy, not AUC

The score consumes a thresholded 40-bit code, and correctness requires landing
inside a radius-2 ball across ~38 bits. Errors compound: at **0.909** per-bit
accuracy the expected code is ~3.5 bits wrong — already outside the ball. AUC
0.929 flatters this, because AUC measures ranking *within* an attribute.

**Measured exchange rate: +0.003 bit accuracy bought +0.022 R@10** — roughly
**7× amplification**. Small accuracy gains are worth real effort here.

### 4.2 Two predictors, both reported

| predictor | where | valid bit accuracy |
|---|---|---|
| linear probe (logistic regression) | `src/probes.py` | 0.9088 |
| **MLP head** (512 → 1024 → 1024 → 40, GELU, dropout) | `src/attribute_head.py` | **0.9119** |

The linear probe is kept because it also supplies the *edit directions* the
combiner needs, and because it is the within-run reference point.

Trained by `scripts/fit_attribute_head.py`: AdamW, cosine schedule, selection on
**held-out valid bit accuracy** (not loss — accuracy is what transfers to the
score). Widths 512–2048 and dropout 0.2–0.4 all land within 0.001 of each other,
so capacity is not the constraint; the open question is data (§9).

### 4.3 Thresholds

`tune_thresholds` fits a per-attribute decision threshold on the valid split.
0.5 is only optimal for a calibrated probe on a balanced attribute, and most
CelebA attributes are far from balanced. Thresholds are saved *with* the
weights — a code produced with different thresholds is a different code.

**Probability calibration was tried and does not help.** Per-attribute Platt
scaling fitted on valid moved NLL 0.2095 → 0.2068 and R@10 0.4609 → 0.4584.
The probes were already calibrated (fitted slopes 0.78–1.08). Calibration
changes probability *values*, not bit accuracy, and bit accuracy is what binds.

---

## 5. Scoring

`src/attribute_retrieval.py`.

```
score(d) = − Σ_{a ∉ query} P(d differs from the target code on a)   # §3.1.1 (2)
           − λ · [ d breaks a queried constraint ]                  # §3.1.1 (1)
           + w · (q · d)                                            # composite embedding
```

- **Expected Hamming, not thresholded Hamming.** Using probabilities keeps the
  predictor's uncertainty in the ranking and is smoother; an attribute at p=0.5
  contributes 0.5 rather than an arbitrary bit.
- **The constraint term is the exclusion re-rank** (`src/rerank.py`) with the
  weight raised. At large λ it is a hard filter, at small λ a soft preference.
  Which is better is a sweep, and on validation **λ = 4 beat λ = 100** — the
  soft version wins, because the filter is applied to *predicted* attributes and
  a hard filter propagates prediction errors irreversibly.
- **The cosine term** keeps a composite query embedding in the ranking, so the
  fusion module still contributes and the assignment's Φ requirement is met by a
  component that is actually in the score.

λ and w are swept on the held-out validation benchmark, never on the 14 test
queries.

---

## 6. The combiner (CPAS-MLP)

`src/cpas_mlp.py`, trained by `scripts/train_cpas.py`. Predicts a reference
weight γ, per-attribute step sizes α, and bounded direction bends Δ, then
composes

```
q = normalize( γ·v_ref + Σ_a s_a · α_a · normalize(ŵ_a + Δ_a) )
```

Trained contrastively (InfoNCE, τ = 0.05) on attribute-flip triplets mined from
train-split labels (`src/mining.py`), with three hard negatives per example:
a violation, an identity distractor, and the reference itself.

**Its role is now a tiebreak.** It is the best cosine-space method measured
(R@10 0.267 versus 0.210 for the fixed γ = 0.6 rule), but in the combined score
the cosine term is worth little — see §7. It is retained because §1 of the
assignment requires a fusion module Φ yielding a composite query embedding, and
because it is the honest upper bound for what cosine-space composition achieves.

**Known issue:** the mining target is selected by agreement on ten
identity-proxy attributes with a CLIP-similarity tiebreak — not the §3.1.1
Hamming rule. Any retrain should fix this first (§9).

---

## 7. Results

All numbers: per-query mean over the 14 benchmark queries, full test split,
identical aggregation. Scoring hyperparameters selected on validation.

| method | scoring | R@1 | R@5 | R@10 | P@10 |
|---|---|---|---|---|---|
| Prompt arithmetic (γ = 1) | cosine | 0.023 | 0.071 | 0.106 | — |
| Probe composition (γ = 0.6) | cosine | 0.051 | 0.144 | 0.210 | 0.033 |
| CPAS-MLP | cosine | 0.062 | 0.187 | 0.267 | 0.043 |
| CPAS-MLP + exclusion re-rank | cosine + hinge | 0.061 | 0.183 | 0.279 | 0.043 |
| **Attribute space, linear probe** | §5 | 0.116 | 0.336 | **0.465** | 0.088 |
| **Attribute space, MLP head** | §5 | — | — | **0.482** | — |
| *oracle attribute codes* | §5 | *1.000* | *1.000* | *1.000* | — |

The MLP-head row was measured with fixed hyperparameters before the sweep
existed; rerun `scripts/run_attribute_retrieval.py --head ...` to fill the row
properly.

**What is established:**

- **Scoring the criterion beats approximating it, by a wide margin.** 0.210 →
  0.465 for the same linear probes and no training whatsoever. Precision@10
  nearly triples (0.033 → 0.088), so it is not a recall-only artifact.
- **Attribute accuracy is the remaining lever**, with 7× amplification and a
  measured ceiling of 1.000 at perfect codes.
- **The soft constraint penalty beats the hard filter** (λ = 4 over λ = 100 on
  validation), because the filter acts on predicted attributes.

**What is not:**

- The MLP-head gain (+0.022) clears the 0.02 resolution limit by little, on one
  seed. It needs repetition.
- The cosine term contributes ~0.004 in the earlier fixed-weight run; the swept
  run selected w = 1 with a small margin. The combiner's real contribution to
  the final score is not yet resolved.

### 7.1 Error decomposition

Which half of the code hurts, measured by substituting true labels on one side:

| | R@10 |
|---|---|
| both codes predicted | 0.461 |
| perfect reference code, predicted database | 0.649 |
| predicted reference, perfect database codes | 0.763 |
| both perfect | 1.000 |

Database-side prediction error costs more than reference-side. Together they
account for the entire gap — nothing is lost to composition or ranking.

---

## 8. Evaluation protocol

The 14-query benchmark against the full test split.

| metric | what it says |
|---|---|
| **R@{1,5,10}, P@{1,5,10}** | the graded objective |
| **V@10** | fraction of returned top-10 breaking a constraint, from true labels |
| **neg_R@10** | mean R@10 over queries carrying a negation |
| **bit accuracy** | per-attribute correctness on held-out data — the leading indicator |

Rules that keep numbers comparable:

1. **Recompute baseline rows inside every run.** Absolute numbers shift with a
   probe or feature refit; only within-run deltas are comparable.
2. **Differences below 0.02 R@10 are unresolved.** Seed spread reaches 0.019.
3. **Tune on validation only.** `build_val_benchmark` constructs held-out ground
   truth with the *same* §3.1.1 rule as the test benchmark. It previously used a
   ten-attribute identity-proxy rule — a different task — which is why earlier
   validation gains did not transfer. Note the val benchmark reuses the 14 query
   *shapes*, so it is held-out data but not a held-out query distribution.

---

## 9. Open work

1. **Train the attribute head on the full train split.** All configurations
   plateau at ~0.911–0.912 on the 30k sample, which is 18% of the ~162k
   available. If the plateau is data-limited this moves; if not, frozen
   ViT-B/32 features are saturated and ~0.50 R@10 is the ceiling for this
   encoder. At 7× amplification this is the highest-value run available.
2. **Repeat the MLP-head result across seeds**, since it clears the resolution
   limit by only 0.002.
3. **Fix the mining target to the §3.1.1 rule** before any combiner retrain
   (§6). Training against a target definition the benchmark does not use has the
   same defect the val benchmark had.
4. **Resolve the cosine term's contribution** — whether CPAS-MLP as the `q` in
   §5 beats raw reference similarity is untested.
5. Per-attribute Hamming weighting: attributes differ in probe reliability, and
   the score currently weights all 38 equally.

**Removed, deliberately:** negation-aware mining — a forced negation fraction,
multiple mined violations, a separately weighted violation loss, and
correlated-pair sampling. It taught the combiner that negation is an exclusion;
the score now enforces that directly, and the combiner reaches the ranking only
through a small cosine term, so it was subsumed twice over. Measured at +0.02
R@10 for ~1 GPU-hour per seed before the scoring change. The code has been
deleted rather than left switched off; it is in git history if the framing ever
changes, and the measurement stands as a reported negative result.

---

## 10. Repository map

Every module is reachable from the current method and has a test file.

### `src/`

| module | role |
|---|---|
| `data.py` | paths, CelebA loading, annotations |
| `features.py` | CLIP wrapper and the feature cache |
| `attribute_head.py` | the MLP attribute predictor, threshold tuning, bit accuracy |
| `attribute_retrieval.py` | target codes, expected Hamming, the §5 score |
| `probes.py` | linear probes: both loaders, `compose_probe`, AUC/AP scoring |
| `rerank.py` | probe probabilities, the hinge penalty, `rank_with_exclusion` |
| `steering.py` | the composition formula, `pad_queries`, `Steerer`, `FixedRule` |
| `cpas_mlp.py` | the combiner: predicts (γ, α, Δ) |
| `mining.py` | attribute-flip triplet mining for the combiner |
| `training.py` | batching, InfoNCE, the epoch loop |
| `evaluation.py` | metrics, benchmark loops, the val benchmark, probe drift |
| `retrieval.py` | query parsing, prompt templates, plain cosine `rank` |

### `scripts/`

| script | produces |
|---|---|
| `extract_train_features.py` | the training pool (`--all` for the full split) |
| `fit_probes.py` | `results/probe_weights.pt` |
| `run_probe_accuracy.py` | `results/probe_accuracy.csv` |
| `fit_attribute_head.py` | `results/attribute_head.pt` (weights + thresholds) |
| `run_attribute_retrieval.py` | `results/attribute_retrieval*.csv` — the current method |
| `run_baseline.py` | `results/baseline_results.csv` (prompt arithmetic) |
| `run_probe_gamma_ablation.py` | `results/probe_gamma_ablation.csv` |
| `train_cpas.py` | a CPAS-MLP checkpoint |
| `run_cpas_ablation.py` | `results/cpas_ablation.csv` — cosine-space methods |
| `run_exclusion_rerank.py` | `results/exclusion_rerank*.csv` |

### `tests/` — 97 tests, model-free

One file per `src` module, ~20 s on CPU, no CelebA or CLIP weights needed. The
contracts most worth keeping: the val benchmark reproduces the §3.1.1 rule
exactly (sound *and* complete); an inactive `Rerank` ranks identically to plain
cosine; a violating candidate is ranked below a compliant one; expected Hamming
is exact for confident probabilities and 0.5 per bit at maximum uncertainty.

### `results/`

Live files are what the current method produces. `results/archive/` holds output
from the abandoned approaches in `docs/method-history.md`; nothing in the code
reads it. Two notes: `archive/cpas_model.pt` is a transformer checkpoint that
**will not load** as a `PerAttributeMLP`, and
`archive/cpas_ablation_transformer.csv` was moved out because
`run_cpas_ablation.py` writes to that filename.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Scoring the ground-truth rule reads as fitting the benchmark | §3.1.1 is the assignment's own definition of a correct answer and of identity preservation; implementing it is the task, and the composite embedding is retained in the score |
| λ too large turns retrieval into attribute classification on predicted labels | λ is swept and the whole curve reported; validation already prefers the soft λ = 4 |
| Bit accuracy measured on valid does not transfer to test | valid is held out from head training; the exchange rate to R@10 is measured, not assumed |
| Thresholds overfit the valid split | 40 scalars on ~20k images; refit them whenever the predictor changes and never on train |
| A predictor refit silently invalidates stored results | baselines are recomputed inside every run |
| Reporting seed noise as a gain | the 0.02 R@10 resolution limit, stated with every result |

---

## 12. Assignment compliance

§1 requires a fusion module Φ yielding a composite query embedding, and names
the objective as *"a dynamic similarity metric where the conditioning process
intelligently integrates multiple conditions, treating them as either positive
(additive) or negative (subtractive) constraints."*

Both are addressed, and the split is deliberate:

- **Φ** is CPAS-MLP, producing the composite query embedding, present in the
  score through the cosine term and reported standalone throughout §7.
- **The dynamic similarity metric** is §5: positive and negative constraints
  enter through different terms with different signs, and the identity
  requirement enters as a distance the assignment itself defines. §3.1 asks
  precisely that we *"define how these positive and negative constraints
  interact in the embedding space"* — §5 is that definition, made explicit
  rather than left implicit in a vector sum.

Every result is reported with and without each component, so the contribution of
the fusion module and of the similarity metric can be read separately.
