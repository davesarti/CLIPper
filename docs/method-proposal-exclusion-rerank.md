# Exclusion re-rank: making negative constraints binding

**Method proposal — compositional image retrieval on CelebA with frozen CLIP ViT-B/32.**

| | |
|---|---|
| Dataset | CelebA (40 binary attributes); test split as fixed retrieval database (19,962 images) |
| Encoder | Frozen CLIP ViT-B/32, d = 512; features pre-computed once |
| Trainable | 41 scalars (one λ per sign, 40 thresholds) — or 2 if thresholds are shared |
| Training required | **None.** Tuned on a held-out validation benchmark, not learned by gradient descent |
| Applies to | Any combiner that emits a query vector: the fixed probe rule, CPAS, CPAS-MLP |

## 0. Context for a reader starting here

The task: given a reference image `v_ref`, a set of positive attribute constraints
`T+` and negative constraints `T−`, retrieve test-split images that keep the
reference's identity, have every attribute in `T+`, and have none in `T−`.

Every method in this project so far builds a single query vector `q` and ranks
the database by cosine similarity `q·d`. The current best combiner is CPAS-MLP
(`src/cpas_mlp.py`), which predicts a reference weight `γ`, per-attribute step
sizes `α_a` and bounded direction bends `Δ_a`, then composes

```
q = normalize( γ·v_ref + Σ_a s_a · α_a · normalize(ŵ_a + Δ_a) ),   s_a = ±1
```

where `ŵ_a` are L2-normalized linear-probe directions fit on frozen CLIP
features. Evaluation is Recall@{1,5,10} and Precision@{1,5,10} over 14 fixed
benchmark queries with provided ground truth (`celeba_evaluation.json`).

## 1. The problem this fixes

**A single query vector scores every candidate with one linear functional, and a
linear functional is compensatory.** `q·d` is a weighted sum of attribute
evidence, so a large surplus on one attribute pays for a violation on another.
The ground truth is conjunctive: a target must satisfy *every* constraint.

Negation is the acute case. It enters the composition only as `−α_a·ŵ_a`, a
subtraction whose single scalar magnitude does two jobs at once. Raising `α_a`
to make the exclusion bite also dilutes `v_ref` and the positive terms, because
the whole sum is renormalized. There is one knob for two objectives.

This is a property of the **output form**, not of the trunk that predicts
`(γ, α, Δ)`. No architecture change to the combiner can remove it, which is why
this proposal changes the scoring function rather than the network.

**Evidence.** The per-query results (`results/cpas_results.csv`) put the
failures on negation-bearing and conjunction-heavy queries:
`+Wearing_Lipstick, -Heavy_Makeup, +Smiling` at R@10 = 0.059 and
`+Chubby, -Young` at 0.086, against 0.30–0.75 for well-populated positive
queries. Across the 14 queries, `corr(R@10, number of negated attributes)` is
−0.39.

**What this is not.** It is not a probe-quality problem. The probes are strong:
`results/probe_accuracy.csv` gives a macro-mean valid AUC of 0.929, and every
attribute appearing in the benchmark scores 0.909 or better. The directions are
fine; the way constraints are combined is not.

## 2. The method

Keep `q` and its cosine term. Add a per-candidate penalty that is **not**
compensatory:

```
s(d) = q·d
       − λ⁻ · Σ_{a ∈ T−} relu( p_a(d) − τ_a )        # forbidden attribute present
       − λ⁺ · Σ_{a ∈ T+} relu( τ_a − p_a(d) )        # required attribute absent
```

`p_a(d) = sigmoid(w_a · d + b_a)` is the probe's predicted probability that
image `d` has attribute `a`, using the **raw** (unnormalized) probe weights and
biases. `τ_a ∈ (0,1)` is a per-attribute threshold; `λ⁻, λ⁺ ≥ 0` weight the two
penalty types.

Three properties matter:

- **The hinge makes it a constraint, not a discount.** Below the threshold the
  penalty is exactly zero, so a compliant candidate is never charged. Above it,
  the cost grows and cannot be bought back by a better cosine elsewhere, which
  is precisely what the compensatory sum could not express.
- **Probabilities, not logits.** Raw probe logits have per-attribute scale, so a
  shared λ would silently weight attributes by their logit magnitude. Squashing
  to `[0,1]` makes one λ meaningful across all 40.
- **It is free at query time.** `P = sigmoid(features @ w.T + b)` is an
  `(N, 40)` matrix computed once for the whole database — 19,962 × 40 floats,
  about 3 MB. Scoring a query is then one gather and one hinge.

The positive term `λ⁺` is included because the conjunction problem is not
exclusive to negation: a candidate can also win on cosine while missing a
required attribute. Whether it earns its place is an ablation (§5), and `λ⁺ = 0`
is a valid outcome.

## 3. Implementation

### 3.1 New module: `src/rerank.py`

```python
def database_probe_probs(features, weights, biases) -> torch.Tensor:
    """(N, A) probability that each database image has each attribute.

    features: (N, D) L2-normalized image features.
    weights:  (A, D) RAW probe weights - not the normalized directions.
    biases:   (A,)
    """
```

```python
def exclusion_scores(
    query_vecs,        # (S, D) normalized query vectors from any combiner
    image_features,    # (N, D) normalized database features
    db_probs,          # (N, A) from database_probe_probs
    pos_rows,          # list[int]: attribute rows in T+
    neg_rows,          # list[int]: attribute rows in T-
    lam_neg=0.0,
    lam_pos=0.0,
    thresholds=None,   # (A,) or None for a shared 0.5
) -> torch.Tensor:     # (S, N) scores, higher is better
```

Then a `rank_with_exclusion` mirroring the contract of `rank` in
`src/retrieval.py` (same `exclude` semantics, returns indices sorted
descending), so the benchmark loop changes in one line.

**Gotcha that will cost an hour if missed.** `src.probes.load_probes` returns
directions **already L2-normalized**, while the saved biases belong to the
*unnormalized* weights. `p_a(d)` needs the raw `saved["weights"]` and
`saved["biases"]` from `probe_weights.pt`, not the normalized directions the
composition uses. Loading the wrong one produces plausible-looking but
meaningless probabilities. Assert that a known attribute reproduces its
`results/probe_accuracy.csv` AUC before going further.

### 3.2 Integration

`run_cpas_benchmark` and `run_probe_benchmark` in `src/evaluation.py` currently
call `rank(query_vecs, image_features, exclude=source_indices)`. Give both an
optional re-rank configuration; when it is absent, behavior must be
bit-identical to today. That default-off requirement is what lets the ablation
attribute any change to the re-rank alone.

### 3.3 Fitting λ and τ

There is no gradient training. Sweep on the **held-out validation benchmark**
built by `build_val_benchmark` / `score_val_benchmark` in `src/evaluation.py`,
never on the 14 test queries.

1. Fix `τ_a = 0.5` for all attributes. Sweep `λ⁻ ∈ {0, 0.1, 0.25, 0.5, 1, 2, 4}`
   with `λ⁺ = 0`. Record val R@10 for each.
2. At the best `λ⁻`, sweep `λ⁺` over the same grid.
3. Only then try per-attribute thresholds. A principled choice is the
   precision-calibrated threshold already implemented as
   `calibrate_threshold_precision` in `src/attributes.py`, converted to
   probability space. Keep it if it beats the shared 0.5 on validation.

Report the full λ sweep as a curve, not just the winner. A curve that rises then
collapses tells you the penalty is fighting the cosine term; a monotone curve
means you have not swept far enough.

## 4. Evaluation

The 14-query benchmark, R@{1,5,10} and P@{1,5,10}, against the full test split.
The comparison is a 2×2 — the re-rank is orthogonal to the combiner, so it must
be shown on both:

| combiner | re-rank off | re-rank on |
|---|---|---|
| Probe composition (γ = 0.6, fixed rule) | | |
| CPAS-MLP | | |

**Recompute the fixed-rule row inside the same run**, with the same probe file
as the model it is compared against. Absolute numbers in this project are not
comparable across probe refits; the comparable quantity is the delta within a
run.

Add one metric the standard ones do not capture, because R@10 can improve for
reasons unrelated to the mechanism:

> **Violation rate @10** — the fraction of returned top-10 images that break at
> least one constraint of their query, computed from CelebA labels. This is what
> the penalty directly targets. Report it alongside R@10 for every cell above.

If R@10 rises but violation rate @10 does not fall, the gain is not coming from
exclusion and the mechanism claim is unsupported.

**Resolution limit.** Seed-to-seed spread on this benchmark reaches 0.019 R@10
for some variants. Treat differences below **0.02 R@10 as unresolved**. The
re-rank itself is deterministic given a combiner, so the noise is inherited from
the combiner's training seed — sweep on validation, and report test numbers for
at least the seeds already available.

## 5. Ablations

| # | Variant | Isolates |
|---|---|---|
| 0 | No re-rank | reference point |
| 1 | `λ⁻ > 0`, `λ⁺ = 0` | negative constraints only |
| 2 | `λ⁺ > 0`, `λ⁻ = 0` | positive constraints only |
| 3 | Both | whether they compose |
| 4 | Best λ, linear penalty instead of hinge (`p_a(d)` with no `relu`/threshold) | **whether the non-compensatory hinge is the active ingredient, or just extra probe signal** |
| 5 | Hinge applied to the top-200 by cosine only | whether a cheap two-stage version suffices |

Row 4 is the one that decides whether the argument in §1 is right. A linear
penalty is still compensatory — it is just another term in a weighted sum, and
it is expressible by moving `q` itself. If row 4 matches row 3, the gain is not
about conjunction and the framing in this document is wrong.

## 6. Assignment compliance

The assignment (§3, "Expressive multimodal conditioning") states: *"the
retrieval function must score images highest if they share the latent identity
of the reference, explicitly contain glasses, and explicitly do not contain red
hair... You must define how these positive and negative constraints interact in
the embedding space."* Defining that interaction is the assigned task, and the
stated objective is a *dynamic similarity metric*, so changing the scoring
function is in scope.

One caveat to handle rather than ignore: §1 describes a fusion module Φ that
"yields a composite query embedding". Keep the composite embedding as the
primary retrieval mechanism, present the penalty as part of the similarity
metric, and report every number with and without it. That converts a possible
objection into an ablation, which §4 of the assignment grades under
methodological thoroughness.

## 7. Risks

| Risk | Mitigation |
|---|---|
| λ large enough to dominate the cosine turns retrieval into probe classification and discards identity | sweep λ and report the whole curve; the violation-rate metric will fall while R@10 also falls, which is the visible signature |
| The penalty only helps because probes add information, not because of the hinge | ablation row 4 (linear penalty) is designed to detect exactly this |
| Raw-vs-normalized probe weights confusion | assert reproduction of a known per-attribute AUC before using `db_probs` |
| Tuning λ on validation and reporting on test looks like fitting the benchmark | the val benchmark is built from held-out references and never touches the 14 queries or their ground truth; state this explicitly in the report |
| Only 6 of the 14 benchmark queries have more than one attribute; 8 involve a negation | the mechanism is exercised by part of the benchmark only; report per-query results, not only the mean |

## 8. Relationship to the negation-mining proposal

`docs/method-proposal-negation-mining.md` attacks the same failure from the
training side. The two are independent and must be measured independently
before being combined, or neither gain is attributable. This proposal requires
no retraining and should be run first for that reason.
