# Weighted attribute-space query — design

**Give the attribute-space query real-valued weights instead of ±1, without
training anything.**

Status: design approved, not implemented.
Scope: rungs 1 and 2 of the proposal. Rung 3 (a learned module) is explicitly
deferred — see §7.

---

## 1. Why

`docs/method.md` §5 ranks by

```
score(d) = − Σ_{a ∉ query} [ p_a(1−r_a) + (1−p_a)r_a ]   − λ·[violation]   + w·(q·d)
```

Expanding the first term and dropping what does not depend on the candidate:

```
E[Hamming] = Σ_a p_a(1−r_a) + (1−p_a)r_a = Σ_a p_a(1 − 2r_a) + Σ_a r_a
```

so, with `s_a = 2r_a − 1`,

```
rank by −E[Hamming]  ≡  rank by ⟨s, p(d)⟩
```

**The method already ranks by an inner product against a 40-dimensional query
vector.** Verified numerically: max deviation 6e-08.

Two facts follow, and they are the whole design:

- `r_a` is currently the reference's *thresholded* bit, so every `s_a` is exactly
  ±1. An attribute the predictor is guessing at (p = 0.51) shouts as loudly as
  one it is certain about (p = 0.99). Worse, two coin flips landing either side
  of the threshold (reference 0.49, candidate 0.52) are recorded as a **full
  disagreement** — half the Hamming budget of 2, spent on noise.
- Nothing weights attributes by how well the predictor actually handles them.
  Probe AUC ranges from 0.731 (`Oval_Face`) to 0.999 (`Male`).

Both are fixed by generalising the term that already exists. Neither requires
training: the assignment (§3) explicitly permits training-free strategies.

## 2. The generalised score

```
score(d) = − Σ_{a ∉ query}  w_a · [ p_a(1−r_a) + (1−p_a)r_a ]   − λ·[violation]   + w_cos·(q·d)
```

Two independent knobs, each defaulting to today's behaviour:

| knob | today | change |
|---|---|---|
| `r_a` — the reference bit | thresholded, 0 or 1 | **rung 1**: the predicted probability |
| `w_a` — the attribute weight | implicitly 1 | **rung 2**: a measured reliability |

`λ` and the cosine term are **untouched**. CPAS-MLP and the constraint penalty
keep working exactly as they do now.

Three properties were verified numerically before the design was fixed:

| property | result |
|---|---|
| `weights = None` reproduces the current term | exact (0.0) |
| soft reference fed hard 0/1 probabilities reproduces the thresholded code | exact |
| `target_code` accepts float input and forces queried bits to 0.0 / 1.0 | yes |

So the current method is a **corner of the new space**, not a neighbour of it.
Any difference is attributable to the knobs.

**The database-side code stays hard.** `constraint_violation` needs a yes/no
decision, so only the Hamming term goes soft.

## 3. The reliability weight

### 3.1 Why not accuracy

Per-attribute accuracy is the obvious choice and is wrong on CelebA, because the
attributes are heavily imbalanced. `Wearing_Necklace` has a 12% positive rate, so
a predictor that always answers "no" scores **88% accuracy while detecting
nothing**. Accuracy would hand a large weight to an attribute the predictor is
blind to — the opposite of the intent.

### 3.2 Youden's J

```
J_a = sensitivity_a + specificity_a − 1
```

- a trivial predictor (always one class) → **J = 0**
- a perfect predictor → **J = 1**

The zero sits where it should: an attribute predicted at chance contributes
nothing to the ranking.

**Negative J is clamped to 0.** A negative weight would invert the target bit —
the score would actively ask for the opposite of what the reference has. On
held-out data a negative J is almost certainly noise, not signal.

**Weights are normalised to mean 1** over the 40 attributes. `λ = 4` was selected
on validation under unit weights; without normalisation the Hamming term changes
scale and `λ` is silently re-tuned. The sweep would re-select it anyway, but
normalisation keeps the rows comparable.

### 3.3 Where it is measured

On the held-out validation slice `run_attribute_retrieval.py` already builds:
the last 10% of the mining pool under a fixed permutation (`VAL_FRACTION = 0.1`).
Its absolute size follows whichever pool `resolve_pool` finds — the 30k sample or
the full train split — so the design does not depend on a particular count.
Never measured on train, never on the 14 test queries.

## 4. Code changes

| file | change |
|---|---|
| `src/attribute_retrieval.py` | optional `weights: torch.Tensor \| None` on `expected_hamming` and `attribute_scores`. Full length `(A,)`, indexed by the same `rows` the term already sums over. `None` reproduces today exactly |
| `src/attribute_head.py` | `attribute_reliability(pred, labels) -> (A,)` returning raw J, and `reliability_weights(pred, labels) -> (A,)` adding the clamp and the normalisation. Both take **thresholded boolean** predictions, so they serve the linear probe and the MLP head alike |
| `scripts/run_attribute_retrieval.py` | `--soft-reference` and `--reliability-weights` flags; weights computed on the val slice; both passed **identically** to the validation sweep and the test benchmark |
| `tests/test_attribute_retrieval.py` | all nine contracts of §6. `attribute_head.py` is already tested from this file (`test_bit_accuracy_and_thresholds` and below), so the reliability tests join them rather than opening a new module |

Rung 1 needs **no change under `src/`**: it is `target_code(probs[src], …)`
instead of `target_code(code[src], …)` at the call site.

The weighted form is one extra elementwise multiply, keeping both matrix
products:

```
pw = p * w ;  pw @ (1−r)ᵀ + (w − pw) @ rᵀ
```

Verified equal to the explicit weighted sum (max deviation 6e-08).

Two small functions rather than one: the raw per-attribute J is a number the
report wants on its own ("which attributes is the predictor effectively blind
to?"), and separating the measurement from its use keeps each testable.

## 5. Evaluation

### 5.1 Ablation grid

Two independent knobs, four configurations, run for both predictors (linear
probe and MLP head), so eight rows:

| reference | weights | what it is |
|---|---|---|
| hard | unit | the current method, recomputed in the same run |
| soft | unit | rung 1 alone |
| hard | reliability | rung 2 alone |
| soft | reliability | both |

The two middle rows are the point: they attribute the gain to a specific knob
instead of leaving an aggregate nobody can decompose.

### 5.2 Protocol

`λ` and `w_cos` are swept on validation **per configuration**, never reused
across configurations. Tuning against one configuration and reporting another
selects hyperparameters for a model that was not measured — the same defect the
`--checkpoint` commit was careful to avoid.

Cost: the sweep goes from 32 to 128 validation evaluations. Minutes, not hours.

### 5.3 Reading the numbers

The repo's *"differences below 0.02 R@10 are unresolved"* rule comes from
seed spread in **training**. Nothing here is trained, so given fixed features
these rows are **deterministic** and small differences are real.

The one residual source of variance is the draw of the 200 validation
references, which decides which `(λ, w_cos)` is selected. Re-run with 2–3 seeds
to confirm the *selection* is stable — not to average the results.

## 6. Test contracts

Model-free, no CelebA and no CLIP weights, matching the existing suite.

**Weighted Hamming**

1. `weights=None` reproduces the current behaviour bit for bit
2. all-ones weights likewise
3. a zero weight removes that attribute's influence entirely — perturbing its
   probabilities arbitrarily leaves the score unchanged
4. doubling a weight doubles that attribute's contribution

**Soft reference**

5. fed hard 0/1 probabilities, reproduces the thresholded code exactly (it is a
   strict generalisation, not a different method)
6. a reference at p = 0.5 on an attribute makes that attribute unable to change
   the ordering of candidates

**Reliability**

7. a perfect predictor gives J = 1
8. **a majority-class predictor on an imbalanced attribute gives J = 0** — the
   accuracy trap of §3.1, and the contract that matters most
9. returned weights have mean 1

## 7. Out of scope

- **Rung 3, a learned module** emitting the weights conditioned on the reference
  and the query. Deferred deliberately: the free rungs establish whether
  weighting has any value before anyone spends GPU time, and the assignment
  permits a training-free method. If rung 1 moves nothing, rung 3 is unlikely to.
- **Absorbing the constraint term into the query vector.** Attractive for
  compliance (the whole score becomes one inner product against one composite
  query embedding) but it would make the score compensatory, and the
  non-compensatory hinge is measured as useful (`λ = 4` beat `λ = 100`). Revisit
  as its own ablation once rungs 1–2 are measured.
- **Retraining anything.** The probes, the attribute head and CPAS-MLP are used
  as they are. The open retrains in `docs/method.md` §9 (attribute head on the
  full split; CPAS with the corrected miner) are independent of this work.

## 8. Risks

| risk | mitigation |
|---|---|
| Youden's J on validation is itself noisy for rare attributes | clamped at 0, and the per-attribute J is reported so a suspicious value is visible rather than silently weighted |
| the reliability weights overfit the validation slice | 40 scalars measured on held-out data; refit whenever the predictor changes, never on train |
| normalisation hides a scale interaction with `λ` | `λ` is re-swept per configuration, and the whole grid is reported |
| gains are too small to resolve | the rows are deterministic (§5.3), so the usual 0.02 caveat does not apply; the selection stability is checked across seeds |
| rung 1 helps and rung 2 does not, or vice versa | the grid separates them by construction |
