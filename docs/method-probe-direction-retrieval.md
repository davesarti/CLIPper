# Probe-Direction Composition: current retrieval method

**Method write-up — compositional image retrieval on CelebA with frozen CLIP ViT-B/32.**

| | |
|---|---|
| Dataset | CelebA (40 binary attributes); test split (19,962 images) as fixed retrieval database |
| Encoder | Frozen CLIP ViT-B/32, d = 512, features pre-computed once and cached |
| Trainable | 40 linear probes (40 × 512 weights + 40 biases ≈ 20k parameters) |
| Benchmark | 14 queries of the form `"+Smiling, -Blond_Hair"`, Recall@K / Precision@K |

## 0. Problem setup

Given a reference image embedding `v_ref`, positive attributes **T+** and negative
attributes **T−**, build a single query embedding `q` such that ranking the database
by cosine similarity to `q` returns images that resemble the reference but have every
T+ attribute and lack every T− attribute. Only the query side may change; the database
features are fixed.

## 1. Method history

Three methods were implemented, in order:

1. **Prompt arithmetic (baseline).** Attribute directions are CLIP *text* embeddings
   of hand-written prompts (`"a photo of a smiling person"`, see `PROMPTS` in
   `src/retrieval.py`): `q = normalize(v_ref + Σ t+ − Σ t−)`. Zero training. Its
   weakness is the modality gap: text embeddings live on a different cone of CLIP
   space than image embeddings, so they are poorly calibrated as edit directions.
2. **Attribute caption** (`docs/method-proposal-attribute-caption.md`). Predict the
   full 40-attribute state of the reference (zero-shot, prompt pairs + calibrated
   thresholds), flip the queried bits, render the *entire* state as one negation-free
   caption, and use its text embedding as `q`. This uses **all** attributes, not just
   the queried ones.
3. **Probe-direction composition (current).** Same arithmetic as the baseline, but
   the directions are learned linear-probe weight vectors — directions that live in
   the *image* embedding space — restricted to the queried attributes only, with a
   tuned reference weight γ. Described below.

## 2. Probe training

One independent logistic regression per attribute, on frozen CLIP features
(`fit_linear_probes` in `src/probes.py`):

- **Data**: L2-normalized image features of a 30k-image train-split sample
  (`features/clip-vit-base-patch32_train30k.pt`), CelebA attribute labels as 0/1
  targets — 40 binary classification problems sharing the same inputs.
- **Model**: score for attribute *a* is `w_a · v + b_a`; all 40 probes are fit
  jointly as a single (40, 512) weight matrix.
- **Optimization**: full-batch Adam, 2000 epochs, lr 0.05, no weight decay,
  `BCEWithLogitsLoss`. CLIP stays frozen throughout. (Reproduce with
  `scripts/fit_probes.py`.)
- **Output**: weights, biases, and the attribute-name order are saved to
  `results/probe_weights.pt`.

## 3. Probe inference

As a classifier, a probe scores an image with its logit `w_a · v + b_a`
(`probe_scores`); thresholding gives has/hasn't predictions.

For retrieval, classification is not run at all — only the *direction* of each probe
is used. The weight vector `w_a` is the normal of the hyperplane separating
"has attribute" from "hasn't": moving an embedding along `ŵ_a = w_a / ‖w_a‖`
is the linear step that most increases the probe's confidence in the attribute.
Because ranking is also a dot product, adding `ŵ_a` to the query shifts every
database image's score by exactly its probe logit (up to the bias constant) — the
composition and the classifier perform the same operation.

## 4. Composition

For a query with positive set T+ and negative set T− (`compose_probe` in
`src/probes.py`):

```
q = normalize( γ · v_ref + Σ_{a∈T+} ŵ_a − Σ_{a∈T−} ŵ_a )
```

- **Only the queried attributes** contribute; the other probes are unused.
- **γ = 0.6**, chosen by a sweep on the benchmark. γ < 1 is essential: database
  candidates are images, and image–image cosines in CLIP space are systematically
  larger than cross-modal ones, so at γ = 1 the reference term dominates and
  retrieval returns near-duplicates of the reference while largely ignoring the
  edits. The reference must be down-weighted, not boosted (γ = 0 also fails —
  the reference carries real identity signal).

## 5. Retrieval

`rank` in `src/retrieval.py`: cosine similarity (dot product of normalized vectors)
between `q` and all cached test-split features, source image excluded, sorted
descending. Evaluation reports Recall@{1,5,10} and Precision@{1,5,10} per query,
averaged over each query's source images (`run_probe_benchmark` in
`src/evaluation.py`; reproduce with `scripts/run_probe_gamma_ablation.py`).

## 6. Key ablation results

MEAN over the 14 benchmark queries, full test-split database:

| method | attributes used | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| Prompt arithmetic baseline (γ = 1) | query only | 0.023 | 0.071 | 0.106 |
| Prompt arithmetic, tuned γ = 0.3 | query only | 0.035 | 0.115 | 0.174 |
| Attribute caption | all 40 | 0.013 | 0.051 | 0.085 |
| **Probe directions, tuned γ = 0.6 (current)** | query only | **0.052** | **0.140** | **0.207** |

Takeaways:

- **Using all attributes is not worth it.** The caption approach ranks last —
  describing all 40 attributes dilutes the few that changed and compounds 40
  attribute-prediction errors into every query.
- **γ is the single biggest lever.** Tuning one scalar lifts the prompt baseline
  from 0.106 to 0.174 R@10 with zero training.
- **Probe directions add real value beyond γ.** They beat the prompt directions at
  every γ, peaking at 0.207 vs 0.174 (~19% relative); the peak sits at a higher γ
  (0.6 vs 0.3) because in-space directions need less reference down-weighting —
  consistent with the modality-gap explanation.
- **Caveat**: γ was selected on the benchmark queries themselves (a single scalar,
  so low overfitting risk, but no held-out query split exists).

## 7. Future work: better combination of directions and reference

The current composition is a fixed linear rule; the planned improvements, cheapest
first:

1. **Per-attribute step sizes.** `q = normalize(γ·v_ref + Σ α_a·ŵ_a − …)`: the probe
   gives the direction but not how far to walk, and the step that flips `Bald` is
   not the one that flips `Arched_Eyebrows`. ~12 scalars fit on validation queries.
2. **Residual correction network.** Keep the arithmetic as backbone and learn a small
   MLP that nudges the result, `q = normalize(arith + MLP(v_ref, dirs))` — models
   attribute interactions, degrades gracefully to the current system.
3. **Full learned combiner** (see `docs/method-proposal-scac.md`): reference-
   conditioned fusion with attention and violation-aware contrastive training.
   Highest ceiling, but requires constructing (reference, edit, target) triplet
   supervision from CelebA labels.
4. **Clean γ calibration** on a held-out subset of source images.
