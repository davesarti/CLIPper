# Method history

**Compositional image retrieval on CelebA with frozen CLIP ViT-B/32.** How the current
method was arrived at, and what each step measured — including the approaches that were
tried and dropped.

**This file is history only. For the method as it stands, read `docs/method.md`;**
nothing described here is part of the current pipeline unless that document says so.
The code for the dropped approaches has been removed (it is in git history); their
result files are under `results/archive/`, where nothing reads them.

Task: given a reference image embedding `v_ref`, positive attributes **T+** and
negative attributes **T−**, build a single query embedding `q` such that ranking the
database by cosine similarity to `q` returns images that resemble the reference but
have every T+ attribute and lack every T−. Only the query side may change; the database
(test split, 19,962 images) is fixed. Benchmark: 14 queries of the form
`"+Smiling, -Blond_Hair"`, Recall@K / Precision@K, averaged over each query's source
images.

## 1. Prompt arithmetic (baseline)

Attribute directions are CLIP *text* embeddings of hand-written prompts
(`"a photo of a smiling person"`, `PROMPTS` in `src/retrieval.py`):

```
q = normalize( γ · v_ref + Σ t+ − Σ t− )
```

Zero training. Its weakness is the **modality gap**: text embeddings live on a
different cone of CLIP space than image embeddings, so they are poorly calibrated as
edit directions. Tuning the single scalar γ from 1 to 0.3 lifted R@10 from 0.106 to
0.174 — the biggest single lever found anywhere in this project, for zero training.

## 2. Attribute caption

Predict the full 40-attribute state of the reference (zero-shot, prompt pairs +
calibrated thresholds), flip the queried bits, render the *entire* state as one
negation-free caption, and use its text embedding as `q`. This uses **all**
attributes, not just the queried ones. (Code removed; results kept in
`results/archive/attribute_caption_results.csv` and
`results/archive/attribute_accuracy.csv`.)

It ranks last (R@10 = 0.085). Describing all 40 attributes dilutes the few that changed
and compounds 40 attribute-prediction errors into every query. **Using all attributes
is not worth it** — every later method uses only the queried ones.

## 3. Probe-direction composition

Same arithmetic as the baseline, but the directions are learned linear-probe weight
vectors — directions that live in the *image* embedding space.

**Probe training** (`fit_linear_probes` in `src/probes.py`, reproduce with
`scripts/fit_probes.py`): one logistic regression per attribute on the L2-normalized
CLIP features of a 30k-image train-split sample; all 40 probes fit jointly as a single
(40, 512) matrix; full-batch Adam, 2000 epochs, lr 0.05, no weight decay,
`BCEWithLogitsLoss`; CLIP frozen throughout. Saved to `results/probe_weights.pt`.

**Why the direction is the right object.** As a classifier the probe scores an image
with the logit `w_a · v + b_a`. For retrieval, classification is never run — only
`ŵ_a = w_a / ‖w_a‖`, the normal of the separating hyperplane, i.e. the linear step that
most increases the probe's confidence in the attribute. Because ranking is also a dot
product, adding `ŵ_a` to the query shifts every database image's score by exactly its
probe logit (up to the bias constant): the composition and the classifier perform the
same operation.

**Composition** (`compose_probe` in `src/probes.py`):

```
q = normalize( γ · v_ref + Σ_{a∈T+} ŵ_a − Σ_{a∈T−} ŵ_a ),   γ = 0.6
```

γ < 1 is essential: database candidates are images, and image–image cosines in CLIP
space are systematically larger than cross-modal ones, so at γ = 1 the reference term
dominates and retrieval returns near-duplicates of the reference while largely ignoring
the edits. γ = 0 also fails — the reference carries real identity signal.

### Where things stood

MEAN over the 14 benchmark queries, full test-split database:

| method | attributes used | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| Prompt arithmetic (γ = 1) | query only | 0.023 | 0.071 | 0.106 |
| Prompt arithmetic, tuned γ = 0.3 | query only | 0.035 | 0.115 | 0.174 |
| Attribute caption | all 40 | 0.013 | 0.051 | 0.085 |
| **Probe directions, tuned γ = 0.6** | query only | **0.052** | **0.140** | **0.207** |

Probe directions beat prompt directions at every γ (0.207 vs 0.174, ~19% relative), and
their peak sits at a *higher* γ — in-space directions need less reference
down-weighting, consistent with the modality-gap explanation. Caveat: γ was selected on
the benchmark queries themselves (one scalar, so low overfitting risk, but no held-out
query split exists).

**The documented weakness** that motivated everything after: the probe directions are
not orthogonal — they share components (blond↔gender, beard↔age, …) — yet the rule adds
them as if independent, with one global reference weight and a unit step for every
attribute regardless of the reference.

## 4. CPAS, transformer version

Keep the formula, make its three fixed choices learned functions of the query:

```
q = normalize( γ(x) · v_ref + Σ_a s_a · α_a(x) · normalize(ŵ_a + Δ_a(x)) )
```

Trunk (code removed; see below): role embeddings for reference / T+ / T−, one
stock `nn.TransformerEncoderLayer`
(d_model 512, 4 heads, FFN 1024, pre-LN) over the token sequence
`[v_ref ; ŵ_a1 ; ŵ_a2 ; …]`, then `γ` from the reference slot and `α_a`, `Δ_a` from
each attribute slot, with `Δ` bounded by `δ_max · tanh(·)` from a full 512×512 head.
≈ 2.4 M parameters. Heads zero-initialized with biases set so the untrained network
reproduces the γ = 0.6 rule exactly. Trained on mined attribute-flip triplets with
InfoNCE and three hard-negative types (the training recipe is unchanged in the current
proposal, `docs/method.md` §5).

The premise was that self-attention is what models direction interaction: each
attribute token sees the reference and the *other* attribute tokens.

### The δ_max ablation

Two seeds per variant, full mining pool, MEAN over the 14 queries
(`results/archive/cpas_ablation_transformer.csv`). "Leakage" is the mean absolute probe drift on the 38
non-queried attributes.

| variant | R@1 | R@5 | R@10 | queried shift | leakage |
|---|---|---|---|---|---|
| Probe composition (γ = 0.6) | 0.052 | 0.140 | 0.207 | 0.78 | 0.084 |
| CPAS, δ_max = 0 (γ and step sizes only) | 0.049 | 0.142 | 0.212 | 0.73 | 0.077 |
| CPAS, δ_max = 0.1 | 0.061 | 0.181 | 0.270 | 0.46 | 0.052 |
| **CPAS, δ_max = 0.3** | 0.062 | 0.188 | **0.277** | 0.40 | 0.050 |
| CPAS, δ_max = 0.3, no cross-attention | 0.064 | 0.183 | 0.275 | 0.41 | 0.051 |

Seed-to-seed range on R@10 is variant-dependent and not uniformly small: 0.0007
(δ_max = 0.3), 0.004 (δ_max = 0.1), 0.002 (δ_max = 0), but **0.019** for the
no-cross-attention variant. Gaps below ~0.02 on this ladder are not resolvable at two
seeds.

### What this settled

1. **Only re-aiming the directions matters.** With directions frozen (δ_max = 0) the
   model lands on the fixed rule — 0.212 vs 0.207, inside seed noise — despite learning
   γ and per-attribute steps that vary substantially per query. Allowing any bend
   recovers the whole gain at once. Learned step sizes buy nothing on their own, so
   per-attribute step sizes are not worth pursuing as a standalone direction.
2. **Cross-attribute attention is not what fixes non-orthogonality.** Masking
   attribute-to-attribute attention costs nothing: 0.275 vs 0.277, with
   δ_max = 0.1 / 0.3 / no-cross spanning only 0.007 against a no-cross seed range of
   0.019. What the model needs is each direction re-aimed *conditioned on the
   reference*, not the directions seeing each other. Caveat: only 6 of 14 benchmark
   queries have k ≥ 2, so attention is barely exercised either way.
3. **The gain is not just "smaller edits".** CPAS cuts leakage 0.084 → 0.050 but also
   shrinks the shift on the *queried* attributes (0.78 → 0.40), so its selectivity
   ratio is no better than the fixed rule's. Shrinking the fixed rule's edit does not
   reproduce the result: in `results/probe_gamma_ablation.csv` accuracy falls
   monotonically as the edit shrinks (γ = 0.9 → 0.189, γ = 1.5 → 0.145). At comparable
   edit magnitude the fixed rule scores ~0.14 and CPAS 0.28 — better retrieval from a
   *smaller* edit, which is re-aiming, not rescaling.
4. **The bend is large.** At δ_max = 0.3, `cos(ŵ_a + Δ_a, ŵ_a)` averages 0.49, a ~60°
   rotation. The model is closer to learning new reference-conditioned directions than
   to correcting the probe ones.

### Why the trunk was dropped

Points 1 and 2 say the transformer's distinguishing feature earns nothing measurable
while the Δ head earns everything, and point 4 says Δ has more freedom than
"correcting" the probe directions needs. That motivated CPAS-MLP (`src/cpas_mlp.py`):
2.1 M of trunk becomes 0.48 M of per-attribute MLP, and the 512×512 Δ head becomes a
rank-32 factorization. On the full pool it lands at Δ +0.057 over the fixed rule
against the transformer's +0.071 — a 0.014 gap, inside the 0.019 seed range measured
above, at **4.8× fewer parameters**. The honest reading is that the trunk is not
clearly earning its cost, not that the MLP is better; a single-machine replication of
both at 3 seeds with identical probes is the only way to call it. On that basis the
MLP is the current method (`docs/method.md` §5).

## 5. SCAC — the proposal that was not built as written

Before CPAS there was **SCAC** (Sign-Conditioned Attentive Combiner with Violation-Aware
Contrastive Training): a ~7 M-parameter transformer combiner in which negative
attributes entered as `+ŵ_a` tagged with a learned *negative sign embedding* rather than
as `−ŵ_a`, so the network was never pre-committed to "negation = subtraction". Exclusion
was to be taught by the loss instead, through mined violation negatives.

It was never implemented in that form, and the reasons are worth keeping because they
shaped what replaced it:

- **The sign-embedding idea was not separable from everything else it changed.** SCAC
  bundled a new trunk, a new attribute representation and a new loss. There was no way
  to attribute a gain to the sign tokens rather than to the extra capacity.
- **What survived is the training half, not the architectural half.** Violation
  negatives went straight into CPAS mining (§4, and `docs/method.md` §5.1) and are the
  foundation of the current negation-aware mining work. The sign-embedding trunk did
  not: the δ_max ablation later showed that cross-attribute attention buys nothing, so a
  larger attentive combiner was the wrong direction.
- **Its diagnosis was right and outlived it.** SCAC's stated failure mode — subtracting
  a probe normal overshoots, because the normal carries everything correlated with the
  attribute, and non-orthogonal directions cannot be added as if independent — is
  exactly the problem the Δ head addresses, and its conjunctive half is what the
  exclusion re-rank addresses (`docs/method.md` §7).

The lesson carried forward: **a proposal that changes representation, capacity and loss
at once cannot be ablated.** The current extensions are each default-off and separately
measurable for that reason.
