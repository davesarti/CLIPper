# CPAS: Conditioned Per-Attribute Steering

**Method proposal — compositional image retrieval on CelebA with frozen CLIP ViT-B/32.**

| | |
|---|---|
| Dataset | CelebA (40 binary attributes); test split as fixed retrieval database |
| Encoder | Frozen CLIP ViT-B/32, d = 512; features pre-computed once |
| Attribute representation | Signed **probe directions** `ŵ_a` (frozen linear-probe weights, `results/probe_weights.pt`) |
| Trainable | ≈ 2.4 M parameters (one transformer encoder layer + three small heads) |
| Position in roadmap | Between probe-direction composition (fixed rule, current best) and SCAC (full learned combiner, `docs/method-proposal-scac.md`) |

## 0. Problem setup

Given a reference image embedding `v_ref`, positive attributes **T+** and negative
attributes **T−**, build a query embedding `q` for cosine-similarity ranking over the
fixed test-split database. Baseline to beat: **probe-direction composition**
(`docs/method-probe-direction-retrieval.md`), R@10 = 0.207:

```
q = normalize( γ · v_ref + Σ ŵ+ − Σ ŵ− ),   γ = 0.6 (global, tuned)
```

Its documented weakness: the probe directions are **not orthogonal** — they share
components (blond↔gender, beard↔age, …) — yet the rule adds them as if independent,
with one global reference weight and unit step for every attribute regardless of the
reference.

## 1. Idea

Keep the *formula* of the current method; make its three fixed choices **learned
functions of the query**:

```
q = normalize( γ(x) · v_ref  +  Σ_a  s_a · α_a(x) · normalize(ŵ_a + Δ_a(x)) )
```

with `s_a = ±1` for T+/T−, and `x` = (reference, queried attribute set):

| Current (fixed) | CPAS (learned per query) | What it fixes |
|---|---|---|
| γ = 0.6 global | scalar `γ(x)` | reference weight adapts to the edit at hand |
| unit step ∀ attributes | scalar `α_a(x)` per attribute | step size depends on the attribute *and* on where `v_ref` already is |
| direction = `ŵ_a` exactly | `ŵ_a + Δ_a(x)`, ‖Δ‖ bounded | **the non-orthogonality fix**: bends each direction away from components shared with the co-queried edits |

Because the model class contains the baseline (γ = 0.6, α = 1, Δ = 0), any benchmark
gain is attributable to the learned conditioning, not to a changed formula — and the
network is *initialized at exactly that point* (§2), so training starts from the
current method and can only move away from it if the data says so.

Every learned quantity is directly inspectable: per-attribute α distributions, γ vs.
edit count, and the alignment of `Δ_a` with correlated probe directions (does the bend
for `+Blond_Hair` point away from `ŵ_Male`?) are all one-line diagnostics — the
"easy to explain and justify" property that motivates trying CPAS before SCAC.

## 2. Architecture

All inputs are frozen and L2-normalized: image embeddings from cached CLIP features,
attribute directions from the trained probes. Built from scratch in plain PyTorch
(stock `nn.TransformerEncoderLayer`); there is no pretrained checkpoint for attention
over CLIP probe directions, so nothing to import.

**Learned components**
- Role embeddings `s_ref, s_pos, s_neg ∈ R^512` (negatives enter as `+ŵ_a` tagged
  `s_neg`, not as `−ŵ_a`; the output sign `s_a` is applied only at composition)
- 1 × transformer encoder layer: d_model = 512, 4 heads, FFN 1024, pre-LN
- Heads: `γ` = Linear(512→1)+sigmoid on the reference slot; `α_a` =
  Linear(512→1)+softplus and `Δ_a` = δ_max·tanh(Linear(512→512)), δ_max = 0.3,
  on each attribute slot

**Forward pass** (batch B, up to K attributes, padded + key-padding mask)

```
tokens = [ v_ref + s_ref ;  ŵ_a1 + s_(pos|neg) ; … ]    # (B, 1+K, 512)
Z      = encoder(tokens)                                 # single layer, full attention
γ      = sigmoid(head_γ(Z[:, 0]))
α_a    = softplus(head_α(Z[:, 1:]))
Δ_a    = δ_max · tanh(head_Δ(Z[:, 1:]))

q = normalize( γ·v_ref + Σ_a s_a · α_a · normalize(ŵ_a + Δ_a) )
```

The single self-attention layer is the component that models direction
interaction: each attribute token sees the reference (steps adapt to where the
reference is) and the *other* attribute tokens (steps adapt to which edits are
co-queried — impossible for any per-attribute rule).

**Initialization = baseline.** Head weight matrices are zero-initialized with biases
set so that at step 0: γ = 0.6, α_a = 1, Δ_a = 0. The untrained network reproduces
probe-direction composition exactly (verified by unit test).

**Parameter count.** Attention ≈ 1.05 M + FFN ≈ 1.05 M + Δ head ≈ 0.26 M + small
heads and role embeddings ≈ 3 k → **≈ 2.4 M**.

## 3. Training

Triplets are mined from CelebA **train-split** attribute labels (test split stays a
clean database). Same signal as the SCAC proposal, §3:

1. Sample a reference from the cached train pool; take its 40-bit label vector.
2. Sample k ∈ {1, 2, 3} attributes to flip (curriculum: k = 1 first, then mixed);
   flips 0→1 form T+, 1→0 form T−. Combos with too few candidate targets are
   rejected and resampled.
3. **Target** = a train image having all T+, lacking all T−, with maximal agreement
   with the reference on ~10 identity-proxy attributes (gender/age/face-shape);
   ties broken by CLIP similarity to the reference.

**Loss**: InfoNCE (τ ≈ 0.05) over image embeddings — pull `q` to its target, push
from in-batch targets plus three mined hard negatives per query:

| Negative | An image that... | Kills the cheat of... |
|---|---|---|
| Violation | has all T+ but also ≥ 1 attribute of T− | ignoring the negative constraints |
| Identity distractor | satisfies T+/T− but disagrees heavily on identity-proxy attributes | ignoring `v_ref` |
| Lazy | the reference itself | γ → 1, α → 0: returning the reference unchanged |

**Honest limitation vs. SCAC.** CPAS's output form makes negation architecturally a
(learned-length, learned-bend) subtraction; the violation negatives can tune it but
not replace it with an exclusion constraint. If T− queries remain the weak spot,
that is the cleanest motivation for graduating to SCAC.

## 4. Ablation ladder

Each row adds one component; the deltas attribute the gain:

| # | Variant | Learns |
|---|---|---|
| 0 | Probe composition (current) | nothing — R@10 = 0.207 reference point |
| 1 | 40 constant α_a (no network) | per-attribute step size only |
| 2 | Conditioning MLP, no attention | steps adapt to the reference, not to each other |
| 3 | **CPAS** (single attention layer) | steps adapt to co-queried attributes → isolates the non-orthogonality gain |
| 4 | CPAS, Δ = 0 frozen | is bending needed, or only rescaling? |

Rows 2–4 are implemented as flags on the same model — `--delta-max` bounds the
bend (0 freezes the directions) and `--no-cross-attention` masks
attribute-to-attribute attention, keeping reference conditioning. Results in §7.

Plus the **probe-drift diagnostic** (as in the SCAC doc): score `q` with all 40
frozen probes and report mean logit shift on queried (large, correct sign) vs.
non-queried (≈ 0) attributes — the direct quantitative measure of edit leakage.

## 5. Evaluation

The existing 14-query benchmark: Recall@{1,5,10} / Precision@{1,5,10} against the
full test-split database, per query and MEAN (`run_cpas_benchmark` in
`src/evaluation.py`). Model selection and early stopping use held-out references
from the train pool, never the benchmark queries.

## 6. First results

Trained with `scripts/train_cpas.py` (20k mined triplets, 8 warm-up epochs on
k = 1 then 25 on k ∈ {1,2,3}, Adam 1e-4, batch 256, ~17 min on CPU) and
evaluated with `scripts/run_cpas_benchmark.py`. Checkpoint selected by
validation batch recall@1 on the held-out image pool: epoch 13 of 33 — training
recall keeps climbing to 0.77 afterwards while validation falls, so early
stopping is load-bearing.

MEAN over the 14 benchmark queries, full test-split database:

| method | R@1 | R@5 | R@10 | P@10 |
|---|---|---|---|---|
| Probe composition (γ = 0.6) | 0.052 | 0.140 | 0.207 | 0.032 |
| **CPAS** | **0.056** | **0.185** | **0.281** | **0.045** |

+36% relative R@10 over the rule it was initialized from. The gain is
concentrated at K = 5–10 rather than K = 1.

**What the model learned**, read off the heads:

- `γ` ranges 0.25–0.70 per query (mean 0.46) against the fixed 0.6 — it keeps
  *more* reference for single-attribute edits like `+Male` (0.70) and much less
  for compound ones like `+Chubby, -Young` (0.25).
- `α` ranges 0.6–1.9 (mean 1.27), with the largest steps for the attributes CLIP
  binds worst (`+Mustache` 1.81, `+Eyeglasses` 1.82) and steps below 1 for
  removals (`-Heavy_Makeup` 0.61).
- Probe drift: leakage into the 38 non-queried attributes drops from 0.084 to
  0.050, so the edit is measurably cleaner — but the signed shift on the queried
  attributes also drops (0.78 → 0.39), i.e. CPAS does not simply push the probe
  logits harder.

**Caveat on Δ.** With `δ_max = 0.3` per coordinate the bends are not small:
`cos(ŵ_a + Δ_a, ŵ_a)` averages 0.49, a ~60° rotation, so the model is closer to
*learning new reference-conditioned directions* than to correcting the probe
ones — which also explains the reduced probe-logit shift.

## 7. Ablations: what actually earns the gain

Two seeds per variant, all trained on the same mined triplet pool
(`scripts/train_cpas.py --triplet-cache`), all evaluated with
`scripts/run_cpas_ablation.py`; MEAN over the 14 queries, seeds averaged:

| variant | R@1 | R@5 | R@10 | leakage |
|---|---|---|---|---|
| Probe composition (γ = 0.6) | 0.052 | 0.140 | 0.207 | 0.084 |
| CPAS, δ_max = 0 (γ and step sizes only) | 0.049 | 0.142 | 0.212 | 0.077 |
| CPAS, δ_max = 0.1 | 0.061 | 0.181 | 0.270 | 0.052 |
| **CPAS, δ_max = 0.3** | 0.062 | 0.188 | **0.277** | 0.050 |
| CPAS, δ_max = 0.3, no cross-attention | 0.064 | 0.183 | 0.275 | 0.051 |

Two findings, both against the original motivation:

1. **Learned γ and per-attribute step sizes are worth nothing.** Frozen
   directions (δ_max = 0) score 0.212 against the fixed rule's 0.207 — inside
   seed noise, despite γ and α varying substantially per query (§6). The entire
   gain comes from **re-aiming the directions**, and it appears as soon as
   bending is allowed at all (δ_max = 0.1 already gives 0.270). Future-work item
   1 of `method-probe-direction-retrieval.md` (per-attribute step sizes) is
   therefore not worth pursuing on its own.
2. **Cross-attribute attention is not what fixes non-orthogonality.** Blocking
   attribute-to-attribute attention costs nothing measurable: 0.275 vs 0.277
   overall, and on the 6 multi-attribute queries — the only ones where the mask
   changes anything — 0.295 vs 0.301, against a seed spread of 0.019 in the
   masked variant itself. What the model needs is each direction re-aimed
   *conditioned on the reference*; it does not need the directions to see each
   other.

The non-orthogonality problem is real (leakage into non-queried attributes drops
from 0.084 to 0.050, and multi-attribute queries gain more than single-attribute
ones: +0.098 vs +0.051 R@10), but the mechanism that solves it is
reference-conditioned re-aiming, not attribute-attribute interaction.

**Consequences for the roadmap.** The attention layer can be replaced by a
per-attribute conditioning MLP on `[v_ref ; ŵ_a ; sign]` at ~0.5 M parameters —
simpler to explain, cheaper, and so far equally accurate; the attention result
should be re-checked on a benchmark with more multi-attribute queries before
being written off for good. It also weakens the case for SCAC's 2-layer
transformer (`docs/method-proposal-scac.md` §2): on this evidence its expected
gain would come from the violation-negative training signal, not the
architecture.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Mining noise: targets are proxies ("another person with the right attributes") | identity-proxy agreement + CLIP-similarity tie-break; monitor identity-distractor loss term |
| Rare attributes/combos have few candidate targets | rejection sampling with a minimum-candidate threshold; report per-attribute results |
| Overfitting the small benchmark | never train or select on it; the 2.4 M model consumes frozen features only |
