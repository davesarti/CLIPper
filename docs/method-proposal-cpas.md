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
attribute-to-attribute attention, keeping reference conditioning. Results in §6–7.

Plus the **probe-drift diagnostic** (as in the SCAC doc): score `q` with all 40
frozen probes and report mean logit shift on queried (large, correct sign) vs.
non-queried (≈ 0) attributes — the direct quantitative measure of edit leakage.

## 5. Evaluation

The existing 14-query benchmark: Recall@{1,5,10} / Precision@{1,5,10} against the
full test-split database, per query and MEAN (`run_cpas_benchmark` in
`src/evaluation.py`). Model selection and early stopping use held-out references
from the train pool, never the benchmark queries.

## 6. Results

One run per variant (seed 0), trained with `scripts/train_cpas.py` on the full
train-split mining pool — 40k triplets re-mined per epoch, 5 warm-up epochs on
k = 1 then up to 45 on k ∈ {1,2,3}, Adam 1e-4, batch 1024, checkpoint selected by
held-out val R@10. Evaluated with `scripts/run_cpas_ablation.py`; MEAN over the
14 queries against the full test-split database. "Leakage" is the mean absolute
probe drift on the 38 non-queried attributes.

| variant | R@1 | R@5 | R@10 | leakage |
|---|---|---|---|---|
| Probe composition (γ = 0.6) | 0.052 | 0.140 | 0.207 | 0.084 |
| CPAS, δ_max = 0 (γ and step sizes only) | 0.045 | 0.127 | 0.191 | 0.059 |
| CPAS, δ_max = 0.1 | 0.062 | 0.181 | 0.262 | 0.047 |
| CPAS, δ_max = 0.3 | 0.066 | 0.185 | 0.259 | 0.047 |
| **CPAS, δ_max = 0.3, no cross-attention** | 0.065 | 0.187 | **0.267** | 0.045 |

CPAS gains ~+0.06 R@10 (+29% relative) over the rule it was initialized from,
concentrated at K = 5–10 rather than K = 1.

## 7. What actually earns the gain

1. **Only re-aiming the directions matters.** With directions frozen
   (δ_max = 0) the model scores *below* the fixed rule — 0.191 vs 0.207 —
   despite learning γ and per-attribute steps that vary substantially per query.
   Allowing any bend recovers the whole gain at once (δ_max = 0.1 already gives
   0.262). Learned step sizes are not merely worthless, they cost accuracy;
   future-work item 1 of `method-probe-direction-retrieval.md` (per-attribute
   step sizes) is not worth pursuing on its own.
2. **Cross-attribute attention is not what fixes non-orthogonality.** Masking
   attribute-to-attribute attention costs nothing: 0.267 vs 0.259, with δ_max =
   0.1/0.3/no-cross spanning only 0.008 against a measured seed spread of 0.019.
   The three are indistinguishable at one seed. What the model needs is each
   direction re-aimed *conditioned on the reference*, not the directions seeing
   each other.
3. **The gain is not just "smaller edits".** CPAS cuts leakage 0.084 → 0.045,
   but also shrinks the shift on the *queried* attributes (0.78 → 0.36), so its
   selectivity ratio is no better than the fixed rule's. Shrinking the fixed
   rule's edit does not reproduce the result: in `results/probe_gamma_ablation.csv`
   accuracy falls monotonically as the edit shrinks (γ = 0.9 → 0.189, γ = 1.5 →
   0.145). At comparable edit magnitude the fixed rule scores ~0.14 and CPAS
   0.26 — better retrieval from a *smaller* edit, which is re-aiming, not
   rescaling. δ_max = 0 fits the same picture from the other side: it shrank the
   step without being allowed to re-aim, and lost accuracy.

**Caveat on Δ.** At δ_max = 0.3, `cos(ŵ_a + Δ_a, ŵ_a)` averages 0.49 — a ~60°
rotation. The model is closer to *learning new reference-conditioned directions*
than to correcting the probe ones.

### Future work

- **Replace attention with a per-attribute conditioning MLP** on
  `[v_ref ; ŵ_a ; sign]` (~0.5 M params): simpler, cheaper, equally accurate on
  this evidence. Re-check on a benchmark with more multi-attribute queries
  before writing attention off for good — only 6 of 14 queries exercise it.
- **Run 4–5 seeds per variant** before claiming any ordering among δ_max = 0.1,
  0.3 and no-cross. The seed spread (0.019) is larger than the gaps (0.008).
- **Explain why heavier training did not help.** An earlier 2-seed run on the
  30k mining pool scored higher across every variant (δ = 0.3: 0.277 vs 0.259).
  Candidates: early stopping (`--patience 15`) truncating runs, or per-epoch
  re-mining from the full pool destabilising the objective. Check the saved
  `epoch` column first.
- **Sweep δ_max between 0 and 0.1** — the entire effect appears inside that
  interval and is currently unresolved.

This weakens the case for SCAC's 2-layer transformer
(`docs/method-proposal-scac.md` §2): on this evidence its expected gain would
come from the violation-negative training signal, not the architecture.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Mining noise: targets are proxies ("another person with the right attributes") | identity-proxy agreement + CLIP-similarity tie-break; monitor identity-distractor loss term |
| Rare attributes/combos have few candidate targets | rejection sampling with a minimum-candidate threshold; report per-attribute results |
| Overfitting the small benchmark | never train or select on it; the 2.4 M model consumes frozen features only |
