# SCAC: Sign-Conditioned Attentive Combiner with Violation-Aware Contrastive Training

**Method proposal — compositional image retrieval on CelebA with frozen CLIP ViT-B/32.**

| | |
|---|---|
| Dataset | CelebA (40 binary attributes); test split as fixed retrieval database |
| Encoder | Frozen CLIP ViT-B/32, d = 512 |
| Trainable | ≈ 7 M parameters (lightweight combiner) |
| Budget | Single Colab GPU; CLIP features pre-computed once |

## 0. Problem setup

Given a reference image `v_ref`, positive attributes **T+** and negative attributes **T−**, build a fusion module Φ producing one query embedding. Retrieval over a fixed, pre-embedded database must return images that keep the reference's identity, satisfy every T+ attribute, and avoid every T− attribute. Baseline to beat: naive CLIP arithmetic

```
v_target ≈ v_ref + Σ t+ − Σ t− ,  ranked by cosine similarity
```

Only the query side (plus optional light re-ranking of a shortlist) may change.

## 1. Fusion mechanism and the failure mode it fixes

**Mechanism.** A small transformer-based combiner fuses `v_ref` with attribute text embeddings tagged by learned *sign embeddings* (one for "must have", one for "must avoid"), producing the query as a **gated residual on `v_ref`**. The T+/T− asymmetry is enforced architecturally (sign tokens) but primarily through the loss: training mines **violation negatives** — images that satisfy T+ but also contain a T− attribute — so the network learns that negative attributes define an exclusion region, not a direction to walk backwards along.

**Why the arithmetic / SVD-subspace baseline fails.** It treats attributes as globally linear, mutually independent directions in CLIP space:

1. **Subtraction overshoots.** Subtracting `t("beard")` doesn't land on "no beard" — it moves toward the semantic opposite of everything correlated with beard (masculinity, age), destroying identity. Negation is not the additive inverse in CLIP space; CLIP is famously bad at "not X".
2. **Modality gap.** CLIP image and text embeddings live on two separated cones (Liang et al., NeurIPS 2022); adding raw text vectors to an image vector mixes the cones with uncalibrated magnitudes, so one strong text direction can dominate `v_ref`.
3. **No conditioning.** The right direction for "add blond hair" depends on the reference (dark-haired man vs. gray-haired woman); a global text direction — or a global SVD subspace — cannot adapt per reference. Attention over the reference token fixes exactly this.

Closest prior art: the **Combiner** of Baldrati et al. (CVPR 2022, CLIP4Cir) and **TIRG** (Vo et al., CVPR 2019) for the gated residual. SCAC's novelty: multi-attribute *set* input with signed tokens, and the violation-negative training signal for explicit negation — neither handles T− at all.

## 2. Architecture

All embeddings are frozen CLIP ViT-B/32 outputs, d = 512, L2-normalized.

**Inputs**
- `v_ref ∈ R^512`
- `t+_i = E_text("a photo of a person with {attr_i}")`, i = 1..P (prompt-ensembled over 4–6 templates)
- `t−_j` likewise, j = 1..N

**Learned components**
- Sign embeddings `s_pos, s_neg, s_ref ∈ R^512` (3 vectors)
- Transformer encoder `f_θ`: 2 layers, d_model = 512, 8 heads, FFN dim 2048, pre-LN
- Gate head `g_θ`: Linear(512→512) → ReLU → Linear(512→1) → sigmoid
- Output head `h_θ`: Linear(512→512)

**Forward pass**

```
X = [ v_ref + s_ref ;  t+_1 + s_pos ; … ; t+_P + s_pos ;
      t−_1 + s_neg ; … ; t−_N + s_neg ]        # (1+P+N, 512)

Z = f_θ(X)                                      # full self-attention, no mask
z = Z[0]                                        # reference-slot output
α = g_θ(z)                                      # scalar gate in (0,1)

q = normalize( (1−α)·v_ref + α·h_θ(z) )        # gated residual → query ∈ R^512
```

```python
class SCAC(nn.Module):
    def __init__(self, d=512, layers=2, heads=8): ...
    def forward(self, v_ref, t_pos, t_neg):
        # v_ref: (B, 512) · t_pos: (B, P, 512) · t_neg: (B, N, 512)
        # returns q: (B, 512), L2-normalized
```

**Retrieval.** Cosine similarity of `q` against the pre-computed database. Optional light re-rank on top-K (K = 100), never touching the database embeddings:

```
score(v) = cos(q, v) − λ · max_j cos(t−_j, v)
```

with a single scalar λ tuned on validation (ablated in §4).

**How T+ and T− differ.** (a) Different sign embeddings let attention treat them differently; (b) the gate α preserves identity when constraints are few/weak; (c) crucially, the loss penalizes T− violations with dedicated hard negatives, so `s_neg` learns "steer away from images scoring high on this text" rather than "subtract this vector"; (d) the λ re-rank term is a third, explicit asymmetry.

**Param count.** Per transformer layer ≈ 4·512² (attn) + 2·512·2048 (FFN) ≈ 3.15 M → ~6.3 M for two layers, + heads ≈ 0.5 M, + 3 sign vectors: **≈ 7 M trainable params**. All CLIP features are cached, so each run trains in well under an hour on a Colab T4.

## 3. Training signal from the 40 attributes

Pairs are built on the CelebA **train split**; the test split stays a clean retrieval database.

**Pair construction (attribute-flip mining)**
1. Sample a reference image with attribute vector `a_ref ∈ {0,1}^40`.
2. Sample a flip count k ∈ {1, 2, 3} (curriculum: k = 1 first, then mix).
3. Choose k flippable attributes; flips 0→1 become T+, flips 1→0 become T− (queries can be all-positive, all-negative, or mixed).
4. **Target** = any train image that has all T+, lacks all T−, and agrees with `a_ref` on ~10 *identity-proxy* attributes (gender, age- and face-shape-related); among candidates, take max attribute-Hamming agreement, tie-broken by CLIP-image similarity to `v_ref`. This is the supervision proxy for "keeps identity + satisfies constraints" without retrieval labels.

**Loss.** InfoNCE over image embeddings, temperature τ ≈ 0.05:

```
L = −log [ exp(cos(q, v_tgt)/τ) / Σ_{v ∈ B ∪ H} exp(cos(q, v)/τ) ]
```

`B` = in-batch targets (batch 256+ is free with cached features). `H` = three mined hard negatives per query:

| Negative | Definition | What it teaches |
|---|---|---|
| **Violation** | Satisfies all T+ but has ≥ 1 attribute from T− | Negation as an exclusion constraint — the load-bearing one |
| **Identity distractor** | Satisfies T+ and T− but disagrees heavily with `a_ref` on identity-proxy attributes | Prevents ignoring `v_ref` |
| **Lazy** | The reference itself (or attribute near-duplicates) | Prevents collapse to q ≈ v_ref (gate α → 0) |

Optional anchor regularizer `+ β·(1 − cos(q, v_ref))`, β ≈ 0.1; tune against the lazy negative — they pull opposite ways.

**Evaluation without retrieval labels.** Recall@K where a test image counts as correct if it satisfies all T+/T− and matches the identity-proxy attributes; plus per-constraint satisfaction rate of the top-10 (fraction having each T+, lacking each T−) — this decomposition shows exactly where the baseline fails on negatives.

## 4. Ablations

| Variant | Change | Question it answers |
|---|---|---|
| **SCAC-noSign** | Drop sign embeddings; feed T− as plain tokens, subtract at output: `q′ = normalize(q − Σ t−)` | Does learned negation beat arithmetic negation? |
| **SCAC-noViolation** | Same architecture, in-batch negatives only | Training signal vs. architecture (prediction: signal matters more) |
| **MLP-Combiner** | CLIP4Cir-style MLP on `[v_ref ; mean(t+) ; mean(t−)]` (~1 M params) | Is attention over individual attribute tokens needed, especially k ≥ 2? |
| **± λ re-rank** | Toggle re-rank term on every variant | Did fusion learn negation, or did the re-ranker patch it? |

Report all variants against the arithmetic baseline and the SVD-subspace method, stratified by k = 1/2/3 and by positive-only / negative-only / mixed queries.

## 5. Likely failure cases and mitigations

| Failure case | Why | Mitigation |
|---|---|---|
| Compound queries (k ≥ 3) | CLIP binds attributes poorly; few images satisfy 3+ simultaneous flips → sparse, noisy supervision | Curriculum on k; compositional-generalization split (train k ≤ 2, test k = 3); relaxed best-effort Hamming targets with down-weighted loss |
| Rare / contradictory combos | "bald ∧ female" ≈ 0.02% of CelebA; "bald + wearing hat" unobservable → mining returns garbage | Inverse-frequency attribute sampling; drop combos with < 20 candidate targets; report performance vs. combo frequency |
| Correlated-attribute leakage | Removing "heavy makeup" drags gender; adding "gray hair" drags age — violation negatives only cover *listed* T− | Fourth negative type: images matching the query but flipped on a correlated non-queried attribute (correlations from train attribute matrix) |
| Weak identity proxy | Attribute agreement only approximates identity | CelebA has identity labels — measure top-K identity match; tighten proxy set or raise β if degraded |
| Gate collapse (α → 0) | Query ≡ reference wins identity metrics while ignoring constraints | Monitor α's distribution; the lazy negative is the direct antidote |

> **Key claim.** The single highest-leverage piece is the **violation negative**: it converts "T− as a vector to subtract" into "T− as a constraint to satisfy" — a distinction neither the arithmetic baseline nor the SVD-subspace method can express at all.

## 6. References

- Baldrati, Bertini, Uricchio, Del Bimbo. *Effective conditioned and composed image retrieval combining CLIP-based features.* CVPR 2022. Code: <https://github.com/ABaldrati/CLIP4Cir>
- Vo, Jiang, Sun, Murphy, Li, Fei-Fei, Hays. *Composing Text and Image for Image Retrieval (TIRG).* CVPR 2019.
- Liang, Zhang, Kwon, Yeung, Zou. *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS 2022.
- Baldrati, Agnolucci, Bertini, Del Bimbo. *Zero-Shot Composed Image Retrieval with Textual Inversion (SEARLE).* ICCV 2023 — zero-shot comparison point for related work.
