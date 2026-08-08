# CPAS-MLP: Conditioned Per-Attribute Steering

**Method proposal — compositional image retrieval on CelebA with frozen CLIP ViT-B/32.**

| | |
|---|---|
| Dataset | CelebA (40 binary attributes); test split (19,962 images) as fixed retrieval database |
| Encoder | Frozen CLIP ViT-B/32, d = 512; features pre-computed once and cached |
| Attribute representation | Signed **probe directions** `ŵ_a` (frozen linear-probe weights, `results/probe_weights.pt`) |
| Trainable | ≈ 0.50 M parameters (per-attribute MLP + three small heads) |
| Baseline it is initialized from | Probe-direction composition, γ = 0.6 (see `docs/method-history.md`) |

## 0. Problem setup

Given a reference image embedding `v_ref`, positive attributes **T+** and negative
attributes **T−**, build a query embedding `q` for cosine-similarity ranking over the
fixed test-split database. Only the query side may change; the database features are
fixed.

The method this replaces is the fixed linear rule

```
q = normalize( γ · v_ref + Σ_{a∈T+} ŵ_a − Σ_{a∈T−} ŵ_a ),   γ = 0.6 (global, tuned)
```

Its weakness: the probe directions are **not orthogonal** — they share components
(blond↔gender, beard↔age, …) — yet the rule adds them as if independent, with one
global reference weight and a unit step for every attribute regardless of the
reference.

## 1. Idea

Keep the *formula*; make its three fixed choices **learned functions of the query**:

```
q = normalize( γ(x) · v_ref  +  Σ_a  s_a · α_a(x) · normalize(ŵ_a + Δ_a(x)) )
```

with `s_a = ±1` for T+/T−, and `x` = (reference, queried attribute set):

| Current (fixed) | CPAS (learned per query) | What it fixes |
|---|---|---|
| γ = 0.6 global | scalar `γ(x)` | reference weight adapts to the edit at hand |
| unit step ∀ attributes | scalar `α_a(x)` per attribute | step size depends on the attribute *and* on where `v_ref` already is |
| direction = `ŵ_a` exactly | `ŵ_a + Δ_a(x)`, ‖Δ‖ bounded | **the non-orthogonality fix**: re-aims each direction conditioned on the reference and the co-queried edits |

The model class contains the baseline (γ = 0.6, α = 1, Δ = 0), and the network is
*initialized at exactly that point* (§2), so training starts from the fixed rule and
can only move away from it if the data says so. Every learned quantity is directly
inspectable: per-attribute α distributions, γ vs. edit count, and the alignment of
`Δ_a` with correlated probe directions.

## 2. Architecture

`src/cpas_mlp.py`, `PerAttributeMLP`. All inputs are frozen and L2-normalized: image
embeddings from cached CLIP features, attribute directions from the trained probes.
There is no trunk with cross-attribute attention — each attribute is processed
independently, then re-processed against a pooled context of the *other* queried
attributes:

```
h1_a = MLP1([ v_ref ; ŵ_a ; sign_emb(s_a) ])      # (2·512 + 64) → 256 → 256, GELU, LayerNorm
c_a  = mean_{b ≠ a} h1_b                          # 0 for single-attribute queries
z_a  = MLP2([ h1_a ; c_a ])                       # 512 → 256, GELU, LayerNorm

γ    = sigmoid( head_γ([ mean_a h1_a ; v_ref ]) )
α_a  = softplus( head_α(z_a) )
Δ_a  = δ_max · tanh( U( V(z_a) ) )                # V: 256→32, U: 32→512, δ_max = 0.3

q = normalize( γ·v_ref + Σ_a s_a · α_a · normalize(ŵ_a + Δ_a) )
```

Notes that matter for correctness:

- **Negatives enter as `+ŵ_a` with a T− sign tag**, not as `−ŵ_a`; the output sign
  `s_a` is applied only at composition (`compose` in `src/steering.py`).
- **Padded slots are zeroed before pooling**, so a padded attribute cannot leak into
  any real attribute's context; `α` and `Δ` are zeroed on padded slots.
- **The context pool excludes the token itself**, so `c_a = 0` for K = 1 queries.
  Setting `cross_attributes = False` zeroes `c_a` without removing MLP2, leaving
  parameter count, initialization and seed unchanged — the cross-attribute ablation
  is exact.
- **The Δ head is rank-32 factorized** (0.025 M instead of 0.26 M for a full 512×512).

**Initialization = baseline.** `head_γ` and `head_α` weights are zeroed with biases
set so γ = 0.6 and α = 1; `delta_up` is zeroed (not `delta_down`, which would leave
the bend permanently dead) so Δ = 0 at step 0. The untrained module reproduces
`compose_probe(γ = 0.6)` exactly — verified by unit test.

## 3. Prerequisite: probe directions

`fit_linear_probes` in `src/probes.py`, reproduce with `scripts/fit_probes.py`:

- **Data**: L2-normalized CLIP features of a 30k-image train-split sample
  (`features/clip-vit-base-patch32_train30k.pt`, which stores the features together
  with the sampled indices used to align the labels), CelebA attribute labels as 0/1
  targets.
- **Model**: score for attribute *a* is `w_a · v + b_a`; all 40 probes are fit jointly
  as a single (40, 512) weight matrix.
- **Optimization**: full-batch Adam, 2000 epochs, lr 0.05, no weight decay,
  `BCEWithLogitsLoss`. Deterministic (zero init), so re-runs reproduce the weights.
- **Output**: `results/probe_weights.pt` (weights, biases, attribute order).

Only the *direction* `ŵ_a = w_a / ‖w_a‖` is used at retrieval time — the normal of the
separating hyperplane, i.e. the linear step that most increases the probe's confidence.

**Caution.** Probes refit from a different 30k sample align with the shipped ones at
cosine > 0.96, not exactly, so absolute benchmark numbers are not comparable across
probe fits. Always recompute the fixed-rule row with the same probes as the models
scored against it, and compare the **Δ over the fixed rule**.

## 4. Training

`scripts/train_cpas.py`. Triplets are mined from CelebA **train-split**
attribute labels (`src/mining.py`); the test split stays a clean database.

1. Sample a reference from the cached train pool; take its 40-bit label vector.
2. Sample k ∈ {1, 2, 3} attributes to flip (curriculum: k = 1 during warm-up, then
   mixed); flips 0→1 form T+, 1→0 form T−. Combos with fewer than
   `min_candidates = 20` targets are rejected and resampled.
3. **Target** = a train image having all T+, lacking all T−, with maximal agreement
   with the reference on ~10 identity-proxy attributes (gender/age/face-shape); ties
   broken by CLIP similarity to the reference.

**Loss**: InfoNCE (τ ≈ 0.05) over image embeddings — pull `q` to its target, push from
in-batch targets plus three mined hard negatives per query:

| Negative | An image that... | Kills the cheat of... |
|---|---|---|
| Violation | has all T+ but also ≥ 1 attribute of T− | ignoring the negative constraints |
| Identity distractor | satisfies T+/T− but disagrees heavily on identity-proxy attributes | ignoring `v_ref` |
| Lazy | the reference itself | γ → 1, α → 0: returning the reference unchanged |

**Hyperparameters** (defaults of `scripts/train_cpas.py`): 40k triplets re-mined every
epoch, 5 warm-up epochs on k = 1 then up to 45 on k ∈ {1,2,3}, Adam lr 1e-4, batch
1024, δ_max = 0.3, Δ rank 32, patience 15.

**Checkpoint selection** is held-out val R@10, never the benchmark queries: a val
benchmark built from val-split references (10% of the train pool) and the same 14
query shapes, mirroring the eval ground-truth rule. The mining-proxy recall@1 tracks
the true metric poorly and is kept only as a diagnostic.

**Use the full train pool.** Run `scripts/extract_train_features.py --all` first; the
script falls back to the 30k sample when the full-pool cache is absent. On the full
pool the held-out val database holds 16,277 images, close to the 19,962 of the test
split, and ranks seeds in the same order as the test benchmark; a 30k pool leaves only
3,000 val images and inflates val R@10 without tracking it better.

**Architectural limitation.** The output form makes negation a (learned-length,
learned-bend) subtraction; the violation negatives can tune it but not replace it with
an exclusion constraint. Negation-heavy queries remain the weak spot.

## 5. Evaluation

The 14-query benchmark: Recall@{1,5,10} / Precision@{1,5,10} against the full
test-split database, per query and MEAN (`run_cpas_benchmark` in `src/evaluation.py`).

```
conda run -n clipper python scripts/extract_train_features.py --all
conda run -n clipper python scripts/fit_probes.py
conda run -n clipper python scripts/train_cpas.py --seed 0 --out runs/mlp_s0.pt
conda run -n clipper python scripts/run_cpas_ablation.py "CPAS-MLP=runs/mlp_s0.pt"
```

`run_cpas_ablation.py` recomputes the fixed-rule row with the loaded probes, averages
over seeds sharing a variant name, and writes `results/cpas_ablation.csv`.

Report alongside recall the **probe-drift diagnostic**: score `q` with all 40 frozen
probes and report the mean logit shift on queried attributes (large, correct sign) vs.
the mean absolute shift on the 38 non-queried ones ("leakage") — the direct measure of
edit leakage.

## 6. Result

MEAN over the 14 queries, full test-split database, full mining pool, 3 seeds
(`results/cpas_ablation.csv`):

| variant | R@1 | R@5 | R@10 | Δ over rule | leakage | params |
|---|---|---|---|---|---|---|
| Probe composition (γ = 0.6) | 0.051 | 0.144 | 0.210 | — | 0.095 | 0 |
| **CPAS-MLP, δ_max = 0.3** | 0.066 | 0.182 | **0.267** | **+0.057** | 0.061 | 0.50 M |

Per-seed R@10: 0.2677 / 0.2689 / 0.2644 — σ = 0.0023, range 0.0045, selected epochs
30 / 35 / 31.

Two things this does *not* show. The gain is **re-aiming, not shrinking**: CPAS-MLP is
no more selective than the rule it started from (queried-shift / leakage ratio 8.07 vs
8.24) — it retrieves better from a smaller, better-aimed edit. And at δ_max = 0.3 the
bend is large: `cos(ŵ_a + Δ_a, ŵ_a)` averages ≈ 0.5, a ~60° rotation, so the model is
closer to learning new reference-conditioned directions than to correcting the probe
ones.

## 7. Open items

- **Sweep δ_max between 0 and 0.1.** The entire effect appears inside that interval
  (δ_max = 0 lands on the fixed rule; δ_max = 0.1 already gives most of the gain).
  Highest-value single experiment: it decides whether Δ is a correction to the probe
  directions or a replacement for them.
- **Enlarge the benchmark.** 14 queries, two of them duplicates (`-Young` appears
  twice), three with < 100 source images, and only 6 with k ≥ 2. Cross-attribute
  conditioning is barely exercised, and gaps below ~0.02 are not resolvable here.
- **Attack negation.** Per-query results (`results/cpas_results.csv`) put every
  remaining failure in the same bucket: negation-heavy or fine-grained queries
  (`+Wearing_Lipstick, -Heavy_Makeup, +Smiling` at R@10 ≈ 0.06, unchanged from the
  vanilla baseline; `+Chubby, -Young` at ≈ 0.09), while positive well-populated
  queries reach 0.30–0.75. This is the limitation §4 predicts.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Mining noise: targets are proxies ("another person with the right attributes") | identity-proxy agreement + CLIP-similarity tie-break; monitor identity-distractor loss term |
| Rare attributes/combos have few candidate targets | rejection sampling with `min_candidates`; report per-attribute results |
| Overfitting the small benchmark | never train or select on it; the 0.50 M model consumes frozen features only |
| Probe refits shift absolute numbers | always recompute the fixed rule with the same probes; compare Δ over the rule |
