# Clipper — compositional image retrieval on CelebA

DL 2026 assignment (`Project assignment - V1.2.pdf`). Given a reference image and a
query like `"+Smiling, -Blond_Hair"`, retrieve images from the 19,962-image CelebA test
split that keep the reference's identity, have every `+` attribute and lack every `−`
one. CLIP ViT-B/32 stays frozen throughout; only the query embedding is built.

- **Baseline**: latent arithmetic over CLIP *text* prompts,
  `q = normalize(v_ref + Σt⁺ − Σt⁻)`. No training.
- **Current method**: retrieval scored directly against the assignment's ground-truth
  criterion (§3.1.1) — expected Hamming distance on the non-queried attributes plus a
  constraint penalty — over attribute codes predicted by a small head on frozen CLIP
  features. The CPAS-MLP combiner supplies the composite query embedding.

Two documents, kept deliberately apart:

- **`docs/method.md`** — the current pipeline and nothing else: the ground-truth
  criterion, attribute prediction, the scoring rule, the combiner, the evaluation
  protocol, current results, and what is still open. **Start here.**
- **`docs/method-history.md`** — how the method was arrived at, and what was tried and
  dropped (prompt arithmetic, attribute captions, the transformer combiner, SCAC).
  Read it for *why* the design is what it is.

## Layout

- `src/` — `data.py`, `features.py` (CLIP wrapper + feature cache), `retrieval.py`
  (prompts, `compose()`/`rank()`), `probes.py` (linear attribute probes and the fixed
  composition rule), `evaluation.py` (metrics, benchmarks, probe-drift diagnostic),
  `steering.py` (shared composition + padding), `mining.py` (triplet mining),
  `training.py` (InfoNCE loop), `cpas_mlp.py` (the combiner),
  `rerank.py` (constraint penalty), `attribute_head.py` (attribute predictor),
  `attribute_retrieval.py` (the ground-truth-criterion score)
- `scripts/` — `run_baseline.py`, `extract_train_features.py`,
  `fit_probes.py`, `run_probe_accuracy.py`, `run_probe_gamma_ablation.py`,
  `train_cpas.py`, `run_cpas_ablation.py`, `run_exclusion_rerank.py`,
  `fit_attribute_head.py`, `run_attribute_retrieval.py`
- `docs/` — `method.md` (the current pipeline) and `method-history.md` (what was tried
  and dropped, and why)
- `results/` — benchmark CSVs and probe weights; `results/archive/` holds output from
  abandoned approaches (nothing reads it)
- `tests/` — 97 pytest tests, model-free

## Setup

Put CelebA under `./celeba/` and `celeba_evaluation.json` at the repo root, then:

```bash
conda create -n clipper python=3.11 -y
conda run -n clipper pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision  # CPU build; skip flag on GPU
conda run -n clipper pip install -r requirements.txt
conda run -n clipper pytest -q
```

## Run

```bash
# baseline (~45 min CPU for feature extraction, then cached)
conda run -n clipper python scripts/run_baseline.py

# probe directions + gamma sweep
conda run -n clipper python scripts/extract_train_features.py         # 30k train sample
conda run -n clipper python scripts/fit_probes.py
conda run -n clipper python scripts/run_probe_gamma_ablation.py

# CPAS-MLP (GPU-oriented)
conda run -n clipper python scripts/extract_train_features.py --all   # full mining pool
conda run -n clipper python scripts/train_cpas.py --seed 0 --out runs/mlp_s0.pt
conda run -n clipper python scripts/run_cpas_ablation.py "CPAS-MLP=runs/mlp_s0.pt"

# current method: attribute head, then attribute-space retrieval
conda run -n clipper python scripts/fit_attribute_head.py
conda run -n clipper python scripts/run_attribute_retrieval.py --head results/attribute_head.pt

# exclusion re-rank on the cosine pipeline (lambda swept on validation)
conda run -n clipper python scripts/run_exclusion_rerank.py --checkpoint runs/mlp_s0.pt

```

## Results

MEAN over the 14 benchmark queries, full test-split database:

| method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Prompt arithmetic baseline (γ = 1) | 0.023 | 0.071 | 0.106 |
| Probe-direction composition (γ = 0.6) | 0.051 | 0.144 | 0.210 |
| CPAS-MLP (best cosine-space method) | 0.066 | 0.182 | 0.267 |
| **Attribute-space scoring, linear probe** | 0.116 | 0.336 | **0.465** |
| **Attribute-space scoring, MLP head** | — | — | **0.482** |
| *oracle attribute codes (ceiling)* | *1.000* | *1.000* | *1.000* |

The jump comes from ranking by the criterion the ground truth is defined by rather
than by cosine similarity to a composed vector — see `docs/method.md` §1 and §7.
Attribute-prediction accuracy is now the only lever: +0.003 bit accuracy bought
+0.022 R@10.

Per-query tables in `results/`. Absolute numbers shift slightly with the probe refit, so
compare against the fixed-rule row recomputed by the same run — see
`docs/method.md` §8.

## Gotchas

- Ground truth is **not** semantic similarity: assignment §3.1.1 defines a correct
  answer as one satisfying the constraints with Hamming distance ≤ 2 to the reference
  on the non-queried attributes. Rank by that, not by cosine.
- Ground-truth keys are **dataset indices**, not filenames: use `celeba[int(key)]`.
- The test split is the retrieval database and is never trained or selected on;
  training mines triplets from the train split, model selection uses a held-out val
  benchmark.
