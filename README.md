# Clipper — compositional image retrieval on CelebA

DL 2026 assignment (`Project assignment - V1.2.pdf`). Given a reference image and a
query like `"+Smiling, -Blond_Hair"`, retrieve images from the 19,962-image CelebA test
split that keep the reference's identity, have every `+` attribute and lack every `−`
one. CLIP ViT-B/32 stays frozen throughout; only the query embedding is built.

- **Baseline**: latent arithmetic over CLIP *text* prompts,
  `q = normalize(v_ref + Σt⁺ − Σt⁻)`. No training.
- **Current method**: CPAS-MLP — the same composition over learned *probe* directions,
  with the reference weight, per-attribute step size and direction bend predicted per
  query by a 0.50 M-parameter MLP. See `docs/method-proposal-cpas.md`.
- **How we got there**, including the methods that were tried and dropped:
  `docs/method-history.md`.

## Layout

- `src/` — `data.py`, `features.py` (CLIP wrapper + feature cache), `retrieval.py`
  (prompts, `compose()`/`rank()`), `probes.py` (linear attribute probes and the fixed
  composition rule), `evaluation.py` (metrics, benchmarks, probe-drift diagnostic),
  `steering.py` (shared composition + padding), `mining.py` (triplet mining),
  `training.py` (InfoNCE loop), `cpas_mlp.py` (the model),
  `rerank.py` (exclusion penalty on top of the cosine score)
- `scripts/` — `smoke_test.py`, `run_baseline.py`, `extract_train_features.py`,
  `fit_probes.py`, `run_probe_accuracy.py`, `run_probe_gamma_ablation.py`,
  `train_cpas.py`, `run_cpas_ablation.py`, `run_exclusion_rerank.py`
- `docs/` — method proposals and history
- `results/` — benchmark CSVs and the trained checkpoint
- `tests/` — 95 pytest tests, model-free

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

# exclusion re-rank: lambda swept on the held-out val benchmark, then the
# 2x2 (combiner x re-rank) test table and the ablations
conda run -n clipper python scripts/run_exclusion_rerank.py --checkpoint runs/mlp_s0.pt

# negation-aware mining (off by default; each flag is one ablation row)
conda run -n clipper python scripts/train_cpas.py --seed 0 \
    --neg-fraction 0.5 --n-violations 8 --lambda-violation 0.5 \
    --correlated-pair-prob 0.3 --out runs/negmine_s0.pt
```

## Results

MEAN over the 14 benchmark queries, full test-split database:

| method | R@1 | R@5 | R@10 |
|---|---|---|---|
| Prompt arithmetic baseline (γ = 1) | 0.023 | 0.071 | 0.106 |
| Probe-direction composition (γ = 0.6) | 0.051 | 0.144 | 0.210 |
| **CPAS-MLP** | 0.066 | 0.182 | **0.267** |

Per-query tables in `results/`. Absolute numbers shift slightly with the probe refit, so
compare against the fixed-rule row recomputed by the same run — see
`docs/method-proposal-cpas.md` §3.

## Gotchas

- Ground-truth keys are **dataset indices**, not filenames: use `celeba[int(key)]`.
- The test split is the retrieval database and is never trained or selected on;
  training mines triplets from the train split, model selection uses a held-out val
  benchmark.
