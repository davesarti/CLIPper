# Clipper — Vanilla CLIP Baseline

Zero-shot baseline for the DL 2026 assignment (compositional image retrieval on
CelebA). Frozen CLIP ViT-B/32, naive latent arithmetic
`q = normalize(v_ref + Σt⁺ − Σt⁻)`, cosine ranking over the 19,962 test images.
No SVD, no training — this is the lower bound our fusion module has to beat.

## Layout

- `src/` — data loading, prompts + `compose()`/`rank()`, metrics, CLIP wrapper with feature cache; `attributes.py` (zero-shot attribute classifier) and `caption.py` (query bit-flips + caption rendering) for the attribute-caption method
- `scripts/` — `smoke_test.py` (quick check), `run_baseline.py` (full run, ~45 min CPU once, then cached), `run_attribute_probe.py` (calibrate attribute thresholds), `run_attribute_caption.py` (predict → flip → caption → retrieve pipeline)
- `baseline.ipynb` — executed notebook, seed of the final deliverable
- `tests/` — 23 pytest tests, model-free

## Setup & run

Put CelebA under `./celeba/` and `celeba_evaluation.json` at the repo root, then:

```bash
conda create -n clipper python=3.11 -y
conda run -n clipper pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision  # CPU build; skip flag on GPU
conda run -n clipper pip install -r requirements.txt

conda run -n clipper pytest -q                        # tests
conda run -n clipper python scripts/run_baseline.py   # full baseline (~45 min CPU, then cached)
```

## Results

**MEAN R@1 / R@5 / R@10 = 0.023 / 0.071 / 0.106** (unweighted macro-average over
the 14 queries; ~13× above chance). Full table in `results/baseline_results.csv`.
Positive single-attribute queries work best (`+Smiling` R@10 = 0.26); negation-only
queries collapse (`-Male, -Mustache` = 0) — that's the gap the fusion module targets.

## Gotchas

- Ground-truth keys are **dataset indices**, not filenames: use `celeba[int(key)]`.
- To extend: replace `compose()` in `src/retrieval.py` — everything else stays.
