# CLIPper: compositional image retrieval on CelebA

CLIPper retrieves images from the CelebA test split given a reference image and
attribute constraints such as `+Smiling, -Blond_Hair`. CLIP ViT-B/32 is kept
frozen. The delivered method predicts the 40 CelebA attributes from CLIP image
features, builds a soft target code, and ranks candidates with the assignment's
ground-truth criterion: expected Hamming distance on non-queried attributes plus
a penalty for violating queried constraints. CPAS-MLP supplies the composed
query embedding used by the small cosine term in the final score.

## Notebook

The primary reproducible artifact is [`Clipper_Report_out.ipynb`](Clipper_Report_out.ipynb).
It is the executed version of the report and contains the complete pipeline:

1. load CelebA and `celeba_evaluation.json`;
2. load or extract frozen CLIP features for train, valid and test;
3. fit the linear probes and the MLP attribute head;
4. train or load CPAS-MLP, selecting it on the held-out validation benchmark;
5. sweep the validation hyperparameters when requested;
6. evaluate the baseline, ablations and delivered method on the 14 mandatory
   test queries.

Open the notebook, restart the kernel, and run all cells from top to bottom. To
execute it without the Jupyter UI:

```bash
conda run -n clipper jupyter nbconvert \
  --to notebook --execute --inplace Clipper_Report_out.ipynb
```

The notebook uses `seed=0` for the train/validation split and sampling. Its
delivered configuration is:

```text
soft_reference = True
lambda_constraint = 4
w_cos = 1
head = results/attribute_head.pt
CPAS-MLP = results/mlp_final_s0.pt
probes = results/probe_weights.pt
```

`RUN_SWEEP` is `False` by default because the reported values already come
from the fixed validation sweep. Set it to `True` in the configuration cell to
recompute the $(lambda, w)$ sweep. The notebook creates missing model files;
the first complete run therefore takes considerably longer than a run using
the cached artifacts.

## Results

The figures below are from the last notebook execution. They average over the
14 benchmark queries and 33,052 query-reference pairs, using the complete
19,962-image CelebA test split. `V@10` is the fraction of top-10 results that
break a queried constraint.

| method | R@10 | V@10 |
|---|---:|---:|
| Zero-shot prompt arithmetic baseline | 0.106 | 0.834 |
| Attribute-space, linear probe, hard reference | 0.482 | 0.307 |
| Attribute-space, MLP head, hard reference | 0.503 | 0.304 |
| Attribute-space, MLP head, soft reference, no cosine | 0.530 | 0.312 |
| **Delivered: MLP head + soft reference + CPAS-MLP** | **0.535** | **0.306** |
| Oracle attribute codes | 1.000 | 0.000 |

The delivered score is selected on held-out data, not on the 14 test queries.
The validation sweep selects the plateau beginning at `lambda=4`; `w=1` is
kept in the delivered configuration because it preserves the CPAS query term,
although the difference between nearby cosine weights is within the observed
sampling noise. The valid bit accuracy is 0.9110 for the linear probe and
0.9149 for the MLP head.

The main conclusion is a change of scoring space, not just a better query
embedding: ranking directly by predicted attributes raises R@10 from 0.106 to
0.535. Results within about 0.02 R@10 should be treated as unresolved because
of benchmark sampling variability. See [`docs/method.md`](docs/method.md) for
the full criterion, ablations and limitations; [`docs/method-history.md`](docs/method-history.md)
records discarded approaches.

## Data and environment

Run commands from the repository root. Place the official CelebA directory at
`./celeba/` and keep `celeba_evaluation.json` at the repository root. The
dataset must contain the standard `img_align_celeba/` directory and annotation
files. The notebook downloads the CLIP text/model components through
Transformers on first use, so network access is needed unless the model cache
already exists.

```bash
conda create -n clipper python=3.11 -y
conda run -n clipper pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
conda run -n clipper pip install -r requirements-min.txt
```

The CPU installation is reproducible but slow for feature extraction and CPAS
training. Install the matching CUDA build of PyTorch first when a GPU is
available. `requirements.txt` is the captured environment from the report; use
it instead of `requirements-min.txt` only when that exact package snapshot is
required.

Before a full notebook run, the model-free test suite provides a cheap sanity
check:

```bash
conda run -n clipper pytest -q
```

## Script-only reproduction

The scripts mirror the notebook for users who prefer a command-line workflow.
The following sequence starts from the dataset and regenerates the main
artifacts. `--all` is important: it uses the complete train/valid split rather
than the optional 30k training sample.

```bash
# frozen CLIP features
conda run -n clipper python scripts/extract_train_features.py --split train --all
conda run -n clipper python scripts/extract_train_features.py --split valid --all

# probe weights and nonlinear attribute head
conda run -n clipper python scripts/fit_probes.py
conda run -n clipper python scripts/fit_attribute_head.py \
  --out results/attribute_head.pt

# CPAS-MLP; defaults match the notebook (seed 0, 5 warmup + 45 main epochs)
conda run -n clipper python scripts/train_cpas.py \
  --seed 0 --out results/mlp_final_s0.pt

# delivered attribute-space benchmark
conda run -n clipper python scripts/run_attribute_retrieval.py \
  --head results/attribute_head.pt \
  --checkpoint results/mlp_final_s0.pt \
  --soft-reference \
  --out results/attribute_retrieval_cpas_soft.csv

# zero-shot baseline for comparison
conda run -n clipper python scripts/run_baseline.py
```

The retrieval script performs its own validation sweep and writes the selected
test results plus per-query output under `results/`. `--no-cosine` reproduces
the attribute-only ablation. Refit probes before comparing new runs: probe
weights define both the attribute classifier and CPAS edit directions, so
changing them invalidates the shipped benchmark rows. Feature caches and model
checkpoints are deliberately kept out of the source-only workflow when they
are not already present; the scripts regenerate them in `features/`,
`results/` or the path supplied with `--out`.

## Repository layout

- `Clipper_Report_out.ipynb` - executed, notebook-first reproduction and report.
- `Clipper_Report.ipynb` - report notebook source.
- `src/` - data loading, feature extraction, probes, CPAS-MLP, scoring and evaluation.
- `scripts/` - command-line equivalents of the notebook stages.
- `features/` - cached CLIP features and shipped attribute/probe artifacts.
- `results/` - current benchmark tables; `results/archive/` contains superseded runs.
- `tests/` - model-free unit tests.

## Evaluation details

- The test split is used only as the retrieval database and final benchmark.
- CPAS-MLP is trained on the train split and selected on a held-out train slice.
- Attribute-head epochs and thresholds are selected on the valid split.
- Ground-truth keys are dataset indices, not filenames: use `celeba[int(key)]`.
- Ground is defined by queried attribute constraints and a maximum Hamming distance of 2 on the remaining
  attributes.
