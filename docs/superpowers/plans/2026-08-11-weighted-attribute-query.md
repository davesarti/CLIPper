# Weighted Attribute-Space Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the attribute-space query carry real-valued per-attribute weights — from the reference's own confidence (rung 1) and from measured predictor reliability (rung 2) — without training anything.

**Architecture:** The score's first term is already an inner product against a 40-dimensional query vector whose entries are ±1. Two independent knobs generalise it: the reference bit `r_a` becomes a probability instead of a threshold, and a per-attribute weight `w_a` multiplies each term. Both default to today's values, so the current method is a corner of the new space rather than a neighbour of it. `λ` and the cosine term are untouched, so CPAS-MLP and the constraint penalty keep working unchanged.

**Tech Stack:** Python 3.11, PyTorch, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-11-weighted-attribute-query-design.md`

## Global Constraints

- **NEVER run `git commit` or `git push`.** This is a standing user instruction and it overrides the commit steps that normally end a task in this plan format. Finish the file changes, run the tests, report what changed, and stop. Do not ask for permission to commit either — the answer is standing.
- **Default behaviour must stay bit-identical.** With `weights=None` and both new flags off, every code path must produce exactly what it produces today. This is what makes the ablation grid attributable.
- **Tests are model-free.** No CelebA, no CLIP weights, no network. The existing 97 tests run in ~20 s on CPU and the new ones must keep that property.
- **Docstrings explain *why*, not *what*.** Match the surrounding style in `src/` — the existing modules justify their choices in prose, and a reviewer will expect the same.
- **Weights are `(A,)`, full length**, indexed by the same `rows` the Hamming term already sums over. Never a pre-sliced vector.
- The database-side code stays hard: `constraint_violation` needs a yes/no decision. Only the Hamming term goes soft.

**Running tests:**

```bash
python -m pytest tests/test_attribute_retrieval.py -v
```

If the `clipper` conda env exists on the machine, prefer `conda run -n clipper pytest -q` as the README specifies.

---

## File Structure

| file | responsibility | change |
|---|---|---|
| `src/attribute_retrieval.py` | target codes, expected Hamming, the score | add optional `weights` to `expected_hamming` and `attribute_scores` |
| `src/attribute_head.py` | the attribute predictor and its held-out measurements | add `attribute_reliability` and `reliability_weights` |
| `scripts/run_attribute_retrieval.py` | the benchmark runner | two flags, weights computed once per predictor, passed identically to sweep and benchmark |
| `tests/test_attribute_retrieval.py` | contracts for both modules (it already covers `attribute_head.py`) | nine new tests |

`rank_by_attributes` needs no change: it forwards `**kwargs` to `attribute_scores` already.

---

## Task 1: Weighted expected Hamming

**Files:**
- Modify: `src/attribute_retrieval.py:50-69` (`expected_hamming`), `src/attribute_retrieval.py:86-112` (`attribute_scores`)
- Test: `tests/test_attribute_retrieval.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `expected_hamming(db_probs, ref_code, rows, weights=None) -> (N, R)` and `attribute_scores(..., weights=None) -> (R, N)`. `weights` is `torch.Tensor | None` of shape `(A,)`. Task 4 passes it a tensor from `reliability_weights`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_attribute_retrieval.py`:

```python
def test_unit_weights_match_the_unweighted_distance():
    db = torch.rand(4, 5)
    ref = torch.rand(2, 5) > 0.5
    rows = [0, 2, 4]
    plain = expected_hamming(db, ref, rows)
    weighted = expected_hamming(db, ref, rows, weights=torch.ones(5))
    assert torch.allclose(plain, weighted)


def test_a_zero_weight_removes_the_attribute_from_the_distance():
    ref = torch.zeros(1, 3, dtype=torch.bool)
    w = torch.tensor([1.0, 0.0, 1.0])
    quiet = expected_hamming(torch.tensor([[0.2, 0.1, 0.3]]), ref, [0, 1, 2], weights=w)
    loud = expected_hamming(torch.tensor([[0.2, 0.9, 0.3]]), ref, [0, 1, 2], weights=w)
    assert float(quiet[0, 0]) == pytest.approx(float(loud[0, 0]))


def test_doubling_a_weight_doubles_that_attributes_contribution():
    db = torch.tensor([[1.0]])
    ref = torch.zeros(1, 1, dtype=torch.bool)
    single = expected_hamming(db, ref, [0], weights=torch.tensor([1.0]))
    double = expected_hamming(db, ref, [0], weights=torch.tensor([2.0]))
    assert float(double[0, 0]) == pytest.approx(2 * float(single[0, 0]))


def test_attribute_scores_forwards_the_weights():
    # Attribute 1 is zeroed, so the two candidates - which differ only there -
    # must score identically.
    probs = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    code = probs > 0.5
    ref = torch.tensor([[True, True]])
    scores = attribute_scores(probs, code, ref, [], [], lam_constraint=0.0,
                              weights=torch.tensor([1.0, 0.0]))
    assert float(scores[0, 0]) == pytest.approx(float(scores[0, 1]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_attribute_retrieval.py -k "weight" -v`

Expected: FAIL — `TypeError: expected_hamming() got an unexpected keyword argument 'weights'`

- [ ] **Step 3: Add the parameter to `expected_hamming`**

Replace `src/attribute_retrieval.py:50-69` with:

```python
def expected_hamming(
    db_probs: torch.Tensor,
    ref_code: torch.Tensor,
    rows: list[int],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """(N, R) expected number of attributes in `rows` where a database image
    disagrees with each reference code.

    E[disagreements] = sum_a  w_a [ p_a (1 - r_a) + (1 - p_a) r_a ], computed as
    two matrix products so the whole database is scored for a block of
    references at once.

    db_probs: (N, A) predicted probabilities; ref_code: (R, A) bool, or float
    probabilities to keep the reference's own uncertainty in the distance.
    weights: optional (A,) per-attribute weight indexed by the same `rows`.
    None means a uniform weight of 1 and reproduces the unweighted distance
    exactly, which is what makes the weighting separately ablatable.
    """
    if not rows:
        return torch.zeros(db_probs.shape[0], ref_code.shape[0],
                           device=db_probs.device)
    p = db_probs[:, rows]
    r = ref_code[:, rows].to(p.dtype)
    if weights is None:
        return p @ (1 - r).T + (1 - p) @ r.T
    # w - pw is w * (1 - p): scaling the candidate side keeps both matmuls.
    w = weights[rows].to(p)
    pw = p * w
    return pw @ (1 - r).T + (w - pw) @ r.T
```

- [ ] **Step 4: Add the parameter to `attribute_scores`**

In `src/attribute_retrieval.py`, change the signature at line 86-95 to add `weights` as the final keyword argument:

```python
def attribute_scores(
    db_probs: torch.Tensor,
    db_code: torch.Tensor,
    ref_code: torch.Tensor,
    pos_rows: list[int],
    neg_rows: list[int],
    lam_constraint: float = 100.0,
    cosine: torch.Tensor | None = None,
    w_cos: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
```

and change the body's first scoring line from

```python
    score = -expected_hamming(db_probs, ref_code, others)          # (N, R)
```

to

```python
    score = -expected_hamming(db_probs, ref_code, others, weights)  # (N, R)
```

Then extend the docstring's parameter list with:

```
    weights: optional (A,) per-attribute weight for the Hamming term. The
        constraint term is deliberately unweighted - it is a conjunctive
        condition, not a distance.
```

- [ ] **Step 5: Run the new tests**

Run: `python -m pytest tests/test_attribute_retrieval.py -k "weight" -v`

Expected: PASS, 4 tests.

- [ ] **Step 6: Run the whole file to prove nothing regressed**

Run: `python -m pytest tests/test_attribute_retrieval.py -v`

Expected: PASS, all tests. The pre-existing ones must still pass unchanged — they exercise the `weights=None` path, which is the bit-identical guarantee.

---

## Task 2: Lock the soft-reference generalisation

No source change. These tests characterise behaviour that already exists and that rung 1 depends on; locking it before Task 4 relies on it means a future edit to `target_code` or `expected_hamming` cannot silently break the soft path.

**Files:**
- Test: `tests/test_attribute_retrieval.py`

**Interfaces:**
- Consumes: `expected_hamming` and `target_code` as they stand after Task 1.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the tests**

Append to `tests/test_attribute_retrieval.py`:

```python
def test_a_float_reference_of_hard_bits_reproduces_the_thresholded_code():
    # The soft reference is a strict generalisation: fed 0/1 it must be exact.
    db = torch.rand(4, 3)
    hard = torch.tensor([[True, False, True]])
    assert torch.allclose(expected_hamming(db, hard, [0, 1, 2]),
                          expected_hamming(db, hard.float(), [0, 1, 2]))


def test_target_code_keeps_probabilities_and_forces_the_queried_bits():
    soft = torch.tensor([[0.90, 0.51, 0.20, 0.80]])
    out = target_code(soft, pos_rows=[2], neg_rows=[0])
    assert out[0].tolist() == pytest.approx([0.0, 0.51, 1.0, 0.80])
    assert float(soft[0, 0]) == pytest.approx(0.90)   # input untouched


def test_an_uncertain_reference_bit_cannot_reorder_candidates():
    # At p = 0.5 the attribute contributes the same amount to every candidate,
    # so it drops out of the ranking instead of deciding it on noise.
    probs = torch.tensor([[0.0], [1.0]])
    ref = torch.tensor([[0.5]])
    d = expected_hamming(probs, ref, [0])
    assert float(d[0, 0]) == pytest.approx(float(d[1, 0]))
```

- [ ] **Step 2: Run them**

Run: `python -m pytest tests/test_attribute_retrieval.py -k "reference or target_code" -v`

Expected: PASS. These describe existing behaviour, so they should pass immediately. **If any fails, stop and report it** — it means the soft-reference premise of the spec is wrong and Task 4 must not proceed.

---

## Task 3: Per-attribute reliability

**Files:**
- Modify: `src/attribute_head.py` — add two functions after `tune_thresholds` (which ends at line 77)
- Test: `tests/test_attribute_retrieval.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `attribute_reliability(pred, labels) -> (A,) float` (raw Youden's J) and `reliability_weights(pred, labels) -> (A,) float` (clamped at 0, mean 1). Both take `(N, A)` **bool** tensors. Task 4 imports `reliability_weights`.

- [ ] **Step 1: Write the failing tests**

Add `attribute_reliability` and `reliability_weights` to the `src.attribute_head` import block at the top of `tests/test_attribute_retrieval.py`, then append:

```python
def test_reliability_is_one_for_a_perfect_predictor():
    labels = torch.tensor([[True, False], [False, True], [True, True]])
    assert attribute_reliability(labels, labels).tolist() == pytest.approx([1.0, 1.0])


def test_reliability_is_zero_for_a_majority_class_predictor():
    # Attribute 0 is 20% positive, so answering "no" every time scores 80%
    # accuracy. It detects nothing and must be worth nothing - this is the
    # trap that rules accuracy out as a weight on CelebA.
    labels = torch.zeros(10, 1, dtype=torch.bool)
    labels[:2, 0] = True
    pred = torch.zeros(10, 1, dtype=torch.bool)
    assert float(attribute_reliability(pred, labels)[0]) == pytest.approx(0.0)


def test_reliability_weights_are_non_negative_and_average_to_one():
    labels = torch.tensor([[True, False], [False, True], [True, True], [False, False]])
    pred = torch.tensor([[True, True], [False, False], [True, False], [False, True]])
    w = reliability_weights(pred, labels)
    assert float(w.mean()) == pytest.approx(1.0)
    assert bool((w >= 0).all())        # an anti-correlated attribute is clamped, not negated
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_attribute_retrieval.py -k "reliability" -v`

Expected: FAIL at collection with `ImportError: cannot import name 'attribute_reliability' from 'src.attribute_head'`

- [ ] **Step 3: Implement both functions**

Append to `src/attribute_head.py`, after `tune_thresholds`:

```python
@torch.no_grad()
def attribute_reliability(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """(A,) Youden's J per attribute: sensitivity + specificity - 1.

    Zero for a predictor that answers one class regardless of its input, one for
    a perfect one. Per-attribute *accuracy* cannot serve here: CelebA attributes
    are heavily imbalanced, so always answering the majority class scores 88% on
    Wearing_Necklace (12% positive rate) while detecting nothing at all, and
    would earn a large weight for an attribute the predictor is blind to.

    pred / labels: (N, A) bool, thresholded predictions and true labels, so the
    same function serves the linear probe and the MLP head. An attribute with no
    positives or no negatives in `labels` returns 0: its rate is undefined, and
    0 is exactly the "carries no information" weight.
    """
    p, y = pred.bool(), labels.bool()
    pos = y.sum(0).float()
    neg = (~y).sum(0).float()
    sensitivity = (p & y).sum(0).float() / pos.clamp(min=1.0)
    specificity = ((~p) & (~y)).sum(0).float() / neg.clamp(min=1.0)
    j = sensitivity + specificity - 1.0
    return torch.where((pos > 0) & (neg > 0), j, torch.zeros_like(j))


@torch.no_grad()
def reliability_weights(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """(A,) non-negative per-attribute weights averaging 1, from held-out data.

    Youden's J clamped at zero and rescaled. The clamp is not cosmetic: a
    negative weight would invert the target bit, so the score would actively ask
    for the opposite of what the reference has, and on held-out data a negative
    J is noise rather than an anti-correlated attribute worth exploiting.

    The rescaling keeps the Hamming term on the scale lam_constraint was swept
    against, so a run with weights stays comparable to one without.
    """
    j = attribute_reliability(pred, labels).clamp(min=0.0)
    mean = j.mean()
    return j / mean if float(mean) > 0 else torch.ones_like(j)
```

- [ ] **Step 4: Run the new tests**

Run: `python -m pytest tests/test_attribute_retrieval.py -k "reliability" -v`

Expected: PASS, 3 tests.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`

Expected: PASS, all tests (97 existing + 10 new).

---

## Task 4: Wire both knobs into the benchmark runner

**Files:**
- Modify: `scripts/run_attribute_retrieval.py`

**Interfaces:**
- Consumes: `expected_hamming`/`attribute_scores` `weights` from Task 1, `reliability_weights` from Task 3.
- Produces: `results/attribute_retrieval[_soft][_rel].csv` with `soft_reference` and `reliability_weights` columns.

- [ ] **Step 1: Import `reliability_weights`**

Change line 25 from

```python
from src.attribute_head import load_attribute_head
```

to

```python
from src.attribute_head import load_attribute_head, reliability_weights
```

- [ ] **Step 2: Add the two flags**

After the `--checkpoint` argument block (which ends around line 63), add:

```python
parser.add_argument("--soft-reference", action="store_true",
                    help="rung 1: build the target code from the reference's "
                         "predicted probabilities instead of its thresholded "
                         "bits, so each non-queried attribute is weighted by "
                         "the predictor's confidence in it. Queried bits are "
                         "still forced to exactly 0/1")
parser.add_argument("--reliability-weights", action="store_true",
                    help="rung 2: weight each attribute in the Hamming term by "
                         "Youden's J measured on the validation slice, so an "
                         "attribute the predictor cannot detect stops voting")
```

- [ ] **Step 3: Extend the output filename**

Replace the `if args.out is None:` block with:

```python
if args.out is None:
    stem = "attribute_retrieval_no_cosine" if args.no_cosine \
        else "attribute_retrieval_cpas" if args.checkpoint \
        else "attribute_retrieval"
    if args.soft_reference:
        stem += "_soft"
    if args.reliability_weights:
        stem += "_rel"
    args.out = REPO_ROOT / "results" / f"{stem}.csv"
```

- [ ] **Step 4: Keep each predictor's threshold**

The sweep currently thresholds the validation probabilities at a flat 0.5 (line 177) while the test benchmark uses the MLP head's tuned thresholds — so the reliability of a *different* predictor would be measured than the one being scored. Fix it minimally by recording the threshold alongside each predictor.

Replace the predictors block (lines 115-129) with:

```python
# ------------------------------------------------------------ predictors
predictors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
thresholds: dict[str, object] = {}
probe_probs = torch.sigmoid(features @ W.T + B)
predictors["linear probe"] = (probe_probs, probe_probs > 0.5)
thresholds["linear probe"] = 0.5
if args.head.is_file():
    head, saved = load_attribute_head(args.head)
    probs = torch.sigmoid(head(features))
    th = saved.get("thresholds")
    thresholds["MLP head"] = 0.5 if th is None else th
    predictors["MLP head"] = (probs, probs > thresholds["MLP head"])
    print(f"loaded {args.head.name}: val bit accuracy "
          f"{saved.get('val_bit_accuracy', float('nan')):.4f}, "
          f"trained on {saved.get('pool', '?')}")
else:
    print(f"no attribute head at {args.head}: linear probe only "
          f"(train one with scripts/fit_attribute_head.py)")
```

- [ ] **Step 5: Add the reference-code helper**

Immediately after the predictors block, add:

```python
def reference_code(probs, code, refs, pos_rows, neg_rows):
    """The target code for a block of references.

    Under --soft-reference the reference contributes its predicted
    probabilities rather than its thresholded bits, which makes each
    non-queried attribute's weight in the Hamming term equal to the predictor's
    confidence in it. target_code forces the queried bits to exactly 0.0 / 1.0
    either way: the query is certain even when the predictor is not.
    """
    source = probs if args.soft_reference else code
    return target_code(source[refs], pos_rows, neg_rows)
```

- [ ] **Step 6: Thread the weights through `val_recall`**

Replace lines 147-169 (`val_recall`) with:

```python
def val_recall(probs, code, lam, w_cos, weights) -> float:
    """Mean R@10 over the val pool under the S3.1.1 ground-truth rule."""
    hits = total = 0
    for pos, neg in queries:
        pr = [attr_index[a] for a in pos]; nr = [attr_index[a] for a in neg]
        others = [a for a in range(len(attributes)) if a not in set(pr) | set(nr)]
        sat = torch.ones(val_features.shape[0], dtype=torch.bool)
        if pr: sat &= val_labels[:, pr].all(dim=1)
        if nr: sat &= ~val_labels[:, nr].any(dim=1)
        rest = val_labels[:, others]
        cos = cosine_term(val_features, val_refs, pr, nr) if w_cos else None
        order = rank_by_attributes(
            probs, code, reference_code(probs, code, val_refs, pr, nr), pr, nr,
            exclude=val_refs.tolist(), lam_constraint=lam,
            cosine=cos, w_cos=w_cos, weights=weights,
        )
        for row, r in enumerate(val_refs.tolist()):
            gt = sat & ((rest != rest[r]).sum(dim=1) <= MAX_HAMMING)
            gt[r] = False
            if int(gt.sum()) < 3:
                continue
            hits += bool(gt[order[row, :10]].any()); total += 1
    return hits / max(total, 1)
```

- [ ] **Step 7: Compute the weights once per predictor, in the sweep**

Replace the sweep block (lines 172-190) with:

```python
best: dict[str, tuple[float, float]] = {}
weights_for: dict[str, torch.Tensor | None] = {}
sweep = []
for name, (probs, code) in predictors.items():
    val_probs = torch.sigmoid(val_features @ W.T + B) if name == "linear probe" \
        else torch.sigmoid(head(val_features))
    val_code = val_probs > thresholds[name]
    # Measured once and reused on the test benchmark: measuring reliability
    # against one configuration and reporting another would select weights for
    # a model that is not the one scored.
    weights_for[name] = (reliability_weights(val_code, val_labels)
                         if args.reliability_weights else None)
    if weights_for[name] is not None:
        w_vec = weights_for[name]
        worst = int(w_vec.argmin()); bestest = int(w_vec.argmax())
        print(f"  reliability weights: {float(w_vec.min()):.2f} "
              f"({attributes[worst]}) to {float(w_vec.max()):.2f} "
              f"({attributes[bestest]}), {int((w_vec == 0).sum())} at zero")
    scores = {}
    for lam in LAM_GRID:
        for w in cos_grid:
            r10 = val_recall(val_probs, val_code, lam, w, weights_for[name])
            scores[(lam, w)] = r10
            sweep.append({"predictor": name, "lam_constraint": lam,
                          "w_cos": w, "val_R@10": r10})
            print(f"[val] {name:14s} lam {lam:<6} w_cos {w:<5} R@10 {r10:.4f}")
    best[name] = max(scores, key=scores.get)
    print(f"  -> best on val: lam {best[name][0]}, w_cos {best[name][1]} "
          f"(R@10 {scores[best[name]]:.4f})\n")
pd.DataFrame(sweep).to_csv(args.out.with_name(args.out.stem + "_sweep.csv"),
                           index=False)
```

- [ ] **Step 8: Use the same knobs on the test benchmark**

Replace the benchmark loop body (lines 193-213) with:

```python
rows, per_query = [], []
for name, (probs, code) in predictors.items():
    lam, w_cos = best[name]
    weights = weights_for[name]
    metrics = []
    for entry, (pos, neg) in zip(annotations, queries):
        pr = [attr_index[a] for a in pos]; nr = [attr_index[a] for a in neg]
        sources = [int(k) for k in entry["ground_truth"].keys()]
        src = torch.tensor(sources)
        cos = cosine_term(features, src, pr, nr) if w_cos else None
        order = rank_by_attributes(
            probs, code, reference_code(probs, code, src, pr, nr), pr, nr,
            exclude=sources, lam_constraint=lam, cosine=cos, w_cos=w_cos,
            weights=weights,
        )
        metrics.append(_query_row(entry, order, sources, labels, pr, nr))
    df = _with_mean_row(metrics)
    per_query.append(df.assign(predictor=name))
    rows.append({"method": f"attribute space ({name})",
                 "soft_reference": args.soft_reference,
                 "reliability_weights": args.reliability_weights,
                 "lam_constraint": lam, "w_cos": w_cos}
                | df[df["query"] == "MEAN"].iloc[0][METRIC_COLS].to_dict()
                | {"neg_R@10": negation_subset(df)})
    print(f"{name:14s} R@10 {rows[-1]['R@10']:.4f}  V@10 {rows[-1]['V@10']:.4f}")
```

- [ ] **Step 9: Check the script parses and the defaults are unchanged**

Run: `python scripts/run_attribute_retrieval.py --help`

Expected: the help text lists `--soft-reference` and `--reliability-weights`. No import errors.

- [ ] **Step 10: Run the full test suite one last time**

Run: `python -m pytest tests/ -q`

Expected: PASS, all tests.

---

## Producing the ablation grid

The four configurations, on a machine that has the features and the probe weights. Each writes its own CSV, so nothing overwrites anything:

```bash
python scripts/run_attribute_retrieval.py
```

```bash
python scripts/run_attribute_retrieval.py --soft-reference
```

```bash
python scripts/run_attribute_retrieval.py --reliability-weights
```

```bash
python scripts/run_attribute_retrieval.py --soft-reference --reliability-weights
```

Each run covers both predictors, so the grid is eight rows. The first is the current method recomputed within-run and is the only row the others should be compared against — absolute numbers shift with a probe refit.

To check that the `(λ, w_cos)` selection is stable rather than an artefact of which 200 validation references were drawn, repeat the fourth command with `--seed 1` and `--seed 2` and confirm the selected pair does not move. Do **not** average the results across seeds: nothing here is trained, so given fixed features the rows are deterministic.

---

## Self-review notes

Checked against the spec:

- §2 (both knobs, defaults preserved) → Tasks 1, 2, 4
- §3 (Youden's J, clamp, mean-1 normalisation) → Task 3
- §4 (file changes) → all four tasks; the `tests/test_attribute_head.py` row in the spec was wrong and has been corrected there — `attribute_head.py` is already tested from `tests/test_attribute_retrieval.py`
- §5 (ablation grid, per-configuration sweep, seed stability) → the "Producing the ablation grid" section
- §6 (nine contracts) → Task 1 covers 1–4, Task 2 covers 5–6, Task 3 covers 7–9
- §7 (out of scope) → no task touches `λ`, the cosine term, or any training

One deliberate addition beyond the spec: Task 4 Step 4 records each predictor's threshold, because the sweep thresholded validation probabilities at a flat 0.5 while the benchmark used the MLP head's tuned thresholds. Measuring reliability under one thresholding rule and scoring under another is the same class of drift the project has already been bitten by three times. It shifts the MLP head's baseline row slightly, which is acceptable because every row is recomputed within-run.
