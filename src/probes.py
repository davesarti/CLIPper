"""Linear attribute probes on frozen CLIP features, and how to score them.

One logistic-regression probe per attribute, fit on train-split features.
The probe weight vectors live in the visual embedding space, so they replace
the prompt text embeddings in the compose step: attribute directions learned
where the database lives, no modality gap to cross.

Measuring a probe: fit on the cached train-split sample, score on the held-out
valid split with `score_attributes`, which returns ROC AUC and average
precision per attribute. Report both. AUC flatters a rare attribute, because
correctly ranking the many easy negatives is most of the score; AP is measured
against a baseline of the attribute's own positive rate, so it shows whether
the top of the ranking is clean. `Wearing_Necklace` scores 0.826 AUC but 0.372
AP at a 12% positive rate.

`scripts/run_probe_accuracy.py` writes these numbers to
`results/probe_accuracy.csv`. The similarly named `attribute_accuracy.csv` under
`results/archive/` belongs to the abandoned zero-shot text-prompt method and
measures prompt classification, not probes.
"""

from pathlib import Path

import torch

PROBE_FILE = "probe_weights.pt"


def load_raw_probes(repo_root: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load saved probes as (raw weights, biases, attribute names).

    Looks in results/ first, then features/ (the weights are a training
    artifact, so either location is valid depending on how they were produced).

    These are the weights the saved biases belong to, so this is what predicts
    a calibrated probability sigmoid(w.d + b) - see src/rerank.py. The
    composition wants `load_probes` instead.
    """
    for folder in ("results", "features"):
        path = repo_root / folder / PROBE_FILE
        if path.is_file():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            return saved["weights"], saved["biases"], saved["attributes"]
    raise FileNotFoundError(
        f"{PROBE_FILE} not found in {repo_root}/results or {repo_root}/features; "
        "run scripts/fit_probes.py first."
    )


def load_probes(repo_root: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load saved probes as (normalized directions, biases, attribute names).

    The directions are L2-normalized but the biases are not rescaled with them:
    the biases belong to the raw weights, and only the directions are used by
    the composition. Anything that needs sigmoid(w.d + b) must use
    `load_raw_probes`.
    """
    w, biases, attributes = load_raw_probes(repo_root)
    return w / w.norm(dim=1, keepdim=True), biases, attributes


def fit_linear_probes(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 2000,
    lr: float = 0.05,
    weight_decay: float = 0.0,
    class_balanced: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit A independent logistic regressions with full-batch Adam.

    The defaults are the recipe that produced results/probe_weights.pt and
    every number in results/ downstream of it, so they must not change:
    changing them silently invalidates the reported benchmarks. At those
    defaults the probes reach a macro-mean valid AUC of 0.929, from 0.731
    (Oval_Face) to 0.999 (Male). The two optional knobs allow a refit under
    regularization without disturbing the default recipe.

    weight_decay is Adam's coupled L2 (added to the gradient), not AdamW's
    decoupled form. class_balanced sets pos_weight = n_neg / n_pos per
    attribute, so a rare attribute's positives are not drowned by its
    negatives; an attribute with no positives gets weight 1, since dividing
    by zero there would poison every other attribute's gradient.

    features: (N, D) L2-normalized; labels: (N, A) 0/1.
    Returns weights (A, D) and biases (A,), detached.
    """
    n, d = features.shape
    a = labels.shape[1]
    w = torch.zeros(a, d, requires_grad=True)
    b = torch.zeros(a, requires_grad=True)
    y = labels.float()

    pos_weight = None
    if class_balanced:
        n_pos = y.sum(dim=0)
        pos_weight = torch.where(n_pos > 0, (n - n_pos) / n_pos.clamp(min=1.0),
                                 torch.ones_like(n_pos))

    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(features @ w.T + b, y)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def compose_probe(
    v_ref: torch.Tensor,
    pos_dirs: torch.Tensor,
    neg_dirs: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Composite query embedding from probe directions.

    q = normalize(gamma * v_ref + sum(pos_dirs) - sum(neg_dirs))

    pos_dirs / neg_dirs: (K, D) L2-normalized probe weight directions for the
    query's T+ / T- attributes. Same shape contract as retrieval.compose,
    with the text embeddings swapped for in-space attribute directions and an
    explicit identity weight gamma.
    """
    q = gamma * v_ref + pos_dirs.sum(dim=0) - neg_dirs.sum(dim=0)
    return q / q.norm()


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Threshold-free ranking quality (Mann-Whitney U / rank-sum form)."""
    order = scores.argsort()
    ranks = torch.empty(len(scores), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    pos = labels.bool()
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    u = ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the precision-recall curve, by the standard step-wise sum.

    AP = (1 / P) * sum over positives of precision-at-that-rank. Returns nan
    when `labels` has no positives, which is the only degenerate case.
    """
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = scores.argsort(descending=True)
    y = labels[order].to(torch.float64)
    ranks = torch.arange(1, len(y) + 1, dtype=torch.float64)
    precision_at_rank = y.cumsum(0) / ranks
    return float((precision_at_rank * y).sum() / n_pos)


def score_attributes(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[float], list[float]]:
    """Per-attribute (AUC, AP) from an (N, A) score matrix and (N, A) labels."""
    aucs, aps = [], []
    for j in range(labels.shape[1]):
        aucs.append(roc_auc(scores[:, j], labels[:, j]))
        aps.append(average_precision(scores[:, j], labels[:, j]))
    return aucs, aps
