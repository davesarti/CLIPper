"""Linear attribute probes on frozen CLIP features and probe-direction fusion.

One logistic-regression probe per attribute, fit on train-split features.
The probe weight vectors live in the visual embedding space, so they replace
the prompt text embeddings in the compose step: attribute directions learned
where the database lives, no modality gap to cross.
"""

import torch


def fit_linear_probes(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit A independent logistic regressions with full-batch Adam.

    features: (N, D) L2-normalized; labels: (N, A) 0/1.
    Returns weights (A, D) and biases (A,), detached.
    """
    n, d = features.shape
    a = labels.shape[1]
    w = torch.zeros(a, d, requires_grad=True)
    b = torch.zeros(a, requires_grad=True)
    y = labels.float()
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(features @ w.T + b, y)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def probe_scores(features: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, D) features -> (N, A) probe logits."""
    return features @ w.T + b


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
