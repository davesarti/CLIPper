"""Predicting the 40-bit attribute code of an image from frozen CLIP features.

Under the assignment's ground-truth rule (S3.1.1) a retrieved image is correct
iff it satisfies the query's constraints and its remaining attributes are within
Hamming distance 2 of the reference's. Both halves are statements about
attribute *codes*, so retrieval quality is bounded by how accurately those codes
can be predicted - with true labels the benchmark is solved exactly (R@10 = 1.0).

That makes per-bit accuracy the quantity to optimize, and it is not the same
quantity the linear probes were fit for. A probe is fit to be a good edit
*direction*; here it is used as a classifier, and correctness needs the whole
38-bit code to land inside a radius-2 ball. Errors compound: at 0.909 per-bit
accuracy the expected code is ~3.5 bits wrong, already outside the ball.

Measured exchange rate on the 14-query benchmark: +0.003 bit accuracy bought
+0.022 R@10, roughly 7x amplification. Small accuracy gains are worth real
effort here.
"""

import torch
from torch import nn


class AttributeMLP(nn.Module):
    """Two hidden layers on frozen CLIP features, one logit per attribute.

    Deliberately small: the features are frozen and the training signal is 40
    binary labels, so capacity is not the constraint. Widths of 512-2048 all
    land within 0.001 bit accuracy of each other.
    """

    def __init__(self, d: int = 512, hidden: int = 1024, attributes: int = 40,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.config = {"d": d, "hidden": hidden, "attributes": attributes,
                       "dropout": dropout}
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, attributes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(N, D) features -> (N, A) logits."""
        return self.net(features)


@torch.no_grad()
def bit_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                 thresholds: torch.Tensor | None = None) -> float:
    """Fraction of individual attribute bits predicted correctly."""
    probs = torch.sigmoid(logits)
    pred = probs > (0.5 if thresholds is None else thresholds)
    return float((pred == labels.bool()).float().mean())


@torch.no_grad()
def tune_thresholds(logits: torch.Tensor, labels: torch.Tensor,
                    steps: int = 91) -> torch.Tensor:
    """Per-attribute decision threshold maximizing bit accuracy on held-out data.

    0.5 is only optimal for a calibrated probe on a balanced attribute, and most
    CelebA attributes are far from balanced. Must be fit on data the predictor
    was not trained on, or it overfits the training split's base rates.

    Returns (A,) thresholds.
    """
    probs = torch.sigmoid(logits)
    grid = torch.linspace(0.05, 0.95, steps)
    out = torch.empty(probs.shape[1])
    for j in range(probs.shape[1]):
        acc = torch.tensor([
            float(((probs[:, j] > t) == labels[:, j].bool()).float().mean())
            for t in grid
        ])
        out[j] = grid[int(acc.argmax())]
    return out


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


def fit_attribute_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    hidden: int = 1024,
    dropout: float = 0.2,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: str = "cpu",
    log: bool = True,
) -> tuple[AttributeMLP, float]:
    """Train an AttributeMLP, keeping the epoch with the best val bit accuracy.

    Selection is on held-out bit accuracy rather than loss: the retrieval score
    consumes the thresholded code, so accuracy is the quantity that transfers.

    Returns (model with the best weights loaded, that val bit accuracy).
    """
    model = AttributeMLP(d=features.shape[1], hidden=hidden,
                         attributes=labels.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    y = labels.float().to(device)
    x = features.to(device)
    vx, vy = val_features.to(device), val_labels.to(device)

    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], batch_size):
            rows = perm[start : start + batch_size]
            opt.zero_grad()
            loss_fn(model(x[rows]), y[rows]).backward()
            opt.step()
        sched.step()
        model.eval()
        acc = bit_accuracy(model(vx), vy)
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if log and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:3d}  val bit accuracy {acc:.4f} "
                  f"(best {best:.4f})", flush=True)
    model.load_state_dict(best_state)
    return model, best


def load_attribute_head(path, device: str = "cpu") -> tuple[AttributeMLP, dict]:
    """Load a head saved by scripts/fit_attribute_head.py."""
    saved = torch.load(path, map_location=device, weights_only=True)
    model = AttributeMLP(**saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved
