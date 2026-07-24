"""Zero-shot CLIP attribute classification: prompts, scoring, calibration.

Complement prompts are deliberately negation-free (CLIP's text encoder
embeds "no beard" near "beard"): affirmative opposite where one exists,
neutral "a person" fallback otherwise.
"""

from collections.abc import Callable

import torch

TEMPLATES: list[str] = [
    "a photo of {}",
    "a cropped photo of {}",
    "a close-up photo of {}",
    "a portrait of {}",
    "a photo of the face of {}",
]

# Canonical list_attr_celeba.txt order. Values: (positive, complement).
ATTRIBUTE_PROMPTS: dict[str, tuple[str, str]] = {
    "5_o_Clock_Shadow": ("a man with five o'clock shadow stubble", "a clean-shaven person"),
    "Arched_Eyebrows": ("a person with arched eyebrows", "a person with straight eyebrows"),
    "Attractive": ("an attractive person", "an ordinary-looking person"),
    "Bags_Under_Eyes": ("a person with bags under the eyes", "a person with smooth skin under the eyes"),
    "Bald": ("a bald person", "a person with a full head of hair"),
    "Bangs": ("a person with bangs covering the forehead", "a person with the forehead visible"),
    "Big_Lips": ("a person with big lips", "a person with thin lips"),
    "Big_Nose": ("a person with a big nose", "a person with a small nose"),
    "Black_Hair": ("a person with black hair", "a person with light-colored hair"),
    "Blond_Hair": ("a person with blond hair", "a person with dark hair"),
    "Blurry": ("a person photographed out of focus", "a person photographed in sharp focus"),
    "Brown_Hair": ("a person with brown hair", "a person"),
    "Bushy_Eyebrows": ("a person with bushy eyebrows", "a person with thin eyebrows"),
    "Chubby": ("a chubby person", "a slim person"),
    "Double_Chin": ("a person with a double chin", "a person with a slim jawline"),
    "Eyeglasses": ("a person wearing eyeglasses", "a person"),
    "Goatee": ("a man with a goatee", "a clean-shaven person"),
    "Gray_Hair": ("a person with gray hair", "a person with dark hair"),
    "Heavy_Makeup": ("a person wearing heavy makeup", "a person with a bare face"),
    "High_Cheekbones": ("a person with high cheekbones", "a person with flat cheekbones"),
    "Male": ("a man", "a woman"),
    "Mouth_Slightly_Open": ("a person with the mouth slightly open", "a person with the mouth closed"),
    "Mustache": ("a person with a mustache", "a clean-shaven person"),
    "Narrow_Eyes": ("a person with narrow eyes", "a person with wide open eyes"),
    "No_Beard": ("a clean-shaven person", "a person with a beard"),
    "Oval_Face": ("a person with an oval face", "a person with a round face"),
    "Pale_Skin": ("a person with pale skin", "a person with tan skin"),
    "Pointy_Nose": ("a person with a pointy nose", "a person with a rounded nose"),
    "Receding_Hairline": ("a person with a receding hairline", "a person with a full head of hair"),
    "Rosy_Cheeks": ("a person with rosy cheeks", "a person with even-toned skin"),
    "Sideburns": ("a man with sideburns", "a clean-shaven person"),
    "Smiling": ("a smiling person", "a person with a serious expression"),
    "Straight_Hair": ("a person with straight hair", "a person with curly hair"),
    "Wavy_Hair": ("a person with wavy hair", "a person with straight hair"),
    "Wearing_Earrings": ("a person wearing earrings", "a person"),
    "Wearing_Hat": ("a person wearing a hat", "a bareheaded person"),
    "Wearing_Lipstick": ("a person wearing lipstick", "a person with bare lips"),
    "Wearing_Necklace": ("a person wearing a necklace", "a person with a bare neck"),
    "Wearing_Necktie": ("a person wearing a necktie", "a person with an open shirt collar"),
    "Young": ("a young person", "an elderly person"),
}


def build_text_embeddings(
    encode_texts: Callable[[list[str]], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensembled (positive, complement) text embeddings, one row per attribute.

    Each phrase is expanded through all TEMPLATES; the template embeddings
    are averaged and re-normalized. Returns two (A, D) tensors in
    ATTRIBUTE_PROMPTS key order.
    """
    def ensemble(phrase: str) -> torch.Tensor:
        embs = encode_texts([t.format(phrase) for t in TEMPLATES])
        mean = embs.mean(dim=0)
        return mean / mean.norm()

    pos_rows = [ensemble(pos) for pos, _ in ATTRIBUTE_PROMPTS.values()]
    neg_rows = [ensemble(neg) for _, neg in ATTRIBUTE_PROMPTS.values()]
    return torch.stack(pos_rows), torch.stack(neg_rows)


def attribute_scores(
    image_features: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> torch.Tensor:
    """(N, A) matrix of cos(v, t_pos) - cos(v, t_neg) per image and attribute.

    All inputs L2-normalized, so cosines are dot products and the
    difference collapses to a single matmul against (pos - neg).
    """
    return image_features @ (pos_emb - neg_emb).T


def _confusion_rates(scores, labels, threshold):
    preds = scores > threshold
    pos = labels.bool()
    tpr = preds[pos].float().mean().item()
    tnr = (~preds[~pos]).float().mean().item()
    return tpr, tnr


def balanced_accuracy(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    """Mean of TPR and TNR at `threshold`. Assumes both classes present."""
    tpr, tnr = _confusion_rates(scores, labels, threshold)
    return (tpr + tnr) / 2


def accuracy(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    correct = ((scores > threshold) == labels.bool()).sum().item()
    return correct / len(labels)


def calibrate_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Threshold maximizing balanced accuracy; midpoint between adjacent scores.

    Sorts once and sweeps all N cut points with cumulative sums.
    Assumes both classes present.
    """
    order = scores.argsort()
    s = scores[order]
    y = labels[order].float()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    # Cut after index i => predict positive for scores > s[i].
    tp = n_pos - y.cumsum(0)          # positives strictly above the cut
    tn = (1 - y).cumsum(0)            # negatives at or below the cut
    bal = (tp / n_pos + tn / n_neg) / 2
    best = int(bal.argmax())
    if best == len(s) - 1:            # degenerate: predict all negative
        return s[best].item() + 1e-6
    return ((s[best] + s[best + 1]) / 2).item()


def calibrate_threshold_prevalence(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Threshold matching the predicted positive rate to the true prevalence.

    With P positives in `labels`, cuts so exactly the top-P scores predict
    positive; midpoint between adjacent scores.
    """
    n_pos = int(labels.sum())
    s = scores.sort(descending=True).values
    if n_pos == 0:
        return s[0].item() + 1e-6
    if n_pos >= len(s):
        return s[-1].item() - 1e-6
    return ((s[n_pos - 1] + s[n_pos]) / 2).item()


def calibrate_threshold_precision(
    scores: torch.Tensor,
    labels: torch.Tensor,
    min_precision: float = 0.7,
) -> float:
    """Lowest threshold whose precision stays >= min_precision (max recall).

    For a caption pipeline a false positive (hallucinated fragment) costs
    more than a false negative (omission), so recall is maximized only
    subject to a precision floor. Falls back to prevalence matching when no
    cut reaches the floor (near-chance attributes).
    """
    order = scores.argsort(descending=True)
    s = scores[order]
    y = labels[order].float()
    precision = y.cumsum(0) / torch.arange(1, len(y) + 1)
    ok = (precision >= min_precision).nonzero()
    if len(ok) == 0:
        return calibrate_threshold_prevalence(scores, labels)
    best = int(ok.max())                  # deepest cut still meeting the floor
    if best == len(s) - 1:
        return s[-1].item() - 1e-6
    return ((s[best] + s[best + 1]) / 2).item()


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
