import re

import torch

from src.attributes import ATTRIBUTE_PROMPTS, TEMPLATES, build_text_embeddings

CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]

NEGATION = re.compile(r"\b(no|not|without)\b|n't", re.IGNORECASE)


def test_prompt_table_covers_all_40_attributes_in_order():
    assert list(ATTRIBUTE_PROMPTS) == CELEBA_ATTRS


def test_prompts_are_negation_free_and_clean():
    for attr, (pos, neg) in ATTRIBUTE_PROMPTS.items():
        for phrase in (pos, neg):
            assert not NEGATION.search(phrase), f"negation in {attr}: {phrase!r}"
            assert "_" not in phrase, f"underscore leaked into {attr}: {phrase!r}"


def test_templates_format_a_phrase():
    for template in TEMPLATES:
        assert template.count("{}") == 1
        assert template.format("a person") != template


def _fake_encode_texts(prompts: list[str]) -> torch.Tensor:
    # Deterministic fake: embedding depends on prompt hash; normalized rows.
    gen = torch.Generator().manual_seed(0)
    base = torch.randn(len(prompts), 8, generator=gen)
    for i, p in enumerate(prompts):
        base[i] += (hash(p) % 1000) / 1000.0
    return base / base.norm(dim=-1, keepdim=True)


def test_build_text_embeddings_shapes_and_norms():
    pos, neg = build_text_embeddings(_fake_encode_texts)
    assert pos.shape == (len(ATTRIBUTE_PROMPTS), 8)
    assert neg.shape == (len(ATTRIBUTE_PROMPTS), 8)
    assert torch.allclose(pos.norm(dim=-1), torch.ones(pos.shape[0]), atol=1e-5)
    assert torch.allclose(neg.norm(dim=-1), torch.ones(neg.shape[0]), atol=1e-5)


from src.attributes import (
    accuracy,
    attribute_scores,
    balanced_accuracy,
    calibrate_threshold,
    roc_auc,
)


def test_attribute_scores_is_cosine_difference():
    v = torch.tensor([[1.0, 0.0]])                 # one image, D=2
    pos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])   # two attributes
    neg = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    s = attribute_scores(v, pos, neg)
    assert s.shape == (1, 2)
    assert torch.allclose(s, torch.tensor([[1.0, -1.0]]))


def test_calibrate_threshold_separable():
    # Positives all above 1.0, negatives all below: any threshold in (0.9, 1.1)
    scores = torch.tensor([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    thr = calibrate_threshold(scores, labels)
    assert 0.9 < thr < 1.1
    assert balanced_accuracy(scores, labels, thr) == 1.0


def test_calibrate_threshold_handles_offset_distributions():
    # Both classes entirely below zero: threshold 0 gets bal.acc 0.5,
    # a calibrated threshold separates them perfectly.
    scores = torch.tensor([-3.0, -2.8, -2.6, -1.4, -1.2, -1.0])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    thr = calibrate_threshold(scores, labels)
    assert balanced_accuracy(scores, labels, thr) == 1.0
    assert balanced_accuracy(scores, labels, 0.0) == 0.5


def test_balanced_accuracy_vs_accuracy_on_imbalance():
    # 8 negatives, 2 positives; predict-all-negative threshold.
    scores = torch.tensor([-1.0] * 8 + [1.0] * 2)
    labels = torch.tensor([0] * 8 + [1] * 2)
    thr = 2.0  # everything predicted negative
    assert accuracy(scores, labels, thr) == 0.8
    assert balanced_accuracy(scores, labels, thr) == 0.5


def test_roc_auc_extremes():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    assert roc_auc(torch.tensor([1.0, 2, 3, 4, 5, 6]), labels) == 1.0
    assert roc_auc(torch.tensor([6.0, 5, 4, 3, 2, 1]), labels) == 0.0


from src.attributes import calibrate_threshold_precision, calibrate_threshold_prevalence


def test_calibrate_threshold_prevalence_matches_positive_rate():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])
    labels = torch.tensor([0, 1, 0, 1, 0, 0])  # prevalence 2/6
    thr = calibrate_threshold_prevalence(scores, labels)
    assert (scores > thr).sum() == 2


def test_calibrate_threshold_precision_separable_maximizes_recall():
    # Perfectly separable: the floor is met by taking every positive.
    scores = torch.tensor([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    thr = calibrate_threshold_precision(scores, labels, min_precision=1.0)
    preds = scores > thr
    assert preds.tolist() == [False, False, False, True, True, True]


def test_calibrate_threshold_precision_floor_cuts_false_positives():
    # Descending scores: labels 1,1,0,0,0,0. Precision at k=2 is 1.0,
    # at k=3 drops to 2/3 < 0.9 -> the cut must stop after the top two.
    scores = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    labels = torch.tensor([1, 1, 0, 0, 0, 0])
    thr = calibrate_threshold_precision(scores, labels, min_precision=0.9)
    assert (scores > thr).sum() == 2
    assert labels[scores > thr].all()


def test_calibrate_threshold_precision_falls_back_to_prevalence():
    # Scores anti-correlated with labels: no cut reaches the floor,
    # so the threshold falls back to prevalence matching.
    scores = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    thr = calibrate_threshold_precision(scores, labels, min_precision=0.9)
    assert thr == calibrate_threshold_prevalence(scores, labels)
    assert (scores > thr).sum() == 2
