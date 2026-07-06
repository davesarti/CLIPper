"""Query parsing, prompt templates, and latent-arithmetic retrieval."""

import torch

# Prompt templates in the style of the original CLIP paper (Radford et al., 2021).
# One entry per attribute appearing in the 14 benchmark queries; extend as needed.
PROMPTS: dict[str, str] = {
    "Smiling": "a photo of a smiling person",
    "Eyeglasses": "a photo of a person wearing eyeglasses",
    "Heavy_Makeup": "a photo of a person with heavy makeup",
    "Male": "a photo of a man",
    "Young": "a photo of a young person",
    "Blond_Hair": "a photo of a person with blond hair",
    "Mustache": "a photo of a person with a mustache",
    "Black_Hair": "a photo of a person with black hair",
    "Wavy_Hair": "a photo of a person with wavy hair",
    "Chubby": "a photo of a chubby person",
    "Wearing_Hat": "a photo of a person wearing a hat",
    "Wearing_Lipstick": "a photo of a person wearing lipstick",
}


def parse_query(query: str) -> tuple[list[str], list[str]]:
    """Parse '+A, -B' into (positives, negatives) lists of attribute names."""
    positives: list[str] = []
    negatives: list[str] = []
    for token in query.split(","):
        token = token.strip()
        if token.startswith("+"):
            positives.append(token[1:].strip())
        elif token.startswith("-"):
            negatives.append(token[1:].strip())
        else:
            raise ValueError(f"Query token must start with '+' or '-': {token!r}")
    return positives, negatives


def compose(
    v_ref: torch.Tensor,
    pos_texts: torch.Tensor,
    neg_texts: torch.Tensor,
) -> torch.Tensor:
    """Naive latent arithmetic: normalize(v_ref + sum(pos) - sum(neg)).

    All inputs are expected L2-normalized. This is the vanilla baseline
    composition; the future fusion module replaces this function.
    """
    q = v_ref + pos_texts.sum(dim=0) - neg_texts.sum(dim=0)
    return q / q.norm()


def rank(
    query_vecs: torch.Tensor,
    image_features: torch.Tensor,
    exclude: list[int] | None = None,
) -> torch.Tensor:
    """Return image indices sorted by descending cosine similarity.

    query_vecs: (S, D) normalized query vectors (one row per source image).
    image_features: (N, D) normalized image features.
    exclude: optional list of length S; exclude[i] (the source image of
        row i) is forced to the bottom of row i's ranking. Must have length S.
    """
    if exclude is not None and len(exclude) != query_vecs.shape[0]:
        raise ValueError(
            f"exclude must have one entry per query row: "
            f"got {len(exclude)} for {query_vecs.shape[0]} rows"
        )
    sims = query_vecs @ image_features.T  # (S, N)
    if exclude is not None:
        rows = torch.arange(sims.shape[0])
        sims[rows, torch.tensor(exclude)] = float("-inf")
    return sims.argsort(dim=1, descending=True)
