"""CLIP ViT-B/32 feature extraction with a disk cache."""

from pathlib import Path

import torch
from tqdm.auto import tqdm


class ClipEncoder:
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor

            self._model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device).eval()
            self._processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        return self._model, self._processor

    @torch.no_grad()
    def encode_texts(self, prompts: list[str]) -> torch.Tensor:
        model, processor = self._load()
        inputs = processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        feats = model.get_text_features(**inputs).pooler_output.cpu().float()
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(self, dataset, batch_size: int = 64,
                      limit: int | None = None) -> torch.Tensor:
        model, processor = self._load()
        n = len(dataset) if limit is None else min(limit, len(dataset))
        chunks = []
        for start in tqdm(range(0, n, batch_size), desc="Encoding images"):
            batch = [dataset[i][0] for i in range(start, min(start + batch_size, n))]
            inputs = processor(images=batch, return_tensors="pt").to(self.device)
            feats = model.get_image_features(**inputs).pooler_output.cpu().float()
            chunks.append(feats)
        feats = torch.cat(chunks)
        return feats / feats.norm(dim=-1, keepdim=True)


def resolve_pool(features_dir, override=None) -> Path:
    """Path to the train-split feature pool: the full split if it exists.

    Every script that trains on train-split features resolves it the same way,
    so extracting the full split with `extract_train_features.py --all` upgrades
    all of them at once instead of only the ones that happened to look for it.
    """
    if override is not None:
        return Path(override)
    slug = ClipEncoder.MODEL_NAME.split("/")[-1]
    full = Path(features_dir) / f"{slug}_train.pt"
    return full if full.is_file() else Path(features_dir) / f"{slug}_train30k.pt"


def load_pool(path) -> tuple[torch.Tensor, torch.Tensor]:
    """(features, train-split indices) from either cache layout.

    The sampled pool stores {"features", "indices"}; the full split is a plain
    tensor in dataset order, so its indices are simply 0..N-1. Callers need the
    indices to align train-split labels with the rows.
    """
    saved = torch.load(path, weights_only=True)
    if isinstance(saved, dict):
        return saved["features"], saved["indices"]
    return saved, torch.arange(saved.shape[0])


def cache_path(features_dir: Path, split: str) -> Path:
    model_slug = ClipEncoder.MODEL_NAME.split("/")[-1]
    return features_dir / f"{model_slug}_{split}.pt"


def load_or_extract(encoder, dataset, features_dir: Path, split: str = "test") -> torch.Tensor:
    path = cache_path(features_dir, split)
    if path.is_file():
        return torch.load(path, weights_only=True)
    features = encoder.encode_images(dataset)
    features_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, path)
    return features
