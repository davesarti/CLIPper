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
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
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
