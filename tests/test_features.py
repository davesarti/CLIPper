from pathlib import Path

import torch

from src.features import ClipEncoder, cache_path, load_or_extract


def test_cache_path_embeds_model_and_split(tmp_path):
    p = cache_path(tmp_path, "test")
    assert p.parent == tmp_path
    assert "clip-vit-base-patch32" in p.name
    assert "test" in p.name
    assert p.suffix == ".pt"


def test_load_or_extract_uses_cache(tmp_path):
    cached = torch.randn(5, 512)
    torch.save(cached, cache_path(tmp_path, "test"))

    class ExplodingEncoder:
        def encode_images(self, dataset, batch_size=64, limit=None):
            raise AssertionError("must not extract when cache exists")

    out = load_or_extract(ExplodingEncoder(), dataset=None,
                          features_dir=tmp_path, split="test")
    assert torch.allclose(out, cached)


def test_load_or_extract_extracts_and_saves(tmp_path):
    fresh = torch.randn(3, 512)

    class FakeEncoder:
        def encode_images(self, dataset, batch_size=64, limit=None):
            return fresh

    out = load_or_extract(FakeEncoder(), dataset="unused",
                          features_dir=tmp_path, split="test")
    assert torch.allclose(out, fresh)
    assert cache_path(tmp_path, "test").is_file()  # saved for next time


def test_model_name_is_the_required_one():
    assert ClipEncoder.MODEL_NAME == "openai/clip-vit-base-patch32"
