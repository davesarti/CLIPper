"""One-off smoke test: encode 100 images, run one query end-to-end.

Run with: conda run -n clipper python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import get_paths, load_annotations, load_dataset
from src.features import ClipEncoder
from src.retrieval import PROMPTS, compose, parse_query, rank

paths = get_paths()
celeba = load_dataset(paths)
annotations = load_annotations(paths)
encoder = ClipEncoder()

print("Encoding 100 images...")
feats = encoder.encode_images(celeba, batch_size=32, limit=100)
assert feats.shape == (100, 512), feats.shape
assert torch.allclose(feats.norm(dim=-1), torch.ones(100), atol=1e-5)

query = annotations[0]["query"]  # "+Smiling"
pos, neg = parse_query(query)
pos_t = encoder.encode_texts([PROMPTS[a] for a in pos])
neg_t = encoder.encode_texts([PROMPTS[a] for a in neg]) if neg else torch.zeros((0, 512))

src = 13  # a source index for +Smiling in the benchmark
q = compose(feats[src], pos_t, neg_t).unsqueeze(0)
order = rank(q, feats, exclude=[src])
print(f"Query {query!r}, source {src}: top-5 = {order[0][:5].tolist()}")
print("Smoke test OK")
