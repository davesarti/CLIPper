"""Full attribute-caption pipeline on the 14-query benchmark.

Predict each reference's attributes with zero-shot CLIP, apply the query as
bit-flips, render the flipped state into a caption, and rank the test split
by caption-to-image cosine (docs/method-proposal-attribute-caption.md,
steps 1-3; no identity term yet).

Run with: conda run -n clipper python scripts/run_attribute_caption.py
Requires results/attribute_thresholds.json from run_attribute_probe.py.
Reuses the cached test-split features; text encoding takes ~1 min on CPU.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attributes import ATTRIBUTE_PROMPTS, attribute_scores, build_text_embeddings
from src.caption import predict_state
from src.data import get_paths, load_annotations, load_dataset
from src.evaluation import run_caption_benchmark
from src.features import ClipEncoder, load_or_extract

REPO_ROOT = Path(__file__).resolve().parent.parent
thresholds_path = REPO_ROOT / "results" / "attribute_thresholds.json"
if not thresholds_path.is_file():
    sys.exit(
        f"Missing {thresholds_path}: run scripts/run_attribute_probe.py first "
        "to calibrate the per-attribute thresholds."
    )
with open(thresholds_path) as f:
    thresholds_by_attr = json.load(f)
thresholds = torch.tensor([thresholds_by_attr[a] for a in ATTRIBUTE_PROMPTS])

paths = get_paths()
celeba = load_dataset(paths)
annotations = load_annotations(paths)
encoder = ClipEncoder()

features = load_or_extract(encoder, celeba, paths.features_dir)
assert features.shape[0] == len(celeba), "feature/dataset size mismatch"

pos_emb, neg_emb = build_text_embeddings(encoder.encode_texts)
states = predict_state(attribute_scores(features, pos_emb, neg_emb), thresholds)

df = run_caption_benchmark(annotations, features, states, encoder.encode_texts)

out_dir = REPO_ROOT / "results"
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "attribute_caption_results.csv", index=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
