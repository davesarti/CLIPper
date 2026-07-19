# Attribute-Caption Retrieval: Negation-Free Composed Retrieval via Text Rendering

**Method proposal — compositional image retrieval on CelebA with frozen CLIP ViT-B/32.**

| | |
|---|---|
| Dataset | CelebA (40 binary attributes); test split as fixed retrieval database |
| Encoder | Frozen CLIP ViT-B/32 only — no captioner, no LLM, no training loop |
| Fitted quantities | 40 per-attribute thresholds + 1 blend scalar γ (tuned, not trained) |
| Closest prior art | CIReVL (ICLR 2024), Pic2Word (CVPR 2023) — language-mediated retrieval |

## 0. Core idea

Route the query through **text** instead of embedding arithmetic. A caption
describes what *is* in an image, never what isn't — so if we describe the
*target* image in words, negative modifiers are handled by **omission or
replacement**, and CLIP never has to parse "not X" or subtract a vector.
This dissolves the baseline's worst failure (negation collapse:
`-Male, -Mustache` → R@1 = 0) by construction.

Key enabler: CelebA is a **closed attribute world**. Queries use the same 40
attributes the dataset annotates, so we don't need free-form captioning —
only to know which attributes the reference has. CLIP can do that alone.

## 1. Pipeline

```
v_ref ──(1)──> â ∈ {0,1}^40 ──(2)──> â' ──(3)──> caption ──> q_text
                                                              │
score(v) = cos(q_text, v) + γ · cos(P·v_ref, P·v)   <──(4)────┘
```

**(1) Read attributes off the reference.** Zero-shot CLIP classification:
for each attribute, compare `cos(v_ref, t("...with X"))` against its
complement prompt (prompt-ensembled), with a **per-attribute threshold**
calibrated on the train split — CLIP is well calibrated on some attributes
(Male, Eyeglasses), poor on others (Oval_Face). Output: estimated state `â`.

**(2) Apply the query as bit-flips.** T+ sets bits to 1, T− sets bits to 0.
Exact set arithmetic on a structured state — no fuzzy text editing.

**(3) Render and encode.** Turn `â'` into a caption via templates (extending
the `PROMPTS` map in `src/retrieval.py` to all 40 attributes). A cleared bit
is simply never mentioned; where a natural complement exists, render the
contrast instead of silence (−Blond_Hair → "dark hair", −Male → "a woman",
−Young → "an older person") via a small hand-written complement table.
Ensemble 4–6 caption templates, average the normalized text embeddings.

**(4) Hybrid scoring.** The caption carries only attribute-level information —
face shape, skin tone, pose are gone, so every "smiling blond woman" would
score the same. Blend image-side identity back in:

```
score(v) = cos(q_text, v) + γ · cos(P·v_ref, P·v)
```

where `P` projects out the span of the flipped attributes' text directions,
so the reference's *old* mustache doesn't fight the query that removes it.
Image-to-image cosine, so the modality gap is irrelevant here. Single scalar
γ tuned on validation. Without this term the method is attribute retrieval,
not composed retrieval.

## 2. Why this beats the arithmetic baseline

1. **Negation is free.** No subtraction overshoot, no "not X" for CLIP to
   misparse — the target caption just doesn't contain the attribute.
2. **Native alignment.** Text→image cosine is exactly what CLIP was trained
   for; the baseline's image+text vector mixing crosses the modality gap
   with uncalibrated magnitudes.
3. **Conditioning for free.** The caption is built from the *reference's*
   full attribute state, so "add blond hair" composes with everything else
   the reference already has.

## 3. Experiments (each step bounds the next)

1. **Oracle ceiling first** (pure lookup + templating, no classifier): run
   the pipeline with ground-truth reference attributes from
   `list_attr_celeba.txt`. This bounds the caption bottleneck in isolation.
2. **Attribute classifier accuracy**: calibrated CLIP vs. true labels,
   per attribute. The oracle-vs-predicted gap isolates classification error.
3. **Full zero-shot pipeline vs. baseline**, stratified positive-only /
   negative-only / mixed — negative-only is where it must decisively win.
4. **Ablations**: γ on/off, projection `P` on/off, complement rendering vs.
   pure omission, confidence cutoff on which bits to render (a long caption
   full of noisy attributes may hurt more than a short accurate one).

**Fallback if zero-shot underperforms:** a single linear adapter on `q_text`,
trained with the same InfoNCE + mined-negatives harness as SCAC — keeps the
two proposals directly comparable.

## 4. Limitations

- **Identity bottleneck** is the central risk: everything not expressible in
  the 40 attributes survives only through the γ term. Measure top-K identity
  match via `identity_CelebA.txt`.
- **Closed vocabulary**: the method works *because* CelebA's query space is
  closed; open-vocabulary composed retrieval would need a real captioner —
  state this trade-off explicitly in the report.
- **Silence may be weak** for complement-less attributes (−Wearing_Hat →
  nothing to say): every hatless caption scores similarly. The γ identity
  term and experiment 4's ablation cover this.
