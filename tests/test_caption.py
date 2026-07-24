import re

import torch

from src.attributes import ATTRIBUTE_PROMPTS
from src.caption import (
    CAPTION_FRAGMENTS,
    encode_caption,
    flip_state,
    predict_state,
    render_caption,
)

ATTRS = list(ATTRIBUTE_PROMPTS)
NEGATION = re.compile(r"\b(no|not|without)\b|n't", re.IGNORECASE)


def make_state(*on: str) -> torch.Tensor:
    state = torch.zeros(len(ATTRS), dtype=torch.bool)
    for attr in on:
        state[ATTRS.index(attr)] = True
    return state


# --- fragment table ---------------------------------------------------------

def test_fragment_table_covers_all_40_attributes_in_order():
    assert list(CAPTION_FRAGMENTS) == ATTRS


def test_fragments_are_negation_free_and_clean():
    for attr, (pos, neg) in CAPTION_FRAGMENTS.items():
        for frag in (pos, neg):
            if frag is None:
                continue
            slot, text = frag
            assert slot in {"adj", "noun", "with", "wearing", "tail"}, (attr, slot)
            assert not NEGATION.search(text), f"negation in {attr}: {text!r}"
            assert "_" not in text, f"underscore leaked into {attr}: {text!r}"


# --- predict ----------------------------------------------------------------

def test_predict_state_applies_per_attribute_thresholds():
    scores = torch.tensor([[0.5, -0.5], [0.1, 0.3]])
    thresholds = torch.tensor([0.2, 0.0])
    state = predict_state(scores, thresholds)
    assert state.dtype == torch.bool
    assert state.tolist() == [[True, False], [False, True]]


# --- flip -------------------------------------------------------------------

def test_flip_sets_positive_and_clears_negative_bits():
    state = make_state("Male", "Mustache")
    new, forced_off = flip_state(state, ["Smiling"], ["Mustache"])
    assert new[ATTRS.index("Smiling")]
    assert not new[ATTRS.index("Mustache")]
    assert new[ATTRS.index("Male")]          # untouched bits survive
    assert not state[ATTRS.index("Smiling")]  # input not mutated
    assert forced_off == ("Mustache",)


def test_flip_reports_forced_off_even_if_bit_was_already_clear():
    # "-Mustache" on a reference without one still signals intent to render.
    new, forced_off = flip_state(make_state("Male"), [], ["Mustache"])
    assert forced_off == ("Mustache",)


def test_flip_clears_conflicting_hair_colors():
    state = make_state("Black_Hair")
    new, _ = flip_state(state, ["Blond_Hair"], [])
    assert new[ATTRS.index("Blond_Hair")]
    assert not new[ATTRS.index("Black_Hair")]


def test_flip_facial_hair_clears_no_beard_and_vice_versa():
    new, _ = flip_state(make_state("No_Beard"), ["Mustache"], [])
    assert not new[ATTRS.index("No_Beard")]
    new, _ = flip_state(make_state("Goatee", "Mustache"), ["No_Beard"], [])
    assert not new[ATTRS.index("Goatee")]
    assert not new[ATTRS.index("Mustache")]


# --- exclusive groups -------------------------------------------------------

from src.caption import EXCLUSIVE_GROUPS, resolve_exclusive_groups


def _margins_for(values: dict[str, float]) -> torch.Tensor:
    m = torch.zeros(1, len(ATTRS))
    for attr, v in values.items():
        m[0, ATTRS.index(attr)] = v
    return m


def test_resolve_keeps_only_max_margin_hair_color():
    states = make_state("Black_Hair", "Brown_Hair", "Smiling").unsqueeze(0)
    margins = _margins_for({"Black_Hair": 0.1, "Brown_Hair": 0.3})
    new = resolve_exclusive_groups(states, margins)
    assert not new[0, ATTRS.index("Black_Hair")]
    assert new[0, ATTRS.index("Brown_Hair")]
    assert new[0, ATTRS.index("Smiling")]  # non-group bits untouched


def test_resolve_settles_no_beard_vs_facial_hair_contradiction():
    states = make_state("No_Beard", "Mustache", "Sideburns").unsqueeze(0)
    margins = _margins_for({"No_Beard": 0.05, "Mustache": 0.2, "Sideburns": 0.1})
    new = resolve_exclusive_groups(states, margins)
    assert new[0, ATTRS.index("Mustache")]
    assert not new[0, ATTRS.index("No_Beard")]
    assert not new[0, ATTRS.index("Sideburns")]


def test_resolve_leaves_single_member_and_empty_groups_alone():
    states = make_state("Blond_Hair", "Male").unsqueeze(0)
    new = resolve_exclusive_groups(states, torch.zeros(1, len(ATTRS)))
    assert (new == states).all()


def test_resolve_does_not_mutate_input():
    states = make_state("Black_Hair", "Brown_Hair").unsqueeze(0)
    resolve_exclusive_groups(states, torch.zeros(1, len(ATTRS)))
    assert states[0, ATTRS.index("Black_Hair")]


def test_exclusive_groups_use_known_attributes():
    for group in EXCLUSIVE_GROUPS:
        assert len(group) >= 2
        for attr in group:
            assert attr in ATTRS


# --- render -----------------------------------------------------------------

def test_render_empty_state_is_a_person():
    assert render_caption(make_state()) == "a person"


def test_render_full_sentence_orders_slots():
    state = make_state("Smiling", "Young", "Male", "Blond_Hair", "Wearing_Lipstick")
    assert render_caption(state) == (
        "a smiling young man with blond hair, wearing lipstick"
    )


def test_render_omits_cleared_bits():
    caption = render_caption(make_state("Male"))
    assert caption == "a man"
    assert "mustache" not in caption


def test_render_complement_for_forced_off_bits():
    # "-Male, -Mustache": flipped state renders the contrast, not silence.
    state = make_state("Male", "Mustache")
    new, forced_off = flip_state(state, [], ["Male", "Mustache"])
    assert render_caption(new, forced_off) == "a clean-shaven woman"


def test_render_uses_an_before_vowel():
    state, forced_off = flip_state(make_state("Young"), [], ["Young"])
    assert render_caption(state, forced_off) == "an elderly person"


def test_render_deduplicates_repeated_fragments():
    # Both -Mustache and -Goatee map to "clean-shaven"; render it once.
    state, forced_off = flip_state(
        make_state("Mustache", "Goatee"), [], ["Mustache", "Goatee"]
    )
    caption = render_caption(state, forced_off)
    assert caption.count("clean-shaven") == 1


def test_render_joins_lists_with_and():
    state = make_state("Blond_Hair", "Mustache", "Eyeglasses", "Wearing_Hat")
    assert render_caption(state) == (
        "a person with blond hair and a mustache, wearing eyeglasses and a hat"
    )


def test_render_is_negation_free_for_random_states():
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        state = torch.rand(len(ATTRS), generator=gen) > 0.7
        assert not NEGATION.search(render_caption(state))


# --- encode -----------------------------------------------------------------

def _fake_encode_texts(prompts: list[str]) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    base = torch.randn(len(prompts), 8, generator=gen)
    for i, p in enumerate(prompts):
        base[i] += (hash(p) % 1000) / 1000.0
    return base / base.norm(dim=-1, keepdim=True)


def test_encode_caption_returns_normalized_vector():
    q = encode_caption("a smiling woman", _fake_encode_texts)
    assert q.shape == (8,)
    assert torch.allclose(q.norm(), torch.tensor(1.0), atol=1e-5)
