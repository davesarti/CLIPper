"""Attribute state -> query bit-flips -> rendered caption -> query embedding.

Implements steps (1)-(3) of docs/method-proposal-attribute-caption.md:
the reference's predicted attribute state is edited by exact set arithmetic
(T+ sets bits, T- clears them), then rendered into a negation-free caption.
A cleared bit is omitted; a bit the *query* removed renders its affirmative
complement instead ("-Male" -> "a woman"), so CLIP never sees "not X".
"""

import torch

from src.attributes import ATTRIBUTE_PROMPTS, TEMPLATES

# Caption grammar slots, assembled as:
#   "a/an {adj...} {noun} with {with...}, wearing {wearing...}, {tail...}"
# Values: (positive fragment, complement fragment) where a fragment is
# (slot, text) and the complement (used when the query clears the bit) is
# None where silence is the only natural rendering. Canonical
# list_attr_celeba.txt order, same as ATTRIBUTE_PROMPTS.
CAPTION_FRAGMENTS: dict[str, tuple[tuple[str, str], tuple[str, str] | None]] = {
    "5_o_Clock_Shadow": (("with", "five o'clock shadow stubble"), ("adj", "clean-shaven")),
    "Arched_Eyebrows": (("with", "arched eyebrows"), ("with", "straight eyebrows")),
    "Attractive": (("adj", "attractive"), None),
    "Bags_Under_Eyes": (("with", "bags under the eyes"), ("with", "smooth skin under the eyes")),
    "Bald": (("adj", "bald"), ("with", "a full head of hair")),
    "Bangs": (("with", "bangs covering the forehead"), ("with", "the forehead visible")),
    "Big_Lips": (("with", "big lips"), ("with", "thin lips")),
    "Big_Nose": (("with", "a big nose"), ("with", "a small nose")),
    "Black_Hair": (("with", "black hair"), ("with", "light-colored hair")),
    "Blond_Hair": (("with", "blond hair"), ("with", "dark hair")),
    "Blurry": (("tail", "photographed out of focus"), ("tail", "photographed in sharp focus")),
    "Brown_Hair": (("with", "brown hair"), None),
    "Bushy_Eyebrows": (("with", "bushy eyebrows"), ("with", "thin eyebrows")),
    "Chubby": (("adj", "chubby"), ("adj", "slim")),
    "Double_Chin": (("with", "a double chin"), ("with", "a slim jawline")),
    "Eyeglasses": (("wearing", "eyeglasses"), None),
    "Goatee": (("with", "a goatee"), ("adj", "clean-shaven")),
    "Gray_Hair": (("with", "gray hair"), ("with", "dark hair")),
    "Heavy_Makeup": (("wearing", "heavy makeup"), ("with", "a bare face")),
    "High_Cheekbones": (("with", "high cheekbones"), ("with", "flat cheekbones")),
    "Male": (("noun", "man"), ("noun", "woman")),
    "Mouth_Slightly_Open": (("with", "the mouth slightly open"), ("with", "the mouth closed")),
    "Mustache": (("with", "a mustache"), ("adj", "clean-shaven")),
    "Narrow_Eyes": (("with", "narrow eyes"), ("with", "wide open eyes")),
    "No_Beard": (("adj", "clean-shaven"), ("with", "a beard")),
    "Oval_Face": (("with", "an oval face"), ("with", "a round face")),
    "Pale_Skin": (("with", "pale skin"), ("with", "tan skin")),
    "Pointy_Nose": (("with", "a pointy nose"), ("with", "a rounded nose")),
    "Receding_Hairline": (("with", "a receding hairline"), ("with", "a full head of hair")),
    "Rosy_Cheeks": (("with", "rosy cheeks"), None),
    "Sideburns": (("with", "sideburns"), ("adj", "clean-shaven")),
    "Smiling": (("adj", "smiling"), ("with", "a serious expression")),
    "Straight_Hair": (("with", "straight hair"), ("with", "curly hair")),
    "Wavy_Hair": (("with", "wavy hair"), ("with", "straight hair")),
    "Wearing_Earrings": (("wearing", "earrings"), None),
    "Wearing_Hat": (("wearing", "a hat"), ("adj", "bareheaded")),
    "Wearing_Lipstick": (("wearing", "lipstick"), ("with", "bare lips")),
    "Wearing_Necklace": (("wearing", "a necklace"), None),
    "Wearing_Necktie": (("wearing", "a necktie"), ("with", "an open shirt collar")),
    "Young": (("adj", "young"), ("adj", "elderly")),
}

_ATTRS = list(ATTRIBUTE_PROMPTS)
_INDEX = {attr: i for i, attr in enumerate(_ATTRS)}

# Setting the key attribute clears the listed ones, so a query like
# "+Blond_Hair" on a black-haired reference doesn't render both colors.
_HAIR_COLORS = ("Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair")
_FACIAL_HAIR = ("5_o_Clock_Shadow", "Goatee", "Mustache", "Sideburns")
CONFLICTS: dict[str, tuple[str, ...]] = (
    {c: tuple(o for o in _HAIR_COLORS if o != c) + ("Bald",) for c in _HAIR_COLORS}
    | {fh: ("No_Beard",) for fh in _FACIAL_HAIR}
    | {
        "Bald": _HAIR_COLORS + ("Straight_Hair", "Wavy_Hair"),
        "Straight_Hair": ("Wavy_Hair",),
        "Wavy_Hair": ("Straight_Hair",),
        "No_Beard": _FACIAL_HAIR,
    }
)

# Families where the *predicted* state should assert at most one member:
# independent per-attribute classifiers routinely set several (e.g. two hair
# colors), and rendering more than the most confident one puts a contradiction
# or a hallucination in every caption. Facial-hair styles can genuinely
# co-occur, but keeping only the best-supported one merely omits the rest —
# harmless for a caption — while resolving the No_Beard contradiction.
EXCLUSIVE_GROUPS: tuple[tuple[str, ...], ...] = (
    _HAIR_COLORS + ("Bald",),
    ("No_Beard",) + _FACIAL_HAIR,
    ("Straight_Hair", "Wavy_Hair"),
)


def resolve_exclusive_groups(states: torch.Tensor, margins: torch.Tensor) -> torch.Tensor:
    """(N, A) bool states -> copy where each EXCLUSIVE_GROUPS family keeps
    only its highest-margin positive member per row.

    margins: (N, A) confidence (score - threshold); only compared within a
    row's positive members, so its scale doesn't matter.
    """
    new = states.clone()
    for group in EXCLUSIVE_GROUPS:
        idx = torch.tensor([_INDEX[a] for a in group])
        members = states[:, idx]
        m = margins[:, idx].masked_fill(~members, float("-inf"))
        winner = torch.zeros_like(members)
        winner[torch.arange(len(members)), m.argmax(dim=1)] = True
        new[:, idx] = winner & members
    return new


# Same-family attributes that can end up simultaneously true in a predicted
# state (each is classified independently). Clearing one on a query "-attr"
# must silently clear the rest too, or the caption keeps stale positive
# fragments (e.g. "-Mustache" leaving "with a goatee, sideburns" standing)
# or, worse, contradicts the forced-off complement text (e.g. "light-colored
# hair" from "-Black_Hair" next to a leftover "brown hair").
_NEGATION_GROUPS: tuple[tuple[str, ...], ...] = (_HAIR_COLORS, _FACIAL_HAIR)


def predict_state(scores: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """(N, A) attribute scores -> (N, A) bool state via per-attribute thresholds."""
    return scores > thresholds


def flip_state(
    state: torch.Tensor,
    positives: list[str],
    negatives: list[str],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Apply the query as bit-flips on a copy of `state` (A,) bool.

    T+ sets bits (clearing CONFLICTS so the caption stays consistent),
    T- clears bits and, if the attribute belongs to a _NEGATION_GROUPS
    family, silently clears other set members of that family too (they
    are not added to `forced_off`, so they're omitted rather than
    rendering a possibly-contradictory complement of their own). Returns
    the new state plus the T- attributes as `forced_off`, which
    render_caption turns into affirmative complements even if the bit
    was already clear.
    """
    new = state.clone()
    for attr in positives:
        new[_INDEX[attr]] = True
        for other in CONFLICTS.get(attr, ()):
            new[_INDEX[other]] = False
    for attr in negatives:
        new[_INDEX[attr]] = False
        for group in _NEGATION_GROUPS:
            if attr in group:
                for other in group:
                    if other != attr:
                        new[_INDEX[other]] = False
    return new, tuple(negatives)


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def render_caption(state: torch.Tensor, forced_off: tuple[str, ...] = ()) -> str:
    """Render a bool attribute state (A,) into one negation-free caption.

    Set bits contribute their positive fragment; query-cleared bits their
    complement fragment (omission where None); everything else is silent.
    """
    slots: dict[str, list[str]] = {"adj": [], "with": [], "wearing": [], "tail": []}
    noun = "person"
    for attr, (positive, complement) in CAPTION_FRAGMENTS.items():
        if attr in forced_off:
            fragment = complement
        elif state[_INDEX[attr]]:
            fragment = positive
        else:
            continue
        if fragment is None:
            continue
        slot, text = fragment
        if slot == "noun":
            noun = text
        elif text not in slots[slot]:  # -Mustache and -Goatee both say clean-shaven
            slots[slot].append(text)

    noun_phrase = " ".join(slots["adj"] + [noun])
    article = "an" if noun_phrase[0] in "aeiou" else "a"
    head = f"{article} {noun_phrase}"
    if slots["with"]:  # "with" binds to the noun: no comma before it
        head += " with " + _join(slots["with"])
    parts = [head]
    if slots["wearing"]:
        parts.append("wearing " + _join(slots["wearing"]))
    parts.extend(slots["tail"])
    return ", ".join(parts)


def encode_caption(caption, encode_texts) -> torch.Tensor:
    """Template-ensembled normalized text embedding of one caption."""
    embs = encode_texts([t.format(caption) for t in TEMPLATES])
    mean = embs.mean(dim=0)
    return mean / mean.norm()
