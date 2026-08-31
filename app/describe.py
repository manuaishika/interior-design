"""Turn what was found in the room into a sentence for the generator.

The generator does not see the photograph's contents — it sees pixels and a
prompt. Left to itself, "bedroom" makes it draw the most average bedroom it
knows, which is one double bed. A room with two single beds comes back with
one, because nothing ever told it there were two.

So the prompt is written from the analysis: count what is actually there, name
it, and hand that to the generator as a constraint. This is the part a generic
chat tool cannot do — it has no structured reading of the room to write from.
"""

from __future__ import annotations

import numpy as np

# Objects worth counting. Counting the wall or the floor says nothing useful.
COUNTABLE = {
    "bed", "chair", "armchair", "sofa", "table", "coffee table", "desk",
    "wardrobe", "cabinet", "chest of drawers", "shelf", "bookcase", "lamp",
    "mirror", "painting", "plant", "pot", "stool", "bench", "nightstand",
}

NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight",
}


def count_instances(mask: np.ndarray, min_frac: float = 0.004) -> int:
    """How many separate objects are in this one class mask.

    Semantic segmentation returns a single region per class, so two beds arrive
    as one 'bed' blob. Splitting it into connected components recovers the
    count — which is exactly the fact the generator needs and would otherwise
    invent for itself.
    """
    if not mask.any():
        return 0
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - scipy is present in Colab
        return 1

    labelled, found = ndimage.label(mask)
    if found <= 1:
        return int(found)

    total = mask.size
    return sum(
        1 for i in range(1, found + 1) if (labelled == i).sum() / total >= min_frac
    )


def plural(word: str, n: int) -> str:
    if n == 1:
        return word
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and not word.endswith(("ay", "ey", "oy", "uy")):
        return word[:-1] + "ies"
    return word + "s"


def phrase(name: str, n: int) -> str:
    if n == 1:
        article = "an" if name[:1].lower() in "aeiou" else "a"
        return f"{article} {name}"
    return f"{NUMBER_WORDS.get(n, str(n))} {plural(name, n)}"


def describe_room(counts: dict[str, int], limit: int = 6) -> str:
    """A phrase like 'two beds, a wardrobe and an air conditioner'.

    Biggest-count-first, capped, so the prompt stays short enough that the
    style direction is not drowned out by an inventory.
    """
    items = [(name, n) for name, n in counts.items() if n > 0 and name in COUNTABLE]
    if not items:
        return ""

    # Multiples first — those are the facts most likely to be lost.
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    parts = [phrase(name, n) for name, n in items[:limit]]

    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def keep_clause(counts: dict[str, int]) -> str:
    """The instruction that stops two beds collapsing into one.

    Only mentions things there is more than one of: telling the model to keep
    "one bed" is noise, telling it to keep "two beds" is load-bearing.
    """
    multiples = [(n, name) for name, n in counts.items() if n > 1 and name in COUNTABLE]
    if not multiples:
        return ""
    multiples.sort(reverse=True)
    listed = ", ".join(f"{NUMBER_WORDS.get(n, n)} separate {plural(name, n)}"
                       for n, name in multiples[:3])
    return f"keep exactly {listed}"
