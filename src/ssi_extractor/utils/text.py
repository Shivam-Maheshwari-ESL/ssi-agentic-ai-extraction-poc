"""Text utilities shared across harvesting, chunking, evidence matching and comparison.

Kept here rather than inside the first stage that needed them, so evidence
matching and semantic comparison cannot drift apart from the normalisation the
extractor used.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

__all__ = [
    "collapse_whitespace",
    "fold_for_comparison",
    "is_probably_label",
    "similarity",
    "split_lines",
    "strip_accents",
    "substring_present",
]

_WHITESPACE_RE = re.compile(r"\s+")
_LABEL_HINT_RE = re.compile(r"[:：]\s*$")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def strip_accents(value: str) -> str:
    """Fold accents so a label matches whether or not OCR kept its diacritics."""
    normalised = unicodedata.normalize("NFKD", value)
    return "".join(character for character in normalised if not unicodedata.combining(character))


def collapse_whitespace(value: str) -> str:
    """Collapse runs of whitespace, including the non-breaking spaces PDFs emit."""
    return _WHITESPACE_RE.sub(" ", value.replace(" ", " ")).strip()


def split_lines(value: str) -> list[str]:
    """Split into non-empty, whitespace-collapsed lines."""
    return [line for line in (collapse_whitespace(part) for part in value.splitlines()) if line]


def fold_for_comparison(value: str) -> str:
    """Aggressive fold for value equality: accent-free, lower-case, alphanumeric only.

    Used by the semantic comparator and by hallucination detection, so
    ``"A/C 4773-002145"`` and ``"ac 4773002145"`` compare equal while the stored
    value stays verbatim.
    """
    return _NON_ALNUM_RE.sub("", strip_accents(value).lower())


def is_probably_label(value: str) -> bool:
    """Heuristic: does this text read as a field label rather than a value?

    Used only to seed harvesting candidates; the kind inference decides what a
    field actually is, so a false positive here is cheap.
    """
    text = collapse_whitespace(value)
    if not text or len(text) > 60:
        return False
    if _LABEL_HINT_RE.search(text):
        return True
    words = text.split()
    if len(words) > 6:
        return False
    alphabetic = sum(character.isalpha() for character in text)
    return alphabetic >= max(2, len(text) // 2)


def substring_present(needle: str, haystack: str, *, min_ratio: float = 0.9) -> bool:
    """Whether ``needle`` appears in ``haystack``, tolerantly.

    Exact folded containment first; otherwise the best sliding-window similarity,
    which absorbs OCR noise without accepting a value the source never contained.
    That distinction is what makes G2's hallucination check meaningful.
    """
    folded_needle = fold_for_comparison(needle)
    folded_haystack = fold_for_comparison(haystack)
    if not folded_needle:
        return True
    if folded_needle in folded_haystack:
        return True
    if len(folded_needle) > len(folded_haystack):
        return False

    window = len(folded_needle)
    best = 0.0
    step = max(1, window // 4)
    for start in range(0, len(folded_haystack) - window + 1, step):
        candidate = folded_haystack[start : start + window]
        best = max(best, SequenceMatcher(None, folded_needle, candidate).ratio())
        if best >= min_ratio:
            return True
    return best >= min_ratio


def similarity(left: str, right: str) -> float:
    """Folded similarity ratio in [0, 1], used for scoring rather than deciding."""
    return SequenceMatcher(None, fold_for_comparison(left), fold_for_comparison(right)).ratio()
