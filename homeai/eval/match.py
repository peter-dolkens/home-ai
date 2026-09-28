"""Phrase matching for the key facts a reply must mention (must_mention in the eval sets).

Plain substring matching failed correct replies on wording alone: "Jack's phone left about 4 hours ago" for
"four hours" / "away", "82%" for "82 percent", "0 watts" for "zero watts". mentions() also accepts those after
normalising number words, percent signs and thousands separators, and a few synonyms for presence words.
It only ever accepts more than a plain substring test, never less.
"""
from __future__ import annotations

import re

_NUM = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_NUM.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90"})
_NUM_RE = re.compile(r"\b(" + "|".join(_NUM) + r")\b")

# must phrase (normalised) -> other ways a correct reply says it
SYNONYMS = {
    "away": ["out", "not home", "not here", "isn't home", "isn't here", "left", "gone", "not in"],
}


def norm(s: str) -> str:
    s = (s or "").lower().replace("’", "'")
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)          # 2,185 -> 2185
    s = _NUM_RE.sub(lambda m: _NUM[m.group(1)], s)      # four -> 4
    s = re.sub(r"\s*(per ?cent)\b", "%", s)             # 82 percent -> 82%
    s = re.sub(r"\s+%", "%", s)
    return re.sub(r"\s+", " ", s)


def mentions(text: str, phrase: str) -> bool:
    """True if text states phrase (one alternative of a must_mention item)."""
    p = phrase.strip().lower()
    t = (text or "").lower()
    if not p or p in t:
        return True
    nt, np_ = norm(t), norm(p)
    if np_ in nt:
        return True
    return any(re.search(r"\b" + re.escape(s) + r"\b", nt) for s in SYNONYMS.get(np_, []))


def item_hit(text: str, item: str) -> bool:
    """A must_mention item is 'a|b|c': any alternative counts."""
    return any(mentions(text, alt) for alt in item.split("|"))
