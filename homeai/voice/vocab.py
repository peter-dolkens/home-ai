"""House vocabulary, transcript normalisation and word error rates for the voice-robustness stream.

Pure Python (no numpy), so the offline tests and confusions.py run on the plain interpreter. roundtrip.py uses it to
score transcripts; confusions.py to find the words worth mishearing on purpose.

  norm_words("Set the lounge to 22°")          -> ['set', 'the', 'lounge', 'to', '22', 'degrees']
  norm_words("set the lounge to twenty-two")   -> ['set', 'the', 'lounge', 'to', '22']

WER here is word-level Levenshtein over those normalised words (fillers dropped on both sides, numbers as digits,
apostrophes and hyphens folded, the Whisper-normaliser convention). House-vocabulary WER only counts reference words
that name a room, a device, a person, a number or a command word ("on", "off", "unlock"), and a turn's meaning is
"changed" when any such term in the written request can no longer be found in the transcript, even allowing for
spacing ("en suite" still finds "ensuite", "petes office" still finds "Pete's office").
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

from ..house.schema import House

FILLERS = {"um", "uh", "er", "erm", "hmm", "mm", "ah", "eh", "uhm"}
COMMANDS = {"on", "off", "open", "close", "shut", "lock", "unlock", "arm", "disarm", "pause", "play", "stop", "start",
            "skip", "mute", "unmute", "louder", "quieter", "up", "down", "dim", "dimmer", "brighter", "warmer", "cooler",
            "raise", "lower", "resume"}
STOP = {"the", "a", "an", "and", "of", "in", "on", "to", "s", "my", "all", "main", "1", "2", "3", "4"}
KINDS = ("room", "device", "person", "number", "command")
# spellings a recogniser may pick either way (Whisper's normaliser folds British/American the same way)
SPELLING = {"veggie": "vegie", "veggies": "vegies", "colour": "color", "grey": "gray", "okay": "ok", "tv": "tv",
            "theatre": "theater", "metre": "meter", "metres": "meters", "centre": "center", "favourite": "favorite",
            "organise": "organize", "aircon": "aircon", "wifi": "wifi", "alright": "all right"}
# entity classes people do not name out loud (a forecast called "Tomorrow" is not a device anyone says)
UNNAMED = {"forecast", "weather", "sun", "device", "meter", "measurement", "detection", "illuminance", "air_quality"}

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_UNIT = {w: i for i, w in enumerate(_ONES)}


def _numbers_to_digits(toks: list[str]) -> list[str]:
    """['twenty', 'two', 'degrees'] -> ['22', 'degrees']; 'a hundred' / 'one hundred and five' -> '100' / '105'."""
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t not in _UNIT and t not in _TENS and not (t in ("a", "one") and i + 1 < len(toks) and toks[i + 1] == "hundred"):
            out.append(t)
            i += 1
            continue
        val, j = 0, i
        if toks[j] in ("a", "one") and j + 1 < len(toks) and toks[j + 1] == "hundred":
            val, j = 100, j + 2
            if j + 1 < len(toks) and toks[j] == "and" and (toks[j + 1] in _UNIT or toks[j + 1] in _TENS):
                j += 1
        elif toks[j] in _UNIT and j + 1 < len(toks) and toks[j + 1] == "hundred":
            val, j = _UNIT[toks[j]] * 100, j + 2
            if j + 1 < len(toks) and toks[j] == "and" and (toks[j + 1] in _UNIT or toks[j + 1] in _TENS):
                j += 1
        rest = 0
        if j < len(toks) and toks[j] in _TENS:
            rest, j = _TENS[toks[j]], j + 1
            if j < len(toks) and toks[j] in _UNIT and 0 < _UNIT[toks[j]] < 10:
                rest, j = rest + _UNIT[toks[j]], j + 1
        elif j < len(toks) and toks[j] in _UNIT and (val or j == i):
            rest, j = _UNIT[toks[j]], j + 1
        out.append(str(val + rest))
        i = j
    return out


def norm_words(text: str) -> list[str]:
    s = (text or "").lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)
    s = re.sub(r"(\d)\s*%", r"\1 percent ", s).replace("per cent", "percent")
    s = re.sub(r"(\d)\s*°\s*c?\b", r"\1 degrees ", s).replace("°", " degrees ")
    s = re.sub(r"(\d)(am|pm)\b", r"\1 \2", s)
    s = re.sub(r"(\d)\.(\d)", r"\1 point \2", s)
    s = re.sub(r"['`]", "", s)                     # pete's -> petes, don't -> dont
    s = re.sub(r"[^a-z0-9]+", " ", s)             # hyphens and punctuation -> space
    s = re.sub(r"\bair con\b", "aircon", s).replace("wi fi", "wifi").replace("t v", "tv")
    toks = [SPELLING.get(t, t) for t in s.split() if t not in FILLERS]
    return _numbers_to_digits(toks)


def compact(words: Iterable[str]) -> str:
    return "".join(words)


# ---------------------------------------------------------------------- house terms
def house_key(ex: dict):
    """What decides an example's names: the house it was built on (patches only change states)."""
    h = ex.get("big") or ex.get("house")
    if h:
        return (h.get("kind", "random"), int(h["seed"]), int(h.get("n") or 0))
    return ("small", ex.get("house_seed"), ex.get("profile"))


def house_of(ex: dict) -> House:
    from ..house.bighouse import big_spec_house, house_for
    if not ex.get("big") and ex.get("house"):  # leaf rows: {"house": {"kind", "seed"}}
        return big_spec_house(ex["house"])
    return house_for(ex)


def device_phrase(house: House, e) -> str:
    """'Pete's Office Downlights' -> 'Downlights': the name minus its room, as a speaker says it inside that room."""
    name = e.name
    for an in sorted({house.area_spoken(e.area), house.area_name(e.area), e.area.replace("_", " ")}, key=len, reverse=True):
        if an and name.lower().startswith(an.lower() + " "):
            return name[len(an) + 1:]
    return name


class Terms:
    """The rooms, devices and people of one house, as normalised word tuples, for tagging and meaning checks."""

    def __init__(self, house: House):
        self.rooms: set[tuple] = set()
        self.room_names: list[str] = []
        for a in house.areas:
            for n in (a.name, a.spoken, re.sub(r"\s*\(.*\)", "", a.name)):
                if n and n not in self.room_names:
                    self.room_names.append(n)
                    self.rooms.add(tuple(norm_words(n)))
                    self.rooms.update((w,) for w in norm_words(n) if w not in STOP and not w.isdigit())  # "the ensuite"
        self.people = {p.name for p in house.people}
        self.persons = {tuple(norm_words(n)) for n in self.people}
        self.devices: set[tuple] = set()
        for e in house.entities.values():
            if e.cls in UNNAMED:
                continue
            ph = tuple(w for w in norm_words(device_phrase(house, e)))
            if ph and not all(w in STOP or w.isdigit() for w in ph):
                self.devices.add(ph)
                content = [w for w in ph if w not in STOP and not w.isdigit()]
                self.devices.update((w,) for w in content)  # "downlights" alone names the "Office Downlights" too
        self.devices -= self.rooms | self.persons
        self.maxlen = max([len(t) for t in self.rooms | self.devices | self.persons] + [1])

    def tag(self, words: list[str]) -> list[Optional[str]]:
        """Kind of each reference word (room / device / person / number / command) or None; longest phrase wins."""
        tags: list[Optional[str]] = [None] * len(words)
        i = 0
        while i < len(words):
            for n in range(min(self.maxlen, len(words) - i), 0, -1):
                ph = tuple(words[i:i + n])
                kind = ("person" if ph in self.persons else "room" if ph in self.rooms else
                        "device" if ph in self.devices else None)
                if kind:
                    for k in range(i, i + n):
                        tags[k] = kind
                    i += n
                    break
            else:
                w = words[i]
                tags[i] = "number" if w.isdigit() else "command" if w in COMMANDS else None
                i += 1
        return tags

    def spans(self, words: list[str]) -> list[tuple[str, tuple]]:
        """[(kind, phrase words)] for every house term the words contain."""
        tags, out, i = self.tag(words), [], 0
        while i < len(words):
            if tags[i] is None:
                i += 1
                continue
            j = i + 1
            while j < len(words) and tags[j] == tags[i] and tags[i] not in ("number", "command"):
                j += 1
            out.append((tags[i], tuple(words[i:j])))
            i = j
        return out


def lost_terms(terms: Terms, ref: str, hyp: str) -> list[tuple[str, str]]:
    """House terms of the written request that the transcript no longer contains ([(kind, term)]): the errors that
    can change what gets done. Spacing is forgiven ("en suite" = "ensuite"); a number must keep its value; a command
    word must still be there as a word ("on" inside "only" does not count)."""
    rw, hw = norm_words(ref), norm_words(hyp)
    hc, hs = compact(hw), set(hw)
    lost = []
    for kind, ph in terms.spans(rw):
        if kind in ("number", "command"):
            ok = ph[0] in hs
        else:
            ok = compact(ph) in hc
        if not ok:
            lost.append((kind, " ".join(ph)))
    return lost


# ---------------------------------------------------------------------- word error rate
def align(ref: list[str], hyp: list[str]) -> list[tuple[str, Optional[str], Optional[str]]]:
    """Levenshtein alignment: [(op, ref word, hyp word)] with op in ok / sub / del / ins."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append(("del", ref[i - 1], None))
            i -= 1
        else:
            out.append(("ins", None, hyp[j - 1]))
            j -= 1
    return out[::-1]


def score(terms: Terms, ref: str, hyp: str) -> dict:
    """Per-utterance counts: words, errors, and per house-vocabulary kind the reference words and how many were lost."""
    rw, hw = norm_words(ref), norm_words(hyp)
    tags = terms.tag(rw)
    al = align(rw, hw)
    errs = sum(op != "ok" for op, _, _ in al)
    vocab = {k: [0, 0] for k in KINDS}
    ri = 0
    for op, r, _ in al:
        if op == "ins":
            continue
        if tags[ri]:
            vocab[tags[ri]][0] += 1
            vocab[tags[ri]][1] += op != "ok"
        ri += 1
    lost = lost_terms(terms, ref, hyp)
    return {"words": len(rw), "errors": errs, "vocab": vocab, "lost": lost, "exact": rw == hw}


# ---------------------------------------------------------------------- biasing prompt
PRIORITY_DOMAINS = {"light", "switch", "fan", "media_player", "climate", "scene", "todo"}


def bias_prompt(house: House, count_tokens: Callable[[str], int] = lambda s: len(s) // 4, max_tokens: int = 200) -> str:
    """The initial_prompt wyoming-faster-whisper builds from Home Assistant's names (--hass-token, vocabulary.py):
    a comma-joined list filled in priority order until the token budget (200 by default) is spent: areas with
    devices, then entity names in the domains people say out loud, then empty areas, then everything else. People's
    names ride with the areas here (HA would list the person entities in the last tier, where a big house never
    reaches them)."""
    used = {e.area for e in house.entities.values()}
    names: list[str] = []
    seen: set[str] = set()

    def add(xs):
        for x in xs:
            x = " ".join((x or "").split())
            if x and x.lower() not in seen:
                seen.add(x.lower())
                names.append(x)

    for a in house.areas:
        if a.area_id in used:
            add([a.name, a.spoken])
    add(p.name for p in house.people)
    add(e.name for e in house.entities.values() if e.domain in PRIORITY_DOMAINS)
    for a in house.areas:
        if a.area_id not in used:
            add([a.name, a.spoken])
    add(e.name for e in house.entities.values() if e.domain not in PRIORITY_DOMAINS)
    chosen: list[str] = []
    for n in names:
        if count_tokens(", ".join(chosen + [n]) + ".") > max_tokens:
            break
        chosen.append(n)
    return ", ".join(chosen) + "." if chosen else ""
