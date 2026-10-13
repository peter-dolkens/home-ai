"""Shared plumbing for scenario builders: the Conv transcript builder, phrase
helpers and the voice-style perturbation applied to user utterances."""
from __future__ import annotations

import random
from typing import Any, Iterable, Optional

from ..house.schema import Entity, House, Speaker
from ..house.snapshot import num, rel_time, system_prompt, system_prompt_b
from ..tools import execute_tool


import re as _re


def fix_articles(text: str) -> str:
    """'the Amber's office' -> 'Amber's office'; 'the the' -> 'the'."""
    text = _re.sub(r"\b([Tt]he) ([A-Z][a-z]+'s )", lambda m: m.group(2) if m.group(1) == "the" else m.group(2)[:1].upper() + m.group(2)[1:], text)
    return _re.sub(r"\b([Tt]he) the\b", r"\1", text)


class Conv:
    """Builds one training transcript against a live simulated house."""

    on_start = None  # optional callback(conv), e.g. bigbench records the house state the transcript starts from
    # Tool design new transcripts are written for: "a" (homeai.tools, device names in the prompt) or "b" (homeai.tools_b,
    # rooms-only prompt, find_devices and filtered get_state). build.py / bigbench.py --toolset b switch it. B-only steps
    # never draw from rng, so a B transcript has exactly the requests, house and answers of its A twin.
    toolset = "a"

    def __init__(self, house: House, speaker: Optional[Speaker], category: str, rng: random.Random, toolset: Optional[str] = None):
        self.house = house
        self.speaker = speaker
        self.category = category
        self.rng = rng
        self.toolset = toolset or Conv.toolset
        prompt = system_prompt_b if self.toolset == "b" else system_prompt
        self.msgs: list[dict[str, Any]] = [{"role": "system", "content": prompt(house, speaker)}]
        self.ncalls = 0
        self.seen: set[str] = set()  # B: entity ids earlier tool results showed
        self.b_issues: list[str] = []  # B: steps whose results do not support the reference (cut off, unseen entity ids)
        if Conv.on_start:
            Conv.on_start(self)

    def user(self, text: str):
        self.msgs.append({"role": "user", "content": fix_articles(text)})

    def fetch(self, areas: Iterable[str] = (), classes: Iterable[str] = (), b=None, need: Iterable[Entity] = ()):
        """get_state(areas, classes). b: what design B asks instead, when the same filters would be too broad at
        big-house scale: a get_state args dict, or a list of dicts / (tool, args) pairs sent together in one turn
        (need: entities the reference answer relies on, flagged if the B results do not show them)."""
        if self.toolset == "b" and b is not None:
            return self.b_step(b, need)
        args: dict[str, Any] = {}
        if areas:
            args["areas"] = list(areas)
        if classes:
            args["classes"] = list(classes)
        self._tool_turn([("get_state", args)])

    def b_step(self, calls, need: Iterable[Entity] = ()):
        """Design B only (no-op in A): a tool turn A did not need, e.g. a follow-up get_state after a filtered one."""
        if self.toolset != "b":
            return
        from ..tools_b import ids_in
        calls = [calls] if isinstance(calls, dict) else calls
        n0 = len(self.msgs)
        self._tool_turn([c if isinstance(c, tuple) else ("get_state", c) for c in calls])  # dict: get_state args (see _tool_turn_b)
        shown = set().union(*[ids_in(m["content"]) for m in self.msgs[n0:] if m["role"] == "tool"])
        unseen = [e.entity_id for e in need if e.entity_id not in shown]
        if unseen:
            self.b_issues.append(f"results do not show {unseen}")

    def find(self, need: Iterable[Entity] = (), queries: Iterable[dict[str, Any]] = (), exact: bool = False):
        """Design B only (no-op in A): find_devices where A read device names off the inventory. Sends the first query
        whose result lists every entity in need without being cut off (exact: and no other entity of their classes, for
        "which ones are there" questions); none: the last one, and the transcript is flagged. Skipped when earlier
        results already showed all of need."""
        if self.toolset != "b":
            return
        from ..tools_b import find_devices_text, ids_in, truncated
        need = list(need)
        need_ids = {e.entity_id for e in need}
        if need_ids and need_ids <= self.seen and not exact:
            return
        classes = {e.cls for e in need}
        queries = list(queries)
        chosen = None
        for q in queries:
            text = find_devices_text(self.house, q.get("areas"), q.get("classes"), q.get("name") or "")
            shown = ids_in(text)
            extra = [i for i in shown - need_ids if self.house.entities[i].cls in classes]
            if need_ids <= shown and not truncated(text) and not (exact and extra):
                chosen = q
                break
        if chosen is None:  # the first query that lists them all among others, else the first
            chosen = next((q for q in queries if need_ids <= ids_in(find_devices_text(self.house, q.get("areas"), q.get("classes"), q.get("name") or ""))),
                          queries[0])
            self.b_issues.append(f"find_devices: no query lists just {sorted(need_ids)}")
        self._tool_turn([("find_devices", dict(chosen))])

    def lookup(self, areas: Iterable[str] = (), classes: Iterable[str] = (), need: Iterable[Entity] = (), find=None):
        """An entity-id lookup before acting. A: get_state(areas, classes). B: nothing when earlier results already
        showed every entity in need, else find_devices (find: candidate queries, default the same areas and classes)."""
        if self.toolset != "b":
            return self.fetch(areas, classes)
        q = {k: list(v) for k, v in (("areas", areas), ("classes", classes)) if v}
        self.find(need, find or [q])

    def act(self, calls: list[dict[str, Any]]):
        self._tool_turn([("call_service", c) for c in calls])

    def _tool_turn(self, calls):
        if self.toolset == "b":
            return self._tool_turn_b(calls)
        tcs, results = [], []
        for name, args in calls:
            self.ncalls += 1
            cid = f"call_{self.ncalls}"
            tcs.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
            results.append((cid, name, execute_tool(self.house, name, args)))
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
        for cid, name, content in results:
            self.msgs.append({"role": "tool", "tool_call_id": cid, "name": name, "content": content})

    def _tool_turn_b(self, calls):
        """Same transcript shape through the design-B tools, checking the reference only uses what the model was shown:
        results cut off by the default cap and call_service entity ids no earlier result listed are flagged. A get_state
        args key "_rooms": True (not sent) marks a call whose answer only needs the rooms, which a cut-off result still
        lists in its count of the rest by area."""
        from ..tools_b import execute_tool_b, ids_in, truncated
        tcs, results = [], []
        for name, args in calls:
            args = dict(args)
            rooms_only = bool(args.pop("_rooms", False))
            if name == "call_service":
                tgt = (args.get("target") or {}).get("entity_id") or []
                unseen = [i for i in ([tgt] if isinstance(tgt, str) else tgt) if i not in self.seen]
                if unseen:
                    self.b_issues.append(f"call_service on unseen {unseen}")
            self.ncalls += 1
            cid = f"call_{self.ncalls}"
            tcs.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
            content = execute_tool_b(self.house, name, args)
            if name != "call_service":
                self.seen |= ids_in(content)
                if truncated(content) and not rooms_only:
                    self.b_issues.append(f"{name} {args} cut off")
            results.append((cid, name, content))
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
        for cid, name, content in results:
            self.msgs.append({"role": "tool", "tool_call_id": cid, "name": name, "content": content})

    def say(self, text: str, must: Optional[list[str]] = None, must_not: Optional[list[str]] = None):
        m: dict[str, Any] = {"role": "assistant", "content": fix_articles(text)}
        ev: dict[str, Any] = {}
        if must:
            ev["must_mention"] = [s.lower() for s in must]
        if must_not:
            ev["must_not_mention"] = [s.lower() for s in must_not]
        if ev:
            m["_eval"] = ev
        self.msgs.append(m)

    def build(self) -> dict[str, Any]:
        sp = None
        if self.speaker:
            sp = {"name": self.speaker.name, "area": self.speaker.area, "channel": self.speaker.channel}
        ex = {"category": self.category, "house_seed": self.house.seed, "speaker": sp, "messages": self.msgs}
        if self.toolset == "b":
            ex["toolset"] = "b"
        return ex


# ---------------------------------------------------------------- phrasing
def pick(rng: random.Random, options: list[str], **kw) -> str:
    return rng.choice(options).format(**kw)


def join(items: list[str], conj: str = "and") -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


def room_ref(house: House, area_id: str) -> str:
    """'the office' / \"Jess's room\" / 'security'."""
    n = house.area_spoken(area_id)
    if "'s " in n:  # "Jess's Room", "Pete's office": possessive, no article
        return n.split("'s ")[0] + "'s " + n.split("'s ", 1)[1].lower()
    return "the " + n.lower()


def room_plain(house: House, area_id: str) -> str:
    """'office' / \"Jess's room\" (no article), for 'the office lights' constructions."""
    n = house.area_spoken(area_id)
    if "'s " in n:
        return n.split("'s ")[0] + "'s " + n.split("'s ", 1)[1].lower()
    return n.lower()


def rooms_ref(house: House, area_ids: list[str]) -> str:
    return join([room_ref(house, a) for a in area_ids])


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def short(e: Entity, house: House) -> str:
    """Device name with the area prefix removed, lower-cased: 'desk lamp'."""
    n = e.name
    for an in (house.area_name(e.area), house.area_spoken(e.area)):
        if n.lower().startswith(an.lower() + " ") and len(n) > len(an) + 1:
            n = n[len(an) + 1:]
            break
    return n.lower()


def isv(name: str) -> str:
    """'Amber's' / 'Marcus is' - contraction that avoids the awkward Marcus's."""
    return f"{name} is" if name.lower().endswith("s") else f"{name}'s"


def is_are(n: int) -> str:
    return "is" if n == 1 else "are"


QUESTION_START = ("is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "will", "should", "have", "has",
                  "what", "whats", "what's", "where", "wheres", "where's", "who", "whos", "who's", "when", "why", "how",
                  "hows", "how's", "which", "any", "anyone", "anybody", "anything", "much", "am")
INTERJECTIONS = ("um", "uh", "er", "erm", "hey", "ok", "okay", "right", "so", "yeah", "yep", "oh", "hmm", "well", "sorry")


def stt_case(text: str) -> str:
    """Format an utterance the way Whisper / Parakeet hand it over: sentence case, a comma after a leading interjection
    or wake phrase, "I" and "OK"/"Okay" capitalised, and a closing full stop or question mark (the heard-set evals
    found the models, trained on voiceify's lower-case unpunctuated text, lose a turn or two in a hundred to casing
    and punctuation alone)."""
    t = " ".join(text.strip().split()).rstrip(",;")
    if not t:
        return text
    t = _re.sub(r"\bi\b", "I", t)
    t = _re.sub(r"\b(ok)\b", "Okay", t, flags=_re.I)
    words = t.split(" ")
    first = words[0].lower().strip(",")
    if first in INTERJECTIONS and len(words) > 1 and not words[0].endswith(","):
        if first == "hey" and words[1].lower().strip(",") == "house":
            words[1] = words[1].rstrip(",") + ","
        else:
            words[0] += ","
    t = " ".join(words)
    t = t[:1].upper() + t[1:]
    if t[-1] not in ".?!":
        body = [w.lower().strip(",") for w in t.split(" ") if w.lower().strip(",") not in INTERJECTIONS + ("house",)]
        t += "?" if body and body[0] in QUESTION_START else "."
    return t


def voiceify(text: str, rng: random.Random, kind: str = "cmd", stt: float = 0.0) -> str:
    """Make a clean utterance look like something a speech-to-text layer emits. stt: probability of handing it over
    in STT style (stt_case) after the perturbation; 0 draws nothing extra from rng, so existing builds are unchanged."""
    out = _voiceify(text, rng, kind)
    return stt_case(out) if stt and rng.random() < stt else out


def _voiceify(text: str, rng: random.Random, kind: str) -> str:
    r = rng.random()
    if r < 0.4:
        return text
    if r < 0.65:
        t = text.rstrip("?.!")
        return t[:1].lower() + t[1:]
    if kind == "cmd":
        t = text.rstrip("?.!")
        first = t.split(" ")[0].lower()
        if first in VERBS:
            pre = rng.choice(["hey, ", "can you ", "could you ", "please ", "ok ", "hey house, ", "um, ", "right, ", ""])
        else:
            pre = rng.choice(["hey, ", "ok ", "hey house, ", "um, ", ""])
        suf = rng.choice(["", "", " please", " thanks", ""])
        return (pre + t[:1].lower() + t[1:] + suf).strip()
    pre = rng.choice(["hey, ", "hey house, ", "quick question, ", "um, ", ""])
    t = text.rstrip("?.!")
    return (pre + t[:1].lower() + t[1:]).strip()


def pct(x: float) -> str:
    return num(x, 0)


def spoken_dur(s: int) -> str:
    """Duration for speech: 'about 3 and a half hours', '20 minutes'."""
    if s < 90:
        return "a minute"
    m = round(s / 60)
    if m < 60:
        return f"{m} minutes"
    h, mm = divmod(m, 60)
    if mm < 15:
        return f"about {h} hour{'s' if h != 1 else ''}"
    if mm < 40:
        return f"about {h} and a half hours"
    return f"nearly {h + 1} hours"


def spoken_ago(s: int) -> str:
    return "just now" if s < 90 else spoken_dur(s) + " ago"


VERBS = ("turn", "switch", "set", "dim", "brighten", "close", "open", "lock", "unlock", "pause", "play", "mute", "kill",
         "put", "resume", "unpause", "check", "disarm", "make", "warm", "cool", "start", "give", "shut", "heat", "arm")


__all__ = ["Conv", "pick", "join", "room_ref", "room_plain", "rooms_ref", "cap", "short", "is_are", "isv",
           "voiceify", "stt_case", "num", "rel_time", "pct", "spoken_ago", "spoken_dur"]
