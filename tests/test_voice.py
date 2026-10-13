"""Offline checks for the voice-robustness stream (homeai/voice): normalisation, house terms, WER, the biasing prompt,
the confusion builders, and that a heard row replays through the outcome eval against the original reference."""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from homeai.eval.e2e import answer_ok, apply_ref, compare  # noqa: E402
from homeai.house.bighouse import house_for  # noqa: E402
from homeai.house.generator import make_house  # noqa: E402
from homeai.voice import confusions, vocab  # noqa: E402
from homeai.voice.check import oracle, strip  # noqa: E402
from homeai.voice.roundtrip import rate_for, user_turns, voice_for, VOICES  # noqa: E402


def test_norm():
    n = vocab.norm_words
    assert n("Set the lounge to 22°") == ["set", "the", "lounge", "to", "22", "degrees"]
    assert n("set the lounge to twenty-two degrees") == n("Set the lounge to 22 degrees.")
    assert n("um, dim it to fifty percent") == n("Dim it to 50%")
    assert n("a hundred and five") == ["105"] and n("one hundred") == ["100"]
    assert n("Pete's office") == ["petes", "office"]
    assert n("the veggie beds") == n("the vegie beds")


def test_terms_and_wer():
    house = make_house(900000, "bundook")
    t = vocab.Terms(house)
    spans = t.spans(vocab.norm_words("turn on the downlights in Pete's office and set Amber's office to 22"))
    kinds = [k for k, _ in spans]
    assert ("room", ("petes", "office")) in spans and ("room", ("ambers", "office")) in spans
    assert "command" in kinds and "number" in kinds and "device" in kinds
    # spacing is forgiven, a flipped command and a changed number are not
    assert vocab.lost_terms(t, "turn on the master ensuite lights", "turn on the master en suite lights") == []
    assert ("command", "on") in vocab.lost_terms(t, "turn on the fan", "turn off the fan")
    assert ("number", "50") in vocab.lost_terms(t, "set it to fifty percent", "set it to fifteen percent")
    assert ("room", "alfresco") in vocab.lost_terms(t, "alfresco lights on", "their fresco lights on")
    s = vocab.score(t, "turn on the alfresco lights", "turn on their fresco lights")
    assert s["words"] == 5 and s["errors"] == 2 and s["vocab"]["room"] == [1, 1] and s["lost"]
    assert vocab.align(["a", "b"], ["a", "c", "b"]) == [("ok", "a", "a"), ("ins", None, "c"), ("ok", "b", "b")]


def test_bias_prompt():
    house = make_house(900000, "bundook")
    p = vocab.bias_prompt(house, lambda s: len(s.split()), max_tokens=60)
    assert p.startswith("Office (Pete), Pete's office") and "Alfresco" in p and p.endswith(".")
    assert len(p.split()) <= 60
    assert "Amber" in vocab.bias_prompt(house)


def test_turns_and_voices():
    ex = {"id": "x", "speaker": {"name": "Amber"}, "messages": [{"role": "system", "content": "s"},
          {"role": "user", "content": "hi", "_written": "hello"}, {"role": "assistant", "content": "yo"}]}
    assert user_turns(ex) == [(1, "hello")]
    assert user_turns({"query": "any smoke"}) == [(None, "any smoke")]
    assert VOICES[voice_for(ex)][3] == "f"
    assert VOICES[voice_for({**ex, "speaker": {"name": "Peter"}})][3] == "m"
    assert voice_for(ex) == voice_for(copy.deepcopy(ex)) and rate_for(ex, 0) == rate_for(ex, 0)


def _heard(ex, transcripts):
    ex = copy.deepcopy(ex)
    users = [m for m in ex["messages"] if m["role"] == "user"]
    for m, t in zip(users, transcripts):
        m["_written"], m["content"] = m["content"], t
    ex["_voice"] = {"stt": "test", "condition": "clean", "turns": []}
    return ex


def test_heard_rows_score_against_the_reference():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "generated", "wild.jsonl")
    if not os.path.exists(path):
        print("  (skipped: data/generated/wild.jsonl not built)")
        return
    rows = [json.loads(l) for l in open(path)]
    ex = next(r for r in rows if r["id"] == "wild-900000-ctrl-compound-alfresco-office")
    heard = _heard(ex, ["turn on their fresco lights and kill the fan in pizza office"])
    assert strip(heard) == strip(ex)  # reference untouched
    assert oracle(heard) == []        # a perfect model still passes: scoring follows the reference, not the transcript
    msgs = heard["messages"]
    j = next(i for i, m in enumerate(msgs) if m["role"] == "user")
    before = house_for(heard)
    apply_ref(before, msgs[:j])
    ref_after = copy.deepcopy(before)
    apply_ref(ref_after, msgs[j + 1:])
    assert not compare(ref_after, before, msgs[j]["_written"])[0]  # doing nothing still fails
    ref_ans = [m for m in msgs if m["role"] == "assistant" and m.get("content")][-1]
    must = ref_ans["_eval"]["must_mention"]
    assert must == ["alfresco"] and not answer_ok("done, fresco lights on", must, "wild_control")[0]


def test_confusion_builders():
    assert confusions.swap("Turn on the Alfresco lights", "alfresco", "al fresco") == "Turn on the Al fresco lights"
    assert confusions.swap("is Pete's office warm", "Pete's office", "Pits office") == "is Pits office warm"
    assert confusions.pattern("on").search("turn it on") and not confusions.pattern("on").search("only")
    assert confusions.spoken(22) == "twenty two" and confusions.TEEN_TY[15] == 50 and confusions.TEEN_TY[50] == 15
    house = make_house(900000, "bundook")
    src = {"id": "t", "category": "wild_control", "house_seed": 900000, "profile": "bundook", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "chuck the kitchen lights on"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "call_service", "arguments": {
            "domain": "light", "service": "turn_on", "target": {"area_id": ["kitchen"]}}}}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "Kitchen lights are on.", "_eval": {"must_mention": ["kitchen"]}}]}
    moved = confusions.relocate(src, 1, house, "battery_shed")  # the battery shed's batten lights are off here
    assert moved and moved["messages"][-1]["content"] == "Battery shed lights are on."
    assert moved["messages"][1]["content"] == "chuck the battery shed lights on"
    assert moved["messages"][2]["tool_calls"][0]["function"]["arguments"]["target"]["area_id"] == ["battery_shed"]
    assert moved["messages"][-1]["_eval"]["must_mention"] == ["battery shed"]
    assert confusions.relocate(src, 1, house, "kitchen") is None  # same room: nothing to move


def test_heard_summary_pairs_turns():
    from homeai.eval.heard_summary import cell
    written = {("a", 0): 1.0, ("a", 1): 1.0, ("b", 0): 0.0, ("c", 0): 1.0}
    heard = {("a", 0): 0.0, ("a", 1): 1.0, ("b", 0): 1.0}          # ("c", 0) not run: left out of both sides
    c = cell(heard, written, {("a", 0): True})
    assert c["turns"] == 3 and c["lost"] == 1 and c["regained"] == 1
    assert abs(c["written_pass"] - 2 / 3) < 1e-3 and abs(c["pass"] - 2 / 3) < 1e-3 and c["drop"] == 0
    assert c["lost_term_changed"] == 1.0 and c["rest_term_changed"] == 0.0
    assert cell({}, written, {}) == {"turns": 0}


def test_v9_augmentations():
    from homeai.datagen import asr_v9
    from homeai.datagen.common import stt_case
    assert stt_case("hey house is the heating on") == "Hey house, is the heating on?"
    assert stt_case("um, turn on the kitchen lights") == "Um, turn on the kitchen lights."
    assert stt_case("Turn off the alarm.") == "Turn off the alarm."
    house = make_house(900000, "bundook")
    terms = vocab.Terms(house)
    sp = asr_v9.splits(terms)
    assert sp.get("ensuite") == "en suite" and sp.get("mudroom") == "mud room" and sp.get("alfresco") == "al fresco"
    assert asr_v9.swap_word("Unlock the door", "lock", "look") is None  # whole words only
    assert asr_v9.swap_word("Turn off the lights", "off", "of") == "Turn of the lights"
    ex = {"id": "t", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "lock the mudroom door"},
          {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "call_service", "arguments": {
              "domain": "lock", "service": "lock", "target": {"entity_id": ["lock.mudroom_door"]}}}}]},
          {"role": "tool", "content": "ok"}, {"role": "assistant", "content": "Locked."}]}
    kinds = {k for k, _, _ in asr_v9.near_misses(ex, 1, terms)}
    assert {"homophone", "split"} <= kinds
    import collections
    row = asr_v9.homophone_row(ex, set(), collections.Counter({"homophone": 1}), terms, house)
    assert row["messages"][1]["content"] == "look the mudroom door" and row["_v9"]["sub"] == "ambiguous"
    assert row["messages"][-1] == {"role": "assistant", "content": "Did you want the mudroom door locked, or just checked?"}
    assert asr_v9.said(22) == "twenty two" and asr_v9.said(50) == "fifty"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
