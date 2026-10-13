"""Offline checks of the eval scorer using replayed reference turns (no Ollama)."""
import json, random, sys
sys.path.insert(0, ".")
from homeai.datagen.build import generate_one
from homeai.eval.run_eval import score_turn, to_ollama
from homeai.house.generator import make_house

rows = [generate_one(i, 5, None, "bundook", 0.5) for i in range(240)]
rows = [r for r in rows if r]
n_ok = n = 0
kinds = {}
for ex in rows:
    house = make_house(ex["house_seed"], ex.get("profile"))
    for j, m in enumerate(ex["messages"]):
        if m["role"] != "assistant":
            continue
        to_ollama(ex["messages"][:j])
        kind, ok, why, over = score_turn(house, m, m)  # perfect replay must pass
        n += 1; n_ok += ok
        kinds.setdefault(kind, 0); kinds[kind] += 1
        assert ok, (ex["category"], why, m)
print(f"perfect replay: {n_ok}/{n} pass, kinds={kinds}")

# equivalence: area_id target vs its entity_ids should score identical
ex = next(r for r in rows if r["category"] == "control_area")
house = make_house(ex["house_seed"], ex.get("profile"))
ref = next(m for m in ex["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
args = ref["tool_calls"][0]["function"]["arguments"]
ids = [e.entity_id for a in args["target"]["area_id"] for e in house.in_area(a) if e.domain == args["domain"]]
alt = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "call_service", "arguments": {**args, "target": {"entity_id": ids}}}}]}
assert score_turn(house, ref, alt)[1], "entity_id form of same action should pass"
wrong = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "call_service", "arguments": {**args, "service": "toggle"}}}]}
assert not score_turn(house, ref, wrong)[1]
# fetch superset passes, subset fails
exf = next(r for r in rows if r["category"] == "presence_where" and any(m.get("tool_calls") for m in r["messages"]))
house = make_house(exf["house_seed"], exf.get("profile"))
ref = next(m for m in exf["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
sup = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_state", "arguments": {}}}]}
k, ok, why, over = score_turn(house, ref, sup)
assert ok and over > 1, (ok, over)
sub = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_state", "arguments": {"classes": ["motion"], "areas": [house.areas[0].area_id]}}}]}
print("subset fetch:", score_turn(house, ref, sub)[:3])
# answer with tool call fails
ans = next(m for m in exf["messages"] if m["role"] == "assistant" and not m.get("tool_calls"))
assert not score_turn(house, ans, sup)[1]
print("scorer OK")

# clean-turn rescoring: turn k = k-th user message, strict carries a seen turn forward, failures rebuild the outcomes
import os, tempfile
from homeai.eval import clean_rescore as cr
turns = {("a", 0): "turn on the light", ("a", 1): "yes please", ("a", 2): "the kitchen one", ("b", 0): "hows the tank"}
flags = cr.contamination(turns, {cr.norm("Yes please.")})
assert flags == {("a", 0): (False, False), ("a", 1): (True, True), ("a", 2): (False, True), ("b", 0): (False, False)}, flags
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "failures.jsonl"), "w") as fh:
        fh.write(json.dumps({"id": "a", "turn": 1, "why": "missing:x"}) + "\n")
        fh.write(json.dumps({"id": "b", "turn": 0, "why": cr.HALF_WHY + ": lock.front"}) + "\n")
    got, how = cr.run_outcomes(d, turns)
    assert how == "failures.jsonl" and got == {("a", 0): 1, ("a", 1): 0, ("a", 2): 1, ("b", 0): 0.5}, got
    sc = cr.score(got, flags)
    assert sc["all"]["n"] == 4 and sc["clean"]["n"] == 3 and sc["strict_clean"]["n"] == 2 and sc["clean"]["pass"] == 2.5 / 3
print("clean rescore OK")

# user_context (v11) and clock_in_user (v12) layouts: the context / clock line is added only to the request the model
# gets; the outcome scorer's number tolerance ("a bit" = loose, a number said = tight) and the failure log read the
# stored, bare utterance
import copy
from homeai.eval.e2e import Run
from homeai.house.schema import Speaker
from homeai.house.snapshot import CLOCK_TAG, CONTEXT_TAG, system_prompt
h11 = make_house(11)
call = lambda b: {"function": {"name": "call_service", "arguments": {"domain": "light", "service": "turn_on",  # noqa: E731
                                                                    "target": {"entity_id": ["light.kitchen_ceiling"]}, "data": {"brightness_pct": b}}}}
row = {"id": "uc", "category": "control_brightness", "house_seed": 11, "profile": None,
       "speaker": {"name": h11.people[0].name, "area": "kitchen", "channel": "satellite"},
       "messages": [{"role": "system", "content": system_prompt(h11, Speaker(h11.people[0].name, "kitchen", "satellite"))},
                    {"role": "user", "content": "dim the kitchen light a bit"},
                    {"role": "assistant", "content": "", "tool_calls": [call(60)]},
                    {"role": "tool", "content": "{\"success\": true}"},
                    {"role": "assistant", "content": "Dimmed it."},
                    {"role": "user", "content": "and what day is it"},
                    {"role": "assistant", "content": "It's Wednesday.", "_eval": {"must_mention": ["wednesday"]}}]}
stored = copy.deepcopy(row)


class ContextModel:
    """Checks what the request looks like, then acts a bit off the reference (45 for 60) and misses the day."""
    def __init__(self, layout): self.layout, self.seen = layout, []
    def chat(self, messages, tools=None):
        users = [m["content"] for m in messages if m["role"] == "user"]
        if self.layout == "user_context":
            assert "Now: " not in messages[0]["content"] and all(u.startswith(CONTEXT_TAG) for u in users), messages
        else:  # clock_in_user: only the clock leaves the system prompt
            assert "Now: " not in messages[0]["content"] and "\nSpeaker: " in messages[0]["content"], messages
            assert all(u.startswith(CLOCK_TAG) and " | " not in u.split("\n")[0] for u in users), messages
        self.seen.append(users[-1])
        if messages[-1]["role"] == "user" and len(users) == 1:
            return {"role": "assistant", "content": "", "tool_calls": [call(45)]}
        return {"role": "assistant", "content": "Done." if len(users) == 1 else "No idea."}


for layout, tag in (("user_context", CONTEXT_TAG + "Now: "), ("clock_in_user", CLOCK_TAG)):
    with tempfile.TemporaryDirectory() as d:
        run, llm = Run(d, "x", layout), ContextModel(layout)
        run.example(llm, row)
        run.fails.close(); run.turns_log.close()
        head = llm.seen[0].split("\n")[0]
        assert head.startswith(tag) and any(c.isdigit() for c in head) and llm.seen[0].endswith("\ndim the kitchen light a bit")
        fails = [json.loads(l) for l in open(os.path.join(d, "failures.jsonl"))]
        # turn 0 passes: the bare utterance has no number, so 45 vs 60 is within the loose tolerance (the context or
        # clock line's digits would have made it tight); turn 1 fails on the missing fact and logs the bare utterance
        assert [f["turn"] for f in fails] == [1] and fails[0]["user"] == "and what day is it" and "missing:wednesday" in fails[0]["why"], (layout, fails)
    assert row == stored, "rendering must not change the stored row"
    print(f"{layout} scoring OK")
