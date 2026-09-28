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
