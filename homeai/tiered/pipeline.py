"""The tiered runtime loop (designs C and D): an orchestrator model with the design-B prompt and tools plus ask_rooms, and
small room leaves that each see one room's tools (homeai.leaf.schema.room_tools) and one short instruction.

  routed (C)   the orchestrator calls ask_rooms(requests: [{area_id, instruction}]) for the rooms the request is for; the
               leaves run in parallel (thread pool), their calls are applied to the house in request order
               (leaf_execute / apply_leaf), and ask_rooms returns one line per room: what changed, what was read, or
               "no action". The orchestrator then answers, or keeps going (get_state, find_devices, call_service for
               house-level entities: security, energy, water, weather).
  fan-out (D)  no routing: the user's utterance goes verbatim to every room leaf of the house before the orchestrator runs;
               the rooms that did something come back as an ask_rooms result in its context, and the orchestrator (B tools
               only) composes the answer from them.

Leaves are Ollama models, called the way homeai.leaf.eval.run_ollama calls them (system = room fact + LEAF_RULE, the room's
tools, the instruction as the user turn). Which model serves a room is pluggable (LeafPicker): one model for every room,
a map from room kind (area.kind: bedroom | office | kitchen | living | bath | entry | utility | garage | outdoor) to model,
and/or area_id -> model; each model may sit on its own Ollama (MODEL@URL). The orchestrator is any make_client URL / model.

    PYTHONPATH=. python3 -m homeai.tiered.pipeline --house bundook_projected --say "turn off the lights in Pete's office" \
        --ollama http://URL1:11434 --model qwen3.5:9b --leaf qwen3.5:4b --leaf-url http://URL2:11434 -v
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from ..bridge.agent import make_client
from ..house.schema import GROUPS, House
from ..leaf.eval import LEAF_RULE
from ..leaf.schema import Room, apply_leaf, room_model, room_tools, system_fact
from ..tools_b import TOOLS_B, execute_tool_b
from .schema import TOOLS_ORCH, leaf_rooms, room_result, system_prompt_fanout, system_prompt_orch, unknown_room_line

KINDS = ("bedroom", "office", "kitchen", "living", "bath", "entry", "utility", "garage", "outdoor")
LEAF_PREDICT = 256  # as homeai.leaf.eval.run_ollama


# ---------------------------------------------------------------------- leaves
class OllamaLeaf:
    """One room leaf served by Ollama: (room, instruction, tools) -> {"calls", "text", "error", "latency_s", "model"}."""

    def __init__(self, url: str, model: str, timeout: float = 120, num_ctx: Optional[int] = None, think: Optional[bool] = False):
        self.url, self.model, self.timeout, self.num_ctx, self.think = url.rstrip("/"), model, timeout, num_ctx, think

    @property
    def label(self) -> str:
        return f"{self.model}@{self.url}"

    def __call__(self, room: Room, instruction: str, tools: list[dict]) -> dict:
        system = "\n".join(x for x in (system_fact(room), LEAF_RULE) if x)
        opts: dict[str, Any] = {"temperature": 0, "num_predict": LEAF_PREDICT}
        if self.num_ctx:
            opts["num_ctx"] = self.num_ctx
        body: dict[str, Any] = {"model": self.model, "stream": False, "options": opts,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": instruction}],
                                "tools": [{"type": "function", "function": t} for t in tools]}
        if self.think is not None:
            body["think"] = self.think
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t = time.time()
        for attempt in range(3):  # Ollama restarts / model swaps surface as connection errors
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    msg = json.loads(r.read())["message"]
                calls = []
                for tc in msg.get("tool_calls") or []:
                    args = tc["function"].get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    calls.append({"name": tc["function"]["name"], "arguments": args})
                return {"calls": calls, "text": msg.get("content") or "", "error": None, "latency_s": time.time() - t, "model": self.label}
            except (urllib.error.URLError, ConnectionError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                if attempt == 2 or (isinstance(e, urllib.error.HTTPError) and e.code < 500):
                    return {"calls": [], "text": "", "error": f"leaf transport: {str(e)[:200]}", "latency_s": time.time() - t,
                            "model": self.label}
                time.sleep(2 * (attempt + 1))
        return {"calls": [], "text": "", "error": "unreachable", "latency_s": time.time() - t, "model": self.label}


class OracleLeaf:
    """Returns the reference leaf calls for (area_id, instruction) (homeai/tiered/data.py records them as "leaf_gold");
    anything else gets no call. For harness sanity checks without a model."""

    label = "oracle"

    def __init__(self, gold: Optional[dict[tuple[str, str], list]] = None):
        self.gold = dict(gold or {})

    def add(self, rows: list[dict]):
        for g in rows or []:
            self.gold[(g["area_id"], g["instruction"])] = g["calls"]

    def __call__(self, room: Room, instruction: str, tools: list[dict]) -> dict:
        calls = self.gold.get((room.area_id, instruction), [])
        return {"calls": json.loads(json.dumps(calls)), "text": "", "error": None, "latency_s": 0.0, "model": self.label}


LeafFn = Callable[[Room, str, list], dict]


class LeafPicker:
    """Which leaf serves a room: by_area[area_id], else by_kind[area.kind], else default."""

    def __init__(self, default: Optional[LeafFn] = None, by_kind: Optional[dict[str, LeafFn]] = None, by_area: Optional[dict[str, LeafFn]] = None):
        self.default, self.by_kind, self.by_area = default, dict(by_kind or {}), dict(by_area or {})

    def __call__(self, house: House, area_id: str) -> Optional[LeafFn]:
        if area_id in self.by_area:
            return self.by_area[area_id]
        a = house.area(area_id)
        if a and a.kind in self.by_kind:
            return self.by_kind[a.kind]
        return self.default

    def describe(self) -> dict:
        lab = lambda f: getattr(f, "label", repr(f)) if f else None  # noqa: E731
        return {"default": lab(self.default), "by_kind": {k: lab(v) for k, v in self.by_kind.items()},
                "by_area": {k: lab(v) for k, v in self.by_area.items()}}


def parse_leaf_args(default: Optional[str], url: str, kinds: list[str] = (), areas: list[str] = (), timeout: float = 120,
                    num_ctx: Optional[int] = None) -> LeafPicker:
    """--leaf MODEL[@URL], --leaf-kind KIND[,KIND]=MODEL[@URL], --leaf-area AREA[,AREA]=MODEL[@URL]; one client per (url, model)."""
    cache: dict[tuple[str, str], OllamaLeaf] = {}

    def leaf(spec: str) -> OllamaLeaf:
        model, _, u = spec.partition("@")
        key = ((u or url).rstrip("/"), model)
        if key not in cache:
            cache[key] = OllamaLeaf(key[0], model, timeout, num_ctx)
        return cache[key]

    def mapping(items, check=None):
        out = {}
        for it in items or []:
            keys, _, spec = it.partition("=")
            if not spec:
                raise SystemExit(f"expected NAME[,NAME]=MODEL[@URL], got {it!r}")
            for k in keys.split(","):
                if check and k not in check:
                    raise SystemExit(f"unknown room kind {k!r} (one of {', '.join(check)})")
                out[k] = leaf(spec)
        return out

    return LeafPicker(leaf(default) if default else None, mapping(kinds, KINDS), mapping(areas))


# ---------------------------------------------------------------------- running leaves
def _requests(args: Any) -> list[dict]:
    """ask_rooms arguments as models emit them: {"requests": [...]}, a JSON string of that, or area_id lists."""
    reqs = args.get("requests") if isinstance(args, dict) else args
    if isinstance(reqs, str):
        try:
            reqs = json.loads(reqs)
        except json.JSONDecodeError:
            reqs = []
    if isinstance(reqs, dict):
        reqs = [reqs]
    out = []
    for r in reqs or []:
        if not isinstance(r, dict):
            continue
        ins = str(r.get("instruction") or "").strip()
        aids = r.get("area_id")
        for aid in ([aids] if isinstance(aids, str) else aids or []):
            out.append({"area_id": str(aid), "instruction": ins})
    return out


def run_leaves(house: House, jobs: list[dict], picker: LeafPicker, pool: cf.Executor) -> tuple[list[dict], dict]:
    """jobs: [{area_id, instruction}]. Leaves run in parallel; their calls are applied to the house afterwards, in job order
    (rooms are disjoint, so the order only matters for two jobs on one room). Returns (per-job results, timing stats)."""
    prepared = []
    for j in jobs:
        aid, ins = j["area_id"], j["instruction"]
        room = room_model(house, aid) if aid in house.area_ids() and aid not in GROUPS else None
        tools = room_tools(house, aid, room) if room else []
        leaf = picker(house, aid) if tools else None
        prepared.append((aid, ins, room, tools, leaf))
    t0 = time.time()
    futs = {i: pool.submit(leaf, room, ins, tools) for i, (aid, ins, room, tools, leaf) in enumerate(prepared) if leaf and ins}
    outs = {i: f.result() for i, f in futs.items()}
    wall = time.time() - t0 if futs else 0.0
    results = []
    for i, (aid, ins, room, tools, leaf) in enumerate(prepared):
        if not tools:
            results.append({"area_id": aid, "instruction": ins, "calls": [], "line": unknown_room_line(house, aid), "acted": False})
            continue
        if leaf is None:
            results.append({"area_id": aid, "instruction": ins, "calls": [], "line": f"{aid}: no room assistant configured", "acted": False})
            continue
        o = outs.get(i) or {"calls": [], "error": None, "latency_s": 0.0}
        calls = [c for c in o["calls"] if isinstance(c, dict)]
        applied = apply_leaf(house, aid, calls, room) if calls else None
        line = room_result(house, room, calls, applied, o.get("error"))
        results.append({"area_id": aid, "instruction": ins, "calls": calls, "line": line, "latency_s": o.get("latency_s", 0.0),
                        "model": o.get("model"), "error": o.get("error"), "acted": bool(calls) or bool(o.get("error"))})
    stats = {"leaf_calls": len(futs), "leaf_s": sum(outs[i].get("latency_s") or 0.0 for i in outs), "leaf_wall_s": wall,
             "leaf_errors": sum(1 for i in outs if outs[i].get("error"))}
    return results, stats


def ask_rooms_text(results: list[dict]) -> str:
    return "\n".join(r["line"] for r in results) if results else "no requests"


def fanout_text(results: list[dict]) -> str:
    """Only the rooms that did something (or failed), plus how many did nothing."""
    acted = [r["line"] for r in results if r["acted"]]
    rest = len(results) - len(acted)
    if rest:
        acted.append(f"no action in the other {rest} rooms" if acted else f"no action in any of the {rest} rooms")
    return "\n".join(acted)


def _args(fn: dict) -> Any:
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args


def room_level_targets(house: House, args: dict) -> list[str]:
    """Real rooms an orchestrator call_service reaches into (routing leaks: those devices belong to a room leaf)."""
    t = args.get("target") if isinstance(args, dict) else None
    if not isinstance(t, dict):
        return []
    rooms = set()
    for aid in ([t.get("area_id")] if isinstance(t.get("area_id"), str) else t.get("area_id") or []):
        if aid in house.area_ids():
            rooms.add(aid)
    for eid in ([t.get("entity_id")] if isinstance(t.get("entity_id"), str) else t.get("entity_id") or []):
        e = house.entities.get(eid)
        if e is not None and e.area not in GROUPS:
            rooms.add(e.area)
    return sorted(rooms)


# ---------------------------------------------------------------------- the loop
class Pipeline:
    """orch: an object with chat(messages, tools) -> assistant message (make_client, or an oracle); picker: LeafPicker.
    mode "routed" (ask_rooms) or "fanout" (every room leaf gets the utterance first)."""

    def __init__(self, orch, picker: LeafPicker, mode: str = "routed", max_rounds: int = 4, workers: int = 8, verbose: bool = False):
        assert mode in ("routed", "fanout"), mode
        self.orch, self.picker, self.mode, self.max_rounds, self.verbose = orch, picker, mode, max_rounds, verbose
        self.pool = cf.ThreadPoolExecutor(max(1, workers))

    def prompt(self, house: House, speaker) -> str:
        return (system_prompt_fanout if self.mode == "fanout" else system_prompt_orch)(house, speaker)

    @property
    def tools(self) -> list[dict]:
        return TOOLS_B if self.mode == "fanout" else TOOLS_ORCH

    def close(self):
        self.pool.shutdown(wait=False)

    def _log(self, s: str):
        if self.verbose:
            print(s, flush=True)

    def run(self, ctx: list[dict], house: House, user_text: str) -> tuple[str, list[dict], dict]:
        """ctx: Ollama-format messages ending with the user turn (system prompt first). Mutates house.
        Returns (spoken answer, calls made, stats)."""
        t_turn = time.time()
        st = {"orch_calls": 0, "orch_s": 0.0, "leaf_calls": 0, "leaf_s": 0.0, "leaf_wall_s": 0.0, "leaf_errors": 0, "ask_rooms": 0,
              "rooms_asked": 0, "rooms_acted": 0, "room_call_service": 0, "leaf_latencies": [], "ask_walls": [], "orch_latencies": []}
        msgs = list(ctx)
        calls_made: list[dict] = []

        def leaves(jobs):
            res, s = run_leaves(house, jobs, self.picker, self.pool)
            for k in ("leaf_calls", "leaf_s", "leaf_wall_s", "leaf_errors"):
                st[k] += s[k]
            st["leaf_latencies"] += [r["latency_s"] for r in res if r.get("latency_s") is not None and r.get("model")]
            if s["leaf_calls"]:
                st["ask_walls"].append(s["leaf_wall_s"])
            st["ask_rooms"] += 1
            st["rooms_asked"] += len(jobs)
            st["rooms_acted"] += sum(1 for r in res if r["calls"])
            for r in res:
                self._log(f"     [{r['area_id']}] {r['instruction']!r} -> {json.dumps(r['calls'])[:300]}")
            return res

        if self.mode == "fanout":
            res = leaves([{"area_id": a, "instruction": user_text} for a in leaf_rooms(house)])
            text = fanout_text(res)
            args = {"requests": [{"area_id": "all", "instruction": user_text}]}
            calls_made.append({"name": "ask_rooms", "arguments": args, "leaf": [{k: r[k] for k in ("area_id", "calls")} for r in res if r["calls"]]})
            msgs += [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "ask_rooms", "arguments": args}}]},
                     {"role": "tool", "content": text, "tool_name": "ask_rooms"}]
            self._log("  <- fan-out: " + text.replace("\n", "\n     "))
        answer = ""
        for _ in range(self.max_rounds):
            t = time.time()
            msg = self.orch.chat(msgs, self.tools)
            st["orch_calls"] += 1
            st["orch_s"] += time.time() - t
            st["orch_latencies"].append(time.time() - t)
            tcs = msg.get("tool_calls") or []
            msgs.append({"role": "assistant", "content": msg.get("content", "") or "", **({"tool_calls": tcs} if tcs else {})})
            if not tcs:
                answer = (msg.get("content") or "").strip()
                break
            for tc in tcs:
                fn = tc.get("function", {})
                name, args = fn.get("name", ""), _args(fn)
                self._log(f"  -> {name}({json.dumps(args)[:400]})")
                if name == "ask_rooms" and self.mode == "routed":
                    res = leaves(_requests(args))
                    content = ask_rooms_text(res)
                    calls_made.append({"name": name, "arguments": args, "leaf": [{k: r[k] for k in ("area_id", "calls")} for r in res]})
                else:
                    if name == "call_service" and room_level_targets(house, args):
                        st["room_call_service"] += 1
                    content = execute_tool_b(house, name, args if isinstance(args, dict) else {})
                    calls_made.append({"name": name, "arguments": args})
                self._log("  <- " + content[:600].replace("\n", "\n     "))
                msgs.append({"role": "tool", "content": content, "tool_name": name})
        st["turn_s"] = time.time() - t_turn
        return answer, calls_made, st


def add_args(ap: argparse.ArgumentParser):
    """Orchestrator and leaf options shared by the pipeline demo and homeai/tiered/eval.py."""
    ap.add_argument("--model", required=True, help="orchestrator model (make_client), or 'oracle' in the eval")
    ap.add_argument("--ollama", default="http://localhost:11434", help="orchestrator URL (an OpenAI-compatible server when it ends in /v1)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--leaf", default=None, help="default leaf MODEL[@URL] for every room (or 'oracle' in the eval)")
    ap.add_argument("--leaf-url", default="http://localhost:11434", help="Ollama URL for leaves given without @URL")
    ap.add_argument("--leaf-kind", action="append", default=[], help=f"KIND[,KIND]=MODEL[@URL]; kinds: {', '.join(KINDS)}")
    ap.add_argument("--leaf-area", action="append", default=[], help="AREA_ID[,AREA_ID]=MODEL[@URL]")
    ap.add_argument("--leaf-num-ctx", type=int, default=None, help="leaf context window (default: the model's)")
    ap.add_argument("--leaf-timeout", type=float, default=120)
    ap.add_argument("--workers", type=int, default=8, help="leaf calls in flight at once")
    ap.add_argument("--fanout", action="store_true", help="blind fan-out (design D): every room leaf gets the utterance, no routing")
    ap.add_argument("--max-rounds", type=int, default=4, help="orchestrator rounds per turn (e2e.py uses 4)")
    ap.add_argument("--num-predict", type=int, default=2048,
                    help="orchestrator token cap per turn (bridge default 512; an ask_rooms to every room of a big house is ~1.2k tokens)")


def orchestrator_from_args(a):
    from ..bridge import agent
    agent.NUM_PREDICT = a.num_predict  # OllamaClient reads the module cap at call time
    return make_client(a.ollama, a.model, a.num_ctx, temperature=0.0, think=False)


def picker_from_args(a) -> LeafPicker:
    if not (a.leaf or a.leaf_kind or a.leaf_area):
        raise SystemExit("give --leaf and/or --leaf-kind / --leaf-area")
    return parse_leaf_args(a.leaf, a.leaf_url, a.leaf_kind, a.leaf_area, a.leaf_timeout, a.leaf_num_ctx)


def main():
    from ..house.bighouse import big_spec_house
    from ..house.generator import make_house
    from ..house.schema import Speaker
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--house", default="bundook_projected", help="bundook_full | bundook_projected | random:N | small")
    ap.add_argument("--seed", type=int, default=9_100_001)
    ap.add_argument("--speaker", default=None, help="NAME@AREA_ID (a satellite), default unknown via app")
    ap.add_argument("--say", action="append", required=True, help="user utterance (repeat for a multi-turn chat)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.house == "small":
        house = make_house(a.seed)
    elif a.house.startswith("random"):
        house = big_spec_house({"kind": "random", "seed": a.seed, "n": int(a.house.split(":")[1]) if ":" in a.house else 10000})
    else:
        house = big_spec_house({"kind": a.house, "seed": a.seed})
    sp = None
    if a.speaker:
        n, _, ar = a.speaker.partition("@")
        sp = Speaker(n or None, ar or None, "satellite" if ar else "app")
    orch = orchestrator_from_args(a)
    pipe = Pipeline(orch, picker_from_args(a), "fanout" if a.fanout else "routed", a.max_rounds, a.workers, verbose=a.verbose)
    print(f"{len(house.entities)} entities, {len(leaf_rooms(house))} leaf rooms")
    history: list[dict] = []
    for u in a.say:
        history.append({"role": "user", "content": u})
        ctx = [{"role": "system", "content": pipe.prompt(house, sp)}] + history
        text, calls, st = pipe.run(ctx, house, u)
        history.append({"role": "assistant", "content": text})
        print(f"user: {u}\nassistant: {text}\n  ({st['orch_calls']} orchestrator calls {st['orch_s']:.1f}s, {st['leaf_calls']} leaf calls "
              f"{st['leaf_s']:.1f}s, parallel leaf wall {st['leaf_wall_s']:.1f}s, turn {st['turn_s']:.1f}s)")
    pipe.close()


if __name__ == "__main__":
    main()
