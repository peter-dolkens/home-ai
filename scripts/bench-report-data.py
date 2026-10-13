#!/usr/bin/env python3
"""Collect the TODO-app benchmark (Qwen via Vineyard + Claude runs via claude-bench.py) into one JSON
for the report page: per-run summary, per-API-call timings from the metrics proxy, and the transcript.

  python3 scripts/bench-report-data.py > outputs/claude-metrics/bench-report.json
"""
import json, os, glob, datetime

M = "outputs/claude-metrics"
LOCAL = os.path.expanduser("~/Projects/local-ai")
QWEN_SESSION = os.path.expanduser(
    "~/.claude/projects/-Users-peter-dolkens-Projects-local-ai/ed27d3bc-5ad7-4d38-b2b7-edcbdb1551b3.jsonl")
CLIP_IN, CLIP_OUT = 6000, 2500


def clip(s, n):
    s = s if isinstance(s, str) else json.dumps(s, indent=1)
    return s if len(s) <= n else s[:n] + f"\n… [{len(s) - n:,} more characters]"


def calls(path, t_from=0, t_to=1e18):
    out = []
    for l in open(path):
        r = json.loads(l)
        if r["method"] != "POST" or not r["path"].startswith("/v1/messages") or not (t_from <= r["t_start"] < t_to):
            continue
        out.append(r)
    t0 = out[0]["t_start"]
    return [{"start": round(r["t_start"] - t0, 2), "ttft": r.get("ttft_s"), "visible": r.get("first_visible_s", r.get("ttft_s")),
             "total": r.get("total_s"), "in": r.get("input_tokens") or 0, "cache_read": r.get("cache_read_input_tokens") or 0,
             "cache_write": r.get("cache_creation_input_tokens") or 0, "out": r.get("output_tokens") or 0,
             "blocks": r.get("blocks") or [], "stop": r.get("stop_reason")} for r in out]


def block_items(content, role):
    items = []
    for b in content if isinstance(content, list) else [{"type": "text", "text": content}]:
        t = b.get("type")
        if t == "text" and b.get("text", "").strip():
            items.append({"kind": "user" if role == "user" else "text", "text": clip(b["text"], CLIP_OUT)})
        elif t == "thinking":
            items.append({"kind": "thinking", "text": clip(b.get("thinking") or "", CLIP_OUT)})
        elif t == "tool_use":
            items.append({"kind": "tool_use", "name": b["name"], "input": clip(b.get("input"), CLIP_IN)})
        elif t == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            items.append({"kind": "tool_result", "error": bool(b.get("is_error")), "text": clip(c or "", CLIP_OUT)})
    return items


def stream_transcript(path, prompt):
    items = [{"kind": "user", "text": prompt}]
    for l in open(path):
        try:
            e = json.loads(l)
        except ValueError:
            continue
        if e.get("type") in ("assistant", "user") and "message" in e:
            items += block_items(e["message"]["content"], e["type"])
    return items


def session_transcript(path):
    items, t0 = [], None
    for l in open(path):
        e = json.loads(l)
        if e.get("type") not in ("assistant", "user") or e.get("isMeta") or "message" not in e:
            continue
        ts = datetime.datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()
        t0 = t0 or ts
        for it in block_items(e["message"]["content"], e["type"]):
            it["t"] = round(ts - t0, 1)
            items.append(it)
    return items


def files(ws):
    out = []
    for root, dirs, fs in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git")]
        for f in fs:
            out.append(os.path.relpath(os.path.join(root, f), ws))
    return sorted(out)


VERDICTS = {
    "qwen-tq1": (False, "No app written. It sent every Write to /Users/peter.dolkens/Desktop/todo-app/src/App.js, a path it made up outside its working folder, so all five attempts were refused by Claude Code's edit permissions. It never tried the folder it was given and ended by pasting the code into chat. The 48 s is the time it took to give up."),
    "haiku55-max": (True, "useReducer with loadState as the lazy initializer, a save-on-change effect, validation that drops malformed entries, and cross-tab sync. Split into components with 22 Vitest tests, and it checked the app in headless Chromium. Build passes."),
    "haiku55-low": (True, "Scaffolded with create-vite into a todo-app/ subfolder. Reads storage through useState(loadTodos), a lazy initializer, and writes on change, so a reload restores the list. Production build present."),
    "sonnet55-high": (True, "useLocalStorage hook reads storage in a lazy useState initializer and writes on change, so a reload restores the list. Also syncs across tabs. Vite project, built cleanly."),
    "opus55-xhigh": (True, "TypeScript hook with a lazy initializer, validation of stored data and cross-tab sync. It wrote 15 tests (Vitest) and checked persistence itself in headless Chromium: added, completed and edited todos survived a reload. Type-check and build pass."),
    "opus55-medium": (True, "Same pattern: lazy initializer plus write-on-change hook, also persisting the active filter and syncing across tabs. Vite project, built cleanly."),
}

PROMPT = "build a frontend-only TODO app in React. Store its state in localStorage"
runs = []

# Qwen: Vineyard session; turn 1 = the API calls before the user's second message (request ids 1-16).
q_all = calls(f"{M}/todo-qwen3-coder/requests.jsonl")
q_turn1 = q_all[:16]
q_out = sum(c["out"] for c in q_turn1)
runs.append({
    "key": "qwen", "label": "Qwen3-Coder · Q3_K_XL", "sub": "UD-Q3_K_XL · cassowary RTX 3090 · Ollama 0.40.3",
    "wall": round(q_turn1[-1]["start"] + q_turn1[-1]["total"], 1), "api": round(sum(c["total"] for c in q_turn1), 1),
    "turns": len(q_turn1), "cost": 0.0, "out": q_out, "thinking": 0,
    "ctx_first": q_turn1[0]["in"], "calls": q_turn1, "calls_later": q_all[16:],
    "transcript": session_transcript(QWEN_SESSION), "files": files(f"{LOCAL}/todo-app"),
    "workspace": "~/Projects/local-ai/todo-app",
    "persists": False,
    "verdict": "Loads in a mount effect, but a second effect writes the empty initial list on the same mount, wiping saved todos. Two follow-up turns did not fix it.",
    "note": "Calls 1, 2, 4 and 5 hit a model reload (our home-ai driver unloaded it), costing about 40 s.",
})

for key, label in (("qwen-tq1", "Qwen3-Coder · TQ1_0"), ("haiku55-low", "Haiku 5.5 · low"), ("haiku55-max", "Haiku 5.5 · max"), ("sonnet55-high", "Sonnet 5.5 · high"),
                   ("opus55-medium", "Opus 5.5 · medium"), ("opus55-xhigh", "Opus 5.5 · xhigh")):
    if not os.path.exists(f"{M}/bench-claude/{key}.summary.json"):
        continue
    s = json.load(open(f"{M}/bench-claude/{key}.summary.json"))
    c = calls(f"{M}/bench-claude/{key}/requests.jsonl")
    u = s.get("usage") or {}
    local = "anthropic.com" not in s.get("upstream", "https://api.anthropic.com")
    runs.append({
        "key": key, "label": label,
        "sub": "UD-TQ1_0 · cassowary RTX 3090 · Ollama 0.40.3" if local else f"{s['model']} · effort {s['effort']} · Anthropic API",
        "wall": s["wall_s"], "api": round((s.get("duration_api_ms") or 0) / 1000, 1),
        "turns": len(c), "cost": 0.0 if local else s.get("total_cost_usd"), "out": u.get("output_tokens"),
        "thinking": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
        "ctx_first": c[0]["in"] + c[0]["cache_read"] + c[0]["cache_write"], "calls": c, "calls_later": [],
        "transcript": stream_transcript(f"{M}/bench-claude/{key}.stream.jsonl", PROMPT),
        "files": files(s["workspace"]), "workspace": s["workspace"].replace(os.path.expanduser("~"), "~"),
        "persists": VERDICTS.get(key, (None, ""))[0], "verdict": VERDICTS.get(key, (None, "Not reviewed yet."))[1],
    })

json.dump({"generated": datetime.datetime.now().isoformat(timespec="minutes"), "prompt": PROMPT, "runs": runs},
          open(1, "w"), indent=1)
