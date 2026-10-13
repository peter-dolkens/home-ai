"""Score a Needle 3 room leaf (.cact) on a leaf test file (homeai/leaf/data.py test).

    PYTHONPATH=. python3 -m homeai.leaf.eval --data data/generated/leaf-test-bundook.jsonl \
        --needle path/to/needle --model path/to/needle3.cact [--workers 5] [--limit N] [--out results.jsonl]

The runner starts one `needle --serve` per distinct (toolset, system fact) with the room's whole toolset (Needle's own
retrieval picks the five tools it shows per turn), POSTs /reset then /complete {"input": query} for each example, and
keeps the returned function_calls, suppressed_calls, reasoning, confidence and latency. --cli runs one
`needle --prompt` process per example instead (slow: the model loads every time).

Scores:
  exact      the calls equal the reference, in order (list arguments compared as sets, numbers as numbers)
  tools      the same tool names, in order
  end_state  applying the calls to a copy of the house (leaf_execute -> sim.py) changes exactly the entities the reference
             changes, to the same state, and reads the same entities (negatives: no change and no read)
  refusal    negatives: no call at all; false_refusal: positives answered with no call
  +suppressed  the same scores counting the calls Needle withheld (suppressed_calls) as made
and, when the model reports confidence, a reliability table: accuracy by confidence bucket."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from typing import Any, Optional

from ..host import host_info
from .data import build_house
from .schema import outcome

BUCKETS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001]


# ---------------------------------------------------------------------- running Needle
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(port: int, path: str, body: dict, timeout: float = 60) -> dict:
    # compact JSON: the server's parser wants '"input":"' with no space after the colon (it reads an empty input otherwise)
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode() or "{}"
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"success": False, "error": f"bad response: {txt[:200]}", "function_calls": []}


def _files(tmp: str, tools: list, system: Optional[str]) -> list[str]:
    tp = os.path.join(tmp, "tools.json")
    with open(tp, "w") as f:
        json.dump(tools, f)
    args = ["--tools", tp]
    if system:
        sp = os.path.join(tmp, "system.txt")
        with open(sp, "w") as f:
            f.write(system)
        args += ["--system", sp]
    return args


def run_group(needle: str, model: str, rows: list[dict], threads: int, extra: list[str]) -> list[dict]:
    """One server for rows that share a toolset and system fact."""
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        port = _free_port()
        cmd = [needle, "--model", model, *_files(tmp, rows[0]["tools"], rows[0].get("system")), "--serve", "--port", str(port),
               "--threads", str(threads), *extra]
        log = open(os.path.join(tmp, "serve.log"), "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            while True:
                try:
                    _post(port, "/reset", {}, timeout=2)
                    break
                except Exception:
                    if proc.poll() is not None or time.time() - t0 > 60:
                        log.close()
                        err = open(os.path.join(tmp, "serve.log")).read()[-500:]
                        return [{"id": r["id"], "response": {"success": False, "error": f"server failed: {err}", "function_calls": []},
                                 "latency_ms": None} for r in rows]
                    time.sleep(0.1)
            for r in rows:
                _post(port, "/reset", {})
                t = time.time()
                try:
                    resp = _post(port, "/complete", {"input": r["query"]})
                except Exception as e:  # noqa: BLE001
                    resp = {"success": False, "error": str(e), "function_calls": []}
                out.append({"id": r["id"], "response": resp, "latency_ms": round((time.time() - t) * 1000)})
        finally:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
    return out


def run_cli(needle: str, model: str, r: dict, threads: int, extra: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [needle, "--model", model, *_files(tmp, r["tools"], r.get("system")), "--prompt", r["query"], "--threads", str(threads), *extra]
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        line = next((l for l in p.stdout.splitlines()[::-1] if l.strip().startswith("{")), "")
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            resp = {"success": False, "error": (p.stderr or p.stdout)[-300:], "function_calls": []}
        return {"id": r["id"], "response": resp, "latency_ms": round((time.time() - t) * 1000)}


def run_all(rows: list[dict], needle: str, model: str, workers: int, threads: int, cli: bool, extra: list[str],
            model_dir: str | None = None) -> dict[str, dict]:
    """model_dir: per-room leaves, <model_dir>/<area_id>.cact; rooms without one fall back to `model`."""
    def model_for(r):
        if model_dir:
            p = os.path.join(model_dir, f"{r['area_id']}.cact")
            if os.path.exists(p):
                return p
        return model
    res: dict[str, dict] = {}
    if cli:
        with cf.ThreadPoolExecutor(workers) as ex:
            for o in ex.map(lambda r: run_cli(needle, model_for(r), r, threads, extra), rows):
                res[o["id"]] = o
        return res
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[json.dumps([r["tools"], r.get("system")], sort_keys=True)].append(r)
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_group, needle, model_for(g[0]), g, threads, extra) for g in groups.values()]
        for f in cf.as_completed(futs):
            for o in f.result():
                res[o["id"]] = o
            done += 1
            if done % 10 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)} toolsets, {len(res)}/{len(rows)} examples", flush=True)
    return res


LEAF_RULE = ("You control only this room's devices, through the tools. Call the tool(s) that carry out the instruction, "
             "with only the arguments it states. If the instruction is not for this room or no tool fits, call no tool.")


def run_ollama(url: str, model: str, rows: list[dict], workers: int, area_models: dict | None = None) -> dict[str, dict]:
    """The same leaf test with an Ollama model in place of Needle (e.g. a Qwen 4B as the upper bound for the task).
    Needle's no-call contract is stated as a rule; the room fact goes in the same system turn."""
    import urllib.request

    def one(r):
        system = "\n".join(x for x in (r.get("system"), LEAF_RULE) if x)
        body = {"model": (area_models or {}).get(r["area_id"], model), "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 256},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": r["query"]}],
                "tools": [{"type": "function", "function": t} for t in r["tools"]]}
        t = time.time()
        try:
            req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                msg = json.loads(resp.read())["message"]
            calls = [{"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or {}} for tc in msg.get("tool_calls") or []]
            out = {"success": True, "function_calls": calls}
        except Exception as e:  # noqa: BLE001
            out = {"success": False, "error": str(e)[:300], "function_calls": []}
        return {"id": r["id"], "response": out, "latency_ms": round((time.time() - t) * 1000)}

    res = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, o in enumerate(ex.map(one, rows), 1):
            res[o["id"]] = o
            if i % 100 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} examples", flush=True)
    return res


# ---------------------------------------------------------------------- scoring
def _canon_val(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return tuple(sorted((_canon_val(x) for x in v), key=repr))
    if isinstance(v, dict):
        return tuple(sorted((k, _canon_val(x)) for k, x in v.items()))
    return v


def canon_calls(calls) -> list:
    out = []
    for c in calls or []:
        if not isinstance(c, dict):
            out.append(("?", repr(c)))
            continue
        out.append((c.get("name"), _canon_val(c.get("arguments") or {})))
    return out


def canon_changes(ch: dict) -> dict:
    return {k: (_canon_val(v[0]), _canon_val(v[1])) for k, v in (ch or {}).items()}


_HOUSES: dict[str, Any] = {}


def house_of(spec: dict):
    key = json.dumps(spec, sort_keys=True)
    if key not in _HOUSES:
        _HOUSES[key] = build_house(spec)
    return _HOUSES[key]


def score_one(row: dict, calls: list) -> dict:
    gold = row["answers"]
    neg = not gold
    exact = canon_calls(calls) == canon_calls(gold)
    tools = [c.get("name") for c in calls if isinstance(c, dict)] == [c["name"] for c in gold]
    house = house_of(row["house"])
    try:
        oc = outcome(house, row["area_id"], [c for c in calls if isinstance(c, dict)])
    except Exception as e:  # noqa: BLE001 - a malformed call is a wrong call
        oc = {"changes": {"_error": [str(e), {}]}, "reads": [], "errors": [str(e)]}
    exp = row["expected"]
    end = canon_changes(oc["changes"]) == canon_changes(exp["changes"]) and sorted(oc["reads"]) == sorted(exp["reads"])
    if neg:
        end = not oc["changes"] and not oc["reads"]
    return {"exact": exact, "tools": tools, "end_state": end, "refused": not calls, "exec_errors": oc.get("errors", []),
            "pred_changes": oc["changes"], "pred_reads": oc["reads"]}


def bucket(c: Optional[float]) -> Optional[str]:
    if c is None:
        return None
    for lo, hi in zip(BUCKETS, BUCKETS[1:]):
        if lo <= c < hi:
            return f"{lo:.1f}-{min(hi, 1.0):.1f}"
    return None


def summarise(results: list[dict]) -> dict:
    def agg(rs):
        n = len(rs)
        if not n:
            return {"n": 0}
        pos = [r for r in rs if not r["negative"]]
        neg = [r for r in rs if r["negative"]]
        f = lambda k, xs: round(sum(1 for r in xs if r[k]) / len(xs), 3) if xs else None  # noqa: E731
        lat = sorted(r["latency_ms"] for r in rs if r.get("latency_ms") is not None)
        return {"n": n, "exact": f("exact", rs), "end_state": f("end_state", rs), "tools": f("tools", rs),
                "pos_n": len(pos), "pos_exact": f("exact", pos), "pos_end_state": f("end_state", pos), "false_refusal": f("refused", pos),
                "neg_n": len(neg), "refusal": f("refused", neg),
                "exact+suppressed": f("exact_s", rs), "end_state+suppressed": f("end_state_s", rs),
                "errors": sum(1 for r in rs if r.get("error")),
                "latency_ms_p50": lat[len(lat) // 2] if lat else None, "latency_ms_p95": lat[int(len(lat) * 0.95)] if lat else None}

    out: dict[str, Any] = {"overall": agg(results)}
    for key in ("source", "style", "negative_type", "first_tool", "n_calls"):
        groups = collections.defaultdict(list)
        for r in results:
            groups[str(r.get(key))].append(r)
        out[f"by_{key}"] = {k: agg(v) for k, v in sorted(groups.items())}
    confs = [r for r in results if r.get("confidence") is not None]
    if confs:
        rel = collections.defaultdict(list)
        for r in confs:
            rel[bucket(r["confidence"])].append(r)
        out["reliability"] = {b: {"n": len(v), "mean_conf": round(sum(r["confidence"] for r in v) / len(v), 3),
                                  "exact": round(sum(r["exact"] for r in v) / len(v), 3),
                                  "end_state": round(sum(r["end_state"] for r in v) / len(v), 3)} for b, v in sorted(rel.items())}
        n = len(confs)
        out["ece_end_state"] = round(sum(len(v) / n * abs(sum(r["confidence"] for r in v) / len(v) - sum(r["end_state"] for r in v) / len(v))
                                         for v in rel.values()), 3)
    return out


def score(rows: list[dict], responses: dict[str, dict]) -> list[dict]:
    out = []
    for row in rows:
        o = responses.get(row["id"]) or {"response": {"success": False, "error": "missing", "function_calls": []}, "latency_ms": None}
        resp = o["response"] or {}
        calls = resp.get("function_calls") or []
        sup = resp.get("suppressed_calls") or []
        s = score_one(row, calls)
        s2 = score_one(row, calls + sup) if sup else s
        meta = row.get("meta") or {}
        out.append({"id": row["id"], "area_id": row["area_id"], "query": row["query"], "gold": row["answers"], "pred": calls, "suppressed": sup,
                    "reasoning": resp.get("reasoning"), "confidence": resp.get("confidence"), "error": resp.get("error") if not resp.get("success", True) else None,
                    "latency_ms": o.get("latency_ms"), "negative": not row["answers"], "negative_type": meta.get("negative"),
                    "source": meta.get("source"), "style": meta.get("style"), "n_calls": len(row["answers"]),
                    "first_tool": (row["answers"][0]["name"].split("_")[0] if row["answers"][0]["name"].startswith("read_") else row["answers"][0]["name"]) if row["answers"] else "(none)",
                    "exact": s["exact"], "tools": s["tools"], "end_state": s["end_state"], "refused": s["refused"],
                    "exact_s": s2["exact"], "end_state_s": s2["end_state"], "exec_errors": s["exec_errors"]})
    return out


def print_summary(summ: dict, label: str):
    o = summ["overall"]
    print(f"\n== {label}: {o['n']} examples")
    print(f"exact {o['exact']}  end-state {o['end_state']}  tools {o['tools']}  | positives {o['pos_n']}: exact {o['pos_exact']} "
          f"end-state {o['pos_end_state']} false-refusal {o['false_refusal']}  | negatives {o['neg_n']}: refusal {o['refusal']}  "
          f"| +suppressed: exact {o['exact+suppressed']} end-state {o['end_state+suppressed']}  | errors {o['errors']}  "
          f"| latency p50 {o['latency_ms_p50']} ms p95 {o['latency_ms_p95']} ms")
    for key in ("by_first_tool", "by_source", "by_negative_type", "by_n_calls"):
        print(f"-- {key[3:]}")
        for k, v in summ[key].items():
            print(f"   {k:<22} n={v['n']:<4} exact={v['exact']}  end_state={v['end_state']}  refusal={v['refusal']}  false_refusal={v['false_refusal']}")
    if "reliability" in summ:
        print(f"-- reliability (confidence bucket: n, mean confidence, exact, end-state); ECE(end-state) {summ['ece_end_state']}")
        for b, v in summ["reliability"].items():
            print(f"   {b:<8} n={v['n']:<4} conf={v['mean_conf']}  exact={v['exact']}  end_state={v['end_state']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--needle", help="the needle CLI binary")
    ap.add_argument("--model", help="the .cact weights (the fallback when --model-dir has no file for a room), or the Ollama model with --ollama")
    ap.add_argument("--ollama", default=None, help="score an Ollama model instead of Needle, e.g. http://localhost:11434")
    ap.add_argument("--area-models", default=None, help="with --ollama: JSON file area_id -> Ollama model (e.g. room-type leaves)")
    ap.add_argument("--model-dir", default=None, help="per-room leaves: <dir>/<area_id>.cact")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2, help="--threads per needle process")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cli", action="store_true", help="one --prompt process per example instead of --serve")
    ap.add_argument("--extra", default="", help="extra needle flags, e.g. '--depth 12'")
    ap.add_argument("--out", default=None, help="per-example results (.jsonl); the summary goes next to it (.summary.json)")
    a = ap.parse_args()
    for path in a.data:
        rows = [json.loads(l) for l in open(path)]
        if a.limit:
            rows = rows[:a.limit]
        t = time.time()
        responses = (run_ollama(a.ollama, a.model, rows, a.workers, json.load(open(a.area_models)) if a.area_models else None) if a.ollama
                     else run_all(rows, a.needle, a.model, a.workers, a.threads, a.cli, a.extra.split(), a.model_dir))
        results = score(rows, responses)
        summ = summarise(results)
        summ["wall_s"] = round(time.time() - t, 1)
        summ.update(host_info())  # latency is per host
        label = os.path.basename(path)
        print_summary(summ, label)
        if a.out:
            base = a.out if len(a.data) == 1 else a.out.replace(".jsonl", f"-{label}")
            with open(base, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(base.replace(".jsonl", "") + ".summary.json", "w") as f:
                json.dump(summ, f, indent=1)


if __name__ == "__main__":
    main()
