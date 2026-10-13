#!/usr/bin/env python3
"""Run one prompt through Claude Code headless per (model, effort), timing every API call.

Each run gets a fresh workspace, its own anthropic-metrics-proxy (forwarding to the real API, or to
any Anthropic-compatible server) and the session's stream-json output. Summarise afterwards with
scripts/anthropic-metrics-report.py <out>/<name>.

  python3 scripts/claude-bench.py --prompt "build ..." --workdir ~/Projects/local-ai \
      --out outputs/claude-metrics/bench1 sonnet-high:claude-sonnet-5-5:high opus-medium:claude-opus-5-5:medium
"""
import argparse, json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CLAUDE = os.path.expanduser("~/.local/bin/claude")


def wait_port(port, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
        except urllib.error.HTTPError:
            return  # proxy answered (upstream said 404 or similar)
        except Exception:
            time.sleep(0.2)
            continue
        return
    raise RuntimeError(f"proxy on :{port} did not come up")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--workdir", required=True, help="parent folder; each run gets <workdir>/bench-<name>")
    p.add_argument("--out", required=True)
    p.add_argument("--upstream", default="https://api.anthropic.com")
    p.add_argument("--port", type=int, default=11501)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra Claude Code env, e.g. ANTHROPIC_AUTH_TOKEN=ollama or CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 for a local model")
    p.add_argument("runs", nargs="+", help="name:model:effort")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{a.port}",
           "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    env.update(kv.split("=", 1) for kv in a.env)
    # One settings file per invocation, named after its first run, so earlier runs keep theirs.
    settings = os.path.join(os.path.abspath(a.out), f"settings-{a.runs[0].split(':')[0]}.json")
    json.dump({"disableClaudeAiConnectors": True, "env": env}, open(settings, "w"), indent=2)

    for spec in a.runs:
        name, model, effort = spec.split(":")
        ws = os.path.join(os.path.expanduser(a.workdir), f"bench-{name}")
        if os.path.exists(ws) and os.listdir(ws):
            sys.exit(f"{ws} is not empty; refusing to reuse it")
        os.makedirs(ws, exist_ok=True)
        run_out = os.path.abspath(os.path.join(a.out, name))
        proxy = subprocess.Popen([sys.executable, os.path.join(HERE, "anthropic-metrics-proxy.py"),
                                  "--upstream", a.upstream, "--port", str(a.port), "--out", run_out],
                                 stdout=subprocess.DEVNULL, stderr=open(os.path.join(a.out, f"{name}.proxy.err"), "w"))
        try:
            wait_port(a.port)
            cmd = [CLAUDE, "-p", "--output-format", "stream-json", "--verbose", "--settings", settings,
                   "--model", model, "--effort", effort, "--thinking-display", "summarized", "--permission-mode", "acceptEdits",
                   "--allowedTools", "Bash"]  # prompt goes on stdin: --allowedTools would swallow it
            print(f"== {name}: {model} effort={effort} in {ws}", flush=True)
            t0 = time.time()
            with open(os.path.join(a.out, f"{name}.stream.jsonl"), "w") as f:
                rc = subprocess.run(cmd, cwd=ws, input=a.prompt, text=True, stdout=f, stderr=open(os.path.join(a.out, f"{name}.stderr"), "w")).returncode
            wall = time.time() - t0
            result = {}
            for line in open(os.path.join(a.out, f"{name}.stream.jsonl")):
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if j.get("type") == "result":
                    result = j
            summary = {"name": name, "model": model, "effort": effort, "upstream": a.upstream, "settings": settings,
                       "exit": rc, "wall_s": round(wall, 1),
                       "t_start": t0, "workspace": ws,
                       **{k: result.get(k) for k in ("is_error", "duration_ms", "duration_api_ms", "num_turns",
                                                     "total_cost_usd", "usage", "modelUsage")}}
            json.dump(summary, open(os.path.join(a.out, f"{name}.summary.json"), "w"), indent=2)
            print(f"   exit={rc} wall={wall:.1f}s turns={result.get('num_turns')} cost=${result.get('total_cost_usd')}", flush=True)
        finally:
            proxy.terminate()
            proxy.wait()


if __name__ == "__main__":
    main()
