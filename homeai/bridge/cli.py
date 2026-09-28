"""Talk to the model.

Mock house (no Home Assistant needed):
    python -m homeai.bridge.cli chat --mock --seed 7 --speaker Peter --room office --model homeai

Live Home Assistant:
    export HA_TOKEN=...   # long-lived access token
    python -m homeai.bridge.cli prompt --ha-url http://homeassistant.local:8123 --config configs/bridge.json
    python -m homeai.bridge.cli chat   --ha-url http://homeassistant.local:8123 --config configs/bridge.json --speaker Peter --room office

HTTP endpoint for a voice pipeline (POST /chat {"text": "...", "speaker": "Peter", "area": "office"}):
    python -m homeai.bridge.cli serve --port 8090 --ha-url ... --config configs/bridge.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..house.generator import make_house
from ..house.schema import Speaker
from ..house.snapshot import system_prompt
from ..tools import execute_tool
from ..tools_b import for_toolset
from .agent import Agent, OllamaClient, live_executor


def load_dotenv(path: str = ".env"):
    """Minimal .env reader: KEY=VALUE lines, no quoting rules beyond stripping quotes."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_provider_and_executor(a):
    if a.mock:
        house = make_house(a.seed)
        if a.everyone_away:
            for p in house.people:
                p.home, p.room = False, None
                house.entities[p.entity_id].state = "not_home"
            for e in house.entities.values():
                if e.cls in ("motion", "occupancy"):
                    e.state = "off"
        return (lambda: house), for_toolset(a.toolset)[1], house
    from .ha import HAClient, build_house
    load_dotenv()
    token = a.token or os.environ.get("HA_TOKEN")
    user, pw = os.environ.get("HA_USERNAME"), os.environ.get("HA_PASSWORD")
    url = a.ha_url or os.environ.get("HA_URL") or (f"https://{os.environ['HA_HOSTNAME']}" if os.environ.get("HA_HOSTNAME") else None)
    if not url:
        sys.exit("set --ha-url or HA_URL/HA_HOSTNAME")
    if not token and not (user and pw):
        sys.exit("set HA_TOKEN, or HA_USERNAME and HA_PASSWORD (a .env file in the project root is read)")
    cfg = json.load(open(a.config)) if a.config and os.path.exists(a.config) else {}
    cfg["toolset"] = a.toolset
    ha = HAClient(url, token, username=user, password=pw)
    registry = None
    try:
        from .ws import fetch_registries
        registry = fetch_registries(ha.url, ha.token)
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not read HA registries over websocket ({e}); falling back to name heuristics", file=sys.stderr)

    def provider():
        return build_house(ha.states(), None if registry else ha.areas(), cfg, registry=registry)

    return provider, live_executor(ha, cfg), None


def speaker_from(a, house):
    if a.mock and not a.speaker and house is not None and house.home_people():
        p = random.Random(a.seed).choice(house.home_people())
        area = a.room or p.room or (house.areas[0].area_id if house.areas else None)
        return Speaker(name=p.name, area=area, channel="satellite")
    if a.room:
        return Speaker(name=a.speaker, area=a.room, channel="satellite")
    return Speaker(name=a.speaker, area=None, channel="app")


def cmd_prompt(a):
    provider, _, house = make_provider_and_executor(a)
    h = provider()
    print(for_toolset(a.toolset)[2](h, speaker_from(a, house)))
    print(f"\n[{len(h.entities)} entities, {len(h.areas)} areas, {len(h.people)} people]")


def cmd_chat(a):
    provider, executor, house = make_provider_and_executor(a)
    sp = speaker_from(a, house)
    agent = Agent(OllamaClient(a.ollama, a.model, a.num_ctx, a.temperature, think={"on": True, "off": False, "default": None}[a.think]), provider, executor, verbose=not a.quiet, toolset=a.toolset)
    print(f"model={a.model}  speaker={sp.name} area={sp.area} channel={sp.channel}   (type /reset, /prompt, /state, or /quit)")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            agent.reset()
            continue
        if text == "/prompt":
            print(for_toolset(a.toolset)[2](provider(), sp))
            continue
        if text == "/state":
            from ..tools import get_state_text
            print(get_state_text(provider()))
            continue
        try:
            print("house> " + agent.chat(text, sp))
        except Exception as e:  # noqa: BLE001
            print(f"error: {e}")


def cmd_serve(a):
    provider, executor, house = make_provider_and_executor(a)
    llm = OllamaClient(a.ollama, a.model, a.num_ctx, a.temperature, think={"on": True, "off": False, "default": None}[a.think])
    sessions: dict[str, Agent] = {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"ok": True, "model": a.model}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            sid = body.get("session", "default")
            agent = sessions.setdefault(sid, Agent(llm, provider, executor, toolset=a.toolset))
            if body.get("reset"):
                agent.reset()
            sp = Speaker(name=body.get("speaker"), area=body.get("area"), channel="satellite" if body.get("area") else "app")
            try:
                reply = agent.chat(body.get("text", ""), sp)
                out = {"reply": reply}
            except Exception as e:  # noqa: BLE001
                out = {"error": str(e)}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f"listening on http://0.0.0.0:{a.port}/chat")
    HTTPServer(("0.0.0.0", a.port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("prompt", cmd_prompt), ("chat", cmd_chat), ("serve", cmd_serve)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--mock", action="store_true")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--everyone-away", action="store_true", help="mock: mark everyone away (for check-on-the-house tests)")
        p.add_argument("--ha-url", default=None, help="default: HA_URL or https://$HA_HOSTNAME from .env")
        p.add_argument("--token", default=None)
        p.add_argument("--config", default="configs/bridge.json")
        p.add_argument("--speaker", default=None)
        p.add_argument("--room", default=None, help="area id of the satellite the speaker is using")
        p.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
        p.add_argument("--model", default=os.environ.get("HOMEAI_MODEL", "homeai"))
        p.add_argument("--num-ctx", type=int, default=8192)
        p.add_argument("--temperature", type=float, default=0.2)
        p.add_argument("--think", choices=["on", "off", "default"], default=os.environ.get("HOMEAI_THINK", "off"), help="thinking mode for hybrid models")
        p.add_argument("--quiet", action="store_true")
        p.add_argument("--toolset", choices=["a", "b"], default=os.environ.get("HOMEAI_TOOLSET", "a"), help="b: rooms-only prompt + find_devices (homeai/tools_b.py)")
        p.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
