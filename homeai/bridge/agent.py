"""The runtime loop: build the system prompt from live (or mock) state, call the
model through Ollama, execute tool calls, repeat until the model answers."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from ..house.schema import House, Speaker
from ..house.snapshot import relayout_messages, split_context, system_prompt
from ..tools import TOOLS, get_state_text
from ..tools_b import find_devices_text, for_toolset, get_state_b_text


# Cap on generated tokens per model turn. Voice replies and tool calls are well under this; without it a model
# that never emits a stop token (seen with LFM2.5 230M and Gemma 4) runs into the 120 s timeout and every retry.
NUM_PREDICT = int(os.environ.get("HOMEAI_NUM_PREDICT", "512"))


class OllamaClient:
    def __init__(self, url: str = "http://localhost:11434", model: str = "homeai", num_ctx: int = 8192, temperature: float = 0.2,
                 timeout: float = 120, think: Optional[bool] = False):
        self.url, self.model, self.num_ctx, self.temperature, self.timeout = url.rstrip("/"), model, num_ctx, temperature, timeout
        self.think = think  # hybrid-thinking models (Qwen3/3.5): False = answer directly, None = model default
        self.last: dict = {}  # the last call's server counters (prompt_eval_count / _duration ...: homeai.eval.prefix_bench)

    def chat(self, messages: list[dict], tools: Optional[list] = TOOLS) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": NUM_PREDICT}}
        if self.think is not None:
            body["think"] = self.think
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        # Ollama restarts (model swaps, OOM) surface as connection errors; retry rather than fail the turn. An HTTP 500
        # is usually Ollama failing to parse the model's own tool call (3 Oct: the 2B's "<parameter> closed by
        # </function>"): a model error, so only two quick retries (temperature may vary the output), not the 75 s of
        # backoff that inflated those turns' timings.
        http500 = 0
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode())
                self.last = {k: v for k, v in resp.items() if k.endswith(("_count", "_duration"))}
                return resp["message"]
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if isinstance(e, urllib.error.HTTPError):
                    if e.code < 500:
                        raise
                    http500 += 1
                    if http500 > 2:
                        raise
                    time.sleep(0.5)
                    continue
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))


PY_CALL_START, PY_CALL_END = "<|tool_call_start|>", "<|tool_call_end|>"
_LITERALS = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}


def _literal(node):
    """ast.literal_eval that also takes JSON's true / false / null (models mix the two styles in call arguments)."""
    import ast
    if isinstance(node, ast.Name) and node.id in _LITERALS:
        return _LITERALS[node.id]
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def parse_pythonic_calls(text: str) -> tuple[list[dict], str]:
    """LFM2's tool calls, "<|tool_call_start|>[name(a=1, b=['x']), other()]<|tool_call_end|>", as Ollama-style tool calls
    plus the remaining text. mlx_lm.server (0.31.3) only recognises this format when the chat template carries
    <|tool_list_start|>, which LFM2.5's does not, and its regex parser splits list arguments at their commas, so the
    client parses the Python call syntax itself (2 Oct: on falcon every LFM call came back as plain text)."""
    import ast
    calls, rest, i = [], [], 0
    while True:
        s = text.find(PY_CALL_START, i)
        if s < 0:
            rest.append(text[i:])
            break
        e = text.find(PY_CALL_END, s)
        e = len(text) if e < 0 else e
        rest.append(text[i:s])
        try:
            tree = ast.parse(text[s + len(PY_CALL_START):e].strip(), mode="eval").body
            for c in (tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                    calls.append({"function": {"name": c.func.id, "arguments": {k.arg: _literal(k.value) for k in c.keywords if k.arg}}})
        except (SyntaxError, ValueError):
            rest.append(text[s:e + len(PY_CALL_END)])  # not parseable: leave it as text
        i = e + len(PY_CALL_END)
    return calls, "".join(rest).strip()


class OpenAIClient:
    """Same interface as OllamaClient for OpenAI-compatible servers (vLLM, SGLang, llama.cpp), used to compare
    base models Ollama cannot run yet. url like http://localhost:8000/v1; extra goes into the request body
    (e.g. {"chat_template_kwargs": {"reasoning_effort": "low"}})."""

    def __init__(self, url: str, model: str, temperature: float = 0.0, timeout: float = 300, max_tokens: int = 4096, extra: Optional[dict] = None):
        self.url, self.model, self.temperature, self.timeout, self.max_tokens, self.extra = url.rstrip("/"), model, temperature, timeout, max_tokens, extra or {}
        self.think = None
        self.last: dict = {}  # the last call's "usage" (prompt_tokens, prompt_tokens_details.cached_tokens ...)

    def chat(self, messages: list[dict], tools: Optional[list] = TOOLS) -> dict:
        out, pending = [], []  # Ollama-style history -> OpenAI (tool calls need ids that tool results refer to)
        n = 0
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                tcs = []
                for tc in m["tool_calls"]:
                    n += 1
                    fn = tc["function"]
                    args = fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
                    tcs.append({"id": f"call_{n}", "type": "function", "function": {"name": fn["name"], "arguments": args}})
                pending = [t["id"] for t in tcs]
                out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else f"call_{n}", "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        body = {"model": self.model, "messages": out, "temperature": self.temperature, "max_tokens": self.max_tokens, **self.extra}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode())
                msg, self.last = resp["choices"][0]["message"], resp.get("usage") or {}
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if attempt == 3 or (isinstance(e, urllib.error.HTTPError) and e.code < 500):
                    raise
                time.sleep(5 * (attempt + 1))
        calls = []
        for tc in msg.get("tool_calls") or []:
            args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                args = {}
            calls.append({"function": {"name": tc["function"]["name"], "arguments": args}})
        content = msg.get("content") or ""
        if not calls and PY_CALL_START in content:  # LFM2 calls the server returned as text
            calls, content = parse_pythonic_calls(content)
        res = {"role": "assistant", "content": content}
        if calls:
            res["tool_calls"] = calls
        if msg.get("reasoning_content"):
            res["thinking"] = msg["reasoning_content"]
        return res


def make_client(url: str, model: str, num_ctx: int = 8192, temperature: float = 0.0, think: Optional[bool] = False):
    """OllamaClient, or OpenAIClient when url ends in /v1 (extra request fields from HOMEAI_OPENAI_EXTRA, JSON)."""
    import os
    if url.rstrip("/").endswith("/v1"):
        return OpenAIClient(url, model, temperature, extra=json.loads(os.environ.get("HOMEAI_OPENAI_EXTRA", "{}")))
    return OllamaClient(url, model, num_ctx, temperature=temperature, think=think)


class Agent:
    """house_provider() -> House (fresh each turn); executor(house, name, args) -> tool result string.
    layout: the prompt layout the model was trained on (homeai/prompt_layout.py; v10 "stable_prefix", v11 "user_context",
    v12 "clock_in_user"). Each user turn in the history keeps the context lines (Now / Speaker / People) it was sent with
    ("_context"), so a user_context or clock_in_user conversation re-sends earlier turns unchanged and only the latest
    turn carries the new context (clock_in_user: the new clock; Speaker and People are the system prompt's)."""

    def __init__(self, llm: OllamaClient, house_provider: Callable[[], House],
                 executor: Callable[[House, str, dict], str], max_rounds: int = 4, verbose: bool = False, toolset: str = "a",
                 layout: Optional[str] = None):
        self.llm, self.house_provider, self.executor, self.max_rounds, self.verbose = llm, house_provider, executor, max_rounds, verbose
        self.tools, _, self.prompt = for_toolset(toolset)  # "b": rooms-only prompt and the homeai/tools_b.py schemas
        self.layout = layout or "classic"
        self.history: list[dict] = []  # user/assistant/tool turns of the current conversation

    def reset(self):
        self.history = []

    def chat(self, text: str, speaker: Optional[Speaker]) -> str:
        house = self.house_provider()
        user = {"role": "user", "content": text}
        self.history.append(user)
        system = None
        for _ in range(self.max_rounds):
            # classic: rebuilt every round, as v1-v9 ran. Otherwise once per request, so a clock tick between rounds
            # cannot change the context in front of this request's history (the whole prompt stays cached)
            if system is None or self.layout == "classic":
                system = self.prompt(house, speaker)  # built classic, then rendered in the model's layout
                user["_context"] = split_context(system)[1]
            messages = relayout_messages([{"role": "system", "content": system}] + self.history, self.layout)
            msg = self.llm.chat(messages, self.tools)
            calls = msg.get("tool_calls") or []
            self.history.append({"role": "assistant", "content": msg.get("content", "") or "", **({"tool_calls": calls} if calls else {})})
            if not calls:
                return msg.get("content", "").strip()
            for tc in calls:
                fn = tc.get("function", {})
                name, args = fn.get("name"), fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if self.verbose:
                    print(f"  -> {name}({json.dumps(args)})")
                result = self.executor(house, name, args)
                if self.verbose:
                    print("  <- " + result[:600].replace("\n", "\n     ") + (" ..." if len(result) > 600 else ""))
                self.history.append({"role": "tool", "content": result, "tool_name": name})
            if any((tc.get("function") or {}).get("name") == "call_service" for tc in calls):
                house = self.house_provider()  # state changed: refresh for the next round
        return "Sorry, I got stuck on that one."


def live_executor(ha, config: dict):
    """Executor against a real Home Assistant: get_state reads the freshly built House,
    call_service goes to the REST API. config "toolset": "b" serves the design-B find_devices / get_state."""
    from .ha import build_house
    design_b = config.get("toolset") == "b"

    def run(house: House, name: str, args: dict) -> str:
        if design_b and name == "find_devices":
            return find_devices_text(house, args.get("areas"), args.get("classes"), args.get("name") or "")
        if design_b and name == "get_state":
            return get_state_b_text(house, args.get("areas"), args.get("classes"), args.get("name") or "", args.get("entity_ids"),
                                    args.get("where"), args.get("sort") or "", args.get("limit"))
        if name == "get_state":
            return get_state_text(house, args.get("areas") or [], args.get("classes") or [])
        if name == "call_service":
            domain, service, target, data = args.get("domain", ""), args.get("service", ""), args.get("target") or {}, args.get("data") or {}
            if not domain or not service:
                return json.dumps({"success": False, "error": "domain and service are required"})
            if target.get("area_id"):
                bad = [a for a in target["area_id"] if a not in house.area_ids()]
                if bad:
                    return json.dumps({"success": False, "error": f"unknown area_id {bad}; groups must be targeted by entity_id"})
            # Music Assistant: the model speaks generic media_player.play_media; translate to MA's service.
            if config.get("media_backend") == "music_assistant" and domain == "media_player" and service == "play_media":
                mtype = {"music": "track", "genre": "playlist", "tvshow": "track"}.get(data.get("media_content_type", "track"), data.get("media_content_type", "track"))
                data = {"media_id": data.get("media_content_id"), "media_type": mtype, "enqueue": "replace"}
                if data["media_type"] == "playlist" and config.get("music_assistant_radio_mode_for_genre", True):
                    data["radio_mode"] = True
                domain, service = "music_assistant", "play_media"
            try:
                changed = ha.call_service(domain, service, target, data)
                ids = [c.get("entity_id") for c in changed] if isinstance(changed, list) else []
                return json.dumps({"success": True, "changed": ids})
            except Exception as e:  # noqa: BLE001
                return json.dumps({"success": False, "error": str(e)[:200]})
        return json.dumps({"success": False, "error": f"unknown tool {name}"})

    return run
