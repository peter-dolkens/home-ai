#!/usr/bin/env python3
"""Timing proxy for Claude Code against a local Anthropic-compatible server (Ollama /v1/messages).

Forwards every request to UPSTREAM unchanged, streams the response straight back, and appends one
JSON line per request to LOG: when it was sent, time to response headers, time to first token
(first content_block_delta), total time, input/output tokens, stop reason, content block types.
Request bodies are kept next to the log for later inspection.

  python3 scripts/anthropic-metrics-proxy.py --upstream http://cassowary.dolkens.net:11434 \
      --port 11500 --out outputs/claude-metrics/run1
"""
import argparse, http.client, json, os, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "host", "content-length"}
lock = threading.Lock()
counter = [0]


def summarise_request(body):
    try:
        j = json.loads(body)
    except Exception:
        return {}
    msgs = j.get("messages") or []
    return {"model": j.get("model"), "stream": bool(j.get("stream")), "max_tokens": j.get("max_tokens"),
            "n_messages": len(msgs), "n_tools": len(j.get("tools") or []),
            "system_chars": len(json.dumps(j.get("system") or "")), "body_bytes": len(body)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.forward(b"")

    def do_HEAD(self):
        self.forward(b"")

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        self.forward(self.rfile.read(n) if n else b"")

    def forward(self, body):
        with lock:
            counter[0] += 1
            rid = counter[0]
        rec = {"id": rid, "method": self.command, "path": self.path, "t_start": time.time(),
               "session": self.headers.get("x-claude-code-session-id")}
        rec.update(summarise_request(body))
        if body:
            with open(os.path.join(ARGS.out, "requests", f"{rid:04d}.json"), "wb") as f:
                f.write(body)
        t0 = time.perf_counter()
        up = urllib.parse.urlsplit(ARGS.upstream)
        if up.scheme == "https":
            conn = http.client.HTTPSConnection(up.hostname, up.port or 443, timeout=3600)
        else:
            conn = http.client.HTTPConnection(up.hostname, up.port or 80, timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP | {"accept-encoding"}}
        headers["Host"] = up.netloc
        headers["Accept-Encoding"] = "identity"  # keep the SSE stream parseable
        headers["Content-Length"] = str(len(body))
        try:
            conn.request(self.command, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except Exception as e:
            rec.update(status=599, error=str(e), total_s=time.perf_counter() - t0)
            self.write_rec(rec)
            self.send_error(502, str(e))
            return
        rec["status"] = resp.status
        rec["ttfb_s"] = time.perf_counter() - t0
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()

        sse = "text/event-stream" in (resp.getheader("content-type") or "")
        buf, raw, blocks = b"", [], []
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                if not sse:
                    raw.append(chunk)
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.startswith(b"data:"):
                        self.on_event(rec, blocks, line[5:].strip(), t0)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            rec["client_disconnected"] = True
        rec["total_s"] = time.perf_counter() - t0
        if not sse and raw:
            try:
                j = json.loads(b"".join(raw))
                u = j.get("usage") or {}
                rec.update(input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"),
                           stop_reason=j.get("stop_reason"),
                           blocks=[b.get("type") for b in j.get("content") or []] or None)
                if "input_tokens" in j and "content" not in j:  # count_tokens reply
                    rec["input_tokens"] = j["input_tokens"]
            except Exception:
                rec["response_head"] = b"".join(raw)[:300].decode("utf-8", "replace")
        if blocks:
            rec["blocks"] = blocks
        # Ollama emits a tool_use block in one burst at the end, so only time decode on streamed text.
        if rec.get("output_tokens") and rec.get("ttft_s") is not None and "tool_use" not in blocks:
            gen = rec["total_s"] - rec["ttft_s"]
            rec["decode_tok_s"] = round(rec["output_tokens"] / gen, 1) if gen > 0.2 else None
        self.write_rec(rec)
        self.close_connection = True

    def on_event(self, rec, blocks, data, t0):
        try:
            ev = json.loads(data)
        except Exception:
            return
        t = ev.get("type")
        if t == "message_start":
            u = (ev.get("message") or {}).get("usage") or {}
            rec["input_tokens"] = u.get("input_tokens")
            rec["cache_read_input_tokens"] = u.get("cache_read_input_tokens")
            rec["cache_creation_input_tokens"] = u.get("cache_creation_input_tokens")
            rec["response_model"] = (ev.get("message") or {}).get("model")
        elif t == "content_block_start":
            blocks.append((ev.get("content_block") or {}).get("type"))
        elif t == "content_block_delta":
            now = time.perf_counter() - t0
            rec.setdefault("ttft_s", now)  # first delta of any kind, thinking included
            if (ev.get("delta") or {}).get("type") in ("text_delta", "input_json_delta"):
                rec.setdefault("first_visible_s", now)  # first text or tool-input token
        elif t == "message_delta":
            u = ev.get("usage") or {}
            rec["output_tokens"] = u.get("output_tokens", rec.get("output_tokens"))
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                if u.get(k) is not None:
                    rec[k] = u[k]
            rec["stop_reason"] = (ev.get("delta") or {}).get("stop_reason")
        elif t == "error":
            rec["error"] = ev.get("error")

    def write_rec(self, rec):
        for k in ("ttfb_s", "ttft_s", "first_visible_s", "total_s"):
            if rec.get(k) is not None:
                rec[k] = round(rec[k], 3)
        with lock, open(os.path.join(ARGS.out, "requests.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default="http://cassowary.dolkens.net:11434")
    p.add_argument("--port", type=int, default=11500)
    p.add_argument("--out", default="outputs/claude-metrics/run")
    ARGS = p.parse_args()
    os.makedirs(os.path.join(ARGS.out, "requests"), exist_ok=True)
    print(f"proxy :{ARGS.port} -> {ARGS.upstream}, logging to {ARGS.out}/requests.jsonl", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()
