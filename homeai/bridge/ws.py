"""Minimal Home Assistant websocket client (stdlib only) used to read the entity
and device registries. Those carry entity_category (config/diagnostic) and
hidden flags that the REST states API does not expose."""
from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import struct
from typing import Any, Optional
from urllib.parse import urlparse


class _WS:
    def __init__(self, url: str, timeout: float = 20.0):
        u = urlparse(url)
        secure = u.scheme in ("https", "wss")
        host, port = u.hostname, u.port or (443 if secure else 80)
        raw = socket.create_connection((host, port), timeout=timeout)
        self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host) if secure else raw
        key = base64.b64encode(os.urandom(16)).decode()
        path = (u.path or "/") + "/api/websocket" if not u.path.endswith("/api/websocket") else u.path
        path = path.replace("//", "/")
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket handshake failed")
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise ConnectionError(head.decode(errors="replace").splitlines()[0])
        self.buf = rest

    def _recv_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self.buf)))
            if not chunk:
                raise ConnectionError("socket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, obj: dict):
        data = json.dumps(obj).encode()
        mask = os.urandom(4)
        n = len(data)
        if n < 126:
            head = struct.pack("!BB", 0x81, 0x80 | n)
        elif n < 65536:
            head = struct.pack("!BBH", 0x81, 0x80 | 126, n)
        else:
            head = struct.pack("!BBQ", 0x81, 0x80 | 127, n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(head + mask + masked)

    def recv(self) -> dict:
        message = b""
        while True:
            b0, b1 = self._recv_exact(2)
            fin, opcode = b0 & 0x80, b0 & 0x0F
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack("!H", self._recv_exact(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._recv_exact(8))[0]
            if b1 & 0x80:
                mask = self._recv_exact(4)
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(self._recv_exact(n)))
            else:
                payload = self._recv_exact(n)
            if opcode == 0x8:
                raise ConnectionError("websocket closed by server")
            if opcode == 0x9:  # ping -> pong
                self.sock.sendall(struct.pack("!BB", 0x8A, 0x80) + b"\x00\x00\x00\x00")
                continue
            message += payload
            if fin:
                return json.loads(message.decode())

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def fetch_registries(url: str, token: str) -> dict[str, Any]:
    """Return {"entities": [...], "devices": [...], "areas": [...]} from HA's registries."""
    ws = _WS(url)
    try:
        hello = ws.recv()
        if hello.get("type") != "auth_required":
            raise ConnectionError(f"unexpected greeting {hello}")
        ws.send({"type": "auth", "access_token": token})
        ok = ws.recv()
        if ok.get("type") != "auth_ok":
            raise PermissionError(f"websocket auth failed: {ok}")
        out = {}
        for i, (key, typ) in enumerate([("entities", "config/entity_registry/list"),
                                        ("devices", "config/device_registry/list"),
                                        ("areas", "config/area_registry/list")], start=1):
            ws.send({"id": i, "type": typ})
            while True:
                msg = ws.recv()
                if msg.get("id") == i:
                    if not msg.get("success", True):
                        raise RuntimeError(msg.get("error"))
                    out[key] = msg["result"]
                    break
        return out
    finally:
        ws.close()
