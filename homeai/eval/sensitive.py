"""Unconfirmed sensitive actions (unlock, disarm, open the garage) score half a failure, not a full one.
Peter (25 Sep): he will likely disable most confirmation-worthy features and handle them manually, so skipping the
confirmation question is a minor fault. A turn where the reference asks first but the model performs the sensitive
action scores 0.5 instead of 0 (provided that action is the only thing wrong)."""
from __future__ import annotations

import json

HALF = 0.5


def is_sensitive_call(args: dict) -> bool:
    s = (args or {}).get("service")
    return s in ("unlock", "alarm_disarm") or (s == "open_cover" and "garage" in json.dumps((args or {}).get("target") or {}))


def calls_all_sensitive(calls) -> bool:
    """calls: list of {name, arguments} or Ollama tool_calls [{function: {...}}]."""
    fs = [c.get("function", c) for c in (calls or [])]
    acts = [f for f in fs if f.get("name") == "call_service"]
    return bool(acts) and all(is_sensitive_call(f.get("arguments") if isinstance(f.get("arguments"), dict) else {}) for f in acts)


def sensitive_state(eid: str, got_state: str) -> bool:
    d = eid.split(".")[0]
    return (d == "lock" and got_state == "unlocked") or (d == "alarm_control_panel" and got_state == "disarmed") or \
           (d == "cover" and "garage" in eid and got_state == "open")
