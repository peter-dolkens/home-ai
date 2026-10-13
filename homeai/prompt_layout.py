"""Which system-prompt layout a model was trained on (homeai/house/snapshot.py: LAYOUTS, relayout), for everything that
runs a model: homeai.eval.e2e / run_eval / latency / prefix_bench, homeai.tiered.eval and the bridge.

The layout is resolved per model, first match wins:
  1. an explicit --prompt-layout (or the bridge config's "prompt_layout")
  2. $HOMEAI_PROMPT_LAYOUT
  3. the model's configs/models.json entry: its own "prompt_layout", else its training config's "prompt_layout"
  4. "classic": every model trained before v10, and the untuned baselines
The eval sets store the conversation in the classic layout with the bare utterances. The evals score on those stored
messages and render() only the request they send (so a user_context / clock_in_user prefix never reaches a scorer);
training renders whole rows with apply() when loading (homeai/train/data.py). Conversion works from and to every layout, so the
*-sp.jsonl copies serve any model too."""
from __future__ import annotations

import argparse
import functools
import json
import os
from typing import Optional

from .house.snapshot import LAYOUTS, relayout_messages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "configs", "models.json")


@functools.lru_cache(maxsize=None)
def _registry(path: str) -> dict:
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return {}


def from_config(path: Optional[str]) -> Optional[str]:
    """A training config's "prompt_layout" (None when it has none or cannot be read)."""
    if not path:
        return None
    try:
        return json.load(open(path if os.path.isabs(path) else os.path.join(ROOT, path))).get("prompt_layout")
    except (OSError, json.JSONDecodeError):
        return None


def for_model(model: Optional[str], explicit: Optional[str] = None, registry: str = REGISTRY) -> str:
    layout = explicit or os.environ.get("HOMEAI_PROMPT_LAYOUT")
    if not layout and model:
        reg = _registry(registry)
        entry = reg.get(model) or reg.get(model.split(":")[0]) if isinstance(reg, dict) else None  # "name:latest"
        if isinstance(entry, dict):
            layout = entry.get("prompt_layout") or from_config(entry.get("config"))
    layout = layout or "classic"
    if layout not in LAYOUTS:
        raise SystemExit(f"unknown prompt layout {layout!r}; have {', '.join(LAYOUTS)}")
    return layout


def render(messages: list[dict], layout: Optional[str]) -> list[dict]:
    """A request's messages (system first) in `layout`: what goes to the model. Callers keep scoring on their stored
    messages, so a user_context / clock_in_user prefix never reaches a scorer (homeai/house/snapshot.py
    relayout_messages)."""
    return relayout_messages(messages, layout)


def apply(ex: dict, layout: Optional[str]) -> dict:
    """The row with its conversation in `layout` (the row itself when nothing changes): for training (homeai/train/
    data.py load_jsonl) and the layout converters, never for rows a scorer reads."""
    msgs = ex.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        return ex
    new = relayout_messages(msgs, layout)
    if new == msgs:
        return ex
    return {**ex, "messages": new}


def add_arg(ap: argparse.ArgumentParser):
    ap.add_argument("--prompt-layout", choices=LAYOUTS, default=None,
                    help="prompt layout the model was trained on (default: $HOMEAI_PROMPT_LAYOUT, else the "
                         "model's configs/models.json entry / training config \"prompt_layout\", else classic)")
