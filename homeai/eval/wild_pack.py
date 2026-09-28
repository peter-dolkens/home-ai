"""Write 'house packs' for agents to author wild (non-templated) eval conversations against.

    python -m homeai.eval.wild_pack --out /tmp/wild --houses 12 --seed 900000

Each pack has the system prompt the model will see, a full state dump, the tool schema, and the
authoring instructions + JSON spec the builder (wild_build.py) expects.
"""
from __future__ import annotations

import argparse
import json
import os
import random

from ..house.generator import make_house, pick_speaker
from ..house.snapshot import system_prompt
from ..tools import TOOLS, get_state_text

SPEC = """
Author conversations as a JSON list. Each item:
{
  "id": "short-unique-id",
  "category": one of ["control", "status", "presence", "routine", "energy", "media", "appliance", "water", "vehicle", "clarify", "confirm", "missing", "smalltalk", "mixed"],
  "turns": [
    {
      "user": "what the person says (natural, spoken, no template feel)",
      "fetch": {"areas": [...], "classes": [...]} | null,      // get_state the assistant should make first, if it needs state
      "calls": [ {"domain": "...", "service": "...", "target": {"area_id": [...]} or {"entity_id": [...]}, "data": {...}} ],  // may be []
      "answer": "what the assistant should say (short, spoken; omit if the turn ends with a question the assistant asks)",
      "answer_must": ["phrase or alt1|alt2", ...]              // lowercase substrings that must appear in a correct answer (facts, room names, numbers). Use [] if any phrasing is fine.
    }
  ]
}
Rules the assistant follows (mirror them exactly):
- It only sees the inventory; it must call get_state before answering about state/presence/power/weather or before acting on a specific named device (needs entity_id). Fetch only the areas/classes needed.
- Whole-room actions use area_id; specific devices use entity_id (from the state dump). Devices in Security/Energy/Water/Weather groups are entity_id only.
- Unlock, disarm alarm, open garage: the assistant asks for confirmation first (turn 1 answer is the question, no calls); the user confirms in turn 2 and the calls happen then.
- Ambiguity ("turn on the lamp" with lamps in several rooms and none in the speaker's room): assistant asks which; user answers; then act.
- Music: [MA] players use music_assistant.play_media with data {media_id, media_type in artist/album/track/playlist/radio/podcast, radio_mode: true for genres}; other players use media_player.play_media with {media_content_type, media_content_id}. Pause/skip/volume/shuffle/join use media_player services for all players.
- Locking never needs confirmation. Climate is targeted by area_id.
- Presence: reason from motion/occupancy + who is home + elimination; say how sure.
- Answers are short and speakable. Numbers as digits are fine.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--houses", type=int, default=12)
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--profile-share", type=float, default=0.4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    for i in range(a.houses):
        seed = a.seed + i
        profile = "bundook" if rng.random() < a.profile_share else None
        house = make_house(seed, profile)
        sp = pick_speaker(house, random.Random(seed))
        pack = {
            "house_seed": seed, "profile": profile,
            "speaker": {"name": sp.name, "area": sp.area, "channel": sp.channel},
            "system_prompt": system_prompt(house, sp),
            "full_state": get_state_text(house),
            "tools": TOOLS,
            "people": [{"name": p.name, "home": p.home, "room_truth": p.room} for p in house.people],
            "spec": SPEC,
        }
        with open(os.path.join(a.out, f"house-{seed}.json"), "w") as f:
            json.dump(pack, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.houses} packs to {a.out}")


if __name__ == "__main__":
    main()
