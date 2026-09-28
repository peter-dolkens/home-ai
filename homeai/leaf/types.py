"""Room-TYPE leaves (Peter, 27 Sep): one leaf model per kind of room, reused by every room of that kind (the request
carries the room's own tools), e.g. one office leaf for both offices. Builds chat-format training data per type from the
generic leaf data (rooms of that kind across many generated houses) plus the Bundook rooms of that type, and writes the
Bundook area -> type map used to route the leaf test and the tiered pipeline.

    PYTHONPATH=. python3 -m homeai.leaf.types            # data/generated/leaf-type/<type>-{train,val}.jsonl + bundook-types.json
"""
from __future__ import annotations

import json
import os
import random

from .chat import to_chat
from .eval import build_house

OUT = "data/generated/leaf-type"
# type -> (generator room kinds, Bundook rooms). Bundook's lounge and media room are outdoor space for a future build
# (Peter, 27 Sep): a second alfresco.
TYPES = {
    "office": ({"office"}, ["office_pete", "office_amber"]),
    "bedroom": ({"bedroom"}, ["master_bed", "master_robe", "guest_bedroom", "guest_robe"]),
    "bathroom": ({"bath"}, ["master_ensuite", "guest_ensuite", "bathroom", "vanity", "toilet", "laundry_toilet"]),
    "kitchen": ({"kitchen", "living"}, ["kitchen", "dining_hall"]),
    "utility": ({"utility"}, ["laundry", "pantry", "storage"]),
    "circulation": ({"entry"}, ["hallway", "mudroom", "entry"]),
    "outdoor": ({"outdoor", "garage"}, ["alfresco", "lounge", "media_room", "carport", "workshop", "battery_shed"]),
}


def area_kinds(meta_path: str) -> list[str]:
    kinds, cache = [], {}
    for l in open(meta_path):
        m = json.loads(l)
        key = json.dumps(m["house"], sort_keys=True)
        if key not in cache:
            cache[key] = {a.area_id: a.kind for a in build_house(m["house"]).areas}
        kinds.append(cache[key].get(m["area_id"], "?"))
    return kinds


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(3)
    gen = [json.loads(l) for l in open("data/generated/leaf-generic-train.jsonl")]
    kinds = area_kinds("data/generated/leaf-generic-train.meta.jsonl")
    for t, (ks, rooms) in TYPES.items():
        rows = [ex for ex, k in zip(gen, kinds) if k in ks]
        for r in rooms:
            p = f"data/generated/leaf-bundook/{r}-train.jsonl"
            rows += [json.loads(l) for l in open(p)] if os.path.exists(p) else []
        rng.shuffle(rows)
        n_val = max(50, len(rows) // 20)
        for split, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
            with open(f"{OUT}/{t}-{split}.jsonl", "w") as f:
                for ex in part:
                    f.write(json.dumps(to_chat(ex, rng), ensure_ascii=False) + "\n")
        print(f"{t:12} train {len(rows) - n_val:5}  val {n_val}")
    json.dump({r: t for t, (_, rooms) in TYPES.items() for r in rooms}, open(f"{OUT}/bundook-types.json", "w"), indent=1)


if __name__ == "__main__":
    main()
