"""The new build at full scale, from its electrical / IoT plan (data/house/bundook-project.json, an
export from plan.dolkens.house; read-only here).

    make_bundook_full(seed)                    what the plan draws today, plus the devices of the small
                                               BUNDOOK profile the plan does not model (HomePods, locks,
                                               alarm, energy loads, tank, irrigation, vehicle, dryer)
    make_bundook_full(seed, projected=True)    "bundook_full_projected": the same house with the devices
                                               the plan does not draw yet filled in (mmWave in every room,
                                               contacts on every door and window, smart plugs on a third
                                               of the GPOs, air quality, leak, BLE nodes, phones, weather
                                               station, irrigation valves, mower, the Shark's own device)

Mapping (plan fixture / circuit -> HA device, entity sets from homeai/house/devices.py):
  board_items             the Shelly modules in the Mains / Submains boxes (Pro Dimmer 2PM, Pro 3EM, Pro 4PM,
                          Pro RGBWW PM, Pro EM, Pro Dual Cover PM): one device per module in the board's
                          area, one channel device per channel in the room its circuit serves (spare
                          channels stay at the board); lighting channels are lights, 3EM / EM CTs meter
  Hue fittings            (bulb hue_st72, hue_discover) one Hue bulb each plus a Zigbee2MQTT group per
                          circuit and room; the Shelly channel on such a circuit only feeds them
  switch (bp_*)           Buster + Punch wall switches: event + input binary_sensor on the channel of each
                          circuit they control (Voltex ones are isolators); towel rails and the laundry
                          exhausts are contactor / timer switched and stay plain entities
  data circuit            a UniFi switch port (PoE switch, PoE power, port enable)
  solar strings           two Fronius GEN24 + the Selectronic SP PRO, 11 battery modules
  motion_sensor           Apollo R PRO-1 mmWave / Hue motion (indoor, outdoor)
  blind_motor             Somfy Sonesse PoE cover; the highlight-window actuators are one Dual Cover channel
  cam_ptz / cam_dome      UniFi Protect G6 PTZ (+ Frigate) / G6 turret; intercom -> G6 Entry
  fire_alarm, ac_head, ceiling_fan, wap, rack items, appliances, TV, vacuum, satellite dish: their models
Structure is fixed (the same entity ids for every seed); the seed only drives time, weather, people and states."""
from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
import collections
import random
import re

from .devices import MODELS, DeviceMixin, finish_devices, pick_names, slug, wall_inputs
from .generator import Gen
from .profiles import BUNDOOK
from .schema import Area, House

PLAN = os.path.join(os.path.dirname(__file__), "..", "..", "data", "house", "bundook-project.json")

PLAN_ROOMS = {"Office (Pete)": "office_pete", "Master Robe": "master_robe", "Master Bed": "master_bed", "Master Suite": "master_ensuite",
              "Vanity": "vanity", "Toilet": "toilet", "Bathroom": "bathroom", "Guest Suite": "guest_ensuite", "Storage": "storage",
              "Guest Bedroom": "guest_bedroom", "Guest Robe": "guest_robe", "Office (Amber)": "office_amber", "Carport": "carport",
              "Laundry Toilet": "laundry_toilet", "Kitchen": "kitchen", "Dining Hall": "dining_hall", "Workshop": "workshop", "Mudroom": "mudroom",
              "Laundry": "laundry", "Pantry": "pantry", "Hallway": "hallway", "Alfresco": "alfresco", "Lounge (future)": "lounge",
              "Media (future)": "media_room", "Bulkhead": "dining_hall", "Entry (Outside)": "entry", "Battery Shed": "battery_shed"}
# the room part of a circuit's purpose ("Kitchen — Downlights") -> area id
CIRCUIT_ROOMS = {"laundry": "laundry", "laundry wc": "laundry_toilet", "mudroom": "mudroom", "pantry": "pantry", "kitchen": "kitchen",
                 "dining": "dining_hall", "living": "dining_hall", "living/dining": "dining_hall", "living / dining": "dining_hall",
                 "lounge (future)": "lounge", "media (future)": "media_room", "dining / hallway": "dining_hall", "hallway": "hallway",
                 "storage": "storage", "master bed": "master_bed", "master robe": "master_robe", "guest robe": "guest_robe", "guest bed": "guest_bedroom",
                 "office (pete)": "office_pete", "office (amber)": "office_amber", "master ensuite": "master_ensuite", "guest ensuite": "guest_ensuite",
                 "toilet": "toilet", "vanity": "vanity", "bathroom": "bathroom", "kitchen / pantry": "kitchen", "kitchen/pantry": "kitchen",
                 "carport": "carport", "alfresco": "alfresco", "workshop": "workshop", "entry / patio": "entry", "battery shed": "battery_shed",
                 "master bed & robe": "master_bed", "guest bed & robe": "guest_bedroom", "storage / server": "storage"}
HUE_BULBS = {"hue_st72", "hue_discover"}
OUTDOOR = ("alfresco", "entry", "carport")
PTZ_NAMES = ["Driveway Camera", "North Paddock Camera", "South Paddock Camera", "Shed Camera"]


@functools.lru_cache(maxsize=1)
def load_plan(path: str = PLAN) -> dict:
    with open(path) as f:
        return json.load(f)


def _inside(x, y, poly) -> bool:
    c = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def _dist(x, y, poly) -> float:
    if _inside(x, y, poly):
        return 0.0
    best = math.inf
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy or 1)))
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best


class Plan:
    """The parts of the plan the generator uses, with every fixture placed in an area."""

    def __init__(self, d: dict):
        self.rooms = [(PLAN_ROOMS[r["name"]], r["polygon"], r.get("zone_only", False))
                      for l in d["layers"] if l["kind"] == "rooms" for r in l["rooms"] if r["name"] in PLAN_ROOMS]
        skip = {"furniture", "cable_weight", "cable_tray"}
        self.fixtures = [f for l in d["layers"] if l["name"] != "Furniture" for f in l.get("fixtures", []) if f["type"] not in skip]
        self.room_ids = {r["id"]: PLAN_ROOMS[r["name"]] for l in d["layers"] if l["kind"] == "rooms" for r in l["rooms"] if r["name"] in PLAN_ROOMS}
        self.circuits = {c["id"]: c for c in d["circuits"]}
        self.by_circuit: dict[str, list[dict]] = {}
        for f in self.fixtures:
            if f.get("circuit_id"):
                self.by_circuit.setdefault(f["circuit_id"], []).append(f)

    def room_at(self, f: dict, among: tuple[str, ...] | None = None) -> tuple[str, float]:
        """(area, distance): the room containing the fixture (real rooms before zones), else the nearest one."""
        cands = [(z, _dist(f["x"], f["y"], poly), aid) for aid, poly, z in self.rooms if among is None or aid in among]
        cands.sort(key=lambda c: (c[1] > 0, c[1], c[0]))
        return cands[0][2], cands[0][1]

    def rooms_touching(self, f: dict, tol: float = 18.0) -> set[str]:
        return {aid for aid, poly, z in self.rooms if not z and _dist(f["x"], f["y"], poly) <= tol}


def circuit_parts(c: dict) -> tuple[str | None, str]:
    """'Kitchen — Island Pendant' -> ('kitchen', 'Island Pendant'); unknown room part -> (None, whole purpose)."""
    purpose = (c.get("purpose") or "").strip()
    if " — " in purpose:
        room, what = purpose.split(" — ", 1)
        return CIRCUIT_ROOMS.get(room.strip().lower()), what.strip()
    return None, purpose


def tidy(label: str) -> str:
    """'Track (low fill)' -> 'Track Low Fill', 'Sitting-area Feature Lights' -> 'Sitting Area Feature Lights'."""
    s = re.sub(r"\((?:monitored|weatherproof|spare)\)", "", label).replace(" — ", " ")
    s = s.replace("-", " ").replace("/", " ").replace("(", " ").replace(")", " ").replace("&", "and")
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w if w.isupper() else w[:1].upper() + w[1:] for w in s.split(" "))


class BundookGen(DeviceMixin, Gen):
    def __init__(self, seed: int, projected: bool = False):
        prof = dataclasses.replace(BUNDOOK, name="bundook_projected" if projected else "bundook_full")
        super().__init__(seed, prof)
        self.projected = projected
        self.plan = Plan(load_plan())
        self.srng = random.Random(20260926)  # structure: identical for every seed
        self._devinit()

    def make(self) -> House:
        house = super().make()
        for e in house.by_class("climate"):
            if e.attributes.get("floor"):  # hydronic zones only heat
                e.attributes["last_mode"] = "heat"
                if e.state == "cool":
                    e.state = "off"
        finish_devices(self, house)
        return house

    def _an(self, aid: str) -> str:
        """Entity-name prefix for an area, as the small profile does it ("Pete's Office" for office_pete)."""
        s = self.spec[aid]
        return s.spoken.split("'s ")[0] + "'s Office" if s.spoken and "'s office" in s.spoken else s.name

    # Gen.make() calls this for profile houses
    def _build_profile(self, house: House, prof):
        rng = self.rng
        self.spec = {s.area_id: s for s in prof.areas}
        for s in prof.areas:
            house.areas.append(Area(s.area_id, s.name, s.kind, spoken=s.spoken))
        mains = next(f for f in self.plan.fixtures if f["type"] == "meter_box" and f.get("name") == "Mains")
        self.board_area, self.board_name = self.plan.room_at(mains)[0], "Switchboard"
        self.energy_params = dict(peak_kw=prof.peak_kw, cap_kwh=prof.cap_kwh)
        self._board(house)
        self._off_board(house)
        self._fixtures(house)
        self._small_profile(house, prof)
        self._network(house)
        self._solar(house)
        if self.projected:
            self._projected(house)

    # ------------------------------------------------------------------ lighting circuits
    # ------------------------------------------------------------------ the boards: Shelly modules and their circuits
    MODULES = {"Shelly Pro Dimmer 2PM": ("bd_pro_dimmer_2pm", 2, "Pro Dimmer 2PM"), "Shelly Pro 3EM": ("bd_pro_3em", 3, "Pro 3EM"),
               "Shelly Pro 4PM": ("shelly_pro_4pm", 4, "Pro 4PM"), "Shelly Pro RGBWW PM": ("shelly_pro_rgbww", 5, "Pro RGBWW PM"),
               "Shelly Pro EM": ("bd_pro_em", 2, "Pro EM"), "Shelly Pro Dual Cover/Shutter PM": ("shelly_dual_cover", 2, "Pro Dual Cover PM")}

    def circuit_room(self, c: dict) -> tuple[str | None, str]:
        """(area, how): fixtures inside a room, else the purpose text, else room_ids; None when the circuit serves no room."""
        plan = self.plan
        fx = [f for f in plan.by_circuit.get(c["id"], []) if f["type"] != "switch"]
        inside = [plan.room_at(f)[0] for f in fx if plan.room_at(f)[1] == 0]
        if inside:
            return max(sorted(set(inside)), key=inside.count), "fixtures"
        room, _ = circuit_parts(c)
        if room:
            return room, "purpose"
        if c.get("room_ids") and plan.room_ids.get(c["room_ids"][0]):
            return plan.room_ids[c["room_ids"][0]], "room_ids"
        if fx:
            return None, "plant outside the rooms"  # heat pump, water and septic pumps: metered at the board
        return None, "unplaced"

    def _board(self, house: House):
        plan = self.plan
        self.placed = collections.Counter()
        self.circuit_rooms = {}
        for c in plan.circuits.values():
            if c["kind"] in ("lighting", "power"):
                aid, how = self.circuit_room(c)
                self.circuit_rooms[c["id"]] = aid
                self.placed[how] += 1
        # wall switches (Buster + Punch plates) -> the circuits they control; Voltex ones are isolators, not inputs
        self.controls: dict[str, list[dict]] = {}
        for f in sorted((f for f in plan.fixtures if f["type"] == "switch"), key=lambda f: f["id"]):
            if str((f.get("props") or {}).get("model", "")).startswith("bp_"):
                for cid in f.get("controls") or []:
                    self.controls.setdefault(cid, []).append(f)
        self.hue_circuits = set()
        for c in plan.circuits.values():  # Hue fittings: a bulb each, grouped per room; their channel only feeds them
            if c["kind"] != "lighting":
                continue
            hue = [f for f in plan.by_circuit.get(c["id"], []) if (f.get("props") or {}).get("bulb") in HUE_BULBS or (f.get("props") or {}).get("model") in HUE_BULBS]
            if not hue:
                continue
            self.hue_circuits.add(c["id"])
            label = self._label(c)
            by_room: dict[str, list] = {}
            for f in hue:
                by_room.setdefault(plan.room_at(f, OUTDOOR if label == "Floodlights" else None)[0], []).append(f)
            for aid, fs in by_room.items():
                self._hue(house, aid, self._an(aid), label, len(fs), watts=7 if "ST72" in c.get("notes", "") else 15, circuit=c["id"])
        self.on_module: set[str] = set()
        self.metered: set[str] = set()
        self.inputs_done = inputs_done = set()
        for box in (f for f in plan.fixtures if f["type"] == "meter_box"):
            box_area = plan.room_at(box)[0]
            for it in (box.get("props") or {}).get("board_items") or []:
                if it.get("profile") not in self.MODULES:
                    continue
                self._module(house, it, box_area, inputs_done)

    def _label(self, c: dict) -> str:
        _, what = circuit_parts(c)
        label = "Floodlights" if "Floodlights" in what else tidy(what.replace("Mirror / Wall Lights", "Mirror Lights"))
        return re.sub(r"RGBWW Strip Ch(\d)", r"RGBWW Strip \1", label)

    def _module(self, house: House, it: dict, box_area: str, inputs_done: set, label: str | None = None):
        """One DIN-rail module: its own entities in the board's area, one channel device per channel in the room its circuit serves."""
        plan = self.plan
        model, n_ch, short = self.MODULES[it["profile"]]
        m = MODELS[model]
        label = label or it.get("label") or it.get("name")
        unit = self._device(house, model, box_area, f"{label} {short}", ents=m.unit)
        unit["is_unit"] = True
        cids = [c for c in dict.fromkeys(it.get("circuit_ids") or []) if c in plan.circuits]
        if model == "shelly_dual_cover":  # the actuator group's common + open/close actives drive one cover
            self.on_module.update(cids)
            cids = cids[:1]
        for k in range(max(n_ch, len(cids))):
            c = plan.circuits.get(cids[k]) if k < len(cids) else None
            kw: dict = {}
            if c is None:  # spare channel: HA still shows it
                area, name = box_area, f"{label} Channel {k + 1}"
                kw = dict(main_domain="light" if model in ("bd_pro_dimmer_2pm", "shelly_pro_rgbww") else "cover" if model == "shelly_dual_cover" else "switch",
                          main_cls="device") if m.ents[0].suffix == "" else {}
                self._device(house, model, area, name, unit_of=unit, **kw)
                continue
            self.on_module.add(c["id"])
            area = self.circuit_rooms.get(c["id"]) or box_area
            an = self._an(area)
            lab = self._label(c)
            lab = lab[len(an):].strip() if lab.startswith(an) else lab
            ents = list(m.ents)
            if model in ("bd_pro_3em", "bd_pro_em"):
                room, what = circuit_parts(c)
                name = tidy((what if room else c.get("purpose") or f"Circuit {c['name']}").replace("→", "to"))
                nm = f"{an} {name}" if room and not name.lower().startswith(an.lower()) else name
                self._device(house, model, area, f"{nm} Circuit", unit_of=unit)["meters"] = c["id"]
                self.metered.add(c["id"])
                continue
            if model == "shelly_dual_cover":
                kw = dict(main_cls="cover", main_id=f"cover.{area}_highlight_windows", load_w=30, circuit=c["id"])
                name = f"{an} Highlight Windows"
            elif c["id"] in self.hue_circuits:  # the channel only feeds the Hue bulbs
                kw = dict(main_domain="light" if model != "shelly_pro_4pm" else "switch", main_cls="device", circuit=None)
                name = f"{an} {lab} {'Feed' if model == 'shelly_pro_4pm' else 'Dimmer'}"
            elif "Exhaust Fan" in lab:
                kw = dict(main_domain="fan", main_cls="fan", main_id=f"fan.{area}_exhaust", load_w=25, circuit=c["id"])
                name = f"{an} Exhaust Fan"
            elif "EV Charger" in lab:
                area, name = "energy", "EV Charger"
                kw = dict(main_domain="switch", main_cls="switch", main_id="switch.ev_charger", main_attrs={"power_w": 7000, "appliance": True}, circuit=c["id"])
            elif c["kind"] == "power":
                area, name = box_area, tidy(lab)
                kw = dict(main_domain="switch", main_cls="device", circuit=c["id"])
            else:
                dumb = [f for f in plan.by_circuit.get(c["id"], []) if f["type"] != "switch"]
                watts = sum(12 if f["type"] == "downlight" else 15 if f["type"] == "track_head" else 30 if "ektor" in str((f.get("props") or {}).get("model")) else 20
                            for f in dumb) or 20
                kw = dict(main_domain="light", main_cls="light", main_id=f"light.{area}_{slug(lab)}", load_w=watts, circuit=c["id"])
                name = f"{an} {lab}"
            if c["id"] not in inputs_done and self.controls.get(c["id"]) and model != "shelly_dual_cover":
                inputs_done.add(c["id"])
                n = len(self.controls[c["id"]])
                ents += wall_inputs(["Wall Switch"] if n == 1 else [f"Wall Switch {i + 1}" for i in range(n)])
            self._device(house, model, area, name, ents=ents, unit_of=unit, **kw)

    def _off_board(self, house: House):
        """Circuits the plan switches by contactor or timer rather than a Shelly: towel rails and the laundry exhaust fans stay
        plain entities (as in the small profile)."""
        for c in self.plan.circuits.values():
            if c["kind"] != "power" or c["id"] in self.on_module:
                continue
            lab = self._label(c)
            aid = self.circuit_rooms.get(c["id"])
            if aid is None:
                continue
            if "Towel Rail" in lab:
                e = self._add(house, "switch", f"switch.{aid}_towel_rail", f"{self._an(aid)} Towel Rail", "switch", aid,
                              state="on" if self.p(0.3) else "off", attributes={"power_w": 200})
                self._track(e, 200, c["id"])
            elif "Exhaust Fan" in lab:
                self._add(house, "fan", f"fan.{aid}_exhaust", f"{self._an(aid)} Exhaust Fan", "fan", aid)

    # ------------------------------------------------------------------ fixtures that are devices
    def _fixtures(self, house: House):
        plan, rng = self.plan, self.rng
        count: dict[tuple, int] = {}

        def nth(aid, what):
            count[(aid, what)] = count.get((aid, what), 0) + 1
            return count[(aid, what)]

        by_type: dict[str, list[dict]] = {}
        for f in plan.fixtures:
            by_type.setdefault(f["type"], []).append(f)
        per_room = lambda fs, among=None: [(plan.room_at(f, among)[0], f) for f in fs]  # noqa: E731
        # presence
        for aid, f in per_room(by_type.get("motion_sensor", [])):
            model = (f.get("props") or {}).get("model") or "hue_motion_indoor"
            if model == "hue_motion_outdoor" or plan.room_at(f)[1] > 0:
                aid = plan.room_at(f, OUTDOOR)[0]
                self._device(house, "hue_motion_outdoor", aid, f"{self._an(aid)} Motion {nth(aid, 'om')}")
            elif model == "apollo_r_pro_1":
                self._device(house, "apollo_r_pro_1_full" if self.projected else "apollo_r_pro_1", aid, f"{self._an(aid)} mmWave {nth(aid, 'mm')}")
            else:
                self._device(house, "hue_motion", aid, f"{self._an(aid)} Motion {nth(aid, 'm')}")
        # blinds and window actuators
        blinds = per_room(by_type.get("blind_motor", []))
        for aid in dict.fromkeys(a for a, _ in blinds):
            n = sum(1 for a, _ in blinds if a == aid)
            for i in range(n):
                nm = f"{self._an(aid)} Blinds" if n == 1 else f"{self._an(aid)} Blind {i + 1}"
                self._device(house, "somfy_poe", aid, nm, main_cls="cover", main_id=f"cover.{aid}_blinds" if n == 1 else None)
        # climate, fans, smoke
        for aid, f in per_room(by_type.get("ac_head", [])):
            if (f.get("props") or {}).get("model", "").startswith("mhi_fdua"):
                aid = "dining_hall"  # the ducted unit's head sits in the storage ceiling but serves the dining hall
            k = nth(aid, "ac")
            self._device(house, "intesis_ac", aid, f"{self._an(aid)} Air Conditioning" + (f" {k}" if k > 1 else ""),
                         main_id=f"climate.{aid}" if k == 1 else None, load_w=1400)
        for aid, f in per_room(by_type.get("ceiling_fan", [])):
            self._device(house, "baf_haiku", aid, f"{self._an(aid)} Fan", main_id=f"fan.{aid}_fan", load_w=30, circuit=f.get("circuit_id"))
        for aid, f in per_room(by_type.get("fire_alarm", [])):
            k = nth(aid, "sm")
            self._device(house, "clipsal_smoke", aid, f"{self._an(aid)} Smoke Alarm" + (f" {k}" if k > 1 else ""))
        # cameras, intercoms
        ptz = by_type.get("cam_ptz", [])
        cx = sum(f["x"] for f in ptz) / max(1, len(ptz))
        cy = sum(f["y"] for f in ptz) / max(1, len(ptz))
        used = set()
        for i, f in enumerate(sorted(ptz, key=lambda f: f["id"])):
            if i < len(PTZ_NAMES):
                cam = PTZ_NAMES[i]
            else:
                ang = math.degrees(math.atan2(cy - f["y"], f["x"] - cx)) % 360
                cam = ["East", "North East", "North", "North West", "West", "South West", "South", "South East"][int((ang + 22.5) // 45) % 8] + " Camera"
                k = 2
                while cam in used:
                    cam = cam.rsplit(" Camera", 1)[0].rsplit(" ", 1)[0] if cam[-1].isdigit() else cam
                    cam = f"{cam.replace(' Camera', '')} Camera {k}"
                    k += 1
            used.add(cam)
            self._device(house, "g6_ptz", "security", cam, main_id=f"camera.{slug(cam)}", circuit=f.get("circuit_id"))
            self._device(house, "frigate_camera", "security", cam)
        for f in by_type.get("cam_dome", []):
            aid, dist = plan.room_at(f)
            if dist > 0:
                aid = plan.room_at(f, OUTDOOR)[0]
            self._device(house, "g6_turret", aid, f"{self._an(aid)} Camera", main_id=f"camera.{aid}")
        for i, f in enumerate(sorted(by_type.get("intercom", []), key=lambda f: f["id"])):
            aid = plan.room_at(f)[0]
            nm = "Entry Intercom" if aid == "entry" else "Workshop Intercom" if not count.get(("ic", "w")) else "Workshop Rear Intercom"
            if aid != "entry":
                count[("ic", "w")] = 1
            self._device(house, "g6_entry", "security", nm, main_id=f"camera.{slug(nm)}")
        # appliances and the rest of the Services layer
        kitchen_kinds = {"dishwasher": "Dishwasher", "microwave": "Microwave", "rangehood": "Rangehood", "fridge": "Fridge", "freezer": "Freezer",
                         "washing_machine": "Washing Machine"}
        for f in by_type.get("oven", []):
            kind = "Cooktop" if "cx492" in (f.get("props") or {}).get("model", "") else "Oven"
            self._appliance(house, "kitchen", self._an("kitchen"), kind)
            self._attach(house, house.entities[f"switch.kitchen_{slug(kind)}"], "home_connect", load_w=2400, circuit=f.get("circuit_id"))
        for t, kind in kitchen_kinds.items():
            for f in by_type.get(t, []):
                aid = plan.room_at(f)[0]
                self._appliance(house, aid, self._an(aid), kind)
                main = house.entities[f"switch.{aid}_{slug(kind)}"]
                if t in ("microwave", "rangehood"):
                    self._attach(house, main, "home_connect", load_w=main.attributes["power_w"], circuit=f.get("circuit_id"))
                elif t == "washing_machine":
                    self._attach(house, main, "lg_washer", load_w=main.attributes["power_w"], circuit=f.get("circuit_id"))
                else:
                    self._track(main, main.attributes["power_w"], f.get("circuit_id"))
        for f in by_type.get("tv", []):
            aid = plan.room_at(f)[0]
            self._device(house, "lg_tv", aid, f"{self._an(aid)} TV", main_id=f"media_player.{aid}_tv", load_w=150, circuit=f.get("circuit_id"))
        for f in by_type.get("robot_vacuum", []):
            self._device(house, "dreame_vacuum", plan.room_at(f)[0], "Robot Vacuum", main_id="vacuum.robot", load_w=60, circuit=f.get("circuit_id"))
        for aid, f in per_room(by_type.get("wap", [])):
            k = nth(aid, "ap")
            self._device(house, "unifi_ap", aid, f"{self._an(aid)} Access Point" + (f" {k}" if k > 1 else ""))
        for f in by_type.get("satellite_dish", []):
            self._device(house, "starlink", "storage", "Starlink")

    # ------------------------------------------------------------------ small-profile devices the plan does not model
    def _small_profile(self, house: House, prof):
        rng = self.rng
        for s in prof.areas:
            an = self._an(s.area_id)
            if s.media and s.media != "TV":
                self._device(house, "homepod", s.area_id, f"{an} {s.media}", main_id=f"media_player.{s.area_id}_{slug(s.media)}", main_attrs={"ma": True}, load_w=8)
            if s.dehumidifier:
                dev = self._device(house, "meross_plug", s.area_id, f"{an} Dehumidifier", main_cls="switch", main_id=f"switch.{s.area_id}_dehumidifier",
                                   main_attrs={"power_w": 300, "appliance": True})
                dev["main"].state = "on" if self.p(0.3) else "off"
            for kind in s.appliances:
                if f"switch.{s.area_id}_{slug(kind)}" not in house.entities:  # the dryer: not drawn yet
                    self._appliance(house, s.area_id, an, kind)
                    self._track(house.entities[f"switch.{s.area_id}_{slug(kind)}"], 2000, None)
        for nm in prof.locks:
            self._device(house, "unifi_lock", "security", nm, main_id=f"lock.{slug(nm)}")
        for nm in prof.door_sensors:
            self._device(house, "aqara_contact", "security", nm, main_domain="binary_sensor", main_cls="door", main_id=f"binary_sensor.{slug(nm)}")
        self._device(house, "udm_se", "security", "Home Alarm", main_id="alarm_control_panel.home_alarm")
        loads = {"hot_water": "MD06.01", "heating": "MD07.01", "pump": "MD03.01"}
        by_name = {c["name"]: cid for cid, c in self.plan.circuits.items()}
        for eid, name, watts, kind in prof.loads:
            on = self.p(0.5 if self.season_cold else 0.03) if kind == "heating" else self.p(0.3) if kind == "hot_water" else self.p(0.15)
            # switched by contactor from the automation channel (SP01), metered by its 3EM CT: a plain entity, as in the small profile
            e = self._add(house, "switch", eid, name, "switch", "energy", state="on" if on else "off",
                          attributes={"power_w": watts, "appliance": True}, changed_s=rng.randint(300, 4 * 3600))
            self._track(e, watts, by_name.get(loads.get(kind)))
        self._water(house, prof.irrigation_zones)
        self._attach(house, house.entities["switch.house_pump"], "dab_esybox", area="water")
        self._vehicle(house, "BYD Shark", "sensor.shark_battery", backup=True)

    # ------------------------------------------------------------------ network rack and solar
    def _network(self, house: House):
        plan = self.plan
        rack = next(f for f in plan.fixtures if f["type"] == "rack" and (f.get("props") or {}).get("model") == "apc_ar3150")
        for it in rack["props"]["rack_items"]:
            prof = it.get("profile") or ""
            if prof.startswith("UniFi NVR"):
                self._device(house, "unvr", "storage", "NVR")
            elif prof.startswith("APC Smart-UPS"):
                self._device(house, "apc_ups", "storage", f"Rack UPS {self._next_n('ups')}")
            elif prof.startswith("APC Metered Rack PDU"):
                self._device(house, "apc_pdu", "storage", f"Rack PDU {self._next_n('pdu')}")
        # every data circuit is a cable run back to a switch port
        for c in plan.circuits.values():
            if c["kind"] == "data":
                self._channel(house, "unifi_switch", "storage", f"Port {c['name']}", unit_area="storage")
        self._unit(house, "unifi_switch", "battery_shed", "Solar Switch")  # USW Enterprise 8 PoE by the inverters

    def _solar(self, house: House):
        self._device(house, "selectronic", "battery_shed", "Battery Inverter")
        for i in (1, 2):
            self._device(house, "fronius_gen24", "battery_shed", f"Solar Inverter {i}")
        for i in range(11):
            self._device(house, "battery_module", "battery_shed", f"Battery Module {i + 1}")

    def _next_n(self, what: str) -> int:
        c = self.__dict__.setdefault("_counters", {})
        c[what] = c.get(what, 0) + 1
        return c[what]

    # ------------------------------------------------------------------ Peter's build spec (26 Sep)
    SMALL = {"master_robe", "guest_robe", "toilet", "vanity", "laundry_toilet", "pantry", "storage", "battery_shed"}
    LARGE = {"dining_hall", "kitchen", "lounge"}
    AC_ROOMS_EXTRA = {"storage", "workshop", "laundry"}
    FLOOR_ZONES = ["dining_hall", "kitchen", "hallway", "lounge", "media_room", "bathroom", "master_ensuite", "guest_ensuite", "mudroom", "vanity"]

    def _build_spec(self, house: House):
        """Apollo R PRO-1 mmWave 1 / 2 / 3 per small / normal / large room plus a Hue PIR everywhere; A/C with its sub-entities in most
        rooms; the SL24 LED strips on their own RGBWW modules and every power circuit metered; Home Connect appliances; connected hot water and hydronic floor heating."""
        mm_models = ("apollo_r_pro_1_full",)  # the plan's sensor, Apollo R PRO-1, everywhere
        for a in house.areas:
            aid, an = a.area_id, self._an(a.area_id)
            devs = [d for d in self.devices if d["area"] == aid]
            if a.kind == "outdoor":
                if not any(d["model"] == "hue_motion_outdoor" for d in devs):
                    self._device(house, "hue_motion_outdoor", aid, f"{an} Motion 1")
                continue
            want = 1 if aid in self.SMALL else 3 if aid in self.LARGE else 2
            have = sum(1 for d in devs if d["model"] in mm_models)
            for k in range(have + 1, want + 1):
                self._device(house, "apollo_r_pro_1_full", aid, f"{an} mmWave {k}")
            if not any(d["model"] == "hue_motion" for d in devs):
                self._device(house, "hue_motion", aid, f"{an} Motion 1")
            # A/C: every bedroom, office, living space and kitchen, plus the server room, workshop and laundry
            if (a.kind in ("bedroom", "office", "living", "kitchen") and aid != "pantry") or aid in self.AC_ROOMS_EXTRA:
                main = house.entities.get(f"climate.{aid}")
                if main is not None:
                    self._attach(house, main, "ac_extras")
                else:
                    self._device(house, "mhi_ac_wifi", aid, f"{an} Air Conditioning", main_id=f"climate.{aid}", load_w=1200)
        # the four floor / bulkhead LED strips (SL24.x, Submains RCBO-24) are provisions: one Shelly Pro RGBWW PM each
        sub = next(f for f in self.plan.fixtures if f["type"] == "meter_box" and f.get("name") == "Submains")
        for k, c in enumerate(sorted((c for c in self.plan.circuits.values() if c["name"].startswith("SL24.")), key=lambda c: c["name"])):
            self._module(house, {"profile": "Shelly Pro RGBWW PM", "circuit_ids": [c["id"]]}, self.plan.room_at(sub)[0], self.inputs_done,
                         label=f"SR{k + 2:02d}")
        # per-circuit metering for every power circuit the boards don't meter yet
        for c in sorted(self.plan.circuits.values(), key=lambda c: c["name"]):
            if c["kind"] == "power" and c["id"] not in self.metered:
                aid = self.circuit_rooms.get(c["id"]) or self.board_area
                room, what = circuit_parts(c)
                name = tidy((what if room else c.get("purpose") or f"Circuit {c['name']}").replace("→", "to"))
                an = self._an(aid)
                nm = f"{an} {name}" if room and not name.lower().startswith(an.lower()) else name
                self._channel(house, "bd_pro_3em", aid, f"{nm} Circuit")["meters"] = c["id"]
        # appliances: Home Connect for the dishwashers and fridges, LG ThinQ for the dryer
        for eid, model in (("switch.kitchen_dishwasher", "home_connect"), ("switch.pantry_dishwasher", "home_connect"), ("switch.kitchen_fridge", "home_connect_fridge"),
                           ("switch.kitchen_freezer", "home_connect_fridge"), ("switch.laundry_dryer", "lg_washer")):
            main = house.entities.get(eid)
            if main is None:
                continue
            tracked = [d for d in self.devices if d["model"] is None and d["main"] is main]
            for d in tracked:
                self.devices.remove(d)
            self._attach(house, main, model, load_w=main.attributes.get("power_w", 500), circuit=tracked[0]["circuit"] if tracked else None)
        # connected hot water and hydronic heating; the plant lives in the carport with the heat pumps
        self._attach(house, house.entities["switch.hot_water_heat_pump"], "reclaim_hwhp", area="carport")
        self._attach(house, house.entities["switch.hydronic_heating"], "intaflo_plant", area="carport")
        for aid in self.FLOOR_ZONES:
            self._device(house, "floor_zone", aid, f"{self._an(aid)} Floor Heating", main_id=f"climate.{aid}_floor", main_attrs={"floor": True})

    # ------------------------------------------------------------------ devices the plan does not draw yet
    def _projected(self, house: House):
        from .bighouse import PLUGS
        plan, srng = self.plan, self.srng
        indoor = [a for a in house.areas if a.kind != "outdoor"]
        self._build_spec(house)
        for a in indoor:
            aid, an = a.area_id, self._an(a.area_id)
            s = self.spec[aid]
            if (s.temperature or s.humidity) and not house.in_area(aid, "temperature"):
                self._device(house, "aqara_th", aid, f"{an} Climate Sensor")
            if a.kind in ("bedroom", "office"):
                self._device(house, "aranet4", aid, f"{an} Air Quality")
            if a.kind in ("kitchen", "bath") or aid in ("laundry", "storage"):
                self._device(house, "aqara_leak", aid, f"{an} Leak Sensor")
            self._device(house, "espresense", aid, f"{an} BLE Node")
            if s.windows and aid != "dining_hall":
                self._channel(house, "shelly_dual_cover", aid, f"{an} Windows", main_domain="cover", main_cls="cover", main_id=f"cover.{aid}_windows",
                              load_w=30)
        self._device(house, "develco_air", "kitchen", f"{self._an('kitchen')} VOC Sensor")
        for aid in ("master_bed", "guest_bedroom"):
            self._device(house, "bed_presence", aid, f"{self._an(aid)} Bed Sensor")
            self._hue(house, aid, self._an(aid), "Bedside Lamps", 2)
        for aid in ("office_pete", "office_amber"):
            self._hue(house, aid, self._an(aid), "Desk Lamp", 1)
        # a contact on every drawn window and door: outside doors in the security group, inside doors in their room
        n_win: dict[str, int] = {}
        for f in sorted((f for f in plan.fixtures if f["type"] == "window"), key=lambda f: f["id"]):
            aid = plan.room_at(f)[0]
            n_win[aid] = n_win.get(aid, 0) + 1
            self._device(house, "aqara_contact", aid, f"{self._an(aid)} Window {n_win[aid]}", main_domain="binary_sensor", main_cls="window")
        ext = 0
        for f in sorted((f for f in plan.fixtures if f["type"] == "door"), key=lambda f: f["id"]):
            touching = plan.rooms_touching(f)
            if len(touching) >= 2:
                a, b = sorted(touching)[:2]
                dev = self._device(house, "aqara_contact", a, f"{self._an(a)} Door to {self._an(b)}", main_domain="binary_sensor", main_cls="door")
                dev["main"].state = "on" if self.p(0.4) else "off"
            else:
                ext += 1
                aid = next(iter(touching), plan.room_at(f)[0])
                self._device(house, "aqara_contact", "security", f"{self._an(aid)} Outside Door {ext}", main_domain="binary_sensor", main_cls="door")
        # smart plugs on a third of the sockets, named for what is plugged in
        gpos: dict[str, int] = {}
        for f in plan.fixtures:
            if f["type"] == "gpo":
                aid, dist = plan.room_at(f)
                gpos[aid] = gpos.get(aid, 0) + 1
        for aid, n in sorted(gpos.items()):
            a = house.area(aid)
            (lo, hi), pool = PLUGS.get(a.kind, ((0, 1), ["Plug"]))
            for label in pick_names(srng, pool or ["Plug"], max(1, round(n / 3)), {e.name[len(self._an(aid)) + 1:] for e in house.in_area(aid)}):
                dev = self._device(house, "meross_plug", aid, f"{self._an(aid)} {label}", main_cls="switch", main_id=f"switch.{aid}_{slug(label)}",
                                   load_w=60, circuit=None)
                dev["main"].state = "on" if self.p(0.45) else "off"
        # people, weather, water, vehicle, mower, NAS
        for nm, _ in BUNDOOK.people:
            self._device(house, "phone", "_people", f"{nm}'s iPhone")
        self._device(house, "ecowitt_station", "weather", "Weather Station")
        self._add(house, "sensor", "sensor.weather_station_outdoor_temperature", "Weather Station Outdoor Temperature", "temperature", "weather",
                  state="15", attributes={"device": "weather_station"}, changed_s=self.rng.randint(30, 600))
        for z in BUNDOOK.irrigation_zones:
            self._attach(house, house.entities[f"switch.irrigation_{slug(z)}"], "ecowitt_valve", area="water")
        self._attach(house, house.entities["sensor.tank_level"], "tank_sensor", area="water")
        self._attach(house, house.entities["sensor.shark_battery"], "byd_shark", area="carport")
        self._device(house, "luba_mower", "alfresco", "Lawn Mower")
        self._device(house, "synology", "storage", "NAS")


def make_bundook_full(seed: int, projected: bool = False) -> House:
    return BundookGen(seed, projected).make()


def summary():
    import collections
    for projected in (False, True):
        h = make_bundook_full(9_300_001, projected)
        cats = collections.Counter(e.attributes.get("entity_category") or "primary" for e in h.entities.values())
        print(("bundook_full_projected" if projected else "bundook_full"), len(h.entities), dict(cats))


if __name__ == "__main__":
    summary()
