"""Random but internally consistent houses. Every training example is built on a
fresh house so the model learns behaviours (how to read an inventory, fetch the
right state, reason about presence and power) rather than one specific home.

make_house(seed) is deterministic: the eval harness regenerates the house from
the seed to resolve area targets against entity ids."""
from __future__ import annotations

import math
import random
import re
from datetime import datetime, timedelta

from .profiles import PROFILES as HOUSE_PROFILES
from .profiles import AreaSpec, Profile
from .schema import Area, Entity, House, Person, Speaker

NAMES = [
    "Amber", "Peter", "Sam", "Jess", "Tom", "Mia", "Liam", "Chloe", "Noah", "Ella", "Jack", "Zoe",
    "Alex", "Ruby", "Max", "Lucy", "Ben", "Grace", "Oliver", "Sophie", "Kate", "Dan", "Emma", "Josh",
    "Hannah", "Leo", "Isla", "Charlie", "Amelia", "Harry", "Freya", "Ethan", "Molly", "Nathan", "Tara",
    "Ryan", "Nina", "Owen", "Priya", "Raj", "Sofia", "Marcus", "Elena", "Kai", "Yuki",
]

# (area_id, name, kind)
ROOM_POOL = [
    ("bedroom", "Bedroom", "bedroom"), ("master_bedroom", "Master Bedroom", "bedroom"),
    ("guest_room", "Guest Room", "bedroom"), ("spare_room", "Spare Room", "bedroom"),
    ("office", "Office", "office"), ("study", "Study", "office"),
    ("kitchen", "Kitchen", "kitchen"),
    ("living_room", "Living Room", "living"), ("lounge", "Lounge", "living"),
    ("dining_room", "Dining Room", "living"), ("family_room", "Family Room", "living"),
    ("media_room", "Media Room", "living"), ("rumpus_room", "Rumpus Room", "living"),
    ("bathroom", "Bathroom", "bath"), ("ensuite", "Ensuite", "bath"),
    ("hallway", "Hallway", "entry"), ("entry", "Entry", "entry"),
    ("laundry", "Laundry", "utility"), ("workshop", "Workshop", "utility"), ("garage", "Garage", "garage"),
    ("patio", "Patio", "outdoor"), ("backyard", "Backyard", "outdoor"), ("front_porch", "Front Porch", "outdoor"),
    ("deck", "Deck", "outdoor"),
]

# Per room kind: which devices exist and with what probability.
PROFILES = {
    "bedroom": dict(lights=[("Ceiling", 1.0), ("Bedside Lamp", 0.55), ("Lamp", 0.2)], motion=0.5, occupancy=0.35,
                    temp=0.7, humidity=0.2, media=("Speaker", 0.55), fan=0.5, cover=("Blinds", 0.5), window=0.5, heater=0.35),
    "office": dict(lights=[("Ceiling", 1.0), ("Desk Lamp", 0.7)], motion=0.7, occupancy=0.35, temp=0.7, humidity=0.1,
                   media=("Speaker", 0.65), fan=0.4, cover=("Blinds", 0.4), window=0.4, heater=0.4),
    "kitchen": dict(lights=[("Ceiling", 1.0), ("Bench Lights", 0.5), ("Pantry Light", 0.3)], motion=0.85, occupancy=0.1,
                    temp=0.6, humidity=0.2, media=("Speaker", 0.7), fan=0.0, cover=("Blinds", 0.2), window=0.5, heater=0.05),
    "living": dict(lights=[("Ceiling", 1.0), ("Lamp", 0.6), ("Downlights", 0.3)], motion=0.8, occupancy=0.25, temp=0.7,
                   humidity=0.15, media=("TV", 0.85), fan=0.3, cover=("Blinds", 0.5), window=0.4, heater=0.3),
    "bath": dict(lights=[("Ceiling", 1.0)], motion=0.6, occupancy=0.05, temp=0.3, humidity=0.6, media=None, fan=0.6,
                 cover=None, window=0.2, heater=0.15),
    "entry": dict(lights=[("Ceiling", 1.0)], motion=0.75, occupancy=0.0, temp=0.15, humidity=0.0, media=None, fan=0.0,
                  cover=None, window=0.0, heater=0.0),
    "utility": dict(lights=[("Ceiling", 1.0)], motion=0.4, occupancy=0.0, temp=0.2, humidity=0.2, media=None, fan=0.0,
                    cover=None, window=0.2, heater=0.0),
    "garage": dict(lights=[("Light", 1.0)], motion=0.5, occupancy=0.0, temp=0.3, humidity=0.0, media=None, fan=0.0,
                   cover=None, window=0.0, heater=0.0),
    "outdoor": dict(lights=[("Lights", 1.0)], motion=0.7, occupancy=0.0, temp=0.3, humidity=0.0, media=None, fan=0.0,
                    cover=None, window=0.0, heater=0.0),
}

CONDITION_FACTOR = {"sunny": 1.0, "partlycloudy": 0.65, "cloudy": 0.3, "rainy": 0.15, "clear-night": 1.0}


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    return s.strip("_")


class Gen:
    def __init__(self, seed: int, profile: Profile | None = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.profile = profile

    def p(self, prob: float) -> bool:
        return self.rng.random() < prob

    # ------------------------------------------------------------------
    def make(self) -> House:
        rng = self.rng
        # --- time and sun ---
        weekday = rng.randrange(7)
        hour = rng.choices(range(24), weights=[1, 1, 1, 1, 1, 2, 5, 8, 8, 6, 5, 5, 5, 5, 5, 6, 7, 9, 10, 10, 9, 8, 6, 3])[0]
        base = datetime(2025, 3, 3) + timedelta(days=weekday)  # 2025-03-03 is a Monday
        now = base.replace(hour=hour, minute=rng.randrange(60))
        day_len = rng.uniform(9.5, 14.5)
        rise_min = int(12.5 * 60 - day_len * 30 + rng.uniform(-20, 20))
        set_min = int(rise_min + day_len * 60)
        now_min = hour * 60 + now.minute
        self.rise_min, self.set_min, self.now_min = rise_min, set_min, now_min
        dark = not (rise_min < now_min < set_min - 30)

        house = House(seed=self.seed, now=now, areas=[], entities={}, people=[])
        prof = self.profile
        if prof:
            house.description, house.profile = prof.description, prof.name

        # --- people ---
        if prof:
            names = [n for n, _ in prof.people]
            if self.p(prof.guest_prob):
                names.append(rng.choice([n for n in NAMES if n not in names]))
        else:
            n_people = rng.choices([1, 2, 3, 4], weights=[0.1, 0.4, 0.32, 0.18])[0]
            names = rng.sample(NAMES, n_people)
        for nm in names:
            house.people.append(Person(name=nm, entity_id=f"person.{slug(nm)}", home=True, room=None, changed_s=0))
        self.season_cold = self.p(0.5)
        self.house_has_ma = self.p(0.7)  # most houses run Music Assistant for some speakers
        if prof:
            self._build_profile(house, prof)
        else:
            self._build_random(house, names)

        # --- presence: who is home and where ---
        indoor = [a for a in house.areas if a.kind not in ("outdoor",)]
        for p in house.people:
            p.home = self.p(prof.home_prob if prof else 0.75)
            if p.home:
                p.changed_s = rng.randint(1800, 12 * 3600)
                p.room = self._choose_room(house, p, indoor)
            else:
                p.changed_s = rng.randint(600, 9 * 3600)
                p.room = None
            self._add(house, "person", p.entity_id, p.name, "person", "_people",
                      state="home" if p.home else "not_home", changed_s=p.changed_s)
        self.place_people(house)
        self._states(house, dark)
        self._weather(house, dark)
        self._energy(house)
        return house

    def _choose_room(self, house: House, p: Person, indoor: list[Area]):
        """Where a person at home is. With a profile, people who work from home are usually
        in their own office in work hours, in the living area in the evening, in bed late."""
        rng = self.rng
        prof = self.profile
        if not self.p(0.85):
            return None
        if prof:
            own = dict(prof.people).get(p.name)
            h, wd = house.now.hour + house.now.minute / 60, house.now.weekday()
            beds = [a.area_id for a in house.areas if a.kind == "bedroom" and "guest" not in a.area_id]
            if own is None:  # guest
                return rng.choice([r for r in prof.guest_rooms if house.area(r)] or [a.area_id for a in indoor])
            for blk in prof.schedules.get(p.name, []):
                if blk.active(wd, h):
                    room = beds[0] if (blk.room == "bed" and beds) else blk.room
                    if house.area(room) and self.p(blk.prob):
                        return room
                    break  # block applies but they're elsewhere this time
            if not prof.schedules.get(p.name):
                if wd < 5 and 8 <= h < 18 and own and house.area(own) and self.p(0.65):
                    return own
                if (h >= 22 or h < 7) and beds and self.p(0.8):
                    return beds[0]
            living = [a.area_id for a in house.areas if a.kind in ("living", "kitchen")]
            if 17 <= h < 22.5 and living and self.p(0.45):
                return rng.choice(living)
        return rng.choice(indoor).area_id

    def _build_random(self, house: House, names: list[str]):
        rng = self.rng
        n_people = len(names)
        chosen: list[tuple[str, str, str]] = []
        pool = list(ROOM_POOL)

        def take(pred):
            opts = [r for r in pool if pred(r)]
            r = rng.choice(opts)
            pool.remove(r)
            chosen.append(r)

        take(lambda r: r[2] == "kitchen")
        take(lambda r: r[0] in ("living_room", "lounge", "family_room"))
        take(lambda r: r[0] in ("bedroom", "master_bedroom"))
        if self.p(0.7):
            take(lambda r: r[2] == "office")
        if n_people >= 3 and self.p(0.5):
            kid = names[-1]
            chosen.append((f"{slug(kid)}_room", f"{kid}'s Room", "bedroom"))
        target = rng.randint(5, 9)
        while len(chosen) < target and pool:
            take(lambda r: True)
        for aid, name, kind in chosen:
            house.areas.append(Area(aid, name, kind))
        for area in house.areas:
            self._area_devices(house, area)
        self._security(house)
        living = [a for a in house.areas if a.kind == "living"]
        if living and self.p(0.7):
            a = living[0]
            self._add(house, "climate", "climate.house", "House Thermostat", "climate", a.area_id,
                      state="off", attributes={"temperature": 21.0, "current_temperature": 20.0})
        if living and self.p(0.3):
            self._add(house, "vacuum", "vacuum.robot", "Robot Vacuum", "vacuum", living[0].area_id,
                      state="docked", changed_s=rng.randint(3600, 36000))
        self.energy_params = dict(peak_kw=rng.choice([3.3, 4.0, 5.0, 6.0, 6.6, 8.0, 10.0]),
                                  cap_kwh=rng.choice([10.0, 14.0, 15.0, 20.0, 28.0, 30.0]))
        if self.p(0.6):
            on = self.p(0.2)
            self._add(house, "switch", "switch.hot_water", "Hot Water", "switch", "energy",
                      state="on" if on else "off", attributes={"power_w": rng.choice([1800, 2400, 3000, 3600]), "appliance": True},
                      changed_s=rng.randint(300, 4 * 3600))
        if self.p(0.45):
            self._add(house, "switch", "switch.generator", "Generator", "switch", "energy", state="off",
                      changed_s=rng.randint(3600, 3 * 86400))
        if self.p(0.6):
            zones = rng.sample(["Vegie Beds", "Orchard", "Front Lawn", "Back Lawn", "Greenhouse", "Fruit Trees"], rng.randint(0, 3)) if self.p(0.7) else []
            self._water(house, zones)
        if self.p(0.35):
            self._vehicle(house, rng.choice(["Car", "EV", "Ute"]), "sensor.car", backup=False)

    def _build_profile(self, house: House, prof: Profile):
        rng = self.rng
        for s in prof.areas:
            house.areas.append(Area(s.area_id, s.name, s.kind, spoken=s.spoken))
            aid, an = s.area_id, (s.spoken.split("'s ")[0] + "'s Office" if s.spoken and "'s office" in s.spoken else s.name)
            for lname in s.lights:
                self._add(house, "light", f"light.{aid}_{slug(lname)}", f"{an} {lname}", "light", aid)
            for i in range(s.occupancy):
                suffix = "" if s.occupancy == 1 else f" {i + 1}"
                nm = f"{an} Occupancy{suffix}" if not (aid == "dining_hall" and i == 1) else "Sitting Area Occupancy"
                self._add(house, "binary_sensor", f"binary_sensor.{slug(nm)}", nm, "occupancy", aid)
            for i in range(s.motion):
                suffix = "" if s.motion == 1 else f" {i + 1}"
                self._add(house, "binary_sensor", f"binary_sensor.{aid}_motion{suffix.replace(' ', '_')}", f"{an} Motion{suffix}", "motion", aid)
            if s.temperature:
                self._add(house, "sensor", f"sensor.{aid}_temperature", f"{an} Temperature", "temperature", aid, state="20", changed_s=rng.randint(30, 900))
            if s.humidity:
                self._add(house, "sensor", f"sensor.{aid}_humidity", f"{an} Humidity", "humidity", aid, state=str(rng.randint(35, 75)), changed_s=rng.randint(30, 900))
            if s.blinds:
                closed = self.p(0.5)
                self._add(house, "cover", f"cover.{aid}_blinds", f"{an} Blinds", "cover", aid, state="closed" if closed else "open", attributes={"position": 0 if closed else 100})
            if s.windows:
                opn = self.p(0.25)
                self._add(house, "cover", f"cover.{aid}_windows", f"{an} Windows", "cover", aid, state="open" if opn else "closed", attributes={"position": 100 if opn else 0})
            if s.window_sensor:
                self._add(house, "binary_sensor", f"binary_sensor.{aid}_window", f"{an} Window", "window", aid, state="on" if self.p(0.2) else "off")
            if s.climate:
                self._add(house, "climate", f"climate.{aid}", f"{an} Air Conditioning", "climate", aid, state="off", attributes={"temperature": 21.0, "current_temperature": 20.0})
            if s.fan:
                self._add(house, "fan", f"fan.{aid}_{slug(s.fan)}", f"{an} {s.fan}", "fan", aid)
            if s.exhaust:
                self._add(house, "fan", f"fan.{aid}_exhaust", f"{an} Exhaust Fan", "fan", aid)
            if s.media:
                self._add(house, "media_player", f"media_player.{aid}_{slug(s.media)}", f"{an} {s.media}", "media_player", aid, state="off",
                          attributes={"ma": True} if s.media != "TV" else {})
            if s.camera:
                self._add(house, "camera", f"camera.{aid}", f"{an} {s.camera}", "camera", aid, state="idle")
            if s.towel_rail:
                self._add(house, "switch", f"switch.{aid}_towel_rail", f"{an} Towel Rail", "switch", aid,
                          state="on" if self.p(0.3) else "off", attributes={"power_w": 200})
            if s.vacuum:
                self._add(house, "vacuum", "vacuum.robot", "Robot Vacuum", "vacuum", aid, state="docked", changed_s=rng.randint(3600, 36000))
            for kind in s.appliances:
                self._appliance(house, aid, an, kind)
            if s.dehumidifier:
                on = self.p(0.3)
                self._add(house, "switch", f"switch.{aid}_dehumidifier", f"{an} Dehumidifier", "switch", aid, state="on" if on else "off",
                          attributes={"power_w": 300, "appliance": True}, changed_s=rng.randint(300, 6 * 3600))
        for nm in prof.locks:
            self._add(house, "lock", f"lock.{slug(nm)}", nm, "lock", "security", state="locked" if self.p(0.7) else "unlocked")
        for nm in prof.door_sensors:
            self._add(house, "binary_sensor", f"binary_sensor.{slug(nm)}", nm, "door", "security", state="on" if self.p(0.08) else "off")
        for nm in prof.cameras:
            self._add(house, "camera", f"camera.{slug(nm)}", nm, "camera", "security", state="idle")
        if prof.alarm:
            self._add(house, "alarm_control_panel", "alarm_control_panel.home_alarm", "Home Alarm", "alarm", "security",
                      state=rng.choices(["disarmed", "armed_home", "armed_away"], weights=[0.7, 0.2, 0.1])[0])
        self.energy_params = dict(peak_kw=prof.peak_kw, cap_kwh=prof.cap_kwh)
        for eid, name, watts, kind in prof.loads:
            if kind == "heating":
                on = self.p(0.5 if self.season_cold else 0.03)
            elif kind == "hot_water":
                on = self.p(0.3)
            else:
                on = self.p(0.15)
            self._add(house, "switch", eid, name, "switch", "energy", state="on" if on else "off",
                      attributes={"power_w": watts, "appliance": True}, changed_s=rng.randint(300, 4 * 3600))
        if prof.water:
            self._water(house, prof.irrigation_zones)
        if prof.vehicle_backup:
            self._vehicle(house, "BYD Shark", "sensor.shark_battery", backup=True)

    def _states(self, house: House, dark: bool):
        """Lights, heaters, media and fans as a function of darkness and occupancy."""
        rng = self.rng

        occupied = set(house.occupied_areas()) | {p.room for p in house.people if p.home and p.room}
        for e in list(house.entities.values()):
            if e.cls == "light":
                a = house.area(e.area)
                outdoor = a is not None and a.kind == "outdoor"
                if outdoor:
                    on = dark and self.p(0.3)
                elif e.area in occupied:
                    on = self.p(0.85 if dark else 0.3)
                else:
                    on = self.p(0.25 if dark else 0.08)
                e.state = "on" if on else "off"
                if on:
                    e.attributes["brightness_pct"] = rng.choice([100, 100, 80, 60, 50, 40, 30, 20])
                e.changed_s = rng.randint(60, 4 * 3600)
            elif e.cls == "switch" and e.attributes.get("heater"):
                on = self.p(0.5 if (self.season_cold and e.area in occupied) else 0.08)
                e.state = "on" if on else "off"
                e.changed_s = rng.randint(300, 4 * 3600)
            elif e.cls == "media_player":
                is_tv = "tv" in e.name.lower()
                if e.area in occupied and self.p(0.5):
                    e.state = rng.choice(["playing", "playing", "paused"])
                    if is_tv:
                        e.attributes["media_title"] = rng.choice(["The Bear", "Bluey", "Planet Earth", "ABC News", "The Great British Bake Off", "Test cricket", "Severance"])
                        e.attributes["media_content_type"] = "tvshow"
                    else:
                        title, artist = rng.choice(TRACKS)
                        e.attributes["media_title"], e.attributes["media_artist"] = title, artist
                        e.attributes["media_content_type"] = rng.choice(["music", "music", "playlist", "radio"])
                        if e.attributes["media_content_type"] == "radio":
                            e.attributes["media_title"], e.attributes["media_artist"] = rng.choice(RADIO), None
                            e.attributes.pop("media_artist")
                        e.attributes["shuffle"] = self.p(0.3)
                    e.attributes["volume_pct"] = rng.choice([15, 20, 25, 30, 40, 50])
                else:
                    e.state = rng.choice(["off", "idle", "idle"])
                    e.attributes["volume_pct"] = rng.choice([20, 30, 40])
                e.changed_s = rng.randint(120, 5 * 3600)
            elif e.cls == "fan":
                on = self.p(0.35 if (not self.season_cold and e.area in occupied) else 0.05)
                e.state = "on" if on else "off"
                if on:
                    e.attributes["percentage"] = rng.choice([33, 66, 100])
                e.changed_s = rng.randint(120, 5 * 3600)
            elif e.cls == "climate":
                pass  # set in _weather

    # ------------------------------------------------------------------
    def _add(self, house: House, domain: str, entity_id: str, name: str, cls: str, area: str,
             state: str = "off", attributes: dict | None = None, changed_s: int | None = None) -> Entity:
        e = Entity(entity_id=entity_id, domain=domain, name=name, cls=cls, area=area, state=state,
                   attributes=dict(attributes or {}),
                   changed_s=self.rng.randint(300, 8 * 3600) if changed_s is None else changed_s)
        house.entities[entity_id] = e
        return e

    def _appliance(self, house: House, aid: str, an: str, kind: str):
        rng = self.rng
        st = rng.choices(["off", "running", "finished", "door open"], weights=[0.6, 0.2, 0.15, 0.05 if kind in ("Fridge", "Freezer", "Oven") else 0.0])[0]
        attrs = {"appliance": kind.lower(), "status": st, "power_w": APPLIANCE_POWER.get(kind, 500)}
        if kind in ("Fridge", "Freezer"):
            st = "door open" if st == "door open" else "on"
            attrs["status"] = st
        if st == "running":
            attrs["remaining_min"] = rng.choice([5, 12, 20, 35, 48, 65, 90])
            if kind in APPLIANCE_PROGRAMS:
                attrs["program"] = rng.choice(APPLIANCE_PROGRAMS[kind])
        state = "on" if st in ("running", "on") else "off"
        self._add(house, "switch", f"switch.{aid}_{slug(kind)}", f"{an} {kind}", "appliance", aid, state=state, attributes=attrs,
                  changed_s=rng.randint(600, 3 * 3600) if st == "finished" else rng.randint(60, 6 * 3600))

    def _water(self, house: House, zones: list[str]):
        rng = self.rng
        cap = rng.choice([10000, 22000, 27000, 45000, 90000])
        pct = rng.randint(15, 100)
        self._add(house, "sensor", "sensor.tank_level", "Water Tank", "water", "water", state=str(pct),
                  attributes={"kind": "level", "litres": cap * pct / 100, "capacity_l": cap}, changed_s=rng.randint(60, 3600))
        on = self.p(0.15)
        self._add(house, "switch", "switch.house_pump", "House Pump", "pump", "water", state="on" if on else "off",
                  attributes={"power_w": rng.choice([750, 900, 1100])}, changed_s=rng.randint(60, 4 * 3600))
        if self.p(0.4):
            self._add(house, "sensor", "sensor.water_pressure", "Water Pressure", "water", "water", state=f"{rng.uniform(2.5, 4.5):.1f}",
                      attributes={"kind": "pressure"}, changed_s=rng.randint(30, 900))
        for z in zones:
            on = self.p(0.1)
            self._add(house, "switch", f"switch.irrigation_{slug(z)}", f"{z} Irrigation", "irrigation", "water", state="on" if on else "off",
                      attributes={"default_min": rng.choice([15, 20, 30, 45]), **({"remaining_min": rng.randint(2, 40)} if on else {})},
                      changed_s=rng.randint(1800, 3 * 86400))

    def _vehicle(self, house: House, name: str, entity_id: str, backup: bool):
        rng = self.rng
        loc = rng.choices(["parked here", "parked here", "at the old house", "away"], weights=[0.4, 0.2, 0.2, 0.2])[0]
        plugged = loc.startswith("parked here") and self.p(0.5)
        soc = rng.randint(15, 100)
        self._add(house, "sensor", entity_id, name, "vehicle", "energy", state=str(soc),
                  attributes={"range_km": round(soc * 5.5), "location": loc + (" charging" if plugged and self.p(0.5) else ""),
                              "plugged": plugged, "locked": self.p(0.85)}, changed_s=rng.randint(60, 3600))
        if backup:
            feeding = plugged and self.p(0.25)
            self._add(house, "switch", "switch.shark_backup", "Shark Backup Feed", "switch", "energy",
                      state="on" if feeding else "off", attributes={"source_w": 3000}, changed_s=rng.randint(600, 6 * 3600))

    def _area_devices(self, house: House, area: Area):
        prof = PROFILES[area.kind]
        aid, an = area.area_id, area.name
        for lname, prob in prof["lights"]:
            if self.p(prob):
                self._add(house, "light", f"light.{aid}_{slug(lname)}", f"{an} {lname}", "light", aid)
        if self.p(prof["motion"]):
            self._add(house, "binary_sensor", f"binary_sensor.{aid}_motion", f"{an} Motion", "motion", aid)
        elif self.p(prof["occupancy"] / max(1e-6, 1 - prof["motion"])) if prof["motion"] < 1 else False:
            self._add(house, "binary_sensor", f"binary_sensor.{aid}_occupancy", f"{an} Occupancy", "occupancy", aid)
        if self.p(prof["temp"]):
            self._add(house, "sensor", f"sensor.{aid}_temperature", f"{an} Temperature", "temperature", aid,
                      state="20", changed_s=self.rng.randint(30, 900))
        if self.p(prof["humidity"]):
            self._add(house, "sensor", f"sensor.{aid}_humidity", f"{an} Humidity", "humidity", aid,
                      state=str(self.rng.randint(35, 70)), changed_s=self.rng.randint(30, 900))
        if prof["media"] and self.p(prof["media"][1]):
            mname = prof["media"][0]
            attrs = {"ma": True} if (mname != "TV" and getattr(self, "house_has_ma", False) and self.p(0.8)) else {}
            self._add(house, "media_player", f"media_player.{aid}_{slug(mname)}", f"{an} {mname}", "media_player", aid, state="off", attributes=attrs)
        if self.p(prof["fan"]):
            fname = "Exhaust Fan" if area.kind == "bath" else "Fan"
            self._add(house, "fan", f"fan.{aid}_{slug(fname)}", f"{an} {fname}", "fan", aid)
        if prof["cover"] and self.p(prof["cover"][1]):
            cname = prof["cover"][0]
            closed = self.p(0.5)
            self._add(house, "cover", f"cover.{aid}_{slug(cname)}", f"{an} {cname}", "cover", aid,
                      state="closed" if closed else "open", attributes={"position": 0 if closed else 100})
        if self.p(prof["window"]):
            self._add(house, "binary_sensor", f"binary_sensor.{aid}_window", f"{an} Window", "window", aid,
                      state="on" if self.p(0.2) else "off")
            if self.p(0.35):  # motorised windows (openable as a cover)
                opn = self.p(0.25)
                self._add(house, "cover", f"cover.{aid}_windows", f"{an} Windows", "cover", aid,
                          state="open" if opn else "closed", attributes={"position": 100 if opn else 0})
        if area.kind == "kitchen":
            for kind, prob in (("Dishwasher", 0.7), ("Oven", 0.6), ("Fridge", 0.5), ("Air Fryer", 0.2)):
                if self.p(prob):
                    self._appliance(house, aid, an, kind)
        if area.kind == "utility" and aid in ("laundry",) or (area.kind == "utility" and self.p(0.3)):
            for kind, prob in (("Washing Machine", 0.7), ("Dryer", 0.5)):
                if self.p(prob):
                    self._appliance(house, aid, an, kind)
        if area.kind in ("utility", "bath", "bedroom") and self.p(0.15):
            on = self.p(0.3)
            self._add(house, "switch", f"switch.{aid}_dehumidifier", f"{an} Dehumidifier", "switch", aid, state="on" if on else "off",
                      attributes={"power_w": 300, "appliance": True}, changed_s=self.rng.randint(300, 6 * 3600))
        if self.p(prof["heater"]):
            self._add(house, "switch", f"switch.{aid}_heater", f"{an} Heater", "switch", aid,
                      attributes={"heater": True, "power_w": self.rng.choice([1200, 1500, 1800, 2000, 2400])})
        if area.kind == "garage":
            self._add(house, "cover", "cover.garage_door", "Garage Door", "garage", aid,
                      state="closed" if self.p(0.8) else "open", attributes={"position": 0})
            house.entities["cover.garage_door"].attributes["position"] = 0 if house.entities["cover.garage_door"].state == "closed" else 100
            if self.p(0.4):
                self._add(house, "binary_sensor", f"binary_sensor.{aid}_side_door", f"{an} Side Door", "door", aid,
                          state="on" if self.p(0.1) else "off")
        if area.kind == "outdoor" and self.p(0.5):
            self._add(house, "camera", f"camera.{aid}", f"{an} Camera", "camera", aid, state="idle")

    def _security(self, house: House):
        self._add(house, "lock", "lock.front_door", "Front Door", "lock", "security",
                  state="locked" if self.p(0.7) else "unlocked")
        if self.p(0.9):
            self._add(house, "binary_sensor", "binary_sensor.front_door", "Front Door", "door", "security",
                      state="on" if self.p(0.08) else "off")
        if self.p(0.7):
            self._add(house, "lock", "lock.back_door", "Back Door", "lock", "security",
                      state="locked" if self.p(0.6) else "unlocked")
            if self.p(0.8):
                self._add(house, "binary_sensor", "binary_sensor.back_door", "Back Door", "door", "security",
                          state="on" if self.p(0.12) else "off")
        if self.p(0.8):
            self._add(house, "alarm_control_panel", "alarm_control_panel.home_alarm", "Home Alarm", "alarm", "security",
                      state=self.rng.choices(["disarmed", "armed_home", "armed_away"], weights=[0.7, 0.2, 0.1])[0])
        if self.p(0.5):
            self._add(house, "camera", "camera.front_door", "Front Door Camera", "camera", "security", state="idle")
        if not house.area("garage") and self.p(0.35):
            closed = self.p(0.8)
            self._add(house, "cover", "cover.garage_door", "Garage Door", "garage", "security",
                      state="closed" if closed else "open", attributes={"position": 0 if closed else 100})

    def place_people(self, house: House):
        """Set motion/occupancy sensors from the ground-truth rooms of people at home."""
        rng = self.rng
        occupied = {p.room for p in house.people if p.home and p.room}
        for e in house.entities.values():
            if e.cls == "motion":
                if e.area in occupied:
                    e.state, e.changed_s = "on", rng.randint(0, 120)
                else:
                    e.state, e.changed_s = "off", rng.choices([rng.randint(300, 1200), rng.randint(1200, 8 * 3600)], weights=[0.3, 0.7])[0]
            elif e.cls == "occupancy":
                if e.area in occupied:
                    e.state, e.changed_s = "on", rng.randint(60, 3 * 3600)
                else:
                    e.state, e.changed_s = "off", rng.randint(300, 8 * 3600)

    def _weather(self, house: House, dark: bool):
        rng = self.rng
        cond = rng.choices(["sunny", "partlycloudy", "cloudy", "rainy"], weights=[0.4, 0.3, 0.2, 0.1])[0]
        self.cond = cond
        shown = "clear-night" if (dark and cond == "sunny") else cond
        if self.season_cold:
            low, high = rng.uniform(2, 9), rng.uniform(11, 18)
        else:
            low, high = rng.uniform(12, 19), rng.uniform(22, 34)
        # crude diurnal curve: min at sunrise, max mid afternoon
        x = ((self.now_min - self.rise_min) % 1440) / 1440
        cur = low + (high - low) * max(0.0, math.sin(math.pi * min(1.0, x * 1.6)))
        self.outdoor_temp = cur
        tcond = rng.choices(["sunny", "partlycloudy", "cloudy", "rainy"], weights=[0.4, 0.3, 0.2, 0.1])[0]
        self.tomorrow_cond = tcond
        self._add(house, "weather", "weather.home", "Home Weather", "weather", "weather", state=shown,
                  attributes={"temperature": round(cur, 1), "humidity": rng.randint(30, 90), "wind_kmh": rng.randint(0, 45),
                              "forecast_tomorrow": {"condition": tcond, "low": round(low + rng.uniform(-3, 3), 0),
                                                    "high": round(high + rng.uniform(-4, 4), 0)}},
                  changed_s=rng.randint(60, 1800))
        above = self.rise_min < self.now_min < self.set_min
        self._add(house, "sun", "sun.sun", "Sun", "sun", "weather", state="above_horizon" if above else "below_horizon",
                  attributes={"next_rising": f"{self.rise_min // 60:02d}:{self.rise_min % 60:02d}",
                              "next_setting": f"{(self.set_min // 60) % 24:02d}:{self.set_min % 60:02d}"},
                  changed_s=abs(self.now_min - (self.set_min if above else self.rise_min)) * 60 % (12 * 3600) or 60)
        if self.p(0.6):
            self._add(house, "sensor", "sensor.outdoor_temperature", "Outdoor Temperature", "temperature", "weather",
                      state=f"{cur:.1f}", changed_s=rng.randint(30, 900))
        # indoor temperatures
        for e in house.entities.values():
            if e.cls == "temperature" and e.area != "weather":
                heater_on = any(h.state == "on" and h.attributes.get("heater") for h in house.in_area(e.area, "switch"))
                indoor = (17.5 if self.season_cold else 23.0) + rng.uniform(-2.5, 2.5) + (2.5 if heater_on else 0)
                e.state = f"{indoor:.1f}"
        for clim in house.by_class("climate"):
            clim.attributes["current_temperature"] = round((17.5 if self.season_cold else 23.5) + rng.uniform(-2, 2), 1)
            clim.attributes["temperature"] = float(rng.choice([20, 21, 21, 22, 23]))
            occupied_here = clim.area in {p.room for p in house.people if p.home}
            w_on = (0.35 if self.season_cold else 0.35) * (1.0 if occupied_here else 0.3)
            clim.state = rng.choices(["off", "heat" if self.season_cold else "cool"], weights=[1 - w_on, w_on])[0]
            clim.attributes["last_mode"] = "heat" if self.season_cold else "cool"

    def _energy(self, house: House):
        rng = self.rng
        peak_kw = self.energy_params["peak_kw"]
        cap_kwh = self.energy_params["cap_kwh"]
        factor = CONDITION_FACTOR[self.cond]
        rise, sset, nowm = self.rise_min, self.set_min, self.now_min
        if rise < nowm < sset:
            x = (nowm - rise) / (sset - rise)
            curve = math.sin(math.pi * x) ** 1.5
            frac = (1 - math.cos(math.pi * x)) / 2
        else:
            curve, frac = 0.0, (1.0 if nowm >= sset else 0.0)
        solar_w = peak_kw * 1000 * curve * factor * rng.uniform(0.85, 1.05)
        expected_daily = peak_kw * 4.8 * factor
        energy_today = expected_daily * frac * rng.uniform(0.9, 1.1)
        forecast_rem = max(0.0, expected_daily - energy_today) * rng.uniform(0.85, 1.15)
        forecast_tom = peak_kw * 4.8 * CONDITION_FACTOR[self.tomorrow_cond] * rng.uniform(0.85, 1.15)

        base_load = rng.uniform(250, 900) * (1.6 if self.profile else 1.0)
        load_w = base_load
        gen_w = 0.0
        for e in house.entities.values():
            if e.cls == "switch" and e.state == "on" and e.attributes.get("source_w"):
                gen_w += e.attributes["source_w"]
            elif e.cls in ("switch", "appliance", "pump") and e.state == "on" and e.attributes.get("power_w"):
                load_w += e.attributes["power_w"] if e.cls != "appliance" or e.attributes.get("status") in ("running", "on") else 0
            elif e.cls == "light" and e.state == "on":
                load_w += 10
            elif e.cls == "media_player" and e.state in ("playing", "paused"):
                load_w += 90
            elif e.cls == "climate" and e.state != "off":
                load_w += 1400
        net = solar_w + gen_w - load_w
        hour = nowm / 60
        if hour < 6:
            soc = rng.uniform(25, 75)
        elif hour < 10:
            soc = rng.uniform(25, 70)
        elif hour < 14:
            soc = rng.uniform(45, 98)
        elif hour < 18:
            soc = rng.uniform(60, 100)
        else:
            soc = rng.uniform(35, 98)
        if net > 0 and soc >= 99.5:
            direction, bat_w = "idle", 0.0
            soc = 100.0
        elif net > 0:
            direction, bat_w = "charging", net
        elif net < 0:
            direction, bat_w = "discharging", net
        else:
            direction, bat_w = "idle", 0.0
        remaining = soc / 100 * cap_kwh
        if direction == "discharging":
            runtime = remaining / (-net / 1000)
            runtime_state = f"{runtime:.1f}"
        else:
            runtime_state = "n/a (charging)"
        ts = rng.randint(5, 120)
        self._add(house, "sensor", "sensor.battery_soc", "Battery Charge", "battery", "energy",
                  state=f"{soc:.0f}", attributes={"kind": "soc", "capacity_kwh": cap_kwh}, changed_s=ts)
        self._add(house, "sensor", "sensor.battery_power", "Battery Power", "battery", "energy",
                  state=f"{bat_w:.0f}", attributes={"kind": "power", "direction": direction}, changed_s=ts)
        self._add(house, "sensor", "sensor.battery_energy_remaining", "Battery Energy Remaining", "battery", "energy",
                  state=f"{remaining:.1f}", attributes={"kind": "remaining"}, changed_s=ts)
        self._add(house, "sensor", "sensor.battery_runtime", "Battery Runtime Estimate", "battery", "energy",
                  state=runtime_state, attributes={"kind": "runtime"}, changed_s=ts)
        self._add(house, "sensor", "sensor.solar_power", "Solar Power", "solar", "energy",
                  state=f"{solar_w:.0f}", attributes={"kind": "power", "peak_kw": peak_kw}, changed_s=ts)
        self._add(house, "sensor", "sensor.solar_energy_today", "Solar Energy Today", "solar", "energy",
                  state=f"{energy_today:.1f}", attributes={"kind": "energy_today"}, changed_s=ts)
        self._add(house, "sensor", "sensor.house_load", "House Load", "load", "energy",
                  state=f"{load_w:.0f}", attributes={}, changed_s=ts)
        self._add(house, "sensor", "sensor.solar_forecast_remaining_today", "Solar Forecast Rest Of Today", "forecast", "energy",
                  state=f"{forecast_rem:.1f}", attributes={"kind": "remaining_today"}, changed_s=rng.randint(600, 3600))
        self._add(house, "sensor", "sensor.solar_forecast_tomorrow", "Solar Forecast Tomorrow", "forecast", "energy",
                  state=f"{forecast_tom:.1f}", attributes={"kind": "tomorrow"}, changed_s=rng.randint(600, 3600))


TRACKS = [("Cruel Summer", "Taylor Swift"), ("Blinding Lights", "The Weeknd"), ("Bohemian Rhapsody", "Queen"), ("So What", "Miles Davis"),
          ("Redbone", "Childish Gambino"), ("The Less I Know the Better", "Tame Impala"), ("Riptide", "Vance Joy"), ("Hey Ya!", "OutKast"),
          ("Clair de Lune", "Debussy"), ("Chameleon", "Herbie Hancock"), ("Down Under", "Men at Work"), ("Heat Waves", "Glass Animals"),
          ("Kiss from a Rose", "Seal"), ("Levitating", "Dua Lipa"), ("Spirit of the Anzacs", "Lee Kernaghan"), ("Big Jet Plane", "Angus & Julia Stone")]
RADIO = ["Triple J", "ABC Radio National", "ABC Classic", "Double J", "Smooth FM", "BBC World Service"]
ARTISTS = [a for _, a in TRACKS] + ["Fleetwood Mac", "Billie Eilish", "Nick Cave", "Missy Higgins", "Daft Punk", "Norah Jones", "The Beatles", "Powderfinger", "Hilltop Hoods"]
GENRES = ["jazz", "classical", "some chill music", "80s music", "country", "hip hop", "lo-fi", "dinner music", "something upbeat", "rock", "blues", "Australian music"]
PLAYLISTS = ["Dinner Party", "Chill Mix", "Discover Weekly", "Road Trip", "Sunday Morning", "Focus", "Favourites", "Peter's Mix", "Amber's Mix", "Workout"]
PODCASTS = ["Conversations", "The Daily", "Hard Fork", "No Such Thing as a Fish", "The Rest Is History", "Coffee Break Spanish"]
ALBUMS = [("Rumours", "Fleetwood Mac"), ("Abbey Road", "The Beatles"), ("Currents", "Tame Impala"), ("Kind of Blue", "Miles Davis"), ("1989", "Taylor Swift"), ("Discovery", "Daft Punk")]

APPLIANCE_POWER = {"Dishwasher": 1200, "Oven": 2400, "Fridge": 120, "Freezer": 100, "Washing Machine": 500, "Dryer": 2000, "Air Fryer": 1500}
APPLIANCE_PROGRAMS = {"Dishwasher": ["eco", "quick", "intensive"], "Washing Machine": ["cottons", "quick wash", "delicates"], "Dryer": ["cupboard dry", "iron dry"], "Oven": ["180 degrees fan-forced", "grill", "220 degrees"]}


def make_house(seed: int, profile: str | None = None) -> House:
    return Gen(seed, HOUSE_PROFILES.get(profile) if profile else None).make()


def regen(house: House) -> Gen:
    """A generator matching the house (for re-deriving sensor states after mutations)."""
    return Gen(house.seed, HOUSE_PROFILES.get(house.profile) if house.profile else None)


def pick_speaker(house: House, rng: random.Random, channel: str | None = None) -> Speaker:
    """Choose who is talking. Satellite speakers are placed in their ground-truth room
    (the satellite location is known, so we make the two agree)."""
    home = [p for p in house.people if p.home]
    if channel is None:
        channel = "satellite" if home and rng.random() < 0.85 else "app"
    if channel == "satellite":
        indoor = [a.area_id for a in house.areas if a.kind != "outdoor"]
        if home:
            p = rng.choice(home)
            if p.room is None:
                p.room = rng.choice(indoor)
                regen(house).place_people(house)
            name = p.name if rng.random() < 0.9 else None
            return Speaker(name=name, area=p.room, channel="satellite")
        return Speaker(name=None, area=rng.choice(indoor), channel="satellite")
    p = rng.choice(house.people)
    return Speaker(name=p.name, area=None, channel="app")
