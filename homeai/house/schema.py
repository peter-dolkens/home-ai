"""Core house model shared by the data generator, the mock simulator, the eval
harness and the live Home Assistant bridge.

The important abstraction is the *class* of an entity (light, motion, battery...)
and its *area*, where an area is either a real Home Assistant area id or one of
three house-wide pseudo-areas ("security", "energy", "weather"). The model only
ever sees an inventory of areas and classes in its system prompt and pulls
detailed state on demand with get_state(areas, classes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

# Classes the model may target with call_service. Names are listed in the inventory.
CONTROLLABLE = {
    "light", "switch", "fan", "cover", "garage", "lock",
    "media_player", "climate", "vacuum", "alarm",
    "appliance",   # dishwasher, washing machine, dryer, oven, fridge (status + on/off)
    "irrigation",  # valves / zones
    "pump",        # water pumps
}
# Read-only classes. Only the class name is listed in the inventory.
SENSORY = {
    "motion", "occupancy", "temperature", "humidity", "door", "window", "camera",
    "battery", "solar", "load", "power", "forecast", "weather", "sun", "person",
    "water",       # tank levels, pressure
    "vehicle",     # car: charge, location, locked, plugged in
    # Only produced by the big-house generators (homeai/house/bighouse.py) so far:
    "illuminance",  # lux
    "air_quality",  # CO2 / VOC / PM2.5
    "meter",        # cumulative totals: kWh, litres
    "measurement",  # other numeric readings with a unit: current, voltage, rain, wind, flow...
    "detection",    # camera / NVR object and audio detections, doorbell events
    "safety",       # smoke, heat, CO, leak
    "device",       # the device itself: battery %, signal, firmware, config numbers/selects/switches
}
CLASSES = CONTROLLABLE | SENSORY

# House-wide pseudo-areas. call_service area_id targeting does not work on these;
# entities inside them must be targeted by entity_id.
GROUPS = ("security", "energy", "water", "weather")
GROUP_TITLES = {"security": "Security", "energy": "Energy", "water": "Water", "weather": "Weather"}

# Services that should be confirmed before execution.
SENSITIVE = {
    ("lock", "unlock"),
    ("alarm_control_panel", "alarm_disarm"),
    ("cover", "open_cover"),  # only sensitive for class == garage; checked in sim
}


@dataclass
class Entity:
    entity_id: str
    domain: str
    name: str
    cls: str
    area: str  # area_id or a group name
    state: str
    attributes: dict = field(default_factory=dict)
    changed_s: int = 0  # seconds since last_changed

    @property
    def in_group(self) -> bool:
        return self.area in GROUPS


@dataclass
class Area:
    area_id: str
    name: str
    kind: str  # bedroom | office | kitchen | living | bath | entry | utility | garage | outdoor
    spoken: Optional[str] = None  # how to say it aloud when the HA name is awkward, e.g. "Pete's office" for "Office (Pete)"


@dataclass
class Person:
    name: str
    entity_id: str
    home: bool
    room: Optional[str]  # ground truth, hidden from the model. None = home, no sensor evidence.
    changed_s: int


@dataclass
class Speaker:
    name: Optional[str]  # None = unidentified
    area: Optional[str]  # area_id of the satellite, None when via app
    channel: str  # "satellite" | "app"


@dataclass
class House:
    seed: int
    now: datetime
    areas: list[Area]
    entities: dict[str, Entity]
    people: list[Person]
    description: str = "an off-grid smart home"
    profile: Optional[str] = None  # generator profile name, if any

    # -- lookups ---------------------------------------------------------
    def area(self, area_id: str) -> Optional[Area]:
        for a in self.areas:
            if a.area_id == area_id:
                return a
        return None

    def area_name(self, area_id: str) -> str:
        if area_id in GROUP_TITLES:
            return GROUP_TITLES[area_id]
        a = self.area(area_id)
        return a.name if a else area_id

    def area_spoken(self, area_id: str) -> str:
        """Name to use in speech: the alias if set, else the display name."""
        a = self.area(area_id)
        if a and a.spoken:
            return a.spoken
        return self.area_name(area_id)

    def area_ids(self) -> list[str]:
        return [a.area_id for a in self.areas]

    def all_area_keys(self) -> list[str]:
        """Real areas followed by the groups that have entities."""
        keys = self.area_ids()
        for g in GROUPS:
            if any(e.area == g for e in self.entities.values()):
                keys.append(g)
        return keys

    def in_area(self, area_id: str, cls: Optional[str] = None) -> list[Entity]:
        return [e for e in self.entities.values()
                if e.area == area_id and (cls is None or e.cls == cls)]

    def by_class(self, cls: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.cls == cls]

    def by_domain(self, domain: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.domain == domain]

    def select(self, areas: Iterable[str] = (), classes: Iterable[str] = ()) -> list[Entity]:
        """Semantics of get_state: an entity matches if its area is in `areas`
        (or areas is empty) AND its class is in `classes` (or classes is empty)."""
        areas = set(areas or ())
        classes = set(classes or ())
        out = []
        for e in self.entities.values():
            if areas and e.area not in areas:
                continue
            if classes and e.cls not in classes:
                continue
            out.append(e)
        return out

    def first(self, cls: str, area: Optional[str] = None) -> Optional[Entity]:
        for e in self.entities.values():
            if e.cls == cls and (area is None or e.area == area):
                return e
        return None

    def person(self, name: str) -> Optional[Person]:
        for p in self.people:
            if p.name.lower() == name.lower():
                return p
        return None

    def home_people(self) -> list[Person]:
        return [p for p in self.people if p.home]

    def away_people(self) -> list[Person]:
        return [p for p in self.people if not p.home]

    def occupied_areas(self) -> list[str]:
        """Areas whose motion/occupancy sensor currently reports presence."""
        out = []
        for a in self.area_ids():
            if any(e.state == "on" for e in self.in_area(a) if e.cls in ("motion", "occupancy")):
                out.append(a)
        return out

    def sensed_areas(self) -> list[str]:
        """Areas that have at least one motion/occupancy sensor."""
        return [a for a in self.area_ids()
                if any(e.cls in ("motion", "occupancy") for e in self.in_area(a))]

    def is_dark(self) -> bool:
        sun = self.first("sun")
        if sun:
            return sun.state == "below_horizon"
        return self.now.hour < 7 or self.now.hour >= 18
