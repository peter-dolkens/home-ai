"""Home Assistant REST client and the mapping from HA entities to the House model.

Classification (entity -> class/area) is the one piece that is specific to your
installation. Auto rules cover the common cases; everything else goes in the
bridge config's "overrides" (see configs/bridge.example.json)."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

from ..house.schema import GROUPS, Area, Entity, House, Person

AREA_TEMPLATE = (
    "{% set ns = namespace(out={}) %}{% for a in areas() %}"
    "{% set ns.out = dict(ns.out, **{a: {'name': area_name(a), 'entities': area_entities(a)}}) %}"
    "{% endfor %}{{ ns.out | to_json }}"
)


class HAClient:
    def __init__(self, url: str, token: Optional[str] = None, timeout: float = 15.0,
                 username: Optional[str] = None, password: Optional[str] = None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.token = token or self.login(username, password)

    def login(self, username: Optional[str], password: Optional[str]) -> str:
        """Obtain an access token through HA's login flow (for setups without a long-lived token)."""
        if not (username and password):
            raise ValueError("need a long-lived token or username+password")
        client_id = self.url + "/"

        def post(path, body, form=False):
            data = urllib.parse.urlencode(body).encode() if form else json.dumps(body).encode()
            req = urllib.request.Request(self.url + path, data=data, method="POST",
                                         headers={"Content-Type": "application/x-www-form-urlencoded" if form else "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())

        flow = post("/auth/login_flow", {"client_id": client_id, "handler": ["homeassistant", None], "redirect_uri": client_id})
        res = post(f"/auth/login_flow/{flow['flow_id']}", {"username": username, "password": password, "client_id": client_id})
        if res.get("type") != "create_entry":
            raise RuntimeError(f"login failed: {res.get('errors') or res}")
        tok = post("/auth/token", {"grant_type": "authorization_code", "code": res["result"], "client_id": client_id}, form=True)
        return tok["access_token"]

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = r.read().decode()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def states(self) -> list[dict]:
        return self._req("GET", "/api/states")

    def areas(self) -> dict[str, dict]:
        raw = self._req("POST", "/api/template", {"template": AREA_TEMPLATE})
        return json.loads(raw) if isinstance(raw, str) else raw

    def call_service(self, domain: str, service: str, target: dict, data: Optional[dict] = None) -> Any:
        body = dict(data or {})
        for k in ("entity_id", "area_id"):
            if target.get(k):
                body[k] = target[k]
        return self._req("POST", f"/api/services/{domain}/{service}", body)


# ------------------------------------------------------------------ mapping
KIND_WORDS = [
    ("bedroom", ("bed", "room's", "'s room", "nursery")), ("office", ("office", "study")), ("kitchen", ("kitchen",)),
    ("bath", ("bath", "ensuite", "toilet", "wc")), ("entry", ("hall", "entry", "entrance", "foyer")),
    ("utility", ("laundry", "workshop", "utility", "shed")), ("garage", ("garage", "carport")),
    ("outdoor", ("patio", "deck", "yard", "porch", "outside", "outdoor", "garden", "pool", "driveway")),
]


def area_kind(name: str) -> str:
    n = name.lower()
    for kind, words in KIND_WORDS:
        if any(w in n for w in words):
            return kind
    return "living"


ENERGY_HINTS = [
    (r"(soc|state_of_charge|battery_(level|charge|soc|percent))", "battery", "soc"),
    (r"battery.*(power|watts)|(power|watts).*battery", "battery", "power"),
    (r"battery.*(remaining|energy_left|kwh)", "battery", "remaining"),
    (r"battery.*(runtime|time_remaining|hours)", "battery", "runtime"),
    (r"(solar|pv).*forecast.*(remaining|today)|forecast.*(remaining|today)", "forecast", "remaining_today"),
    (r"(solar|pv).*forecast.*tomorrow|forecast.*tomorrow", "forecast", "tomorrow"),
    (r"(solar|pv).*(today|energy|yield|kwh)", "solar", "energy_today"),
    (r"(solar|pv).*(power|watts)|(power|watts).*(solar|pv)", "solar", "power"),
    (r"(house|home|total|ac|site).*(load|consumption|power)|(load|consumption)", "load", None),
]


def classify(st: dict, override: dict) -> Optional[tuple[str, str | None, dict]]:
    """Return (cls, group_or_None, extra_attrs) or None to skip the entity."""
    if "cls" in override:
        return override["cls"], override.get("area"), {k: v for k, v in override.items() if k not in ("cls", "area", "name")}
    eid = st["entity_id"]
    domain = eid.split(".")[0]
    attrs = st.get("attributes", {})
    dc = attrs.get("device_class")
    low_name = (eid + " " + str(attrs.get("friendly_name", ""))).lower()
    if domain == "light":
        return "light", None, {}
    if domain == "switch":
        if re.search(r"(dishwasher|washing_machine|washer|dryer|tumbletron|\boven\b|fridge|freezer|air_fryer|air fryer)", low_name):
            return "appliance", None, {"appliance": re.search(r"(dishwasher|washing machine|washer|dryer|oven|fridge|freezer|air fryer)", low_name.replace("_", " ")).group(1) if re.search(r"(dishwasher|washing machine|washer|dryer|oven|fridge|freezer|air fryer)", low_name.replace("_", " ")) else "appliance"}
        if re.search(r"(valve|irrigation|sprinkler|zone \d|beds \d|watering)", low_name):
            return "irrigation", "water", {}
        if re.search(r"\bpump\b", low_name):
            return "pump", "water", {}
        return "switch", None, {}
    if domain == "fan" and re.search(r"\bpump\b", low_name):
        return "pump", "water", {}
    if domain == "fan":
        return "fan", None, {}
    if domain == "lock":
        return "lock", "security", {}
    if domain == "cover":
        return ("garage", None, {}) if dc in ("garage", "gate") else ("cover", None, {})
    if domain == "climate":
        return "climate", None, {}
    if domain == "media_player":
        return "media_player", None, {}
    if domain == "camera":
        return "camera", "security", {}
    if domain == "alarm_control_panel":
        return "alarm", "security", {}
    if domain == "vacuum":
        return "vacuum", None, {}
    if domain == "person":
        return "person", "_people", {}
    if domain == "weather":
        return "weather", "weather", {}
    if domain == "sun":
        return "sun", "weather", {}
    if domain == "binary_sensor":
        if dc == "motion":
            return "motion", None, {}
        if dc in ("occupancy", "presence"):
            return "occupancy", None, {}
        if dc in ("door", "opening", "garage_door"):
            return "door", "security", {}
        if dc == "window":
            return "window", None, {}
        return None
    if domain == "sensor":
        if dc == "temperature":
            return "temperature", None, {}
        if dc == "humidity":
            return "humidity", None, {}
        unit0 = str(attrs.get("unit_of_measurement", "")).lower()
        if re.search(r"(tank|water level|water_level|rainwater)", low_name) and unit0 in ("%", "l", "litres", "liters"):
            return "water", "water", {"kind": "level"}
        if re.search(r"(water pressure|water_pressure|mains pressure)", low_name):
            return "water", "water", {"kind": "pressure"}
        if dc == "battery" and re.search(r"(shark|\bcar\b|\bev\b|vehicle|byd|tesla)", low_name) and not re.search(r"(phone|remote|sensor|12v|aux)", low_name):
            return "vehicle", "energy", {}
        low = eid.lower() + " " + str(attrs.get("friendly_name", "")).lower()
        try:
            float(st.get("state"))
        except (TypeError, ValueError):
            return None  # timestamps, enums etc. are never energy readings
        unit = str(attrs.get("unit_of_measurement", "")).lower()
        val = float(st.get("state"))
        if re.search(r"(shark|vehicle|\bev\b|car\b|phone|tablet|remote|sensor battery)", low):
            return None  # vehicles and device batteries are not the house battery
        if dc in ("power", "energy", "battery", None) or "forecast" in low:
            for pat, cls, kind in ENERGY_HINTS:
                if re.search(pat, low):
                    if cls == "battery" and dc == "battery" and kind != "soc":
                        continue
                    if kind == "soc" and not (unit in ("%", "") and 0 <= val <= 100):
                        continue
                    if kind in ("power", None) and cls in ("battery", "solar", "load") and unit not in ("w", "kw", ""):
                        continue
                    if kind in ("energy_today", "remaining_today", "tomorrow") and (unit not in ("kwh", "wh", "") or val > 500):
                        continue  # lifetime totals are not today's yield
                    if dc == "battery" and not re.search(r"(solar|inverter|house|home|bank|bms|victron|powerwall|selectronic|soc)", low):
                        return None
                    return cls, "energy", ({"kind": kind} if kind else {})
            if dc == "power" and re.search(r"(heater|dryer|hot_water|pump|charger|oven|washer|dishwasher)", low):
                return "power", None, {}
        return None
    return None


def _changed_s(st: dict, now: datetime) -> int:
    try:
        t = datetime.fromisoformat(st["last_changed"].replace("Z", "+00:00"))
        return max(0, int((now - t).total_seconds()))
    except Exception:  # noqa: BLE001
        return 0


MA_PLAYERS: set[str] = set()

NOISE_PATTERNS = re.compile(
    r"(led indication|auto update|prerelease|arduino ota|overlay|privacy mode|status light|hdr mode|ssh enabled|"
    r"system sounds|\bdnd\b|config overtemp|detection\b|recordings|snapshots|review (alerts|detections)|"
    r"disable polling|holding engine|flip y|single target|bluetooth|nerd mode|resolution channel)", re.I)


def registry_maps(registry: Optional[dict]) -> tuple[dict[str, str], set[str], dict[str, str]]:
    """From HA registries: entity->area (via device when needed), entities to skip
    (config/diagnostic/hidden/disabled), and area id->name."""
    if not registry:
        return {}, set(), {}
    dev_area = {d["id"]: d.get("area_id") for d in registry.get("devices", [])}
    ent_area, skip = {}, set()
    global MA_PLAYERS
    MA_PLAYERS = {e["entity_id"] for e in registry.get("entities", []) if e.get("platform") == "music_assistant"}
    for e in registry.get("entities", []):
        eid = e["entity_id"]
        if e.get("entity_category") in ("config", "diagnostic") or e.get("hidden_by") or e.get("disabled_by"):
            skip.add(eid)
            continue
        aid = e.get("area_id") or dev_area.get(e.get("device_id"))
        if aid:
            ent_area[eid] = aid
    area_names = {a["area_id"]: a["name"] for a in registry.get("areas", [])}
    return ent_area, skip, area_names


def registry_area_info(registry: Optional[dict]) -> dict[str, dict]:
    """area_id -> {floor_id, aliases} from the area registry."""
    if not registry:
        return {}
    return {a["area_id"]: {"floor_id": a.get("floor_id"), "aliases": a.get("aliases") or []} for a in registry.get("areas", [])}


def build_house(states: list[dict], areas: dict[str, dict], config: dict, now: Optional[datetime] = None,
                registry: Optional[dict] = None) -> House:
    now = now or datetime.now(timezone.utc)
    overrides: dict[str, dict] = config.get("overrides", {})
    exclude = set(config.get("exclude_entities", []))
    include_areas = config.get("areas")  # optional allowlist of area ids
    reg_area, skip, reg_names = registry_maps(registry)
    area_info = registry_area_info(registry)
    include_floors = config.get("floors")  # optional allowlist of HA floor ids, e.g. ["new_build"]
    ent_area: dict[str, str] = dict(reg_area)
    house_areas: list[Area] = []
    area_names = dict(reg_names)
    for aid, info in (areas or {}).items():
        area_names.setdefault(aid, info["name"])
        for e in info.get("entities", []):
            ent_area.setdefault(e, aid)
    excluded_areas = set(config.get("exclude_areas", []))
    for aid, name in area_names.items():
        if include_areas and aid not in include_areas:
            excluded_areas.add(aid)
            continue
        if include_floors and area_info.get(aid, {}).get("floor_id") not in include_floors:
            excluded_areas.add(aid)
            continue
        if aid in excluded_areas:
            continue
        if aid not in GROUPS:  # an HA area called "energy"/"security"/"weather" is treated as that group
            aliases = area_info.get(aid, {}).get("aliases") or []
            house_areas.append(Area(aid, name, area_kind(name), spoken=aliases[0] if aliases else None))
    by_id = {st["entity_id"]: st for st in states}
    house = House(seed=0, now=now.astimezone(), areas=house_areas, entities={}, people=[],
                  description=config.get("house_description", "a smart home"))
    use_heuristics = not registry or config.get("noise_filter", True)

    for st in states:
        eid = st["entity_id"]
        if eid in exclude or eid in skip or st.get("state") in ("unavailable", "unknown"):
            continue
        if eid.split(".")[0] in ("switch", "light", "fan", "camera") and use_heuristics and eid not in overrides \
                and NOISE_PATTERNS.search(str(st.get("attributes", {}).get("friendly_name", ""))):
            continue
        ha_area = ent_area.get(eid)
        if ha_area in excluded_areas:
            continue
        ov = overrides.get(eid, {})
        res = classify(st, ov)
        if not res:
            continue
        cls, group, extra = res
        attrs = st.get("attributes", {})
        if cls in ("lock", "alarm", "person", "weather", "sun") or group == "energy":
            area = ov.get("area") or group
        else:
            area = ov.get("area") or ha_area or {"door": "security", "garage": "security", "camera": "security", "temperature": "weather"}.get(cls)
        if area is None:
            continue  # unassigned room device: not shown until given an area
        if area in excluded_areas:
            continue
        if cls in ("battery", "solar", "load", "forecast"):
            area = "energy"
        name = ov.get("name") or attrs.get("friendly_name") or {"sun": "Sun", "weather": "Home Weather"}.get(cls, eid)
        a: dict[str, Any] = dict(extra)
        state = st["state"]
        if cls == "light" and attrs.get("brightness") is not None:
            a["brightness_pct"] = round(attrs["brightness"] / 255 * 100)
        if cls in ("cover", "garage") and attrs.get("current_position") is not None:
            a["position"] = attrs["current_position"]
        if cls == "fan" and attrs.get("percentage") is not None:
            a["percentage"] = attrs["percentage"]
        if cls == "climate":
            a["temperature"] = attrs.get("temperature")
            a["current_temperature"] = attrs.get("current_temperature")
        if cls == "media_player":
            if eid in MA_PLAYERS or eid in config.get("music_assistant_players", []):
                a["ma"] = True
            a["media_artist"] = attrs.get("media_artist")
            a["media_title"] = attrs.get("media_title")
            if attrs.get("volume_level") is not None:
                a["volume_pct"] = round(attrs["volume_level"] * 100)
        if cls == "switch":
            ps = ov.get("power_sensor")
            if ps and ps in by_id:
                try:
                    a["power_w"] = round(float(by_id[ps]["state"]))
                except ValueError:
                    pass
            elif ov.get("power_w"):
                a["power_w"] = ov["power_w"]
        if cls == "battery" and a.get("kind") == "soc":
            a["capacity_kwh"] = ov.get("capacity_kwh", config.get("battery_capacity_kwh"))
        if cls == "battery" and a.get("kind") == "power":
            try:
                w = float(state)
                a["direction"] = "charging" if w > 25 else "discharging" if w < -25 else "idle"
                if config.get("battery_power_positive_is_discharge"):
                    a["direction"] = "discharging" if w > 25 else "charging" if w < -25 else "idle"
                    state = str(-w)
            except ValueError:
                pass
        if cls == "weather":
            a["temperature"] = attrs.get("temperature")
            a["humidity"] = attrs.get("humidity")
            a["wind_kmh"] = attrs.get("wind_speed")
            fc = (attrs.get("forecast") or [{}])
            if len(fc) > 1 and isinstance(fc[1], dict):
                a["forecast_tomorrow"] = {"condition": fc[1].get("condition"), "low": fc[1].get("templow"), "high": fc[1].get("temperature")}
        if cls == "sun":
            for k in ("next_rising", "next_setting"):
                try:
                    a[k] = datetime.fromisoformat(attrs[k].replace("Z", "+00:00")).astimezone().strftime("%H:%M")
                except Exception:  # noqa: BLE001
                    a[k] = "?"
        if cls == "person":
            house.people.append(Person(name=name, entity_id=eid, home=(state == "home"), room=None, changed_s=_changed_s(st, now)))
        house.entities[eid] = Entity(entity_id=eid, domain=eid.split(".")[0], name=name, cls=cls, area=area,
                                     state=state, attributes=a, changed_s=_changed_s(st, now))
    _dedupe_energy(house, overrides)
    return house


def _dedupe_energy(house: House, overrides: dict):
    """Auto-classification on a real install can match many solar/battery sensors. Keep one
    per (class, kind): overrides win, then the shortest entity id (usually the primary sensor)."""
    groups: dict[tuple, list[Entity]] = {}
    for e in house.entities.values():
        if e.area == "energy" and e.cls in ("battery", "solar", "load", "forecast"):
            groups.setdefault((e.cls, e.attributes.get("kind")), []).append(e)
    for key, ents in groups.items():
        if len(ents) <= 1:
            continue
        def score(e: Entity):
            low = (e.entity_id + " " + e.name).lower()
            s = -len(e.entity_id)
            s += 40 if re.search(r"(total|house|home|site|combined)", low) else 0
            s -= 60 if (e.cls != "forecast" and re.search(r"(forecast|estimat|next_hour|current_hour)", low)) else 0
            s -= 30 if re.search(r"(_1_|_2_| 1 | 2 |string|mppt)", low) else 0
            s -= 50 if (e.cls == "forecast" and re.search(r"hour", low)) else 0
            return s
        keep = [e for e in ents if e.entity_id in overrides] or sorted(ents, key=score, reverse=True)[:1]
        for e in ents:
            if e not in keep:
                del house.entities[e.entity_id]
