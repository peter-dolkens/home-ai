"""Large, heavily instrumented houses (thousands to tens of thousands of HA entities) for comparing
prompt / tool architectures at the scale of the new build. Built on the same schema, simulator and
state logic as the small houses: BigGen subclasses generator.Gen, so presence, lights, media,
climate, weather and the energy group follow exactly the same rules, and every existing scenario
builder runs on a big house unchanged.

    make_big_house(seed, n_entities_target)     random big house, within a few % of the target
    make_bundook_full(seed) / (seed, projected=True)   the new build from its plan (homeai/house/bundook_full.py)
    house_for(ex)                               rebuild the house of any dataset example (small or big)

    PYTHONPATH=. python3 -m homeai.house.bighouse --report     prompt size vs entity count

Devices come from homeai/house/devices.py: every device exposes its real integration's entity set,
config / diagnostic entities included (marked with attributes["entity_category"])."""
from __future__ import annotations

import argparse
import copy
import functools

from .devices import DeviceMixin, finish_devices, pick_names, slug
from .generator import Gen, make_house
from .schema import Area, House

# (area_id, name, kind). CORE rooms exist in every big house; EXTRA rooms are added in order until
# the house has as many areas as its entity target calls for.
CORE_ROOMS = [
    ("entry", "Entry", "entry"), ("hallway", "Hallway", "entry"), ("kitchen", "Kitchen", "kitchen"), ("pantry", "Pantry", "kitchen"),
    ("living_room", "Living Room", "living"), ("dining_room", "Dining Room", "living"), ("family_room", "Family Room", "living"),
    ("master_bedroom", "Master Bedroom", "bedroom"), ("master_ensuite", "Master Ensuite", "bath"), ("master_robe", "Master Robe", "utility"),
    ("bathroom", "Bathroom", "bath"), ("powder_room", "Powder Room", "bath"), ("laundry", "Laundry", "utility"), ("mudroom", "Mudroom", "entry"),
    ("office", "Office", "office"), ("bedroom_2", "Bedroom 2", "bedroom"), ("garage", "Garage", "garage"), ("plant_room", "Plant Room", "utility"),
    ("comms_room", "Comms Room", "utility"), ("battery_shed", "Battery Shed", "utility"), ("front_deck", "Front Deck", "outdoor"),
    ("back_deck", "Back Deck", "outdoor"), ("driveway", "Driveway", "outdoor"), ("vegie_garden", "Vegie Garden", "outdoor"),
    ("machinery_shed", "Machinery Shed", "utility"),
]
EXTRA_ROOMS = [
    ("scullery", "Scullery", "kitchen"), ("bedroom_3", "Bedroom 3", "bedroom"), ("guest_room", "Guest Room", "bedroom"),
    ("guest_ensuite", "Guest Ensuite", "bath"), ("media_room", "Media Room", "living"), ("alfresco", "Alfresco", "outdoor"),
    ("workshop", "Workshop", "utility"), ("study", "Study", "office"), ("toilet", "Toilet", "bath"), ("rumpus_room", "Rumpus Room", "living"),
    ("bedroom_4", "Bedroom 4", "bedroom"), ("ensuite_2", "Ensuite 2", "bath"), ("gym", "Gym", "living"), ("carport", "Carport", "outdoor"),
    ("upstairs_hallway", "Upstairs Hallway", "entry"), ("stairs", "Stairs", "entry"), ("library", "Library", "living"),
    ("wine_cellar", "Wine Cellar", "utility"), ("pool", "Pool", "outdoor"), ("orchard", "Orchard", "outdoor"), ("front_garden", "Front Garden", "outdoor"),
    ("back_garden", "Back Garden", "outdoor"), ("games_room", "Games Room", "living"), ("sunroom", "Sunroom", "living"),
    ("bedroom_5", "Bedroom 5", "bedroom"), ("kids_bathroom", "Kids Bathroom", "bath"), ("chicken_coop", "Chicken Coop", "outdoor"),
    ("pump_shed", "Pump Shed", "utility"), ("studio", "Studio", "office"), ("guest_cottage", "Guest Cottage", "bedroom"),
    ("cottage_bathroom", "Cottage Bathroom", "bath"), ("cottage_kitchenette", "Cottage Kitchenette", "kitchen"), ("side_yard", "Side Yard", "outdoor"),
    ("playroom", "Playroom", "living"), ("loft", "Loft", "utility"), ("bedroom_6", "Bedroom 6", "bedroom"), ("nursery", "Nursery", "bedroom"),
    ("boot_room", "Boot Room", "entry"), ("walk_in_pantry", "Walk-in Pantry", "kitchen"), ("hay_shed", "Hay Shed", "utility"),
    ("dam", "Dam", "outdoor"), ("fire_pit", "Fire Pit", "outdoor"), ("greenhouse", "Greenhouse", "outdoor"), ("stables", "Stables", "utility"),
    ("upstairs_bathroom", "Upstairs Bathroom", "bath"), ("reading_nook", "Reading Nook", "living"), ("music_room", "Music Room", "living"),
    ("cellar", "Cellar", "utility"), ("courtyard", "Courtyard", "outdoor"), ("north_paddock", "North Paddock", "outdoor"),
    ("south_paddock", "South Paddock", "outdoor"),
]

# light fittings per room kind: (label, probability, how: dim | relay | rgbww | hue:<n>)
LIGHTS = {
    "bedroom": [("Downlights", 1.0, "dim"), ("Wall Lights", 0.5, "hue:2"), ("Bedside Lamps", 0.6, "hue:2"), ("Strip Light", 0.2, "rgbww")],
    "living": [("Downlights", 1.0, "dim"), ("Pendants", 0.6, "hue:3"), ("Floor Lamp", 0.5, "hue:1"), ("Cove Strip", 0.4, "rgbww"),
               ("Track Lights", 0.3, "relay"), ("Wall Washers", 0.2, "dim")],
    "kitchen": [("Downlights", 1.0, "dim"), ("Island Pendants", 0.6, "hue:3"), ("Bench Strip", 0.6, "rgbww"), ("Toe Kick Strip", 0.4, "rgbww")],
    "office": [("Downlights", 1.0, "dim"), ("Desk Lamp", 0.7, "hue:1"), ("Track Lights", 0.4, "relay")],
    "bath": [("Downlights", 1.0, "dim"), ("Mirror Light", 0.6, "relay"), ("Night Light", 0.3, "rgbww")],
    "entry": [("Downlights", 1.0, "dim"), ("Pendants", 0.4, "hue:2"), ("Floor Strip", 0.4, "rgbww")],
    "utility": [("Downlights", 1.0, "relay")],
    "garage": [("Battens", 1.0, "relay")],
    "outdoor": [("Floodlights", 0.8, "hue:2"), ("Downlights", 0.5, "relay"), ("Garden Spots", 0.4, "hue:3"), ("Path Lights", 0.3, "relay")],
}
for _k in ("pantry", "scullery", "walk_in_pantry", "cottage_kitchenette"):
    LIGHTS[_k] = [("Downlights", 1.0, "dim"), ("Bench Strip", 0.4, "rgbww")]
WINDOWS = {"bedroom": (1, 3), "living": (2, 4), "kitchen": (1, 3), "office": (1, 2), "bath": (0, 1), "entry": (0, 1), "utility": (0, 1), "garage": (0, 1)}
WINDOW_NAMES = ["North Window", "East Window", "South Window", "West Window", "Bay Window", "Side Window", "Corner Window", "Garden Window"]
PLUGS = {
    "bedroom": ((1, 3), ["Bedside Charger", "Electric Blanket", "Lamp Plug", "Humidifier", "Phone Charger", "Air Purifier"]),
    "living": ((2, 4), ["TV Unit", "Lamp Plug", "Fish Tank", "Record Player", "Christmas Tree", "Game Console", "Air Purifier"]),
    "kitchen": ((2, 4), ["Kettle", "Coffee Machine", "Toaster", "Rice Cooker", "Thermomix", "Bread Maker"]),
    "office": ((2, 4), ["Monitor", "Printer", "Desk Plug", "Speakers", "3D Printer", "Soldering Station", "Laptop Dock"]),
    "utility": ((1, 2), ["Iron", "Chest Freezer", "Beer Fridge", "Battery Charger", "Wine Fridge"]),
    "garage": ((1, 3), ["Compressor", "Battery Charger", "Beer Fridge", "Bench Grinder", "Tool Charger"]),
    "outdoor": ((0, 1), ["Pond Pump", "Festoon Lights", "Heat Lamp", "Fountain"]),
    "bath": ((0, 1), ["Hair Dryer", "Toothbrush Charger"]),
    "entry": ((0, 1), ["Lamp Plug", "Phone Charger"]),
}
PLUG_WATTS = {"Kettle": 2200, "Coffee Machine": 1400, "Toaster": 900, "Rice Cooker": 700, "Thermomix": 1000, "Bread Maker": 600, "Compressor": 1800,
              "Heat Lamp": 1500, "Electric Blanket": 120, "Chest Freezer": 110, "Beer Fridge": 90, "Wine Fridge": 80, "Monitor": 60, "Printer": 30,
              "3D Printer": 250, "Iron": 2000, "Hair Dryer": 1800, "Fish Tank": 60, "Game Console": 180, "TV Unit": 150}
IRRIGATION = ["Vegie Beds", "Orchard", "Front Lawn", "Back Lawn", "Greenhouse", "Fruit Trees", "Native Garden", "Hedge Line", "Pots", "Paddock Trough",
              "Rose Garden", "Herb Garden", "Driveway Beds", "Pool Garden"]
EXT_CAMERAS = ["Driveway Camera", "Front Gate Camera", "North Paddock Camera", "South Paddock Camera", "Shed Camera", "Back Yard Camera",
               "Side Path Camera", "Dam Camera", "Orchard Camera", "Road Camera", "Deck Camera", "Pool Camera", "Chicken Coop Camera",
               "Hay Shed Camera", "East Fence Camera", "West Fence Camera", "Tank Camera", "Garden Camera", "Stables Camera", "Courtyard Camera"]
DOORS = ["Front Door", "Back Door", "Garage Side Door", "Deck Door", "Mudroom Door", "Laundry Door", "Workshop Door", "Sliding Door",
         "Cottage Door", "Shed Door"]


class BigGen(DeviceMixin, Gen):
    """A random big house. Structure (rooms, devices) and state both come from the seed."""

    def __init__(self, seed: int, n_target: int, density: float = 1.0):
        super().__init__(seed, None)
        self.n_target = n_target
        self.d = density  # 0..1: how much optional kit each room and the infrastructure get (small targets thin it out)
        self.board_area, self.board_name = "plant_room", "Switchboard"
        self._devinit()

    def q(self, prob: float) -> bool:
        """An optional device: its probability scales with the house's density."""
        return self.p(prob * self.d)

    def make(self) -> House:
        house = super().make()
        finish_devices(self, house)
        house.description = "a large off-grid country home"
        house.profile = "big"
        return house

    # Gen.make() calls this for houses without a profile
    def _build_random(self, house: House, names: list[str]):
        rng = self.rng
        n_areas = max(25, min(len(CORE_ROOMS) + len(EXTRA_ROOMS), round(22 + self.n_target / 450)))
        rooms = list(CORE_ROOMS)
        kids = [n for n in names[2:]]
        extra = [(f"{slug(k)}_room", f"{k}'s Room", "bedroom") for k in kids] + list(EXTRA_ROOMS)
        head, tail = extra[:12], extra[12:]
        rng.shuffle(tail)
        for r in head + tail:
            if len(rooms) >= n_areas:
                break
            rooms.append(r)
        for aid, name, kind in rooms:
            house.areas.append(Area(aid, name, kind))
        self.energy_params = dict(peak_kw=rng.choice([15.0, 20.0, 25.0, 30.0]), cap_kwh=rng.choice([40.0, 51.7, 60.0, 80.0]))
        self._infra(house, names)
        for a in house.areas:
            self._room(house, a)
        self._fill(house)

    # ------------------------------------------------------------------ rooms
    def _room(self, house: House, area: Area):
        rng, p = self.rng, self.p
        aid, an, kind = area.area_id, area.name, area.kind
        indoor = kind != "outdoor"
        circ = f"{aid}_lights"
        for label, prob, how in LIGHTS.get(aid, LIGHTS[kind]):
            if not (p(prob) if prob >= 1 else self.q(prob)):
                continue
            if self.d < 0.25:  # a lightly automated house: smart bulbs, no relays at the board
                how = "hue:1" if how in ("dim", "relay", "rgbww") else how
            if how.startswith("hue:"):
                self._hue(house, aid, an, label, int(how[4:]))
            else:
                model = {"dim": "shelly_pro_dimmer", "relay": "shelly_pro_4pm", "rgbww": "shelly_pro_rgbww"}[how]
                self._channel(house, model, aid, f"{an} {label}", main_domain="light", main_cls="light", main_id=f"light.{aid}_{slug(label)}",
                              load_w=rng.choice([30, 45, 60, 80, 120]), circuit=circ)
        # presence
        if kind == "outdoor":
            if self.q(0.8):
                self._device(house, "hue_motion_outdoor", aid, f"{an} Motion")
        elif kind in ("bedroom", "living", "office", "kitchen"):
            if p(0.5 + self.d):
                model = "sensy_one" if self.q(0.06) else "apollo_r_pro_1" if self.q(0.75) else "hue_motion"
                self._device(house, model, aid, f"{an} {'Motion' if model == 'hue_motion' else 'mmWave'}")
            if kind == "living" and self.q(0.3):
                self._device(house, "hue_motion", aid, f"{an} Motion {self._next(area, 'Motion')}")
        elif p(0.3 + 0.55 * self.d):
            self._device(house, "hue_motion", aid, f"{an} Motion")
        # windows, blinds, awning windows
        lo, hi = WINDOWS.get(kind, (0, 0))
        wins = pick_names(rng, WINDOW_NAMES, round(rng.randint(lo, hi) * self.d), set())
        for wn in wins:
            self._device(house, "aqara_contact", aid, f"{an} {wn}", main_domain="binary_sensor", main_cls="window")
        if wins and kind in ("bedroom", "living", "office", "kitchen"):
            for wn in wins:
                if self.q(0.6):
                    bn = wn.replace("Window", "Blind")
                    self._device(house, "somfy_poe" if p(0.6) else "aqara_blind", aid, f"{an} {bn}", main_cls="cover")
            if self.q(0.2):
                self._channel(house, "shelly_dual_cover", aid, f"{an} Windows", main_domain="cover", main_cls="cover", main_id=f"cover.{aid}_windows",
                              load_w=40, circuit=f"{aid}_power")
        # climate, fans
        if kind in ("bedroom", "living", "office") and self.q(0.7) or kind == "kitchen" and self.q(0.3):
            self._device(house, "intesis_ac", aid, f"{an} Air Conditioning", main_id=f"climate.{aid}", load_w=1400)
        if kind in ("bedroom", "living") and self.q(0.4) or kind == "office" and self.q(0.2):
            self._device(house, "baf_haiku", aid, f"{an} Fan", main_id=f"fan.{aid}_fan", load_w=30)
        if (kind == "bath" or aid in ("laundry", "kitchen")) and self.q(1.0):
            self._channel(house, "shelly_pro_4pm", aid, f"{an} Exhaust Fan", main_domain="fan", main_cls="fan", main_id=f"fan.{aid}_exhaust_fan",
                          load_w=25, circuit=f"{aid}_power")
        # media
        media_p = {"bedroom": 0.6, "living": 0.5, "kitchen": 0.7, "office": 0.6, "bath": 0.2, "outdoor": 0.3 if "deck" in aid or aid == "alfresco" else 0}.get(kind, 0)
        if kind == "living" and p(0.8) or kind == "bedroom" and self.q(0.25):
            self._device(house, "lg_tv" if p(0.6) else "apple_tv", aid, f"{an} TV", main_id=f"media_player.{aid}_tv", load_w=150)
        if p(media_p * (0.5 + 0.5 * self.d)):
            ma = {"ma": True} if (self.house_has_ma and p(0.8)) else {}
            self._device(house, "homepod", aid, f"{an} Speaker", main_id=f"media_player.{aid}_speaker", main_attrs=ma, load_w=8)
        # plugs, heaters, towel rails, dehumidifiers
        (plo, phi), pool = PLUGS.get(kind, ((0, 0), []))
        taken: set[str] = set()
        for label in pick_names(rng, pool, round(rng.randint(plo, phi) * self.d), taken) if pool else []:
            dev = self._device(house, "meross_plug", aid, f"{an} {label}", main_cls="switch", main_id=f"switch.{aid}_{slug(label)}",
                               load_w=PLUG_WATTS.get(label.rsplit(" ", 1)[0] if label[-1].isdigit() else label, 40), circuit=f"{aid}_power")
            dev["main"].state = "on" if p(0.45) else "off"
        if kind in ("bedroom", "office", "bath") and p(0.35):
            self._device(house, "meross_plug", aid, f"{an} Heater", main_cls="switch", main_id=f"switch.{aid}_heater", circuit=f"{aid}_power",
                         main_attrs={"heater": True, "power_w": rng.choice([1200, 1500, 1800, 2000, 2400])})
        if kind == "bath" and self.q(0.6):
            self._channel(house, "shelly_pro_4pm", aid, f"{an} Towel Rail", main_domain="switch", main_cls="switch", main_id=f"switch.{aid}_towel_rail",
                          main_attrs={"power_w": 200}, circuit=f"{aid}_power")["main"].state = "on" if p(0.3) else "off"
        if kind in ("utility", "bath", "bedroom") and indoor and p(0.12):
            self._device(house, "meross_plug", aid, f"{an} Dehumidifier", main_cls="switch", main_id=f"switch.{aid}_dehumidifier", circuit=f"{aid}_power",
                         main_attrs={"power_w": 300, "appliance": True})["main"].state = "on" if p(0.3) else "off"
        # environment and safety
        if kind in ("bedroom", "office", "living") and self.q(0.5):
            self._device(house, "aranet4", aid, f"{an} Air Quality")
        if kind == "kitchen" and self.q(0.5):
            self._device(house, "develco_air", aid, f"{an} Air Quality")
        if kind in ("bath", "utility", "entry", "garage") and self.q(0.6):
            self._device(house, "aqara_th", aid, f"{an} Climate Sensor")
        if (kind in ("bedroom", "kitchen", "garage") or aid in ("hallway", "living_room", "upstairs_hallway")) and self.q(1.0):
            self._device(house, "clipsal_smoke", aid, f"{an} Smoke Alarm")
        if kind in ("kitchen", "bath") or aid in ("laundry", "plant_room", "pump_shed"):
            if self.q(0.7):
                self._device(house, "aqara_leak", aid, f"{an} Leak Sensor")
        if indoor and self.q(0.5):
            self._device(house, "espresense", aid, f"{an} BLE Node")
        if indoor and self.q(0.6):
            self._device(house, "zigbee_switch", aid, f"{an} Scene Switch")
        if aid in ("master_bedroom", "guest_room") and self.q(0.5):
            self._device(house, "bed_presence", aid, f"{an} Bed Sensor")
        if aid in ("garage", "comms_room", "entry") and self.q(0.5):
            self._device(house, "g6_turret", aid, f"{an} Camera", main_id=f"camera.{aid}")
        if indoor and self.q(0.25):
            self._device(house, "unifi_ap", aid, f"{an} Access Point")
        # appliances
        if aid == "kitchen":
            for kd, prob in (("Dishwasher", 1.0), ("Oven", 1.0), ("Fridge", 0.9), ("Freezer", 0.5), ("Microwave", 0.5), ("Rangehood", 0.5)):
                if p(prob):
                    self._appliance(house, aid, an, kd)
                    main = house.entities[f"switch.{aid}_{slug(kd)}"]
                    if self.q(1.0):
                        self._attach(house, main, "home_connect", load_w=main.attributes.get("power_w", 500), circuit=f"{aid}_{slug(kd)}")
                    else:
                        self._track(main, main.attributes.get("power_w", 500), f"{aid}_{slug(kd)}")
        if aid == "scullery" and p(0.6):
            self._appliance(house, aid, an, "Dishwasher")
            self._attach(house, house.entities[f"switch.{aid}_dishwasher"], "home_connect", load_w=1200, circuit=f"{aid}_power")
        if aid == "laundry":
            for kd in ("Washing Machine", "Dryer"):
                self._appliance(house, aid, an, kd)
                main = house.entities[f"switch.{aid}_{slug(kd)}"]
                if self.q(1.0):
                    self._attach(house, main, "lg_washer", load_w=main.attributes.get("power_w", 500), circuit=f"{aid}_power")
                else:
                    self._track(main, main.attributes.get("power_w", 500), f"{aid}_power")
        if aid == "garage" and self.d < 0.3:
            closed = p(0.8)
            self._add(house, "cover", "cover.garage_door", "Garage Door", "garage", aid, state="closed" if closed else "open",
                      attributes={"position": 0 if closed else 100})
        elif aid == "garage":
            self._device(house, "ratgdo", aid, "Garage Door", main_id="cover.garage_door", load_w=300)
            if p(0.4):
                self._device(house, "ratgdo", aid, "Second Garage Door", main_id="cover.second_garage_door", load_w=300)
        if aid == "living_room" and p(0.6):
            if self.d < 0.4:
                self._add(house, "vacuum", "vacuum.robot", "Robot Vacuum", "vacuum", aid, state="docked", changed_s=rng.randint(3600, 36000))
            else:
                self._device(house, "dreame_vacuum", aid, "Robot Vacuum", main_id="vacuum.robot", load_w=60)
        # circuits: every room's sockets and lights are metered at the board
        if self.q(1.0):
            self._channel(house, "shelly_pro_3em", aid, f"{an} GPOs")["meters"] = f"{aid}_power"
        if self.q(1.0) and any(e.cls == "light" for e in house.in_area(aid)):
            self._channel(house, "shelly_pro_3em", aid, f"{an} Lighting Circuit")["meters"] = circ
        for e in list(house.in_area(aid, "appliance")):
            if e.attributes.get("appliance") in ("oven", "dishwasher", "dryer") and self.q(1.0):
                self._channel(house, "shelly_pro_3em", aid, f"{e.name} Circuit")["meters"] = f"{aid}_{slug(e.attributes['appliance'])}"

    # ------------------------------------------------------------------ house-wide
    def _infra(self, house: House, names: list[str]):
        rng, p = self.rng, self.p
        n = self.n_target
        # security: locks and contacts on the outside doors, UniFi Protect cameras (+ Frigate), intercoms, alarm
        doors = DOORS[:2 + round(self.d * rng.randint(3, len(DOORS) - 2))]
        for d in doors:
            if d in ("Front Door", "Back Door", "Mudroom Door", "Workshop Door", "Cottage Door") or self.q(0.4):
                self._device(house, "unifi_lock", "security", d, main_id=f"lock.{slug(d)}")
            if p(0.4 + 0.6 * self.d):
                self._device(house, "aqara_contact", "security", d, main_domain="binary_sensor", main_cls="door", main_id=f"binary_sensor.{slug(d)}")
        if self.d >= 0.3:
            self._device(house, "udm_se", "security", "Home Alarm", main_id="alarm_control_panel.home_alarm")
        else:
            self._add(house, "alarm_control_panel", "alarm_control_panel.home_alarm", "Home Alarm", "alarm", "security",
                      state=rng.choices(["disarmed", "armed_home", "armed_away"], weights=[0.7, 0.2, 0.1])[0])
        for cam in EXT_CAMERAS[:max(1, min(len(EXT_CAMERAS), round(n / 1000)))]:
            self._device(house, "g6_ptz", "security", cam, main_id=f"camera.{slug(cam)}")
            if self.q(0.8):
                self._device(house, "frigate_camera", "security", cam)
        if self.q(1.0):
            self._device(house, "g6_entry", "security", "Front Door Intercom", main_id="camera.front_door_intercom")
        if self.q(0.6):
            self._device(house, "g6_entry", "security", "Gate Intercom", main_id="camera.gate_intercom")
        # energy: big loads on relay channels, plant in the battery shed
        loads = [("Hot Water Heat Pump", 1000, "hot_water", 1.0), ("Hydronic Heating", 3000, "heating", 0.6), ("Pool Pump", 800, "pump", 0.4),
                 ("EV Charger", 7000, "charger", 0.5), ("Pressure Pump", 900, "pump", 0.3)]
        for name, watts, kind, prob in loads:
            if not p(prob):
                continue
            on = p(0.5 if (kind == "heating" and self.season_cold) else 0.03 if kind == "heating" else 0.3 if kind == "hot_water" else 0.15)
            dev = self._channel(house, "shelly_pro_4pm", "energy", name, main_domain="switch", main_cls="switch", main_id=f"switch.{slug(name)}",
                                main_attrs={"power_w": watts, "appliance": True}, unit_area="plant_room")
            dev["main"].state = "on" if on else "off"
            dev["main"].changed_s = rng.randint(300, 4 * 3600)
        if p(0.5):
            self._add(house, "switch", "switch.generator", "Generator", "switch", "energy", state="off", changed_s=rng.randint(3600, 3 * 86400))
        if self.d >= 0.2:
            self._device(house, "selectronic", "battery_shed", "Battery Inverter")
        if self.d >= 0.3:
            for i in range(max(1, round(self.energy_params["peak_kw"] / 10))):
                self._device(house, "fronius_gen24", "battery_shed", f"Solar Inverter {i + 1}")
        if self.d >= 0.6:
            for i in range(round(self.energy_params["cap_kwh"] / 4.7)):
                self._device(house, "battery_module", "battery_shed", f"Battery Module {i + 1}")
        # water: tanks, pump, irrigation valves
        zones = rng.sample(IRRIGATION, min(len(IRRIGATION), rng.randint(4, 6) + round(n / 4000)))
        self._water(house, zones)
        if self.d >= 0.5:
            self._attach(house, house.entities["switch.house_pump"], "dab_esybox", area="water")
        if self.d >= 0.4:
            for z in zones:
                self._attach(house, house.entities[f"switch.irrigation_{slug(z)}"], "ecowitt_valve", area="water")
        for t in ["Garden Tank", "Fire Tank", "Stock Tank"][:round(rng.randint(1, 3) * self.d)]:
            cap = rng.choice([22000, 45000, 90000])
            dev = self._device(house, "tank_sensor", "water", t, main_attrs={"kind": "level", "capacity_l": cap})
            dev["main"].attributes["litres"] = cap * float(dev["main"].state) / 100
        # weather station
        if self.d >= 0.4:
            self._device(house, "ecowitt_station", "weather", "Weather Station")
            self._add(house, "sensor", "sensor.weather_station_outdoor_temperature", "Weather Station Outdoor Temperature", "temperature", "weather",
                      state="15", attributes={"device": "weather_station"}, changed_s=rng.randint(30, 600))
        # network and rack
        for model, nm in (("unvr", "NVR"), ("synology", "NAS"), ("apc_ups", "Rack UPS"), ("apc_ups", "Network UPS"), ("apc_pdu", "Rack PDU A"),
                          ("apc_pdu", "Rack PDU B"), ("starlink", "Starlink")):
            if self.q(1.0):
                self._device(house, model, "comms_room", nm)
        # people's phones; vehicle; mower
        for nm in names:
            if self.q(1.0):
                self._device(house, "phone", "_people", f"{nm}'s iPhone")
        if p(0.6):
            self._vehicle(house, rng.choice(["EV", "Ute", "Car"]), "sensor.car", backup=False)
            if self.d >= 0.6:
                self._attach(house, house.entities["sensor.car"], "byd_shark", area="garage")
        if self.q(0.4):
            self._device(house, "luba_mower", "back_deck", "Lawn Mower")

    def _fill(self, house: House):
        """Keep adding plausible devices until the house reaches its entity target."""
        rng = self.rng
        reserve = 16 + 2 * len(house.people)  # weather, sun, energy sensors and people Gen.make() adds afterwards
        indoor = [a for a in house.areas if a.kind != "outdoor"]
        living = [a for a in house.areas if a.kind in ("living", "bedroom", "office", "kitchen")]
        outdoor = [a for a in house.areas if a.kind == "outdoor"] or indoor
        counter = [0]

        def nm(a, label):
            counter[0] += 1
            return f"{a.name} {label} {counter[0]}"

        def lamp():
            a = rng.choice(living)
            self._hue(house, a.area_id, a.name, f"Lamp {self._next(a, 'Lamp')}", 1)

        def plug():
            a = rng.choice(indoor)
            label = f"Plug {self._next(a, 'Plug')}"
            self._device(house, "meross_plug", a.area_id, f"{a.name} {label}", main_cls="switch", main_id=f"switch.{a.area_id}_{slug(label)}",
                         load_w=rng.choice([10, 40, 80, 150]), circuit=f"{a.area_id}_power")["main"].state = "on" if rng.random() < 0.45 else "off"

        def window():
            a = rng.choice(living)
            wn = f"Window {self._next(a, 'Window')}"
            self._device(house, "aqara_contact", a.area_id, f"{a.name} {wn}", main_domain="binary_sensor", main_cls="window")
            if rng.random() < 0.6:
                self._device(house, "somfy_poe", a.area_id, f"{a.name} {wn.replace('Window', 'Blind')}", main_cls="cover")

        def presence():
            a = rng.choice(living)
            self._device(house, "sensy_one" if rng.random() < 0.3 else "apollo_r_pro_1", a.area_id, f"{a.name} mmWave {self._next(a, 'mmWave')}")

        def camera():
            k = self._next(None, "Camera")
            cam = f"Perimeter Camera {k}"
            self._device(house, "g6_ptz", "security", cam, main_id=f"camera.{slug(cam)}")
            self._device(house, "frigate_camera", "security", cam)

        def sensor():
            a = rng.choice(indoor)
            model = rng.choice(["aqara_th", "aqara_leak", "aranet4", "espresense", "zigbee_switch", "aqara_th"])
            self._device(house, model, a.area_id, nm(a, {"aqara_th": "Climate Sensor", "aqara_leak": "Leak Sensor", "aranet4": "Air Quality",
                                                         "espresense": "BLE Node", "zigbee_switch": "Scene Switch"}[model]))

        def circuit():
            a = rng.choice(indoor)
            self._channel(house, "shelly_pro_3em", a.area_id, nm(a, "Circuit"))["meters"] = f"{a.area_id}_power"

        def garden():
            a = rng.choice(outdoor)
            self._hue(house, a.area_id, a.name, f"Spot {self._next(a, 'Spot')}", 1)

        def valve():
            k = self._next(None, "Zone")
            z = f"Garden Zone {k}"
            dev = self._device(house, "ecowitt_valve", "water", f"{z} Irrigation", main_id=f"switch.irrigation_{slug(z)}",
                               main_attrs={"default_min": rng.choice([15, 20, 30, 45])})
            dev["main"].changed_s = rng.randint(1800, 3 * 86400)

        def ap():
            a = rng.choice(indoor + outdoor)
            self._device(house, "unifi_ap", a.area_id, nm(a, "Access Point"))

        pool = [(lamp, 5), (plug, 5), (window, 4), (presence, 0.6), (camera, 0.12), (sensor, 5), (circuit, 1.5), (garden, 1.5), (valve, 0.3), (ap, 0.5)]
        fns, ws = [f for f, _ in pool], [w for _, w in pool]
        while len(house.entities) < self.n_target - reserve:
            rng.choices(fns, weights=ws)[0]()
        # PoE ports on the UniFi switches for everything that is cabled back to the rack
        poe = [d for d in self.devices if d["model"] in ("g6_ptz", "g6_turret", "g6_entry", "apollo_r_pro_1", "sensy_one", "unifi_ap", "somfy_poe")]
        for d in poe:
            if len(house.entities) >= self.n_target + 0.03 * self.n_target - reserve:
                break
            self._channel(house, "unifi_switch", "comms_room", f"{d['name']} Port", unit_area="comms_room")

    def _next(self, area, what: str) -> int:
        c = self.__dict__.setdefault("_counters", {})
        key = (area.area_id if area else "", what)
        c[key] = c.get(key, 0) + 1
        return c[key]


# ---------------------------------------------------------------------- building and caching
@functools.lru_cache(maxsize=8)
def _pristine(kind: str, seed: int, n: int) -> House:
    if kind == "random":
        return _search(seed, n)
    from .bundook_full import make_bundook_full
    return make_bundook_full(seed, projected=(kind == "bundook_projected"))


def _search(seed: int, n: int) -> House:
    """Build at the density the target suggests; if the rooms and infrastructure alone overshoot, thin them out and
    rebuild (deterministic: same seed, same sequence of densities). Undershoot is topped up by BigGen._fill."""
    d = min(1.0, n / 6000)
    best = None
    for _ in range(8):
        h = BigGen(seed, n, d).make()
        if best is None or abs(len(h.entities) - n) < abs(len(best.entities) - n):
            best = h
        if len(h.entities) <= n * 1.06:
            return h
        d *= 0.7 * n / len(h.entities)
    return best


def make_big_house(seed: int, n_entities_target: int = 2000) -> House:
    """A random big house with about n_entities_target entities (config / diagnostic included)."""
    return copy.deepcopy(_pristine("random", seed, n_entities_target))


def big_spec_house(big: dict) -> House:
    """{"kind": "random", "seed": s, "n": n} or {"kind": "bundook_full" | "bundook_projected", "seed": s}, plus an optional
    "patch" (entity and people changes the scenario made before the transcript starts)."""
    house = copy.deepcopy(_pristine(big.get("kind", "random"), int(big["seed"]), int(big.get("n") or 0)))
    patch = big.get("patch") or {}
    for eid, (state, attrs, changed) in (patch.get("entities") or {}).items():
        e = house.entities[eid]
        e.state, e.attributes, e.changed_s = state, dict(attrs), changed
    for pp in patch.get("people") or []:
        p = house.person(pp["name"])
        p.home, p.room, p.changed_s = pp["home"], pp["room"], pp["changed_s"]
    return house


def house_for(ex: dict) -> House:
    """The house a dataset example was built on: big houses from their "big" spec, the rest from make_house."""
    if ex.get("big"):
        return big_spec_house(ex["big"])
    return make_house(ex["house_seed"], ex.get("profile"))


def state_patch(base: House, house: House) -> dict:
    """Entity and people differences between a freshly built house and the same house after a scenario set it up."""
    ents = {}
    for eid, e in house.entities.items():
        b = base.entities[eid]
        if (e.state, e.attributes, e.changed_s) != (b.state, b.attributes, b.changed_s):
            ents[eid] = [e.state, copy.deepcopy(e.attributes), e.changed_s]
    people = [{"name": p.name, "home": p.home, "room": p.room, "changed_s": p.changed_s}
              for p, q in zip(house.people, base.people) if (p.home, p.room, p.changed_s) != (q.home, q.room, q.changed_s)]
    return {"entities": ents, "people": people}


# ---------------------------------------------------------------------- prompt-size report
def primary_only(house: House) -> House:
    h = copy.copy(house)
    h.entities = {k: e for k, e in house.entities.items() if not e.attributes.get("entity_category")}
    return h


def report(targets=(62, 500, 1000, 2000, 3000, 5000, 10000, 20000), seeds=(9_100_001, 9_100_002)):
    from ..tools import get_state_text
    from .snapshot import system_prompt
    print(f"{'target':>7} {'entities':>8} {'primary':>8} {'areas':>5} {'named':>6} {'prompt chars':>12} {'~tokens':>8} "
          f"{'prompt(prim)':>12} {'full dump chars':>15} {'dump(prim)':>10}")
    for t in list(targets) + ["bundook_full", "bundook_projected"]:
        rows = []
        for s in seeds:
            h = make_house(s) if t == 62 else _pristine(t, s, 0) if isinstance(t, str) else make_big_house(s, t)
            prim = primary_only(h)
            named = sum(1 for e in h.entities.values() if e.cls in ("light", "switch", "fan", "cover", "garage", "lock", "media_player", "climate",
                                                                    "vacuum", "alarm", "appliance", "irrigation", "pump"))
            sp_all, sp_prim = len(system_prompt(h, None)), len(system_prompt(prim, None))
            dump_all = len(get_state_text(h, [], ["device"] + sorted({e.cls for e in h.entities.values()})))  # everything, categories included
            dump_prim = len(get_state_text(h))
            rows.append((len(h.entities), len(prim.entities), len(h.areas), named, sp_all, sp_prim, dump_all, dump_prim))
        m = [sum(r[i] for r in rows) / len(rows) for i in range(8)]
        label = "small" if t == 62 else "b_full" if t == "bundook_full" else "b_proj" if t == "bundook_projected" else str(t)
        print(f"{label:>7} {m[0]:8.0f} {m[1]:8.0f} {m[2]:5.0f} {m[3]:6.0f} {m[4]:12.0f} {m[4] / 3.6:8.0f} {m[5]:12.0f} {m[6]:15.0f} {m[7]:10.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed", type=int, default=9_100_001)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    if a.report:
        return report()
    h = make_big_house(a.seed, a.n)
    print(len(h.entities), "entities,", len(h.areas), "areas")


if __name__ == "__main__":
    main()
