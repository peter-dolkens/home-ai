"""Fixed house profiles. A profile pins the areas, devices and people of a real
house while the generator still randomises time, weather, states, and where
people are. Mixing profile houses with fully random ones teaches the model the
real room names and layout without letting it memorise a single state.

BUNDOOK is the new build at 319 Bundook Rd, taken from plan.dolkens.house on
2026-09-18 (layout not final; adjust as the sensor plan changes)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AreaSpec:
    area_id: str
    name: str
    kind: str
    lights: list[str] = field(default_factory=list)
    occupancy: int = 0          # number of mmWave presence sensors (Apollo R PRO-1 etc.)
    motion: int = 0             # PIR motion sensors (Hue)
    temperature: bool = False
    humidity: bool = False
    blinds: bool = False        # Somfy motorised blinds (one cover entity per room)
    windows: bool = False       # window actuators (openable windows as a cover)
    window_sensor: bool = False
    climate: bool = False       # AC head / ducted zone
    fan: str | None = None
    exhaust: bool = False
    media: str | None = None
    camera: str | None = None
    towel_rail: bool = False
    vacuum: bool = False
    door_sensor: str | None = None
    spoken: str | None = None   # alias for speech when the HA name is awkward
    appliances: list[str] = field(default_factory=list)  # Dishwasher, Oven, Fridge, Washing Machine, Dryer, Freezer
    dehumidifier: bool = False


@dataclass
class Block:
    """A recurring schedule block: on `days` (0=Mon..6=Sun), between start and end hour
    (end may be <= start to wrap past midnight), the person is in `room` with probability `prob`."""
    days: tuple[int, ...]
    start: float
    end: float
    room: str          # area id, or "bed" for the household's own bedroom
    prob: float = 0.8

    def active(self, weekday: int, hour: float) -> bool:
        if weekday not in self.days:
            return False
        if self.start < self.end:
            return self.start <= hour < self.end
        return hour >= self.start or hour < self.end


WEEKDAYS = (0, 1, 2, 3, 4)
ALL_DAYS = (0, 1, 2, 3, 4, 5, 6)


@dataclass
class Profile:
    name: str
    description: str
    areas: list[AreaSpec]
    people: list[tuple[str, str | None]]     # (name, own office/room for work hours)
    guest_prob: float
    guest_rooms: list[str]
    locks: list[str]                          # security group
    door_sensors: list[str]
    cameras: list[str]                        # external cameras (security group)
    alarm: bool
    peak_kw: float
    cap_kwh: float
    loads: list[tuple[str, str, int, str]]    # (entity_id, name, watts, kind: hot_water|heating|pump|appliance)
    vehicle_backup: bool = False              # BYD Shark used as a portable generator
    home_prob: float = 0.85
    water: bool = True                        # tank level + house pump in the water group
    irrigation_zones: list[str] = field(default_factory=list)
    schedules: dict[str, list[Block]] = field(default_factory=dict)  # per person, first matching block wins


BUNDOOK = Profile(
    name="bundook",
    description="an off-grid rural home",
    people=[("Peter", "office_pete"), ("Amber", "office_amber")],
    schedules={
        # Peter: Mon-Fri shifts that start in the afternoon and run through midnight to ~4am of the
        # next calendar day (Friday's shift ends Saturday 4am). Sleeps from somewhere between 3 and
        # 6am until 11 or midday, every day including weekends.
        "Peter": [Block((1, 2, 3, 4, 5), 0, 4, "office_pete", 0.8),
                  Block(ALL_DAYS, 3, 5.5, "bed", 0.4),
                  Block(ALL_DAYS, 5.5, 11.5, "bed", 0.9),
                  Block(ALL_DAYS, 11.5, 12.5, "bed", 0.4),
                  Block(WEEKDAYS, 13, 17.5, "office_pete", 0.6)],
        # Amber: 9 to 6 weekdays; Tuesdays and Thursdays she stops mid-afternoon and takes calls 7pm to 11pm.
        "Amber": [Block((1, 3), 9, 15, "office_amber", 0.8),
                  Block((1, 3), 19, 23, "office_amber", 0.85),
                  Block((0, 2, 4), 9, 18, "office_amber", 0.8),
                  Block(ALL_DAYS, 23, 7, "bed", 0.85)],
    },
    guest_prob=0.15,
    guest_rooms=["guest_bedroom", "dining_hall", "alfresco"],
    locks=["Front Door", "Mudroom Door", "Workshop Door"],
    door_sensors=["Front Door", "Mudroom Door", "Workshop Door"],
    cameras=["Driveway Camera", "Entry Intercom", "Workshop Intercom", "North Paddock Camera", "South Paddock Camera", "Shed Camera"],
    alarm=True,
    peak_kw=20.0,   # 63 x 475 W = 30 kW DC, but clipped by 2 x 10 kW Fronius inverters
    cap_kwh=51.7,   # 11 x PowerPlus Eco4847P
    loads=[("switch.hot_water_heat_pump", "Hot Water Heat Pump", 1000, "hot_water"),
           ("switch.hydronic_heating", "Hydronic Heating", 3000, "heating"),
           ("switch.water_pump", "Water Pump", 800, "pump")],
    vehicle_backup=True,
    water=True,
    irrigation_zones=["Vegie Beds", "Orchard", "House Lawn"],
    areas=[
        # HA area names follow the existing install's convention ("Office (Pete)"); `spoken` is what the assistant says.
        AreaSpec("office_pete", "Office (Pete)", "office", spoken="Pete's office", lights=["Downlights", "Track Accent", "Track Wash", "Track Spot"], occupancy=1, temperature=True, blinds=True, windows=True, climate=True, fan="Fan", camera="Camera", media="HomePod"),
        AreaSpec("office_amber", "Office (Amber)", "office", spoken="Amber's office", lights=["Downlights", "Track Accent", "Track Wash"], occupancy=1, temperature=True, blinds=True, windows=True, climate=True, camera="Camera", media="HomePod"),
        AreaSpec("master_bed", "Master Bedroom", "bedroom", ["Downlights", "Wall Lights"], occupancy=1, temperature=True, blinds=True, windows=True, climate=True, window_sensor=True, media="HomePod"),
        AreaSpec("master_robe", "Master Robe", "utility", ["Downlights"], occupancy=1, temperature=True),
        AreaSpec("master_ensuite", "Master Ensuite", "bath", ["Downlights", "Mirror Lights"], occupancy=1, temperature=True, humidity=True, blinds=True, exhaust=True, towel_rail=True),
        AreaSpec("guest_bedroom", "Guest Bedroom", "bedroom", ["Downlights", "Wall Lights"], occupancy=1, motion=1, temperature=True, blinds=True, windows=True, climate=True, media="HomePod"),
        AreaSpec("guest_robe", "Guest Robe", "utility", ["Downlight"], motion=1),
        AreaSpec("guest_ensuite", "Guest Ensuite", "bath", ["Downlights"], occupancy=1, temperature=True, humidity=True, blinds=True, exhaust=True, towel_rail=True),
        AreaSpec("bathroom", "Bathroom", "bath", ["Downlights"], occupancy=1, motion=1, temperature=True, humidity=True, blinds=True, exhaust=True),
        AreaSpec("vanity", "Vanity", "bath", ["Downlights"], occupancy=1, temperature=True),
        AreaSpec("toilet", "Toilet", "bath", ["Downlight"], occupancy=1),
        AreaSpec("kitchen", "Kitchen", "kitchen", ["Downlights", "Island Pendant", "Floor Strip"], occupancy=1, temperature=True, blinds=True, windows=True, climate=True, vacuum=True, appliances=["Dishwasher", "Oven", "Fridge", "Freezer"], media="HomePod"),
        AreaSpec("pantry", "Pantry", "kitchen", ["Downlights"], occupancy=1, temperature=True),
        AreaSpec("dining_hall", "Dining Hall", "living", ["Table Feature Lights", "Sitting Area Feature Lights", "Track Low Fill", "Track High Fill", "Floor Strip", "Bulkhead Strip"], occupancy=2, temperature=True, blinds=True, windows=True, climate=True, fan="Fan", media="TV"),
        AreaSpec("lounge", "Lounge", "living", ["Downlights"], occupancy=1, motion=1, temperature=True),
        AreaSpec("media_room", "Media Room", "living", ["Downlights"], occupancy=1, motion=1, temperature=True),
        AreaSpec("hallway", "Hallway", "entry", ["Downlights", "Pendants", "Floor Strip"], motion=2, temperature=True),
        AreaSpec("mudroom", "Mudroom", "entry", ["Downlights"], occupancy=1, temperature=True, blinds=True),
        AreaSpec("laundry", "Laundry", "utility", ["Downlights"], motion=1, temperature=True, blinds=True, windows=True, exhaust=True, appliances=["Washing Machine", "Dryer"], dehumidifier=True),
        AreaSpec("laundry_toilet", "Laundry Toilet", "bath", ["Downlight"], occupancy=1, exhaust=True),
        AreaSpec("storage", "Storage", "utility", ["Downlights"], climate=True, camera="Camera"),
        AreaSpec("workshop", "Workshop", "utility", ["Downlights"], occupancy=1, temperature=True),
        AreaSpec("carport", "Carport", "outdoor", ["Batten Lights"], motion=1, camera="Camera"),
        AreaSpec("alfresco", "Alfresco", "outdoor", ["Downlights", "Floodlights"], occupancy=1, motion=1, temperature=True, media="HomePod"),
        AreaSpec("entry", "Entry", "outdoor", ["Downlights", "Floodlights"], motion=1),
        AreaSpec("battery_shed", "Battery Shed", "utility", ["Batten Lights"], occupancy=1, temperature=True),
    ],
)

PROFILES: dict[str, Profile] = {BUNDOOK.name: BUNDOOK}
