"""Device models for the big-house generators: every physical device exposes the set of Home
Assistant entities its real integration does, including config and diagnostic entities.

Entity counts per device follow data/house/ha-entity-profiles.json (per-model stats from the
existing house's HA registries, 26 Sep 2026: 3,499 enabled entities on 443 devices, 1,953 primary).
Models that are not in that dump (Apollo R PRO-1, Somfy PoE, Home Connect, Dreame, Fronius,
Selectronic, APC PDU, Starlink, Aranet) are estimates from their integrations; they are marked
`est` below.

Config / diagnostic entities carry attributes["entity_category"] like HA and get cls "device" so
the inventory and the scenario builders never mistake them for user-facing devices; get_state
hides them unless classes=["device"] is asked for, and area targeting skips them (as HA does)."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from .schema import Entity, House

P, C, D = None, "config", "diagnostic"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


@dataclass
class Ent:
    domain: str
    suffix: str                   # appended to the device name; "" is the device's main entity
    cls: str = "device"
    cat: Optional[str] = None     # HA entity_category: None (primary), "config", "diagnostic"
    attrs: dict = field(default_factory=dict)


@dataclass
class Model:
    make: str                     # "manufacturer | model", as in ha-entity-profiles.json
    ents: list[Ent]               # per device (or per channel for multi-channel units)
    unit: list[Ent] = field(default_factory=list)  # once per physical unit (multi-channel relays, meters)
    channels: int = 1


def E(domain, suffix, cls="device", cat=None, **attrs) -> Ent:
    return Ent(domain, suffix, cls, cat, attrs)


def many(domain, cat, names, cls="device", **attrs) -> list[Ent]:
    return [Ent(domain, n, cls, cat, dict(attrs)) for n in names]


def numbered(domain, cat, base, n, cls="device", **attrs) -> list[Ent]:
    return [Ent(domain, f"{base} {i + 1}", cls, cat, dict(attrs)) for i in range(n)]


MAIN = "*"  # placeholder class for the main entity: the caller says what the device is (light, switch, window...)

W = dict(unit="W", nd=0)
KWH = dict(unit="kWh", nd=1)
AMP = dict(unit="A", nd=2)
VOLT = dict(unit="V", nd=0)
PCT = dict(unit="%", nd=0)
DEGC = dict(unit="°C", nd=1)
DBM = dict(unit="dBm", nd=0)

ESPHOME_DIAG = many("sensor", D, ["Uptime", "IP Address", "ESP Temperature"]) + [E("binary_sensor", "Online", cat=D, kind="connectivity")]

MODELS: dict[str, Model] = {
    # ---------------------------------------------------------------- presence / environment
    "hue_motion": Model("Philips | Hue motion sensor", [  # 9 / 3 (Zigbee2MQTT)
        E("binary_sensor", "Occupancy", "occupancy"), E("sensor", "Illuminance", "illuminance"), E("sensor", "Temperature", "temperature"),
        E("sensor", "Battery", cat=D, **PCT), E("number", "Occupancy Timeout", cat=C), E("select", "Motion Sensitivity", cat=C),
        E("switch", "LED Indication", cat=C), E("button", "Identify", cat=C), E("update", "Firmware", cat=C, kind="update")]),
    "hue_motion_outdoor": Model("Philips | Hue motion outdoor sensor", [  # 9 / 3
        E("binary_sensor", "Occupancy", "occupancy"), E("sensor", "Illuminance", "illuminance"), E("sensor", "Temperature", "temperature"),
        E("sensor", "Battery", cat=D, **PCT), E("number", "Occupancy Timeout", cat=C), E("select", "Motion Sensitivity", cat=C),
        E("switch", "LED Indication", cat=C), E("button", "Identify", cat=C), E("update", "Firmware", cat=C, kind="update")]),
    "apollo_r_pro_1": Model("Apollo Automation | R PRO-1 (est)", [  # est ~42 / 6: ESPHome mmWave, LD2412 gates
        E("binary_sensor", "Presence", "occupancy"), *numbered("binary_sensor", P, "Zone Presence", 3, cls="occupancy"),
        E("sensor", "Target Count"), E("sensor", "Illuminance", "illuminance"),
        E("number", "Radar Max Distance", cat=C), E("number", "Radar Timeout", cat=C),
        *[E("number", f"Zone {z} {b}", cat=C) for z in (1, 2, 3) for b in ("Begin", "End")],
        *[E("number", f"Gate {g} {k} Threshold", cat=C) for g in range(8) for k in ("Move", "Still")],
        E("select", "Radar Distance Resolution", cat=C), E("select", "Presence Mode", cat=C),
        E("switch", "Radar Engineering Mode", cat=C), E("switch", "Bluetooth Proxy", cat=C),
        E("button", "Restart", cat=C), E("button", "Factory Reset Radar", cat=C), E("update", "Firmware", cat=C, kind="update"),
        *ESPHOME_DIAG, E("sensor", "Radar Firmware", cat=D)]),
    "sensy_one": Model("Sensy-One | S1 mmWave Sensor", [  # 124 / 111: most of the tuning numbers are primary
        E("binary_sensor", "Presence", "occupancy"), E("binary_sensor", "Motion", "motion"),
        *numbered("binary_sensor", P, "Zone Presence", 4, cls="occupancy"),
        *[E("sensor", f"Target {t} {k}") for t in (1, 2, 3) for k in ("X", "Y", "Speed", "Distance", "Angle")],
        E("sensor", "Target Count"), *numbered("sensor", P, "Zone Target Count", 4),
        *[E("number", f"Zone {z} {k}") for z in (1, 2, 3, 4) for k in ("X Begin", "X End", "Y Begin", "Y End")],
        *[E("number", f"Gate {g} {k} Sensitivity") for g in range(16) for k in ("Move", "Still")],
        *numbered("number", P, "Zone Timeout", 4), *numbered("number", P, "Exclusion Bound", 16), *numbered("number", P, "Tuning", 17),
        E("sensor", "Temperature", cat=D, **DEGC), E("binary_sensor", "Online", cat=D, kind="connectivity"),
        E("button", "Restart", cat=C), E("button", "Calibrate", cat=C), E("button", "Identify", cat=C),
        *many("switch", C, ["Bluetooth", "LED", "Zone Mode", "Auto Calibrate"]),
        *many("sensor", D, ["Uptime", "WiFi Signal", "IP Address", "Firmware Version"])]),
    "espresense": Model("ESPresense (esp32) | ESP32-D0WDQ6-V3", [  # 11 / 1
        E("binary_sensor", "Online", kind="connectivity"),
        *many("number", C, ["Absorption", "Max Distance", "TX Ref RSSI", "RX Adj RSSI"]), E("switch", "Status LED", cat=C),
        E("switch", "Auto Update", cat=C), E("button", "Restart", cat=C), *many("sensor", D, ["Uptime", "Free Memory", "WiFi Signal"])]),
    "bed_presence": Model("ElevatedSensors | BedPresenceMk1", [  # 27 / 8
        E("binary_sensor", "Bed Occupied", "occupancy"), E("binary_sensor", "Left Side", "occupancy"), E("binary_sensor", "Right Side", "occupancy"),
        E("sensor", "Left Pressure"), E("sensor", "Right Pressure"), E("sensor", "Combined Pressure"), E("sensor", "Occupied Time"), E("sensor", "Last Occupied"),
        *numbered("number", C, "Trigger Pressure", 6), *numbered("number", C, "Calibration", 4), E("select", "Mode", cat=C), E("switch", "Calibrate", cat=C),
        E("button", "Calibrate Empty", cat=C), E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update"),
        E("binary_sensor", "Online", cat=D, kind="connectivity"), *many("sensor", D, ["WiFi Signal", "Uptime", "IP Address", "ESP Temperature"])]),
    "aqara_contact": Model("Aqara | Door and window sensor", [  # 4 / 1
        E("binary_sensor", "", MAIN), E("sensor", "Device Temperature", cat=D, **DEGC), E("sensor", "Battery", cat=D, **PCT), E("button", "Identify", cat=C)]),
    "aqara_th": Model("Aqara | Temperature and humidity sensor (est)", [  # est 5 / 3
        E("sensor", "Temperature", "temperature"), E("sensor", "Humidity", "humidity"), E("sensor", "Pressure", "measurement", unit="hPa", nd=0),
        E("sensor", "Battery", cat=D, **PCT), E("button", "Identify", cat=C)]),
    "aqara_leak": Model("Aqara | Water leak sensor (est)", [  # est 4 / 1
        E("binary_sensor", "Leak", "safety", kind="leak"), E("sensor", "Device Temperature", cat=D, **DEGC), E("sensor", "Battery", cat=D, **PCT),
        E("button", "Identify", cat=C)]),
    "develco_air": Model("Develco | Air quality sensor", [  # 7 / 4
        E("sensor", "Temperature", "temperature"), E("sensor", "Humidity", "humidity"), E("sensor", "VOC", "air_quality", unit="ppb", nd=0),
        E("sensor", "Air Quality", "air_quality"), E("sensor", "Battery", cat=D, **PCT), E("binary_sensor", "Battery Low", cat=D, kind="problem"),
        E("update", "Firmware", cat=C, kind="update")]),
    "aranet4": Model("SAF Tehnika | Aranet4 (est)", [  # est 7 / 4
        E("sensor", "CO2", "air_quality", unit="ppm", nd=0), E("sensor", "Temperature", "temperature"), E("sensor", "Humidity", "humidity"),
        E("sensor", "Pressure", "measurement", unit="hPa", nd=0), E("sensor", "Battery", cat=D, **PCT), E("sensor", "Signal", cat=D, **DBM),
        E("sensor", "Update Interval", cat=D)]),
    "clipsal_smoke": Model("Schneider Electric | Clipsal Wiser smoke alarm", [  # 7 / 2
        E("binary_sensor", "Smoke", "safety", kind="smoke"), E("binary_sensor", "Heat", "safety", kind="heat"),
        E("binary_sensor", "Tamper", cat=D, kind="problem"), E("sensor", "Battery", cat=D, **PCT), E("binary_sensor", "Battery Low", cat=D, kind="problem"),
        E("button", "Identify", cat=C), E("update", "Firmware", cat=C, kind="update")]),
    "zigbee_switch": Model("Zigbee | wall scene switch (est)", [  # est 3 / 1
        E("event", "Action", "device"), E("sensor", "Link Quality", cat=D), E("update", "Firmware", cat=C, kind="update")]),

    # ---------------------------------------------------------------- lights, relays, covers
    "hue_bulb": Model("Philips | Hue white A60 bulb E27 806lm bluetooth", [  # 6 / 1
        E("light", "", "light"), E("number", "Startup Brightness", cat=C), E("button", "Identify", cat=C),
        E("text", "Scene", cat=C), E("update", "Firmware", cat=C, kind="update"), E("select", "Power-on Behaviour", cat=C)]),
    "light_group": Model("Zigbee2MQTT | Group", [E("light", "", "light")]),
    "shelly_pro_4pm": Model("Shelly | Shelly Pro 4PM", [  # per channel ~7.5 / 3
        E("*", "", MAIN), E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH),
        E("sensor", "Current", cat=D, **AMP), E("sensor", "Voltage", cat=D, **VOLT), E("sensor", "Temperature", cat=D, **DEGC),
        E("binary_sensor", "Overpower", cat=D, kind="problem")],
        unit=[E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update")], channels=4),
    "shelly_pro_dimmer": Model("Shelly | Shelly Pro Dimmer 2PM (est)", [  # est per channel 6 / 3
        E("light", "", "light"), E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH),
        E("sensor", "Temperature", cat=D, **DEGC), E("binary_sensor", "Overpower", cat=D, kind="problem")],
        unit=[E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update")], channels=2),
    "shelly_pro_rgbww": Model("Shelly | Shelly Pro RGBWW PM (est)", [  # est per channel 3 / 3 + unit 4
        E("light", "", "light"), E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH)],
        unit=[E("sensor", "Temperature", cat=D, **DEGC), E("sensor", "Voltage", cat=D, **VOLT), E("button", "Restart", cat=C),
              E("update", "Firmware", cat=C, kind="update")], channels=5),
    "shelly_dual_cover": Model("Shelly | Shelly Pro Dual Cover PM", [  # per cover ~5.5 / 3
        E("cover", "", MAIN), E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH), E("binary_sensor", "Overpower", cat=D, kind="problem")],
        unit=[E("sensor", "Temperature", cat=D, **DEGC), E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update")], channels=2),
    "somfy_poe": Model("Somfy | Sonesse PoE motor (est)", [  # est 5 / 1
        E("cover", "", MAIN), E("button", "Identify", cat=C), E("button", "Go To My Position", cat=C),
        E("sensor", "Motor Temperature", cat=D, **DEGC), E("update", "Firmware", cat=C, kind="update")]),
    "aqara_blind": Model("Aqara | Roller shade driver E1", [  # 9 / 4
        E("cover", "", MAIN), E("binary_sensor", "Charging"), E("switch", "Reverse Direction"), E("sensor", "Motor State"),
        E("sensor", "Device Temperature", cat=D, **DEGC), E("sensor", "Battery", cat=D, **PCT), E("update", "Firmware", cat=C, kind="update"),
        E("select", "Speed", cat=C), E("sensor", "Link Quality", cat=D)]),
    "shelly_pro_3em": Model("Shelly | Shelly Pro 3EM", [  # per channel ~5.3 / 4.7 (profile 4.6 / 4.1)
        E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH), E("sensor", "Current", "measurement", **AMP),
        E("sensor", "Power Factor", "measurement", unit="", nd=2)],
        unit=[E("sensor", "Temperature", **DEGC), E("sensor", "Voltage", "measurement", **VOLT),
              E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update")], channels=3),
    "meross_plug": Model("Meross | mss305 8.0.0", [  # 13 / 7
        E("switch", "", MAIN), E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH), E("sensor", "Current", "measurement", **AMP),
        E("sensor", "Voltage", "measurement", **VOLT), E("sensor", "Status"), E("sensor", "Energy Today", "meter", **KWH),
        E("light", "DND Mode", cat=C), E("switch", "Child Lock", cat=C), E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update"),
        E("sensor", "WiFi Signal", cat=D, **DBM), E("sensor", "Uptime", cat=D)]),

    # ---------------------------------------------------------------- climate, fans, media, appliances
    "intesis_ac": Model("Intesis | 550", [  # 9 / 2
        E("climate", "", "climate"), E("sensor", "Temperature", "temperature"),
        E("binary_sensor", "Connected", cat=D, kind="connectivity"), E("binary_sensor", "Fault", cat=D, kind="problem"), E("button", "Restart", cat=C),
        E("switch", "Display", cat=C), E("sensor", "Run Time", cat=D), E("sensor", "WiFi Signal", cat=D, **DBM),
        E("sensor", "Outdoor Unit Temperature", cat=D, **DEGC)]),
    "baf_haiku": Model("Big Ass Fans | Haiku (est)", [  # est 19 / 5
        E("fan", "", "fan"), E("light", "Light", "light"), E("sensor", "Temperature", "temperature"), E("sensor", "Humidity", "humidity"),
        E("binary_sensor", "Occupancy", "occupancy"),
        *many("switch", C, ["Auto Comfort", "Eco Mode", "Motion Sense", "Return To Auto", "Sleep Mode", "Whoosh", "Light Auto"]),
        *many("number", C, ["Fan Min Speed", "Fan Max Speed", "Light Min Brightness", "Light Max Brightness", "Motion Timeout"]),
        E("sensor", "WiFi Signal", cat=D, **DBM), E("update", "Firmware", cat=C, kind="update")]),
    "homepod": Model("Apple | HomePod Mini", [E("media_player", "", "media_player"), E("remote", "")]),  # 2 / 2
    "apple_tv": Model("Apple | Apple TV 4K (gen 3)", [E("media_player", "", "media_player"), E("remote", ""), E("binary_sensor", "Keyboard Focus")]),  # 3 / 3
    "lg_tv": Model("LG | webOS TV (est)", [E("media_player", "", "media_player"), E("remote", ""), E("button", "Restart", cat=C)]),
    "lg_washer": Model("LG | TLVA5BA-TL (WASHER)", [  # 16 / 15
        E("switch", "", "appliance"), *many("sensor", P, ["Run State", "Remaining Time", "Current Course", "Error", "Spin Speed", "Water Temperature",
                                                             "Rinse Option", "Tub Clean Counter", "Initial Time", "Total Time"]),
        E("binary_sensor", "Remote Start"), E("binary_sensor", "Door Lock"), E("binary_sensor", "Detergent Low"),
        E("button", "Update Status"), E("sensor", "WiFi Signal", cat=D, **DBM)]),
    "home_connect": Model("Home Connect | appliance (est)", [  # est 20 / 13 (Gaggenau, Neff, Bosch)
        E("switch", "", "appliance"), *many("sensor", P, ["Operation State", "Program Progress", "Remaining Program Time", "Active Program",
                                                             "Selected Program", "Door", "Power State"]),
        E("binary_sensor", "Remote Control"), E("binary_sensor", "Remote Start"), E("binary_sensor", "Local Control"),
        E("button", "Pause Program"), E("button", "Stop Program"),
        *many("switch", C, ["Child Lock", "Sound", "Eco Mode"]), *many("select", C, ["Program", "Start In"]), E("number", "Setpoint", cat=C),
        E("sensor", "Connection", cat=D), E("update", "Firmware", cat=C, kind="update")]),
    "dreame_vacuum": Model("Dreame | X60 Ultra (est)", [  # est 58 / 22
        E("vacuum", "", "vacuum"), *many("sensor", P, ["Battery", "Status", "Cleaning Time", "Cleaned Area", "Error", "Water Tank", "Dust Bag",
                                                          "Mop Pad", "Main Brush Left", "Side Brush Left", "Filter Left", "Current Room",
                                                          "Cleaning History", "Total Cleaning Time", "Total Cleaned Area", "Cleaning Count"]),
        *many("select", P, ["Suction Level", "Mop Humidity", "Cleaning Mode", "Map"]), E("camera", "Map"),
        *many("switch", C, ["Carpet Boost", "Child Lock", "Do Not Disturb", "Auto Empty", "Obstacle Avoidance", "AI Detection", "Auto Mop Drying",
                             "Resume Cleaning", "Multi Floor Map", "Tight Mopping", "Auto Detergent", "Hot Water Mop Wash"]),
        *many("number", C, ["Volume", "Mop Wash Level", "Drying Time", "DND Start", "DND End"]),
        *many("button", C, ["Reset Main Brush", "Reset Side Brush", "Reset Filter", "Reset Mop Pad", "Self Clean", "Empty Dust Bin"]),
        *many("sensor", D, ["WiFi Signal", "Firmware Version", "Uptime", "Last Clean", "Map Size", "Cleaning Sequence", "Detergent Left", "Dirty Water Tank"]),
        E("update", "Firmware", cat=C, kind="update")]),

    # ---------------------------------------------------------------- security
    "g6_ptz": Model("Ubiquiti | G6 PTZ", [  # 48 / 17
        E("camera", "", "camera"), E("binary_sensor", "Motion", "detection", kind="motion"),
        *[E("binary_sensor", f"{o} Detected", "detection", kind=o.lower()) for o in ("Person", "Vehicle", "Animal", "Package", "License Plate", "Face")],
        E("binary_sensor", "Smoke Alarm", "safety", kind="smoke"), E("binary_sensor", "CO Alarm", "safety", kind="co"),
        E("binary_sensor", "Glass Break", "safety", kind="glass"), E("binary_sensor", "Speaking Detected", "detection", kind="speaking"),
        E("binary_sensor", "Bark Detected", "detection", kind="bark"), E("media_player", "Speaker"), E("event", "Smart Detection", "detection"),
        E("sensor", "Last Motion"), E("sensor", "Detected Object"),
        *many("number", C, ["Microphone Level", "Zoom Level", "Motion Sensitivity", "Wide Dynamic Range", "Infrared Lux Trigger"]),
        *many("select", C, ["Recording Mode", "Infrared Mode", "HDR Mode", "PTZ Preset"]),
        *many("switch", C, ["Privacy Mode", "Status Light", "Overlay: Show Name", "Overlay: Show Date", "Overlay: Show Logo", "Overlay: Nerd Mode",
                             "Detections: Motion", "Detections: Person", "Detections: Vehicle", "Detections: Animal", "Detections: Package",
                             "Detections: License Plate", "Detections: Smoke", "Detections: CO", "Tracking: Person", "System Sounds", "Microphone"]),
        E("button", "Restart", cat=C), *many("sensor", D, ["Storage Used", "Uptime", "Received Data", "Transferred Data"])]),
    "g6_turret": Model("Ubiquiti | G5 Turret Ultra", [  # 42 / 15
        E("camera", "", "camera"), E("binary_sensor", "Motion", "detection", kind="motion"),
        *[E("binary_sensor", f"{o} Detected", "detection", kind=o.lower()) for o in ("Person", "Vehicle", "Animal", "Package", "License Plate", "Face")],
        E("binary_sensor", "Smoke Alarm", "safety", kind="smoke"), E("binary_sensor", "CO Alarm", "safety", kind="co"),
        E("binary_sensor", "Glass Break", "safety", kind="glass"), E("binary_sensor", "Speaking Detected", "detection", kind="speaking"),
        E("binary_sensor", "Bark Detected", "detection", kind="bark"), E("event", "Smart Detection", "detection"), E("sensor", "Last Motion"),
        *many("number", C, ["Microphone Level", "Motion Sensitivity", "Wide Dynamic Range", "Infrared Lux Trigger"]),
        *many("select", C, ["Recording Mode", "Infrared Mode", "HDR Mode"]),
        *many("switch", C, ["Privacy Mode", "Status Light", "Overlay: Show Name", "Overlay: Show Date", "Overlay: Show Logo", "Overlay: Nerd Mode",
                             "Detections: Motion", "Detections: Person", "Detections: Vehicle", "Detections: Animal", "Detections: Package",
                             "Detections: License Plate", "Detections: Smoke", "Detections: CO", "System Sounds", "Microphone"]),
        E("button", "Restart", cat=C), *many("sensor", D, ["Storage Used", "Uptime", "Received Data"])]),
    "g6_entry": Model("Ubiquiti | G6 Entry (est)", [  # est 30 / 11
        E("camera", "", "camera"), E("binary_sensor", "Motion", "detection", kind="motion"),
        *[E("binary_sensor", f"{o} Detected", "detection", kind=o.lower()) for o in ("Person", "Vehicle", "Package", "Face")],
        E("event", "Doorbell", "detection", kind="doorbell"), E("binary_sensor", "Ringing", "detection", kind="doorbell"), E("media_player", "Speaker"),
        E("binary_sensor", "Smoke Alarm", "safety", kind="smoke"), E("binary_sensor", "CO Alarm", "safety", kind="co"),
        *many("switch", C, ["Privacy Mode", "Status Light", "Overlay: Show Name", "Overlay: Show Date", "Detections: Motion", "Detections: Person",
                             "Detections: Vehicle", "Detections: Package", "System Sounds", "Microphone"]),
        *many("number", C, ["Microphone Level", "Chime Duration", "Motion Sensitivity"]), *many("select", C, ["Recording Mode", "Doorbell Text"]),
        E("button", "Restart", cat=C), *many("sensor", D, ["Storage Used", "Uptime", "Received Data"])]),
    "frigate_camera": Model("Frigate | 0.17", [  # 28 / 22
        E("camera", "Frigate", "camera"), E("binary_sensor", "Frigate Motion", "detection", kind="motion"),
        *[e for o in ("Person", "Car", "Dog", "Cat") for e in (E("binary_sensor", f"{o} Occupancy", "detection", kind=o.lower()),
                                                               E("sensor", f"{o} Count"), E("image", f"{o}"))],
        E("binary_sensor", "All Occupancy", "detection", kind="object"), E("sensor", "All Count"),
        *many("sensor", P, ["Camera FPS", "Detection FPS", "Process FPS", "Skipped FPS", "Review Status"]),
        *many("switch", C, ["Detect", "Recordings", "Snapshots", "Motion", "Review Alerts", "Review Detections"])]),
    "unifi_lock": Model("Unifi | UA-Hub-Door-Mini", [  # 10 / 7
        E("lock", "", "lock"), E("binary_sensor", "Door Position", "device"), E("event", "Doorbell", "detection", kind="doorbell"),
        E("event", "Access", "detection", kind="access"), E("binary_sensor", "Occupancy", "detection", kind="person"), E("select", "Lock Rule"),
        E("image", "Thumbnail"), E("number", "Unlock Duration", cat=C), E("sensor", "Last Access", cat=D), E("sensor", "Firmware", cat=D)]),
    "udm_se": Model("Ubiquiti | UDM SE", [  # 9 / 1
        E("alarm_control_panel", "", "alarm"), E("sensor", "CPU Temperature", cat=D, **DEGC), E("sensor", "Uptime", cat=D), E("switch", "Guest Network", cat=C),
        *many("sensor", D, ["WAN Download", "WAN Upload", "CPU Usage", "Memory Usage"]), E("update", "Firmware", cat=C, kind="update")]),
    "ratgdo": Model("ratgdo | garage door opener (est)", [  # est 10 / 4
        E("cover", "", "garage"), E("light", "Light", "light"), E("binary_sensor", "Obstruction", "safety", kind="obstruction"),
        E("binary_sensor", "Motion", "motion"), E("button", "Sync", cat=C), E("number", "Opening Duration", cat=C), E("number", "Closing Duration", cat=C),
        E("sensor", "Openings", cat=D), E("sensor", "WiFi Signal", cat=D, **DBM), E("update", "Firmware", cat=C, kind="update")]),

    # ---------------------------------------------------------------- network, rack, energy plant
    "unifi_ap": Model("Ubiquiti | U7 access point (est)", [  # est 7 / 1
        E("sensor", "Clients"), E("sensor", "State", cat=D), E("sensor", "Uptime", cat=D), E("sensor", "Temperature", cat=D, **DEGC),
        E("light", "LED", cat=C), E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update")]),
    "unifi_switch": Model("Ubiquiti | USW (est)", [  # per used port 3 (PoE switch, PoE power, port enable)
        E("switch", "PoE", cat=C), E("sensor", "PoE Power", cat=D, **W), E("switch", "Enabled", cat=C)],
        unit=[E("sensor", "Clients"), E("sensor", "State", cat=D), E("sensor", "Uptime", cat=D), E("sensor", "Temperature", cat=D, **DEGC),
              E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update")], channels=24),
    "unvr": Model("Ubiquiti | UNVR (est)", [  # est 11 / 1
        E("sensor", "Storage Used", **PCT), *many("sensor", D, ["Uptime", "CPU Usage", "Memory Usage", "CPU Temperature", "Recording Capacity"]),
        *[E("binary_sensor", f"Disk {i} Health", cat=D, kind="problem") for i in (1, 2, 3, 4)], E("button", "Restart", cat=C)]),
    "apc_ups": Model("American Power Conversion | Smart-UPS_750", [  # 14 / 3
        E("sensor", "Battery", **PCT), E("sensor", "Status"), E("sensor", "Load", **PCT),
        *many("sensor", D, ["Runtime", "Input Voltage", "Output Voltage", "Battery Voltage", "Last Transfer", "Transfer Count", "Self Test",
                             "Time On Battery", "Nominal Power", "Internal Temperature", "Model"])]),
    "apc_pdu": Model("American Power Conversion | Metered Rack PDU (est)", [
        E("sensor", "Power", "power", **W), E("sensor", "Current", "measurement", **AMP), E("sensor", "Energy", "meter", **KWH),
        E("sensor", "Voltage", cat=D, **VOLT), E("sensor", "Status", cat=D)]),
    "synology": Model("Synology | DS418", [  # 7.5 / 4
        E("sensor", "Volume Used", **PCT), E("sensor", "Download Throughput"), E("sensor", "Upload Throughput"), E("sensor", "Status"),
        E("sensor", "Temperature", cat=D, **DEGC), E("sensor", "Uptime", cat=D), E("update", "DSM", cat=C, kind="update"), E("button", "Reboot", cat=C)]),
    "starlink": Model("SpaceX | Starlink (est)", [  # est 20 / 10
        *many("sensor", P, ["Downlink Throughput", "Uplink Throughput", "Ping", "Ping Drop Rate", "Azimuth", "Elevation"]),
        E("binary_sensor", "Obstructed", kind="problem"), E("binary_sensor", "Heating"), E("binary_sensor", "Connected", kind="connectivity"),
        E("switch", "Stowed"), E("switch", "Sleep Schedule", cat=C), *many("number", C, ["Sleep Start", "Sleep Duration"]),
        E("button", "Restart", cat=C), *many("sensor", D, ["Uptime", "Last Boot", "Hardware Version", "Software Version", "Power Draw", "Alerts"])]),
    "fronius_gen24": Model("Fronius | Primo GEN24 (est, SolarEdge SE8250H profile 36 / 12)", [
        E("sensor", "AC Power", "power", **W), E("sensor", "DC Power MPPT 1", "power", **W), E("sensor", "DC Power MPPT 2", "power", **W),
        E("sensor", "Energy Today", "meter", **KWH), E("sensor", "Energy Total", "meter", unit="MWh", nd=2),
        E("sensor", "AC Voltage", "measurement", **VOLT), E("sensor", "AC Current", "measurement", **AMP), E("sensor", "Frequency", "measurement", unit="Hz", nd=2),
        E("sensor", "DC Voltage MPPT 1", "measurement", **VOLT), E("sensor", "DC Voltage MPPT 2", "measurement", **VOLT), E("sensor", "Status"),
        E("binary_sensor", "Producing", kind="connectivity"),
        *many("sensor", D, ["DC Current MPPT 1", "DC Current MPPT 2", "Inverter Temperature", "Error Code", "Last Update", "Energy Year", "Isolation Resistance",
                             "Apparent Power", "Reactive Power", "Power Factor", "Modbus Status", "Firmware"]),
        *many("number", C, ["Export Limit", "Power Limit"]), *many("select", C, ["Operating Mode", "Storage Control"]), E("switch", "Enabled", cat=C),
        E("button", "Restart", cat=C), *numbered("sensor", D, "String Current", 6)]),
    "selectronic": Model("Selectronic | SP PRO (est, select.live)", [  # est 24 / 12
        E("sensor", "Battery Voltage", "measurement", **VOLT), E("sensor", "Battery Current", "measurement", **AMP),
        E("sensor", "Battery Temperature", "measurement", **DEGC), E("sensor", "Load Energy Today", "meter", **KWH), E("sensor", "Solar Energy Today", "meter", **KWH),
        E("sensor", "Battery In Today", "meter", **KWH), E("sensor", "Battery Out Today", "meter", **KWH), E("sensor", "Generator Power", "power", **W),
        E("sensor", "Inverter Power", "power", **W), E("sensor", "AC Load Voltage", "measurement", **VOLT), E("sensor", "Status"),
        E("binary_sensor", "Generator Running"),
        *many("sensor", D, ["Inverter Temperature", "Transformer Temperature", "Fault Code", "Last Update", "Shunt 1 Power", "Shunt 2 Power",
                             "Battery Cycles", "Firmware"]), *many("number", C, ["Generator Start SoC", "Generator Stop SoC"]), E("button", "Sync Time", cat=C),
        E("switch", "Generator Auto Start", cat=C)]),
    "battery_module": Model("PowerPlus | Eco4847P BMS (est)", [  # est 6 / 0 per module
        *many("sensor", D, ["State Of Charge", "Voltage", "Current", "Temperature", "Cycles", "Cell Delta"])]),
    "dab_esybox": Model("DAB Pumps | esybox_mini_maxiCPU", [  # 69 / 33 (the pump switch itself is the house pump)
        E("sensor", "Flow Rate", "water", kind="flow"), E("sensor", "Water Delivered", "meter", unit="L", nd=0), E("sensor", "Voltage", "measurement", **VOLT),
        E("sensor", "Current", "measurement", **AMP), E("sensor", "Energy", "meter", **KWH), E("sensor", "Power", "power", **W),
        E("sensor", "Power Factor", "measurement", unit="", nd=2), E("sensor", "Temperature", "measurement", **DEGC), E("sensor", "Status"),
        E("sensor", "Run Time"), E("sensor", "Starts"), E("sensor", "Pressure", "water", kind="pressure"),
        *numbered("sensor", P, "Pump Parameter", 21),
        *numbered("number", C, "Setpoint", 12), *numbered("select", C, "Mode", 4), E("switch", "Power Shower", cat=C), E("switch", "Sleep Mode", cat=C),
        E("button", "Reset Alarms", cat=C), E("button", "Restart", cat=C),
        *numbered("sensor", D, "Diagnostic", 14), E("sensor", "Firmware", cat=D), E("sensor", "Signal", cat=D)]),
    "ecowitt_valve": Model("Ecowitt | WFC01", [  # 11 / 4
        E("switch", "", "irrigation"), E("sensor", "Water Total", "meter", unit="L", nd=0), E("sensor", "Water Temperature", "measurement", **DEGC),
        E("sensor", "Flow", "water", kind="flow"),
        E("binary_sensor", "Heat Alarm", cat=D, kind="problem"), E("binary_sensor", "Cold Alarm", cat=D, kind="problem"),
        E("binary_sensor", "Moisture", cat=D, kind="problem"), E("binary_sensor", "Connected", cat=D, kind="connectivity"),
        E("binary_sensor", "Problem", cat=D, kind="problem"), E("sensor", "Battery", cat=D, **PCT), E("binary_sensor", "Battery Low", cat=D, kind="problem")]),
    "tank_sensor": Model("Tuya | tank level sensor (est)", [  # est 4 / 2
        E("sensor", "", "water", kind="level"), E("sensor", "Depth", "measurement", unit="m", nd=2), E("sensor", "Battery", cat=D, **PCT),
        E("sensor", "Signal", cat=D, **DBM)]),
    "ecowitt_station": Model("Ecowitt | GW1100C", [  # 49 / 38 (lives in the weather group)
        E("sensor", "Outdoor Humidity", "humidity"), E("sensor", "Illuminance", "illuminance"),
        *[E("sensor", n, "measurement", unit=u, nd=nd) for n, u, nd in (
            ("Dew Point", "°C", 1), ("Feels Like", "°C", 1), ("Wind Speed", "km/h", 0), ("Wind Gust", "km/h", 0), ("Max Daily Gust", "km/h", 0),
            ("Wind Direction", "°", 0), ("Rain Rate", "mm/h", 1), ("Event Rain", "mm", 1), ("Hourly Rain", "mm", 1), ("Daily Rain", "mm", 1),
            ("Weekly Rain", "mm", 1), ("Monthly Rain", "mm", 1), ("Yearly Rain", "mm", 0), ("Solar Radiation", "W/m²", 0), ("UV Index", "", 0),
            ("Relative Pressure", "hPa", 0), ("Absolute Pressure", "hPa", 0), ("Console Temperature", "°C", 1), ("Console Humidity", "%", 0),
            ("Soil Moisture 1", "%", 0), ("Soil Moisture 2", "%", 0), ("Soil Moisture 3", "%", 0), ("Soil Temperature", "°C", 1),
            ("Lightning Strikes", "", 0), ("Lightning Distance", "km", 0), ("Leaf Wetness", "%", 0), ("PM2.5 Outdoor", "µg/m³", 0),
            ("PM2.5 24h Average", "µg/m³", 0), ("Vapour Pressure Deficit", "kPa", 2), ("Heat Index", "°C", 1), ("Wind Chill", "°C", 1),
            ("Rain Today Piezo", "mm", 1), ("Rain State", "", 0), ("Evapotranspiration", "mm", 1), ("Last Rain", "", 0), ("Sunshine Hours", "h", 1))],
        *numbered("sensor", D, "Sensor Battery", 8), E("binary_sensor", "Rain Gauge Battery Low", cat=D, kind="problem"),
        E("sensor", "Console Voltage", cat=D, **VOLT), E("sensor", "Firmware", cat=D)]),
    "phone": Model("Apple | iPhone16,1", [  # 24.5 / 23
        E("device_tracker", ""), *many("sensor", P, ["Battery Level", "Battery State", "Activity", "Steps", "Distance", "Floors Ascended",
                                                         "Geocoded Location", "Connection Type", "SSID", "BSSID", "Storage", "Last Update Trigger",
                                                         "Audio Output", "App Version", "Location Permission", "SIM 1", "SIM 2", "Average Active Pace",
                                                         "Floors Descended", "Watch Battery"]),
        E("binary_sensor", "Focus"), E("notify", "Notify"), E("sensor", "Last Seen", cat=D), E("sensor", "Push Registered", cat=D)]),
    "byd_shark": Model("BYD | BYD SHARK", [  # 116 / 40 (main battery sensor is the energy-group vehicle entity)
        E("device_tracker", "Location"), E("lock", "Doors"), E("climate", "Climate"), E("switch", "Charging"),
        *many("sensor", P, ["Speed", "Odometer", "Range", "Fuel Level", "Fuel Range", "Charging Power", "Charge Time Remaining", "Cabin Temperature",
                             "Outside Temperature", "Tyre Pressure FL", "Tyre Pressure FR", "Tyre Pressure RL", "Tyre Pressure RR", "Energy Consumption",
                             "Trip Distance", "12V Battery", "Coolant Temperature", "Engine State", "Gear", "Power Output", "Last Trip", "Drive Mode",
                             "Charge Limit"]),
        *many("binary_sensor", P, ["Driver Door", "Passenger Door", "Rear Left Door", "Rear Right Door", "Tailgate", "Bonnet", "Windows", "Plugged In",
                                    "Charging", "Engine Running", "Online", "Problem"]),
        *numbered("number", C, "Setting", 18), *numbered("select", C, "Option", 8), *numbered("switch", C, "Feature", 10), *numbered("button", C, "Command", 12),
        *numbered("sensor", D, "Telemetry", 26), E("update", "Firmware", cat=C, kind="update")]),
    "luba_mower": Model("Mammotion | Luba2AWD5000", [  # 114 / 14
        E("lawn_mower", ""), E("button", "Return To Dock"), *many("sensor", P, ["Battery", "Status", "Progress", "Area Mowed", "Blade Height", "Speed",
                                                                                    "Work Time", "Total Time", "Position", "Error", "Last Mow", "Next Mow"]),
        *numbered("number", C, "Mowing Setting", 30), *numbered("select", C, "Mowing Option", 14), *numbered("switch", C, "Mowing Feature", 16),
        *numbered("button", C, "Mowing Command", 8), *numbered("sensor", D, "Mower Telemetry", 31), E("device_tracker", "Tracker", cat=D),
        E("update", "Firmware", cat=C, kind="update")]),
}


# Models only the projected new build uses (Peter's spec for 319 Bundook Rd, 26 Sep): richer mmWave, per-room A/C
# with its sub-entities, Home Connect fridges, the connected hot-water heat pump and hydronic floor heating.
MODELS.update({
    "apollo_r_pro_1_full": Model("Apollo Automation | R PRO-1 (est, all zones and tuning exposed, Sensy-One class)", [  # est ~120 / ~104
        E("binary_sensor", "Presence", "occupancy"), E("binary_sensor", "Motion", "motion"),
        *numbered("binary_sensor", P, "Zone Presence", 3, cls="occupancy"), *numbered("binary_sensor", P, "Zone Motion", 3, cls="motion"),
        *numbered("sensor", P, "Zone Distance", 3),
        *[E("sensor", f"Target {t} {k}") for t in (1, 2, 3) for k in ("X", "Y", "Speed", "Distance", "Angle")],
        E("sensor", "Target Count"), *numbered("sensor", P, "Zone Target Count", 3), E("sensor", "Illuminance", "illuminance"),
        *[E("number", f"Zone {z} {k}") for z in (1, 2, 3) for k in ("X Begin", "X End", "Y Begin", "Y End")],
        *[E("number", f"Gate {g} {k} Threshold") for g in range(14) for k in ("Move", "Still")],
        *numbered("number", P, "Zone Timeout", 3), *numbered("number", P, "Tuning", 30),
        *many("select", C, ["Radar Distance Resolution", "Presence Mode", "Tracking Mode"]),
        *many("switch", C, ["Radar Engineering Mode", "Bluetooth Proxy", "Status LED"]),
        *many("button", C, ["Restart", "Factory Reset Radar", "Calibrate"]), E("update", "Firmware", cat=C, kind="update"),
        *ESPHOME_DIAG, E("sensor", "Radar Firmware", cat=D), E("sensor", "Network Signal", cat=D)]),
    "mhi_ac_wifi": Model("Mitsubishi Heavy | split system WiFi (est)", [  # est 14 / 9
        E("climate", "", "climate"), E("sensor", "Temperature", "temperature"), E("sensor", "Humidity", "humidity"),
        E("sensor", "Outdoor Temperature", "measurement", **DEGC), E("select", "Fan Mode"), E("select", "Swing Mode"),
        E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH), E("select", "Mode"),
        E("sensor", "Filter Hours", cat=D), E("binary_sensor", "Connected", cat=D, kind="connectivity"), E("sensor", "Error Code", cat=D),
        E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update")]),
    "ac_extras": Model("Intesis | 550 (richer mapping, est)", [  # added to a drawn head: 5 / 5
        E("sensor", "Humidity", "humidity"), E("select", "Fan Mode"), E("select", "Swing Mode"), E("sensor", "Power", "power", **W),
        E("sensor", "Energy", "meter", **KWH)]),
    "shelly_dimmer_input": Model("Shelly | Pro Dimmer switch input", [  # the wall switch wired to the channel's input: 2 / 2
        E("event", "Switch", "device"), E("binary_sensor", "Switch Input", "device")]),
    "home_connect_fridge": Model("Home Connect | fridge / freezer (est)", [  # est 14 / 10
        E("switch", "", "appliance"), E("binary_sensor", "Door"), E("number", "Setpoint"), E("sensor", "Door Alarm"), E("sensor", "Temperature Alarm"),
        *many("switch", P, ["Super Mode", "Eco Mode", "Vacation Mode", "Fresh Mode"]), E("sensor", "Operation State"),
        *many("switch", C, ["Child Lock", "Sabbath Mode"]), E("sensor", "Connection", cat=D), E("update", "Firmware", cat=C, kind="update")]),
    "reclaim_hwhp": Model("Reclaim | CO2 heat pump hot water (est)", [  # est 22 / 15
        *[E("sensor", f"{n} Temperature", "measurement", **DEGC) for n in ("Tank Top", "Tank Middle", "Tank Bottom", "Outlet", "Inlet", "Ambient")],
        E("sensor", "Compressor Power", "power", **W), E("sensor", "Energy", "meter", **KWH), E("sensor", "Status"), E("select", "Mode"),
        E("number", "Setpoint"), E("binary_sensor", "Defrost"), E("sensor", "Error"), E("sensor", "COP"), E("switch", "Boost"),
        *many("sensor", D, ["Run Hours", "Compressor Starts", "WiFi Signal"]), E("switch", "Legionella Cycle", cat=C),
        E("number", "Legionella Day", cat=C), E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update")]),
    "intaflo_plant": Model("Intaflo | HH12 hydronic heat pump (est)", [  # est 24 / 15
        *[E("sensor", f"{n} Temperature", "measurement", **DEGC) for n in ("Flow", "Return", "Buffer Tank", "Outdoor")],
        E("sensor", "Delta T", "measurement", unit="°C", nd=1), E("sensor", "Compressor Power", "power", **W), E("sensor", "Energy", "meter", **KWH),
        E("binary_sensor", "Circulation Pump"), E("sensor", "Pump Speed", "measurement", **PCT), E("sensor", "Flow Rate", "measurement", unit="L/min", nd=1),
        E("select", "Mode"), E("number", "Flow Setpoint"), E("sensor", "Status"), E("binary_sensor", "Defrost"), E("sensor", "Error"),
        *many("sensor", D, ["Run Hours", "Compressor Starts", "WiFi Signal", "Firmware Version"]),
        *many("number", C, ["Heating Curve Slope", "Heating Curve Offset"]), E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update"),
        E("switch", "Night Setback", cat=C)]),
    "floor_zone": Model("Hydronic floor heating zone (est, wired thermostat + actuator)", [  # est 11 / 8
        E("climate", "", "climate"), E("sensor", "Floor Temperature", "measurement", **DEGC), E("sensor", "Room Temperature", "temperature"),
        E("sensor", "Flow Temperature", "measurement", **DEGC), E("sensor", "Return Temperature", "measurement", **DEGC),
        E("binary_sensor", "Valve"), E("sensor", "Valve Position", "measurement", **PCT), E("sensor", "Heating Demand", "measurement", **PCT),
        E("number", "Max Floor Temperature", cat=C), E("switch", "Window Detection", cat=C), E("sensor", "Signal", cat=D)]),
})

# The plan's own DIN-rail modules (board_items in the Mains / Submains boxes), per-channel entity sets per Peter's spec.
MODELS.update({
    "bd_pro_dimmer_2pm": Model("Shelly | Shelly Pro Dimmer 2PM", [  # per channel 8 / 5 (+ 2 per wall switch wired to its inputs)
        E("light", "", MAIN), E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH), E("sensor", "Current", "measurement", **AMP),
        E("sensor", "Voltage", "measurement", **VOLT), E("sensor", "Temperature", cat=D, **DEGC), E("binary_sensor", "Overpower", cat=D, kind="problem"),
        E("binary_sensor", "Overtemperature", cat=D, kind="problem")],
        unit=[E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update"), E("sensor", "Uptime", cat=D)], channels=2),
    "bd_pro_3em": Model("Shelly | Shelly Pro 3EM", [  # per CT 5 / 5
        E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH), E("sensor", "Current", "measurement", **AMP),
        E("sensor", "Voltage", "measurement", **VOLT), E("sensor", "Power Factor", "measurement", unit="", nd=2)],
        unit=[E("sensor", "Temperature", **DEGC), E("sensor", "Frequency", "measurement", unit="Hz", nd=2), E("button", "Restart", cat=C),
              E("update", "Firmware", cat=C, kind="update")], channels=3),
    "bd_pro_em": Model("Shelly | Shelly Pro EM", [  # per CT 5 / 5
        E("sensor", "Power", "power", **W), E("sensor", "Energy", "meter", **KWH), E("sensor", "Current", "measurement", **AMP),
        E("sensor", "Voltage", "measurement", **VOLT), E("sensor", "Power Factor", "measurement", unit="", nd=2)],
        unit=[E("sensor", "Temperature", **DEGC), E("button", "Restart", cat=C), E("update", "Firmware", cat=C, kind="update")], channels=2),
})


def wall_inputs(names: list[str]) -> list[Ent]:
    """A wall switch wired to a Shelly channel's input: an event and an input binary_sensor each."""
    return [e for n in names for e in (E("event", n), E("binary_sensor", f"{n} Input"))]


def shelly_i4(labels: list[str]) -> list[Ent]:
    """A Shelly i4 behind a wall plate: an event and an input binary_sensor per button, plus the unit's own entities."""
    return [e for lab in labels for e in (E("event", f"{lab} Button"), E("binary_sensor", f"{lab} Input"))] + [
        E("sensor", "Temperature", cat=D, **DEGC), E("sensor", "WiFi Signal", cat=D, **DBM), E("button", "Restart", cat=C),
        E("update", "Firmware", cat=C, kind="update")]


class DeviceMixin:
    """Instantiates catalog devices on a Gen subclass (needs self.rng, self._add)."""

    def _devinit(self):
        self.devices: list[dict] = []
        self._eids: set[str] = set()

    def _eid(self, house: House, domain: str, name: str) -> str:
        base = f"{domain}.{slug(name)}"
        eid, i = base, 2
        while eid in house.entities:
            eid, i = f"{base}_{i}", i + 1
        return eid

    def _device(self, house: House, model_key: str, area: str, name: str, main_cls: Optional[str] = None,
                main_domain: Optional[str] = None, main_id: Optional[str] = None, main_name: Optional[str] = None,
                main_attrs: Optional[dict] = None, load_w: float = 0.0, circuit: Optional[str] = None,
                ents: Optional[list[Ent]] = None, unit_of: Optional[dict] = None) -> dict:
        """Add one device (or one channel of a multi-channel unit) and return its record."""
        m = MODELS[model_key]
        dev = {"slug": slug(name), "model": model_key, "area": area, "name": name, "main": None, "ents": [], "load_w": load_w,
               "circuit": circuit, "unit": unit_of}
        for spec in (ents if ents is not None else m.ents):
            is_main = spec.suffix == "" and dev["main"] is None
            domain = main_domain if (is_main and main_domain) else spec.domain
            cls = main_cls if (is_main and main_cls) else spec.cls
            if cls == MAIN:
                raise ValueError(f"{model_key}: main class not given for {name}")
            nm = (main_name or name) if is_main else f"{name} {spec.suffix}".strip()
            eid = main_id if (is_main and main_id and main_id not in house.entities) else self._eid(house, domain, nm)
            attrs = dict(spec.attrs)
            if is_main and main_attrs:
                attrs.update(main_attrs)
            attrs["device"] = dev["slug"]
            if spec.cat:
                attrs["entity_category"] = spec.cat
            e = self._add(house, domain, eid, nm, cls, area, state=self._initial(domain, cls, attrs), attributes=attrs)
            dev["ents"].append(e)
            if is_main:
                dev["main"] = e
        self.devices.append(dev)
        return dev

    def _attach(self, house: House, main: Entity, model_key: str, area: Optional[str] = None, load_w: float = 0.0,
                circuit: Optional[str] = None) -> dict:
        """Give an entity made by the small-house helpers (_appliance, _water, _vehicle) the rest of its device."""
        m = MODELS[model_key]
        rest = [e for e in m.ents if not (e.suffix == "" and e.domain in (main.domain, "*"))]
        dev = self._device(house, model_key, area or main.area, main.name, ents=rest, load_w=load_w, circuit=circuit)
        dev["main"] = main
        main.attributes["device"] = dev["slug"]
        return dev

    def _track(self, main: Entity, load_w: float, circuit: Optional[str]):
        """A bare entity (no device extras) whose draw still counts on its circuit."""
        self.devices.append({"slug": slug(main.name), "model": None, "area": main.area, "name": main.name, "main": main, "ents": [],
                             "load_w": load_w, "circuit": circuit, "unit": None})

    def _channel(self, house: House, model_key: str, area: str, name: str, **kw) -> dict:
        """One channel of a multi-channel DIN-rail unit (Shelly Pro 4PM / Dimmer / RGBWW / Dual Cover / 3EM,
        UniFi switch port). Units fill up in order; each unit's own entities live in the board's area."""
        m = MODELS[model_key]
        units = self.__dict__.setdefault("_units", {})
        cur = units.get(model_key)
        if cur is None or cur[1] >= m.channels:
            counts = self.__dict__.setdefault("_unit_counts", {})
            n = counts[model_key] = counts.get(model_key, 0) + 1
            label = {"shelly_pro_4pm": "Relay", "shelly_pro_dimmer": "Dimmer", "shelly_pro_rgbww": "LED Driver", "shelly_dual_cover": "Cover Controller",
                     "shelly_pro_3em": "Circuit Meter", "unifi_switch": "Switch"}.get(model_key, "Unit")
            uname = f"USW {n}" if model_key == "unifi_switch" else f"{self.board_name} {label} {n}"
            u = self._device(house, model_key, kw.pop("unit_area", None) or self.board_area, uname, ents=m.unit)
            u["is_unit"] = True
            cur = units[model_key] = [u, 0]
        cur[1] += 1
        kw.pop("unit_area", None)
        return self._device(house, model_key, area, name, unit_of=cur[0], **kw)

    def _hue(self, house: House, area: str, an: str, label: str, n: int, watts: float = 9.0, circuit: Optional[str] = None) -> Entity:
        """Hue bulbs: one bulb is the light itself, several get a Zigbee2MQTT group named for the fitting."""
        aid = area
        if n == 1:
            return self._device(house, "hue_bulb", aid, f"{an} {label}", main_id=f"light.{aid}_{slug(label)}", load_w=watts, circuit=circuit)["main"]
        grp = self._device(house, "light_group", aid, f"{an} {label}", main_id=f"light.{aid}_{slug(label)}")["main"]
        one = label[:-1] if label.endswith("s") else label
        members = []
        for i in range(n):
            b = self._device(house, "hue_bulb", aid, f"{an} {one} {i + 1}", load_w=watts, circuit=circuit)
            members.append(b["main"].entity_id)
        grp.attributes["entity_id"] = members
        return grp

    def _unit(self, house: House, model_key: str, area: str, name: str) -> dict:
        """The per-unit entities of a multi-channel device (restart button, firmware, board temperature)."""
        m = MODELS[model_key]
        return self._device(house, model_key, area, name, ents=m.unit)

    def _initial(self, domain: str, cls: str, attrs: dict) -> str:
        rng = self.rng
        kind = attrs.get("kind")
        if domain in ("binary_sensor",):
            if kind == "connectivity":
                return "on" if rng.random() < 0.98 else "off"
            return "off"
        if domain == "update":
            return "on" if rng.random() < 0.08 else "off"
        if domain in ("button", "event"):
            return "unknown"
        if domain == "number":
            return str(rng.choice([0, 1, 5, 10, 15, 30, 50, 60, 100, 120, 300]))
        if domain in ("select", "text"):
            return rng.choice(["auto", "default", "high", "medium", "low", "off", "on"])
        if cls == "lock":
            return "locked" if rng.random() < 0.7 else "unlocked"
        if cls == "alarm":
            return rng.choices(["disarmed", "armed_home", "armed_away"], weights=[0.7, 0.2, 0.1])[0]
        if cls == "camera":
            return "idle"
        if cls == "vacuum":
            return "docked"
        if cls == "climate":
            attrs.setdefault("temperature", 21.0)
            attrs.setdefault("current_temperature", 20.0)
            return "off"
        if cls == "media_player":
            return "off"
        if cls == "garage":
            closed = rng.random() < 0.8
            attrs["position"] = 0 if closed else 100
            return "closed" if closed else "open"
        if cls == "window" or cls == "door":
            return "on" if rng.random() < (0.05 if cls == "window" else 0.06) else "off"
        if cls == "water" and kind == "level":
            return str(rng.randint(15, 100))
        if cls == "water" and kind == "flow":
            return "0"
        if cls == "water" and kind == "pressure":
            return f"{rng.uniform(2.5, 4.5):.1f}"
        if domain == "switch" and cls == "device":
            return rng.choice(["on", "off"])
        if domain == "light" and cls == "device":
            return "off"
        if domain in ("remote", "device_tracker", "lawn_mower", "notify", "image", "media_player") and cls == "device":
            return {"remote": "on", "device_tracker": "home", "lawn_mower": "docked", "notify": "unknown", "image": "unknown",
                    "media_player": "idle"}[domain]
        if cls == "humidity":
            return str(rng.randint(35, 72))
        if cls in ("light", "switch", "fan", "appliance", "irrigation", "pump"):
            return "off"
        if cls == "cover":
            closed = rng.random() < 0.5
            attrs["position"] = 0 if closed else 100
            return "closed" if closed else "open"
        if cls == "temperature":
            return "20"
        if attrs.get("unit") is not None:
            return f"{rng.uniform(1, 100):.1f}"
        if domain == "sensor":
            return rng.choice(["ok", "normal", "idle", "online", "12", "3", "0"])
        return "off"


def pick_names(rng: random.Random, pool: list[str], k: int, taken: set[str]) -> list[str]:
    """k names from pool not yet used in the room; numbered fallbacks when the pool runs out."""
    free = [n for n in pool if n not in taken]
    rng.shuffle(free)
    out = free[:k]
    i = 2
    while len(out) < k:
        base = pool[len(out) % len(pool)]
        cand = f"{base} {i}"
        if cand not in taken and cand not in out:
            out.append(cand)
        i += 1
    taken.update(out)
    return out


# ---------------------------------------------------------------- derived states
DAYLIGHT_LUX = {"sunny": 40000, "partlycloudy": 20000, "cloudy": 8000, "rainy": 3000, "clear-night": 0}


def finish_devices(gen, house: House):
    """After Gen.make(): make light groups agree with their bulbs, switch off most lights in empty rooms
    (big houses run occupancy automations), and derive every device's power, energy, current, lux and
    outdoor temperature readings from the states the base generator settled on."""
    rng = gen.rng
    occupied = set(house.occupied_areas()) | {p.room for p in house.people if p.home and p.room}
    outdoor = {a.area_id for a in house.areas if a.kind == "outdoor"}
    for e in house.entities.values():
        if e.cls == "light" and e.state == "on" and e.area not in occupied and e.area not in outdoor and e.area in house.area_ids() and rng.random() < 0.75:
            e.state = "off"
            e.attributes.pop("brightness_pct", None)
    for e in list(house.entities.values()):
        members = e.attributes.get("entity_id") if e.cls == "light" else None
        if members:
            for mid in members:
                m = house.entities[mid]
                m.state, m.changed_s = e.state, e.changed_s
                if e.state == "on":
                    m.attributes["brightness_pct"] = e.attributes.get("brightness_pct", 100)
                else:
                    m.attributes.pop("brightness_pct", None)
    w = house.entities.get("weather.home")
    out_t = float(w.attributes["temperature"]) if w else 15.0
    sun_up = house.entities.get("sun.sun") is not None and house.entities["sun.sun"].state == "above_horizon"
    day_lux = DAYLIGHT_LUX.get(w.state if w else "cloudy", 5000) if sun_up else 0
    lit = {e.area for e in house.entities.values() if e.cls == "light" and e.state == "on"}
    circuits: dict[str, float] = {}
    for dev in gen.devices:
        main = dev["main"]
        draw = _draw(dev, main, rng)
        if dev["circuit"]:
            circuits[dev["circuit"]] = circuits.get(dev["circuit"], 0.0) + draw
        _fill_readings(dev, draw, rng)
    for dev in gen.devices:
        if dev.get("meters"):  # circuit meter channel: what everything on the circuit draws plus unmetered sockets
            draw = circuits.get(dev["meters"], 0.0) + rng.uniform(*dev.get("base_w", (3, 60)))
            _fill_readings(dev, draw, rng)
    for e in house.entities.values():
        if e.cls == "illuminance":
            if e.area in outdoor or e.area == "weather":
                e.state = str(int(day_lux * rng.uniform(0.6, 1.2)))
            else:
                e.state = str(int((rng.uniform(120, 450) if e.area in lit else 0) + day_lux * rng.uniform(0.002, 0.01)))
            e.changed_s = rng.randint(20, 900)
        elif e.cls == "temperature" and (e.area in outdoor or e.area == "weather"):
            e.state = f"{out_t + rng.uniform(-0.8, 0.8):.1f}"
        elif e.cls == "air_quality" and e.attributes.get("unit") == "ppm":
            e.state = str(int(rng.uniform(650, 1400) if e.area in occupied else rng.uniform(420, 700)))
        elif e.cls == "air_quality" and e.attributes.get("unit") == "ppb":
            e.state = str(int(rng.uniform(40, 600)))


def _draw(dev: dict, main: Optional[Entity], rng) -> float:
    """What the device's load is drawing right now, in watts."""
    if main is None or not dev["load_w"]:
        return 0.0
    a = main.attributes
    if main.cls == "light":
        return dev["load_w"] * a.get("brightness_pct", 100) / 100 if main.state == "on" else 0.0
    if main.cls == "appliance":
        st = a.get("status", main.state)
        return float(a.get("power_w", dev["load_w"])) * (1.0 if st == "running" else 0.35 if st == "on" else 0.0)
    if main.state in ("on", "heat", "cool", "playing", "cleaning"):
        if a.get("power_w"):
            return float(a["power_w"])
        return dev["load_w"] * (rng.uniform(0.05, 1.0) if main.cls == "switch" else 1.0)
    return 0.0


def _fill_readings(dev: dict, draw: float, rng):
    standby = rng.uniform(0.2, 1.5)
    for e in dev["ents"]:
        if e.attributes.get("entity_category") or e is dev["main"]:
            continue
        sfx = e.name[len(dev["name"]):].strip() if e.name.startswith(dev["name"]) else e.name
        if e.cls == "power":
            e.state, e.changed_s = f"{draw if draw else standby:.0f}", rng.randint(5, 120)
        elif e.cls == "meter" and e.attributes.get("unit") == "kWh":
            per_day = max(draw, dev["load_w"] * 0.15, 1.0) * rng.uniform(2, 8) / 1000
            e.state = f"{per_day * (1 if 'Today' in sfx else rng.uniform(30, 400)):.1f}"
        elif sfx == "Current":
            e.state = f"{draw / 240:.2f}"
        elif sfx == "Voltage":
            e.state = f"{rng.uniform(236, 248):.0f}"
        elif sfx == "Power Factor":
            e.state = f"{rng.uniform(0.85, 0.99) if draw > 5 else 0:.2f}"
