"""Bridge checks without Home Assistant or Ollama: HA payload -> House mapping, and the
agent loop driven by a scripted fake model."""
import json, sys
from datetime import datetime, timedelta, timezone
sys.path.insert(0, ".")
from homeai.bridge.ha import build_house
from homeai.bridge.agent import Agent
from homeai.house.snapshot import system_prompt
from homeai.house.schema import Speaker
from homeai.tools import execute_tool, get_state_text
from homeai.house.generator import make_house

now = datetime.now(timezone.utc)
ago = lambda s: (now - timedelta(seconds=s)).isoformat()
states = [
    {"entity_id": "light.office_ceiling", "state": "on", "attributes": {"friendly_name": "Office Ceiling", "brightness": 204}, "last_changed": ago(3600)},
    {"entity_id": "light.office_desk_lamp", "state": "off", "attributes": {"friendly_name": "Office Desk Lamp"}, "last_changed": ago(300)},
    {"entity_id": "binary_sensor.office_motion", "state": "on", "attributes": {"friendly_name": "Office Motion", "device_class": "motion"}, "last_changed": ago(30)},
    {"entity_id": "binary_sensor.bedroom_mmwave", "state": "off", "attributes": {"friendly_name": "Bedroom Presence", "device_class": "occupancy"}, "last_changed": ago(7200)},
    {"entity_id": "sensor.bedroom_temp", "state": "19.4", "attributes": {"friendly_name": "Bedroom Temperature", "device_class": "temperature"}, "last_changed": ago(60)},
    {"entity_id": "lock.front_door", "state": "unlocked", "attributes": {"friendly_name": "Front Door"}, "last_changed": ago(900)},
    {"entity_id": "binary_sensor.front_door_contact", "state": "off", "attributes": {"friendly_name": "Front Door", "device_class": "door"}, "last_changed": ago(900)},
    {"entity_id": "cover.garage_door", "state": "closed", "attributes": {"friendly_name": "Garage Door", "device_class": "garage", "current_position": 0}, "last_changed": ago(9000)},
    {"entity_id": "person.peter", "state": "home", "attributes": {"friendly_name": "Peter"}, "last_changed": ago(20000)},
    {"entity_id": "person.amber", "state": "not_home", "attributes": {"friendly_name": "Amber"}, "last_changed": ago(2700)},
    {"entity_id": "sensor.victron_battery_soc", "state": "62", "attributes": {"friendly_name": "Battery SOC", "device_class": "battery"}, "last_changed": ago(10)},
    {"entity_id": "sensor.victron_battery_power", "state": "-1250", "attributes": {"friendly_name": "Battery Power", "device_class": "power"}, "last_changed": ago(10)},
    {"entity_id": "sensor.victron_pv_power", "state": "0", "attributes": {"friendly_name": "PV Power", "device_class": "power"}, "last_changed": ago(10)},
    {"entity_id": "sensor.victron_ac_loads", "state": "1250", "attributes": {"friendly_name": "AC Loads", "device_class": "power"}, "last_changed": ago(10)},
    {"entity_id": "sensor.solcast_forecast_tomorrow", "state": "18.2", "attributes": {"friendly_name": "Solcast Forecast Tomorrow", "device_class": "energy"}, "last_changed": ago(1000)},
    {"entity_id": "sensor.phone_battery_level", "state": "80", "attributes": {"friendly_name": "Peter's Phone Battery Level", "device_class": "battery"}, "last_changed": ago(10)},
    {"entity_id": "switch.hot_water", "state": "on", "attributes": {"friendly_name": "Hot Water"}, "last_changed": ago(1200)},
    {"entity_id": "sensor.hot_water_power", "state": "2980", "attributes": {"friendly_name": "Hot Water Power", "device_class": "power"}, "last_changed": ago(10)},
    {"entity_id": "weather.home", "state": "partlycloudy", "attributes": {"friendly_name": "Home", "temperature": 14.2, "humidity": 60, "wind_speed": 12,
        "forecast": [{"condition": "partlycloudy", "temperature": 20, "templow": 9}, {"condition": "sunny", "temperature": 23, "templow": 10}]}, "last_changed": ago(600)},
    {"entity_id": "sun.sun", "state": "below_horizon", "attributes": {"next_rising": (now + timedelta(hours=9)).isoformat(), "next_setting": (now + timedelta(hours=20)).isoformat()}, "last_changed": ago(3000)},
    {"entity_id": "update.core", "state": "off", "attributes": {}, "last_changed": ago(1)},
    {"entity_id": "light.unassigned", "state": "on", "attributes": {"friendly_name": "Mystery Light"}, "last_changed": ago(1)},
]
areas = {"office": {"name": "Office", "entities": ["light.office_ceiling", "light.office_desk_lamp", "binary_sensor.office_motion"]},
         "bedroom": {"name": "Bedroom", "entities": ["binary_sensor.bedroom_mmwave", "sensor.bedroom_temp"]},
         "energy": {"name": "Energy", "entities": ["switch.hot_water"]}}
cfg = {"battery_capacity_kwh": 14, "overrides": {"switch.hot_water": {"power_sensor": "sensor.hot_water_power", "area": "energy"}}}
h = build_house(states, areas, cfg, now)
sp = Speaker(name="Peter", area="office", channel="satellite")
prompt = system_prompt(h, sp)
print(prompt)
print()
print(get_state_text(h, ["energy", "weather"]))
print(get_state_text(h, [], ["motion", "occupancy"]))
assert "sensor.phone_battery_level" not in h.entities, "phone battery must be filtered"
assert "update.core" not in h.entities and "light.unassigned" not in h.entities
assert h.entities["sensor.victron_battery_soc"].cls == "battery" and h.entities["sensor.victron_battery_soc"].attributes["kind"] == "soc"
assert h.entities["sensor.victron_battery_power"].attributes["direction"] == "discharging"
assert h.entities["sensor.victron_ac_loads"].cls == "load"
assert h.entities["sensor.solcast_forecast_tomorrow"].cls == "forecast"
assert h.entities["switch.hot_water"].attributes["power_w"] == 2980
assert h.entities["lock.front_door"].area == "security" and h.entities["cover.garage_door"].area == "security"
assert [p.name for p in h.people] == ["Peter", "Amber"] and not h.person("Amber").home
assert "Office [office]: light(Ceiling, Desk Lamp); sensors: motion" in prompt

# --- agent loop with a scripted fake model against the mock simulator
class FakeLLM:
    def __init__(self): self.n = 0
    def chat(self, messages, tools=None):
        self.n += 1
        last = messages[-1]
        if self.n == 1:
            assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
            return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_state", "arguments": {"areas": [LIGHT_AREA], "classes": ["light"]}}}]}
        if self.n == 2:
            assert last["role"] == "tool" and LIGHT_ID in last["content"], last
            return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "call_service", "arguments": {"domain": "light", "service": "turn_off", "target": {"entity_id": [LIGHT_ID]}}}}]}
        assert json.loads(last["content"])["success"]
        return {"role": "assistant", "content": "Office ceiling is off."}

mock = make_house(11)
light = next(e for e in mock.by_class("light"))
LIGHT_AREA, LIGHT_ID = light.area, light.entity_id
agent = Agent(FakeLLM(), lambda: mock, execute_tool)
reply = agent.chat("turn off the ceiling light", Speaker("Peter", LIGHT_AREA, "satellite"))
assert reply == "Office ceiling is off." and light.state == "off", (reply, light.state)
assert [m["role"] for m in agent.history] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
print("\nbridge OK")
