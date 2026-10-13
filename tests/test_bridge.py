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

# --- prompt layouts (homeai/house/snapshot.py relayout; v10 on trains "stable_prefix")
import collections, os, tempfile
from homeai import prompt_layout
from homeai.house.snapshot import CONTEXT_HEAD, relayout, system_prompt_b
sp_prompt = system_prompt(h, sp, "stable_prefix")
assert sp_prompt.split("\n")[:2] == prompt.split("\n")[:2] and sp_prompt.split("\n")[-4] == CONTEXT_HEAD
assert sp_prompt.split("\n")[-3].startswith("Now: ") and sp_prompt.split("\n")[-1].startswith("People: ")
assert collections.Counter(sp_prompt.split("\n")) - collections.Counter(prompt.split("\n")) == collections.Counter([CONTEXT_HEAD])
assert relayout(sp_prompt, "classic") == prompt and relayout(prompt, "stable_prefix") == sp_prompt == relayout(sp_prompt, "stable_prefix")
assert relayout(system_prompt_b(h, sp), "stable_prefix") == system_prompt_b(h, sp, "stable_prefix")
later = system_prompt(h, Speaker("Amber", None, "app"), "stable_prefix")
assert len(os.path.commonprefix([sp_prompt, later])) > sp_prompt.index(CONTEXT_HEAD)  # a new speaker: only the block differs
assert len(os.path.commonprefix([prompt, system_prompt(h, Speaker("Amber", None, "app"))])) < prompt.index("Areas")  # classic
ex = {"id": "x", "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": "hi"}]}
assert prompt_layout.apply(ex, "stable_prefix")["messages"][0]["content"] == sp_prompt and prompt_layout.apply(ex, "classic") is ex
from homeai.house.snapshot import relayout_messages
conv = ex["messages"] + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "lights off"}]
for x in prompt_layout.LAYOUTS:
    for y in prompt_layout.LAYOUTS:  # every layout converts to every other and back
        assert relayout_messages(relayout_messages(conv, x), y) == relayout_messages(conv, y), (x, y)
uc_conv = relayout_messages(conv, "user_context")
assert uc_conv[0]["content"] == sp_prompt[:sp_prompt.index("\n\n" + CONTEXT_HEAD)] and relayout_messages(uc_conv, "classic") == conv
assert [m["content"].split("\n")[1] for m in uc_conv if m["role"] == "user"] == ["hi", "lights off"]
# clock_in_user (v12): classic without the clock line, every user turn starts with "[now] <clock>"
from homeai.house.snapshot import CLOCK_TAG, split_context, split_user
now_line = next(l for l in prompt.split("\n") if l.startswith("Now: "))
cu_conv = relayout_messages(conv, "clock_in_user")
assert cu_conv[0]["content"] == prompt.replace(now_line + "\n", "") == relayout(prompt, "clock_in_user")
assert cu_conv[0]["content"].split("\n")[2].startswith("Speaker: ") and "Now: " not in cu_conv[0]["content"]
assert [m["content"] for m in cu_conv if m["role"] == "user"] == [CLOCK_TAG + now_line[5:] + "\n" + u for u in ("hi", "lights off")]
assert relayout_messages(cu_conv, "clock_in_user") == cu_conv and relayout_messages(cu_conv, "classic") == conv
assert relayout_messages(cu_conv, "user_context") == uc_conv and relayout_messages(uc_conv, "clock_in_user") == cu_conv
assert relayout(cu_conv[0]["content"], "clock_in_user") == cu_conv[0]["content"]
assert split_user("[now] is it 9 yet") == (None, "[now] is it 9 yet") and split_user("[now] \nhi")[0] is None  # no clock
# per-turn contexts (as the bridge records them): each turn keeps its own clock, Speaker / People are the latest turn's
ctx1 = split_context(prompt)[1]
ctx2 = ["Now: Friday 23:59", "Speaker: Amber, via the app (home)", ctx1[2]]
turns = [{**conv[1], "_context": ctx1}, conv[2], {**conv[3], "_context": ctx2}]
cu2 = relayout_messages([conv[0]] + turns, "clock_in_user")
assert [m["content"] for m in cu2 if m["role"] == "user"] == [CLOCK_TAG + now_line[5:] + "\nhi", CLOCK_TAG + "Friday 23:59\nlights off"]
assert "\nSpeaker: Amber, via the app (home)\n" in cu2[0]["content"] and all("_context" not in m for m in cu2)
assert relayout_messages(cu2, "classic")[0]["content"] == relayout_messages([conv[0]] + turns, "classic")[0]["content"]
assert relayout_messages(relayout_messages([conv[0]] + turns, "user_context"), "clock_in_user") == cu2
with tempfile.TemporaryDirectory() as t:
    json.dump({"prompt_layout": "stable_prefix"}, open(t + "/cfg.json", "w"))
    json.dump({"m-new": {"config": t + "/cfg.json"}, "m-old": {"config": None}}, open(t + "/models.json", "w"))
    assert prompt_layout.for_model("m-new:latest", registry=t + "/models.json") == "stable_prefix"
    assert prompt_layout.for_model("m-old", registry=t + "/models.json") == "classic"
    assert prompt_layout.for_model("m-new", "classic", registry=t + "/models.json") == "classic"


class Recorder(FakeLLM):
    def __init__(self): super().__init__(); self.systems = []
    def chat(self, messages, tools=None):
        self.systems.append(messages[0]["content"])
        return super().chat(messages, tools)


light.state = "on"
rec = Recorder()
Agent(rec, lambda: mock, execute_tool, layout="stable_prefix").chat("turn off the ceiling light", Speaker("Peter", LIGHT_AREA, "satellite"))
assert len(rec.systems) == 3 and len(set(rec.systems)) == 1 and CONTEXT_HEAD in rec.systems[0]  # built once per request

# user_context (v11): a static system message; the latest user turn gets the current context line, earlier turns keep
# the one they were sent with, and the history itself keeps bare utterances
import datetime
from homeai.house.snapshot import CONTEXT_TAG, split_context


class Seen(FakeLLM):
    def __init__(self): super().__init__(); self.calls = []
    def chat(self, messages, tools=None):
        self.calls.append(messages)
        if self.n >= 3:
            return {"role": "assistant", "content": "It's warm enough."}
        return super().chat(messages, tools)


light.state = "on"
seen = Seen()
uc = Agent(seen, lambda: mock, execute_tool, layout="user_context")
uc.chat("turn off the ceiling light", Speaker("Peter", LIGHT_AREA, "satellite"))
first = [m["content"] for m in seen.calls[0] if m["role"] == "user"]
mock.now += datetime.timedelta(minutes=7)
uc.chat("is it warm in here", Speaker("Amber", None, "app"))
last = seen.calls[-1]
users = [m["content"] for m in last if m["role"] == "user"]
assert "Now: " not in last[0]["content"] and split_context(last[0]["content"])[1] == []
assert users[0] == first[0] and users[0].startswith(CONTEXT_TAG) and "Speaker: Peter" in users[0]  # sent as before
assert users[1].startswith(CONTEXT_TAG) and "Speaker: Amber" in users[1] and users[1].endswith("\nis it warm in here")
assert users[0].split(" | ")[0] != users[1].split(" | ")[0]  # the clock moved on for the new turn only
assert [m["content"] for m in uc.history if m["role"] == "user"] == ["turn off the ceiling light", "is it warm in here"]
assert all("_context" not in m for c in seen.calls for m in c)  # nothing internal goes to the server

# clock_in_user (v12): only the clock moves into the user turn. From the same satellite with nobody coming or going the
# system message is byte-identical across requests (the cache resumes after it); earlier turns are re-sent byte for byte
light.state = "on"
seen = Seen()
cu = Agent(seen, lambda: mock, execute_tool, layout="clock_in_user")
here = Speaker("Peter", LIGHT_AREA, "satellite")
cu.chat("turn off the ceiling light", here)
first = seen.calls[0]
mock.now += datetime.timedelta(minutes=7)
n = len(seen.calls)
cu.chat("is it warm in here", here)
second, prev = seen.calls[n], seen.calls[n - 1]
assert second[:len(prev)] == prev  # the whole last call of the first request, byte for byte, then its answer and the new turn
assert second[0] == first[0] and "Now: " not in first[0]["content"] and "\nSpeaker: Peter, from " in first[0]["content"]
assert len({c[0]["content"] for c in seen.calls}) == 1  # every round of both requests
users = [m["content"] for m in second if m["role"] == "user"]
assert users[0] == first[1]["content"] and users[0].startswith(CLOCK_TAG) and users[0].endswith("\nturn off the ceiling light")
assert users[1].startswith(CLOCK_TAG) and users[1].endswith("\nis it warm in here") and users[0].split("\n")[0] != users[1].split("\n")[0]
mock.now += datetime.timedelta(minutes=3)
n = len(seen.calls)
cu.chat("and in the kitchen", Speaker("Amber", None, "app"))  # another speaker: the system message follows them
third, prev = seen.calls[n], seen.calls[n - 1]
assert "\nSpeaker: Amber, via the app" in third[0]["content"] and third[0] != first[0] and third[1:len(prev)] == prev[1:]
assert [m["content"] for m in third if m["role"] == "user"][:2] == users  # earlier turns still byte-identical
assert [m["content"] for m in cu.history if m["role"] == "user"] == ["turn off the ceiling light", "is it warm in here", "and in the kitchen"]
assert all("_context" not in m for c in seen.calls for m in c)
print("\nbridge OK")
