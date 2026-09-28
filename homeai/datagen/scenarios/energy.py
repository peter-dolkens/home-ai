"""Off-grid power management: battery, solar, load, forecasts, weather."""
from __future__ import annotations

import random
from typing import Optional

from ...house.schema import House, Speaker
from ...house.snapshot import num
from ..common import Conv, cap, is_are, join, pick, voiceify

COND_WORDS = {"sunny": "sunny", "partlycloudy": "partly cloudy", "cloudy": "overcast", "rainy": "raining", "clear-night": "clear"}
COND_TOMORROW = {"sunny": "sunny", "partlycloudy": "partly cloudy", "cloudy": "cloudy", "rainy": "rainy"}


class EnergyView:
    """Numbers pulled from the house in the same rounding the model sees."""

    def __init__(self, house: House):
        g = lambda eid: house.entities.get(eid)
        soc = g("sensor.battery_soc")
        self.soc = float(soc.state)
        self.cap = float(soc.attributes["capacity_kwh"])
        self.remaining = float(g("sensor.battery_energy_remaining").state)
        bp = g("sensor.battery_power")
        self.bat_w = float(bp.state)
        self.direction = bp.attributes["direction"]
        self.solar_w = float(g("sensor.solar_power").state)
        self.load_w = float(g("sensor.house_load").state)
        self.energy_today = float(g("sensor.solar_energy_today").state)
        self.fc_rem = float(g("sensor.solar_forecast_remaining_today").state)
        self.fc_tom = float(g("sensor.solar_forecast_tomorrow").state)
        rt = g("sensor.battery_runtime").state
        self.runtime = None if rt.startswith("n/a") else float(rt)
        sun = g("sun.sun")
        self.sun_up = sun.state == "above_horizon"
        self.rise = sun.attributes["next_rising"]
        self.sset = sun.attributes["next_setting"]
        rh, rm = map(int, self.rise.split(":"))
        now_m = house.now.hour * 60 + house.now.minute
        rise_m = rh * 60 + rm
        self.h_to_rise = ((rise_m - now_m) % 1440) / 60
        w = g("weather.home")
        self.cond = w.state
        self.temp = float(w.attributes["temperature"])
        self.wind = w.attributes["wind_kmh"]
        self.humidity = w.attributes["humidity"]
        self.tom = w.attributes["forecast_tomorrow"]
        self.sheddable = [e for e in house.entities.values() if e.cls == "switch" and e.state == "on" and e.attributes.get("power_w")]
        self.generator = house.entities.get("switch.generator")
        self.shark = house.entities.get("sensor.shark_battery")
        self.shark_feed = house.entities.get("switch.shark_backup")

    def backup_option(self):
        """(sentence, fix_call or None) describing the vehicle/generator backup, if any."""
        if self.shark is not None and self.shark_feed is not None:
            loc = self.shark.attributes.get("location", "")
            plugged = self.shark.attributes.get("plugged")
            soc = num(self.shark.state, 0)
            if self.shark_feed.state == "on":
                return (f"The Shark is already feeding the house at {num(self.shark_feed.attributes.get('source_w', 3000), 0)} watts.", None)
            if loc.startswith("parked here") and plugged and float(self.shark.state) >= 30:
                return (f"The Shark is plugged in at {soc} percent, so I can switch its backup feed on to cover the gap.",
                        {"domain": "switch", "service": "turn_on", "target": {"entity_id": [self.shark_feed.entity_id]}})
            if loc.startswith("parked here") and float(self.shark.state) >= 30:
                return (f"The Shark is here at {soc} percent but not plugged in; plug it in and I can bring the backup feed on.", None)
            if loc.startswith("at the old house"):
                return (f"The Shark is at the old house, {soc} percent, so bringing it over would cover the gap.", None)
            return ("The Shark isn't here, so no backup feed tonight.", None)
        if self.generator is not None:
            return ("The generator is available if it comes to that.", None)
        return (None, None)

    def flow_sentence(self, rng: random.Random) -> str:
        sw, lw = num(self.solar_w, 0), num(self.load_w, 0)
        if self.direction == "charging":
            return pick(rng, ["Solar's making {s} watts against {l} watts of load, so it's charging at about {c} watts.",
                              "We're charging at {c} watts, {s} watts of solar in and {l} out."], s=sw, l=lw, c=num(self.bat_w, 0))
        if self.direction == "discharging":
            d = num(-self.bat_w, 0)
            rt = num(self.runtime) if self.runtime is not None else "?"
            if self.solar_w < 50:
                return pick(rng, ["Solar's done for the day, the load is {l} watts, so we're drawing {d} watts from the battery. That's roughly {r} hours at this rate.",
                                  "No solar right now and {l} watts of load, so the battery's covering {d} watts, about {r} hours' worth."], l=lw, d=d, r=rt)
            return pick(rng, ["Solar's only {s} watts and the load is {l}, so we're pulling {d} watts from the battery, roughly {r} hours at this rate.",
                              "We're using {l} watts with {s} coming from solar, so the battery is covering {d} watts, about {r} hours' worth."], s=sw, l=lw, d=d, r=rt)
        if self.soc >= 99:
            return pick(rng, ["The battery's full and solar is covering the {l} watt load.", "Battery's full, solar's carrying the {l} watts of load on its own."], l=lw)
        return f"Solar and load are balanced at about {lw} watts, so the battery's holding."


def energy_status(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    v = EnergyView(house)
    focus = rng.choice(["battery", "battery", "usage", "solar", "overall", "solar_tomorrow"])
    c = Conv(house, sp, "energy_status", rng)
    soc, rem = num(v.soc, 0), num(v.remaining)
    if focus == "battery":
        c.user(voiceify(rng.choice(["How's the battery?", "What's the battery at?", "How much battery do we have?", "What's our state of charge?", "Battery level?", "How much charge is left?"]), rng, "q"))
        c.fetch(areas=["energy"])
        c.say(pick(rng, ["Battery's at {p} percent, about {r} kilowatt hours. ", "{p} percent, which is {r} kilowatt hours. "], p=soc, r=rem) + v.flow_sentence(rng), must=[soc])
    elif focus == "usage":
        c.user(voiceify(rng.choice(["How much power are we using?", "What's the house drawing right now?", "What's our load at the moment?", "How much are we pulling?", "What's using power right now?"]), rng, "q"))
        big = max(v.sheddable, key=lambda e: e.attributes["power_w"]) if v.sheddable else None
        c.fetch(areas=["energy"], b=[{"areas": ["energy"]}, {"classes": ["switch"], "where": ["power>0"], "sort": "power_desc", "limit": 5}],
                need=[big] if big else [])
        text = pick(rng, ["The house is drawing {l} watts right now. ", "About {l} watts at the moment. "], l=num(v.load_w, 0))
        if v.sheddable:
            text += pick(rng, ["The {n} is the big one at {w} watts. ", "Most of that is the {n}, {w} watts. "], n=big.name.lower(), w=big.attributes["power_w"])
        text += v.flow_sentence(rng)
        c.say(text, must=[num(v.load_w, 0)])
    elif focus == "solar_tomorrow":
        # "sun tomorrow" means the solar forecast (kWh), not the weather description
        c.user(voiceify(rng.choice(["Will we get much sun tomorrow?", "Is tomorrow a good solar day?", "How's the solar looking for tomorrow?",
                                    "Are we getting better sun tomorrow?", "What's the solar forecast for tomorrow?", "Much sun coming tomorrow?",
                                    "Is tomorrow going to be any good for the panels?", "How much solar tomorrow?"]), rng, "q"))
        c.fetch(areas=["energy"])
        today = v.energy_today + v.fc_rem
        cmp = "better than" if v.fc_tom > today * 1.1 else "less than" if v.fc_tom < today * 0.9 else "about the same as"
        c.say(pick(rng, ["Tomorrow's forecast is {f} kilowatt hours, {c} today's {t}.", "About {f} kilowatt hours forecast tomorrow, {c} the {t} we'll get today.",
                         "{F} kilowatt hours tomorrow, which is {c} today at {t}."], f=num(v.fc_tom), F=num(v.fc_tom), c=cmp, t=num(today)), must=[num(v.fc_tom)])
    elif focus == "solar":
        c.user(voiceify(rng.choice(["How much solar are we making?", "How's the solar looking?", "What's the solar doing?", "How much solar have we made today?", "Is the solar producing?"]), rng, "q"))
        c.fetch(areas=["energy"])
        if v.solar_w < 50:
            c.say(pick(rng, ["Solar's finished for the day. We made {t} kilowatt hours today, and tomorrow's forecast is {f}.",
                             "Nothing coming in right now, it's dark. Today totalled {t} kilowatt hours; tomorrow should bring about {f}."], t=num(v.energy_today), f=num(v.fc_tom)), must=[num(v.energy_today)])
        else:
            q = c.msgs[1]["content"].lower()
            c.say(pick(rng, ["Solar's at {s} watts, {t} kilowatt hours so far today, with about {f} more expected before sunset.",
                             "{s} watts right now. That's {t} kilowatt hours today, and the forecast has another {f} to come."], s=num(v.solar_w, 0), t=num(v.energy_today), f=num(v.fc_rem)),
                  must=[num(v.energy_today) if "today" in q or "made" in q else f"{num(v.solar_w, 0)}|{num(v.energy_today)}"])
    else:
        c.user(voiceify(rng.choice(["What's our power situation?", "Give me an energy update.", "How are we doing for power?", "Power report please.", "How's the system looking?"]), rng, "q"))
        c.fetch(areas=["energy"])
        text = pick(rng, ["Battery's at {p} percent, {r} kilowatt hours. ", "{p} percent on the battery, about {r} kilowatt hours. "], p=soc, r=rem) + v.flow_sentence(rng)
        if v.solar_w < 50:
            text += f" Tomorrow's forecast is {num(v.fc_tom)} kilowatt hours of solar."
        else:
            text += f" About {num(v.fc_rem)} kilowatt hours of solar still to come today."
        c.say(text, must=[soc])
    return c


def energy_night(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    v = EnergyView(house)
    c = Conv(house, sp, "energy_night", rng)
    c.user(voiceify(rng.choice(["Will we make it through the night?", "Will the battery last till morning?", "Have we got enough power for tonight?",
                                "Are we going to run out of power overnight?", "Is the battery going to be okay overnight?", "Are we good for the night?"]), rng, "q"))
    c.fetch(areas=["energy", "weather"], b=[{"areas": ["energy"]}, {"classes": ["sun"]}, {"classes": ["switch"], "where": ["power>0"], "sort": "power_desc"}],
            need=v.sheddable)
    soc, rem = num(v.soc, 0), num(v.remaining)
    if v.sun_up and v.direction != "discharging":
        c.say(pick(rng, ["It's still daylight and the battery's at {p} percent and charging, with about {f} kilowatt hours of solar still to come today. You should go into the night close to full. Ask me again at sunset for a proper number.",
                         "Too early to say precisely. We're at {p} percent, charging, and the forecast has {f} kilowatt hours more today, so it's looking fine. Check with me after dark."],
                   p=soc, f=num(v.fc_rem)), must=[soc])
        return c
    if v.direction != "discharging":
        c.say(f"Battery's at {soc} percent and holding, nothing's being drawn from it right now, so yes.", must=[soc])
        return c
    need = v.h_to_rise + 1.0
    rt = v.runtime or 0.0
    d = num(-v.bat_w, 0)
    hrs = num(v.h_to_rise)
    shed = ""
    fixes = []
    if v.sheddable:
        saved = sum(e.attributes["power_w"] for e in v.sheddable)
        names = join([e.name.lower() for e in v.sheddable])
        new_draw = -v.bat_w - saved
        fixes = [{"domain": "switch", "service": "turn_off", "target": {"entity_id": [e.entity_id for e in v.sheddable]}}]
        if new_draw <= 50:
            shed = f" Turning off the {names} would stop the drain almost entirely."
        else:
            new_rt = v.remaining / (new_draw / 1000)
            if new_rt >= need:
                shed = f" Turning off the {names} would drop the draw to about {num(new_draw, 0)} watts and get you through to morning."
            else:
                shed = f" Turning off the {names} would stretch it to about {num(new_rt)} hours, still short" + (", so the generator may be needed." if v.generator else ".")
    if rt >= need * 1.15:
        c.say(pick(rng, ["Yes. Battery's at {p} percent, {r} kilowatt hours, drawing {d} watts. That's about {rt} hours and sunrise is at {rise}, {h} hours away.",
                         "Comfortably. {rt} hours at the current {d} watts, and the sun's up at {rise}, in {h} hours."],
                   p=soc, r=rem, d=d, rt=num(rt), rise=v.rise, h=hrs), must=["yes|comfortably"])
    elif rt >= need:
        text = pick(rng, ["Just. About {rt} hours at the current {d} watts, and sunrise is {h} hours away at {rise}, so it's tight.",
                          "It'll be close. {p} percent gives about {rt} hours at {d} watts, with sunrise {h} hours off."], rt=num(rt), d=d, h=hrs, rise=v.rise, p=soc) + shed
        if fixes:
            text += " Want me to do that?"
        c.say(text, must=["tight|close"])
        if fixes:
            c.user(rng.choice(["Yes please.", "Yes.", "Do it."]))
            c.act(fixes)
            c.say(f"Done, {join([e.name.lower() for e in v.sheddable])} off.")
    else:
        text = pick(rng, ["Probably not. At {d} watts you've got about {rt} hours, but sunrise isn't until {rise}, {h} hours away.",
                          "No, not at this rate. {p} percent is roughly {rt} hours at {d} watts, and the sun's not up until {rise}."], d=d, rt=num(rt), rise=v.rise, h=hrs, p=soc) + shed
        backup_sentence, backup_fix = v.backup_option()
        if not v.sheddable:
            text += " There's nothing big I can switch off."
        if backup_sentence:
            text += " " + backup_sentence
        elif not v.sheddable:
            text += " You might want to shed some load manually."
        if backup_fix:
            fixes.append(backup_fix)
        asks = []
        if v.sheddable:
            asks.append("turn those off")
        if backup_fix:
            asks.append("bring the Shark feed on")
        if asks:
            text += f" Shall I {join(asks)}?"
        c.say(text, must=["not|no"])
        if fixes:
            if rng.random() < 0.8:
                c.user(rng.choice(["Yes please.", "Yes.", "Do it.", "Go ahead."]))
                c.act(fixes)
                done = []
                if v.sheddable:
                    done.append(f"{join([e.name.lower() for e in v.sheddable])} off")
                if backup_fix:
                    done.append("Shark backup feed on")
                c.say(f"Done, {join(done)}." + (" I'll leave the rest to you." if v.sheddable else ""))
            else:
                c.user(rng.choice(["No, leave it.", "Not yet."]))
                c.say("Okay.")
    return c


APPLIANCES = [("the dryer", 2.0), ("the dishwasher", 1.2), ("a load of washing", 0.8), ("the washing machine", 0.8), ("the oven", 1.5), ("the kettle", 0.1), ("the pool pump", 1.0)]


def energy_appliance(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    v = EnergyView(house)
    app, kwh = rng.choice(APPLIANCES)
    c = Conv(house, sp, "energy_appliance", rng)
    c.user(voiceify(pick(rng, ["Can I run {a} now?", "Is it a good time to run {a}?", "Should I put {a} on?", "Is there enough power to run {a}?", "Can I put {a} on or should I wait?", "Okay to use {a}?"], a=app), rng, "q"))
    c.fetch(areas=["energy"], b=[{"areas": ["energy"]}, {"classes": ["weather"]}])
    surplus = v.solar_w - v.load_w
    soc = num(v.soc, 0)
    pctb = num(kwh / v.cap * 100, 0)
    if surplus >= kwh * 500:
        c.say(pick(rng, ["Go for it. Solar's making {s} watts and we're only using {l}, so there's {x} watts spare.",
                         "Yes, now's a good time. {x} watts of solar surplus at the moment."], s=num(v.solar_w, 0), l=num(v.load_w, 0), x=num(surplus, 0)), must=["go for it|yes|good time"])
    elif v.sun_up and v.soc >= 60:
        c.say(pick(rng, ["Yes, that's fine. Battery's at {p} percent and solar's still producing {s} watts. {A} will use about {k} kilowatt hours, only {b} percent of the battery.",
                         "Sure. {p} percent and still charging from {s} watts of solar; {a} is about {k} kilowatt hours, roughly {b} percent."],
                   p=soc, s=num(v.solar_w, 0), A=cap(app), a=app, k=num(kwh), b=pctb), must=["yes|sure|fine"])
    elif v.soc >= 75 and kwh <= 0.15 * v.cap:
        c.say(pick(rng, ["Should be okay. Battery's at {p} percent with {r} kilowatt hours; {a} will take about {k}, roughly {b} percent, and tomorrow's forecast is {f} kilowatt hours.",
                         "Yes, we can afford it. {a} is about {k} kilowatt hours, {b} percent of the battery, and we're at {p} percent."],
                   p=soc, r=num(v.remaining), a=app, k=num(kwh), b=pctb, f=num(v.fc_tom)), must=["okay|yes|afford"])
    elif kwh <= 0.15:
        c.say("That's tiny, go ahead.", must=["go ahead|yes|sure|fine|okay"])
    else:
        why = "solar's finished for the day" if v.solar_w < 50 else f"solar's only making {num(v.solar_w, 0)} watts"
        c.say(pick(rng, ["I'd wait. Battery's at {p} percent and {w}. Tomorrow looks {cnd} with about {f} kilowatt hours forecast, so late morning would be better.",
                         "Better to hold off. We're at {p} percent, {w}, and {a} would take about {b} percent of the battery. Tomorrow's forecast is {f} kilowatt hours, so try mid-morning."],
                   p=soc, w=why, cnd=COND_TOMORROW.get(v.tom.get("condition"), "unclear"), f=num(v.fc_tom), a=app, b=pctb), must=["wait|hold off"])
    return c


def weather(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    v = EnergyView(house)
    kind = rng.choice(["now", "now", "tomorrow", "rain", "sun", "temp"])
    c = Conv(house, sp, "weather", rng)
    wb = {"classes": ["sun"]} if kind == "sun" else [{"classes": ["weather"]}, {"classes": ["forecast"]}] if kind == "rain" else {"classes": ["weather"]}
    if kind == "now":
        c.user(voiceify(rng.choice(["What's the weather like?", "What's it like outside?", "How's the weather?", "What's the weather doing?"]), rng, "q"))
        c.fetch(areas=["weather"], b=wb)
        c.say(pick(rng, ["It's {cnd} and {t} degrees outside, wind {w} kilometres an hour.", "{Cnd} out there, {t} degrees, with {w} k's of wind."],
                   cnd=COND_WORDS.get(v.cond, v.cond), Cnd=cap(COND_WORDS.get(v.cond, v.cond)), t=num(v.temp), w=v.wind), must=[num(v.temp)])
    elif kind == "temp":
        c.user(voiceify(rng.choice(["How cold is it outside?", "What's the temperature outside?", "How hot is it out there?"]), rng, "q"))
        c.fetch(areas=["weather"], b=wb)
        c.say(pick(rng, ["It's {t} degrees outside.", "{t} degrees out there right now."], t=num(v.temp)), must=[num(v.temp)])
    elif kind == "tomorrow":
        c.user(voiceify(rng.choice(["What's the forecast for tomorrow?", "What's tomorrow looking like?", "Weather tomorrow?"]), rng, "q"))
        c.fetch(areas=["weather"], b=wb)
        c.say(pick(rng, ["Tomorrow looks {cnd}, {lo} to {hi} degrees.", "{Cnd} tomorrow, with a low of {lo} and a high of {hi}."],
                   cnd=COND_TOMORROW.get(v.tom["condition"]), Cnd=cap(COND_TOMORROW.get(v.tom["condition"])), lo=num(v.tom["low"]), hi=num(v.tom["high"])), must=[num(v.tom["high"])])
    elif kind == "rain":
        c.user(voiceify(rng.choice(["Is it going to rain tomorrow?", "Will it rain tomorrow?", "Any rain coming?"]), rng, "q"))
        c.fetch(areas=["weather"], b=wb)
        if v.tom["condition"] == "rainy":
            c.say(pick(rng, ["Yes, rain's forecast for tomorrow, {lo} to {hi} degrees. Solar will be down too, about {f} kilowatt hours.", "Looks like it. Tomorrow's rainy, so expect a weak solar day, around {f} kilowatt hours."],
                       lo=num(v.tom["low"]), hi=num(v.tom["high"]), f=num(v.fc_tom)), must=["yes|looks like"])
        else:
            c.say(pick(rng, ["No rain forecast. Tomorrow looks {cnd}, {lo} to {hi}.", "Doesn't look like it, tomorrow should be {cnd}."],
                       cnd=COND_TOMORROW.get(v.tom["condition"]), lo=num(v.tom["low"]), hi=num(v.tom["high"])), must=["no|doesn't"])
    else:
        c.user(voiceify(rng.choice(["When's sunset?", "What time is sunrise?", "When does the sun come up?", "What time does the sun go down?"]), rng, "q"))
        c.fetch(areas=["weather"], b=wb)
        q = c.msgs[1]["content"].lower()
        if "sunset" in q or "go down" in q:
            c.say(pick(rng, ["Sunset's at {s}.", "The sun sets at {s}."], s=v.sset), must=[v.sset])
        else:
            c.say(pick(rng, ["Sunrise is at {r}.", "The sun's up at {r}."], r=v.rise), must=[v.rise])
    return c
