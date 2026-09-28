"""Small talk and capability questions, so the model doesn't reach for tools on everything."""
from __future__ import annotations

import random
from typing import Optional

from ...house.schema import House, Speaker
from ..common import Conv

PAIRS = [
    (["Hello.", "Hi.", "Hey house.", "Hey there."], ["Hi! What can I do for you?", "Hey. What do you need?", "Hello. How can I help?"]),
    (["Thanks.", "Thank you.", "Cheers.", "Ta."], ["No worries.", "Anytime.", "You're welcome."]),
    (["What can you do?", "What are you able to help with?", "How can you help me?"],
     ["I can control the lights, locks, blinds, fans and media, tell you who's home and roughly where they are, keep an eye on doors and windows, and report on the battery, solar and weather.",
      "Lights, locks, blinds, heating and media, plus who's home, whether the house is secure, and how the battery and solar are doing."]),
    (["Tell me a joke.", "Got a joke?"], ["Why did the solar panel go to therapy? It had too many issues with its current situation.", "I'd tell you a joke about the battery, but it's a bit draining."]),
    (["Never mind.", "Forget it.", "Cancel that."], ["Okay.", "No problem.", "Sure."]),
    (["What time is it?", "What's the time?"], ["It's {time}.", "{time}."]),
    (["What day is it?", "What's today?"], ["It's {day}.", "{day}."]),
    (["Are you there?", "You awake?"], ["I'm here.", "Yep, listening."]),
]


def smalltalk(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    us, rs = rng.choice(PAIRS)
    c = Conv(house, sp, "smalltalk", rng)
    u = rng.choice(us)
    c.user(u if rng.random() < 0.6 else u.rstrip(".?").lower())
    r = rng.choice(rs).format(time=house.now.strftime("%H:%M"), day=house.now.strftime("%A"))
    c.say(r)
    return c
