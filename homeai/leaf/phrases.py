"""Phrasing for room-leaf instructions. Every table is (train, test): the held-out test set is written from the second
list only, so it measures fit to the room rather than memorised sentences (data.py also draws paraphrases from the
v1-train / v1-held banks and, for test only, NATURAL).

Slots: {act} a verb phrase before the object ("turn off"), {pol} a particle after it ("off"), {dev} the device phrase
with its article ("the island pendant", "all the kitchen lights", "blind 2"), {devn} the same without a leading "the",
{inr} " in <room>" or "", {room} the room as a noun ("the kitchen", or "this room" when the room is not named),
{pct} a percentage, {deg} a temperature, {n} a bare number, {mode}, {speed}, {q} a music query, {are} is / are."""
from __future__ import annotations

# action value -> (train verbs, test verbs); {act} slot
ACT = {
    "on": (["turn on", "switch on", "put on"], ["flick on", "pop on"]),
    "off": (["turn off", "switch off", "kill", "shut off"], ["flick off", "cut"]),
    "open": (["open", "raise", "pull up"], ["lift", "roll up"]),
    "close": (["close", "lower", "shut", "draw"], ["pull down", "roll down"]),
    "start": (["start", "run", "kick off"], ["get going", "set off"]),
    "stop": (["stop", "cancel"], ["halt", "abort"]),
}
# {pol} slot (after the object)
POL = {
    "on": (["on"], ["on"]),
    "off": (["off"], ["off"]),
    "open": (["up", "open"], ["up", "open"]),
    "close": (["down", "closed"], ["down", "shut"]),
}
# words that evidence an argument value in the final text (after paraphrase and speech noise); first found is quoted
EVID = {
    "on": ["turn on", "switch on", "put on", "flick on", "pop on", "on"],
    "off": ["turn off", "switch off", "shut off", "flick off", "kill", "cut", "off", "out"],
    "dimmer": ["dimmer", "too bright", "dim", "darker", "down", "lower", "drop", "tone"],
    "brighter": ["brighter", "brighten", "too dim", "more light", "bump", "up"],
    "open": ["open", "raise", "pull up", "lift", "roll up", "up"],
    "close": ["close", "lower", "shut", "draw", "pull down", "roll down", "down"],
    "stop": ["stop", "hold", "halt", "freeze", "cancel", "abort"],
    "start": ["start", "run", "kick off", "get going", "set off", "going"],
    "warmer": ["warmer", "hotter", "heat it up", "warm it up", "warm"],
    "cooler": ["cooler", "colder", "cool it down", "cool down", "cool"],
    "play": ["resume", "unpause", "keep playing", "carry on playing", "play", "start"],
    "pause": ["pause", "hold"],
    "next": ["next", "skip"],
    "previous": ["previous", "go back", "back one", "last song", "back a track"],
    "turn on": ["turn on", "switch on", "power on", "on"],
    "turn off": ["turn off", "switch off", "power off", "shut off", "off"],
    "mute": ["mute", "silence", "shh"],
    "unmute": ["unmute", "sound back on", "sound back"],
    "louder": ["louder", "crank", "bump", "up"],
    "quieter": ["quieter", "too loud", "lower", "softer", "down"],
    "dock": ["dock", "home", "base"],
    "except": ["except for", "except", "but not", "apart from", "other than", "but leave", "not"],
    "heat": ["heating", "heat"], "cool": ["cooling", "cool"], "auto": ["auto"], "dry": ["dry"], "fan only": ["fan only"],
    "low": ["low"], "medium": ["medium"], "high": ["high"],
    "playlist": ["playlist"], "album": ["album"], "track": ["song", "track"], "radio": ["radio", "station"],
}
READ_EVID = {
    "presence": ["anyone", "anybody", "someone", "occupied", "empty", "presence", "people"],
    "temperature": ["temperature", "temp", "warm", "cold", "hot", "chilly"],
    "humidity": ["humidity", "humid", "muggy"],
    "lights": ["lights", "light"],
    "blinds": ["blinds", "blind"],
    "windows": ["windows", "window"],
    "doors": ["doors", "door"],
    "air quality": ["air quality", "co2", "stuffy", "air"],
    "light level": ["light level", "how bright", "how dark", "dark", "bright"],
    "power use": ["power", "watts", "electricity"],
    "smoke and leaks": ["smoke", "leak", "leaking"],
}

T: dict[str, tuple[list[str], list[str]]] = {
    # ---------------------------------------------------------------- lights
    "light_onoff": (
        ["{act} {dev}{inr}", "{devn} {pol}{inr}", "turn {dev} {pol}{inr}", "can you {act} {dev}{inr}", "switch {dev} {pol}{inr}",
         "{act} {dev} please", "{act} {dev}"],
        ["could you {act} {dev}{inr}", "i need {dev} {pol}{inr}", "{devn} {pol} now{inr}", "go ahead and {act} {dev}", "get {dev} {pol}{inr}"]),
    "light_all_onoff": (
        ["{act} {dev}{inr}", "{devn} {pol}{inr}", "turn {dev} {pol}{inr}", "can you turn {dev} {pol}{inr}", "{act} {dev} please"],
        ["could you {act} {dev}{inr}", "i'd like {dev} {pol}{inr}", "{devn} {pol} thanks{inr}", "get {dev} {pol}{inr}"]),
    "light_except": (
        ["{act} {all}{inr} except {dev}", "{act} {all} but not {dev}{inr}", "{all} {pol}{inr} except for {dev}", "{act} {all} except {dev}"],
        ["{all} {pol}{inr} apart from {dev}", "{act} {all}{inr} other than {dev}", "{act} {all} but leave {dev}{inr}"]),
    "light_bright": (
        ["set {dev} to {pct}{inr}", "dim {dev} to {pct}{inr}", "{devn} to {pct}{inr}", "brighten {dev} to {pct}{inr}", "put {dev} at {pct}{inr}"],
        ["bring {dev} to {pct}{inr}", "make {dev} {pct}{inr}", "can i have {dev} at {pct}{inr}", "{devn} at {pct} please"]),
    "light_on_bright": (
        ["{act} {dev} at {pct}{inr}", "{act} {dev} to {pct}{inr}"],
        ["{act} {dev}{inr} at about {pct}"]),
    "light_dimmer": (
        ["dim {dev} a bit{inr}", "{devn} a bit dimmer{inr}", "make {dev} dimmer{inr}", "turn {dev} down a bit{inr}", "lower {dev}{inr}"],
        ["{dev} {are} too bright{inr}", "tone {dev} down{inr}", "can you drop {dev} a little{inr}"]),
    "light_brighter": (
        ["brighten {dev}{inr}", "make {dev} brighter{inr}", "turn {dev} up a bit{inr}", "{devn} a bit brighter{inr}"],
        ["{dev} {are} too dim{inr}", "bump {dev} up a little{inr}", "i need more light from {dev}{inr}"]),
    # ---------------------------------------------------------------- covers
    "cover_move": (
        ["{act} {dev}{inr}", "{devn} {pol}{inr}", "can you {act} {dev}{inr}", "{act} {dev} please"],
        ["could you {act} {dev}{inr}", "i'd like {dev} {pol}{inr}", "{act} {dev} for me{inr}"]),
    "cover_stop": (["stop {dev}{inr}", "hold {dev} there{inr}"], ["freeze {dev}{inr}", "halt {dev}{inr}"]),
    "cover_pos": (
        ["set {dev} to {pct}{inr}", "{devn} to {pct}{inr}", "{act} {dev} to {pct}{inr}"],
        ["put {dev} at {pct}{inr}", "i want {dev} at {pct}{inr}"]),
    "cover_half": (["{act} {dev} halfway{inr}"], ["{act} {dev} about halfway{inr}"]),
    # ---------------------------------------------------------------- climate
    "clim_onoff": (
        ["{act} {dev}{inr}", "turn {dev} {pol}{inr}", "{devn} {pol}{inr}", "can you {act} {dev}{inr}"],
        ["could you {act} {dev}{inr}", "go ahead and {act} {dev}{inr}"]),
    "clim_mode": (
        ["put {dev} on {mode}{inr}", "set {dev} to {mode}{inr}", "{devn} to {mode} mode{inr}", "switch {dev} to {mode}{inr}"],
        ["change {dev} over to {mode}{inr}", "i want {dev} on {mode}{inr}"]),
    "clim_temp": (
        ["set {dev} to {deg}{inr}", "{devn} to {deg}{inr}", "make it {deg}{inr}", "set the temperature to {deg}{inr}"],
        ["can you get it to {deg}{inr}", "i want it at {deg}{inr}", "change the temperature to {deg}{inr}"]),
    "clim_on_temp": (["{act} {dev} at {deg}{inr}", "{act} {dev} and set it to {deg}{inr}"], ["{act} {dev}{inr}, {deg} please"]),
    "clim_mode_temp": (["set {dev} to {mode} at {deg}{inr}", "{devn} on {mode}, {deg}{inr}"], ["{mode} on {dev} at {deg}{inr}"]),
    "clim_warmer": (["make it warmer{inr}", "a bit warmer{inr}", "warm it up a bit{inr}", "make {dev} warmer{inr}"],
                    ["it's a bit cold{inr}, can you make it warmer", "can you heat it up a little{inr}"]),
    "clim_cooler": (["make it cooler{inr}", "a bit cooler{inr}", "cool it down a bit{inr}", "make {dev} cooler{inr}"],
                    ["it's too hot{inr}, make it colder", "can you cool it down a touch{inr}"]),
    # ---------------------------------------------------------------- fans
    "fan_onoff": (["{act} {dev}{inr}", "{devn} {pol}{inr}", "turn {dev} {pol}{inr}", "can you {act} {dev}{inr}"],
                  ["could you {act} {dev}{inr}", "get {dev} {pol}{inr}"]),
    "fan_speed": (["set {dev} to {speed}{inr}", "{devn} on {speed}{inr}", "turn {dev} to {speed}{inr}"],
                  ["put {dev} on {speed} speed{inr}", "{speed} on {dev}{inr} please"]),
    # ---------------------------------------------------------------- media
    "media_play": (["resume {dev}{inr}", "unpause {dev}{inr}", "resume{inr}", "keep playing{inr}", "play {dev} again{inr}"],
                   ["carry on playing{inr}", "resume the music{inr} please"]),
    "media_pause": (["pause {dev}{inr}", "pause{inr}", "pause the music{inr}", "hold the music{inr}"],
                    ["put {dev} on pause{inr}", "pause that for a sec{inr}"]),
    "media_stop": (["stop {dev}{inr}", "stop the music{inr}"], ["that's enough music{inr}, stop it"]),
    "media_next": (["skip this song{inr}", "next track{inr}", "skip{inr}", "next song{inr}"], ["play the next one{inr}", "skip ahead a track{inr}"]),
    "media_previous": (["previous track{inr}", "go back a track{inr}", "play the last song again{inr}"], ["back one song{inr}"]),
    "media_turn on": (["turn on {dev}{inr}", "switch {dev} on{inr}"], ["power on {dev}{inr}"]),
    "media_turn off": (["turn off {dev}{inr}", "switch {dev} off{inr}", "{devn} off{inr}"], ["power off {dev}{inr}", "shut {dev} off{inr}"]),
    "media_mute": (["mute {dev}{inr}", "mute{inr}", "mute the music{inr}"], ["silence {dev}{inr}"]),
    "media_unmute": (["unmute {dev}{inr}", "sound back on{inr}"], ["turn the sound back on{inr}"]),
    "vol_set": (["set the volume to {pct}{inr}", "volume {n}{inr}", "volume to {pct} on {dev}", "set {dev} volume to {pct}"],
                ["can you put the volume at {pct}{inr}", "make {dev} {pct} volume"]),
    "vol_louder": (["louder{inr}", "turn it up{inr}", "turn {dev} up{inr}", "a bit louder{inr}", "crank it{inr}"],
                   ["i can't hear it, louder please{inr}", "bump the volume up{inr}"]),
    "vol_quieter": (["quieter{inr}", "turn it down{inr}", "turn {dev} down a bit{inr}", "a bit quieter{inr}"],
                    ["it's too loud{inr}", "lower the volume{inr}"]),
    "music_any": (["play {q}{inr}", "put on some {q}{inr}", "play some {q}{inr}", "put {q} on{inr}"],
                  ["can we have {q}{inr}", "i feel like some {q}{inr}", "chuck on {q}{inr}"]),
    "music_playlist": (["play the {q} playlist{inr}", "put on my {q} playlist{inr}", "start the {q} playlist{inr}"],
                       ["throw the {q} playlist on{inr}", "can we get my {q} playlist going{inr}"]),
    "music_album": (["play the album {q}{inr}", "put on the album {q}{inr}"], ["i want to hear the album {q}{inr}"]),
    "music_radio": (["tune in to {q} radio{inr}", "play the radio station {q}{inr}", "put {q} on the radio{inr}"],
                    ["can you find {q} on the radio{inr}"]),
    "music_track": (["play the song {q}{inr}", "play the track {q}{inr}"], ["put on that song {q}{inr}"]),
    # ---------------------------------------------------------------- switches, appliances, vacuum
    "switch_onoff": (["{act} {dev}{inr}", "{devn} {pol}{inr}", "turn {dev} {pol}{inr}", "switch {dev} {pol}{inr}", "can you {act} {dev}{inr}"],
                     ["could you {act} {dev}{inr}", "i need {dev} {pol}{inr}", "get {dev} {pol}{inr}"]),
    "appliance": (["{act} {dev}{inr}", "can you {act} {dev}{inr}", "{act} {dev} please"],
                  ["could you {act} {dev} for me{inr}", "time to {act} {dev}{inr}"]),
    "vac_start": (["start the vacuum", "start the robot vacuum", "run the robot vacuum"], ["get the robot vacuum started"]),
    "vac_stop": (["stop the vacuum", "stop the robot vacuum"], ["halt the robot vacuum"]),
    "vac_dock": (["dock the vacuum", "send the robot home", "send the vacuum back to its base"], ["tell the robot vacuum to go home"]),
    # ---------------------------------------------------------------- reads
    "read_presence": (["is anyone{inr}", "is anybody{inr}", "is {room} occupied", "is there someone{inr}"],
                      ["has anyone been{inr} lately", "is {room} empty"]),
    "read_temperature": (["what's the temperature{inr}", "how warm is it{inr}", "how cold is it{inr}", "how hot is {room}"],
                         ["is it chilly{inr}", "what's it sitting at{inr} temperature wise"]),
    "read_humidity": (["how humid is it{inr}", "what's the humidity{inr}"], ["is it muggy{inr}"]),
    "read_lights": (["are the lights on{inr}", "which lights are on{inr}", "did i leave the lights on{inr}", "are any lights on{inr}"],
                    ["is there a light on{inr}", "what lights are on{inr}"]),
    "read_blinds": (["are the blinds open{inr}", "are the blinds closed{inr}"], ["did anyone close the blinds{inr}"]),
    "read_windows": (["are the windows open{inr}", "is a window open{inr}", "did we leave a window open{inr}"], ["any windows open{inr}"]),
    "read_doors": (["are the doors shut{inr}", "is the door open{inr}", "is a door open{inr}"], ["did someone leave the door open{inr}"]),
    "read_air quality": (["how's the air quality{inr}", "what's the co2 like{inr}", "is the air stuffy{inr}"], ["how fresh is the air{inr}"]),
    "read_light level": (["how bright is it{inr}", "what's the light level{inr}", "how dark is it{inr}"], ["is it dark{inr}"]),
    "read_power use": (["how much power is being used{inr}", "what's using power{inr}", "what's the power draw{inr}"], ["how many watts are we pulling{inr}"]),
    "read_smoke and leaks": (["any leaks{inr}", "is there a leak{inr}", "is the smoke alarm going off{inr}"], ["is water leaking anywhere{inr}", "any smoke{inr}"]),
    "read_device": (["is {dev} on{inr}", "is {dev} running{inr}", "has {dev} finished{inr}", "what's {dev} doing{inr}"],
                    ["is {dev} still going{inr}", "can you check {dev}{inr}", "what's the status of {dev}{inr}"]),
    "read_climate": (["is {dev} on{inr}", "what's {dev} set to{inr}", "is {dev} running{inr}"], ["what's {dev} doing{inr}"]),
    "read_media": (["what's playing on {dev}", "what's playing{inr}", "is {dev} on{inr}"], ["what song is this{inr}", "what's {dev} playing{inr}"]),
}

# planner hand-over: terse, normalised, room given as a prefix
P: dict[str, tuple[list[str], list[str]]] = {
    "light_onoff": (["{devn} {pol}", "turn {pol} {devn}", "switch {pol} {devn}"], ["{devn}: {pol}", "set {devn} {pol}"]),
    "light_all_onoff": (["{devn} {pol}", "turn {pol} {devn}", "all lights {pol}"], ["lights: {pol}", "set all lights {pol}"]),
    "light_except": (["all lights {pol} except {devn}", "lights {pol} except {devn}", "turn {pol} all lights except {devn}"], ["all lights {pol}, not {devn}"]),
    "light_bright": (["{devn} to {pct}", "set {devn} {pct}", "dim {devn} to {pct}"], ["{devn} brightness {pct}"]),
    "light_on_bright": (["turn on {devn} at {pct}"], ["{devn} on, {pct}"]),
    "light_dimmer": (["{devn} dimmer", "dim {devn} a bit"], ["{devn} down a little"]),
    "light_brighter": (["{devn} brighter", "brighten {devn}"], ["{devn} up a little"]),
    "cover_move": (["{devn} {pol}", "{act} {devn}"], ["{devn}: {act}"]),
    "cover_stop": (["stop {devn}"], ["{devn}: stop"]),
    "cover_pos": (["{devn} to {pct}", "set {devn} {pct}"], ["{devn} position {pct}"]),
    "cover_half": (["{act} {devn} halfway"], ["{devn} halfway {pol}"]),
    "clim_onoff": (["{devn} {pol}", "turn {pol} {devn}"], ["{devn}: {pol}"]),
    "clim_mode": (["{devn} {mode}", "{devn} to {mode}"], ["{devn} mode {mode}"]),
    "clim_temp": (["{devn} to {deg}", "set {devn} {deg}"], ["{devn} setpoint {deg}"]),
    "clim_on_temp": (["turn on {devn} at {deg}"], ["{devn} on, {deg}"]),
    "clim_mode_temp": (["{devn} {mode} {deg}", "{devn} to {mode}, {deg}"], ["{devn}: {mode} at {deg}"]),
    "clim_warmer": (["{devn} warmer", "warmer"], ["{devn} a degree warmer"]),
    "clim_cooler": (["{devn} cooler", "cooler"], ["{devn} a degree cooler"]),
    "fan_onoff": (["{devn} {pol}", "turn {pol} {devn}"], ["{devn}: {pol}"]),
    "fan_speed": (["{devn} {speed}", "{devn} speed {speed}"], ["{devn} to {speed}"]),
    "media_play": (["resume {devn}", "resume playback"], ["{devn}: resume"]),
    "media_pause": (["pause {devn}", "pause playback"], ["{devn}: pause"]),
    "media_stop": (["stop {devn}", "stop playback"], ["{devn}: stop"]),
    "media_next": (["next track", "skip track on {devn}"], ["{devn}: next"]),
    "media_previous": (["previous track", "previous track on {devn}"], ["{devn}: previous"]),
    "media_turn on": (["turn on {devn}"], ["{devn}: on"]),
    "media_turn off": (["turn off {devn}"], ["{devn}: off"]),
    "media_mute": (["mute {devn}", "mute"], ["{devn}: mute"]),
    "media_unmute": (["unmute {devn}", "unmute"], ["{devn}: unmute"]),
    "vol_set": (["volume {n}", "{devn} volume {n}", "set volume {pct}"], ["volume to {pct}"]),
    "vol_louder": (["volume up", "louder"], ["{devn} louder"]),
    "vol_quieter": (["volume down", "quieter"], ["{devn} quieter"]),
    "music_any": (["play {q}", "play {q} on {devn}"], ["put on {q}"]),
    "music_playlist": (["play playlist {q}", "play the {q} playlist"], ["playlist: {q}"]),
    "music_album": (["play album {q}"], ["album: {q}"]),
    "music_radio": (["play radio {q}", "radio station {q}"], ["radio: {q}"]),
    "music_track": (["play song {q}", "play track {q}"], ["song: {q}"]),
    "switch_onoff": (["{devn} {pol}", "turn {pol} {devn}"], ["{devn}: {pol}", "set {devn} {pol}"]),
    "appliance": (["{act} {devn}"], ["{devn}: {act}"]),
    "vac_start": (["start vacuum", "start the robot vacuum"], ["vacuum: start"]),
    "vac_stop": (["stop vacuum"], ["vacuum: stop"]),
    "vac_dock": (["dock vacuum", "send vacuum home"], ["vacuum: dock"]),
    "read_presence": (["check presence", "is anyone here"], ["presence?"]),
    "read_temperature": (["report temperature", "temperature?"], ["read temperature"]),
    "read_humidity": (["report humidity"], ["humidity?"]),
    "read_lights": (["check lights", "which lights are on"], ["lights status?"]),
    "read_blinds": (["check blinds"], ["blinds status?"]),
    "read_windows": (["check windows"], ["windows open?"]),
    "read_doors": (["check doors"], ["doors open?"]),
    "read_air quality": (["report air quality"], ["air quality?"]),
    "read_light level": (["report light level"], ["light level?"]),
    "read_power use": (["report power use"], ["power draw?"]),
    "read_smoke and leaks": (["check smoke and leaks"], ["any leaks or smoke?"]),
    "read_device": (["check {devn}", "status of {devn}"], ["{devn} status?"]),
    "read_climate": (["check {devn}", "status of {devn}"], ["{devn} status?"]),
    "read_media": (["what's playing on {devn}", "check {devn}"], ["{devn} status?"]),
}

# test only: written to be natural, not templated from the lists above
NATURAL: dict[str, list[str]] = {
    "light_all_onoff:off": ["we're heading to bed, lights off{inr}", "nobody's using {room}, you can turn the lights off", "it's too bright, lights off{inr}"],
    "light_all_onoff:on": ["it's getting dark{inr}, lights on please", "can't see a thing{inr}, turn the lights on"],
    "light_onoff:off": ["{dev} {are} still on{inr}, turn {dev} off", "no need for {dev}{inr}, switch it off"],
    "light_onoff:on": ["i need some light, put {dev} on{inr}", "i'm reading, turn on {dev}{inr}"],
    "light_bright": ["it's movie time, drop {dev} to {pct}{inr}", "just {pct} on {dev}{inr} thanks"],
    "light_dimmer": ["it's a bit harsh{inr}, dim {dev} a bit"],
    "cover_move:close": ["the sun's in my eyes, close {dev}{inr}", "it's getting dark, shut {dev}{inr}"],
    "cover_move:open": ["let some light in, open {dev}{inr}"],
    "clim_temp": ["i'm freezing, set it to {deg}{inr}", "it's stuffy{inr}, drop it to {deg}"],
    "clim_onoff:off": ["nobody's in {room}, turn {dev} off"],
    "clim_warmer": ["brr, it's cold{inr}, a bit warmer please"],
    "clim_cooler": ["i'm sweating{inr}, make it cooler"],
    "media_pause": ["i've got a phone call, pause the music{inr}"],
    "vol_quieter": ["the baby's asleep, turn it down{inr}"],
    "music_any": ["we're cooking, put on some {q}{inr}"],
    "switch_onoff:on": ["i'd love a cuppa, turn {dev} on{inr}", "flick {dev} on for me{inr}"],
    "switch_onoff:off": ["we're done with {dev}{inr}, turn it off"],
    "appliance:start": ["it's full, start {dev}{inr}"],
    "read_presence": ["is someone still{inr} or can i switch things off"],
    "read_temperature": ["should i grab a jumper, how cold is it{inr}"],
    "read_lights": ["did someone leave a light on{inr}"],
}

JOIN = ([" and ", " and then ", ", then ", " and also "], [" plus ", " and after that ", ", and while you're at it "])  # raw compound joins
P_JOIN = ([", ", "; ", " and "], ["; then ", ", also "])

OFF_TOPIC = (
    ["lock the front door", "is the back door locked", "arm the alarm", "what's the weather tomorrow", "how much solar are we making",
     "what's the battery at", "who's home", "where is Amber", "set a timer for ten minutes", "what time is it", "tell me a joke",
     "water the vegie beds for 20 minutes", "how full is the water tank", "is the car charged", "add milk to the shopping list",
     "call mum", "good night", "thanks", "turn on the hot water", "start the generator", "disarm the alarm", "is it going to rain",
     "how long is the battery going to last", "open the front gate"],
    ["unlock the front door", "what's the forecast for the weekend", "remind me to call the plumber", "how long until sunset",
     "is the gate shut", "who's at the door", "run the irrigation", "what's the pool pump doing", "how much power did we make today",
     "is Peter home yet", "what's on my calendar", "turn the hydronic heating on"],
)

# music queries: (train, test) per kind, disjoint
MUSIC = {
    "artist": (["Taylor Swift", "The Weeknd", "Queen", "Miles Davis", "Fleetwood Mac", "Billie Eilish", "Nick Cave", "Daft Punk",
                "Norah Jones", "The Beatles", "Powderfinger", "Hilltop Hoods", "Tame Impala", "Vance Joy"],
               ["Missy Higgins", "Midnight Oil", "Kasey Chambers", "Radiohead", "Paul Kelly", "Crowded House"]),
    "genre": (["jazz", "classical", "80s music", "country", "hip hop", "lo-fi", "rock", "blues"],
              ["dinner music", "Australian music", "folk", "reggae"]),
    "playlist": (["Dinner Party", "Chill Mix", "Discover Weekly", "Road Trip", "Sunday Morning", "Focus", "Workout"],
                 ["Favourites", "Peter's Mix", "Amber's Mix", "Summer Hits"]),
    "album": (["Rumours", "Abbey Road", "Currents", "Kind of Blue"], ["1989", "Discovery", "Woodface"]),
    "radio": (["Triple J", "ABC Radio National", "Double J", "Smooth FM"], ["ABC Classic", "BBC World Service", "Triple M"]),
    "track": (["Cruel Summer", "Blinding Lights", "Bohemian Rhapsody", "Riptide", "Heat Waves", "Down Under"],
              ["Redbone", "Levitating", "Big Jet Plane", "Chameleon"]),
}

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n == 100:
        return "a hundred"
    t, o = divmod(n, 10)
    return TENS[t] + ("" if o == 0 else " " + ONES[o])
