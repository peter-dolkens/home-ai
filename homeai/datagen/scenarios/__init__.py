"""Scenario registry with sampling weights. Weights reflect what matters most for
the target use: presence inference, the away-check routine, room control and power."""
from . import appliances, control, energy, media, misc, presence, routines, status

SCENARIOS = [
    # (name, builder, weight)
    ("control_area", control.control_area, 12),
    ("control_here", control.control_here, 6),
    ("control_brightness", control.control_brightness, 4),
    ("control_all_lights", control.control_all_lights, 2),
    ("control_named", control.control_named, 8),
    ("control_compound", control.control_compound, 5),
    ("control_climate", control.climate_control, 5),
    ("clarify", control.clarify, 9),
    ("clarify_in_room", control.clarify_in_room, 2),
    ("act_only_one", control.act_only_one, 2),
    ("missing", control.missing, 3),
    ("confirm_sensitive", control.confirm_sensitive, 7),
    ("control_lock", control.control_lock, 4),
    ("status_entity", status.status_entity, 8),
    ("status_aggregate", status.status_aggregate, 6),
    ("who_home", status.who_home, 3),
    ("presence_where", presence.presence_where, 14),
    ("presence_in_room", presence.presence_in_room, 3),
    ("check_house", routines.check_house, 10),
    ("away_motion", routines.away_motion, 4),
    ("goodnight", routines.goodnight, 9),
    ("leaving", routines.leaving, 3),
    ("energy_status", energy.energy_status, 6),
    ("energy_night", energy.energy_night, 5),
    ("energy_appliance", energy.energy_appliance, 4),
    ("weather", energy.weather, 3),
    ("smalltalk", misc.smalltalk, 2),
    # media (HomePod-parity is a stated priority)
    ("media_play", media.media_play, 9),
    ("media_transport", media.media_transport, 7),
    ("media_volume", media.media_volume, 6),
    ("media_what", media.media_what, 3),
    ("media_multiroom", media.media_multiroom, 3),
    # appliances, water, vehicle
    ("appliance_status", appliances.appliance_status, 4),
    ("appliance_control", appliances.appliance_control, 2),
    ("water_status", appliances.water_status, 3),
    ("irrigation_control", appliances.irrigation_control, 4),
    ("vehicle_status", appliances.vehicle_status, 3),
]

BY_NAME = {n: b for n, b, _ in SCENARIOS}
