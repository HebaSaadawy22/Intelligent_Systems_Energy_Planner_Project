"""Generated (SIMULATED) parameters for P2 Home Energy Planner.

Everything in this file is made up by the project, not measured.
Only the household load (UCI dataset) is real.
"""

SEED = 42
HOURS = 24

# Grid power limit in kW (generated)
POWER_LIMIT_KW = 5.0

# Hourly price per kWh (generated, SIMULATED): cheap night, expensive evening peak
PRICES = (
    [0.10] * 6      # 00-05 night
    + [0.18] * 11   # 06-16 day
    + [0.35] * 5    # 17-21 evening peak
    + [0.18] * 1    # 22
    + [0.10] * 1    # 23
)
assert len(PRICES) == HOURS

# Flexible appliances (generated): power kW, duration h, earliest start, latest END hour
APPLIANCES = {
    "washer":     {"power": 0.5, "duration": 2, "earliest": 7,  "latest_end": 20},
    "dryer":      {"power": 2.0, "duration": 1, "earliest": 8,  "latest_end": 23},
    "dishwasher": {"power": 1.8, "duration": 2, "earliest": 12, "latest_end": 24},
    "ev":         {"power": 3.3, "duration": 3, "earliest": 0,  "latest_end": 24},
    "oven":       {"power": 2.2, "duration": 1, "earliest": 11, "latest_end": 22},
}

# (before, after): "after" must start when "before" has finished
ORDER_CONSTRAINTS = [("washer", "dryer")]

# Appliances on the same circuit cannot run at the same time
CIRCUITS = {
    "kitchen": ["dishwasher", "oven"],
    "heavy": ["dryer", "ev"],
}

# Battery (generated). Levels are integer kWh.
BATTERY = {
    "capacity_kwh": 4,
    "max_rate_kwh_per_h": 2,
    "initial_kwh": 0,
    "final_min_kwh": 0,
}

MAX_FEEDBACK_LOOPS = 5
LIMIT_RELAX_STEP = 0.10  # relax the power limit in 10% steps if no solution
