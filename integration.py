"""integration.py - forecast -> appliance CSP -> battery search, with feedback loop.

Loop (max config.MAX_FEEDBACK_LOOPS):
  1. CSP schedules appliances on top of the forecast load
     (limit relaxed in 10% steps if needed, see csp_appliances.plan_appliances).
  2. Battery search (UCS / A*) plans charge/discharge for forecast + appliances,
     with the ORIGINAL grid limit as a hard limit.
  3. If the battery plan is impossible (e.g. the CSP only found a schedule by
     relaxing the limit and the battery cannot cover the excess), the schedule
     is added to `forbidden` and the CSP is solved again.

All prices / costs are SIMULATED (see config.py). Only the load is real.
"""
import config as cfg
from csp_appliances import ApplianceCSP, plan_appliances
from search_battery import plan_battery

B = cfg.BATTERY
EPS = 1e-9


def total_load(forecast, schedule):
    """forecast + appliances for a schedule {appliance: start_hour}."""
    csp = ApplianceCSP(forecast, cfg.APPLIANCES, cfg.POWER_LIMIT_KW)
    return csp.load_profile(schedule)


def plan_day(forecast, price=None, algo="astar", use_fc=True, use_ac3=True,
             max_loops=cfg.MAX_FEEDBACK_LOOPS):
    price = price or cfg.PRICES
    forecast = [max(0.0, float(x)) for x in forecast]
    forbidden, log = [], []
    for loop in range(1, max_loops + 1):
        res = plan_appliances(
            forecast, cfg.APPLIANCES, cfg.POWER_LIMIT_KW, cfg.CIRCUITS,
            cfg.ORDER_CONSTRAINTS, forbidden, use_fc=use_fc, use_ac3=use_ac3,
            relax_step=cfg.LIMIT_RELAX_STEP)
        if res["status"] == "no_solution":
            return {"status": "no_solution", "loops": loop, "log": log,
                    "message": res["message"], "forecast": forecast}
        sched = res["schedule"]
        load = total_load(forecast, sched)
        bat = plan_battery(load, price, B["capacity_kwh"],
                           B["max_rate_kwh_per_h"], B["initial_kwh"],
                           B["final_min_kwh"], cfg.POWER_LIMIT_KW, algo=algo)
        log.append({"loop": loop, "csp_status": res["status"],
                    "schedule": dict(sched), "battery_ok": bat is not None})
        if bat is not None:
            return {"status": res["status"], "loops": loop, "log": log,
                    "schedule": sched, "forecast": forecast, "load": load,
                    "battery": bat, "cost": bat["cost"], "csp": res}
        forbidden.append(dict(sched))  # feedback: reject this schedule
    return {"status": "failed_after_max_loops", "loops": max_loops, "log": log,
            "message": f"No schedule accepted by the battery in {max_loops} loops.",
            "forecast": forecast}


def realized(plan, actual_load, price=None):
    """Execute a plan on the ACTUAL load. No export: grid is clipped at 0.

    Returns cost (simulated) and the number of hours above the grid limit.
    """
    price = price or cfg.PRICES
    app = [t - f for t, f in zip(plan["load"], plan["forecast"])]
    grid = [max(0.0, a + p + d) for a, p, d in
            zip(actual_load, app, plan["battery"]["actions"])]
    cost = sum(pr * g for pr, g in zip(price, grid))
    viol = sum(g > cfg.POWER_LIMIT_KW + EPS for g in grid)
    return {"cost": cost, "violation_hours": viol}


def unplanned(actual_load, price=None):
    """Baseline: every appliance starts at its earliest hour, no battery."""
    price = price or cfg.PRICES
    load = list(actual_load)
    for a in cfg.APPLIANCES.values():
        for h in range(a["earliest"], a["earliest"] + a["duration"]):
            load[h] += a["power"]
    cost = sum(p * x for p, x in zip(price, load))
    viol = sum(x > cfg.POWER_LIMIT_KW + EPS for x in load)
    return {"cost": cost, "violation_hours": viol}
