"""run_demo.py - plan ONE test day end to end and print a readable summary.

    python run_demo.py                  # first test day, data/hourly.csv
    python run_demo.py --day 2010-06-15 # a specific test day
    python run_demo.py --synthetic      # DEV ONLY fake data
All prices / costs are SIMULATED. Only the load is real.
"""
import argparse

import pandas as pd

import config as cfg
import nn_forecast as nf
from integration import plan_day, realized, unplanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=nf.DATA_CSV)
    ap.add_argument("--day", default=None, help="YYYY-MM-DD inside the test period")
    ap.add_argument("--synthetic", action="store_true", help="DEV ONLY")
    a = ap.parse_args()

    series = nf._fake_series() if a.synthetic else nf.load_hourly(a.data)
    D = nf.prepare(series)
    model = nf.train_nn(D, seed=cfg.SEED)
    test = D["sl"]["test"]
    idx = test.start
    if a.day:
        idx = list(D["dates"]).index(pd.Timestamp(a.day))
        assert test.start <= idx < test.stop, "day is not in the test period"
    day = D["dates"][idx]
    forecast = nf.predict_day(model, D, idx).clip(min=0)
    actual = D["y_raw"][idx]

    plan = plan_day(forecast)
    print(f"Day {day.date()}   status: {plan['status']}   loops: {plan['loops']}")
    if "schedule" not in plan:
        print(plan.get("message"))
        return
    print("\nAppliance start hours (CSP):")
    for n, s in sorted(plan["schedule"].items(), key=lambda x: x[1]):
        d = cfg.APPLIANCES[n]["duration"]
        print(f"  {n:11s} {s:02d}:00 - {s + d:02d}:00")
    c = plan["csp"]
    print(f"  limit used: {c['limit_used']:.2f} kW (relaxed {c['relax_percent']}%), "
          f"counters: {c['counters']}")
    b = plan["battery"]
    print("\nBattery action per hour (kWh, + charge / - discharge):")
    print("  ", b["actions"])
    print(f"  search counters: expanded={b['expanded']} generated={b['generated']}")
    r, u = realized(plan, actual), unplanned(actual)
    print(f"\nSimulated cost on the ACTUAL load: planned {r['cost']:.3f}  "
          f"vs unplanned {u['cost']:.3f}  "
          f"(saving {100 * (1 - r['cost'] / u['cost']):.1f}%)")
    print(f"Hours over the {cfg.POWER_LIMIT_KW} kW limit: planned "
          f"{r['violation_hours']}  unplanned {u['violation_hours']}")
    if a.synthetic:
        print("\nDEV ONLY - fake data, never report these numbers.")


if __name__ == "__main__":
    main()
