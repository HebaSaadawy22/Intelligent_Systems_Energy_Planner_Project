"""experiments.py - results for the report (writes results/*.csv and summary.md).

    python experiments.py                 # 5 seeds on the real test year
    python experiments.py --synthetic     # DEV ONLY (never report)
Parts:
  A. End-to-end: forecast source (NN / yesterday / last week / oracle) ->
     plan -> SIMULATED cost on the ACTUAL load. NN is averaged over seeds
     (mean/std); the other sources do not depend on the seed (std = 0).
  B. CSP: BT vs FC vs FC+AC-3 (counters).  C. Battery: UCS vs A* (counters)
     with a DP optimality check.
"""
import argparse
import os

import numpy as np
import pandas as pd

import config as cfg
import nn_forecast as nf
from csp_appliances import ApplianceCSP
from integration import plan_day, realized, unplanned, total_load
from search_battery import plan_battery, dp_battery

OUT = "results"
B = cfg.BATTERY


def run_source(forecasts, actuals):
    """Plan every test day with given forecasts; return mean metrics."""
    rows = []
    for f, y in zip(forecasts, actuals):
        f = np.clip(f, 0, None)
        p = plan_day(f)
        u = unplanned(y)
        if "schedule" not in p:
            rows.append({"ok": 0, "saving_pct": np.nan, "viol": np.nan,
                         "viol_unplanned": u["violation_hours"], "loops": p["loops"],
                         "relaxed": np.nan})
            continue
        r = realized(p, y)
        rows.append({"ok": 1, "saving_pct": 100 * (1 - r["cost"] / u["cost"]),
                     "viol": r["violation_hours"],
                     "viol_unplanned": u["violation_hours"], "loops": p["loops"],
                     "relaxed": float(p["status"] == "relaxed")})
    d = pd.DataFrame(rows)
    return {"plan_found_%": 100 * d["ok"].mean(),
            "saving_vs_unplanned_%": d["saving_pct"].mean(),
            "viol_hours/day": d["viol"].mean(),
            "viol_hours/day_unplanned": d["viol_unplanned"].mean(),
            "avg_loops": d["loops"].mean(),
            "relaxed_days_%": 100 * d["relaxed"].mean()}


def algo_tables(forecasts, actuals):
    csp_rows, bat_rows, dp_ok, n = [], [], 0, 0
    for f in forecasts:
        f = list(np.clip(f, 0, None))
        for name, fc, ac in [("BT", False, False), ("FC", True, False),
                             ("FC+AC3", True, True)]:
            c = ApplianceCSP(f, cfg.APPLIANCES, cfg.POWER_LIMIT_KW, cfg.CIRCUITS,
                             cfg.ORDER_CONSTRAINTS)
            sol = c.solve(use_fc=fc, use_ac3=ac)
            csp_rows.append({"algo": name, "solved": sol is not None,
                             "assignments": c.assignments,
                             "backtracks": c.backtracks,
                             "ac3_revisions": c.ac3_revisions})
        p = plan_day(f, algo="ucs")
        if "schedule" not in p:
            continue
        load = p["load"]
        args = (B["capacity_kwh"], B["max_rate_kwh_per_h"], B["initial_kwh"],
                B["final_min_kwh"], cfg.POWER_LIMIT_KW)
        dp, _ = dp_battery(load, cfg.PRICES, *args)
        for algo in ("ucs", "astar"):
            r = plan_battery(load, cfg.PRICES, *args, algo=algo)
            bat_rows.append({"algo": algo, "expanded": r["expanded"],
                             "generated": r["generated"],
                             "matches_dp": abs(r["cost"] - dp) < 1e-6})
        n += 1
    csp = pd.DataFrame(csp_rows).groupby("algo")[
        ["solved", "assignments", "backtracks", "ac3_revisions"]].mean().round(2)
    bat = pd.DataFrame(bat_rows).groupby("algo")[
        ["expanded", "generated", "matches_dp"]].mean().round(2)
    return csp, bat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=nf.DATA_CSV)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--algo_days", type=int, default=60,
                    help="test days used for the CSP / battery counters")
    ap.add_argument("--synthetic", action="store_true", help="DEV ONLY")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    D = nf.prepare(nf._fake_series() if a.synthetic else nf.load_hourly(a.data))
    te = D["sl"]["test"]
    actual = D["y_raw"][te]

    # deterministic sources
    res = {"Yesterday": [run_source(D["yest"][te], actual)],
           "Last week": [run_source(D["week"][te], actual)],
           "Oracle (actual load)": [run_source(actual, actual)]}
    # NN over seeds
    res["NN"] = []
    fc_first = None
    for k in range(a.seeds):
        m = nf.train_nn(D, seed=cfg.SEED + k)
        fc = nf._unscale(D, m.predict(D["X"][te]))
        fc_first = fc if fc_first is None else fc_first
        res["NN"].append(run_source(fc, actual))
        print(f"seed {cfg.SEED + k} done")

    rows = []
    for name, lst in res.items():
        df = pd.DataFrame(lst)
        for col in df.columns:
            rows.append({"forecast": name, "metric": col,
                         "mean": df[col].mean(),
                         "std": df[col].std(ddof=0)})
    summ = pd.DataFrame(rows)
    wide = summ.pivot(index="forecast", columns="metric", values="mean").round(3)
    wide_std = summ.pivot(index="forecast", columns="metric", values="std").round(3)
    summ.to_csv(f"{OUT}/end_to_end.csv", index=False)

    csp, bat = algo_tables(fc_first[:a.algo_days], actual[:a.algo_days])
    csp.to_csv(f"{OUT}/csp_counters.csv")
    bat.to_csv(f"{OUT}/battery_counters.csv")

    with open(f"{OUT}/summary.md", "w", encoding="utf-8") as f:
        if a.synthetic:
            f.write("**DEV ONLY - synthetic data, never report these numbers.**\n\n")
        f.write(f"## A. End to end (mean over test days; NN: mean over {a.seeds} seeds)\n\n")
        f.write(wide.to_markdown() + "\n\nStd over seeds (0 = no randomness):\n\n")
        f.write(wide_std.to_markdown() + "\n\n## B. Appliance CSP counters (avg per day)\n\n")
        f.write(csp.to_markdown() + "\n\n## C. Battery search counters (avg per day)\n\n")
        f.write(bat.to_markdown() + "\n")
    print(open(f"{OUT}/summary.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
