"""CSP tests: 5 normal, 3 edge, 2 impossible."""
import pytest

from config import APPLIANCES, CIRCUITS, ORDER_CONSTRAINTS
from csp_appliances import ApplianceCSP, plan_appliances

# forecast: ~3.4 kW at 18:00-20:00 (evening peak), 1.0 kW otherwise
FORECAST = [1.0] * 24
for h in (18, 19, 20):
    FORECAST[h] = 3.4
LIMIT = 5.0

MODES = [(False, False), (True, False), (False, True), (True, True)]


def make(forecast=FORECAST, limit=LIMIT, apps=APPLIANCES, forbidden=None):
    return ApplianceCSP(forecast, apps, limit, CIRCUITS, ORDER_CONSTRAINTS, forbidden)


def check_valid(csp, sched):
    load = csp.load_profile(sched)
    assert max(load) <= csp.limit + 1e-9
    assert sched["dryer"] >= sched["washer"] + APPLIANCES["washer"]["duration"]
    for grp in CIRCUITS.values():
        for i, a in enumerate(grp):
            for b in grp[i + 1:]:
                sa, sb = sched[a], sched[b]
                assert (sa + APPLIANCES[a]["duration"] <= sb
                        or sb + APPLIANCES[b]["duration"] <= sa)


# ---------------- 5 normal ----------------
@pytest.mark.parametrize("fc,ac3", MODES)
def test_normal_all_modes_find_valid_schedule(fc, ac3):
    csp = make()
    sol = csp.solve(use_fc=fc, use_ac3=ac3)
    assert sol is not None
    check_valid(csp, sol)


def test_normal_dishwasher_moved_out_of_peak():
    # 1.8 kW dishwasher at 18:00 on top of 3.4 kW would break 5 kW -> must move
    csp = make()
    sol = csp.solve(use_fc=True, use_ac3=True)
    span = range(sol["dishwasher"], sol["dishwasher"] + 2)
    assert all(FORECAST[h] + 1.8 <= LIMIT + 1e-9 for h in span)


def test_normal_all_modes_agree_on_feasibility():
    results = [make().solve(use_fc=fc, use_ac3=ac3) is not None for fc, ac3 in MODES]
    assert all(results)


def test_normal_fc_not_worse_than_plain_bt():
    a, b = make(), make()
    a.solve(use_fc=False)
    b.solve(use_fc=True)
    assert b.backtracks <= a.backtracks


def test_normal_forbidden_combination_changes_solution():
    first = make().solve(use_fc=True, use_ac3=True)
    forbid = [{"washer": first["washer"], "dryer": first["dryer"]}]
    csp = make(forbidden=forbid)
    second = csp.solve(use_fc=True, use_ac3=True)
    assert second is not None
    assert (second["washer"], second["dryer"]) != (first["washer"], first["dryer"])


# ---------------- 3 edge ----------------
def test_edge_single_appliance():
    csp = make(apps={"oven": APPLIANCES["oven"]})
    sol = csp.solve(use_fc=True, use_ac3=True)
    assert sol is not None and 11 <= sol["oven"] <= 21


def test_edge_limit_exactly_equal_to_load():
    # forecast 1.0 + oven 2.2 = 3.2 exactly the limit -> still allowed
    csp = make(forecast=[1.0] * 24, limit=3.2, apps={"oven": APPLIANCES["oven"]})
    assert csp.solve(use_fc=True) is not None


def test_edge_window_of_exactly_one_start():
    apps = {"oven": {"power": 2.2, "duration": 1, "earliest": 12, "latest_end": 13}}
    csp = make(apps=apps)
    assert csp.solve(use_fc=True, use_ac3=True) == {"oven": 12}


# ---------------- 2 impossible ----------------
def test_impossible_limit_below_forecast_gives_clear_message():
    res = plan_appliances([2.0] * 24, APPLIANCES, 1.0, CIRCUITS, ORDER_CONSTRAINTS,
                          relax_steps=2)
    assert res["status"] == "no_solution"
    assert "NO SOLUTION" in res["message"]
    assert res["windows_that_cannot_fit"]


def test_impossible_order_cycle_ac3_detects():
    apps = {
        "a": {"power": 0.5, "duration": 2, "earliest": 0, "latest_end": 4},
        "b": {"power": 0.5, "duration": 2, "earliest": 0, "latest_end": 4},
    }
    csp = ApplianceCSP([1.0] * 24, apps, 5.0, {}, [("a", "b"), ("b", "a")])
    assert csp.solve(use_fc=True, use_ac3=True) is None


# ---------------- relaxation (10% steps) ----------------
def test_relaxation_reports_violation():
    # at 3.0 kW limit the 3.3 kW EV cannot run alone; 10% steps reach it
    res = plan_appliances([0.0] * 24, {"ev": APPLIANCES["ev"]}, 3.0, {}, [])
    assert res["status"] == "relaxed"
    assert res["violation_kw"] == pytest.approx(0.3)
