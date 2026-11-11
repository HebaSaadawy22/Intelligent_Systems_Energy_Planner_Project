"""Battery search tests: UCS and A* must match dynamic programming."""
import random

import pytest

from config import PRICES, BATTERY
from search_battery import plan_battery, dp_battery, heuristic, INF

CAP, RATE = BATTERY["capacity_kwh"], BATTERY["max_rate_kwh_per_h"]


def rand_case(seed):
    rnd = random.Random(seed)
    load = [float(rnd.randint(1, 4)) for _ in range(24)]  # whole kWh
    price = [round(rnd.uniform(0.08, 0.4), 3) for _ in range(24)]
    return load, price


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("algo", ["ucs", "astar"])
def test_search_matches_dp(seed, algo):
    load, price = rand_case(seed)
    res = plan_battery(load, price, CAP, RATE, 0, 0, 10.0, algo=algo)
    dp, _ = dp_battery(load, price, CAP, RATE, 0, 0, 10.0)
    assert res["cost"] == pytest.approx(dp, abs=1e-6)


def test_small_known_answer():
    # 2 hours: cheap then expensive, load 1 kWh each, capacity 1, max rate 1
    load, price = [1.0, 1.0], [0.1, 0.5]
    res = plan_battery(load, price, 1, 1, 0, 0, 5.0, algo="ucs")
    # charge in hour 0 (buy 2 @0.1 = 0.2), discharge in hour 1 (buy 0)
    assert res["cost"] == pytest.approx(0.2)
    assert res["actions"] == [1, -1]


def test_no_battery_equals_plain_cost():
    load, price = rand_case(1)
    res = plan_battery(load, price, 0, 0, 0, 0, 10.0)
    assert res["cost"] == pytest.approx(sum(l * p for l, p in zip(load, price)))


@pytest.mark.parametrize("seed", range(5))
def test_heuristic_never_overestimates(seed):
    load, price = rand_case(seed)
    _, V = dp_battery(load, price, CAP, RATE, 0, 0, 10.0)
    for h in range(24):
        for l in range(CAP + 1):
            if V[h][l] < INF:
                assert heuristic(load, price, h, l) <= V[h][l] + 1e-9


def test_astar_expands_no_more_than_ucs_on_example():
    load = [2.0] * 24
    res_u = plan_battery(load, PRICES, CAP, RATE, 0, 0, 10.0, "ucs")
    res_a = plan_battery(load, PRICES, CAP, RATE, 0, 0, 10.0, "astar")
    assert res_a["expanded"] <= res_u["expanded"]


def test_impossible_grid_limit_returns_none():
    # load 3 kWh every hour but grid limit 2 and no battery -> infeasible
    assert plan_battery([3.0] * 24, PRICES, 0, 0, 0, 0, 2.0) is None
