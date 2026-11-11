"""Battery planning by search: UCS and A*, checked against dynamic programming.

State  = (hour, battery level in whole kWh)
Action = charge / discharge / idle by up to max_rate kWh in that hour
Grid purchase in hour h = load[h] + delta   (delta > 0 charge, delta < 0 discharge)
Rules  : 0 <= grid <= grid_limit (no export to the grid)
Cost   = price[h] * grid           (SIMULATED cost, prices are generated)
Goal   = hour == H and final level >= final_min
Counters: expanded, generated.
"""
from __future__ import annotations

import heapq
import itertools

EPS = 1e-9
INF = float("inf")


def transitions(hour, level, load, cap, max_rate, grid_limit):
    for delta in range(-max_rate, max_rate + 1):
        nl = level + delta
        if nl < 0 or nl > cap:
            continue
        grid = load[hour] + delta
        if grid < -EPS or grid > grid_limit + EPS:
            continue
        yield delta, nl, grid


def heuristic(load, price, hour, level):
    """Remaining energy not covered by the battery x cheapest remaining price.

    Admissible: energy bought later >= remaining load - current level,
    and every kWh costs at least the cheapest remaining price.
    """
    if hour >= len(load):
        return 0.0
    uncovered = max(0.0, sum(load[hour:]) - level)
    return uncovered * min(price[hour:])


def plan_battery(load, price, cap, max_rate, initial, final_min, grid_limit,
                 algo="ucs"):
    H = len(load)
    start = (0, initial)
    counters = {"expanded": 0, "generated": 1}
    use_h = algo == "astar"
    h0 = heuristic(load, price, 0, initial) if use_h else 0.0
    tie = itertools.count()
    pq = [(h0, next(tie), 0.0, start)]
    best = {start: 0.0}
    parent = {start: None}
    while pq:
        f, _, g, state = heapq.heappop(pq)
        if g > best.get(state, INF) + EPS:
            continue
        hour, level = state
        if hour == H:
            actions, grids = [], []
            s = state
            while parent[s] is not None:
                prev, delta, grid = parent[s]
                actions.append(delta)
                grids.append(grid)
                s = prev
            return {"cost": g, "actions": actions[::-1], "grid": grids[::-1],
                    **counters}
        counters["expanded"] += 1
        for delta, nl, grid in transitions(hour, level, load, cap, max_rate,
                                           grid_limit):
            nh = hour + 1
            if nh == H and nl < final_min:
                continue
            ng = g + price[hour] * grid
            ns = (nh, nl)
            if ng < best.get(ns, INF) - EPS:
                best[ns] = ng
                parent[ns] = (state, delta, grid)
                counters["generated"] += 1
                hh = heuristic(load, price, nh, nl) if use_h else 0.0
                heapq.heappush(pq, (ng + hh, next(tie), ng, ns))
    return None  # battery plan cannot stay under the grid limit


def dp_battery(load, price, cap, max_rate, initial, final_min, grid_limit):
    """Backward dynamic programming. Returns (optimal cost, cost-to-go table V)."""
    H = len(load)
    V = [[INF] * (cap + 1) for _ in range(H + 1)]
    for l in range(cap + 1):
        V[H][l] = 0.0 if l >= final_min else INF
    for h in range(H - 1, -1, -1):
        for l in range(cap + 1):
            for delta, nl, grid in transitions(h, l, load, cap, max_rate,
                                               grid_limit):
                v = price[h] * grid + V[h + 1][nl]
                if v < V[h][l]:
                    V[h][l] = v
    return V[0][initial], V
