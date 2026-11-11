# P2 Home Energy Planner: Report

*Max 12 pages. All prices, costs and savings in this project are SIMULATED. Only the household load is real.*

## 1. Problem

Plan a household's next day of electricity use: forecast the base load, choose start hours for flexible appliances (washer, dryer, dishwasher, EV charger, oven) under a power limit, and plan a home battery to minimise the simulated electricity cost.

Users: households and energy advisers (simulation only).

## 2. Data

- Source: UCI Individual Household Electric Power Consumption (one house near Paris, 2006-2010, CC BY 4.0).
- Preprocessing (`make_data.py`, seed 42): minutes aggregated to hourly mean kW; gaps of up to 3 hours interpolated; days with remaining gaps dropped; chronological split (final 365 days test, 90 days before it validation, rest train).
- Real vs generated: see `data/data_dictionary.md`. Prices, appliance profiles, power limit and battery are generated in `config.py`.

## 3. Design choices

### 3.1 Appliance CSP (`csp_appliances.py`)
- Variables: appliance start hours. Domains: start hours inside each appliance's window where forecast + appliance power stays under the limit.
- Constraints: dryer after washer, no overlap on a shared circuit, pairwise and total load under the limit, forbidden combinations (from the feedback loop).
- Three switchable parts: Backtracking, Forward Checking, AC-3. Counters: assignments, backtracks, AC-3 revisions.
- If no solution: list windows that cannot fit; relax the limit in 10% steps and report the violation.

### 3.2 Battery search (`search_battery.py`)
- State = (hour, battery level in kWh). Actions: charge, discharge or idle by up to the maximum rate. No export to the grid.
- Cost = price x energy bought from the grid (simulated).
- UCS and A*. A* heuristic: remaining energy not covered by the battery x cheapest remaining price. It never overestimates (tested against the exact cost-to-go from DP).
- Correctness check: optimal cost equals backward dynamic programming on random cases.

## 4. Neural network forecast

*TODO.* Inputs: previous 24 hourly loads, same hour last week, hour, weekday, month. Target: next-day hourly load (24 values). Baselines: same hour yesterday, same hour last week. Report test MAE/RMSE.

## 5. Integration

*TODO.* NN forecast -> CSP schedules appliances -> Search plans the battery. If the battery plan cannot stay under the limit, the clashing appliance combination is forbidden and the CSP runs again (up to 5 loops). Include evidence that the loop was triggered at least once.

## 6. Experiments

*TODO. At least 5 runs/seeds, mean and standard deviation.*
1. UCS vs A*: nodes and time over 30 days; heuristic never overestimates.
2. BT vs BT+FC vs BT+FC+AC-3 with 3, 5, 8 appliances.
3. NN forecast vs yesterday baseline: cost and limit violations using the real load.

## 7. Limits

- One household only; no holidays; no real price data.
- Costs and savings are simulated, not measured.
- Whole-kWh battery levels simplify real battery behaviour (no efficiency losses yet).
- *(add NN findings here once measured)*

## 8. Conclusions

*TODO after experiments.*
