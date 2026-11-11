# P2 Home Energy Planner - Report

**Forecast -> appliance CSP -> battery search.**
All prices, costs and savings are **SIMULATED** (generated in `config.py`). Only the household load (UCI dataset) is real.

## 1. Problem
Plan one day (24 hours) for a single household: forecast the load, schedule flexible appliances under a grid power limit, then decide when to charge/discharge a small battery to reduce the (simulated) electricity cost.

## 2. Data
- UCI household power consumption, aggregated to hourly load (`make_data.py` -> `data/hourly.csv`).
- Chronological split: last 365 days test, 90 days before validation, rest train. No shuffling.
- Generated (not measured): hourly prices (cheap night 0.10, day 0.18, evening peak 0.35), appliance specs, circuits, battery, power limit (5 kW). See `config.py`.

## 3. Method

### 3.1 Forecast (`nn_forecast.py`)
- Task: predict the 24 hourly loads of day D.
- Input: load of D-1 (24 values), load of D-7 (24 values), day-of-week and month (sin/cos). Loads standardised with train statistics.
- Model: MLP 48+4 -> 64 -> 32 -> 24 (ReLU, L2 = 1e-3, Adam), manual early stopping on validation MAE (patience 15).
- Baselines: **yesterday** (copy D-1) and **last week** (copy D-7).
- Metrics: MAE, RMSE on the test year; mean/std over 5 seeds.

### 3.2 Appliance CSP (`csp_appliances.py`)
- Variables: start hour of each appliance. Domain: start hours in the appliance window where forecast + power <= limit.
- Constraints: washer before dryer; no overlap on the same circuit (kitchen, heavy); total load <= limit in every hour; forbidden schedules (feedback).
- Algorithms: backtracking (BT), forward checking (FC), AC-3; counters: assignments, backtracks, AC-3 revisions.
- If there is no solution, the limit is relaxed in 10% steps (up to 50%) and the violation is reported; if still impossible, a "NO SOLUTION" message lists the windows that cannot fit.

### 3.3 Battery search (`search_battery.py`)
- State = (hour, battery level in whole kWh); action = charge/discharge/idle by up to 2 kWh/h; no export to the grid; the grid limit is a hard limit.
- Cost = price[h] x grid purchase[h] (simulated).
- Algorithms: UCS and A* with the heuristic "remaining load not covered by the battery x cheapest remaining price" (admissible). Counters: nodes expanded / generated.
- Checks: result compared with backward dynamic programming (DP); heuristic admissibility test.

### 3.4 Integration and feedback loop (`integration.py`)
1. CSP schedules appliances on the forecast.
2. Battery search plans on forecast + appliances with the **original** grid limit.
3. If the battery plan is impossible (typically when the CSP only succeeded by relaxing the limit), that schedule is added to the forbidden list and the CSP is solved again. At most 5 loops (`MAX_FEEDBACK_LOOPS`).
4. If no schedule is accepted, the result is `failed_after_max_loops` with a message. If the CSP has no solution, the result is `no_solution`.

Evaluation uses the **actual** load of the day: the planned appliance schedule and battery actions are executed on it, grid purchase is clipped at 0 (no export), and cost and hours above the limit are measured. Baseline "unplanned": every appliance starts at its earliest hour, no battery.

## 4. Experiments (`experiments.py`)
Run: `python make_data.py && python experiments.py` (5 seeds). Output in `results/`: `summary.md`, `end_to_end.csv`, `csp_counters.csv`, `battery_counters.csv`.

Compared forecast sources: NN, yesterday, last week, and an oracle that knows the actual load (upper bound).

### 4.1 Forecast accuracy
> Fill from `python nn_forecast.py --seeds 5` (MAE / RMSE, mean +- std).

### 4.2 End-to-end results
> Paste section A of `results/summary.md` (cost saving vs unplanned, hours over the limit, loops, relaxed days).

### 4.3 CSP algorithms
> Paste section B of `results/summary.md` (BT vs FC vs FC+AC-3 counters).

### 4.4 Battery search
> Paste section C of `results/summary.md` (UCS vs A*: expanded/generated; `matches_dp` should be 1).

## 5. Discussion
> Write after seeing the real numbers. Suggested points: Does the NN beat both baselines? How close is each forecast source to the oracle? How often did the limit need relaxing, and did the feedback loop ever trigger? How much does A* save over UCS, and did AC-3 reduce backtracks?

## 6. Limitations
- One household only; no holidays; no real price data (all prices and savings are simulated).
- Battery levels are integer kWh; battery has no efficiency loss and no export.
- Appliance schedules are fixed per day (no re-planning during the day).
- The forecast uses only the previous day and the previous week as input (no weather).

## 7. How to reproduce
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # needs numpy, pandas, scikit-learn, tabulate, pytest
python make_data.py
python -m pytest -q
python nn_forecast.py --seeds 5
python run_demo.py
python experiments.py
```
