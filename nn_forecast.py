"""nn_forecast.py - day-ahead hourly load forecast (NN) + baselines.

Predicts the 24 hourly loads of day D from:
  - the 24 hourly loads of day D-1   ("yesterday")
  - the 24 hourly loads of day D-7   ("last week")
  - day-of-week (sin/cos) and month (sin/cos)

Baselines: yesterday (copy D-1), last week (copy D-7).
Split is chronological (as in README): last 365 days = test,
90 days before = validation, rest = train.

Usage:
    python nn_forecast.py                      # uses data/hourly.csv
    python nn_forecast.py --seeds 5            # mean/std over 5 seeds
    python nn_forecast.py --check              # self-test on fake data (DEV ONLY)

From other code:
    from nn_forecast import prepare, train_nn, evaluate, predict_day
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

try:  # use the project's config when available
    import config as cfg
except Exception:  # pragma: no cover
    cfg = None

SEED = getattr(cfg, "SEED", 42)
DATA_CSV = os.path.join("data", "hourly.csv")
LOAD_COL = getattr(cfg, "LOAD_COL", None)  # optional name of the load column
TEST_DAYS, VAL_DAYS = 365, 90


# ---------------------------------------------------------------- data
def load_hourly(path=DATA_CSV):
    """Return a pd.Series of hourly load indexed by timestamp (regular, no gaps)."""
    df = pd.read_csv(path)
    # find the timestamp column
    ts_col = None
    for c in df.columns:
        if str(c).lower() in ("timestamp", "datetime", "date_time", "time", "date"):
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    # find the load column
    if LOAD_COL and LOAD_COL in df.columns:
        col = LOAD_COL
    else:
        pref = [c for c in df.columns if any(k in str(c).lower() for k in
                ("load", "active_power", "consumption", "kwh", "kw"))]
        num = df.select_dtypes("number").columns.tolist()
        col = (pref or num)[0]
    s = df[col].astype(float)
    s = s[~s.index.duplicated()].asfreq("h").interpolate(limit_direction="both")
    return s


def _day_features(day):
    dow, mon = day.dayofweek, day.month
    return [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)]


def make_days(series):
    """Build per-day arrays. Returns dates, X, y, yesterday, lastweek."""
    s = series.copy()
    s = s[s.index.normalize().isin(
        s.groupby(s.index.normalize()).count().loc[lambda c: c == 24].index)]
    days = sorted(set(s.index.normalize()))
    mat = {d: s[s.index.normalize() == d].values for d in days}
    dates, X, y, yest, week = [], [], [], [], []
    for d in days:
        d1, d7 = d - pd.Timedelta(days=1), d - pd.Timedelta(days=7)
        if d1 in mat and d7 in mat:
            dates.append(d)
            y.append(mat[d])
            yest.append(mat[d1])
            week.append(mat[d7])
            X.append(np.concatenate([mat[d1], mat[d7], _day_features(d)]))
    return pd.DatetimeIndex(dates), np.array(X), np.array(y), np.array(yest), np.array(week)


def prepare(series):
    """Chronological split + scaling. Returns a dict used by the other functions."""
    dates, X, y, yest, week = make_days(series)
    n = len(dates)
    if n < TEST_DAYS + VAL_DAYS + 60:
        raise ValueError(f"Only {n} usable days; need > {TEST_DAYS + VAL_DAYS + 60}.")
    i_te, i_va = n - TEST_DAYS, n - TEST_DAYS - VAL_DAYS
    sl = {"train": slice(0, i_va), "val": slice(i_va, i_te), "test": slice(i_te, n)}
    mu, sd = y[sl["train"]].mean(), y[sl["train"]].std() + 1e-9  # scale loads only
    Xs = X.copy()
    Xs[:, :48] = (Xs[:, :48] - mu) / sd
    ys = (y - mu) / sd
    return dict(dates=dates, X=Xs, y=ys, y_raw=y, yest=yest, week=week,
                mu=mu, sd=sd, sl=sl)


# --------------------------------------------------------------- model
def train_nn(D, seed=SEED, max_epochs=200, patience=15):
    """Train an MLP with manual early stopping on the validation split."""
    rng = np.random.RandomState(seed)
    m = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                     learning_rate_init=1e-3, batch_size=32, random_state=seed)
    tr, va = D["sl"]["train"], D["sl"]["val"]
    Xtr, ytr, Xva, yva = D["X"][tr], D["y"][tr], D["X"][va], D["y"][va]
    best, best_m, bad = np.inf, None, 0
    for _ in range(max_epochs):
        idx = rng.permutation(len(Xtr))
        m.partial_fit(Xtr[idx], ytr[idx])
        v = np.mean(np.abs(m.predict(Xva) - yva))
        if v < best - 1e-5:
            best, best_m, bad = v, copy.deepcopy(m), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_m


def _unscale(D, p):
    return p * D["sd"] + D["mu"]


def predict_day(model, D, i):
    """24 hourly forecasts (original units) for the day at index i of D['dates']."""
    return _unscale(D, model.predict(D["X"][i:i + 1]))[0]


def _metrics(y, p):
    return {"MAE": float(np.mean(np.abs(y - p))),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2)))}


def evaluate(model, D, split="test"):
    s = D["sl"][split]
    y = D["y_raw"][s]
    return {"NN": _metrics(y, _unscale(D, model.predict(D["X"][s]))),
            "Yesterday": _metrics(y, D["yest"][s]),
            "Last week": _metrics(y, D["week"][s])}


# ---------------------------------------------------------------- main
def run(path=DATA_CSV, seeds=1):
    D = prepare(load_hourly(path))
    rows = []
    for k in range(seeds):
        res = evaluate(train_nn(D, seed=SEED + k), D)
        for name, m in res.items():
            rows.append({"seed": SEED + k, "model": name, **m})
    df = pd.DataFrame(rows)
    return df.groupby("model")[["MAE", "RMSE"]].agg(["mean", "std"]).round(4)


def _fake_series(years=4, seed=0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2018-01-01", periods=24 * 365 * years, freq="h")
    h, d = idx.hour.values, idx.dayofweek.values
    base = 1 + 0.8 * np.exp(-((h - 19) ** 2) / 8) + 0.4 * np.exp(-((h - 8) ** 2) / 6)
    base *= 1 + 0.15 * (d >= 5) + 0.2 * np.cos(2 * np.pi * idx.dayofyear.values / 365)
    return pd.Series(base + 0.1 * rng.randn(len(idx)), index=idx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_CSV)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="DEV ONLY: fake data")
    a = ap.parse_args()
    if a.check:
        D = prepare(_fake_series())
        print(pd.DataFrame(evaluate(train_nn(D), D)).T.round(4))
        print("DEV ONLY - fake data, never report these numbers.")
    else:
        print(run(a.data, a.seeds))
