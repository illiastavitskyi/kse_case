# main.py  —  python main.py [volunteer|official|splice]
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "splice"   # splice (офіційні + волонтерські після 7.09) | official | volunteer
T0 = pd.Timestamp("2026-01-01")
EXCLUDE = {"Луганська область"}
CITY = "м. Київ"
MAX_DUR_H = 72
CAP_H = 720
WARMUP_H = 24 * 7
VAL_DAYS = int(os.environ.get("VAL_DAYS", 10))
TEST_DAYS = int(os.environ.get("TEST_DAYS", 14))
TAG = os.environ.get("TAG", "_t14")
USE_SRC_FLAG = os.environ.get("SRC_FLAG", "0") == "1"
SPLICE_CUT = pd.Timestamp.max
ALERT_THRESHOLD = 0.26   # «тривога очікується»: recall>=65% на val при мінімальному FPR
LEVEL_CUTS = [0.15, ALERT_THRESHOLD, 0.40]
LEVEL_NAMES = ["низька", "підвищена", "тривога очікується", "висока ймовірність"]
OUT_DIR = "artifacts"

CENTROIDS = {
    "Вінницька область": (49.23, 28.47), "Волинська область": (51.00, 24.70),
    "Дніпропетровська область": (48.45, 34.98), "Донецька область": (48.00, 37.80),
    "Житомирська область": (50.25, 28.65), "Закарпатська область": (48.60, 22.60),
    "Запорізька область": (47.50, 35.60), "Івано-Франківська область": (48.90, 24.70),
    "Київська область": (50.20, 30.40), "Кіровоградська область": (48.50, 32.30),
    "Львівська область": (49.60, 23.70), "Миколаївська область": (47.60, 32.00),
    "Одеська область": (46.50, 30.70), "Полтавська область": (49.60, 34.50),
    "Рівненська область": (50.70, 26.20), "Сумська область": (50.90, 34.80),
    "Тернопільська область": (49.50, 25.60), "Харківська область": (49.50, 36.50),
    "Херсонська область": (46.60, 33.40), "Хмельницька область": (49.40, 26.90),
    "Черкаська область": (49.40, 31.80), "Чернівецька область": (48.30, 25.90),
    "Чернігівська область": (51.50, 32.00), CITY: (50.45, 30.52),
}

OBLAST_NEIGHBORS = {
    "Вінницька область": ["Житомирська область", "Хмельницька область", "Черкаська область",
                          "Кіровоградська область", "Одеська область", "Чернівецька область"],
    "Волинська область": ["Рівненська область", "Львівська область"],
    "Дніпропетровська область": ["Запорізька область", "Донецька область", "Харківська область",
                                 "Полтавська область", "Кіровоградська область", "Миколаївська область"],
    "Донецька область": ["Харківська область", "Дніпропетровська область", "Запорізька область"],
    "Житомирська область": ["Київська область", "Чернігівська область", "Рівненська область",
                            "Хмельницька область", "Вінницька область", "Черкаська область"],
    "Закарпатська область": ["Львівська область", "Івано-Франківська область"],
    "Запорізька область": ["Дніпропетровська область", "Донецька область", "Херсонська область"],
    "Івано-Франківська область": ["Львівська область", "Тернопільська область",
                                  "Чернівецька область", "Закарпатська область"],
    "Київська область": ["Житомирська область", "Чернігівська область", "Полтавська область",
                         "Черкаська область", CITY],
    "Кіровоградська область": ["Черкаська область", "Полтавська область", "Дніпропетровська область",
                               "Миколаївська область", "Одеська область", "Вінницька область"],
    "Львівська область": ["Волинська область", "Рівненська область", "Тернопільська область",
                          "Івано-Франківська область", "Закарпатська область"],
    "Миколаївська область": ["Одеська область", "Херсонська область",
                             "Кіровоградська область", "Дніпропетровська область"],
    "Одеська область": ["Миколаївська область", "Кіровоградська область",
                        "Вінницька область", "Чернівецька область"],
    "Полтавська область": ["Харківська область", "Сумська область", "Чернігівська область",
                           "Київська область", "Черкаська область", "Кіровоградська область",
                           "Дніпропетровська область"],
    "Рівненська область": ["Волинська область", "Львівська область", "Тернопільська область",
                           "Хмельницька область", "Житомирська область"],
    "Сумська область": ["Харківська область", "Полтавська область", "Чернігівська область"],
    "Тернопільська область": ["Рівненська область", "Хмельницька область", "Чернівецька область",
                              "Івано-Франківська область", "Львівська область"],
    "Харківська область": ["Сумська область", "Полтавська область",
                           "Дніпропетровська область", "Донецька область"],
    "Херсонська область": ["Миколаївська область", "Запорізька область"],
    "Хмельницька область": ["Житомирська область", "Вінницька область", "Тернопільська область",
                            "Чернівецька область", "Рівненська область"],
    "Черкаська область": ["Київська область", "Полтавська область", "Кіровоградська область",
                          "Вінницька область", "Житомирська область"],
    "Чернівецька область": ["Одеська область", "Вінницька область", "Хмельницька область",
                            "Тернопільська область", "Івано-Франківська область"],
    "Чернігівська область": ["Київська область", "Житомирська область",
                             "Сумська область", "Полтавська область"],
    CITY: ["Київська область"],
}

FEATURES = [
    "hour", "hour_sin", "hour_cos", "dow", "sun_elev_deg",
    "since_active_h", "since_last_onset_h", "gap_ratio",
    "onset_rate_7d", "onset_rate_30d", "onset_cnt_24h",
    "national_active_units", "nat_onset_cnt_24h", "nat_onset_rate_72h",
    "nat_onset_rate_7d", "nat_regime_ratio",
    "nb_active_share_1h", "nb_active_share_2h", "nb_active_share_3h",
    "nb_onset_share_1h", "nb_onset_share_6h",
]

if USE_SRC_FLAG:
    FEATURES.append("src_volunteer")


# ---------------------------------------------------------------- дані
def find(name: str) -> str:
    for base in ("datasets", "."):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def to_kyiv(s: pd.Series) -> pd.Series:
    return (pd.to_datetime(s, utc=True)
              .dt.tz_convert("Europe/Kyiv")
              .dt.tz_localize(None))


def norm_region(r: str) -> str:
    r = str(r).strip()
    return CITY if r in {"Київ", "місто Київ", "м. Київ"} else r


def load_official() -> pd.DataFrame:
    df = pd.read_csv(find("official_data_uk.csv"))
    return pd.DataFrame({
        "oblast": df["oblast"].map(norm_region),
        "start": to_kyiv(df["started_at"]),
        "end": to_kyiv(df["finished_at"]),
    })


def load_volunteer() -> pd.DataFrame:
    df = pd.read_csv(find("volunteer_data_uk.csv"))
    naive = df["naive"].astype(str).str.lower().isin(["true", "1"])
    print(f"Волонтерські: naive (вигаданий кінець) {naive.mean():.3%} рядків")
    return pd.DataFrame({
        "oblast": df["region"].map(norm_region),
        "start": to_kyiv(df["started_at"]),
        "end": to_kyiv(df["finished_at"]),
    })


def load_events(now=None):
    if SOURCE == "official":
        ev = load_official()
    elif SOURCE == "volunteer":
        ev = load_volunteer()
    elif SOURCE == "splice":
        off, vol = load_official(), load_volunteer()
        global SPLICE_CUT
        cut = off["start"].max()
        SPLICE_CUT = cut
        ev = pd.concat([off, vol[vol["start"] > cut]], ignore_index=True)
        print(f"Склейка: official до {cut}, volunteer після")
    else:
        raise ValueError("SOURCE: volunteer | official | splice")

    ev = ev[~ev["oblast"].isin(EXCLUDE)]
    ev = ev[(ev["start"] >= T0) & (ev["end"] > ev["start"])].copy()

    dur_h = (ev["end"] - ev["start"]).dt.total_seconds() / 3600
    capped = dur_h > MAX_DUR_H
    ev.loc[capped, "end"] = ev.loc[capped, "start"] + pd.Timedelta(hours=MAX_DUR_H)

    if now is None:
        # навчання: остання година даних може бути неповною — не використовуємо її
        t_end = ev["start"].max().floor("h")
    else:
        # сервіс: T = поточна година, але не далі за межу даних (щоб не вважати прогалину "тишею")
        t_end = min(pd.Timestamp(now).floor("h"), ev["start"].max().floor("h") + pd.Timedelta(hours=1))
    ev = ev[ev["start"] < t_end]
    print(f"Режим: {SOURCE}. Події: {len(ev)}, обрізано >{MAX_DUR_H} год: {int(capped.sum())}, "
          f"дані до {t_end}")
    return ev[["oblast", "start", "end"]].drop_duplicates(), t_end


def merge_intervals(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["unit", "start"]).reset_index(drop=True)
    prev_max_end = ev.groupby("unit")["end"].transform(lambda s: s.cummax().shift())
    block = (prev_max_end.isna() | (ev["start"] > prev_max_end)).cumsum()
    out = ev.groupby(block).agg(unit=("unit", "first"), start=("start", "min"), end=("end", "max"))
    return out.reset_index(drop=True)


def build_units(ev: pd.DataFrame) -> pd.DataFrame:
    units = merge_intervals(ev.rename(columns={"oblast": "unit"}))
    print(f"Одиниць: {units['unit'].nunique()}, інтервалів після злиття: {len(units)}")
    return units


# ---------------------------------------------------------------- фічі
def build_adjacency(names: list) -> np.ndarray:
    pairs = set()
    for a, bs in OBLAST_NEIGHBORS.items():
        for b in bs:
            pairs.add((a, b))
            pairs.add((b, a))
    U = len(names)
    W = np.zeros((U, U), dtype=np.float32)
    for i, oi in enumerate(names):
        for j, oj in enumerate(names):
            if i != j and (oi, oj) in pairs:
                W[i, j] = 1.0
    deg = W.sum(axis=1, keepdims=True)
    return np.divide(W, deg, out=np.zeros_like(W), where=deg > 0)


def sun_elevation(H: pd.DatetimeIndex, names: list) -> np.ndarray:
    """Висота сонця (градуси) для кожної години і області. Спрощена формула NOAA."""
    loc = H.tz_localize("Europe/Kyiv", ambiguous="NaT", nonexistent="NaT")
    utc = pd.Series(loc.tz_convert("UTC").tz_localize(None))
    utc = utc.fillna(pd.Series(H - pd.Timedelta(hours=3)))
    d = ((utc - pd.Timestamp("2000-01-01 12:00")) / pd.Timedelta("1D")).to_numpy(dtype=float)
    L = (280.460 + 0.9856474 * d) % 360
    g = np.radians((357.528 + 0.9856003 * d) % 360)
    lam = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.0000004 * d)
    delta = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    gmst = np.radians(((18.697374558 + 24.06570982441908 * d) * 15) % 360)
    coords = np.array([CENTROIDS[u] for u in names], dtype=float)
    lat, lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    ha = gmst[:, None] + lon[None, :] - ra[:, None]
    sin_el = (np.sin(lat)[None, :] * np.sin(delta)[:, None]
              + np.cos(lat)[None, :] * np.cos(delta)[:, None] * np.cos(ha))
    return np.degrees(np.arcsin(np.clip(sin_el, -1, 1))).astype(np.float32)


def build_panel(units: pd.DataFrame, t_end: pd.Timestamp):
    H = pd.date_range(T0, t_end, freq="h", inclusive="left")
    n = len(H)
    names = sorted(units["unit"].unique())
    U = len(names)
    ui = pd.Series(np.arange(U), index=names)[units["unit"]].to_numpy()

    s_h = ((units["start"] - T0) / pd.Timedelta("1h")).to_numpy()
    e_h = ((units["end"] - T0) / pd.Timedelta("1h")).to_numpy()

    # A[T] = 1, якщо тривога триває в момент T
    a_s = np.clip(np.floor(s_h).astype(int) + 1, 0, n)
    a_e = np.clip(np.ceil(e_h).astype(int), 0, n)
    diff = np.zeros((n + 1, U), dtype=np.int32)
    np.add.at(diff, (a_s, ui), 1)
    np.add.at(diff, (a_e, ui), -1)
    A = pd.DataFrame((np.cumsum(diff, axis=0)[:n] > 0).astype(np.int8), index=H, columns=names)

    # O[T] = 1, якщо тривога починається в [T, T+1h)
    o_idx = np.floor(s_h).astype(int)
    keep = (o_idx >= 0) & (o_idx < n)
    O_arr = np.zeros((n, U), dtype=np.int8)
    O_arr[o_idx[keep], ui[keep]] = 1
    O = pd.DataFrame(O_arr, index=H, columns=names)

    # Усі ознаки використовують стан не пізніше T-1
    a1 = A.shift(1).fillna(0)
    a2 = A.shift(2).fillna(0)
    a3 = A.shift(1).rolling(3, min_periods=1).max().fillna(0)
    o1 = O.shift(1).fillna(0)

    W = build_adjacency(names)
    nb_active_1h = a1.to_numpy(np.float32) @ W.T
    nb_active_2h = a2.to_numpy(np.float32) @ W.T
    nb_active_3h = a3.to_numpy(np.float32) @ W.T
    nb_onset_1h = o1.to_numpy(np.float32) @ W.T
    nb_onset_6h = o1.rolling(6, min_periods=1).sum().to_numpy(np.float32) @ W.T

    idx = np.arange(n)[:, None]
    last_act = pd.DataFrame(np.where(a1.to_numpy() > 0, idx, np.nan), index=H, columns=names).ffill().to_numpy()
    since_active = np.where(np.isnan(last_act), CAP_H, np.minimum(idx - last_act, CAP_H))
    last_on = pd.DataFrame(np.where(o1.to_numpy() > 0, idx, np.nan), index=H, columns=names).ffill().to_numpy()
    since_last_onset = np.where(np.isnan(last_on), CAP_H, np.minimum(idx - last_on, CAP_H))

    # Медіана міжпочаткових інтервалів за 30 днів, тільки з минулих подій
    gap_med = np.full((n, U), CAP_H, dtype=np.float32)
    for j in range(U):
        on_idx = np.flatnonzero(O_arr[:, j])
        if len(on_idx) < 3:
            continue
        gaps = np.full(n, np.nan)
        gaps[on_idx[1:]] = np.diff(on_idx)
        med = pd.Series(gaps, index=H).rolling("720h", min_periods=2).median().shift(1)
        gap_med[:, j] = med.fillna(CAP_H).to_numpy()
    gap_ratio = np.clip(since_last_onset / (gap_med + 1), 0, 10)

    r7 = o1.rolling(168, min_periods=24).mean().fillna(0).to_numpy()
    r30 = o1.rolling(720, min_periods=168).mean().fillna(0).to_numpy()
    onset_cnt_24h = o1.rolling(24, min_periods=1).sum().to_numpy()

    national_active = np.repeat(a1.sum(axis=1).to_numpy()[:, None], U, axis=1)
    nat_on = o1.sum(axis=1)
    nat_24 = nat_on.rolling(24, min_periods=1).sum().to_numpy() / U
    nat_72 = nat_on.rolling(72, min_periods=24).mean().to_numpy() / U
    nat_7 = nat_on.rolling(168, min_periods=24).mean().to_numpy() / U
    nat_30 = nat_on.rolling(720, min_periods=168).mean().to_numpy() / U
    nat_ratio = np.nan_to_num(np.clip((nat_7 + 1e-4) / (nat_30 + 1e-4), 0, 5))

    def bc(v):
        return np.repeat(np.nan_to_num(np.asarray(v, dtype=np.float32))[:, None], U, axis=1)

    hr = H.hour.to_numpy()
    feats = {
        "hour": np.broadcast_to(hr[:, None], (n, U)),
        "hour_sin": np.broadcast_to(np.sin(2 * np.pi * hr / 24)[:, None], (n, U)),
        "hour_cos": np.broadcast_to(np.cos(2 * np.pi * hr / 24)[:, None], (n, U)),
        "dow": np.broadcast_to(H.dayofweek.to_numpy()[:, None], (n, U)),
        "sun_elev_deg": sun_elevation(H, names),
        "src_volunteer": np.broadcast_to((H > SPLICE_CUT).astype(np.float32)[:, None], (n, U)),
        "since_active_h": since_active,
        "since_last_onset_h": since_last_onset,
        "gap_ratio": gap_ratio,
        "onset_rate_7d": r7,
        "onset_rate_30d": r30,
        "onset_cnt_24h": onset_cnt_24h,
        "national_active_units": national_active,
        "nat_onset_cnt_24h": bc(nat_24),
        "nat_onset_rate_72h": bc(nat_72),
        "nat_onset_rate_7d": bc(nat_7),
        "nat_regime_ratio": bc(nat_ratio),
        "nb_active_share_1h": nb_active_1h,
        "nb_active_share_2h": nb_active_2h,
        "nb_active_share_3h": nb_active_3h,
        "nb_onset_share_1h": nb_onset_1h,
        "nb_onset_share_6h": nb_onset_6h,
    }
    X = pd.DataFrame({k: np.asarray(v, dtype=np.float32).ravel() for k, v in feats.items()})
    X["unit_id"] = np.tile(np.arange(U), n).astype(np.int32)
    X["t"] = np.repeat(np.arange(n), U)
    X["y"] = O.to_numpy().ravel()
    X["active_now"] = A.to_numpy().ravel()
    X = X[(X["active_now"] == 0) & (X["t"] >= WARMUP_H)].reset_index(drop=True)
    return X, n, names


# ---------------------------------------------------------------- модель і звіти
def split(X: pd.DataFrame, n: int):
    test_start = n - TEST_DAYS * 24
    val_start = test_start - VAL_DAYS * 24
    if val_start <= WARMUP_H:
        raise ValueError("Замало даних для val/test")
    return (X[X["t"] < val_start],
            X[(X["t"] >= val_start) & (X["t"] < test_start)],
            X[X["t"] >= test_start])


def train(tr, va):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1)
    dtr = lgb.Dataset(tr[FEATURES], tr["y"])
    dva = lgb.Dataset(va[FEATURES], va["y"], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"\nBest iteration: {model.best_iteration}")
    return model


def score(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": brier_score_loss(y, p), "logloss": log_loss(y, p),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p)}


def baseline_report(tr, te, p_model):
    rows = {"constant": np.full(len(te), tr["y"].mean()),
            "onset_rate_30d": te["onset_rate_30d"].to_numpy(),
            "lightgbm": p_model}
    print("\nBaseline порівняння (test):")
    for name, p in rows.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        print(f"  {name:16s} pr_auc={average_precision_score(te['y'], p):.4f} "
              f"brier={brier_score_loss(te['y'], p):.4f}")


def monthly_calibration(te, p):
    hours = T0 + pd.to_timedelta(te["t"].to_numpy(), unit="h")
    df = pd.DataFrame({"month": hours.to_period("M"), "y": te["y"].to_numpy(), "p": p})
    print("\nКалібрування по місяцях (test):")
    print(df.groupby("month").agg(rows=("y", "size"), observed=("y", "mean"),
                                  predicted=("p", "mean")).round(4).to_string())


def level_report(y, p, title):
    y = np.asarray(y, dtype=int)
    lvl = pd.cut(np.asarray(p), bins=[-0.001, *LEVEL_CUTS, 1.0001], labels=LEVEL_NAMES, right=False)
    df = pd.DataFrame({"level": lvl, "y": y})
    rep = df.groupby("level", observed=False).agg(hours=("y", "size"), observed_rate=("y", "mean"),
                                                  onsets=("y", "sum"))
    rep["share_hours"] = rep["hours"] / len(df)
    rep["share_onsets"] = rep["onsets"] / max(int(y.sum()), 1)
    rep["lift"] = rep["observed_rate"] / max(y.mean(), 1e-9)
    print(f"\n{title} (база onset={y.mean():.4f})")
    print(rep[["hours", "share_hours", "observed_rate", "lift", "share_onsets"]].round(4).to_string())


def per_oblast(te, p, names):
    df = pd.DataFrame({"unit": [names[i] for i in te["unit_id"]], "y": te["y"].to_numpy(), "p": p})
    rows = []
    for u, g in df.groupby("unit"):
        if g["y"].nunique() < 2:
            continue
        rows.append({"oblast": u, "rows": len(g), "base": g["y"].mean(),
                     "pr_auc": average_precision_score(g["y"], g["p"]),
                     "roc_auc": roc_auc_score(g["y"], g["p"])})
    print("\nМетрики по областях (test):")
    print(pd.DataFrame(rows).sort_values("base", ascending=False).round(3).to_string(index=False))


def leakage_check(units, t_end, X_full, n):
    """Дописуємо до історії події після Tcut -> фічі рядків до Tcut не мають змінитися."""
    cut_t = n - 15 * 24
    cut_ts = T0 + pd.Timedelta(hours=cut_t)
    X_cut, _, _ = build_panel(units[units["start"] < cut_ts], t_end)
    a = X_full[X_full["t"] < cut_t].set_index(["t", "unit_id"])
    b = X_cut[X_cut["t"] < cut_t].set_index(["t", "unit_id"])
    common = a.index.intersection(b.index)
    cols = FEATURES + ["y"]
    diff = np.abs(a.loc[common, cols].to_numpy(np.float64) - b.loc[common, cols].to_numpy(np.float64))
    print(f"\nLeakage-перевірка: рядків {len(common)}, макс. розбіжність фіч {np.nanmax(diff):.2e} "
          f"({'OK' if np.nanmax(diff) < 1e-5 else 'ПРОБЛЕМА'})")


def main():
    ev, t_end = load_events()
    units = build_units(ev)
    X, n, names = build_panel(units, t_end)
    tr, va, te = split(X, n)

    bounds = lambda d: (T0 + pd.Timedelta(hours=int(d['t'].min())), T0 + pd.Timedelta(hours=int(d['t'].max())))
    for name, d in [("Train", tr), ("Val", va), ("Test", te)]:
        lo, hi = bounds(d)
        print(f"{name:5s} {len(d):6d} рядків, онсетів {d['y'].mean():.4f}, {lo:%Y-%m-%d} — {hi:%Y-%m-%d}")

    model = train(tr, va)
    p_va = model.predict(va[FEATURES], num_iteration=model.best_iteration)
    p_te = model.predict(te[FEATURES], num_iteration=model.best_iteration)

    s = score(te["y"].to_numpy(), p_te)
    print("\nTest metrics:")
    for k, v in s.items():
        print(f"  {k:10s} {v:.4f}")

    baseline_report(tr, te, p_te)
    monthly_calibration(te, p_te)
    level_report(va["y"], p_va, "VAL: рівні ризику")
    level_report(te["y"], p_te, "TEST: рівні ризику (межі задані заздалегідь)")
    per_oblast(te, p_te, names)

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES)
    print("\nFeature importance (gain):")
    print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())

    leakage_check(units, t_end, X, n)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_model(os.path.join(OUT_DIR, f"model_{SOURCE}{TAG}.txt"), num_iteration=model.best_iteration)
    meta = {"source": SOURCE, "data_from": str(T0), "data_to": str(t_end),
            "train_to": str(bounds(tr)[1]), "val": [str(x) for x in bounds(va)],
            "test": [str(x) for x in bounds(te)], "features": FEATURES, "units": names,
            "level_cuts": LEVEL_CUTS, "level_names": LEVEL_NAMES,
            "best_iteration": int(model.best_iteration), "test_metrics": s}
    with open(os.path.join(OUT_DIR, f"meta_{SOURCE}{TAG}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nЗбережено: {OUT_DIR}/model_{SOURCE}{TAG}.txt, meta_{SOURCE}{TAG}.json")


if __name__ == "__main__":
    main()
