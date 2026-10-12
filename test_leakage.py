"""Тести на витік даних з майбутнього. Запуск: pytest -q"""
import sys

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], "splice"]
import main as m  # noqa: E402

OBLASTS = ["Київська область", "Житомирська область", "Чернігівська область",
           "м. Київ", "Харківська область", "Полтавська область"]
FEATS = m.FEATURES + ["y"]


def synth_units(days=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for o in OBLASTS:
        t = m.T0 + pd.Timedelta(hours=float(rng.uniform(0, 10)))
        while t < m.T0 + pd.Timedelta(days=days):
            dur = pd.Timedelta(minutes=float(rng.uniform(10, 90)))
            rows.append((o, t, t + dur))
            t = t + dur + pd.Timedelta(hours=float(rng.exponential(8)))
    return pd.DataFrame(rows, columns=["unit", "start", "end"])


def _cmp(a, b):
    a = a.set_index(["t", "unit_id"])
    b = b.set_index(["t", "unit_id"])
    idx = a.index.intersection(b.index)
    assert len(idx) > 0
    return np.nanmax(np.abs(a.loc[idx, FEATS].to_numpy(np.float64) - b.loc[idx, FEATS].to_numpy(np.float64)))


def test_future_events_do_not_change_past_features():
    units = synth_units()
    t_end = m.T0 + pd.Timedelta(days=60)
    full, n, _ = m.build_panel(units, t_end)
    cut_t = n - 10 * 24
    cut_ts = m.T0 + pd.Timedelta(hours=cut_t)
    part, _, _ = m.build_panel(units[units["start"] < cut_ts], t_end)
    assert _cmp(full[full["t"] < cut_t], part[part["t"] < cut_t]) < 1e-5


def test_forecast_row_ignores_events_at_or_after_T():
    """Рядок T, який бере сервіс, однаковий з подіями після T і без них."""
    units = synth_units()
    T = m.T0 + pd.Timedelta(days=40)
    t_idx = int((T - m.T0) / pd.Timedelta("1h"))
    full, _, _ = m.build_panel(units, m.T0 + pd.Timedelta(days=60))
    past, _, _ = m.build_panel(units[units["start"] < T], T + pd.Timedelta(hours=1))
    a, b = full[full["t"] == t_idx], past[past["t"] == t_idx]
    assert len(b) > 0
    a = a[a["unit_id"].isin(b["unit_id"])]
    cols = m.FEATURES
    diff = np.abs(a.sort_values("unit_id")[cols].to_numpy(np.float64) - b.sort_values("unit_id")[cols].to_numpy(np.float64))
    assert np.nanmax(diff) < 1e-5


def test_chronological_split_has_no_overlap():
    units = synth_units(days=150)
    X, n, _ = m.build_panel(units, m.T0 + pd.Timedelta(days=150))
    tr, va, te = m.split(X, n)
    assert tr["t"].max() < va["t"].min() <= va["t"].max() < te["t"].min()


def test_alert_now_is_not_a_feature():
    assert "alert_now" not in m.FEATURES and "unit_id" not in m.FEATURES
