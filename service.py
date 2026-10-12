"""Обчислення прогнозу: дані -> ознаки на момент T -> збережена модель. Без перенавчання."""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], "splice"]
import main as m  # noqa: E402

m.SOURCE = "splice"
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model_splice_t14.txt")
META_PATH = os.environ.get("META_PATH", "artifacts/meta_splice_t14.json")

ALERT_THRESHOLD = 0.26  # «тривога очікується»: recall>=65% на val при мінімальному FPR
LEVEL_CUTS = [0.15, ALERT_THRESHOLD, 0.40]
LEVEL_NAMES = ["низька", "підвищена", "тривога очікується", "висока ймовірність"]

DATA_URL = os.environ.get(
    "DATA_URL",
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_uk.csv")
VOLUNTEER_FILE = "volunteer_data_uk.csv"
KYIV = ZoneInfo("Europe/Kyiv")

_lock = threading.Lock()
_cache: dict | None = None
_status = {"refresh_error": None, "downloaded_at": None}


def now_kyiv() -> pd.Timestamp:
    return pd.Timestamp.now(tz=KYIV).tz_localize(None)


def refresh_data() -> None:
    """Підтягує свіжий волонтерський CSV. Офіційний файл потрібен лише до 07.09 і не змінюється."""
    try:
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "kse-alerts"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
        if raw.count(b"\n") < 10000:
            raise ValueError("файл підозріло малий")
        target = os.path.join(os.path.dirname(m.find(VOLUNTEER_FILE)) or ".", VOLUNTEER_FILE)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(target)))
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        os.replace(tmp, target)
        _status.update(refresh_error=None, downloaded_at=now_kyiv())
    except Exception as e:  # лишаємо попередній файл
        _status["refresh_error"] = f"{type(e).__name__}: {e}"


def _level(p: float) -> str:
    return str(pd.cut([p], bins=[-0.001, *LEVEL_CUTS, 1.0001], labels=LEVEL_NAMES, right=False)[0])


def compute() -> dict:
    """Один прогноз на годину T = кінець даних. Ознаки в T використовують стан не пізніше T-1."""
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    now = now_kyiv()
    ev, t_end = m.load_events(now=now)
    units = m.build_units(ev)
    # панель до t_end + 1h: останній рядок = година T = t_end; її y невідомий і не використовується
    X, n, names = m.build_panel(units, t_end + pd.Timedelta(hours=1))
    last = X[X["t"] == n - 1]
    p = model.predict(last[model.feature_name()])
    pred = dict(zip((names[int(i)] for i in last["unit_id"]), p))

    rows = []
    for name in names:
        if name in pred:
            pv = float(pred[name])
            rows.append({"oblast": name, "p": pv, "level": _level(pv),
                         "expected": pv >= ALERT_THRESHOLD, "active": False})
        else:  # на момент T тривога вже триває — задача «початок» не визначена
            rows.append({"oblast": name, "p": None, "level": "триває тривога",
                         "expected": False, "active": True})
    rows.sort(key=lambda r: (r["p"] is None, -(r["p"] or 0)))

    return {
        "now": now, "lag_h": (now.floor("h") - t_end) / pd.Timedelta("1h"),
        "updated_at": now, "downloaded_at": _status["downloaded_at"], "refresh_error": _status["refresh_error"],
        "forecast_for": t_end, "window_end": t_end + pd.Timedelta(hours=1),
        "rows": rows, "threshold": ALERT_THRESHOLD,
        "levels": list(zip(LEVEL_NAMES, [0.0, *LEVEL_CUTS])),
        "metrics": meta["test_metrics"], "test_period": meta["test"],
        "train_to": meta["train_to"], "splice_cut": m.SPLICE_CUT,
    }


def update() -> None:
    """Завантажити дані й перерахувати прогноз; при збої лишається попередній."""
    global _cache
    refresh_data()
    try:
        res = compute()
        with _lock:
            _cache = res
    except Exception as e:
        _status["refresh_error"] = f"compute: {type(e).__name__}: {e}"
        if _cache is not None:
            _cache = {**_cache, "refresh_error": _status["refresh_error"]}


def get_forecast() -> dict:
    with _lock:
        if _cache is None:
            raise RuntimeError("прогноз ще не готовий")
        return _cache


def _loop() -> None:
    while True:
        nxt = (now_kyiv().floor("h") + pd.Timedelta(hours=1, minutes=2))   # +2 хв: дати файлу оновитись
        time.sleep(max(30, (nxt - now_kyiv()).total_seconds()))
        update()


def start() -> None:
    update()
    threading.Thread(target=_loop, daemon=True).start()
