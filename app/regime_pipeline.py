"""Regime prediction for the Delhi air-pollution project.

Takes one day of pollutant and weather readings and returns the nearest regime.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
_scaler = joblib.load(BASE / "models" / "scaler.pkl")
_kmeans = joblib.load(BASE / "models" / "kmeans_k5.pkl")
_config = json.loads((BASE / "models" / "regime_config.json").read_text(encoding="utf-8"))

NAMES = {int(k): v for k, v in _config["names"].items()}
REQUIRED = ["PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "O3", "CO", "Benzene", "temp", "rain", "wind_speed"]
LOG_COLS = ["PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "O3", "CO", "Benzene", "rain"]
NON_NEGATIVE = [c for c in REQUIRED if c != "temp"]


def _check(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    if raw[REQUIRED].isna().any().any():
        raise ValueError("Inputs contain missing values")
    if (raw[NON_NEGATIVE] < 0).any().any():
        raise ValueError("Pollutant, rain and wind inputs must be non-negative")
    if (raw["PM10"] <= 0).any():
        raise ValueError("PM10 must be positive")


def _prepare(raw):
    """Same steps as notebooks 04 and 05: PM ratio, log transform, scaling."""
    _check(raw)
    X = raw[REQUIRED].astype(float).copy()
    X["PM_ratio"] = (X["PM2.5"] / X["PM10"]).clip(upper=1.0)
    X[LOG_COLS] = np.log1p(X[LOG_COLS])
    # wind direction is not used by the final model; neutral values keep the scaler happy
    X["wind_dir_sin"] = 0.0
    X["wind_dir_cos"] = 0.0
    X = X[list(_scaler.feature_names_in_)]
    Z = pd.DataFrame(_scaler.transform(X), index=X.index, columns=X.columns)
    return Z[list(_kmeans.feature_names_in_)]


def predict_regimes(raw):
    """Many days at once. Returns cluster id, regime name and distance to every regime centre."""
    Z = _prepare(raw)
    dist = _kmeans.transform(Z)
    labels = _kmeans.predict(Z)
    out = pd.DataFrame({"cluster": labels, "regime": [NAMES[i] for i in labels]}, index=raw.index)
    for i in range(dist.shape[1]):
        out[f"dist_{NAMES[i]}"] = dist[:, i]
    return out


def predict_regime(**inputs):
    """One day. Distances are not probabilities; a small margin means a borderline day."""
    res = predict_regimes(pd.DataFrame([inputs])).iloc[0]
    dist = {NAMES[i]: float(res[f"dist_{NAMES[i]}"]) for i in range(len(NAMES))}
    ranked = sorted(dist.items(), key=lambda kv: kv[1])
    (first, d1), (second, d2) = ranked[0], ranked[1]
    return {
        "regime": first,
        "runner_up": second,
        "margin": round((d2 - d1) / d2, 3),
        "distances": {k: round(v, 2) for k, v in ranked},
    }
