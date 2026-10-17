"""Seeded controls, explicitly not exchange data or evidence of profitability."""
import numpy as np
from .model import sigmoid, logit


def generate(n=1200, seed=21, scenario="edge"):
    if scenario not in {"edge", "null", "drift"}:
        raise ValueError("Unknown scenario")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ts = 1704067200+i*21600
        p = float(sigmoid(rng.normal(0, 1.35)))
        signal = float(rng.normal())
        coefficient = 0. if scenario == "null" else (.9 if scenario == "edge" or i < n//2 else -.9)
        true = float(sigmoid(logit(p)+coefficient*signal))
        y = int(rng.random() < true)
        rows.append({"market_id":f"{scenario}-{i}", "event_id":f"synthetic:{scenario}-{i}",
                     "venue":"polymarket" if i%2 else "kalshi", "source":"SYNTHETIC_CONTROL",
                     "ts":ts, "feature_available_ts":ts-1, "scheduled_close_ts":ts+86400,
                     "market_p":p, "evidence_p":float(sigmoid(logit(p)+signal)),
                     "evidence_available_ts":ts-300, "y":y, "label_available_ts":ts+90000,
                     "yes_ask":min(p+.01, .99999), "no_ask":min(1-p+.01, .99999),
                     "quote_ts":ts-1, "spread":.02, "momentum":0.,
                     "yes_depth":100, "no_depth":100, "fee_rate":.07, "fee_available_ts":ts-3600})
    return rows
