"""Strict, point-in-time snapshot contract. One predeclared horizon per market."""
import json
import math
from collections import Counter
from pathlib import Path

REQUIRED = {"market_id", "event_id", "venue", "ts", "feature_available_ts",
            "scheduled_close_ts", "market_p", "y", "label_available_ts"}


def validate(rows, labeled=True):
    seen = set()
    for r in rows:
        missing = REQUIRED - r.keys()
        if not labeled:
            missing -= {"y", "label_available_ts"}
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        key = (r["venue"], r["market_id"])
        if key in seen:
            raise ValueError("One snapshot per market required; choose a horizon before evaluation")
        seen.add(key)
        if not r["event_id"] or r["venue"] not in {"kalshi", "polymarket", "synthetic"}:
            raise ValueError("Event group and supported venue required")
        for k in ["ts", "feature_available_ts", "scheduled_close_ts", "market_p"]:
            if not isinstance(r[k], (int, float)) or not math.isfinite(r[k]):
                raise ValueError(f"Nonfinite/nonnumeric {k}")
        if not 0 < r["market_p"] < 1:
            raise ValueError("market_p must be strictly between 0 and 1")
        if not r["feature_available_ts"] <= r["ts"] < r["scheduled_close_ts"]:
            raise ValueError("Feature leakage or forecast after scheduled close")
        if labeled:
            if r["y"] not in (0, 1) or not math.isfinite(r["label_available_ts"]):
                raise ValueError("Only verified binary final settlements supported")
            if r["label_available_ts"] <= r["ts"]:
                raise ValueError("Outcome was already available at forecast time")
        if r.get("evidence_p") is not None:
            if not 0 < r["evidence_p"] < 1 or r.get("evidence_available_ts", float("inf")) > r["ts"]:
                raise ValueError("External forecast must have a point-in-time vintage")
        if r.get("yes_ask") is not None or r.get("no_ask") is not None:
            if not all(0 < r.get(k, -1) < 1 for k in ["yes_ask", "no_ask"]):
                raise ValueError("Both side asks required and must be inside (0,1)")
            if not r.get("quote_ts", float("inf")) <= r["ts"]:
                raise ValueError("Future quote")
            if r["yes_ask"] + r["no_ask"] < 1 - 1e-9:
                raise ValueError("Crossed/inconsistent quotes")
        for k in ["momentum", "spread", "yes_depth", "no_depth", "fee_rate"]:
            if r.get(k) is not None and not math.isfinite(r[k]):
                raise ValueError(f"Nonfinite {k}")
        if r.get("spread", 0) < 0 or r.get("fee_rate", 0) < 0:
            raise ValueError("Negative spread or fee")
        for k in ["yes_depth", "no_depth"]:
            if r.get(k, 0) < 0:
                raise ValueError("Negative depth")
    return sorted(rows, key=lambda r: (r["ts"], r["event_id"], r["market_id"]))


def read(path, labeled=True):
    return validate([json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()], labeled)


def write(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))


def weights(rows):
    """Each event has total weight one, including mutually exclusive buckets."""
    import numpy as np
    n = Counter(r["event_id"] for r in rows)
    return np.array([1 / n[r["event_id"]] for r in rows])
