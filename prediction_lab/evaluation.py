"""Event-purged expanding-window evaluation with label-availability embargo."""
from collections import defaultdict
import numpy as np
from .data import validate, weights
from .model import Forecaster, loss


def stages(pool, origin, embargo=3600, minimum=12):
    groups = defaultdict(list)
    for r in pool:
        groups[r["event_id"]].append(r)
    # Reject partially resolved events, including sibling buckets with late labels.
    groups = {e: rs for e, rs in groups.items() if max(r["label_available_ts"] for r in rs) < origin-embargo}
    events = sorted(groups, key=lambda e: min(r["ts"] for r in groups[e]))
    if len(events) < minimum * 5:
        return None
    a, b = int(.6*len(events)), int(.8*len(events))
    cal_start = min(r["ts"] for e in events[a:b] for r in groups[e])
    sel_start = min(r["ts"] for e in events[b:] for r in groups[e])
    train = [r for e in events[:a] for r in groups[e]
             if max(v["label_available_ts"] for v in groups[e]) < cal_start-embargo]
    cal = [r for e in events[a:b] for r in groups[e]
           if max(v["label_available_ts"] for v in groups[e]) < sel_start-embargo]
    select = [r for e in events[b:] for r in groups[e]]
    if any(len({r["event_id"] for r in rs}) < minimum or len({r["y"] for r in rs}) < 2
           for rs in (train, cal, select)):
        return None
    return train, cal, select


def metrics(rows, key):
    p, y, w = np.array([r[key] for r in rows]), np.array([r["y"] for r in rows]), weights(rows)
    bins = []
    for i in range(10):
        mask = (p >= i/10) & (p < (i+1)/10 if i < 9 else p <= 1)
        if mask.any():
            bins.append({"lower": i/10, "n": int(mask.sum()), "weight": float(w[mask].sum()),
                         "predicted": float(np.average(p[mask], weights=w[mask])),
                         "observed": float(np.average(y[mask], weights=w[mask]))})
    return {"brier": float(np.average((p-y)**2, weights=w)), "log_loss": loss(p, rows),
            "ece_10_bins": float(sum(b["weight"]*abs(b["predicted"]-b["observed"]) for b in bins)/w.sum()),
            "reliability": bins}


def paired_ci(rows, cluster="event", seed=42, draws=2000):
    groups = defaultdict(list)
    for r in rows:
        k = r["event_id"] if cluster == "event" else int(r["ts"]//(86400*7))
        groups[k].append(r)
    # Event-weighted sums inside each resampled cluster; preserves unequal bucket counts.
    all_w = weights(rows)
    sums = defaultdict(lambda: [0., 0.])
    for r, w in zip(rows, all_w):
        k = r["event_id"] if cluster == "event" else int(r["ts"]//(86400*7))
        sums[k][0] += w*((r["market"]-r["y"])**2-(r["forecast"]-r["y"])**2)
        sums[k][1] += w
    vals = np.array(list(sums.values()))
    rng = np.random.default_rng(seed)
    sample = vals[rng.integers(len(vals), size=(draws, len(vals)))].sum(axis=1)
    delta = sample[:, 0]/sample[:, 1]
    return {"cluster": cluster, "clusters": len(vals), "draws": draws,
            "brier_improvement_ci95": np.quantile(delta, [.025, .975]).tolist(),
            "warning": "Exploratory; few clusters and regime dependence weaken coverage"}


def backtest(rows, folds=4, warmup=.5, embargo=3600, minimum=12):
    rows = validate(rows)
    times = sorted({r["ts"] for r in rows})
    if len(times) < folds+2 or not 0 < warmup < 1 or folds < 1:
        raise ValueError("Insufficient dates or invalid backtest configuration")
    start = int(len(times)*warmup)
    chunks = [c.tolist() for c in np.array_split(np.array(times[start:]), folds) if len(c)]
    predictions, audit, last_model = [], [], None
    # Events spanning an outer boundary are assigned exclusively to their first test fold.
    tested_events = set()
    for i, chunk in enumerate(chunks):
        origin, end = chunk[0], chunk[-1]
        test = [r for r in rows if origin <= r["ts"] <= end and r["event_id"] not in tested_events]
        te = {r["event_id"] for r in test}
        tested_events |= te
        pool = [r for r in rows if r["ts"] < origin and r["event_id"] not in te]
        split = stages(pool, origin, embargo, minimum)
        entry = {"fold": i, "origin": origin, "end": end, "test_rows": len(test), "test_events": len(te)}
        if not test:
            continue
        if split is None:
            market = np.array([r["market_p"] for r in test])
            ps = {k: market for k in ["market", "calibrated_market", "raw_model", "calibrated_model", "forecast"]}
            entry.update({"status": "insufficient_history_market_fallback", "alpha": 0.})
        else:
            train, cal, select = split
            model = Forecaster().fit(train, cal, select)
            ps = model.predict(test)
            entry.update({"status": "fitted", "alpha": model.alpha,
                          "stages": [{"name": name, "rows": len(rs), "events": len({r["event_id"] for r in rs}),
                                      "max_label_available_ts": max(r["label_available_ts"] for r in rs),
                                      "min_forecast_ts": min(r["ts"] for r in rs)}
                                     for name, rs in zip(["train", "calibration", "selection"], split)]})
            last_model = model
        for j, r in enumerate(test):
            predictions.append(dict(r, **{k: float(v[j]) for k, v in ps.items()}, fold=i,
                                    fitted=split is not None))
        audit.append(entry)
    if not predictions:
        raise ValueError("No out-of-sample predictions")
    scores = {k: metrics(predictions, k) for k in ["market", "calibrated_market", "raw_model", "calibrated_model", "forecast"]}
    for entry in audit:
        fold_rows = [r for r in predictions if r["fold"] == entry["fold"]]
        entry["scores"] = {k:metrics(fold_rows, k) for k in ["market", "forecast"]}
    improvement = scores["market"]["brier"] - scores["forecast"]["brier"]
    report = {"rows": len(predictions), "events": len({r["event_id"] for r in predictions}),
              "fitted_rows": sum(r["fitted"] for r in predictions), "scores": scores,
              "brier_skill": 1-scores["forecast"]["brier"]/scores["market"]["brier"] if scores["market"]["brier"] else None,
              "brier_improvement": improvement, "uncertainty": [paired_ci(predictions, c) for c in ["event", "calendar_week"]],
              "folds": audit, "config": {"warmup": warmup, "embargo_seconds": embargo, "minimum_stage_events": minimum},
              "by_venue": {v: {k: metrics([r for r in predictions if r["venue"] == v], k)
                               for k in ["market", "forecast"]} for v in sorted({r["venue"] for r in predictions})}}
    return predictions, report, last_model
