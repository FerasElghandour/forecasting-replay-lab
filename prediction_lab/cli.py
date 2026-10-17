import argparse
import hashlib
import json
from pathlib import Path
import time
from . import data
from .apis import PublicHTTP, GAMMA, CLOB, polymarket_assets, polymarket_history, polymarket_row, collect_kalshi
from .evaluation import backtest, stages
from .model import Forecaster
from .paper import simulate
from .synthetic import generate


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def run_backtest(args):
    rows = data.read(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predictions, report, model = backtest(rows, folds=args.folds, minimum=args.minimum)
    report["input_sha256"] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    report["input_source_counts"] = dict(__import__("collections").Counter(r.get("source", "user_supplied") for r in rows))
    data.write(out/"predictions.jsonl", predictions)
    dump(out/"evaluation.json", report)
    paper = simulate(predictions, assume_depth=args.assume_depth, assume_fee_rate=args.assume_fee_rate,
                     max_quote_age=args.max_quote_age)
    dump(out/"paper.json", paper)
    stress = []
    for slip in [0., .005, .02]:
        s = simulate(predictions, slippage=slip, assume_depth=args.assume_depth,
                     assume_fee_rate=args.assume_fee_rate, max_quote_age=args.max_quote_age)
        stress.append({k:s[k] for k in ["net_pnl", "trade_count", "total_fees", "assumptions"]})
    dump(out/"cost_sensitivity.json", stress)
    if model:
        # This artifact is the LAST evaluated fold model, never a model fitted on test labels.
        model.save(out/"last_fold_model.json")
        origin = max(f["origin"] for f in report["folds"] if f["status"] == "fitted")
        dump(out/"last_fold_model.json.metadata.json", {"trained_asof":origin,
             "input_sha256":report["input_sha256"], "role":"last outer fold, not refit to test outcomes"})
    print(json.dumps({"rows":report["rows"], "events":report["events"], "fitted_rows":report["fitted_rows"],
                      "brier_market":report["scores"]["market"]["brier"],
                      "brier_forecast":report["scores"]["forecast"]["brier"],
                      "ci":report["uncertainty"], "paper_net_pnl":paper["net_pnl"], "trades":paper["trade_count"]}, indent=2))


def main():
    p = argparse.ArgumentParser(description="Public-data research and paper forecasts; no live trading")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--scenario", choices=["edge", "null", "drift"], default="edge")
    d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=21)
    d.add_argument("--n", type=int, default=1200)
    b = sub.add_parser("backtest")
    b.add_argument("--input", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--folds", type=int, default=4)
    b.add_argument("--minimum", type=int, default=12)
    b.add_argument("--assume-depth", action="store_true")
    b.add_argument("--assume-fee-rate", type=float)
    b.add_argument("--max-quote-age", type=float, default=300)
    k = sub.add_parser("collect-kalshi")
    k.add_argument("--series", default="KXHIGHNY")
    k.add_argument("--cache", default="data/raw")
    k.add_argument("--out", required=True)
    k.add_argument("--pages", type=int, default=1)
    k.add_argument("--horizon-hours", type=float, default=24)
    h = sub.add_parser("polymarket-history")
    h.add_argument("--market-id", required=True)
    h.add_argument("--start", type=int, required=True)
    h.add_argument("--end", type=int, required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--cache", default="data/raw")
    h.add_argument("--legacy", action="store_true")
    c = sub.add_parser("collect-polymarket")
    c.add_argument("--audit-manifest", required=True, help="JSON array of independently verified final labels and schedule vintages")
    c.add_argument("--out", required=True)
    c.add_argument("--cache", default="data/raw")
    s = sub.add_parser("polymarket-snapshot")
    s.add_argument("--market-id", required=True)
    s.add_argument("--event-id", required=True, help="Audited group ID shared by all related markets")
    s.add_argument("--scheduled-close", type=float, required=True, help="Rule-audited epoch time known now")
    s.add_argument("--cache", default="data/raw")
    s.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--input", required=True)
    f.add_argument("--as-of", type=float, required=True)
    f.add_argument("--out", required=True)
    q = sub.add_parser("predict")
    q.add_argument("--input", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "demo":
        data.write(args.out, generate(args.n, args.seed, args.scenario))
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "collect-kalshi":
        print(collect_kalshi(PublicHTTP(args.cache), args.series, args.out, args.pages, args.horizon_hours))
    elif args.command == "polymarket-history":
        http = PublicHTTP(args.cache)
        m = http.get(GAMMA, "/markets/"+args.market_id)
        asset = polymarket_assets(m)["yes"]
        data.write(args.out, polymarket_history(http, asset, args.start, args.end, args.legacy))
    elif args.command == "collect-polymarket":
        http = PublicHTTP(args.cache)
        audits = json.loads(Path(args.audit_manifest).read_text())
        rows = []
        for audit in audits:
            m = http.get(GAMMA, "/markets/"+str(audit["market_id"]))
            asset = polymarket_assets(m)["yes"]
            ts = audit["scheduled_close_ts"]-86400
            points = polymarket_history(http, asset, ts-2*86400, ts)
            rows.append(polymarket_row(audit, points))
        data.write(args.out, data.validate(rows))
    elif args.command == "polymarket-snapshot":
        http = PublicHTTP(args.cache)
        m = http.get(GAMMA, "/markets/"+args.market_id, refresh=True)
        assets = polymarket_assets(m)
        books = {s:http.get(CLOB, "/book", {"token_id":a}, refresh=True) for s,a in assets.items()}
        ts = time.time()
        row = {"market_id":args.market_id, "event_id":args.event_id, "venue":"polymarket", "ts":ts,
               "feature_available_ts":ts, "scheduled_close_ts":args.scheduled_close,
               "source":"public_orderbook_snapshot"}
        for side, book in books.items():
            if not book["asks"] or not book["bids"]:
                raise ValueError("Two-sided order book required")
            ask = min(float(v["price"]) for v in book["asks"])
            row[side+"_ask"] = ask
            row[side+"_depth"] = sum(float(v["size"]) for v in book["asks"] if float(v["price"]) == ask)
        yes_bid = max(float(v["price"]) for v in books["yes"]["bids"])
        row["market_p"] = (yes_bid+row["yes_ask"])/2
        row["spread"] = row["yes_ask"]-yes_bid
        row["quote_ts"] = min(float(book["timestamp"])/1000 for book in books.values())
        schedule = m.get("feeSchedule")
        if m.get("feesEnabled") is False:
            row.update(fee_rate=0., fee_available_ts=ts)
        elif schedule and schedule.get("exponent") == 1:
            row.update(fee_rate=float(schedule["rate"]), fee_available_ts=ts)
        data.write(args.out, data.validate([row], labeled=False))
    elif args.command == "fit":
        rows = data.read(args.input)
        split = stages([r for r in rows if r["ts"] < args.as_of], args.as_of)
        if split is None:
            raise ValueError("Insufficient resolved event history for honest train/calibration/selection")
        model = Forecaster().fit(*split)
        model.save(args.out)
        dump(args.out+".metadata.json", {"trained_asof":args.as_of, "stage_rows":[len(r) for r in split],
             "input_sha256":hashlib.sha256(Path(args.input).read_bytes()).hexdigest()})
    elif args.command == "predict":
        rows = data.read(args.input, labeled=False)
        meta = json.loads(Path(args.model+".metadata.json").read_text())
        if any(r["ts"] < meta["trained_asof"] for r in rows):
            raise ValueError("Model trained after requested forecast")
        ps = Forecaster.load(args.model).predict(rows)
        # Deliberately drop retrospective labels if present in input.
        data.write(args.out, [dict({k:v for k,v in r.items() if k not in {"y", "label_available_ts"}},
                                  **{k:float(v[j]) for k,v in ps.items()}, action="PAPER_FORECAST_ONLY")
                              for j,r in enumerate(rows)])


if __name__ == "__main__":
    main()
