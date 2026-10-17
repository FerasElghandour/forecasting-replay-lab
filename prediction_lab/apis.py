"""Public GET-only API adapters. Immutable raw-response cache with provenance.

No keys, wallets, signatures, proxying, order submission, or account endpoints.
"""
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from .data import validate, write

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"


def epoch(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class PublicHTTP:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def get(self, base, path, params=None, refresh=False):
        if base not in {KALSHI, GAMMA, CLOB, DATA} or not path.startswith("/") or ".." in path:
            raise ValueError("Only allowlisted public hosts supported")
        allowed = ((base == KALSHI and path.startswith(("/markets", "/historical/markets", "/historical/cutoff", "/series/")))
                   or (base == GAMMA and path.startswith("/markets"))
                   or (base == CLOB and path in {"/book", "/prices-history"})
                   or (base == DATA and path == "/v2/prices-history"))
        if not allowed:
            raise ValueError("Endpoint outside public research allowlist")
        url = base + path + ("?"+urlencode(params) if params else "")
        key = hashlib.sha256(url.encode()).hexdigest()
        target = self.cache / (key+".json")
        if target.exists() and not refresh:
            return json.loads(target.read_text())["response"]
        if refresh:
            target = self.cache / (key+"-"+str(time.time_ns())+".json")
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "prediction-lab/0.1 read-only research"}), timeout=30) as f:
                    raw = f.read()
                result = json.loads(raw)
                record = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                          "sha256_body": hashlib.sha256(raw).hexdigest(), "response": result}
                target.write_text(json.dumps(record, sort_keys=True))
                time.sleep(.06)
                return result
            except HTTPError as e:
                if e.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
                time.sleep(min(float(e.headers.get("Retry-After", 2**attempt)), 10))
        raise RuntimeError("Request retry limit")


def number(obj, field):
    if obj.get(field+"_dollars") is not None:
        return float(obj[field+"_dollars"])
    if obj.get(field) is not None:
        return float(obj[field])/100
    return None


def kalshi_row(m, candles, horizon_hours=24, lag_seconds=60):
    """Fixed horizon before rule-defined close, not before observed settlement.

    Caller must audit that close_time was fixed at listing. Final metadata is
    never used as a feature except this audited schedule and the event identity.
    Candle close is available no earlier than end_period_ts + explicit lag.
    """
    if m.get("result") not in {"yes", "no"} or m.get("is_provisional", False):
        raise ValueError("unverified_or_nonbinary_settlement")
    if not m.get("settlement_ts"):
        raise ValueError("missing_settlement_timestamp")
    close = epoch(m["close_time"])
    ts = close-horizon_hours*3600
    eligible = sorted([c for c in candles if c["end_period_ts"]+lag_seconds <= ts], key=lambda c:c["end_period_ts"])
    if not eligible or ts-eligible[-1]["end_period_ts"] > 7200:
        raise ValueError("missing_or_stale_candle")
    c = eligible[-1]
    bid, ask = number(c.get("yes_bid", {}), "close"), number(c.get("yes_ask", {}), "close")
    if bid is None or ask is None or not 0 < bid <= ask < 1:
        raise ValueError("missing_or_invalid_two_sided_quote")
    p = (bid+ask)/2
    old = eligible[-7] if len(eligible) >= 7 else eligible[0]
    ob, oa = number(old.get("yes_bid", {}), "close"), number(old.get("yes_ask", {}), "close")
    momentum = p-(ob+oa)/2 if ob is not None and oa is not None else 0.
    return {"market_id": m["ticker"], "event_id": "kalshi:"+m["event_ticker"], "venue": "kalshi",
            "ts": ts, "feature_available_ts": c["end_period_ts"]+lag_seconds,
            "scheduled_close_ts": close, "market_p": p, "y": int(m["result"] == "yes"),
            "label_available_ts": epoch(m["settlement_ts"])+lag_seconds,
            "yes_ask": ask, "no_ask": 1-bid, "quote_ts": c["end_period_ts"],
            "spread": ask-bid, "momentum": momentum,
            "source": "kalshi_hourly_candle_close", "schedule_assumption": "close_time_fixed_in_rules_audit_required"}


def collect_kalshi(http, series, out, pages=2, horizon_hours=24):
    if series != "KXHIGHNY":
        raise ValueError("Automatic importer currently limited to KXHIGHNY fixed local-midnight close; audit new series first")
    markets = {}
    cutoff = http.get(KALSHI, "/historical/cutoff")
    # Include both storage tiers. Keep all raw pages and de-duplicate ticker overlap.
    for path, params in [("/historical/markets", {"series_ticker":series, "limit":1000}),
                         ("/markets", {"series_ticker":series, "status":"settled", "limit":1000})]:
        cursor = None
        for _ in range(pages):
            query = dict(params, **({"cursor": cursor} if cursor else {}))
            result = http.get(KALSHI, path, query)
            for m in result["markets"]:
                markets[m["ticker"]] = (m, path.startswith("/historical"))
            cursor = result.get("cursor")
            if not cursor:
                break
    events = defaultdict(list)
    for m, historical in markets.values():
        events[m["event_ticker"]].append((m, historical))
    rows, excluded, errors = [], Counter(), []
    for j, (event, members) in enumerate(sorted(events.items())):
        # Current event endpoint efficiently returns all sibling buckets.
        current = [m for m, historical in members if not historical]
        event_candles = {}
        if current:
            ts = epoch(current[0]["close_time"])-horizon_hours*3600
            result = http.get(KALSHI, f"/series/{quote(series)}/events/{quote(event)}/candlesticks",
                              {"start_ts":int(ts-10*3600), "end_ts":int(ts), "period_interval":60})
            event_candles = dict(zip(result["market_tickers"], result["market_candlesticks"]))
        for m, historical in members:
            try:
                candles = event_candles.get(m["ticker"])
                if candles is None:
                    ts = epoch(m["close_time"])-horizon_hours*3600
                    path = f"/historical/markets/{quote(m['ticker'])}/candlesticks" if historical else f"/series/{quote(series)}/markets/{quote(m['ticker'])}/candlesticks"
                    result = http.get(KALSHI, path, {"start_ts":int(ts-10*3600), "end_ts":int(ts), "period_interval":60})
                    candles = result["candlesticks"]
                rows.append(kalshi_row(m, candles, horizon_hours))
            except ValueError as e:
                excluded[str(e)] += 1
            except HTTPError as e:
                errors.append({"ticker":m["ticker"], "http_status":e.code})
        if (j+1)%20 == 0:
            print(f"Processed {j+1}/{len(events)} events; accepted {len(rows)} snapshots", flush=True)
    rows = validate(rows)
    write(out, rows)
    manifest = {"series":series, "horizon_hours":horizon_hours, "markets_discovered":len(markets),
                "events_discovered":len(events), "accepted_rows":len(rows), "excluded":dict(excluded),
                "http_errors":errors, "cutoff_response":cutoff, "pages_per_tier_limit":pages,
                "selection": "API order bounded by pages; not a complete universe; settled survivors only",
                "schedule": "KXHIGHNY rules fix last trading at 11:59 PM local; supplied close_time assumed faithful",
                "historical_latency_seconds_assumed":60, "historical_depth_and_fee_vintages": "unavailable"}
    Path(str(out)+".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def polymarket_assets(m):
    def array(x):
        return json.loads(x) if isinstance(x, str) else x
    labels = array(m["outcomes"])
    if len(labels) != 2 or {s.lower() for s in labels} != {"yes", "no"}:
        raise ValueError("Only explicit Yes/No markets supported")
    version = m.get("version", "v1")
    if version not in {"v1", "v2"}:
        raise ValueError("Unsupported Polymarket version")
    ids = array(m["positionIds"] if version == "v2" else m["clobTokenIds"])
    if len(ids) != 2:
        raise ValueError("Missing outcome identifiers")
    return {label.lower(): asset for label, asset in zip(labels, ids)}


def polymarket_history(http, asset, start, end, legacy=False):
    """Current Data API v2 with <=15-day chunks and cursor pagination.

    Legacy CLOB reader is explicit, never an unnoticed schema fallback.
    Returned points are prices only, never fabricated historical bid/ask quotes.
    """
    if start >= end:
        raise ValueError("History start must precede end")
    if legacy:
        d = http.get(CLOB, "/prices-history", {"market":asset, "startTs":int(start), "endTs":int(end), "fidelity":60})
        return [{"ts":int(p["t"]), "price":float(p["p"]), "source":"legacy_clob_history"} for p in d["history"]]
    points = {}
    cursor_start = int(start)
    while cursor_start < end:
        stop = min(cursor_start+15*86400, int(end))
        params = {"token_id":asset, "start":cursor_start, "end":stop}
        seen = set()
        while True:
            d = http.get(DATA, "/v2/prices-history", params)
            for p in d["data"]:
                points[p["timestamp"]] = {"ts":int(p["timestamp"]), "price":float(p["price"]),
                                          "resolution_seconds":p.get("resolution_seconds"), "source":"data_api_v2"}
            pagination = d["pagination"]
            if not pagination["has_more"]:
                break
            cursor = pagination["next_cursor"]
            if not cursor or cursor in seen:
                raise ValueError("Repeated/missing history cursor")
            seen.add(cursor)
            params = dict(params, cursor=cursor)
        cursor_start = stop
    return sorted(points.values(), key=lambda p:p["ts"])


def polymarket_row(audit, points, horizon_hours=24):
    """Join prices to independently verified labels; prices alone never imply y.

    The audit is a user-supplied record with rule/vintage evidence. Enforcing
    timestamps cannot prove that the supplied sources themselves are truthful.
    """
    if not audit.get("resolution_evidence") or not audit.get("schedule_evidence"):
        raise ValueError("Independent final resolution and schedule evidence required")
    ts = audit["scheduled_close_ts"]-horizon_hours*3600
    if audit.get("schedule_available_ts", float("inf")) > ts:
        raise ValueError("Schedule not known at forecast time")
    def available(p):
        # Conservative interval-end assumption until a documented point-time
        # convention is established. Never read a partly observed bucket.
        return p["ts"]+p.get("resolution_seconds", 3600)+60
    eligible = sorted([p for p in points if available(p) <= ts], key=available)
    if not eligible or ts-available(eligible[-1]) > 12*3600:
        raise ValueError("Insufficient pre-forecast history")
    p = eligible[-1]
    row = {k:audit[k] for k in ["market_id", "event_id", "scheduled_close_ts", "y", "label_available_ts"]}
    row.update(venue="polymarket", ts=ts, feature_available_ts=available(p), market_p=p["price"],
               source="polymarket_history_with_audited_label", resolution_evidence=audit["resolution_evidence"],
               schedule_evidence=audit["schedule_evidence"])
    return validate([row])[0]
