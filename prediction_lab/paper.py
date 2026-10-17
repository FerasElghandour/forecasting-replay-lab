"""Conservative hold-to-settlement cash ledger. All fills are hypothetical."""
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
import heapq
import math


def fee(venue, price, qty, rate, quantum=None):
    """Explicit scenario rate; venue-specific precision is caller controlled.

    Kalshi default rounds total cost+fee up to cents conservatively. Current docs
    mention centicent precision, while their fee table still illustrates cents.
    Supply quantum='0.0001' for that interpretation and stress-test both.
    Polymarket current formula is rounded to five decimal places.
    """
    p, n, r = Decimal(str(price)), Decimal(str(qty)), Decimal(str(rate))
    f = n*r*p*(1-p)
    if venue == "polymarket":
        return float(f.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP))
    q = Decimal(quantum or "0.01")
    cost = n*p
    return float(((cost+f)/q).to_integral_value(rounding=ROUND_CEILING)*q-cost)


def simulate(rows, capital=1000., contracts=10, min_edge=.03, slippage=.005,
             max_event_cost=50., max_quote_age=300, assume_depth=False, assume_fee_rate=None):
    if capital <= 0 or contracts < 1 or int(contracts) != contracts or min_edge < 0 or slippage < 0 or max_event_cost <= 0:
        raise ValueError("Invalid paper configuration")
    cash, queue, locked, seq = capital, [], Counter(), 0
    trades, skip, equity = [], Counter(), []
    peak, drawdown = capital, 0.
    def mark(ts):
        nonlocal peak, drawdown
        value = cash+sum(locked.values())
        peak = max(peak, value)
        drawdown = max(drawdown, peak-value)
        equity.append({"ts":ts, "cash":cash, "cost_basis_equity":value})
    for r in sorted(rows, key=lambda r: (r["ts"], r["market_id"])):
        while queue and queue[0][0] <= r["ts"]:
            settled_ts, _, payout, event, cost = heapq.heappop(queue)
            cash += payout
            locked[event] -= cost
            mark(settled_ts)
        if r.get("yes_ask") is None or r.get("no_ask") is None:
            skip["missing_quotes"] += 1
            continue
        if not 0 <= r["ts"]-r.get("quote_ts", float("-inf")) <= max_quote_age:
            skip["stale_quote"] += 1
            continue
        rate = r.get("fee_rate")
        if rate is None or r.get("fee_available_ts", float("inf")) > r["ts"]:
            rate = assume_fee_rate
        if rate is None:
            skip["unknown_historical_fee"] += 1
            continue
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("Invalid fee rate")
        options = []
        for side, p in [("yes", r["forecast"]), ("no", 1-r["forecast"])]:
            price = r[side+"_ask"] + slippage
            if price >= 1:
                continue
            depth = r.get(side+"_depth")
            if (depth is None and not assume_depth) or (depth is not None and depth < contracts):
                continue
            f = fee(r["venue"], price, contracts, rate, r.get("fee_quantum"))
            cost = price*contracts+f
            options.append((p-cost/contracts, side, price, f, cost))
        if not options:
            skip["insufficient_or_unknown_depth"] += 1
            continue
        edge, side, price, f, cost = max(options)
        if edge < min_edge:
            skip["edge_below_threshold"] += 1
            continue
        if cost > cash or locked[r["event_id"]]+cost > max_event_cost:
            skip["cash_or_event_limit"] += 1
            continue
        cash -= cost
        locked[r["event_id"]] += cost
        payout = contracts*int(r["y"] == (side == "yes"))
        seq += 1
        heapq.heappush(queue, (r["label_available_ts"], seq, payout, r["event_id"], cost))
        trades.append({"market_id": r["market_id"], "event_id": r["event_id"], "ts": r["ts"],
                       "settlement_ts": r["label_available_ts"], "side": side, "contracts": contracts,
                       "fill_price": price, "fee": f, "cost": cost, "payout": payout,
                       "net_pnl": payout-cost, "estimated_edge": edge, "fee_rate": rate,
                       "depth_assumed": r.get(side+"_depth") is None,
                       "fee_assumed": r.get("fee_rate") is None or r.get("fee_available_ts", float("inf")) > r["ts"]})
        # Carry unsettled positions at cost. This is NOT a mark-to-market risk metric.
        mark(r["ts"])
    while queue:
        ts, _, payout, event, cost = heapq.heappop(queue)
        cash += payout
        locked[event] -= cost
        mark(ts)
    return {"hypothetical_only": True, "initial_capital": capital, "final_cash": cash,
            "net_pnl": cash-capital, "return_on_initial_capital": cash/capital-1,
            "total_fees": sum(t["fee"] for t in trades), "trade_count": len(trades),
            "max_cost_basis_drawdown": drawdown, "skipped": dict(skip),
            "assumptions": {"contracts": contracts, "min_edge": min_edge, "slippage_per_share": slippage,
                            "max_event_cost": max_event_cost, "assume_depth": assume_depth,
                            "assume_fee_rate": assume_fee_rate, "max_quote_age_seconds": max_quote_age},
            "trades": trades, "equity": equity}
