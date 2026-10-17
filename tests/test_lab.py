import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from prediction_lab.data import validate, weights
from prediction_lab.model import BetaCalibration, Forecaster, fit_logistic, sigmoid
from prediction_lab.evaluation import backtest, stages, metrics
from prediction_lab.paper import fee, simulate
from prediction_lab.synthetic import generate
from prediction_lab.apis import polymarket_assets, polymarket_history, polymarket_row, kalshi_row, PublicHTTP, GAMMA


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.rows = generate(4)

    def test_future_features_rejected(self):
        self.rows[0]["feature_available_ts"] += 2
        with self.assertRaises(ValueError): validate(self.rows)

    def test_future_evidence_rejected(self):
        self.rows[0]["evidence_available_ts"] = self.rows[0]["ts"]+1
        with self.assertRaises(ValueError): validate(self.rows)

    def test_already_known_outcome_rejected(self):
        self.rows[0]["label_available_ts"] = self.rows[0]["ts"]
        with self.assertRaises(ValueError): validate(self.rows)

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError): validate(self.rows + self.rows[:1])

    def test_nan_rejected(self):
        self.rows[0]["market_p"] = float("nan")
        with self.assertRaises(ValueError): validate(self.rows)

    def test_crossed_quotes_rejected(self):
        self.rows[0].update(yes_ask=.3, no_ask=.4)
        with self.assertRaises(ValueError): validate(self.rows)

    def test_unlabeled_inference(self):
        for r in self.rows:
            r.pop("y"); r.pop("label_available_ts")
        self.assertEqual(len(validate(self.rows, labeled=False)), 4)

    def test_event_weights(self):
        self.rows[1]["event_id"] = self.rows[0]["event_id"]
        np.testing.assert_allclose(weights(self.rows), [.5, .5, 1, 1])


class ModelTests(unittest.TestCase):
    def test_identity_beta_map(self):
        c = BetaCalibration(); c.beta = np.zeros(3)
        p = np.array([.001, .1, .5, .9, .999])
        np.testing.assert_allclose(c.predict(p), p)

    def test_optimizer_recovers_intercept(self):
        y = np.array([1.]*70+[0.]*30)
        b = fit_logistic(np.ones((100,1)), y, np.zeros(100), np.ones(100), penalty=1e-9)
        self.assertAlmostEqual(float(sigmoid(b)[0]), .7, places=6)

    def test_probabilities_and_serialization(self):
        rows = generate(500)
        split = stages(rows, rows[-1]["label_available_ts"]+86400)
        m = Forecaster().fit(*split)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"model.json"; m.save(p)
            z = Forecaster.load(p)
            for k, v in m.predict(rows[-20:]).items():
                self.assertTrue(((v>0)&(v<1)).all())
                np.testing.assert_allclose(v, z.predict(rows[-20:])[k])

    def test_labels_not_used_in_prediction(self):
        rows = generate(500)
        m = Forecaster().fit(*stages(rows, rows[-1]["label_available_ts"]+86400))
        changed = copy.deepcopy(rows[-10:])
        for r in changed: r["y"] = 1-r["y"]
        np.testing.assert_equal(m.predict(changed)["forecast"], m.predict(rows[-10:])["forecast"])


class TimeSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate(600)

    def test_stage_order_and_disjoint_events(self):
        origin = self.rows[-1]["label_available_ts"]+86400
        split = stages(self.rows, origin)
        for previous, following in zip(split, split[1:]):
            self.assertLess(max(r["label_available_ts"] for r in previous), min(r["ts"] for r in following)-3600)
            self.assertFalse({r["event_id"] for r in previous}&{r["event_id"] for r in following})
        self.assertLess(max(r["label_available_ts"] for r in split[-1]), origin-3600)

    def test_unresolved_labels_purged(self):
        origin = self.rows[-1]["label_available_ts"]+86400
        rows = copy.deepcopy(self.rows)
        for r in rows[:20]: r["label_available_ts"] = origin+1
        split = stages(rows, origin)
        self.assertTrue(all(r["label_available_ts"] < origin for rs in split for r in rs))

    def test_future_outcomes_cannot_change_earlier_forecasts(self):
        a, report, _ = backtest(self.rows, folds=2)
        origin = report["folds"][0]["origin"]
        changed = copy.deepcopy(self.rows)
        for r in changed:
            if r["label_available_ts"] >= origin: r["y"] = 1-r["y"]
        b, _, _ = backtest(changed, folds=2)
        np.testing.assert_allclose([r["forecast"] for r in a if r["fold"]==0],
                                   [r["forecast"] for r in b if r["fold"]==0])

    def test_sibling_event_cannot_leak_from_train_to_test(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["event_id"] = rows[300]["event_id"]
        # Label at event settlement, not the early sibling's old availability.
        rows[0]["label_available_ts"] = rows[300]["label_available_ts"]
        ps, _, _ = backtest(rows, folds=2)
        # Changing the early sibling outcome must not affect that event's first test fold.
        rows[0]["y"] = 1-rows[0]["y"]
        qs, _, _ = backtest(rows, folds=2)
        np.testing.assert_allclose([r["forecast"] for r in ps if r["fold"]==0],
                                   [r["forecast"] for r in qs if r["fold"]==0])

    def test_planted_signal_detected(self):
        _, report, _ = backtest(generate(1200), folds=4)
        self.assertGreater(report["brier_improvement"], .005)

    def test_metric_reference(self):
        rows = [dict(r, forecast=.5) for r in generate(10)]
        m = metrics(rows, "forecast")
        self.assertAlmostEqual(m["brier"], .25)
        self.assertAlmostEqual(m["log_loss"], np.log(2))


class PaperTests(unittest.TestCase):
    def row(self, **kw):
        r = generate(1)[0]
        r.update(market_p=.4, yes_ask=.41, no_ask=.61, forecast=.7, y=1)
        return dict(r, **kw)

    def test_polymarket_current_fee(self):
        self.assertEqual(fee("polymarket", .5, 100, .07), 1.75)

    def test_kalshi_conservative_rounding(self):
        self.assertEqual(fee("kalshi", .5, 1, .07), .02)
        self.assertAlmostEqual(fee("kalshi", .5, 1, .07, "0.0001"), .0175)

    def test_no_double_count_spread_and_settlement_pnl(self):
        p = simulate([self.row(venue="polymarket")], slippage=0)
        self.assertEqual(p["trade_count"], 1)
        self.assertAlmostEqual(p["net_pnl"], 10-4.1-fee("polymarket",.41,10,.07))

    def test_no_synthetic_midpoint_fills(self):
        r = self.row(); r.pop("yes_ask"); r.pop("no_ask")
        self.assertEqual(simulate([r])["trade_count"],0)

    def test_missing_fee_and_depth_default_skip(self):
        r = self.row(); r.pop("fee_rate")
        self.assertEqual(simulate([r])["skipped"]["unknown_historical_fee"],1)
        r = self.row(); r.pop("yes_depth"); r.pop("no_depth")
        self.assertEqual(simulate([r])["trade_count"],0)

    def test_future_fees_do_not_leak(self):
        r = self.row(fee_available_ts=1e20)
        self.assertEqual(simulate([r])["trade_count"],0)

    def test_stale_quotes_skip(self):
        r = self.row(); r["quote_ts"] -= 10000
        self.assertEqual(simulate([r])["trade_count"],0)

    def test_limited_depth_skip(self):
        self.assertEqual(simulate([self.row(yes_depth=0,no_depth=0)])["trade_count"],0)

    def test_cash_is_locked(self):
        a = self.row(market_id="a"); b = self.row(market_id="b",event_id="other")
        p = simulate([a,b], capital=5)
        self.assertEqual(p["trade_count"],1)
        self.assertGreaterEqual(p["final_cash"],0)

    def test_event_exposure_cap(self):
        rs = [self.row(market_id=str(i)) for i in range(3)]
        self.assertEqual(simulate(rs,max_event_cost=5)["trade_count"],1)

    def test_cost_stress_monotonic_for_fixed_trade(self):
        a = simulate([self.row()],slippage=0)["net_pnl"]
        b = simulate([self.row()],slippage=.02)["net_pnl"]
        self.assertLess(b,a)


class APITests(unittest.TestCase):
    def test_polymarket_labels_must_be_independently_audited(self):
        with self.assertRaises(ValueError): polymarket_row({}, [])

    def test_polymarket_history_never_invents_executable_quotes(self):
        audit={"market_id":"m", "event_id":"e", "scheduled_close_ts":200000,
               "y":1,"label_available_ts":210000,"schedule_available_ts":0,
               "schedule_evidence":"rule vintage", "resolution_evidence":"final payout source"}
        row=polymarket_row(audit,[{"ts":110000,"price":.4,"resolution_seconds":300}])
        self.assertEqual(row["y"],1)
        self.assertNotIn("yes_ask",row)
        audit["schedule_available_ts"]=199999
        with self.assertRaises(ValueError): polymarket_row(audit,[])

    def test_outcome_label_mapping_and_protocol_version(self):
        m = {"outcomes":'["No","Yes"]',"clobTokenIds":'["n","y"]'}
        self.assertEqual(polymarket_assets(m)["yes"],"y")
        m.update(version="v2",positionIds=["vn","vy"])
        self.assertEqual(polymarket_assets(m)["yes"],"vy")

    def test_nonbinary_rejected(self):
        with self.assertRaises(ValueError): polymarket_assets({"outcomes":'["A","B","C"]'})

    def test_v2_pagination_and_range_chunking(self):
        class Fake:
            calls=[]
            def get(self,base,path,params):
                self.calls.append(params.copy())
                more = len(self.calls)==1
                return {"data":[{"timestamp":len(self.calls),"price":.4,"resolution_seconds":60}],
                        "pagination":{"has_more":more,"next_cursor":"next" if more else None}}
        f=Fake(); ps=polymarket_history(f,"asset",0,16*86400)
        self.assertEqual(len(ps),3)
        self.assertEqual(f.calls[1]["cursor"],"next")
        self.assertEqual(f.calls[2]["start"],15*86400)

    def test_public_adapter_rejects_account_endpoints(self):
        with tempfile.TemporaryDirectory() as d:
            h=PublicHTTP(d)
            with self.assertRaises(ValueError): h.get("https://attacker.example","/markets")
            with self.assertRaises(ValueError): h.get(GAMMA,"/orders")

    def test_candle_close_time_not_open_time(self):
        m={"ticker":"t","event_ticker":"e","close_time":"2026-01-04T00:00:00Z",
           "settlement_ts":"2026-01-04T01:00:00Z","result":"yes"}
        from prediction_lab.apis import epoch
        ts=epoch(m["close_time"])-86400
        candles=[{"end_period_ts":ts,"yes_bid":{"close_dollars":"0.4"},"yes_ask":{"close_dollars":"0.42"}}]
        with self.assertRaises(ValueError): kalshi_row(m,candles)
        candles[0]["end_period_ts"]-=3600
        r=kalshi_row(m,candles)
        self.assertAlmostEqual(r["market_p"],.41)
        self.assertAlmostEqual(r["no_ask"],.6)


if __name__ == "__main__":
    unittest.main()
