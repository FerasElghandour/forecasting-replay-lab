"""Market-anchored logistic correction with identity-shrunk beta calibration.

Optimizer is damped Newton with line search on a convex penalized log loss.
No preprocessing or model selection is fitted on the outer test fold.
"""
import numpy as np
from .data import weights

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -np.asarray(z)))


def fit_logistic(x, y, offset, w, penalty=0.03, prior=None):
    prior = np.zeros(x.shape[1]) if prior is None else np.asarray(prior)
    b = prior.copy()
    w = w / w.sum()

    def objective(v):
        z = offset + x @ v
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)) + penalty / 2 * np.sum((v-prior)**2))

    for _ in range(100):
        p = sigmoid(offset + x @ b)
        g = x.T @ (w * (p - y)) + penalty * (b - prior)
        h = x.T @ ((w * p * (1-p))[:, None] * x) + penalty * np.eye(x.shape[1])
        step = np.linalg.solve(h, g)
        if np.max(np.abs(g)) < 1e-9:
            break
        rate, old = 1., objective(b)
        while rate > 1e-10 and objective(b - rate * step) > old - 1e-4 * rate * float(g @ step):
            rate *= .5
        b -= rate * step
    if not np.isfinite(b).all():
        raise ValueError("Optimizer failed")
    return b


def features(rows):
    p = np.array([r["market_p"] for r in rows])
    z = logit(p)
    evidence = np.array([logit(r["evidence_p"]) - a if r.get("evidence_p") is not None else 0.
                         for r, a in zip(rows, z)])
    horizon = np.array([np.log1p((r["scheduled_close_ts"]-r["ts"])/86400) for r in rows])
    return np.column_stack([np.ones(len(rows)), z, z * z / 4,
                            horizon, [r.get("momentum", 0.) for r in rows],
                            [r.get("spread", 0.) for r in rows], evidence,
                            [float(r.get("evidence_p") is not None) for r in rows]])


class Correction:
    def fit(self, rows):
        x = features(rows)
        self.scale = np.maximum(np.sqrt(np.average(x*x, axis=0, weights=weights(rows))), .1)
        self.beta = fit_logistic(x/self.scale, np.array([r["y"] for r in rows]),
                                 logit([r["market_p"] for r in rows]), weights(rows))
        return self

    def predict(self, rows):
        return sigmoid(logit([r["market_p"] for r in rows]) + features(rows)/self.scale @ self.beta)


class BetaCalibration:
    """Identity-centered beta map: sigmoid(logit(p) + b0 + b1 log(p) - b2 log(1-p)).

    Monotonicity requires b1,b2 >= -1. Reject a nonmonotone fit to identity.
    This conservative fallback avoids reversing the order of probabilities.
    """
    def fit(self, p, rows):
        p = np.clip(p, EPS, 1-EPS)
        self.beta = fit_logistic(self.x(p), np.array([r["y"] for r in rows]), logit(p), weights(rows), .05)
        if min(self.beta[1:]) < -1:
            self.beta[:] = 0
        return self

    @staticmethod
    def x(p):
        p = np.clip(p, EPS, 1-EPS)
        return np.column_stack([np.ones(len(p)), np.log(p), -np.log1p(-p)])

    def predict(self, p):
        return sigmoid(logit(p) + self.x(p) @ self.beta)


def loss(p, rows):
    y = np.array([r["y"] for r in rows])
    p = np.clip(p, EPS, 1-EPS)
    return float(np.average(-y*np.log(p)-(1-y)*np.log1p(-p), weights=weights(rows)))


class Forecaster:
    def fit(self, train, calibration, selection):
        self.model = Correction().fit(train)
        self.calibrator = BetaCalibration().fit(self.model.predict(calibration), calibration)
        self.market_calibrator = BetaCalibration().fit(np.array([r["market_p"] for r in calibration]), calibration)
        raw = self.calibrator.predict(self.model.predict(selection))
        market = np.array([r["market_p"] for r in selection])
        # Tie-break toward the market. Fixed grid; no outer-fold tuning.
        self.alpha = min([0., .25, .5, .75, 1.], key=lambda a: loss((1-a)*market+a*raw, selection))
        return self

    def predict(self, rows):
        market = np.array([r["market_p"] for r in rows])
        raw = self.model.predict(rows)
        calibrated = self.calibrator.predict(raw)
        return {"market": market, "calibrated_market": self.market_calibrator.predict(market),
                "raw_model": raw, "calibrated_model": calibrated,
                "forecast": (1-self.alpha)*market+self.alpha*calibrated}

    def save(self, path):
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps({"schema_version": 1, "alpha": self.alpha,
            "scale": self.model.scale.tolist(), "model_beta": self.model.beta.tolist(),
            "calibration_beta": self.calibrator.beta.tolist(),
            "market_calibration_beta": self.market_calibrator.beta.tolist()}, indent=2))

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path
        d = json.loads(Path(path).read_text())
        if d["schema_version"] != 1:
            raise ValueError("Unsupported model version")
        obj = cls()
        obj.alpha = d["alpha"]
        obj.model = Correction()
        obj.model.scale, obj.model.beta = np.array(d["scale"]), np.array(d["model_beta"])
        obj.calibrator, obj.market_calibrator = BetaCalibration(), BetaCalibration()
        obj.calibrator.beta = np.array(d["calibration_beta"])
        obj.market_calibrator.beta = np.array(d["market_calibration_beta"])
        return obj
