"""Offline example using the public implementation; see README for evidence scope."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from prediction_lab.synthetic import generate
from prediction_lab.evaluation import backtest
rows=generate(500,21,'edge')
predictions,report,model=backtest(rows)
print({'rows':report['rows'],'events':report['events'],'fitted_rows':report['fitted_rows'],'market_brier':report['scores']['market']['brier'],'forecast_brier':report['scores']['forecast']['brier']})
