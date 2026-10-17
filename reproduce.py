"""Offline synthetic-control reproduction. Writes to a separate output folder."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args, log=None):
    p = subprocess.run([sys.executable, *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    if log:
        (ROOT/log).write_text(p.stdout)
    print(p.stdout)
    p.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced')
    args = parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for scenario in ["edge", "null", "drift"]:
        run("-m", "prediction_lab.cli", "demo", "--scenario", scenario,
            "--out", str(out/f"synthetic_{scenario}.jsonl"))
        run("-m", "prediction_lab.cli", "backtest", "--input", str(out/f"synthetic_{scenario}.jsonl"),
            "--out", str(out/f"synthetic_{scenario}"))
