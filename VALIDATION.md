# Validation record

Release review: 17 October 2026.

**36 tests pass.** Three reproducible synthetic controls exercise planted signal, no added signal, and regime reversal. The retained controls show Brier scores of **0.201646 → 0.175714**, **0.181112 → 0.180819**, and **0.197706 → 0.218955**, respectively. A separate historical market study did not establish a reliable advantage; absent fee/depth evidence prevents strict simulated fills.

The listed fresh checks exercise the included public edition. Historical full experiments, representative replays, and synthetic fixtures are identified separately. Test success does not certify commercial deployment.

```sh
python -m unittest discover -s tests -q
```

A research and replay tool. Synthetic gains do not establish attainable financial returns. Historical model scores and assumed-fill results have different evidential scope. Fee schedules and API behavior require independent current verification before a new study.
