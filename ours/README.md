# Our submission — ball possession from automatic tracking

Everything we added to the challenge repository lives in this folder. The method, training data,
possession rules, results and limitations are in **[Ousyas.md](Ousyas.md)**.

## Regenerate the submission

```bash
py -3.12 -m pip install -r ours/requirements.txt
py ours/reproduce.py --name Ousyas          # -> evaluation/results/Ousyas.csv + validation + SHA-256
```

Needs the challenge data bundle in `data/` (see `data/README.md`). The 7 IDSSE training matches
are downloaded by `kloppy` on first run into `data/ext/` (git-ignored; or place the figshare files
in `data/ext/raw/`). `--skip-train` reuses the committed models in `ours/models/`.

## Map of the folder

| file | role |
|---|---|
| `reproduce.py` | one command: train → predict → validate |
| `features.py`, `v1.py` … `v4.py`, `v42.py` | ATD parsing and rule-based evidence (touches, control, stoppages, restarts) |
| `v5.py`, `v5_train.py` | frame classifier P(H/A/D): features, training on IDSSE |
| `v5_em.py` | EM estimate of the alive-play rate on the test game |
| `shape.py`, `v8.py`, `v8_train.py` | team-shape features, turnover detector, time-varying Viterbi (v8) |
| `events.py`, `v11.py`, `v11_train.py` | event-timing features on top of v8 — **the submission** |
| `idsse_bench.py` | IDSSE loading, label extraction, ATD-like degradation, dead-ball simulation |
| `validate_*.py`, `bench_results*.csv` | leave-one-match-out experiments and their logs |
| `mobility.py`, `tf_model.py` | rejected experiments (v9 mobility features, v10 Transformer) |
| `analyze.py` | diagnostics, including the public 5-minute window (diagnostic only) |
