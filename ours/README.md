# Our submission — ball possession from automatic tracking

Everything we added to the challenge repository lives in this folder. The method, training data,
possession rules, results and limitations are in **[Ousyas.md](Ousyas.md)**.

## Regenerate the submission

```bash
py -3.12 -m pip install -r ours/requirements.txt
py ours/reproduce.py --skip-train --name Ousyas   # committed models -> the submitted file, any OS
py ours/reproduce.py --name Ousyas                # retrain on IDSSE first (~40 min)
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
| `events.py`, `v11.py`, `v11_train.py` | event-timing features on top of v8 (first submission); `--version` selects v11 / v13r / v14 |
| `idsse_bench.py` | IDSSE loading, label extraction, ATD-like degradation, dead-ball simulation |
| `validate_*.py`, `bench_results*.csv` | leave-one-match-out experiments and their logs |
| `mobility.py`, `tf_model.py` | rejected experiments (v9 mobility features, v10 Transformer) |
| `analyze.py` | diagnostics, including the public 5-minute window (diagnostic only) |
| `gap_evidence.py` | post-submission: label-free ball-gap evidence on D (`v11.py --gap-evidence`), see [GAP_EVIDENCE.md](GAP_EVIDENCE.md) |
| `validate_gap.py`, `bench_results_gap.csv` | its leave-one-match-out test in several "ball-gap worlds" (`idsse_bench.set_world`) |
| `atd_gap_stats.py` | label-free statistics of how the ATD loses the ball, to set the bench's gap world |
| `seq_model.py`, `validate_seq.py` | temporal GRU head over v11's features, 50/50 with the classifier (`v13r`) |
| `validate_world_train.py`, `bench_results_world.csv` | training on BAR-ATH-like ball gaps as well — **the submission `v14`** |
| `validate_relabel.py` | v12 experiment: training on labels aligned to the challenge's control definition (`idsse_bench.align_to_control`) |
