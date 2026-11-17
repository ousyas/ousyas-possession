# Ball possession from automatic tracking — technical document

Team **Ousyas** — More than a Hack 2027, Phase 1 (BAR–ATH). Submitted file SHA-256 `af916ebaf5389ebf0557f321a08a82f22566e66e9312038e4e4331813fe462e0`. Code: `ours/`. Reproduction: §7.

## 1. Approach in one paragraph

Per-frame evidence → a learned frame classifier P(H, A, D) → a learned **turnover detector** →
an HMM/Viterbi decoder whose transition costs vary in time → minimum-duration clean-up. The
classifier and the detector are gradient-boosted trees (scikit-learn
`HistGradientBoostingClassifier`) trained **only on 7 public Bundesliga matches (IDSSE, CC-BY)**
whose tracking we degrade to look like the challenge's automatic tracking data (ATD). Every
design decision was taken on that external benchmark with leave-one-match-out validation; the
public 5-minute window of the challenge game was used only to check that the pipeline runs end to
end and that the output is sane.

Why this rather than something else: spellF1 rewards getting *whole possessions* right, so the
problem is a segmentation problem, not a per-frame one. Per-frame models flicker; pure rules
(our v1–v4) are brittle on noisy ATD where the ball is missing 41 % of the time. A
learned frame model + a structured decoder keeps the best of both: the model weighs many weak
cues, the decoder enforces that possessions are contiguous and that stoppages start and end at
plausible events.

## 2. Pipeline

| step | file | what it does |
|---|---|---|
| tracking | `features.py` | parse the ATD (46 anonymous track slots + ball), assign tracks to teams (`starter/baseline.py` helper), cache |
| rule evidence | `v1.py`–`v4.py`, `v42.py` | ball interpolation (gaps ≤ 0.5 s), touch detection, team-in-control state, stoppage score, restart signatures |
| frame features | `v5.py` | 43 features: rule state (v4.2), stoppage logit (v3), distance of nearest H/A player to ball, players within 5/10 m, team spread, ball speed/visibility/gap length/distance to touchline/out of pitch, collective speed, plus 0.5 s / 2 s rolling means and ±1 s trends |
| frame classifier | `v5_train.py` | HGB, 300 iterations, lr 0.08, 31 leaves, L2 1.0, trained on 1 frame out of 3, **H↔A swap augmentation** (every frame also seen with teams mirrored) |
| alive rate | `v5_em.py`, `v8.py` | EM re-estimation of the class prior on the test game (Saerens et al. 2002), clipped to [50 %, 65 %] → target fraction of alive play for the decoder |
| turnover detector | `v8.py`, `v8_train.py`, `shape.py` | binary HGB: "a direct H↔A change happens within ±0.5 s"; features = frame features + team shape (width, length, players beyond the last defender, in attack direction) + future-minus-past trends over 2 s / 4 s. AUC 0.89–0.92 on held-out IDSSE matches |
| decoding | `v8.py` (`viterbi_switch`) | 3-state Viterbi (H, A, D). Entering D is cheap only near a stoppage-start event, leaving it only near a restart event (v3). The H↔A switch cost goes from 30 down to 8 at local maxima of the turnover probability. The D bias is bisected so that the decoded alive fraction matches the EM estimate |
| event timing | `events.py` | 18 features, all team-symmetric: seconds since / until the ball was last / next out of the pitch, still (< 0.6 m/s), controlled (player < 1.5 m, ball < 12 m/s), visible, and since / until a collective burst (≥ 3 players > 4 m/s); ball speed 1 s / 2 s before and after; mean player speed over 6 s and its ±2 s change (capped at 30 s). Added to **both** the frame classifier and the turnover detector |
| clean-up | `v3.py` | minimum durations; outside the two graded halves → `X` |

Submission = `v11`: classifier `models/v11.joblib` + detector `models/v11_turnover.joblib` (`v11.py`, `v11_train.py`).

## 3. Assets used and dropped

| asset | used? | how |
|---|---|---|
| tactical ATD | **yes** | everything above |
| Smart Tagging (automatic) | **yes, weakly** | team phase tags give a weak prior on the team (`v1.team_from_tags`), set-piece tags feed the stoppage score. It is automatic, not human data |
| homography | no | ATD is already in pitch coordinates |
| broadcast / tactical video | no (as input) | watched only to understand failure types during development |
| external data | **IDSSE** (7 Bundesliga 2022/23 matches, Bassek et al., figshare 10.6084/m9.figshare.28196177, CC-BY 4.0), loaded with `kloppy` | training and validation |

## 4. Training data

* **IDSSE, 7 matches**: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 (≈ 7 × 135 k
  frames at 25 fps). Labels come from the provider's own per-frame flags — `ball_state` dead →
  `D`, otherwise `ball_owning_team` home → `H` / away → `A`. **Nobody on our team labelled
  anything**, and no label of any kind from the challenge game was used.
* **Degradation to look like the ATD** (`idsse_bench.py`, fixed seeds): smooth position noise,
  players dropping out in Markov gaps, ball missing ≈ 41 % of the time (more often in crowded
  zones), identities removed.
* **Dead-ball simulation** ("bench 2"). In IDSSE the ball position is frozen during 100 % of
  stoppages, whereas in the challenge ATD the ball moves inside the pitch during ~42 % of dead
  frames (it is carried, rolled, placed). A model trained on frozen balls learned "ball moves ⇒
  alive". We replace each dead segment by a roll → carry to the restart point → placed sequence
  (15 % left frozen). This is a property of the *data source*, measured on the ATD as a whole, not
  a rule fitted to frames.

## 5. Possession rules implemented

* **Team change needs control** (CHALLENGE.md): two consecutive touches of the new team, or the
  ball staying near one of its players ≥ 0.8 s; the change is dated at the first touch.
* **Touches**: local maximum of smoothed ball acceleration (≥ ~4 m/s²) or a sharp change of
  direction, attributed to the nearest player if within 2 m (ATD noise).
* **Ball in flight** belongs to the team of the last touch.
* **Challenges / loose balls**: a touch with an opponent within 1.5 m is *disputed* and keeps the
  previous team — except "duel won then played": a disputed touch by team T followed within 2 s
  by a clean touch by T moves possession to T at the duel.
* **Stoppages**: start events (ball out of the pitch, missed shot — fast ball towards goal then
  lost/out —, sharp collective deceleration); restart events (collective acceleration + kicked
  ball) and Law-of-the-game signatures (goal kick, corner, throw-in, kick-off, free kick with a
  9.15 m wall).
* **Spells**: minimum durations; abstain (`X`) only outside graded halves.

## 6. Results (IDSSE, leave-one-match-out, official scorer)

Each row was accepted or rejected **before** looking at the challenge game, with a criterion
fixed in advance: beat the current version on mean spellF1 **and** on at least 5 of 7 matches.

| version | change | spellF1 | verdict |
|---|---|---|---|
| baseline smooth50 | | 0.207 | |
| v4 | touches + control rules | 0.409 | |
| v4.1 | stricter duel rules | 0.386 | rejected |
| v4.2 | "duel won then played" | 0.415 | kept |
| v5 | learned frame classifier (fixed 58 % alive) | 0.456 | kept |
| v6 | + EM alive rate | 0.479 (7/7) | kept (oracle alive rate: 0.481) |
| v7 | + team-shape features in the classifier | +0.007, 4/7 | rejected |
| **v8** | + turnover detector / time-varying switch cost | **0.495** (+0.017, 7/7) | **kept** |
| on bench 2 (moving dead ball): | | | |
| v8 trained on bench 1 | | 0.406 (D recall 75 %) | |
| **v8 trained on bench 2** | | **0.477** (D recall 96 %), 7/7 | kept |
| v9 | + player-mobility features | −0.002, 3/7 | rejected |
| v10 | spatio-temporal Transformer (players → time, ~95 k params) | −0.037 on 4/4 folds | rejected (stopped once 5/7 was impossible) |
| v8 + v10 average | | −0.013 on 4/4 | rejected |
| v11 (classifier only) | + event-timing features in the frame classifier | +0.009, 4/7 | rejected |
| **v11** | + event-timing features in classifier **and** turnover detector | **0.486** (+0.010, 5/7; macroF1 +0.009, 7/7) | **kept → submission** |

The event-timing idea (time since / until stoppage-bounding events) was suggested by reading
another team's public repository (github.com/hadisotudeh/more-than-a-hack-2027); our features are
our own implementation and were accepted only on the IDSSE benchmark above.

Pipeline check on the public 5 minutes of BAR–ATH (diagnostic only, never used for decisions):
v8 scored spellF1 0.555, macroF1 0.793, recall of dead frames 52 %; v11 (submitted) scored spellF1 0.560, macroF1 0.807, recall of dead frames 59 %. Note the old bench-1 v8 scored higher
on spellF1 there (0.593) and we still submit the bench-2 model, because the decision belongs to
the external benchmark.

Experiment logs: `ours/bench_results*.csv`.

## 7. Reproduction

```bash
git clone https://github.com/ousyas/ousyas-possession && cd ousyas-possession
py -3.12 -m pip install -r ours/requirements.txt
# challenge data bundle unpacked in data/ (see data/README.md)
# IDSSE: downloaded automatically by kloppy on first run into data/ext/ (or put the figshare
# files in data/ext/raw/)
py ours/reproduce.py --name Ousyas
```

On macOS / Linux (no `py` launcher), with the data bundle anywhere:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r ours/requirements.txt
MTAH_DATA_DIR=/path/to/data .venv/bin/python ours/reproduce.py --name Ousyas
```

`reproduce.py` trains the classifier and the detector (`v11_train.py`) on the 7 IDSSE matches, writes
`evaluation/results/Ousyas.csv`, runs `evaluation/validate_submission.py` and prints the SHA-256.
All randomness is seeded; on Python 3.12 / scikit-learn 1.7.2 the file is byte-identical to the
submitted one. Runtime ≈ 5 min on a 12-core CPU, no GPU. The validation experiments of §6 are
the `ours/validate_*.py` scripts. `--skip-train` reuses the committed models in `ours/models/` and
rebuilds the file in under a minute. The CSV is always written with CRLF line endings, so its
SHA-256 does not depend on the operating system: with `--skip-train`, Windows 11 and macOS both
give the submitted file.

## 8. Limitations and what we would do next

* **Stoppages are the weak point.** The bench is still easier than the real ATD (dead-frame recall
  97 % on IDSSE vs ≈ 59 % on the public window): free kicks where the ball is visible and moving
  slowly are read as alive play. Next: richer dead-ball simulation (players walking to positions,
  wall formation), and restart detection from player configuration.
* **Possession definitions differ.** IDSSE possessions are short (≈ 7 s mean) compared with the
  challenge ground truth (≈ 14 s on the public window), so the decoder may switch too often.
* **Smart Tagging** is absent from IDSSE, so the model never saw it during training; its effect
  at test time is untested.
* **Ball track loss in crowded play** makes team changes in duels late by 1–2 s.
* **Deep sequence models** (our Transformer) underperformed with only 6 training matches;
  they would need more external matches or self-supervised pre-training on unlabelled tracking.
