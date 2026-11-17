# Ball possession from automatic tracking — technical document

Team **Ousyas** — More than a Hack 2027, Phase 1 (BAR–ATH). Submitted file (`v13r`) SHA-256 `da9ff3292002ec089d80378e4f3eff7303ccd52a2d7d79448826b2a31fc46165`. Code: `ours/`. Reproduction: §7.

## 1. Approach in one paragraph

Per-frame evidence → a learned frame classifier P(H, A, D) → a learned **turnover detector** →
an HMM/Viterbi decoder whose transition costs vary in time → minimum-duration clean-up. The
classifier and the detector are gradient-boosted trees (scikit-learn
`HistGradientBoostingClassifier`); the classifier's probabilities are averaged 50/50 with those of
a small **bidirectional GRU** that reads the same per-frame features over ~100 s of context
(submission `v13r`). Everything is trained **only on 7 public Bundesliga matches (IDSSE, CC-BY)**
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
| temporal head | `seq_model.py` | 2-layer bidirectional GRU (hidden 64, ~110 k parameters) over the same 61 per-frame features (43 frame + 18 event timing), robust-scaled, at 5 fps; trained on 512-step chunks (~100 s), one random 5 fps phase per half and epoch, H↔A swap augmentation, 10 epochs, Adam 2e-3, seed 0, single-threaded deterministic torch. At prediction, the five 5 fps phases are run over each half and interleaved. Its P(H, A, D) is averaged 50/50 with the classifier's (weight fixed in advance) before the alive-rate EM and the decoder |
| clean-up | `v3.py` | minimum durations; outside the two graded halves → `X` |

Submission = `v13r`: classifier `models/v13r.joblib` + detector `models/v13r_turnover.joblib` + GRU `models/v13r_gru.joblib` (`v11.py --version v13r`, `v11_train.py --version v13r`). The classifier and detector are the v11 ones (same labels, same training); `v13r` adds only the GRU. Our first submission was `v11` (`models/v11*.joblib`, `--version v11`, SHA-256 `af916eba…`).

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

### 6.1 Second submission (17 November): `v13r`

**Scoring under the challenge's definition.** DFL's `ball_owning_team` flips on every touch
(8.4 possession changes per live minute, 7.3 s mean possession), while the challenge only counts a
change when the new team gains *control* ("a simple touch of the ball without control" and "a
deflection" do not count, CHALLENGE.md; 4.2 per live minute, 14.2 s on the public window). For the
experiments below, each held-out IDSSE match is therefore also scored against its labels aligned
to control (`idsse_bench.align_to_control`): a team spell shorter than 2 s ("keeps the ball for a
few seconds") that ends with the previous team on the ball again, or with a stoppage, goes back to
the previous team. This brings IDSSE to 5.5 changes per live minute and 11.0 s possessions. The
2 s threshold comes from the rule text, not from the public window. Same protocol and criterion as
above (leave-one-match-out, mean spellF1 **and** at least 5 of 7 matches, fixed before running):

| version | change | spellF1 (control truth) | verdict |
|---|---|---|---|
| v11 | first submission | 0.631 (raw DFL truth: 0.486) | |
| v11 + gap evidence | label-free EM of the dead-ball rate per ball-gap length on the test game, added to the D emission (`gap_evidence.py`, `validate_gap.py`) | −0.0005 (1/7); −0.0006 to −0.0022 in test worlds whose ball gaps copy BAR-ATH's (`atd_gap_stats.py`) | rejected: the EM estimates are accurate, but the classifier already uses the information through the event-timing features |
| v12 | training labels aligned to control as well (`validate_relabel.py`) | 0.634 (+0.004, 5/7; raw truth −0.012) | passes narrowly; superseded by v13r |
| **v13r** | + temporal GRU head over the same features, 50/50 with the classifier (`seq_model.py`, `validate_seq.py`) | **0.650** (+0.020, **7/7**; macroF1 +0.007; raw DFL truth 0.496, +0.010, 6/7) | **kept → submission** |

The GRU alone scored +0.022 (7/7); the 50/50 average was the variant fixed before running, so it is
the one submitted. v13r passes under both definitions of possession.

On BAR-ATH, v13r changes 5.6 % of graded frames, predicts 236 possessions (3.8 per live minute,
v11: 283, 4.65; public truth 4.2) and 41.5 % dead ball. Public 5-minute window (diagnostic only,
never used for any decision): spellF1 0.560 (unchanged; 17 possessions), macroF1 0.807 → 0.840. Unlike our v10 Transformer (raw players, ~95 k parameters to learn the geometry
itself), this head reads features the classifier already uses and only adds temporal context.

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

`reproduce.py` (default `--version v13r`) trains the classifier, the detector and the GRU
(`v11_train.py`) on the 7 IDSSE matches, writes `evaluation/results/Ousyas.csv`, runs
`evaluation/validate_submission.py` and prints the SHA-256. Runtime ≈ 15 min on an 8-core CPU, no
GPU. The validation experiments of §6 are the `ours/validate_*.py` scripts.

**Exact reproduction: `--skip-train`** reuses the committed models in `ours/models/` and rebuilds
the submitted file in about a minute, byte for byte, on any OS: the CSV is always written with CRLF
line endings, and the decoder is insensitive to last-bit floating-point differences (we checked
that perturbing the model outputs by 1e-6 changes no label). The boosted trees retrain
identically across machines (our leave-one-match-out scores of §6 reproduce to six decimals on
Windows and macOS); the GRU is trained seeded and single-threaded, but another CPU may differ in the last
bits of its weights, which is why the weights are committed. `--version v11` rebuilds our first
submission (`af916eba…`).

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
* **Sequence models.** A Transformer over raw players (v10) underperformed with only 6 training
  matches; a GRU over our own features (v13r) helped. Both would gain from more external matches
  (Metrica sample games, PFF World Cup 2022, SkillCorner open data).
* **Ball gaps.** On BAR-ATH the ball is lost in long blocks (85 % of missing frames in gaps > 10 s)
  that go with stoppages; our bench now has gap worlds that copy this (`idsse_bench.set_world`),
  but the models are still trained in the legacy world.
