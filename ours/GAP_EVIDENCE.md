# Ball-gap evidence for stoppages (post-submission experiment)

Not part of the submitted file. `v11.py` without `--gap-evidence` still writes the submitted CSV.

## The problem

Stoppages are our weak point: dead-frame recall is 97 % on IDSSE but about 59 % on the public
window. One reason is that our bench loses the ball the wrong way:

| | our bench (`legacy`) | challenge ATD (reported by other teams) |
|---|---|---|
| ball missing | 41 % of frames | 41 % of graded frames |
| gap shape | short gaps, mean 1.2 s | long blocks (median about 10 s), some over 30 s |
| where | more often in crowded *live* play | more often in *stoppages* (ball out of shot, held, set-piece preparation) |

So the model learns that a missing ball says nothing about a stoppage, and the decoder throws away
one of the strongest signals in the real tracking.

## The idea (label-free, rules-compliant)

How strongly gaps go with stoppages is a property of the tracker, so we do **not** hard-code it
in training. On the test game, `gap_evidence.py`:

1. puts every graded frame in a gap class from the raw ball track: visible, gap of ≤1 s, 1–3 s,
   3–10 s, 10–30 s, >30 s;
2. re-estimates the dead-ball rate inside each class with the same EM (Saerens et al. 2002) that
   v6 already uses for the global alive rate, from the classifier's own posteriors. No label of
   the test game is read;
3. adds the log-odds "class rate vs global rate" to the D emission at **half weight** (fixed in
   advance; we don't want to over-trust an estimate);
4. keeps the decoder's alive-rate calibration unchanged, so this only **moves** dead frames
   towards long gaps and away from visible-ball play.

## How we test it

`validate_gap.py`: exactly v11 (same models, hyper-parameters and decoder), leave-one-match-out on
the 7 IDSSE matches, official scorer. Training stays in the current bench; the held-out match is
degraded in four "ball-gap worlds" (`idsse_bench.set_world`):

| world | long blocks | link to stoppages |
|---|---|---|
| `legacy` | no (current bench) | none |
| `blocks_a1` | yes | none |
| `blocks_a10` | yes | blocks 10× more likely to start in a stoppage |
| `blocks_a30` | yes | 30× |

Adoption rule, fixed before running: v11+gap must beat v11 on mean spellF1 in every world with a
link to stoppages, and lose at most 0.002 in `legacy` and `blocks_a1`, where there is nothing to find.

## What the real ATD looks like (`atd_gap_stats.py`, inputs only)

On BAR-ATH, 59 % of graded frames see the ball, **no gap is shorter than 1 s**, 85 % of missing
frames sit in gaps > 10 s and 44 % in gaps > 30 s (frame-weighted median 25.6 s), and players move
at 58 % of their visible-ball speed inside long gaps. The `atd_*` worlds copy this gap shape
(lognormal blocks, median 11.2 s, sigma 0.91); `atd_stop` also loses the ball through 80 % of
stoppages until the restart, which reproduces the speed ratio (0.65). On BAR-ATH itself, the
label-free EM gives a dead-ball rate of 12 % with the ball visible, 89 % in 10–30 s gaps and 97.5 %
in gaps over 30 s.

## Results (leave-one-match-out, official scorer, `bench_results_gap.csv`)

| test world | v11 | v11 + gap evidence | matches won |
|---|---|---|---|
| `legacy` | 0.4861 | −0.0005 | 1/7 |
| `atd_a1` | 0.4448 | −0.0006 | 1/7 |
| `atd_a10` | 0.4528 | −0.0021 | 2/7 |
| `atd_stop` | 0.4791 | −0.0022 | 1/7 |

**Rejected.** The estimator works: the EM rates per gap class track the true ones closely (in
`atd_stop`, true 0.72 / 0.74 for 10–30 s / > 30 s gaps, EM 0.73 / 0.77). But v11 already reads the
same information through its event-timing features (`ev_since_seen`, `ev_until_seen`), and its
dead-frame recall stays at 93–97 % even in the realistic worlds, so adding the term counts the
evidence twice. The legacy row also reproduces v11's logged leave-one-match-out score on macOS to
six decimals.
