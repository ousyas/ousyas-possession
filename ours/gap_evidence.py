"""Label-free ball-gap evidence for the dead-ball state.

Why. On the challenge ATD the ball does not disappear at random: it is lost in long blocks, and
long blocks tend to sit in stoppages (ball out of shot, held by a player, set-piece preparation).
How strongly is a property of the tracker, not of football, so it is NOT learned from IDSSE: the
training mask stays independent of the label, and the model learns nothing it could not transfer.

How. On the test game itself, frames are grouped by the length of the ball gap they sit in
(visible, <=1 s, 1-3 s, 3-10 s, 10-30 s, >30 s). Inside each group the dead-ball rate is
re-estimated with the EM of Saerens et al. (2002) -- the same estimator v6 uses for the global
alive rate -- from the classifier's own posteriors. No label of the test game is read. The
log-odds of the group rate against the global rate is added to the D emission at half weight
(we do not want to over-trust an estimate), and the decoder's usual alive-rate calibration is
unchanged, so this only MOVES dead frames towards long gaps and away from visible-ball play.

    lr, info = gap_log_ratio(P, bx_raw, graded, prior)
    P_adj = apply_log_ratio(P, lr)          # then em_alive(P) + decode_probs(P_adj, ...) as before
"""
import numpy as np

from v2 import gap_length

FPS = 25
GAP_EDGES_S = (1.0, 3.0, 10.0, 30.0)   # -> categories 0 visible | 1 <=1 s | 2 | 3 | 4 | 5 >30 s
N_CAT = len(GAP_EDGES_S) + 2
WEIGHT = 0.5                           # fixed a priori, not tuned
MIN_FRAMES = 30 * FPS                  # a category needs >= 30 s of graded frames to get a term


def gap_category(bx):
    """Category of every frame from the RAW ball track (NaN = missing), before any interpolation."""
    g = gap_length(np.asarray(bx)) / FPS
    cat = np.zeros(len(g), np.int8)
    miss = g > 0
    cat[miss] = 1 + np.searchsorted(GAP_EDGES_S, g[miss], side="left")
    return cat


def em_dead_rate(p_dead, prior_dead, iters=500, tol=1e-8):
    """Saerens EM, two classes (alive / dead): the dead rate that makes the prior-corrected mean
    posterior self-consistent. p_dead are classifier outputs under the training prior."""
    p = np.clip(np.asarray(p_dead, np.float64), 1e-6, 1 - 1e-6)
    pi = float(prior_dead)
    a, b = 1.0 / prior_dead, 1.0 / (1 - prior_dead)
    for _ in range(iters):
        num = p * pi * a
        q = num / (num + (1 - p) * (1 - pi) * b)
        new = float(q.mean())
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def _logit(x):
    x = min(max(x, 1e-6), 1 - 1e-6)
    return np.log(x / (1 - x))


def gap_log_ratio(P, bx, graded, prior, weight=WEIGHT, min_frames=MIN_FRAMES):
    """Per-frame additive log-odds term for D. Returns (lr, info) with info[c] = (frames, rate)."""
    cat = gap_category(bx)
    p_dead = P[:, 2]
    glob = em_dead_rate(p_dead[graded], prior[2])
    lr = np.zeros(len(P))
    info = {"global": glob}
    for c in range(N_CAT):
        m = graded & (cat == c)
        if m.sum() < min_frames:
            continue
        rate = em_dead_rate(p_dead[m], prior[2])
        lr[cat == c] = weight * (_logit(rate) - _logit(glob))
        info[c] = (int(m.sum()), rate)
    return lr, info


def apply_log_ratio(P, lr):
    """Shift the odds of D against alive by exp(lr); H:A ratio unchanged."""
    Q = np.asarray(P, np.float64).copy()
    Q[:, 2] *= np.exp(lr)
    return Q / Q.sum(1, keepdims=True)
