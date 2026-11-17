"""v10 -- Transformer "joueurs puis temps" pour P(H), P(A), P(D) par frame.

  1. Chaque frame (sous-echantillonnee a 5 fps) = un ENSEMBLE de jetons : jusqu'a 24 joueurs
     visibles + le ballon. Jeton joueur : position, position dans le sens d'attaque de son equipe,
     vitesse, equipe (+1 H / -1 A), distance au ballon. Aucune identite de joueur : l'attention
     est indifferente a l'ordre des jetons.
  2. Transformer spatial (2 couches) -> un vecteur par frame, concatene aux features v5 + mobilite.
  3. Transformer temporel bidirectionnel (2 couches) sur des fenetres de 10 s.
  4. Sorties remontees a 25 fps, puis MEME decodage que v8 (Viterbi, detecteur de changements, EM).
Augmentation : echange H <-> A (equipes, features, labels). Entraine uniquement sur IDSSE.
"""
import numpy as np

FPS = 25
STEP = 5                    # 25 fps -> 5 fps
WIN = 50                    # 10 s a 5 fps
STRIDE = 25                 # fenetres qui se chevauchent de moitie
MAX_P = 24
LEN_M, WID_M = 105.0, 68.0
TOK_DIM = 9


def attack_sign(px, teams, blocks, n):
    sign = np.ones(n, np.float32)
    for _, a, b in blocks:
        with np.errstate(all="ignore"):
            sign[a:b + 1] = 1.0 if np.nanmean(px[a:b + 1][:, teams == 0]) < 0.5 else -1.0
    return sign


def build_tokens(tracking, blocks):
    """Jetons a 5 fps : P (m, MAX_P, TOK_DIM), masque joueurs (m, MAX_P), B (m, 6)."""
    teams, px, py, ps, bx, by, bs = tracking
    n = len(bx)
    sign_h = attack_sign(px, teams, blocks, n)
    idx = np.arange(0, n, STEP)
    X, Y = px.astype(np.float64) * LEN_M, py.astype(np.float64) * WID_M
    VX = np.diff(X, axis=0, prepend=np.nan) * FPS
    VY = np.diff(Y, axis=0, prepend=np.nan) * FPS
    BX, BY = bx.astype(np.float64) * LEN_M, by.astype(np.float64) * WID_M
    BVX = np.diff(BX, prepend=np.nan) * FPS
    BVY = np.diff(BY, prepend=np.nan) * FPS
    X, Y, VX, VY = X[idx], Y[idx], VX[idx], VY[idx]
    BX, BY, BVX, BVY, sg = BX[idx], BY[idx], BVX[idx], BVY[idx], sign_h[idx]

    vis = ~np.isnan(X)
    order = np.argsort(~vis, axis=1, kind="stable")[:, :MAX_P]        # visibles d'abord
    take = lambda a: np.take_along_axis(a, order, 1)                   # noqa: E731
    X, Y, VX, VY, vis = take(X), take(Y), take(VX), take(VY), take(vis)
    team = np.broadcast_to(np.where(teams == 0, 1.0, -1.0), (len(idx), len(teams)))
    team = take(np.ascontiguousarray(team))
    att = np.where(team > 0, sg[:, None], -sg[:, None])                # sens d'attaque du joueur
    xc = (X - LEN_M / 2) / (LEN_M / 2)
    yc = (Y - WID_M / 2) / (WID_M / 2)
    bvis = ~np.isnan(BX)
    dball = np.hypot(X - BX[:, None], Y - BY[:, None]) / 30.0
    P = np.stack([xc, yc, xc * att, VX / 10, VY / 10, (VX * att) / 10, team,
                  np.where(bvis[:, None], dball, 0.0), np.repeat(bvis[:, None], MAX_P, 1)], -1)
    P = np.nan_to_num(P).astype(np.float32)
    P[~vis] = 0.0
    bxc = (BX - LEN_M / 2) / (LEN_M / 2)
    B = np.stack([bxc, (BY - WID_M / 2) / (WID_M / 2), bxc * sg, BVX / 30, BVY / 30, bvis], -1)
    B = np.nan_to_num(B).astype(np.float32)
    B[~bvis, :5] = 0.0
    return P, vis, B


def swap_tokens(P, B):
    """Echange H <-> A : signe d'equipe, et coordonnees ballon dans le sens d'attaque de H."""
    P, B = P.copy(), B.copy()
    P[..., 6] = -P[..., 6]
    B[:, 2] = -B[:, 2]
    return P, B


def make_model(frame_dim, d=32, d_t=64):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Linear(TOK_DIM, d)
            self.ball = nn.Linear(6, d)
            self.spatial = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True), 2)
            self.frame = nn.Sequential(nn.Linear(2 * d + frame_dim, d_t), nn.GELU())
            self.pos = nn.Parameter(torch.zeros(1, WIN, d_t))
            self.temporal = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_t, 4, 2 * d_t, dropout=0.1, batch_first=True), 2)
            self.head = nn.Linear(d_t, 3)

        def forward(self, P, M, B, F):
            # P (b, T, MAX_P, TOK_DIM), M (b, T, MAX_P) visibles, B (b, T, 6), F (b, T, frame_dim)
            b, T = P.shape[:2]
            tok = torch.cat([self.ball(B).reshape(b * T, 1, -1),
                             self.tok(P).reshape(b * T, MAX_P, -1)], 1)
            pad = torch.cat([torch.zeros(b * T, 1, dtype=torch.bool, device=P.device),
                             ~M.reshape(b * T, MAX_P)], 1)
            h = self.spatial(tok, src_key_padding_mask=pad)
            w = (~pad).float().unsqueeze(-1)
            pooled = (h * w).sum(1) / w.sum(1).clamp(min=1)
            z = torch.cat([h[:, 0], pooled], -1).reshape(b, T, -1)
            z = self.frame(torch.cat([z, F], -1)) + self.pos[:, :T]
            return self.head(self.temporal(z))

    return Net()


def windows(m):
    starts = list(range(0, max(1, m - WIN + 1), STRIDE))
    if starts[-1] + WIN < m:
        starts.append(m - WIN)
    return starts


class Data:
    """Une sequence (match) prete pour le reseau, a 5 fps."""

    def __init__(self, P, vis, B, F, y=None):
        self.P, self.vis, self.B, self.F, self.y = P, vis, B, F, y


def train(seqs, epochs=6, seed=0, threads=12, verbose=True):
    import time
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    frame_dim = seqs[0].F.shape[1]
    model = make_model(frame_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    def epoch_items(offset):
        # entrainement : fenetres SANS chevauchement, decalees aleatoirement a chaque epoque
        return [(si, s) for si, sq in enumerate(seqs) for s in range(offset, len(sq.y) - WIN + 1, WIN)]
    per_epoch = (len(epoch_items(0)) + 63) // 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * per_epoch + 10)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-1)
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        items = epoch_items(int(np.random.randint(WIN)))
        model.train()
        perm = np.random.permutation(len(items))
        for k in range(0, len(perm), 64):
            batch = [items[i] for i in perm[k:k + 64]]
            get = lambda a: torch.from_numpy(np.stack([getattr(seqs[si], a)[s:s + WIN] for si, s in batch]))  # noqa: E731
            out = model(get("P"), get("vis"), get("B"), get("F"))
            loss = lossf(out.reshape(-1, 3), get("y").reshape(-1).long())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(batch)
        if verbose:
            print("    epoque %d : perte %.4f (%.0f s)" % (ep + 1, tot / len(items), time.time() - t0), flush=True)
    return model


def predict(model, sq, n_frames):
    """Probabilites (n_frames, 3) a 25 fps (moyenne des fenetres qui se chevauchent)."""
    import torch
    model.eval()
    m = len(sq.P)
    acc = np.zeros((m, 3))
    cnt = np.zeros(m)
    starts = windows(m)
    with torch.no_grad():
        for k in range(0, len(starts), 64):
            ss = starts[k:k + 64]
            get = lambda a: torch.from_numpy(np.stack([getattr(sq, a)[s:s + WIN] for s in ss]))  # noqa: E731
            pr = torch.softmax(model(get("P"), get("vis"), get("B"), get("F")), -1).numpy()
            for j, s in enumerate(ss):
                acc[s:s + WIN] += pr[j]
                cnt[s:s + WIN] += 1
    p5 = acc / np.maximum(cnt, 1)[:, None]
    t5 = np.arange(m) * STEP + STEP // 2
    t = np.arange(n_frames)
    P = np.stack([np.interp(t, t5, p5[:, c]) for c in range(3)], 1)
    return P / P.sum(1, keepdims=True)
