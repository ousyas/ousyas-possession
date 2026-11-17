"""Temporal head (v13): a small bidirectional GRU over v11's own per-frame features.

Why: v11's frame classifier sees at most +/- 2 s of context (rolling means and trends) plus the
event-timing features; possessions and stoppages last 5-30 s. Other teams measured the largest
gains of the challenge from sequence models over per-frame features (+0.04 to +0.06 spellF1).
Our v10 Transformer read raw players with ~95 k parameters and failed on 6 training matches; this
head reads the 61 features v11 already computes (43 frame + 18 event-timing), at 5 fps, with ~100 s
of context per training chunk, so it has far less to learn.

    bundle = train(datas, idx)                  # datas[j]: F, Fs (H<->A swapped), y, blocks
    P = predict(bundle, F, blocks)              # (n, 3) = P(H), P(A), P(D)

Ensembled 50/50 with v11's gradient-boosted probabilities (fixed a priori), then the unchanged v11
decoder (EM alive rate + turnover-driven Viterbi). Training is seeded and single-threaded; the
weights are shipped, so `reproduce.py --skip-train` rebuilds the file on any OS.
"""
import numpy as np

STEP = 5            # 25 fps -> 5 fps
CHUNK = 512         # ~102 s of context per training chunk
HIDDEN = 64
EPOCHS = 10
BATCH = 16
LR = 2e-3
SEED = 0


def _torch():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def _net(n_in):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU())
            self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = nn.Linear(2 * HIDDEN, 3)

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.out(h)
    return Net()


def norm_stats(Fs):
    F = np.concatenate([f[::25] for f in Fs]).astype(np.float64)
    q25, med, q75 = np.nanpercentile(F, [25, 50, 75], axis=0)
    iqr = q75 - q25
    iqr[~(iqr > 1e-6)] = 1.0
    return med.astype(np.float32), iqr.astype(np.float32)


def _norm(F, stats):
    med, iqr = stats
    return np.clip(np.nan_to_num((F - med) / iqr), -10, 10).astype(np.float32)


def _swap_y(y):
    return np.where(y == 0, 1, np.where(y == 1, 0, y))


def train(datas, idx):
    torch = _torch()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    stats = norm_stats([datas[j]["F"] for j in idx])
    seqs = []                                    # (features, labels) per half, both orientations
    for j in idx:
        d = datas[j]
        y = np.where(d["graded"], d["y"], -1)
        for F, yy in ((d["F"], y), (d["Fs"], _swap_y(y))):
            Fn = _norm(F, stats)
            for _, a, b in d["blocks"]:
                seqs.append((Fn[a:b + 1], yy[a:b + 1]))
    model = _net(seqs[0][0].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    for ep in range(EPOCHS):
        chunks = []
        for F, y in seqs:                        # one random 5 fps phase per half per epoch
            ph = int(rng.integers(STEP))
            Fp, yp = F[ph::STEP], y[ph::STEP]
            off = int(rng.integers(CHUNK))
            for s in range(-off, len(Fp), CHUNK):
                s0 = max(0, s)
                if len(Fp) - s0 >= CHUNK // 4:
                    chunks.append((Fp[s0:s0 + CHUNK], yp[s0:s0 + CHUNK]))
        order = rng.permutation(len(chunks))
        model.train()
        for k in range(0, len(order), BATCH):
            batch = [chunks[i] for i in order[k:k + BATCH]]
            L = max(len(c[0]) for c in batch)
            X = np.zeros((len(batch), L, batch[0][0].shape[1]), np.float32)
            Y = np.full((len(batch), L), -1, np.int64)
            for i, (f, y) in enumerate(batch):
                X[i, :len(f)], Y[i, :len(y)] = f, y
            opt.zero_grad()
            out = model(torch.from_numpy(X))
            loss = loss_fn(out.reshape(-1, 3), torch.from_numpy(Y).reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return {"state": {k: v.detach().numpy().copy() for k, v in model.state_dict().items()},
            "stats": stats, "n_in": seqs[0][0].shape[1]}


def predict(bundle, F, blocks):
    torch = _torch()
    model = _net(bundle["n_in"])
    model.load_state_dict({k: torch.from_numpy(v) for k, v in bundle["state"].items()})
    model.eval()
    Fn = _norm(F, bundle["stats"])
    P = np.full((len(F), 3), 1.0 / 3)
    with torch.no_grad():
        for _, a, b in blocks:
            for ph in range(STEP):
                idx = np.arange(a + ph, b + 1, STEP)
                out = model(torch.from_numpy(Fn[idx][None]))[0]
                P[idx] = torch.softmax(out, -1).numpy()
    return P
