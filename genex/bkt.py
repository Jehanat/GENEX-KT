"""Bayesian Knowledge Tracing baseline (Corbett & Anderson, 1994).

One HMM per skill with parameters P(L0), P(T), P(G), P(S). Fitted by
maximising the likelihood of TRAIN responses with gradient descent on the
exact forward algorithm, vectorised over (student, skill) sequences.
Guess and slip are capped at 0.3 to avoid the degenerate solutions BKT is
known for. Predictions for val/test rows use the full preceding history.

Scaling (EdNet): parameters are fitted on up to `fit_seqs` sequences, each
truncated to `fit_len` steps; prediction runs over every sequence in
length-sorted chunks, so memory stays bounded.
"""
import numpy as np
import pandas as pd
import torch


class BKT:
    name = "BKT"

    def __init__(self, n_skills, max_gs=0.3, fit_seqs=200_000, fit_len=200, device="cpu"):
        self.K, self.max_gs = n_skills, max_gs
        self.fit_seqs, self.fit_len, self.device = fit_seqs, fit_len, device
        self.params = None

    @staticmethod
    def _index(df):
        df = df.sort_values(["user", "pos"])
        g = df.groupby(["user", "skill"], sort=False)
        return df, g.ngroup().to_numpy(), g.cumcount().to_numpy()

    def _dense(self, seq_id, step, values, ids, length):
        """Dense [len(ids), length] array for the selected sequence ids."""
        pos = np.full(seq_id.max() + 1, -1)
        pos[ids] = np.arange(len(ids))
        keep = (pos[seq_id] >= 0) & (step < length)
        out = {}
        for name, v, fill in values:
            a = np.full((len(ids), length), fill, dtype=np.int64)
            a[pos[seq_id[keep]], step[keep]] = v[keep]
            out[name] = torch.tensor(a, device=self.device)
        return out

    def _forward(self, raw, obs, skill):
        L0, T, G, S = (torch.sigmoid(raw[:, i])[skill] for i in range(4))
        G, S = G * self.max_gs, S * self.max_gs
        pL = L0
        preds = []
        for t in range(obs.shape[1]):
            pc = pL * (1 - S) + (1 - pL) * G
            preds.append(pc)
            o = obs[:, t]
            a, b = pL * (1 - S), (1 - pL) * G           # P(correct, known), P(correct, unknown)
            c, d = pL * S, (1 - pL) * (1 - G)            # P(wrong, known),   P(wrong, unknown)
            post_c = a / (a + b).clamp(min=1e-8)
            post_w = c / (c + d).clamp(min=1e-8)
            post = torch.where(o == 1, post_c, torch.where(o == 0, post_w, pL))
            pL = (post + (1 - post) * T).clamp(1e-6, 1 - 1e-6)
        return torch.stack(preds, 1)

    def fit(self, df, iters=150, lr=0.1, seed=0):
        df, seq_id, step = self._index(df)
        n_seq = seq_id.max() + 1
        ids = np.arange(n_seq)
        if n_seq > self.fit_seqs:
            ids = np.sort(np.random.default_rng(seed).choice(n_seq, self.fit_seqs, replace=False))
        length = int(min(self.fit_len, step.max() + 1))
        d = self._dense(seq_id, step, [("obs", df["correct"].to_numpy(), -1),
                                       ("split", df["split"].to_numpy(), 0)], ids, length)
        skill = np.zeros(n_seq, dtype=np.int64)
        skill[seq_id] = df["skill"].to_numpy()
        skill = torch.tensor(skill[ids], device=self.device)
        raw = torch.zeros(self.K + 1, 4, device=self.device)
        raw[:, 1], raw[:, 2], raw[:, 3] = -2.0, -0.5, -0.5
        raw.requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr)
        m = d["split"] == 1
        y = d["obs"][m].float()
        for _ in range(iters):
            p = self._forward(raw, d["obs"], skill)[m].clamp(1e-5, 1 - 1e-5)
            loss = torch.nn.functional.binary_cross_entropy(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.params = raw.detach()
        return float(loss.item())

    @torch.no_grad()
    def predict(self, df, chunk_cells=4_000_000):
        df, seq_id, step = self._index(df)
        n_seq = seq_id.max() + 1
        lens = np.bincount(seq_id, minlength=n_seq)
        skill_all = np.zeros(n_seq, dtype=np.int64)
        skill_all[seq_id] = df["skill"].to_numpy()
        corr = df["correct"].to_numpy()
        p_rows = np.zeros(len(df), dtype=np.float32)
        order = np.argsort(lens, kind="stable")
        i = 0
        while i < n_seq:                                     # chunks of similar length
            rest = lens[order[i:]]                           # ascending
            fits = np.arange(1, len(rest) + 1) * rest <= chunk_cells
            n = len(rest) if fits.all() else max(1, int(np.argmin(fits)))
            ids = order[i:i + n]
            L = int(lens[ids].max())
            d = self._dense(seq_id, step, [("obs", corr, -1)], ids, L)
            p = self._forward(self.params, d["obs"],
                              torch.tensor(skill_all[ids], device=self.device)).cpu().numpy()
            pos = np.full(n_seq, -1)
            pos[ids] = np.arange(len(ids))
            sel = pos[seq_id] >= 0
            p_rows[sel] = p[pos[seq_id[sel]], step[sel]]
            i += n
        out = df[["user", "pos", "split", "correct"]].copy()
        out["p"] = p_rows
        return out

    def describe(self, skill_names):
        r = torch.sigmoid(self.params).cpu()
        rows = []
        for k in range(1, self.K + 1):
            rows.append({"skill": k, "skill_name": skill_names.get(k, str(k)),
                         "p_L0": float(r[k, 0]), "p_T": float(r[k, 1]),
                         "p_G": float(r[k, 2]) * self.max_gs, "p_S": float(r[k, 3]) * self.max_gs})
        return pd.DataFrame(rows)
