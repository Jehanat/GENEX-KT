"""Phase 2 - Knowledge tracing models.

All models share one contract:
    logits = model(batch)            # [B, L]
predicting P(correct at position t) from interactions < t plus the skill
(and attempt count) of the item at t. Inputs are shifted one step to the
right, so position t only ever sees interaction t-1 and earlier.

Models
  * GenexDiffusionKT - the proposed DDPM-style generative backbone
  * DKT               - LSTM baseline (Piech et al., 2015)
  * SAKT              - self-attentive baseline (Pandey & Karypis, 2019)
  * AKTLite           - attention with monotonic distance decay, a simplified
                        AKT (Ghosh et al., 2020); use pyKT's AKT for the exact model
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def shift_right(x, start_value):
    out = torch.roll(x, 1, dims=1)
    out[:, 0] = start_value
    return out


def history_inputs(batch):
    """Indices describing interaction t-1 at position t.
    interaction id: 0 pad, 1 start token, 2 + 2*(skill-1) + correct otherwise."""
    skill, correct = batch["skill"], batch["correct"]
    inter = torch.where(skill > 0, 2 + 2 * (skill - 1) + correct, torch.zeros_like(skill))
    return {
        "inter": shift_right(inter, 1),
        "tbin": shift_right(batch["tbin"], 0),
        "attempts": shift_right(batch["attempts"], 0),
    }


def causal_mask(L, device):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)


class InteractionEmbedding(nn.Module):
    """Embeds the 4-D feature vector of the previous interaction."""

    def __init__(self, n_skills, d, n_time_bins, max_attempt):
        super().__init__()
        self.inter = nn.Embedding(2 * n_skills + 2, d, padding_idx=0)
        self.tbin = nn.Embedding(n_time_bins + 1, d)
        self.att = nn.Embedding(max_attempt + 1, d)

    def forward(self, h):
        return self.inter(h["inter"]) + self.tbin(h["tbin"]) + self.att(h["attempts"])


class QueryEmbedding(nn.Module):
    """Embeds what is known about the item being predicted (skill, attempts so far)."""

    def __init__(self, n_skills, d, max_attempt):
        super().__init__()
        self.skill = nn.Embedding(n_skills + 1, d, padding_idx=0)
        self.att = nn.Embedding(max_attempt + 1, d)

    def forward(self, skill, attempts):
        return self.skill(skill) + self.att(attempts)


def masked_bce(logits, batch, split_code):
    """BCE over positions whose split code is in split_code (int or tuple)."""
    codes = (split_code,) if isinstance(split_code, int) else split_code
    m = torch.zeros_like(batch["split"], dtype=torch.bool)
    for c in codes:
        m |= batch["split"] == c
    if m.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[m], batch["correct"][m].float())


# --------------------------------------------------------------------------
# GENEX-KT diffusion backbone
# --------------------------------------------------------------------------
def cosine_alpha_bar(T, s=0.008):
    """Cumulative signal level abar_k for k = 0..T (index 0 = clean).

    Uses Hugging Face Diffusers' DDPMScheduler with its cosine schedule
    ("squaredcos_cap_v2", Nichol & Dhariwal 2021), as named in the proposal. Falls
    back to the identical closed-form schedule if diffusers is not installed."""
    try:
        from diffusers import DDPMScheduler
        sch = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2")
        ab = sch.alphas_cumprod.to(torch.float64)
        return torch.cat([torch.ones(1, dtype=torch.float64), ab]).float()
    except ImportError:
        pass
    t = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(min=1e-5)
    # enforce beta <= 0.999 as in Nichol & Dhariwal (2021)
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    ab = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1 - betas, 0)])
    return ab.float()  # index 0 = clean, 1..T = increasingly noisy


class SinusoidalStep(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, k):
        half = self.d // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=k.device) / half)
        a = k.float()[:, None] * freq[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=-1))


class GenexDiffusionKT(nn.Module):
    """Forward process: x_k = sqrt(abar_k) x0 + sqrt(1-abar_k) eps, where x0 is the
    (normalised) embedded interaction history. Reverse network: a causal
    transformer encoder that predicts x0 from x_k. Its last hidden layer H_t is
    the knowledge-state vector used for prediction, explanation and intervention."""

    name = "GENEX-KT"

    def __init__(self, n_skills, n_time_bins, max_attempt, max_len, d_model=128, n_heads=4,
                 n_layers=2, dropout=0.2, diffusion_steps=100, lambda_diff=0.5,
                 lambda_noisy_bce=0.5, mc_samples=8, mc_noise_step=10, **_):
        super().__init__()
        self.n_skills, self.T = n_skills, diffusion_steps
        self.lambda_diff, self.lambda_noisy_bce = lambda_diff, lambda_noisy_bce
        self.mc_samples, self.mc_noise_step = mc_samples, mc_noise_step
        self.embed = InteractionEmbedding(n_skills, d_model, n_time_bins, max_attempt)
        self.norm0 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.query = QueryEmbedding(n_skills, d_model, max_attempt)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.step = SinusoidalStep(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.denoiser = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.x0_head = nn.Linear(d_model, d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.register_buffer("alpha_bar", cosine_alpha_bar(diffusion_steps))

    def init_concept_embeddings(self, vectors):
        """Initialise concept embeddings from pretrained word vectors (proposal:
        word2vec 'where available'). vectors: [K, D] array, rows of NaN = no name
        vector (kept random). Projected to d_model by PCA, then scaled to the
        embedding's own std."""
        import numpy as np
        V = np.asarray(vectors, dtype=np.float64)
        ok = ~np.isnan(V).any(1)
        if ok.sum() < 2:
            return 0
        d = self.query.skill.embedding_dim
        X = V[ok] - V[ok].mean(0)
        U, S, Wt = np.linalg.svd(X, full_matrices=False)
        k = min(d, Wt.shape[0])
        P = X @ Wt[:k].T                                   # [n_ok, k]
        if k < d:                                          # pad with small noise dims
            P = np.concatenate([P, np.random.default_rng(0).normal(0, P.std() * 0.1,
                                                                   (len(P), d - k))], 1)
        with torch.no_grad():
            W = self.query.skill.weight
            P = torch.tensor(P, dtype=W.dtype)
            P = P / P.std() * W[1:].std()
            idx = torch.tensor(np.where(ok)[0] + 1)
            W[idx] = P
            Wi = self.embed.inter.weight                   # (skill, correct) rows
            for c in (0, 1):
                rows = 2 + 2 * (idx - 1) + c
                Wi[rows] = P + 0.5 * Wi[rows]
        return int(ok.sum())

    # --- pieces ---------------------------------------------------------
    def clean_latent(self, batch):
        return self.norm0(self.embed(history_inputs(batch)))

    def noise(self, x0, k):
        ab = self.alpha_bar[k][:, None, None]
        eps = torch.randn_like(x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def denoise(self, xk, k):
        B, L, _ = xk.shape
        pos = torch.arange(L, device=xk.device)
        h = xk + self.pos(pos)[None] + self.step(k)[:, None, :]
        h = self.denoiser(h, mask=causal_mask(L, xk.device), is_causal=True)
        h = self.out_norm(h)
        return h, self.x0_head(h)            # knowledge state, x0 estimate

    def predict_from_state(self, H, skill, attempts):
        q = self.query(skill, attempts)
        return self.head(torch.cat([H, q], dim=-1)).squeeze(-1)

    # --- public API -----------------------------------------------------
    def forward(self, batch, k=None):
        x0 = self.clean_latent(batch)
        B = x0.shape[0]
        if k is None:
            k = torch.zeros(B, dtype=torch.long, device=x0.device)
            xk = x0
        else:
            xk = self.noise(x0, k)
        H, _ = self.denoise(xk, k)
        return self.predict_from_state(H, batch["skill"], batch["attempts"])

    def training_loss(self, batch):
        x0 = self.clean_latent(batch)
        B = x0.shape[0]
        zero = torch.zeros(B, dtype=torch.long, device=x0.device)
        H, _ = self.denoise(x0, zero)
        loss = masked_bce(self.predict_from_state(H, batch["skill"], batch["attempts"]), batch, 1)
        # noised pass: denoising objective + prediction from a noisy history
        k = torch.randint(1, self.T + 1, (B,), device=x0.device)
        Hn, x0_hat = self.denoise(self.noise(x0, k), k)
        valid = batch["split"] > 0
        diff = ((x0_hat - x0.detach()) ** 2).mean(-1)[valid].mean()
        bce_n = masked_bce(self.predict_from_state(Hn, batch["skill"], batch["attempts"]), batch, 1)
        return loss + self.lambda_diff * diff + self.lambda_noisy_bce * bce_n

    @torch.no_grad()
    def predict_mc(self, batch, samples=None, noise_step=None):
        """Generative sampling: re-noise the history to a small step and denoise
        M times. Returns mean and std of P(correct) - the std is the model's
        uncertainty about the knowledge state."""
        samples = samples or self.mc_samples
        noise_step = noise_step or self.mc_noise_step
        x0 = self.clean_latent(batch)
        B = x0.shape[0]
        k = torch.full((B,), noise_step, dtype=torch.long, device=x0.device)
        ps = []
        for _ in range(samples):
            H, _ = self.denoise(self.noise(x0, k), k)
            ps.append(torch.sigmoid(self.predict_from_state(H, batch["skill"], batch["attempts"])))
        ps = torch.stack(ps)
        return ps.mean(0), ps.std(0)

    def concept_logits(self, H, skill, attempts, lag=None):
        """Mastery query for a concept (no specific item): used by explanation and SHAP."""
        return self.predict_from_state(H, skill, attempts)

    @torch.no_grad()
    def mastery_profile(self, batch, attempts_per_skill, samples=None, noise_step=None, lag_per_skill=None):
        """Knowledge state after the LAST interaction in each row of `batch`
        (rows must end with one dummy padding step appended by the caller),
        queried against every skill. Returns state [B,d], mean/std mastery [B,K]."""
        samples = samples or self.mc_samples
        noise_step = noise_step or self.mc_noise_step
        lengths = (batch["skill"] > 0).sum(1)          # dummy step has skill>0
        x0 = self.clean_latent(batch)
        B, K = x0.shape[0], self.n_skills
        idx = torch.arange(B, device=x0.device)
        skills = torch.arange(1, K + 1, device=x0.device)[None].expand(B, K)
        att = attempts_per_skill.clamp(max=self.query.att.num_embeddings - 1)

        def query_all(H):
            h_last = H[idx, lengths - 1]                       # [B,d]
            hk = h_last[:, None, :].expand(B, K, -1)
            return h_last, torch.sigmoid(self.concept_logits(hk, skills, att, lag_per_skill))

        zero = torch.zeros(B, dtype=torch.long, device=x0.device)
        state, _ = self.denoise(x0, zero)
        state, clean = query_all(state)
        k = torch.full((B,), noise_step, dtype=torch.long, device=x0.device)
        draws = torch.stack([query_all(self.denoise(self.noise(x0, k), k)[0])[1]
                             for _ in range(samples)])
        return state, clean, draws.std(0)


class GenexRaschKT(GenexDiffusionKT):
    """Exploratory ablation (post hoc, Chapter 5): GENEX-KT plus AKT-style Rasch
    item difficulty. Each item gets one scalar difficulty mu_i; the concept
    embedding is shifted by mu_i along a learned per-concept variation vector,
    both for the past interaction (input) and the item being predicted (query).
    Everything else - diffusion, objective, MC uncertainty - is unchanged."""

    name = "GENEX-KT + Rasch"

    def __init__(self, n_skills, n_time_bins, max_attempt, max_len, n_items=1, d_model=128, **kw):
        super().__init__(n_skills, n_time_bins, max_attempt, max_len, d_model=d_model, **kw)
        self.mu = nn.Embedding(n_items + 1, 1, padding_idx=0)
        nn.init.zeros_(self.mu.weight)
        self.var_q = nn.Embedding(n_skills + 1, d_model, padding_idx=0)
        self.var_i = nn.Embedding(2 * n_skills + 2, d_model, padding_idx=0)

    def clean_latent(self, batch):
        h = history_inputs(batch)
        x = self.embed(h) + self.mu(shift_right(batch["item"], 0)) * self.var_i(h["inter"])
        return self.norm0(x)

    def _query_shift(self, batch):
        return self.mu(batch["item"]) * self.var_q(batch["skill"])

    def predict_from_state(self, H, skill, attempts, shift=None):
        q = self.query(skill, attempts)
        if shift is not None:
            q = q + shift
        return self.head(torch.cat([H, q], dim=-1)).squeeze(-1)

    def forward(self, batch, k=None):
        x0 = self.clean_latent(batch)
        B = x0.shape[0]
        if k is None:
            k = torch.zeros(B, dtype=torch.long, device=x0.device)
            xk = x0
        else:
            xk = self.noise(x0, k)
        H, _ = self.denoise(xk, k)
        return self.predict_from_state(H, batch["skill"], batch["attempts"], self._query_shift(batch))

    def training_loss(self, batch):
        x0 = self.clean_latent(batch)
        B = x0.shape[0]
        sh = self._query_shift(batch)
        zero = torch.zeros(B, dtype=torch.long, device=x0.device)
        H, _ = self.denoise(x0, zero)
        loss = masked_bce(self.predict_from_state(H, batch["skill"], batch["attempts"], sh), batch, 1)
        k = torch.randint(1, self.T + 1, (B,), device=x0.device)
        Hn, x0_hat = self.denoise(self.noise(x0, k), k)
        valid = batch["split"] > 0
        diff = ((x0_hat - x0.detach()) ** 2).mean(-1)[valid].mean()
        bce_n = masked_bce(self.predict_from_state(Hn, batch["skill"], batch["attempts"], sh), batch, 1)
        return loss + self.lambda_diff * diff + self.lambda_noisy_bce * bce_n

    @torch.no_grad()
    def predict_mc(self, batch, samples=None, noise_step=None):
        samples = samples or self.mc_samples
        noise_step = noise_step or self.mc_noise_step
        x0 = self.clean_latent(batch)
        B = x0.shape[0]
        sh = self._query_shift(batch)
        k = torch.full((B,), noise_step, dtype=torch.long, device=x0.device)
        ps = torch.stack([torch.sigmoid(self.predict_from_state(
            self.denoise(self.noise(x0, k), k)[0], batch["skill"], batch["attempts"], sh))
            for _ in range(samples)])
        return ps.mean(0), ps.std(0)


class GenexV2KT(GenexRaschKT):
    """GENEX-KT v2 (accuracy study): GENEX-KT + Rasch item difficulty, plus
    (1) forgetting-aware attention - each head adds a learned penalty that grows
        linearly with the distance between two interactions (in the spirit of AKT's
        monotonic attention), and
    (2) a forgetting feature - interactions since the learner last practised the same
        concept, for both the past interaction and the item being predicted.
    The diffusion objective and Monte Carlo uncertainty are unchanged."""

    name = "GENEX-KT v2"

    def __init__(self, n_skills, n_time_bins, max_attempt, max_len, n_items=1, d_model=128,
                 n_heads=4, n_lag_bins=13, **kw):
        super().__init__(n_skills, n_time_bins, max_attempt, max_len, n_items=n_items,
                         d_model=d_model, n_heads=n_heads, **kw)
        self.nh = n_heads
        self.lag_h = nn.Embedding(n_lag_bins + 1, d_model)
        self.lag_q = nn.Embedding(n_lag_bins + 1, d_model)
        # per-head decay rates, spread from slow to fast (softplus^-1 of 0.005..0.2)
        rates = torch.logspace(math.log10(0.005), math.log10(0.2), n_heads)
        self.decay = nn.Parameter(torch.log(torch.expm1(rates)))

    def clean_latent(self, batch):
        h = history_inputs(batch)
        x = self.embed(h) + self.mu(shift_right(batch["item"], 0)) * self.var_i(h["inter"])
        x = x + self.lag_h(shift_right(batch["lag"], 0))
        return self.norm0(x)

    def _query_shift(self, batch):
        return self.mu(batch["item"]) * self.var_q(batch["skill"]) + self.lag_q(batch["lag"])

    def concept_logits(self, H, skill, attempts, lag=None):
        shift = self.lag_q(lag) if lag is not None else None
        return self.predict_from_state(H, skill, attempts, shift)

    def denoise(self, xk, k):
        B, L, _ = xk.shape
        pos = torch.arange(L, device=xk.device)
        h = xk + self.pos(pos)[None] + self.step(k)[:, None, :]
        dist = (pos[:, None] - pos[None, :]).float()                         # i - j
        bias = -F.softplus(self.decay)[:, None, None] * dist.clamp(min=0)    # [H, L, L]
        bias = bias.masked_fill(dist[None] < 0, float("-inf"))
        mask = bias[None].expand(B, -1, -1, -1).reshape(B * self.nh, L, L).to(h.dtype)
        h = self.denoiser(h, mask=mask)
        h = self.out_norm(h)
        return h, self.x0_head(h)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
class DKT(nn.Module):
    name = "DKT"

    def __init__(self, n_skills, n_time_bins, max_attempt, max_len, d_model=128,
                 dropout=0.2, **_):
        super().__init__()
        # classic DKT input: (skill, correct) only
        self.inter = nn.Embedding(2 * n_skills + 2, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, n_skills + 1)

    def forward(self, batch):
        h, _ = self.lstm(self.inter(history_inputs(batch)["inter"]))
        y = self.out(self.drop(h))
        return y.gather(-1, batch["skill"].unsqueeze(-1)).squeeze(-1)

    def training_loss(self, batch):
        return masked_bce(self(batch), batch, 1)


class CausalAttention(nn.Module):
    """Multi-head attention over positions <= t, with optional AKT-style
    monotonic decay: score_ij -= softplus(gamma_h) * (i - j)."""

    def __init__(self, d, heads, dropout, decay=False):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)
        self.decay = decay
        if decay:
            self.gamma = nn.Parameter(torch.full((heads,), -2.0))

    def forward(self, q, kv):
        B, L, d = q.shape
        Q = self.q(q).view(B, L, self.h, self.dk).transpose(1, 2)
        K = self.k(kv).view(B, L, self.h, self.dk).transpose(1, 2)
        V = self.v(kv).view(B, L, self.h, self.dk).transpose(1, 2)
        s = Q @ K.transpose(-1, -2) / math.sqrt(self.dk)
        if self.decay:
            pos = torch.arange(L, device=q.device)
            dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()
            s = s - F.softplus(self.gamma)[None, :, None, None] * dist
        s = s.masked_fill(causal_mask(L, q.device), float("-inf"))
        a = self.drop(torch.softmax(s, -1))
        return self.o((a @ V).transpose(1, 2).reshape(B, L, d))


class AttnBlock(nn.Module):
    def __init__(self, d, heads, dropout, decay):
        super().__init__()
        self.attn = CausalAttention(d, heads, dropout, decay)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv):
        x = self.n1(q + self.drop(self.attn(q, kv)))
        return self.n2(x + self.drop(self.ff(x)))


class SAKT(nn.Module):
    name = "SAKT"
    decay = False

    def __init__(self, n_skills, n_time_bins, max_attempt, max_len, d_model=128, n_heads=4,
                 n_layers=1, dropout=0.2, **_):
        super().__init__()
        self.inter = nn.Embedding(2 * n_skills + 2, d_model, padding_idx=0)
        self.skill = nn.Embedding(n_skills + 1, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.encoder = nn.ModuleList(AttnBlock(d_model, n_heads, dropout, self.decay)
                                     for _ in range(n_layers - 1))
        self.retriever = AttnBlock(d_model, n_heads, dropout, self.decay)
        self.out = nn.Linear(d_model, 1)

    def forward(self, batch):
        L = batch["skill"].shape[1]
        p = self.pos(torch.arange(L, device=batch["skill"].device))[None]
        kv = self.inter(history_inputs(batch)["inter"]) + p
        for blk in self.encoder:           # SAKT: n_layers=1 -> no encoder
            kv = blk(kv, kv)
        q = self.skill(batch["skill"]) + p
        return self.out(self.retriever(q, kv)).squeeze(-1)

    def training_loss(self, batch):
        return masked_bce(self(batch), batch, 1)


class AKTLite(SAKT):
    name = "AKT-lite"
    decay = True

    def __init__(self, *a, n_layers=2, **kw):
        super().__init__(*a, n_layers=max(2, n_layers), **kw)


# --------------------------------------------------------------------------
# pyKT reference baselines (proposal: "implemented through the pyKT library")
# --------------------------------------------------------------------------
_PYKT = {}


def _load_pykt():
    """Load pyKT's dkt/sakt/akt model files directly. pyKT's package __init__
    imports every model (one of which needs tkinter), so the three files are
    loaded on their own; the model code itself is used unchanged."""
    if _PYKT:
        return _PYKT
    import importlib.util
    import os
    import sys
    import types
    spec = importlib.util.find_spec("pykt")
    if spec is None:
        raise ImportError("pip install pykt-toolkit")
    base = os.path.join(os.path.dirname(spec.origin), "models")
    pkg = types.ModuleType("_pykt_models")
    pkg.__path__ = [base]
    sys.modules["_pykt_models"] = pkg
    for name in ["utils", "dkt", "sakt", "akt"]:
        s = importlib.util.spec_from_file_location(f"_pykt_models.{name}",
                                                   os.path.join(base, name + ".py"))
        m = importlib.util.module_from_spec(s)
        sys.modules[s.name] = m
        s.loader.exec_module(m)
        _PYKT[name] = m
    return _PYKT


# settings follow pyKT's example training scripts; not re-tuned per dataset
PYKT_DEFAULTS = {
    "dkt": dict(emb_size=200, dropout=0.1),
    "sakt": dict(emb_size=256, num_attn_heads=8, dropout=0.2, num_en=2),
    "akt": dict(d_model=256, n_blocks=4, dropout=0.05, d_ff=256, kq_same=1,
                final_fc_dim=512, num_attn_heads=8, l2=1e-5),
}
PYKT_LR = {"dkt": 1e-3, "sakt": 1e-3, "akt": 1e-4}
PYKT_BATCH = {"akt": 32}          # AKT's attention memory on a 15 GB T4 at length 200


class PyKTBaseline(nn.Module):
    """A pyKT model run on exactly the same windows and split as GENEX-KT.

    pyKT predicts position t+1 from positions <= t, so it has no prediction for
    the first step of a sequence. A START concept (index K) with response 0 is
    prepended so every position gets a prediction, like the other models here."""

    def __init__(self, kind, n_skills, n_items, max_len, **_):
        super().__init__()
        m = _load_pykt()
        self.kind, self.K = kind, n_skills
        self.name = {"dkt": "DKT", "sakt": "SAKT", "akt": "AKT"}[kind] + " (pyKT)"
        num_c, hp = n_skills + 1, PYKT_DEFAULTS[kind]
        if kind == "dkt":
            self.m = m["dkt"].DKT(num_c, **hp)
        elif kind == "sakt":
            self.m = m["sakt"].SAKT(num_c, seq_len=max_len, **hp)
        else:
            self.m = m["akt"].AKT(n_question=num_c, n_pid=n_items, **hp)
        self._reg = 0.0

    def _seq(self, batch):
        skill = batch["skill"]
        valid = skill > 0
        zero = torch.zeros_like(skill)
        c = torch.where(valid, skill - 1, zero)            # 0..K-1
        r = torch.where(valid, batch["correct"], zero)
        q = torch.where(valid, batch["item"], zero)         # 0 = none, 1..Q
        start = torch.full_like(c[:, :1], self.K)
        return c, r, q, start, zero[:, :1]

    def forward(self, batch):
        c, r, q, start, z = self._seq(batch)
        if self.kind in ("dkt", "sakt"):
            hc = torch.cat([start, c[:, :-1]], 1)           # history shifted by one
            hr = torch.cat([z, r[:, :-1]], 1)
            if self.kind == "dkt":
                p = self.m(hc, hr).gather(-1, c.unsqueeze(-1)).squeeze(-1)
            else:
                p = self.m(hc, hr, c)
        else:
            y, reg = self.m(torch.cat([start, c], 1), torch.cat([z, r], 1),
                            torch.cat([z, q], 1))
            p, self._reg = y[:, 1:], reg
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)              # logits

    def training_loss(self, batch):
        loss = masked_bce(self(batch), batch, 1)
        return loss + (self._reg if self.kind == "akt" else 0.0)


def _pykt(kind):
    def make(n_skills, n_time_bins, max_attempt, max_len, n_items=1, **hp):
        return PyKTBaseline(kind, n_skills, n_items, max_len)
    return make


MODELS = {"genex": GenexDiffusionKT, "genex_rasch": GenexRaschKT, "genex_v2": GenexV2KT,
          "dkt": _pykt("dkt"), "sakt": _pykt("sakt"), "akt": _pykt("akt"),
          # earlier self-written baselines, kept for reference / offline use
          "dkt_lite": DKT, "sakt_lite": SAKT, "akt_lite": AKTLite}


def build_model(kind, meta, **hp):
    return MODELS[kind](n_skills=meta["n_skills"], n_time_bins=meta["n_time_bins"],
                        max_attempt=meta["max_attempt_bucket"], max_len=meta["max_len"],
                        n_items=meta.get("n_items", meta["n_skills"]),
                        **({"n_lag_bins": meta.get("n_lag_bins", 13)} if kind == "genex_v2" else {}), **hp)
