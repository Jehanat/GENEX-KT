"""SHAP attribution for the GENEX-KT backbone (proposal: SHAP "has not been adapted
for the sequential, latent-state architectures characteristic of deep KT").

For one student and a few target concepts (their weakest), Kernel SHAP explains the
predicted mastery of each target in terms of *concept groups* of the history: a
feature is "all of the student's interactions on concept j". Removing a feature turns
those interactions into blank steps (no concept, no response), keeping every other
interaction and the sequence length unchanged; the all-removed state is the baseline.
The answer reads as: "this student's predicted mastery of Circle Graph is 0.19;
their answers on Circle Graph pull it down by 0.21, their answers on Median by 0.05...".
"""
import numpy as np
import torch

FEAT = ["skill", "correct", "tbin", "attempts", "item", "lag"]


@torch.no_grad()
def _mastery(model, b, masks, groups_pos, targets, att, lag=None):
    """Predicted mastery of `targets` at the final step for each mask row."""
    n = masks.shape[0]
    bb = {k: v.repeat(n, 1) for k, v in b.items()}
    for j, pos in enumerate(groups_pos):
        off = torch.as_tensor(masks[:, j] == 0, device=bb["skill"].device)
        if off.any():
            rows = off.nonzero().squeeze(1)
            for k in [f for f in FEAT if f in bb]:
                bb[k][rows[:, None], pos[None]] = 0
    x0 = model.clean_latent(bb)
    zero = torch.zeros(n, dtype=torch.long, device=x0.device)
    H, _ = model.denoise(x0, zero)
    h = H[:, -1]                                            # state after the last answer
    T = len(targets)
    sk = torch.as_tensor(targets, device=x0.device)[None].expand(n, T)
    at = torch.as_tensor(att, device=x0.device).clamp(
        max=model.query.att.num_embeddings - 1)[None].expand(n, T)
    hk = h[:, None, :].expand(n, T, -1)
    lg = None if lag is None else torch.as_tensor(lag, device=x0.device)[None].expand(n, T)
    fn = getattr(model, "concept_logits", None)
    out = fn(hk, sk, at, lg) if fn else model.predict_from_state(hk, sk, at)
    return torch.sigmoid(out).cpu().numpy()


def explain_student(model, b, targets, att, max_groups=20, nsamples=256, seed=0, lag=None):
    """b: batch dict for one student (window + dummy final step), as built by
    explain.student_batch. targets: concept ids (1..K). att: their attempt counts.
    Returns {target: {"value", "base", "phi": {concept_id: contribution}}}."""
    import shap
    skill = b["skill"][0, :-1].cpu().numpy()                # exclude the dummy step
    order = []
    for s in skill[::-1]:                                   # most recent first
        if s > 0 and s not in order:
            order.append(int(s))
    groups = order[:max_groups]
    groups_pos = [torch.as_tensor(np.where(skill == g)[0], device=b["skill"].device)
                  for g in groups]
    F = len(groups)
    if F == 0:
        return {}
    model.eval()

    def f(Z):
        return _mastery(model, b, np.asarray(Z), groups_pos, targets, att, lag)

    np.random.seed(seed)
    ex = shap.KernelExplainer(f, np.zeros((1, F)))
    sv = ex.shap_values(np.ones((1, F)), nsamples=min(nsamples, 2 ** F + 2), silent=True)
    sv = np.stack(sv, -1)[0] if isinstance(sv, list) else np.asarray(sv)[0]   # [F, T]
    full = f(np.ones((1, F)))[0]
    base = np.atleast_1d(ex.expected_value)
    out = {}
    for t, tgt in enumerate(targets):
        out[int(tgt)] = {"value": float(full[t]), "base": float(base[t]),
                         "phi": {int(groups[j]): float(sv[j, t]) for j in range(F)}}
    return out


def evidence_lines(shap_res, names, top=3):
    """Short text lines for the explainer prompt."""
    lines = []
    for tgt, r in shap_res.items():
        top_items = sorted(r["phi"].items(), key=lambda kv: -abs(kv[1]))[:top]
        parts = ", ".join(f"{names[c]} {v:+.2f}" for c, v in top_items)
        lines.append(f"- {names[tgt]} (mastery {r['value']:.2f}): biggest influences - {parts}")
    return lines
