"""Phase 3 - LLM-based explainability module.

For each student:
  1. run the trained GENEX-KT backbone over their whole history and take the
     knowledge state after the final interaction;
  2. query that state against every skill -> mastery mean and uncertainty;
  3. project the state with a linear (PCA) projection to a compact descriptor;
  4. add structured metadata (accuracy, recent trend, attempts, labels);
  5. prompt an LLM with chain-of-thought instructions to write a three-part
     diagnosis: demonstrated mastery, evidenced knowledge gaps, review priority;
  6. check the output for faithfulness against the model's own numbers.

Backends: "openai" (GPT-4o; needs OPENAI_API_KEY), "hf" (local
LLaMA-3-8B-Instruct via transformers), "template" (offline, deterministic -
lets the pipeline run without any LLM and serves as a control condition).

    python -m genex.explain --backend template --n 100
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from .config import DATASETS, OUT_DIR, DataConfig, ExplainConfig, get_device
from .config import out_root as dataset_out_root, processed_dir as dataset_processed_dir
from .data import load_processed
from .train import load_model

SYSTEM_PROMPT = (
    "You are an experienced mathematics teacher and learning-analytics expert. "
    "You receive the output of a knowledge-tracing model for ONE student and write a "
    "diagnosis a teacher can act on. Use ONLY the concepts and numbers provided; never "
    "invent concepts, scores or events. Model mastery is the model's estimated probability "
    "the student answers the next item on that concept correctly; uncertainty is the spread "
    "of that estimate across generative samples."
)

USER_TEMPLATE = """Student {sid}: {n_int} interactions on {n_sk} concepts. Overall accuracy {acc:.0%}, last 20 items {acc_recent:.0%}.
Latent state descriptor (8-d linear projection of the knowledge state; for reference only): {desc}

Concept table (mastery = model estimate, unc = uncertainty, acc = observed accuracy, recent = accuracy on the last 5 attempts, trend = recent minus earlier accuracy):
{table}

Think step by step before answering:
1. Which concepts does the model judge as mastered (mastery >= {mt})? Do observed accuracy and trend agree?
2. Which concepts are gaps (mastery < {gt})? What in the attempts, accuracy, trend and SHAP evidence supports this? What misconception might explain the pattern, given the concept's content?
3. Which gaps matter most to review first (lowest mastery, worsening trend, prerequisite-like basics first)?

Return ONLY a JSON object:
{{"reasoning": "<your brief step-by-step reasoning>",
 "mastery_summary": {{"text": "<2-3 sentences>", "concepts": ["<concept name>", ...]}},
 "knowledge_gaps": [{{"concept": "<concept name>", "evidence": "<cite the numbers>", "likely_misconception": "<1 sentence>"}}, ...],
 "review_priority": ["<concept name>", ...]}}
Use concept names exactly as written in the table. At most 4 gaps and 5 review items."""

# Grounded mode (default): the KT model, not the LLM, decides which concepts are
# mastered / gaps; the LLM explains the evidence, suggests misconceptions and
# orders the review list. Small open LLMs otherwise tend to judge from raw
# accuracy and contradict the model (see the "free" ablation).
GROUNDED_SUFFIX = """

The knowledge-tracing model has already classified the concepts. You MUST respect it:
- CANDIDATE MASTERED (mastery >= {mt}): {mastered}
- CANDIDATE GAPS (mastery < {gt}): {gaps}
- DEVELOPING (between): {developing}
Rules: "mastery_summary.concepts" may contain ONLY names from CANDIDATE MASTERED.
"knowledge_gaps" may contain ONLY names from CANDIDATE GAPS. "review_priority" may contain
only names from CANDIDATE GAPS or DEVELOPING, ordered by what to review first. If a list is
"none", leave the matching JSON field empty. Your job is to explain the evidence and the
likely misconceptions, not to re-classify."""


# --------------------------------------------------------------------------
# 1-4: knowledge state -> descriptor
# --------------------------------------------------------------------------
def student_batch(g, max_len, max_attempt, device):
    """Last max_len-1 interactions + one dummy step, so the final position's
    state reflects every real interaction."""
    g = g.tail(max_len - 1)
    cols = {c: g[c].to_numpy() for c in ["skill", "correct", "tbin", "attempts", "item", "lag"] if c in g}
    cols = {c: np.append(v, 1 if c == "skill" else 0) for c, v in cols.items()}
    cols["split"] = np.ones(len(g) + 1, dtype=np.int64)
    return {c: torch.as_tensor(v[None], dtype=torch.long, device=device) for c, v in cols.items()}


def concept_stats(g, recent=5):
    rows = []
    for k, s in g.groupby("skill"):
        c = s["correct"].to_numpy()
        early = c[:-recent] if len(c) > recent else c[:0]
        rows.append({"skill": int(k), "attempts": len(c), "acc": c.mean(),
                     "recent": c[-recent:].mean(),
                     "trend": (c[-recent:].mean() - early.mean()) if len(early) else 0.0,
                     "last_pos": int(s["pos"].max())})
    return pd.DataFrame(rows)


def shap_targets(st, ecfg, n=3):
    """The student's weakest practised concepts (gaps first)."""
    cand = st[st.attempts >= ecfg.min_attempts]
    cand = cand if len(cand) else st
    return cand.nsmallest(n, "mastery")


def build_profiles(model, df, meta, users, device, ecfg, shap_users=()):
    out = []
    K, max_len = meta["n_skills"], meta["max_len"]
    groups = dict(tuple(df[df.user.isin(set(users))].groupby("user")))
    shap_users = set(shap_users)
    for u in users:
        g = groups[u].sort_values("pos")
        b = student_batch(g, max_len, meta["max_attempt_bucket"], device)
        counts = np.bincount(g["skill"], minlength=K + 1)[1:]
        att = torch.as_tensor(counts[None], device=device)
        lagk = None
        if "lag" in g:      # practice gap of every concept at the end of the history (GENEX-KT v2)
            last = g.reset_index(drop=True).groupby("skill").apply(lambda s: s.index.max())
            gap = np.zeros(K, dtype=np.int64)
            gap[last.index.to_numpy() - 1] = len(g) - last.to_numpy()
            lag = np.where(gap > 0, np.minimum(1 + np.floor(np.log2(np.maximum(gap, 1))), meta.get("n_lag_bins", 13) - 1), 0)
            lagk = torch.as_tensor(lag[None].astype(np.int64), device=device)
        state, mean, std = model.mastery_profile(b, att, lag_per_skill=lagk)
        st = concept_stats(g)
        st["mastery"] = mean[0, st.skill.to_numpy() - 1].cpu().numpy()
        st["unc"] = std[0, st.skill.to_numpy() - 1].cpu().numpy()
        st["concept"] = st.skill.map(meta["skill_names"])
        prof = {"user": int(u), "state": state[0].cpu().numpy(), "stats": st,
                "mastery_all": mean[0].cpu().numpy(), "unc_all": std[0].cpu().numpy(),
                "n_int": len(g), "acc": g.correct.mean(),
                "acc_recent": g.correct.tail(20).mean()}
        if u in shap_users:
            from .shap_attr import explain_student
            tg = shap_targets(st, ecfg)
            prof["shap"] = explain_student(model, b, tg.skill.tolist(),
                                           counts[tg.skill.to_numpy() - 1].tolist(),
                                           nsamples=ecfg.shap_nsamples,
                                           lag=None if lagk is None else lagk[0, tg.skill.to_numpy() - 1].tolist())
        out.append(prof)
    return out


def fit_projection(profiles, dims=8):
    X = np.stack([p["state"] for p in profiles])
    pca = PCA(n_components=min(dims, X.shape[0], X.shape[1])).fit(X)
    for p in profiles:
        p["descriptor"] = pca.transform(p["state"][None])[0]
    return pca


def prompt_for(p, ecfg, max_rows=15):
    st = p["stats"][p["stats"].attempts >= ecfg.min_attempts].copy()
    if st.empty:
        st = p["stats"].copy()
    # most informative rows: weakest, strongest, most practiced
    pick = pd.concat([st.nsmallest(6, "mastery"), st.nlargest(5, "mastery"),
                      st.nlargest(6, "attempts")]).drop_duplicates("skill").head(max_rows)
    pick = pick.sort_values("mastery")
    lines = [f"- {r.concept}: mastery {r.mastery:.2f} (unc {r.unc:.2f}), attempts {r.attempts}, "
             f"acc {r.acc:.0%}, recent {r.recent:.0%}, trend {r.trend:+.0%}"
             for r in pick.itertuples()]
    if p.get("shap"):
        from .shap_attr import evidence_lines
        names = dict(zip(p["stats"].skill, p["stats"].concept))
        lines += ["", "SHAP evidence (how the student's answers on each concept move the "
                  "model's mastery estimate; negative = pulls it down):"]
        lines += evidence_lines(p["shap"], names)
    desc = ", ".join(f"{v:+.2f}" for v in p["descriptor"])
    user = USER_TEMPLATE.format(sid=p["user"], n_int=p["n_int"], n_sk=len(p["stats"]),
                                acc=p["acc"], acc_recent=p["acc_recent"], desc=desc,
                                table="\n".join(lines), mt=ecfg.mastered_threshold,
                                gt=ecfg.gap_threshold)
    if ecfg.prompt_mode == "grounded":
        def names(mask):
            return ", ".join(pick[mask].concept) or "none"
        user += GROUNDED_SUFFIX.format(
            mt=ecfg.mastered_threshold, gt=ecfg.gap_threshold,
            mastered=names(pick.mastery >= ecfg.mastered_threshold),
            gaps=names(pick.mastery < ecfg.gap_threshold),
            developing=names((pick.mastery >= ecfg.gap_threshold) &
                             (pick.mastery < ecfg.mastered_threshold)))
    return user, pick


# --------------------------------------------------------------------------
# 5: LLM backends
# --------------------------------------------------------------------------
class TemplateBackend:
    """Deterministic, rule-written explanation from the same table the LLM sees."""
    name = "template"

    def __init__(self, ecfg):
        self.e = ecfg

    def __call__(self, system, user, table):
        m = table[table.mastery >= self.e.mastered_threshold].nlargest(4, "mastery")
        gaps = table[table.mastery < self.e.gap_threshold].nsmallest(4, "mastery")
        txt = ("The student shows secure understanding of " + ", ".join(m.concept) + "."
               if len(m) else "No concept has yet reached a secure level of mastery.")
        gl = []
        for r in gaps.itertuples():
            trend = "and is getting worse" if r.trend < -0.1 else (
                "but is improving" if r.trend > 0.1 else "with no clear change")
            if r.acc >= 0.7 or r.attempts < 5:
                why = (f"Limited evidence so far ({r.attempts} attempts); the model is not yet "
                       "confident the skill is secure, so a short check is advisable.")
            elif r.trend < -0.1:
                why = ("Accuracy is falling on recent items, which suggests the procedure is "
                       "being forgotten or confused with a related skill.")
            else:
                why = ("Errors persist after repeated practice, which suggests a procedural "
                       "misunderstanding rather than a slip.")
            gl.append({"concept": r.concept,
                       "evidence": f"model mastery {r.mastery:.2f}; {r.acc:.0%} correct over "
                                   f"{r.attempts} attempts ({r.recent:.0%} on the last 5), {trend}.",
                       "likely_misconception": why})
        return {"reasoning": "rule-based: thresholds on model mastery",
                "mastery_summary": {"text": txt, "concepts": list(m.concept)},
                "knowledge_gaps": gl, "review_priority": list(gaps.concept)}


class OpenAIBackend:
    name = "openai"

    def __init__(self, ecfg):
        from openai import OpenAI
        self.client, self.model = OpenAI(), ecfg.openai_model

    def __call__(self, system, user, table):
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.2, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return parse_json(r.choices[0].message.content)


class ClaudeBackend:
    """Anthropic Claude via the official SDK; reads ANTHROPIC_API_KEY."""
    name = "claude"

    def __init__(self, ecfg):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), ecfg.claude_model

    def __call__(self, system, user, table):
        r = self.client.messages.create(
            model=self.model, max_tokens=1500, temperature=0.2, system=system,
            messages=[{"role": "user", "content": user}])
        return parse_json("".join(b.text for b in r.content if b.type == "text"))


class HFBackend:
    """Local open-weights LLM through Hugging Face transformers (4-bit on small GPUs).
    Generates in batches: one prompt at a time leaves most of the GPU idle."""
    name = "hf"
    model_id = None          # subclasses fix the model; "hf" reads GENEX_HF_MODEL
    batch_size = 8

    def __init__(self, ecfg):
        from transformers import pipeline
        model = self.model_id or ecfg.hf_model
        kw = {}
        if ecfg.hf_4bit:      # fits an 8B model in ~6 GB of GPU memory (Colab T4, RTX 3060)
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4")
            dtype = torch.float16
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        import inspect
        # transformers >= 4.56 calls it `dtype`; older versions `torch_dtype`
        dkey = "dtype" if "dtype" in inspect.signature(pipeline).parameters else "torch_dtype"
        self.pipe = pipeline("text-generation", model=model, device_map="auto",
                             model_kwargs=kw, **{dkey: dtype})
        tok = self.pipe.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"                      # required for batched decoding
        self.model_name = model

    @staticmethod
    def _msgs(system, user):
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def generate(self, pairs, max_new_tokens=900):
        """pairs: list of (system, user). Returns raw text answers, in order."""
        out = self.pipe([self._msgs(s, u) for s, u in pairs], batch_size=self.batch_size,
                        max_new_tokens=max_new_tokens, do_sample=False)
        return [o[0]["generated_text"][-1]["content"] for o in out]

    def batch(self, pairs):
        return [parse_json(t) for t in self.generate(pairs)]

    def __call__(self, system, user, table):
        return self.batch([(system, user)])[0]

    def close(self):
        del self.pipe
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Llama3Backend(HFBackend):
    """Meta-Llama-3-8B-Instruct - the local model named in the proposal (gated: needs
    accepted licence + HF_TOKEN)."""
    name, model_id = "llama3", "meta-llama/Meta-Llama-3-8B-Instruct"


class Qwen25Backend(HFBackend):
    """Qwen2.5-7B-Instruct (Apache-2.0) - second open model, replacing GPT-4o."""
    name, model_id = "qwen25", "Qwen/Qwen2.5-7B-Instruct"


class Phi35Backend(HFBackend):
    """Phi-3.5-mini-instruct (MIT) - fallback if LLaMA 3 access is not granted."""
    name, model_id = "phi35", "microsoft/Phi-3.5-mini-instruct"


BACKENDS = {"template": TemplateBackend, "openai": OpenAIBackend, "claude": ClaudeBackend,
            "hf": HFBackend, "llama3": Llama3Backend, "qwen25": Qwen25Backend,
            "phi35": Phi35Backend}
OPEN_LLMS = ["llama3", "qwen25", "phi35"]


def _normalise(d):
    """Coerce a parsed LLM answer to the expected schema (small models sometimes return
    plain strings where objects are expected); anything unusable becomes a parse error."""
    if not isinstance(d, dict):
        return {"parse_error": True, "raw": str(d)}
    ms = d.get("mastery_summary", {})
    if isinstance(ms, str):
        ms = {"text": ms, "concepts": []}
    if not isinstance(ms, dict):
        ms = {"text": "", "concepts": []}
    ms["concepts"] = [c for c in (ms.get("concepts") or []) if isinstance(c, str)]
    d["mastery_summary"] = ms
    gaps = []
    for g in d.get("knowledge_gaps") or []:
        if isinstance(g, str):
            gaps.append({"concept": g, "evidence": "", "likely_misconception": ""})
        elif isinstance(g, dict):
            gaps.append({"concept": str(g.get("concept", "")), "evidence": str(g.get("evidence", "")),
                         "likely_misconception": str(g.get("likely_misconception", ""))})
    d["knowledge_gaps"] = gaps
    rp = d.get("review_priority") or []
    d["review_priority"] = [c if isinstance(c, str) else str(c.get("concept", "")) if isinstance(c, dict)
                            else str(c) for c in (rp if isinstance(rp, list) else [rp])]
    return d


def parse_json(text):
    try:
        return _normalise(json.loads(text))
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return _normalise(json.loads(m.group(0)))
            except Exception:
                pass
    return {"parse_error": True, "raw": text}


# --------------------------------------------------------------------------
# 6: faithfulness
# --------------------------------------------------------------------------
def faithfulness(expl, table, ecfg):
    """Every concept the explanation names must exist in the model output and
    its claim must agree with the model's mastery estimate."""
    if expl.get("parse_error"):
        return {"pass": False, "claims": 0, "faithful_claims": 0, "issues": ["unparseable output"]}
    mast = dict(zip(table.concept, table.mastery))
    issues, claims, ok = [], 0, 0

    def check(name, cond, why):
        nonlocal claims, ok
        claims += 1
        if name not in mast:
            issues.append(f"unknown concept '{name}'")
        elif not cond(mast[name]):
            issues.append(f"'{name}' {why} (mastery {mast[name]:.2f})")
        else:
            ok += 1

    for c in expl.get("mastery_summary", {}).get("concepts", []):
        check(c, lambda v: v >= ecfg.mastered_threshold, "called mastered")
    for g in expl.get("knowledge_gaps", []):
        check(g.get("concept", ""), lambda v: v < ecfg.gap_threshold, "called a gap")
    for c in expl.get("review_priority", []):
        check(c, lambda v: v < ecfg.mastered_threshold, "recommended for review")
    return {"pass": not issues, "claims": claims, "faithful_claims": ok,
            "issues": issues if claims else ["no concept-level claims"]}


def render_text(expl):
    if expl.get("parse_error"):
        return expl.get("raw", "")
    parts = ["Demonstrated mastery: " + expl.get("mastery_summary", {}).get("text", "")]
    gaps = expl.get("knowledge_gaps", [])
    if gaps:
        parts.append("Knowledge gaps:\n" + "\n".join(
            f"  - {g.get('concept')}: {g.get('evidence')} Likely misconception: "
            f"{g.get('likely_misconception')}" for g in gaps))
    rp = expl.get("review_priority", [])
    if rp:
        parts.append("Review first: " + " -> ".join(rp))
    return "\n".join(parts)


def validate_against_future(records, df_test, ecfg):
    """Do concepts called gaps really go worse in the student's test period?"""
    gap_acc, mast_acc = [], []
    for r in records:
        t = df_test[df_test.user == r["user"]]
        acc = t.groupby("concept")["correct"].mean()
        for g in r["explanation"].get("knowledge_gaps", []):
            if g.get("concept") in acc:
                gap_acc.append(acc[g["concept"]])
        for c in r["explanation"].get("mastery_summary", {}).get("concepts", []):
            if c in acc:
                mast_acc.append(acc[c])
    return {"future_acc_on_flagged_gaps": float(np.mean(gap_acc)) if gap_acc else None,
            "n_gap_concepts_seen_in_test": len(gap_acc),
            "future_acc_on_flagged_mastered": float(np.mean(mast_acc)) if mast_acc else None,
            "n_mastered_concepts_seen_in_test": len(mast_acc)}


def run(backend="template", n=100, processed_dir=None, out_root=None, seed=42,
        cutoff="pre_test", prompt_mode="grounded", log=print, use_shap=True):
    """cutoff="pre_test" builds every diagnosis from the student's train+val
    history only, so it can be checked against their unseen test period."""
    ecfg = ExplainConfig(backend=backend, n_students=n, prompt_mode=prompt_mode,
                         use_shap=use_shap)
    out_root = out_root or OUT_DIR
    # file tag: "<backend>" for the grounded default, "<backend>_free" for the ablation
    tag = backend if prompt_mode == "grounded" else f"{backend}_{prompt_mode}"
    processed_dir = processed_dir or DataConfig().processed_dir
    _, meta = load_processed(processed_dir)
    df_all = pd.read_csv(os.path.join(processed_dir, "interactions.csv.gz"))
    df = df_all[df_all.split < 3] if cutoff == "pre_test" else df_all
    device = get_device()
    model = load_model(os.path.join(out_root, "genex"), meta, device)

    users = np.sort(df.user.unique())
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(users, min(n, len(users)), replace=False).tolist())
    # projection is fit on the chosen students plus a random reference sample
    ref = set(rng.choice(users, min(500, len(users)), replace=False).tolist())
    out_dir = os.path.join(out_root, "explanations")
    os.makedirs(out_dir, exist_ok=True)
    # SHAP is computed once per trained model and shared by all explainer backends
    shap_path = os.path.join(out_dir, "shap.json")
    model_mtime = os.path.getmtime(os.path.join(out_root, "genex", "model.pt"))
    cache = json.load(open(shap_path)) if os.path.exists(shap_path) else {}
    reuse = ecfg.use_shap and cache.get("model_mtime") == model_mtime
    shap_users = () if (not ecfg.use_shap or reuse) else chosen
    if shap_users:
        log(f"[explain] computing SHAP attributions for {len(chosen)} students")
    profiles_all = build_profiles(model, df, meta, sorted(chosen | ref), device, ecfg,
                                  shap_users=shap_users)
    pca = fit_projection(profiles_all)
    profiles = [p for p in profiles_all if p["user"] in chosen]
    if reuse:
        for p in profiles:
            p["shap"] = {int(k): {**v, "phi": {int(c): x for c, x in v["phi"].items()}}
                         for k, v in cache["students"].get(str(p["user"]), {}).items()}
    elif ecfg.use_shap:
        with open(shap_path, "w") as fh:
            json.dump({"model_mtime": model_mtime,
                       "students": {str(p["user"]): p.get("shap", {}) for p in profiles}}, fh)

    llm = BACKENDS[backend](ecfg)
    np.save(os.path.join(out_dir, "pca_components.npy"), pca.components_)
    prompts = [prompt_for(p, ecfg) for p in profiles]
    t0 = time.time()
    if hasattr(llm, "batch"):                       # open LLMs: batched generation
        expls = llm.batch([(SYSTEM_PROMPT, u) for u, _ in prompts])
    else:
        expls = [llm(SYSTEM_PROMPT, u, t) for u, t in prompts]
    gen_seconds = time.time() - t0
    if hasattr(llm, "close"):
        llm.close()
    records = []
    for i, (p, (user, table), expl) in enumerate(zip(profiles, prompts, expls)):
        f = faithfulness(expl, table, ecfg)
        records.append({"user": p["user"], "backend": tag, "prompt_mode": prompt_mode,
                        "cutoff": cutoff, "prompt": user,
                        "explanation": expl, "text": render_text(expl), "faithfulness": f,
                        "stats": p["stats"].to_dict("records"),
                        "shap": p.get("shap", {}),
                        "n_int": int(p["n_int"]), "acc": float(p["acc"]),
                        "mastery": {int(s): float(v) for s, v in
                                    zip(range(1, meta["n_skills"] + 1), p["mastery_all"])},
                        "uncertainty": {int(s): float(v) for s, v in
                                        zip(range(1, meta["n_skills"] + 1), p["unc_all"])}})
    with open(os.path.join(out_dir, f"explanations_{tag}.jsonl"), "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    rate = float(np.mean([r["faithfulness"]["pass"] for r in records]))
    claim = sum(r["faithfulness"]["faithful_claims"] for r in records) / max(
        1, sum(r["faithfulness"]["claims"] for r in records))
    summary = {"backend": tag, "prompt_mode": prompt_mode, "n": len(records), "cutoff": cutoff,
               "faithfulness_pass_rate": rate, "claim_level_faithfulness": claim,
               "n_without_claims": sum(r["faithfulness"]["claims"] == 0 for r in records),
               "n_parse_errors": sum(bool(r["explanation"].get("parse_error")) for r in records),
               "llm": getattr(llm, "model_name", backend),
               "generation_seconds_per_student": round(gen_seconds / max(1, len(records)), 2)}
    if cutoff == "pre_test":
        t = df_all[df_all.split == 3].copy()
        t["concept"] = t.skill.map(meta["skill_names"])
        summary.update(validate_against_future(records, t, ecfg))
    with open(os.path.join(out_dir, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[explain] {tag}: faithfulness pass rate {rate:.1%} (claims {claim:.1%})")
    return records, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="template", choices=list(BACKENDS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--processed_dir")
    ap.add_argument("--out_root")
    ap.add_argument("--cutoff", default="pre_test", choices=["pre_test", "all"])
    ap.add_argument("--prompt", default="grounded", choices=["grounded", "free"],
                    help="grounded: model decides gaps/mastery, LLM explains (default); "
                         "free: LLM classifies on its own (ablation)")
    ap.add_argument("--no_shap", action="store_true")
    a = ap.parse_args()
    pd_ = a.processed_dir or (dataset_processed_dir(a.dataset) if a.dataset else None)
    or_ = a.out_root or (dataset_out_root(a.dataset) if a.dataset else None)
    run(a.backend, a.n, pd_, or_, cutoff=a.cutoff, prompt_mode=a.prompt, use_shap=not a.no_shap)
