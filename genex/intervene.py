"""Phase 4 - Intervention generation.

Inputs per student: the LLM diagnosis (Phase 3) and the backbone's
predicted mastery for every concept. Outputs:
  * a learning pathway of 5 concepts, ranked by low predicted mastery and
    ordered by a prerequisite graph mined from TRAIN data;
  * 3 practice-exercise descriptors (type, difficulty, target misconception);
  * a formative feedback statement for the learner.

A rule-based baseline (fixed accuracy threshold, no model, no prerequisites)
is produced for the same students. Both are (a) validated algorithmically
against each student's unseen test period and (b) written to a blinded
A/B sheet for the expert panel.

    python -m genex.intervene --backend template
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import DATASETS, OUT_DIR, DataConfig, ExplainConfig
from .config import out_root as dataset_out_root, processed_dir as dataset_processed_dir
from .data import load_processed


# --------------------------------------------------------------------------
# prerequisite graph
# --------------------------------------------------------------------------
def mine_prerequisites(df, min_support=30, min_precedence=0.75, min_lift=0.10,
                       mastery_acc=0.6, max_parents=3, max_users=5000, seed=0):
    """Edge A -> B when (i) students who meet both usually meet A first and
    (ii) students who did well on A before starting B do better on their
    first attempts at B than students who did not. Train rows only."""
    tr = df[df.split == 1]
    users = tr.user.unique()
    if len(users) > max_users:                      # EdNet: mine on a random sample
        keep = np.random.default_rng(seed).choice(users, max_users, replace=False)
        tr = tr[tr.user.isin(set(keep))]
    tr = tr.sort_values(["user", "pos"])
    sup, prec = defaultdict(int), defaultdict(int)
    hi, lo = defaultdict(list), defaultdict(list)
    for u, g in tr.groupby("user"):
        sk, c = g.skill.to_numpy(), g.correct.to_numpy()
        firsts = {}
        for i, s in enumerate(sk):
            firsts.setdefault(s, i)
        for b, fb in firsts.items():
            early_b = c[sk == b][:3].mean()
            prefix_s, prefix_c = sk[:fb], c[:fb]
            for a, fa in firsts.items():
                if a == b:
                    continue
                sup[(a, b)] += 1
                if fa < fb:
                    prec[(a, b)] += 1
                    acc_a = prefix_c[prefix_s == a].mean()
                    (hi if acc_a >= mastery_acc else lo)[(a, b)].append(early_b)
    edges = []
    for (a, b), n in sup.items():
        if n < min_support or prec[(a, b)] / n < min_precedence:
            continue
        if len(hi[(a, b)]) < 5 or len(lo[(a, b)]) < 5:
            continue
        lift = np.mean(hi[(a, b)]) - np.mean(lo[(a, b)])
        if lift >= min_lift:
            edges.append((int(a), int(b), prec[(a, b)] / n, float(lift), n))
    # keep only the strongest few prerequisites of each concept (by lift)
    by_post = defaultdict(list)
    for e in edges:
        by_post[e[1]].append(e)
    edges = [e for v in by_post.values() for e in sorted(v, key=lambda e: -e[3])[:max_parents]]
    edges.sort(key=lambda e: (-e[2], -e[3]))
    # keep a DAG: add strongest edges first, skip any that would close a cycle
    adj = defaultdict(set)

    def reaches(x, y):
        stack, seen = [x], set()
        while stack:
            n = stack.pop()
            if n == y:
                return True
            if n not in seen:
                seen.add(n)
                stack.extend(adj[n])
        return False

    kept = []
    for a, b, p, l, n in edges:
        if not reaches(b, a):
            adj[a].add(b)
            kept.append({"pre": a, "post": b, "precedence": p, "lift": l, "support": n})
    return pd.DataFrame(kept)


def topo_order(nodes, graph, key):
    """Kahn's algorithm restricted to `nodes`; ties broken by `key`."""
    nodes = list(nodes)
    sub = graph[graph.pre.isin(nodes) & graph.post.isin(nodes)] if len(graph) else graph
    indeg = {n: 0 for n in nodes}
    out = defaultdict(list)
    for r in sub.itertuples():
        indeg[r.post] += 1
        out[r.pre].append(r.post)
    order, ready = [], sorted([n for n in nodes if indeg[n] == 0], key=key)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in out[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
        ready.sort(key=key)
    return order


# --------------------------------------------------------------------------
# GENEX-KT plan
# --------------------------------------------------------------------------
def difficulty_band(m):
    return "foundational" if m < 0.4 else "intermediate" if m < 0.65 else "stretch"


def genex_plan(rec, graph, names, answer_types, ecfg, k=5):
    mastery = {int(s): v for s, v in rec["mastery"].items()}
    stats = pd.DataFrame(rec["stats"]).set_index("skill")
    # all practised concepts, including just-started ones: those are often the
    # student's current assignment (requiring >=3 attempts lowered future-error coverage)
    practiced = [s for s in stats.index if mastery[s] < ecfg.mastered_threshold]
    picked = sorted(practiced, key=lambda s: mastery[s])[:k]
    reason = {}
    # pull in weak prerequisites of picked concepts, ahead of the weakest-ranked picks
    if len(graph):
        for s in list(picked):
            for pre in graph[graph.post == s].pre:
                pre = int(pre)
                if pre not in picked and mastery.get(pre, 1) < ecfg.gap_threshold:
                    picked.insert(0, pre)
                    reason.setdefault(pre, f"Weak prerequisite of {names[s]}"
                                      + ("" if pre in stats.index else " - not yet practised") + ".")
        picked = sorted(dict.fromkeys(picked), key=lambda s: mastery[s])[:k]
    # fill with new concepts whose prerequisites are mastered
    if len(picked) < k and len(graph):
        ready = []
        for s in set(graph.post) - set(stats.index):
            pres = graph[graph.post == s].pre
            if all(mastery.get(p, 0) >= ecfg.mastered_threshold for p in pres):
                ready.append(int(s))
        picked += sorted(ready, key=lambda s: mastery[s])[:k - len(picked)]
    path = topo_order(picked, graph, key=lambda s: mastery[s])

    gaps = {g.get("concept"): g for g in rec["explanation"].get("knowledge_gaps", [])}
    exercises = []
    for s in path[:3]:
        name = names[s]
        st = stats.loc[s] if s in stats.index else None
        mis = gaps.get(name, {}).get("likely_misconception") or reason.get(s) or (
            f"Accuracy {st['acc']:.0%} over {int(st['attempts'])} attempts"
            f" (recent {st['recent']:.0%})." if st is not None
            else "New concept - its prerequisites are mastered.")
        exercises.append({"concept": name, "exercise_type": answer_types.get(s, "mixed"),
                          "difficulty": difficulty_band(mastery[s]), "n_items": 5,
                          "target_misconception": mis})
    strengths = rec["explanation"].get("mastery_summary", {}).get("concepts", [])[:2]
    first = names[path[0]] if path else None
    fb = ("You're doing well on " + " and ".join(strengths) + ". " if strengths else
          "You've put in steady practice. ")
    if first:
        fb += (f"Next, focus on {first}: work through a few {exercises[0]['difficulty']} "
               f"problems slowly and check each step before you submit.")
    return {"pathway": [names[s] for s in path], "pathway_ids": path,
            "pathway_mastery": [round(mastery[s], 3) for s in path],
            "exercises": exercises, "feedback": fb}


def rule_baseline(rec, names, k=5, threshold=0.6, min_attempts=3):
    """Fixed mastery threshold on raw accuracy - no model, no prerequisites."""
    st = pd.DataFrame(rec["stats"])
    weak = st[(st.attempts >= min_attempts) & (st.acc < threshold)].nsmallest(k, "acc")
    path = [int(s) for s in weak.skill]
    ex = [{"concept": names[s], "exercise_type": "mixed", "difficulty": "intermediate",
           "n_items": 5, "target_misconception": "Below the 60% accuracy threshold."}
          for s in path[:3]]
    fb = ("Keep practising the topics below 60% accuracy." if path else
          "All practised topics are above 60% accuracy. Keep going!")
    return {"pathway": [names[s] for s in path], "pathway_ids": path, "exercises": ex,
            "feedback": fb}


FEEDBACK_SYSTEM = "You are a supportive tutor writing short feedback to a learner."


def feedback_prompt(rec, plan):
    return ("Write a 2-3 sentence formative feedback message addressed to the student. "
            "Be encouraging and specific. Use only this information.\n"
            f"Diagnosis: {rec['text']}\nPlanned next concepts in order: "
            f"{', '.join(plan['pathway'])}\n"
            'Return JSON: {"feedback": "<message>"}')


def llm_feedback_all(backend, recs, plans):
    """LLM-written feedback for every plan (batched for open LLMs); falls back to
    the rule-written statement when the model returns no usable JSON."""
    if backend.name == "template":
        return [p["feedback"] for p in plans]
    pairs = [(FEEDBACK_SYSTEM, feedback_prompt(r, p)) for r, p in zip(recs, plans)]
    outs = backend.batch(pairs) if hasattr(backend, "batch") else \
        [backend(s, u, None) for s, u in pairs]
    return [o.get("feedback") or p["feedback"] for o, p in zip(outs, plans)]


# --------------------------------------------------------------------------
# algorithmic validation against the unseen test period
# --------------------------------------------------------------------------
def validate(plans, df_test, graph):
    rows = []
    edges = set(zip(graph.pre, graph.post)) if len(graph) else set()
    for user, p in plans.items():
        t = df_test[df_test.user == user]
        if t.empty:
            continue
        acc = t.groupby("skill").correct.mean()
        struggled = set(acc[acc < 0.6].index)
        errors = t[t.correct == 0]
        for kind in ("genex", "baseline"):
            path = p[kind]["pathway_ids"]
            seen = [s for s in path if s in acc.index]
            pos = {s: i for i, s in enumerate(path)}
            pairs = [(a, b) for a, b in edges if a in pos and b in pos]
            rows.append({
                "user": user, "plan": kind, "len": len(path),
                "error_coverage": errors.skill.isin(path).mean() if len(errors) else np.nan,
                "struggle_recall": (len(struggled & set(path)) / len(struggled)
                                    if struggled else np.nan),
                "precision_seen": (np.mean([acc[s] < 0.6 for s in seen]) if seen else np.nan),
                "prereq_order_ok": (np.mean([pos[a] < pos[b] for a, b in pairs])
                                    if pairs else np.nan)})
    r = pd.DataFrame(rows)
    return r, r.groupby("plan")[["len", "error_coverage", "struggle_recall", "precision_seen",
                                 "prereq_order_ok"]].mean()


def plan_text(p):
    lines = ["Learning pathway: " + " -> ".join(p["pathway"]) if p["pathway"] else
             "Learning pathway: (none)"]
    for i, e in enumerate(p["exercises"], 1):
        lines.append(f"Exercise {i}: {e['n_items']} {e['difficulty']} {e['exercise_type']} items "
                     f"on {e['concept']}. Target: {e['target_misconception']}")
    lines.append("Feedback to student: " + p["feedback"])
    return "\n".join(lines)


def run(backend="template", processed_dir=None, out_root=None, seed=42, log=print):
    out_root = out_root or OUT_DIR
    from .explain import BACKENDS
    ecfg = ExplainConfig(backend=backend)
    processed_dir = processed_dir or DataConfig().processed_dir
    _, meta = load_processed(processed_dir)
    names = meta["skill_names"]
    df = pd.read_csv(os.path.join(processed_dir, "interactions.csv.gz"))
    answer_types = df.groupby("skill").answer_type.agg(lambda s: s.mode().iat[0]).to_dict()

    out_dir = os.path.join(out_root, "interventions")
    os.makedirs(out_dir, exist_ok=True)
    graph = mine_prerequisites(df)
    g = graph.copy()
    if len(g):
        g["pre_name"], g["post_name"] = g.pre.map(names), g.post.map(names)
    g.to_csv(os.path.join(out_dir, "prerequisite_graph.csv"), index=False)
    log(f"[intervene] prerequisite graph: {len(graph)} edges")

    path = os.path.join(out_root, "explanations", f"explanations_{backend}.jsonl")
    with open(path) as fh:
        recs = [json.loads(l) for l in fh]
    base = backend.split("_")[0]                    # e.g. hf_free -> hf for the feedback text
    llm = BACKENDS[base](ecfg)
    gps = [genex_plan(r, graph, names, answer_types, ecfg) for r in recs]
    for gp, fb in zip(gps, llm_feedback_all(llm, recs, gps)):
        gp["feedback"] = fb
    if hasattr(llm, "close"):
        llm.close()
    plans = {r["user"]: {"genex": gp, "baseline": rule_baseline(r, names)}
             for r, gp in zip(recs, gps)}
    with open(os.path.join(out_dir, f"plans_{backend}.json"), "w") as fh:
        json.dump({str(k): v for k, v in plans.items()}, fh, indent=2)

    per_user, summary = validate(plans, df[df.split == 3], graph)
    per_user.to_csv(os.path.join(out_dir, f"validation_per_student_{backend}.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, f"validation_summary_{backend}.csv"))
    log("[intervene] validation on unseen test period:\n" + summary.round(3).to_string())

    # blinded A/B sheet for the expert panel
    rng = np.random.default_rng(seed)
    sheet, key = [], []
    rec_by_user = {r["user"]: r for r in recs}
    for user, p in plans.items():
        flip = bool(rng.integers(2))
        a, b = (p["baseline"], p["genex"]) if flip else (p["genex"], p["baseline"])
        sheet.append({"case_id": user, "student_diagnosis": rec_by_user[user]["text"],
                      "plan_A": plan_text(a), "plan_B": plan_text(b),
                      "A_relevance_1to5": "", "A_appropriateness_1to5": "",
                      "B_relevance_1to5": "", "B_appropriateness_1to5": "",
                      "preferred_A_or_B": "", "comments": ""})
        key.append({"case_id": user, "A": "baseline" if flip else "genex",
                    "B": "genex" if flip else "baseline"})
    pd.DataFrame(sheet).to_csv(os.path.join(out_dir, f"expert_sheet_interventions_{backend}.csv"), index=False)
    pd.DataFrame(key).to_csv(os.path.join(out_dir, f"expert_sheet_interventions_{backend}_KEY.csv"), index=False)
    return plans, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="template", choices=["template", "openai", "claude", "hf", "llama3", "qwen25",
                                                       "phi35", "llama3_free", "qwen25_free",
                                                       "phi35_free", "hf_free", "claude_free",
                                                       "template_free"])
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--processed_dir")
    ap.add_argument("--out_root")
    a = ap.parse_args()
    run(a.backend, a.processed_dir or (dataset_processed_dir(a.dataset) if a.dataset else None),
        a.out_root or (dataset_out_root(a.dataset) if a.dataset else None))
