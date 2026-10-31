"""Evaluation protocol (Objective O3).

    python -m genex.evaluate report --dataset assist2009   # one dataset
    python -m genex.evaluate cross                         # all three datasets together
    python -m genex.evaluate ratings --explanations r1.csv r2.csv r3.csv
    python -m genex.evaluate ratings --interventions r1.csv r2.csv r3.csv

* Predictive: AUC-ROC, accuracy, BCE; DeLong's test (paired, p < 0.05) for
  GENEX-KT vs every baseline on identical test positions.
* Explanations: faithfulness pass rate (automatic) + expert Likert sheet
  (factual accuracy, pedagogical coherence, specificity, actionability);
  median and IQR per criterion, Krippendorff's alpha (ordinal).
* Interventions: algorithmic validation (from intervene.py) + blinded A/B
  expert sheet; Wilcoxon signed-rank test and preference rate.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from .config import DATASETS, OUT_BASE, OUT_DIR
from .config import out_root as dataset_out_root

MODEL_DIRS = ["bkt", "dkt", "sakt", "akt", "genex", "genex_rasch", "genex_v2"]
# open LLM explainers rated by the expert panel (proposal: 100 explanations per dataset)
MAIN_LLMS = ["llama3", "qwen25", "phi35"]
EXPL_CRITERIA = ["factual_accuracy", "pedagogical_coherence", "specificity", "actionability"]


# --------------------------------------------------------------------------
# DeLong's test (fast implementation, Sun & Xu 2014)
# --------------------------------------------------------------------------
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y, p1, p2):
    """Two-sided p-value for H0: AUC(p1) == AUC(p2) on the same labels."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([p1, p2])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def report(out_root=None, log=print):
    out_root = out_root or OUT_DIR
    rows, preds = [], {}
    for d in MODEL_DIRS:
        mpath = os.path.join(out_root, d, "metrics.json")
        if not os.path.exists(mpath):
            continue
        m = json.load(open(mpath))
        rows.append({"model": m["model"], "test_auc": m["test"]["auc"], "test_acc": m["test"]["acc"],
                     "test_bce": m["test"]["bce"], "val_auc": m["val"]["auc"],
                     "n_test": m["test"]["n"], "train_s": m["train_seconds"]})
        if "test_mc" in m:
            rows.append({"model": m["model"] + " (MC mean)", "test_auc": m["test_mc"]["auc"],
                         "test_acc": m["test_mc"]["acc"], "test_bce": m["test_mc"]["bce"],
                         "val_auc": m["val_mc"]["auc"], "n_test": m["test_mc"]["n"],
                         "train_s": m["train_seconds"]})
        p = pd.read_csv(os.path.join(out_root, d, "predictions.csv.gz"))
        preds[m["model"]] = p[p.split == 3].set_index(["user", "pos"])
    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(out_root, "report"), exist_ok=True)
    table.to_csv(os.path.join(out_root, "report", "predictive_results.csv"), index=False)
    log("\n" + table.round(4).to_string(index=False))

    tests = []
    mains = ["GENEX-KT", "GENEX-KT + Rasch", "GENEX-KT v2"]
    for mi, main in enumerate(mains):
        if main not in preds:
            continue
        g = preds[main]
        for name, p in preds.items():
            if name == main or name in mains[:mi]:      # each pair of GENEX variants tested once
                continue
            j = g[["correct", "p"]].join(p[["p"]], rsuffix="_b", how="inner")
            a1, a2, z, pv = delong_test(j.correct.to_numpy(), j.p.to_numpy(), j.p_b.to_numpy())
            tests.append({"comparison": f"{main} vs {name}", "auc_genex": a1, "auc_baseline": a2,
                          "delta": a1 - a2, "z": z, "p_value": pv, "significant_p<0.05": pv < 0.05,
                          "n": len(j)})
    if tests:
        tests = pd.DataFrame(tests)
        tests.to_csv(os.path.join(out_root, "report", "delong_tests.csv"), index=False)
        log("\n" + tests.round(4).to_string(index=False))

    rep = os.path.join(out_root, "report")
    # explanation and intervention summaries across explainer backends
    ex = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(
        out_root, "explanations", "summary_*.json")))]
    if ex:
        pd.DataFrame(ex).to_csv(os.path.join(rep, "explanation_summary.csv"), index=False)
        log("\n" + pd.DataFrame(ex)[["backend", "faithfulness_pass_rate",
                                     "claim_level_faithfulness", "future_acc_on_flagged_gaps",
                                     "future_acc_on_flagged_mastered"]].round(3).to_string(index=False))
    iv = []
    for f in sorted(glob.glob(os.path.join(out_root, "interventions", "validation_summary_*.csv"))):
        d = pd.read_csv(f)
        d.insert(0, "explainer", os.path.basename(f)[len("validation_summary_"):-4])
        iv.append(d)
    if iv:
        pd.concat(iv).to_csv(os.path.join(rep, "intervention_summary.csv"), index=False)
    shap_summary(out_root, log)
    expert_sheets(out_root)
    return table, tests


def expert_sheets(out_root, n=100, seed=0):
    """Blinded sheet of n explanations per dataset, split evenly over the LLM
    explainers in MAIN_LLMS, with anonymous case ids; the key is a separate file."""
    rep = os.path.join(out_root, "report")
    recs = {}
    for b in MAIN_LLMS:
        f = os.path.join(out_root, "explanations", f"explanations_{b}.jsonl")
        if os.path.exists(f):
            recs[b] = [json.loads(l) for l in open(f)]
    if not recs:
        return
    rng = np.random.default_rng(seed)
    per = n // len(recs)
    pool = []
    for b, rs in recs.items():
        idx = rng.choice(len(rs), min(per, len(rs)), replace=False)
        pool += [(b, rs[i]) for i in idx]
    rng.shuffle(pool)
    sheet, key = [], []
    for i, (b, r) in enumerate(pool, 1):
        cid = f"E{i:03d}"
        sheet.append({"case_id": cid, "explanation": r["text"],
                      **{c + "_1to5": "" for c in EXPL_CRITERIA}, "comments": ""})
        key.append({"case_id": cid, "explainer": b, "user": r["user"],
                    "faithful": r["faithfulness"]["pass"]})
    pd.DataFrame(sheet).to_csv(os.path.join(rep, "expert_sheet_explanations.csv"), index=False)
    pd.DataFrame(key).to_csv(os.path.join(rep, "expert_sheet_explanations_KEY.csv"), index=False)


def shap_summary(out_root, log=print):
    """How the backbone's predictions are attributed: own-concept history vs.
    mined prerequisites vs. other concepts (share of total |SHAP|)."""
    sp = os.path.join(out_root, "explanations", "shap.json")
    if not os.path.exists(sp):
        return None
    students = json.load(open(sp))["students"]
    gp = os.path.join(out_root, "interventions", "prerequisite_graph.csv")
    g = pd.read_csv(gp) if os.path.exists(gp) else pd.DataFrame(columns=["pre", "post"])
    pre = g.groupby("post")["pre"].apply(set).to_dict() if len(g) else {}
    rows = []
    for u, targets in students.items():
        for t, r in targets.items():
            t = int(t)
            phi = {int(c): v for c, v in r["phi"].items()}
            tot = sum(abs(v) for v in phi.values()) or np.nan
            own = phi.get(t, 0.0)
            rows.append({"user": u, "target": t, "value": r["value"],
                         "own_share": abs(own) / tot,
                         "prereq_share": sum(abs(phi[c]) for c in pre.get(t, ()) if c in phi) / tot,
                         "own_sign_negative": own < 0, "total_abs": tot})
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    out = {"n_attributions": int(len(d)),
           "mean_own_concept_share": float(d.own_share.mean()),
           "mean_prerequisite_share": float(d.prereq_share.mean()),
           "mean_other_share": float(1 - d.own_share.mean() - d.prereq_share.mean()),
           "own_history_lowers_weak_concepts": float(d.own_sign_negative.mean()),
           "mean_total_abs_attribution": float(d.total_abs.mean())}
    d.to_csv(os.path.join(out_root, "report", "shap_attributions.csv"), index=False)
    with open(os.path.join(out_root, "report", "shap_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("\n[SHAP] " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                 for k, v in out.items()))
    return out


# --------------------------------------------------------------------------
# cross-dataset report (proposal Phase 5 / Objective O3)
# --------------------------------------------------------------------------
def kendall_w(ranks):
    """ranks: [m datasets, n models]. Kendall's coefficient of concordance."""
    m, n = ranks.shape
    R = ranks.sum(0)
    S = ((R - R.mean()) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def cross(out_base=None, datasets=DATASETS, log=print):
    out_base = out_base or OUT_BASE
    done = [d for d in datasets if os.path.exists(os.path.join(
        out_base, d, "report", "predictive_results.csv"))]
    if not done:
        log("no per-dataset reports found")
        return None
    dest = os.path.join(out_base, "cross_dataset")
    os.makedirs(dest, exist_ok=True)
    pr = []
    for d in done:
        t = pd.read_csv(os.path.join(out_base, d, "report", "predictive_results.csv"))
        t["dataset"] = d
        dl = os.path.join(out_base, d, "report", "delong_tests.csv")
        if os.path.exists(dl):
            sig = pd.read_csv(dl)
            sig["model"] = sig.comparison.str.replace("GENEX-KT vs ", "", regex=False)
            t = t.merge(sig[["model", "p_value"]], on="model", how="left")
        pr.append(t)
    pr = pd.concat(pr)
    pr.to_csv(os.path.join(dest, "predictive_all.csv"), index=False)
    main = pr[~pr.model.str.contains("MC mean")]
    auc = main.pivot_table(index="model", columns="dataset", values="test_auc")
    auc.to_csv(os.path.join(dest, "auc_by_dataset.csv"))
    summary = {"datasets": done}
    full = auc.dropna()
    if full.shape[1] >= 2 and full.shape[0] >= 3:
        ranks = full.rank(ascending=False).T.to_numpy()
        summary["kendall_w_model_ranking"] = kendall_w(ranks)
    if "GENEX-KT" in auc.index:
        base = auc.drop(index="GENEX-KT")
        delta = auc.loc["GENEX-KT"] - base.max()
        summary["genex_minus_best_baseline_auc"] = {k: float(v) for k, v in delta.items()}
        summary["genex_rank"] = {k: int(auc[k].rank(ascending=False)["GENEX-KT"]) for k in auc}
        summary["genex_delta_mean"] = float(delta.mean())
        summary["genex_delta_sd"] = float(delta.std()) if len(delta) > 1 else 0.0
    ex = []
    for d in done:
        f = os.path.join(out_base, d, "report", "explanation_summary.csv")
        if os.path.exists(f):
            e = pd.read_csv(f)
            e["dataset"] = d
            ex.append(e)
    if ex:
        ex = pd.concat(ex)
        ex.to_csv(os.path.join(dest, "explanations_all.csv"), index=False)
        ex.pivot_table(index="backend", columns="dataset", values="faithfulness_pass_rate").to_csv(
            os.path.join(dest, "faithfulness_by_dataset.csv"))
    iv = []
    for d in done:
        f = os.path.join(out_base, d, "report", "intervention_summary.csv")
        if os.path.exists(f):
            e = pd.read_csv(f)
            e["dataset"] = d
            iv.append(e)
    if iv:
        pd.concat(iv).to_csv(os.path.join(dest, "interventions_all.csv"), index=False)
    sh = {}
    for d in done:
        f = os.path.join(out_base, d, "report", "shap_summary.json")
        if os.path.exists(f):
            sh[d] = json.load(open(f))
    if sh:
        pd.DataFrame(sh).T.to_csv(os.path.join(dest, "shap_by_dataset.csv"))
    with open(os.path.join(dest, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("\nTest AUC by dataset:\n" + auc.round(4).to_string())
    log("\n" + json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------
# expert ratings analysis
# --------------------------------------------------------------------------
def kalpha(matrix):
    import krippendorff
    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"))


def analyse_explanation_ratings(files):
    raters = [pd.read_csv(f).set_index("case_id") for f in files]
    out = []
    for c in EXPL_CRITERIA:
        col = c + "_1to5"
        mat = np.array([r[col].reindex(raters[0].index).astype(float).to_numpy() for r in raters])
        vals = mat[~np.isnan(mat)]
        out.append({"criterion": c, "median": np.median(vals),
                    "IQR": np.subtract(*np.percentile(vals, [75, 25])),
                    "krippendorff_alpha": kalpha(mat) if len(raters) > 1 else np.nan})
    return pd.DataFrame(out)


def analyse_intervention_ratings(files, key_path):
    key = pd.read_csv(key_path).set_index("case_id")
    long = []
    for i, f in enumerate(files):
        r = pd.read_csv(f).set_index("case_id").join(key)
        for slot in ("A", "B"):
            for crit in ("relevance", "appropriateness"):
                long.append(pd.DataFrame({"rater": i, "case": r.index, "plan": r[slot],
                                          "criterion": crit,
                                          "score": r[f"{slot}_{crit}_1to5"].astype(float)}))
        pref = r["preferred_A_or_B"].str.upper().map(lambda s: None if pd.isna(s) else s)
        long.append(pd.DataFrame({"rater": i, "case": r.index, "plan": "pref",
                                  "criterion": "prefers_genex",
                                  "score": [float(r.loc[c, p] == "genex") if p in ("A", "B")
                                            else np.nan for c, p in pref.items()]}))
    d = pd.concat(long)
    res = []
    for crit in ("relevance", "appropriateness"):
        x = d[d.criterion == crit].pivot_table(index=["rater", "case"], columns="plan",
                                               values="score").dropna()
        w = stats.wilcoxon(x["genex"], x["baseline"]) if len(x) > 5 else None
        mats = [d[(d.criterion == crit) & (d.plan == pl)].pivot(index="rater", columns="case",
                                                                values="score").to_numpy()
                for pl in ("genex", "baseline")]
        res.append({"criterion": crit, "genex_median": x["genex"].median(),
                    "baseline_median": x["baseline"].median(),
                    "wilcoxon_p": w.pvalue if w else np.nan,
                    "alpha_genex": kalpha(mats[0]) if len(files) > 1 else np.nan,
                    "alpha_baseline": kalpha(mats[1]) if len(files) > 1 else np.nan})
    pref = d[d.criterion == "prefers_genex"].score.dropna()
    res.append({"criterion": "preference_rate_genex", "genex_median": pref.mean()})
    return pd.DataFrame(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--dataset", choices=DATASETS)
    r.add_argument("--out_root")
    c = sub.add_parser("cross")
    c.add_argument("--out_base")
    g = sub.add_parser("ratings")
    g.add_argument("--explanations", nargs="*")
    g.add_argument("--interventions", nargs="*")
    g.add_argument("--key", default=os.path.join(OUT_DIR, "interventions",
                                                 "expert_sheet_interventions_template_KEY.csv"),
                   help="the _KEY file matching the rated sheet")
    a = ap.parse_args()
    if a.cmd == "report":
        report(a.out_root or (dataset_out_root(a.dataset) if a.dataset else None))
    elif a.cmd == "cross":
        cross(a.out_base)
    else:
        if a.explanations:
            print(analyse_explanation_ratings(a.explanations).round(3).to_string(index=False))
        if a.interventions:
            print(analyse_intervention_ratings(a.interventions, a.key).round(3).to_string(index=False))
