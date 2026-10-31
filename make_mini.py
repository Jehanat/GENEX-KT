"""Compact copy of the results (everything the thesis analysis needs, ~100 KB) so it
can be moved off Colab as text if a file download is not possible.

    python make_mini.py            # -> /content/genex_mini.tgz.b64 (and prints its size)
"""
import base64
import glob
import json
import os
import shutil
import tarfile

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

B = "/content/genex_kt/outputs"
M = "/content/genex_mini"
shutil.rmtree(M, ignore_errors=True)
keep = ["*/eda/*", "*/tuning/best_params.json", "*/tuning/trials.csv", "*/tuning_rasch/best_params.json", "*/tuning_rasch/trials.csv", "*/tuning_v2/best_params.json", "*/tuning_v2/trials.csv", "*/*/metrics.json",
        "*/*/history.csv", "*/report/*.csv", "*/report/*.json", "*/explanations/summary_*.json",
        "*/interventions/validation_per_student_*.csv", "*/interventions/validation_summary_*.csv",
        "*/interventions/prerequisite_graph.csv", "cross_dataset/*"]
for pat in keep:
    for f in glob.glob(os.path.join(B, pat)):
        if "expert_sheet" in f or "shap_attributions" in f:
            continue
        dst = os.path.join(M, os.path.relpath(f, B))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(f, dst)
# trimmed explanations: counts + a few examples per explainer
for f in glob.glob(os.path.join(B, "*/explanations/explanations_*.jsonl")):
    recs = [json.loads(l) for l in open(f)]
    ok = [r for r in recs if r["faithfulness"]["pass"] and r["explanation"].get("knowledge_gaps")]
    bad = [r for r in recs if not r["faithfulness"]["pass"]]
    pick = (ok[len(ok) // 2:len(ok) // 2 + 1] if ok else []) + bad[:1]
    for r in pick:
        r["prompt"] = r["prompt"][:3000]
    dst = os.path.join(M, os.path.relpath(f, B))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        for r in pick:
            fh.write(json.dumps(r) + "\n")
    kinds = {}
    for r in recs:
        for iss in r["faithfulness"].get("issues", []):
            k = ("unknown concept" if iss.startswith("unknown concept") else
                 "unparseable output" if "unparseable" in iss else
                 "no claims" if "no concept-level" in iss else iss.split("' ")[-1].split(" (")[0])
            kinds[k] = kinds.get(k, 0) + 1
    json.dump({"n": len(recs), "n_unfaithful": len(bad), "issue_kinds": kinds},
              open(dst + ".counts.json", "w"))
# uncertainty summary instead of the prediction files
for f in glob.glob(os.path.join(B, "*/genex*/predictions.csv.gz")):
    p = pd.read_csv(f)
    p = p[p.split == 3]
    if "p_std" not in p:
        continue
    p["err"] = np.abs(p.correct - p.p_mc)
    p["dec"] = pd.qcut(p.p_std.rank(method="first"), 10, labels=False)
    g = p.groupby("dec").agg(std=("p_std", "mean"), err=("err", "mean"))
    rho = stats.spearmanr(p.p_std, p.err)
    lo, hi = p[p.p_std <= p.p_std.median()], p[p.p_std > p.p_std.median()]
    out = {"spearman_std_vs_abs_error": float(rho.correlation), "p": float(rho.pvalue),
           "auc_certain_half": float(roc_auc_score(lo.correct, lo.p_mc)),
           "auc_uncertain_half": float(roc_auc_score(hi.correct, hi.p_mc)),
           "mean_std": float(p.p_std.mean()), "err_by_decile": g.err.round(4).tolist(),
           "std_by_decile": g["std"].round(5).tolist()}
    dst = os.path.join(M, os.path.relpath(os.path.dirname(f), B), "uncertainty.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w"))
with tarfile.open("/content/genex_mini.tgz", "w:gz") as t:
    t.add(M, arcname="genex_mini")
b = base64.b64encode(open("/content/genex_mini.tgz", "rb").read()).decode()
open("/content/genex_mini.tgz.b64", "w").write(b)
print("mini archive", os.path.getsize("/content/genex_mini.tgz"), "bytes; base64", len(b))
