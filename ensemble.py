"""Post-hoc ensemble of the GENEX-KT variants (no training): average P(correct) of
GENEX-KT, GENEX-KT + Rasch and GENEX-KT v2 on identical test positions; DeLong vs AKT and vs the best single variant."""
import json, os, sys
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/content/genex_kt")
from genex.evaluate import delong_test
B = "/content/genex_kt/outputs"
for ds in ["assist2009", "oulad", "ednet"]:
    P = {}
    for k in ["genex", "genex_rasch", "genex_v2", "akt"]:
        f = os.path.join(B, ds, k, "predictions.csv.gz")
        if os.path.exists(f):
            p = pd.read_csv(f); P[k] = p[p.split == 3].set_index(["user", "pos"])
    var = [k for k in ["genex", "genex_rasch", "genex_v2"] if k in P]
    if len(var) < 2:
        continue
    j = P[var[0]][["correct"]].copy()
    for k in var:
        j[k] = P[k]["p"]
    j = j.dropna()
    j["ens"] = j[var].mean(1)
    out = {"members": var, "n": len(j), "auc_ensemble": roc_auc_score(j.correct, j.ens),
           "auc_members": {k: roc_auc_score(j.correct, j[k]) for k in var}}
    best = max(var, key=lambda k: out["auc_members"][k])
    out["delong_vs_best_member"] = dict(zip(["auc_a", "auc_b", "z", "p"], delong_test(j.correct.values, j.ens.values, j[best].values)), member=best)
    if "akt" in P:
        a = j.join(P["akt"][["p"]].rename(columns={"p": "akt"}), how="inner")
        out["delong_vs_akt"] = dict(zip(["auc_a", "auc_b", "z", "p"], delong_test(a.correct.values, a.ens.values, a.akt.values)))
    os.makedirs(os.path.join(B, ds, "report"), exist_ok=True)
    json.dump(out, open(os.path.join(B, ds, "report", "ensemble.json"), "w"), indent=1)
    print(ds, "ensemble AUC", round(out["auc_ensemble"], 4), {k: round(v, 4) for k, v in out["auc_members"].items()})
