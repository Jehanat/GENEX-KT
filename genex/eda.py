"""Exploratory data analysis for the thesis (Chapter 4).

    python -m genex.eda --dataset assist2009      # or: --all

Writes small summary tables to outputs/<dataset>/eda/ (no raw rows), so the
figures can be drawn after the results zip has been downloaded.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .config import DATASETS, out_root, processed_dir


def run(ds):
    pdir, odir = processed_dir(ds), os.path.join(out_root(ds), "eda")
    os.makedirs(odir, exist_ok=True)
    meta = json.load(open(os.path.join(pdir, "meta.json")))
    df = pd.read_csv(os.path.join(pdir, "interactions.csv.gz"),
                     usecols=["user", "skill", "item", "correct", "response_ms", "split", "attempts"])
    names = {int(k): v for k, v in meta["skill_names"].items()}
    per_user = df.groupby("user").agg(n=("correct", "size"), acc=("correct", "mean"),
                                      n_skills=("skill", "nunique"))
    rt = df.response_ms.dropna()
    rt = rt[rt > 0]
    summary = {
        "dataset": ds, "n_students": int(df.user.nunique()), "n_interactions": int(len(df)),
        "n_concepts": int(df.skill.nunique()), "n_items": int(df.item.nunique()),
        "overall_accuracy": float(df.correct.mean()),
        "seq_len": {q: float(per_user.n.quantile(v)) for q, v in
                    [("min", 0), ("p25", .25), ("median", .5), ("mean", None), ("p75", .75), ("max", 1)]
                    if v is not None} | {"mean": float(per_user.n.mean())},
        "share_seq_over_200": float((per_user.n > 200).mean()),
        "concepts_per_student_median": float(per_user.n_skills.median()),
        "student_accuracy_median": float(per_user.acc.median()),
        "student_accuracy_iqr": [float(per_user.acc.quantile(.25)), float(per_user.acc.quantile(.75))],
        "response_time_available": float(df.response_ms.notna().mean()),
        "response_time_median_s": float(rt.median() / 1000) if len(rt) else None,
        "split_counts": {k: int(v) for k, v in df.split.map({1: "train", 2: "val", 3: "test"})
                         .value_counts().items()},
        "split_accuracy": {k: float(v) for k, v in df.groupby(df.split.map(
            {1: "train", 2: "val", 3: "test"})).correct.mean().items()},
        "items_per_concept_median": float(df.groupby("skill").item.nunique().median()),
    }
    json.dump(summary, open(os.path.join(odir, "eda_summary.json"), "w"), indent=2)
    # sequence-length histogram (log bins) and student accuracy histogram
    edges = np.unique(np.round(np.logspace(0, np.log10(per_user.n.max() + 1), 40)))
    h, e = np.histogram(per_user.n, bins=edges)
    pd.DataFrame({"left": e[:-1], "right": e[1:], "count": h}).to_csv(
        os.path.join(odir, "seq_len_hist.csv"), index=False)
    h, e = np.histogram(per_user.acc, bins=np.linspace(0, 1, 21))
    pd.DataFrame({"left": e[:-1], "right": e[1:], "count": h}).to_csv(
        os.path.join(odir, "student_acc_hist.csv"), index=False)
    # per-concept table
    pc = df.groupby("skill").agg(interactions=("correct", "size"), accuracy=("correct", "mean"),
                                 students=("user", "nunique"), items=("item", "nunique"))
    pc["name"] = pc.index.map(names)
    pc.sort_values("interactions", ascending=False).to_csv(os.path.join(odir, "per_concept.csv"))
    # accuracy by prior attempts on the concept (learning curve)
    lc = df.groupby("attempts").correct.agg(["mean", "size"])
    lc.to_csv(os.path.join(odir, "learning_curve.csv"))
    # response-time quantiles vs accuracy
    if len(rt) > 100:
        d = df[df.response_ms > 0].copy()
        d["rt_decile"] = pd.qcut(d.response_ms, 10, labels=False, duplicates="drop")
        d.groupby("rt_decile").agg(median_s=("response_ms", lambda s: s.median() / 1000),
                                   accuracy=("correct", "mean")).to_csv(
            os.path.join(odir, "rt_vs_accuracy.csv"))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for ds in (DATASETS if a.all or not a.dataset else [a.dataset]):
        if os.path.exists(os.path.join(processed_dir(ds), "interactions.csv.gz")):
            run(ds)
