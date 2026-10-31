"""Phases 3-4 (explanations, SHAP, interventions) on the most accurate GENEX-KT variant.

    python explain_best.py <dataset> [--kind auto|genex_rasch|genex_v2] [--explainers qwen25 phi35] [--free]

Writes to outputs/<dataset>_best/ (the original base-model results stay untouched); inside it,
'genex' is a link to the chosen variant, so explain.py / intervene.py / evaluate.py run unchanged."""
import argparse
import json
import os

from genex import evaluate, explain, intervene
from genex.config import out_root, processed_dir

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--kind", default="auto")
ap.add_argument("--explainers", nargs="+", default=["template", "qwen25", "phi35"])
ap.add_argument("--free", action="store_true", help="also the free-prompt ablation for the first LLM")
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()

src = out_root(a.dataset)
kind = a.kind
if kind == "auto":      # highest validation AUC among the variants (test AUC is never used to choose)
    cands = {}
    for k in ("genex", "genex_rasch", "genex_v2"):
        f = os.path.join(src, k, "metrics.json")
        if os.path.exists(f):
            cands[k] = json.load(open(f))["val"]["auc"]
    kind = max(cands, key=cands.get)
    print(f"[{a.dataset}] variants by validation AUC: {cands} -> using {kind}", flush=True)
dst = src.rstrip("/") + "_best"
os.makedirs(dst, exist_ok=True)
link = os.path.join(dst, "genex")
if os.path.islink(link) and os.readlink(link) != os.path.join(src, kind):
    os.remove(link)
if not os.path.exists(link):
    os.symlink(os.path.join(src, kind), link)
json.dump({"dataset": a.dataset, "variant": kind}, open(os.path.join(dst, "variant.json"), "w"))
pdir = processed_dir(a.dataset)
runs = [(e, "grounded") for e in a.explainers]
llms = [e for e in a.explainers if e in ("qwen25", "phi35", "llama3", "hf")]
if a.free and llms:
    runs.append((llms[0], "free"))
for backend, mode in runs:
    tag = backend if mode == "grounded" else f"{backend}_{mode}"
    if not os.path.exists(os.path.join(dst, "explanations", f"summary_{tag}.json")):
        print(f"== explanations ({tag}) on {kind} ==", flush=True)
        explain.run(backend, a.n, pdir, dst, prompt_mode=mode)
    if not os.path.exists(os.path.join(dst, "interventions", f"validation_summary_{tag}.csv")):
        print(f"== interventions ({tag}) ==", flush=True)
        intervene.run(tag, pdir, dst)
evaluate.report(dst)
