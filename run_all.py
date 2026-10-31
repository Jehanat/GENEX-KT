"""Run the whole GENEX-KT protocol for one or more datasets.

    python run_all.py --datasets assist2009 ednet oulad --explainers template llama3 qwen25
    python run_all.py --datasets assist2009 --explainers template      # offline, no LLM

For each dataset (modelled independently, as in the proposal):
  Phase 1  preprocessing                         -> data/processed/<dataset>/
  Phase 2  BKT, DKT/SAKT/AKT (pyKT), GENEX-KT     -> outputs/<dataset>/<model>/
  Phase 3  explanations (+ SHAP) per explainer    -> outputs/<dataset>/explanations/
  Phase 4  interventions per explainer            -> outputs/<dataset>/interventions/
  Report   tables, DeLong, SHAP, expert sheets    -> outputs/<dataset>/report/
Then the cross-dataset report                     -> outputs/cross_dataset/

Every step is skipped if its output already exists (use --force to redo), so an
interrupted Colab session can simply be restarted.
"""
import argparse
import json
import os

from genex import data, evaluate, explain, intervene, train
from genex.config import (DATASETS, ModelConfig, TrainConfig, out_root, processed_dir)

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
ap.add_argument("--explainers", nargs="+", default=["template"],
                choices=["template", "llama3", "qwen25", "phi35", "hf", "claude", "openai"])
ap.add_argument("--ablation_free_prompt", action="store_true",
                help="also run the 'free' (ungrounded) prompt for the first open LLM")
ap.add_argument("--models", nargs="+", default=["bkt", "dkt", "sakt", "akt", "genex"],
                help="add genex_rasch for the item-difficulty ablation")
ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
ap.add_argument("--params", help="best_params.json from genex.tune (used for every dataset "
                                 "unless outputs/<dataset>/tuning/best_params.json exists)")
ap.add_argument("--n_explain", type=int, default=100)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()


def genex_cfg(ds):
    mcfg, tcfg = ModelConfig(), TrainConfig(epochs=a.epochs)
    own = os.path.join(out_root(ds), "tuning", "best_params.json")
    path = own if os.path.exists(own) else a.params
    if path and os.path.exists(path):
        p = json.load(open(path))
        for k in ("d_model", "n_heads", "n_layers", "dropout", "diffusion_steps",
                  "lambda_diff", "lambda_noisy_bce"):
            if k in p:
                setattr(mcfg, k, p[k])
        tcfg.lr, tcfg.batch_size = p.get("lr", tcfg.lr), p.get("batch_size", tcfg.batch_size)
        print(f"[{ds}] GENEX-KT hyper-parameters from {path}")
    return mcfg, tcfg


for ds in a.datasets:
    pdir, odir = processed_dir(ds), out_root(ds)
    print(f"\n======== {ds} ========", flush=True)
    if a.force or not os.path.exists(os.path.join(pdir, "windows.npz")):
        print("== Phase 1: preprocessing ==", flush=True)
        _, _, m = data.preprocess(ds)
        print({k: m[k] for k in ("n_students", "n_skills", "n_items", "n_interactions")})

    print("== Phase 2: knowledge tracing models ==", flush=True)
    for m in a.models:
        mcfg, tcfg = genex_cfg(ds) if m.startswith("genex") else (ModelConfig(), TrainConfig(epochs=a.epochs))
        train.run(m, mcfg, tcfg, pdir, odir, force=a.force)

    runs = [(e, "grounded") for e in a.explainers]
    open_llms = [e for e in a.explainers if e in ("llama3", "qwen25", "phi35", "hf")]
    if a.ablation_free_prompt and open_llms:
        runs.append((open_llms[0], "free"))
    for backend, mode in runs:
        tag = backend if mode == "grounded" else f"{backend}_{mode}"
        if a.force or not os.path.exists(os.path.join(odir, "explanations", f"summary_{tag}.json")):
            print(f"== Phase 3: explanations ({tag}) ==", flush=True)
            explain.run(backend, a.n_explain, pdir, odir, prompt_mode=mode)
        if a.force or not os.path.exists(os.path.join(odir, "interventions",
                                                      f"validation_summary_{tag}.csv")):
            print(f"== Phase 4: interventions ({tag}) ==", flush=True)
            intervene.run(tag, pdir, odir)

    print("== Report ==", flush=True)
    evaluate.report(odir)

print("\n======== cross-dataset ========")
evaluate.cross()
