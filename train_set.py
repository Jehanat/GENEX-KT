"""Train a set of models on one dataset with the same settings as run_all.py, then write the
report (metrics, DeLong tests).   python train_set.py <dataset> <model> [<model> ...] [--epochs N]

GENEX-KT uses outputs/<ds>/tuning/best_params.json (EdNet: the ASSISTments file).
GENEX-KT + Rasch uses outputs/<ds>/tuning_rasch/best_params.json (EdNet: the ASSISTments file)."""
import argparse
import json
import os

from genex import evaluate, train
from genex.config import ModelConfig, TrainConfig, out_root, processed_dir

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("models", nargs="+")
ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
ap.add_argument("--force", action="store_true")
ap.add_argument("--seed", type=int, default=42, help="other seeds write to outputs/<dataset>_seed<k>/")
a = ap.parse_args()


def params(kind):
    subs = {"genex_rasch": ["tuning_rasch"], "genex_v2": ["tuning_v2", "tuning_rasch"]}.get(kind, ["tuning"])
    for ds, sub in [(d, s) for s in subs for d in (a.dataset, "assist2009")]:
        f = os.path.join(out_root(ds), sub, "best_params.json")
        if os.path.exists(f):
            print(f"[{a.dataset}] {kind} hyper-parameters from {f}", flush=True)
            return json.load(open(f))
    raise SystemExit(f"no {subs} best_params.json for {a.dataset}")


pdir, odir = processed_dir(a.dataset), out_root(a.dataset)
if a.seed != 42:
    odir = odir.rstrip("/") + f"_seed{a.seed}"
for kind in a.models:
    mcfg, tcfg = ModelConfig(), TrainConfig(epochs=a.epochs, seed=a.seed)
    if kind.startswith("genex"):
        p = params(kind)
        for k in ("d_model", "n_heads", "n_layers", "dropout", "diffusion_steps", "lambda_diff", "lambda_noisy_bce"):
            if k in p:
                setattr(mcfg, k, p[k])
        tcfg.lr, tcfg.batch_size = p.get("lr", tcfg.lr), p.get("batch_size", tcfg.batch_size)
        tcfg.weight_decay = p.get("weight_decay", tcfg.weight_decay)
    train.run(kind, mcfg, tcfg, pdir, odir, force=a.force)
evaluate.report(odir)
