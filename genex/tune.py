"""Optuna hyper-parameter search for the GENEX-KT backbone (validation AUC).

    python -m genex.tune --trials 100 --epochs 15

The proposal specifies 100 trials. On a single GPU each trial takes a few
minutes; use --trials 20 for a quick search. Unpromising trials are pruned
early (median pruner). Best parameters go to outputs/tuning/best_params.json
and can be passed to genex.train.
"""
import argparse
import json
import os

import optuna

from .config import DATASETS, ModelConfig, TrainConfig
from .config import out_root as dataset_out_root, processed_dir as dataset_processed_dir
from .data import load_processed
from .train import train_neural


def objective_factory(win, meta, epochs, device, processed_dir=None, kind="genex", tune_wd=False):
    def objective(trial):
        m = ModelConfig(
            d_model=trial.suggest_categorical("d_model", [64, 128, 256]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
            n_layers=trial.suggest_int("n_layers", 1, 4),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            diffusion_steps=trial.suggest_categorical("diffusion_steps", [50, 100, 200, 500]),
            lambda_diff=trial.suggest_float("lambda_diff", 0.05, 2.0, log=True),
            lambda_noisy_bce=trial.suggest_float("lambda_noisy_bce", 0.0, 1.0),
        )
        t = TrainConfig(epochs=epochs, patience=3, device=device,
                        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]))
        if tune_wd:   # stronger regularisation against the early overfitting seen on OULAD
            t.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        _, best, _ = train_neural(kind, win, meta, m, t, out_dir=None,
                                  log=lambda *_: None, trial=trial, processed_dir=processed_dir)
        return best
    return objective


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dataset", default="assist2009", choices=DATASETS)
    ap.add_argument("--kind", default="genex", choices=["genex", "genex_rasch", "genex_v2"])
    ap.add_argument("--out_name", default="tuning")
    ap.add_argument("--tune_wd", action="store_true")
    a = ap.parse_args()
    pdir = dataset_processed_dir(a.dataset)
    win, meta = load_processed(pdir)
    out = os.path.join(dataset_out_root(a.dataset), a.out_name)
    os.makedirs(out, exist_ok=True)
    study = optuna.create_study(direction="maximize", study_name=f"{a.kind}_kt",
                                storage=f"sqlite:///{out}/optuna.db", load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    left = a.trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:                                    # resumable: only runs the missing trials
        study.optimize(objective_factory(win, meta, a.epochs, a.device, pdir, a.kind, a.tune_wd), n_trials=left)
    print("best val AUC", study.best_value, study.best_params)
    with open(os.path.join(out, "best_params.json"), "w") as f:
        json.dump({"val_auc": study.best_value, **study.best_params}, f, indent=2)
    study.trials_dataframe().to_csv(os.path.join(out, "trials.csv"), index=False)
