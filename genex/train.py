"""Train and evaluate one KT model on the processed data.

    python -m genex.train --model genex          # proposed backbone
    python -m genex.train --model dkt|sakt|akt|bkt

Writes to outputs/<model>/: checkpoint, metrics.json, test/val predictions.
"""
import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .bkt import BKT
from .config import (DATASET_SETTINGS, DATASETS, ModelConfig, TrainConfig, get_device,
                     out_root as dataset_out_root, processed_dir as dataset_processed_dir,
                     save_config)
from .data import FEATS, load_processed
from .models import PYKT_BATCH, PYKT_LR, build_model


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def batches(win, idx, bs, device):
    """Mini-batches, trimmed to the longest real sequence in the batch (padding is
    only ever at the end of a window, so this changes no prediction)."""
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        L = int((win["skill"][j] > 0).sum(1).max())
        yield j, {k: torch.as_tensor(win[k][j, :L], device=device) for k in FEATS if k in win}


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": float(roc_auc_score(y, p)), "acc": float(accuracy_score(y, p >= 0.5)),
            "bce": float(log_loss(y, p)), "n": int(len(y))}


@torch.no_grad()
def predict_rows(model, win, device, bs=128, mc=False):
    """Row-level predictions for every non-pad position, keyed by (user, pos)."""
    model.eval()
    out = []
    idx = np.arange(len(win["user"]))
    for j, b in batches(win, idx, bs, device):
        if mc and hasattr(model, "predict_mc"):
            p, s = model.predict_mc(b)
            p_clean = torch.sigmoid(model(b))
        else:
            p_clean = torch.sigmoid(model(b))
            p, s = p_clean, torch.zeros_like(p_clean)
        m = (b["split"] > 0).cpu().numpy()
        L = m.shape[1]
        users = np.repeat(win["user"][j][:, None], L, 1)
        pos = win["start"][j][:, None] + np.arange(L)[None]
        out.append(pd.DataFrame({
            "user": users[m], "pos": pos[m], "split": b["split"].cpu().numpy()[m],
            "correct": b["correct"].cpu().numpy()[m], "p": p_clean.cpu().numpy()[m],
            "p_mc": p.cpu().numpy()[m], "p_std": s.cpu().numpy()[m]}))
    return pd.concat(out, ignore_index=True)


class _WandB:
    """Optional Weights & Biases tracking (proposal section 8.2). On when
    GENEX_WANDB=1; runs offline unless WANDB_MODE=online, so no account is needed."""

    def __init__(self, kind, meta, cfg, out_dir):
        self.run = None
        if os.environ.get("GENEX_WANDB") != "1" or not out_dir:
            return
        try:
            import wandb
            self.run = wandb.init(project=os.environ.get("WANDB_PROJECT", "genex-kt"),
                                  name=f"{meta.get('dataset', 'data')}-{kind}", dir=out_dir,
                                  mode=os.environ.get("WANDB_MODE", "offline"),
                                  config=cfg, reinit=True)
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    def log(self, d):
        if self.run:
            self.run.log(d)

    def finish(self, d=None):
        if self.run:
            if d:
                self.run.summary.update(d)
            self.run.finish()


def eval_windows(win):
    """Only windows containing validation/test positions need predicting
    (val/test are each student's latest 20%); gives identical metrics, less work."""
    j = np.where((win["split"] >= 2).any(1))[0]
    return {k: v[j] for k, v in win.items()}


def train_neural(kind, win, meta, mcfg, tcfg, out_dir, log=print, trial=None,
                 processed_dir=None):
    device = get_device(tcfg.device)
    set_seed(tcfg.seed)
    hp = asdict(mcfg) if kind.startswith("genex") or kind.endswith("_lite") else {}
    if kind == "sakt_lite":
        hp["n_layers"] = 1
    model = build_model(kind, meta, **hp)
    if (kind.startswith("genex") and mcfg.use_word2vec and processed_dir
            and DATASET_SETTINGS.get(meta.get("dataset"), {}).get("word2vec")):
        from .word2vec import concept_vectors
        vec = concept_vectors(processed_dir, meta, log)
        if vec is not None:
            log(f"[{model.name}] word2vec initialised {model.init_concept_embeddings(vec)} "
                f"concept embeddings")
    model = model.to(device)
    lr = PYKT_LR.get(kind, tcfg.lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg.weight_decay)
    wb = _WandB(kind, meta, {"model": hp, "train": asdict(tcfg), "lr": lr}, out_dir)
    # only windows that contain at least one train position are used for training
    train_idx = np.where((win["split"] == 1).any(1))[0]
    best, best_state, bad = -1, None, 0
    history = []
    ev = eval_windows(win)
    for ep in range(tcfg.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        rng = np.random.default_rng(tcfg.seed + ep)
        for _, b in batches(win, rng.permutation(train_idx), PYKT_BATCH.get(kind, tcfg.batch_size),
                            device):
            loss = model.training_loss(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        rows = predict_rows(model, ev, device)
        v = rows[rows.split == 2]
        auc = roc_auc_score(v.correct, v.p)
        history.append({"epoch": ep + 1, "train_loss": tot / nb, "val_auc": float(auc),
                        "sec": round(time.time() - t0, 1)})
        wb.log(history[-1])
        log(f"[{model.name}] epoch {ep + 1:02d} loss {tot / nb:.4f} val AUC {auc:.4f} "
            f"({time.time() - t0:.0f}s)")
        if trial is not None:
            trial.report(auc, ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if auc > best + 1e-4:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg.patience:
                break
    model.load_state_dict(best_state)
    wb.finish({"best_val_auc": best})
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"kind": kind, "hp": hp, "state": best_state}, os.path.join(out_dir, "model.pt"))
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    return model, best, history


def load_model(out_dir, meta, device="cpu"):
    ck = torch.load(os.path.join(out_dir, "model.pt"), map_location=device)
    m = build_model(ck["kind"], meta, **ck["hp"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def run(kind, mcfg=None, tcfg=None, processed_dir=None, out_root=None, log=print,
        dataset=None, force=False):
    mcfg, tcfg = mcfg or ModelConfig(), tcfg or TrainConfig()
    if dataset:
        processed_dir = processed_dir or dataset_processed_dir(dataset)
        out_root = out_root or dataset_out_root(dataset)
    else:
        processed_dir = processed_dir or dataset_processed_dir()
        out_root = out_root or dataset_out_root()
    out_dir = os.path.join(out_root, kind)
    if not force and os.path.exists(os.path.join(out_dir, "metrics.json")):
        res = json.load(open(os.path.join(out_dir, "metrics.json")))
        log(f"[{res['model']}] already trained - TEST AUC {res['test']['auc']:.4f} (use --force)")
        return res
    win, meta = load_processed(processed_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if kind == "bkt":
        df = pd.read_csv(os.path.join(processed_dir, "interactions.csv.gz"))
        bkt = BKT(meta["n_skills"], device=get_device(tcfg.device))
        bkt.fit(df)
        rows = bkt.predict(df)
        rows["p_mc"], rows["p_std"] = rows["p"], 0.0
        bkt.describe(meta["skill_names"]).to_csv(os.path.join(out_dir, "bkt_params.csv"), index=False)
        name = "BKT"
    else:
        model, _, _ = train_neural(kind, win, meta, mcfg, tcfg, out_dir, log,
                                   processed_dir=processed_dir)
        rows = predict_rows(model, eval_windows(win), get_device(tcfg.device), mc=kind.startswith("genex"))
        name = model.name
    res = {"model": name, "train_seconds": round(time.time() - t0, 1)}
    for code, s in [(2, "val"), (3, "test")]:
        r = rows[rows.split == code]
        res[s] = metrics(r.correct.to_numpy(), r.p.to_numpy())
        if kind.startswith("genex"):
            res[s + "_mc"] = metrics(r.correct.to_numpy(), r.p_mc.to_numpy())
    rows[rows.split >= 2].to_csv(os.path.join(out_dir, "predictions.csv.gz"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    save_config(os.path.join(out_dir, "config.json"), model=mcfg, train=tcfg)
    log(f"[{name}] TEST AUC {res['test']['auc']:.4f}  ACC {res['test']['acc']:.4f}  "
        f"BCE {res['test']['bce']:.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="genex",
                    choices=["genex", "genex_rasch", "genex_v2", "dkt", "sakt", "akt", "bkt", "dkt_lite", "sakt_lite", "akt_lite"])
    ap.add_argument("--dataset", default=None, choices=DATASETS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_word2vec", action="store_true")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--d_model", type=int, default=ModelConfig.d_model)
    ap.add_argument("--n_layers", type=int, default=ModelConfig.n_layers)
    ap.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    ap.add_argument("--lambda_diff", type=float, default=ModelConfig.lambda_diff)
    ap.add_argument("--diffusion_steps", type=int, default=ModelConfig.diffusion_steps)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--processed_dir")
    ap.add_argument("--out_root")
    a = ap.parse_args()
    m = ModelConfig(d_model=a.d_model, n_layers=a.n_layers, dropout=a.dropout,
                    lambda_diff=a.lambda_diff, diffusion_steps=a.diffusion_steps,
                    use_word2vec=not a.no_word2vec)
    t = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, device=a.device)
    run(a.model, m, t, a.processed_dir, a.out_root, dataset=a.dataset, force=a.force)
