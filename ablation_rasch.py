"""Exploratory ablation (Chapter 5): GENEX-KT + Rasch item difficulty, trained with
each dataset's tuned GENEX-KT hyper-parameters.   python ablation_rasch.py oulad assist2009"""
import json
import os
import sys

from genex import train
from genex.config import ModelConfig, TrainConfig, out_root

for ds in sys.argv[1:]:
    mcfg, tcfg = ModelConfig(), TrainConfig()
    p = json.load(open(os.path.join(out_root(ds), "tuning", "best_params.json")))
    for k in ("d_model", "n_heads", "n_layers", "dropout", "diffusion_steps", "lambda_diff", "lambda_noisy_bce"):
        setattr(mcfg, k, p[k])
    tcfg.lr, tcfg.batch_size = p["lr"], p["batch_size"]
    train.run("genex_rasch", mcfg, tcfg, dataset=ds)
