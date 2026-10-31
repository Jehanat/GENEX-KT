#!/usr/bin/env bash
# Second Colab session: finish EdNet (all phases) + the Rasch ablation. Sequential on purpose
# (running jobs in parallel exhausted the VM's RAM in the first session).
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 GENEX_HF_4BIT=1
step() { echo; echo "=== $(date +%H:%M) $* ==="; }
step "install"
pip install -q pykt-toolkit --no-deps
pip install -q optuna krippendorff shap diffusers gensim wandb bitsandbytes accelerate gdown
step "OULAD: data + Rasch ablation (1 min)"
python -m genex.download oulad && python -m genex.data --dataset oulad > /dev/null
python ablation_rasch.py oulad 2>&1 | grep -E "TEST|epoch"
step "EdNet: download + preprocess"
python -m genex.download ednet && python -m genex.data --dataset ednet > /dev/null
python -m genex.eda --dataset ednet > /dev/null
step "EdNet: BKT, DKT, SAKT, GENEX-KT, explanations, interventions"
python run_all.py --datasets ednet --models bkt dkt sakt genex --explainers template qwen25 phi35 \
    --ablation_free_prompt --params outputs/assist2009/tuning/best_params.json 2>&1 | grep --line-buffered -v "it/s\]"
step "EdNet: AKT (epoch budget 6 - compute limit, documented)"
python run_all.py --datasets ednet --models bkt dkt sakt akt genex --epochs 6 --explainers template \
    --params outputs/assist2009/tuning/best_params.json 2>&1 | grep --line-buffered -v "it/s\]"
step "ASSISTments: data + Rasch ablation"
python -m genex.download assist2009 && python -m genex.data --dataset assist2009 > /dev/null
python ablation_rasch.py assist2009 2>&1 | grep -E "TEST|epoch"
step "package"
python make_mini.py
ls -la /content/genex_mini.tgz
step "ALL DONE"
