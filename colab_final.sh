#!/usr/bin/env bash
# Third Colab session: re-create the results lost when the second runtime was reset.
# Strictly sequential; a compact checkpoint (make_mini.py) is written after each stage.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 GENEX_HF_4BIT=1
step() { echo; echo "=== $(date +%H:%M) $* ==="; }
ckpt() { python make_mini.py > /dev/null && cp /content/genex_mini.tgz "/content/GENEX-KT_Results_$1.tgz" && echo "CHECKPOINT $1 ready"; }
step "install"
pip install -q pykt-toolkit --no-deps
pip install -q optuna krippendorff shap diffusers gensim wandb bitsandbytes accelerate gdown
step "A: ASSISTments item-difficulty ablation"
python -m genex.download assist2009 && python -m genex.data --dataset assist2009 > /dev/null
python ablation_rasch.py assist2009 2>&1 | grep --line-buffered -E "TEST|epoch"
ckpt A
step "B: EdNet data + BKT, DKT, SAKT, GENEX-KT"
python -m genex.download ednet && python -m genex.data --dataset ednet > /dev/null
python run_all.py --datasets ednet --models bkt dkt sakt genex --explainers template \
    --params outputs/assist2009/tuning/best_params.json 2>&1 | grep --line-buffered -E "TEST|===|==|rror"
ckpt B
step "C: EdNet explanations (Qwen2.5, Phi-3.5, free prompt) + interventions + report"
python run_all.py --datasets ednet --models bkt dkt sakt genex --explainers template qwen25 phi35 \
    --ablation_free_prompt --params outputs/assist2009/tuning/best_params.json 2>&1 \
    | grep --line-buffered -E "TEST|===|==|rror|faithful|SHAP"
ckpt C
step "D: EdNet AKT (6-epoch budget) + final report"
python run_all.py --datasets ednet --models bkt dkt sakt akt genex --epochs 6 --explainers template qwen25 phi35 \
    --ablation_free_prompt --params outputs/assist2009/tuning/best_params.json 2>&1 \
    | grep --line-buffered -E "TEST|===|==|rror|epoch"
ckpt final
step "ALL DONE"
