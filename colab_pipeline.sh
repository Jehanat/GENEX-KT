#!/usr/bin/env bash
# GENEX-KT - full open-source protocol on a Colab GPU, following the proposal.
#
#   bash colab_pipeline.sh            # everything: data -> tuning -> models -> LLMs -> reports
#   GENEX_EXPLAINERS="qwen25 phi35" bash colab_pipeline.sh
#
# Resumable: rerun after a disconnect and finished steps are skipped.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 GENEX_HF_4BIT=1
EXPLAINERS=${GENEX_EXPLAINERS:-"llama3 qwen25"}
mkdir -p outputs/logs /content/genex_raw
if [ ! -L data/raw ]; then             # raw downloads on the local disk (large, re-downloadable)
  [ -d data/raw ] && cp -rn data/raw/* /content/genex_raw/ 2>/dev/null
  rm -rf data/raw && ln -s /content/genex_raw data/raw
fi
step() { echo; echo "=== $(date +%H:%M) $* ==="; }

step "install"
pip install -q pykt-toolkit --no-deps
pip install -q optuna krippendorff shap diffusers gensim wandb bitsandbytes accelerate gdown

step "Phase 1: download + preprocess"
python -m genex.download assist2009 oulad ednet
for ds in assist2009 oulad ednet; do
  if [ ! -f data/processed/$ds/windows.npz ]; then
    python -m genex.data --dataset $ds | grep -E '"n_(students|skills|items|interactions)"'
  fi
done
python -c "from genex.data import load_processed; from genex.word2vec import concept_vectors; \
from genex.config import processed_dir as p; _, m = load_processed(p('assist2009')); concept_vectors(p('assist2009'), m)" \
  2>&1 | grep -v "downloaded"

step "Phase 2a: Optuna (100 trials per dataset; EdNet reuses ASSISTments)"
for ds in assist2009 oulad; do
  if [ -f outputs/$ds/tuning/best_params.json ]; then echo "$ds: saved result reused"; continue; fi
  python -m genex.tune --dataset $ds --trials 100 --epochs 15 2>&1 | grep -v "^\[I " | tail -1
done

step "Phase 2b-4: models, explanations ($EXPLAINERS + template), interventions, reports"
python run_all.py --datasets assist2009 oulad ednet --explainers template $EXPLAINERS \
    --ablation_free_prompt --params outputs/assist2009/tuning/best_params.json 2>&1 \
  | grep -v "epoch [0-9]*[02-46-9] "

step "exploratory data analysis tables (Chapter 4)"
python -m genex.eda --all > /dev/null

step "package results"
NAME=GENEX-KT_Results_$(date +%Y-%m-%d)
cp -r data/processed outputs/processed_data_summary 2>/dev/null
rm -f outputs/processed_data_summary/*/windows.npz outputs/processed_data_summary/*/interactions.csv.gz
(cd outputs && zip -qr ../$NAME.zip . -x "*/model.pt")
(cd outputs && zip -qr ../${NAME}_models.zip . -i "*/model.pt")
ls -la $NAME*.zip
step "ALL DONE"
