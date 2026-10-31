#!/usr/bin/env bash
# Accuracy-improvement session: tune GENEX-KT + Rasch (item difficulty), train it on all three
# datasets, and re-train the baselines so DeLong tests can be computed. Strictly sequential;
# a compact checkpoint (make_mini.py) is written after each stage.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
step() { echo; echo "=== $(date +%H:%M) $* ==="; }
ckpt() { python make_mini.py > /dev/null && cp /content/genex_mini.tgz "/content/GENEX-KT_Accuracy_$1.tgz" && echo "CHECKPOINT $1 ready"; }
F="TEST|===|rror|best val|Trial .* finished"
step "install"
pip install -q pykt-toolkit --no-deps
pip install -q optuna shap diffusers gensim gdown
for ds in assist2009 oulad; do
  step "$ds: data + baselines + GENEX-KT"
  python -m genex.download $ds && python -m genex.data --dataset $ds > /dev/null
  python train_set.py $ds bkt dkt sakt akt genex 2>&1 | grep --line-buffered -E "$F"
  N=40; [ $ds = oulad ] && N=50
  step "$ds: Optuna search for GENEX-KT + Rasch ($N trials)"
  python -m genex.tune --dataset $ds --kind genex_rasch --out_name tuning_rasch --tune_wd --trials $N --epochs 15 2>&1 \
      | grep --line-buffered -E "best val|finished|rror"
  step "$ds: final GENEX-KT + Rasch + report"
  python train_set.py $ds genex_rasch 2>&1 | grep --line-buffered -E "$F"
  ckpt $ds
done
step "ednet: data + GENEX-KT + Rasch, GENEX-KT, BKT, DKT, SAKT"
python -m genex.download ednet && python -m genex.data --dataset ednet > /dev/null
python train_set.py ednet genex_rasch genex bkt dkt sakt 2>&1 | grep --line-buffered -E "$F"
ckpt ednet
step "ednet: AKT (6-epoch budget, as before) + report"
python train_set.py ednet akt --epochs 6 2>&1 | grep --line-buffered -E "$F"
ckpt final
step "ALL DONE"
