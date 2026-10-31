#!/usr/bin/env bash
# Second accuracy session (run AFTER colab_accuracy.sh, in the same Colab runtime):
# GENEX-KT v2 = Rasch item difficulty + forgetting-aware attention + "time since last
# practice" feature. Data are re-prepared to add the lag column (all other columns are
# unchanged, so the baseline predictions from the first session stay valid for DeLong).
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
step() { echo; echo "=== $(date +%H:%M) $* ==="; }
ckpt() { python make_mini.py > /dev/null && cp /content/genex_mini.tgz "/content/GENEX-KT_V2_$1.tgz" && echo "CHECKPOINT $1 ready"; }
F="TEST|===|rror|best val"
pip install -q pykt-toolkit --no-deps; pip install -q optuna shap diffusers gensim gdown
for ds in assist2009 oulad ednet; do
  step "$ds: re-prepare data with the lag feature"
  [ -d data/raw/$ds ] || python -m genex.download $ds
  grep -q n_lag_bins data/processed/$ds/meta.json 2>/dev/null || python -m genex.data --dataset $ds > /dev/null
  if [ $ds != ednet ]; then
    N=30; [ $ds = oulad ] && N=40
    step "$ds: Optuna search for GENEX-KT v2 ($N trials)"
    python -m genex.tune --dataset $ds --kind genex_v2 --out_name tuning_v2 --tune_wd --trials $N --epochs 15 2>&1 \
        | grep --line-buffered -E "best val|finished|rror"
  fi
  step "$ds: final GENEX-KT v2 + report"
  python train_set.py $ds genex_v2 2>&1 | grep --line-buffered -E "$F|vs"
  ckpt $ds
done
step "ALL DONE"
