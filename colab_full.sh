#!/usr/bin/env bash
# Full plan, chained after the fast run. Order chosen so that the important stages fit in the
# ~12-hour Colab limit and every model is final before it is explained:
#  0 wait: parallel OULAD job done; stop colab_fast.sh when it reaches EdNet (EdNet is redone here
#    with the final ASSISTments settings)
#  1 full-length tuning (resumes the saved Optuna studies) + final variants on ASSISTments/OULAD
#  2 EdNet variants with the final ASSISTments settings
#  3 explanations + SHAP + pathways on the best variant (chosen by validation AUC), all datasets
#  4 EdNet baselines + AKT (DeLong), 5 extra seeds
set -uo pipefail
cd /content/genex_kt
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 GENEX_HF_4BIT=1
step() { echo; echo "=== $(date +%H:%M) [full] $* ==="; }
ckpt() { python ensemble.py > /dev/null 2>&1; python make_mini.py > /dev/null && cp /content/genex_mini.tgz "/content/GENEX-KT_Full_$1.tgz" && echo "CHECKPOINT $1 ready"; }
F="TEST|===|rror|best val|vs|faithful|SHAP|variants"
step "waiting for the fast run and the parallel OULAD job"
while kill -0 4191 2>/dev/null; do sleep 60; done
until grep -q "ednet: data" /content/GENEX-KT_fast.log || ! kill -0 2420 2>/dev/null; do sleep 60; done
kill -9 -- -2420 2>/dev/null && echo "stopped colab_fast.sh before its EdNet stage"
sleep 5; rm -rf data/raw/ednet data/processed/ednet outputs/ednet/genex_rasch outputs/ednet/genex_v2
pip install -q bitsandbytes accelerate transformers krippendorff
step "1: full-length tuning"
python -m genex.tune --dataset assist2009 --kind genex_rasch --out_name tuning_rasch --tune_wd --trials 40 --epochs 15 2>&1 | grep --line-buffered -E "best val|rror"
python -m genex.tune --dataset assist2009 --kind genex_v2 --out_name tuning_v2 --tune_wd --trials 30 --epochs 15 2>&1 | grep --line-buffered -E "best val|rror"
python -m genex.tune --dataset oulad --kind genex_rasch --out_name tuning_rasch --tune_wd --trials 50 --epochs 15 2>&1 | grep --line-buffered -E "best val|rror"
python -m genex.tune --dataset oulad --kind genex_v2 --out_name tuning_v2 --tune_wd --trials 40 --epochs 15 2>&1 | grep --line-buffered -E "best val|rror"
for ds in assist2009 oulad; do python train_set.py $ds genex_rasch genex_v2 --force 2>&1 | grep --line-buffered -E "$F"; done
ckpt tuned
step "2: EdNet data + variants (final ASSISTments settings)"
python -m genex.download ednet && python -m genex.data --dataset ednet > /dev/null
python train_set.py ednet genex_rasch genex_v2 2>&1 | grep --line-buffered -E "$F"
ckpt ednet_variants
for ds in assist2009 oulad ednet; do
  step "3: $ds explanations + SHAP + pathways on the best variant"
  python explain_best.py $ds --explainers template qwen25 phi35 --free 2>&1 | grep --line-buffered -E "$F"
  ckpt explain_$ds
done
step "4: EdNet baselines + AKT (6-epoch budget) for DeLong"
python train_set.py ednet genex bkt dkt sakt 2>&1 | grep --line-buffered -E "$F"
python train_set.py ednet akt --epochs 6 2>&1 | grep --line-buffered -E "$F"
ckpt ednet_all
step "5: extra seeds 43, 44"
for sd in 43 44; do for ds in assist2009 oulad; do
  python train_set.py $ds dkt akt genex genex_rasch genex_v2 --seed $sd 2>&1 | grep --line-buffered -E "TEST"
done; done
ckpt final
step "FULL DONE"
