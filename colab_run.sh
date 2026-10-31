#!/usr/bin/env bash
# One-shot end-to-end GENEX-KT run on a Colab GPU.
#   bash colab_run.sh                 # full run (all phases, open-source LLM)
#   OUT=outputs_gpu MODEL=Qwen/Qwen2.5-7B-Instruct bash colab_run.sh
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-outputs_gpu}
export GENEX_HF_MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
export GENEX_HF_4BIT=${HF_4BIT:-1}
export PYTHONWARNINGS=ignore

echo "== setup =="
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "WARNING: no GPU found"
pip install -q optuna krippendorff bitsandbytes accelerate anthropic
mkdir -p data/raw
if [ ! -f data/raw/as.csv ]; then
  echo "downloading ASSISTments 2009 skill-builder data"
  wget -q -O data/raw/as.csv https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv
fi

echo "== Phase 1: preprocessing =="
python -u -m genex.data | grep -E '"n_(students|skills|interactions|windows)"'

echo "== Phase 2-4 with open-source LLM ($GENEX_HF_MODEL) =="
python -u run_all.py --backend hf --d_model 128 --out_root "$OUT"

echo "== offline template explainer (control condition) =="
python -u -m genex.explain --backend template --out_root "$OUT"
python -u -m genex.intervene --backend template --out_root "$OUT"
python -u -m genex.evaluate report --out_root "$OUT"

echo "== explanation summaries =="
cat "$OUT"/explanations/summary_*.json
zip -qr "${OUT}.zip" "$OUT"
echo "DONE - results in $OUT/ and ${OUT}.zip"
