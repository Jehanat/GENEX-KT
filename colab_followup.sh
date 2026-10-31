#!/usr/bin/env bash
# Runs after colab_resume.sh: EdNet Phi-3.5 + free-prompt explanations (after the
# parsing fix), then the full EdNet report and the final compact results.
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 GENEX_HF_4BIT=1
while pgrep -f colab_resume.sh > /dev/null; do sleep 60; done
echo "=== $(date +%H:%M) EdNet: remaining explanations ==="
python run_all.py --datasets ednet --models bkt dkt sakt akt genex --explainers template qwen25 phi35 \
    --ablation_free_prompt --params outputs/assist2009/tuning/best_params.json 2>&1 | grep --line-buffered -v "it/s\]"
echo "=== $(date +%H:%M) final package ==="
python make_mini.py && cp /content/genex_mini.tgz /content/GENEX-KT_Results_final.tgz && ls -la /content/GENEX-KT_Results_final.tgz
echo "=== $(date +%H:%M) FOLLOWUP DONE ==="
