#!/usr/bin/env bash
# Build a ready-to-upload Hugging Face Space for the GENEX-KT demo.
#   bash make_space.sh                 # uses ./outputs
#   OUT=outputs_gpu bash make_space.sh # uses the GPU-trained model instead
# Produces hf_space/ with 4 files: app.py (launcher), README.md, requirements.txt and
# bundle.zip (code, processed data, trained model). The launcher unpacks the bundle on
# first start. Upload the 4 files to a Gradio Space (huggingface.co/new-space).
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-outputs}
rm -rf hf_space && mkdir -p hf_space/build/outputs/genex hf_space/build/outputs/interventions hf_space/build/data
B=hf_space/build
cp -r genex $B/ && rm -rf $B/genex/__pycache__
cp app.py $B/genex_app.py
cp demo_example.csv $B/
cp -r data/processed $B/data/
cp "$OUT"/genex/model.pt "$OUT"/genex/metrics.json $B/outputs/genex/
cp "$OUT"/interventions/prerequisite_graph.csv $B/outputs/interventions/
(cd $B && zip -qr -9 ../bundle.zip .)
rm -rf $B

cat > hf_space/app.py <<'EOF'
"""Hugging Face Spaces launcher for the GENEX-KT demo: unpacks bundle.zip, then runs it."""
import os
import runpy
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(HERE, "genex", "models.py")):
    zipfile.ZipFile(os.path.join(HERE, "bundle.zip")).extractall(HERE)
os.chdir(HERE)
runpy.run_path(os.path.join(HERE, "genex_app.py"), run_name="__main__")
EOF

cat > hf_space/requirements.txt <<'EOF'
--extra-index-url https://download.pytorch.org/whl/cpu
torch
numpy
pandas
scikit-learn
scipy
plotly
anthropic
openai
EOF

cat > hf_space/README.md <<'EOF'
---
title: GENEX-KT Demo
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
short_description: Explainable, generative knowledge tracing (MSc prototype)
---

# GENEX-KT — explainable, generative knowledge tracing

MSc research prototype (Liverpool John Moores University). Pick an ASSISTments 2009
student or upload an answer history to see the diffusion-KT mastery profile, a
faithfulness-checked diagnosis and a prerequisite-aware learning plan, next to a
rule-based baseline.

Data: ASSISTments 2009 skill-builder (public, anonymised). Not validated with real
classes; recommendations need teacher review.

To use Claude as the explainer, add `ANTHROPIC_API_KEY` under
*Settings → Variables and secrets* as a **secret**. Without it the app uses the
offline template explainer.
EOF
ls -la hf_space
echo "hf_space/ is ready to upload"
