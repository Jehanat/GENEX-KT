"""Start the full GENEX-KT run from a Colab notebook cell:

    %run colab_launch.py

Reads HF_TOKEN from Colab secrets (if added), checks LLaMA-3 access, picks the open
explainers, and starts colab_pipeline.sh in the background (log: GENEX-KT_run.log).
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ)
try:
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
    if tok:
        env["HF_TOKEN"] = tok
        print("HF_TOKEN found in Colab secrets")
except Exception as e:                      # secret missing or notebook access not granted
    print("No HF_TOKEN secret available:", type(e).__name__)

explainers = "qwen25 phi35"
if env.get("HF_TOKEN"):
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("meta-llama/Meta-Llama-3-8B-Instruct", "config.json",
                        token=env["HF_TOKEN"])
        explainers = "llama3 qwen25"
    except Exception as e:
        print("LLaMA-3 not accessible yet:", type(e).__name__)
env["GENEX_EXPLAINERS"] = explainers
print("Open-source explainers:", explainers)

log = open(os.path.join(os.path.dirname(HERE), "GENEX-KT_run.log"), "a")
proc = subprocess.Popen(["bash", os.path.join(HERE, "colab_pipeline.sh")], cwd=HERE, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Started, pid", proc.pid, "- follow with: !tail -f /content/GENEX-KT_run.log")
