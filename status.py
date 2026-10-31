"""Progress of the Colab run:  %run /content/status.py"""
import glob
import json
import os
import sqlite3
import subprocess
import time

B = "/content/genex_kt/outputs"
print(time.strftime("%H:%M UTC"))
for db in sorted(glob.glob(f"{B}/*/tuning/optuna.db")):
    c = sqlite3.connect(db)
    rows = c.execute("select state, count(*) from trials group by state").fetchall()
    best = c.execute("select max(value) from trial_values").fetchone()[0]
    print(db.split("/")[-3], "optuna:", rows, "best", best)
for m in sorted(glob.glob(f"{B}/*/*/metrics.json")):
    j = json.load(open(m))
    print(m.split("/")[-3], j["model"], "test AUC %.4f" % j["test"]["auc"])
for s in sorted(glob.glob(f"{B}/*/explanations/summary_*.json")):
    j = json.load(open(s))
    print(s.split("/")[-3], j["backend"], "faithful", j["faithfulness_pass_rate"])
import re
log = open("/content/GENEX-KT_run.log", errors="ignore").read()[-200000:]
lines = [l for l in re.split(r"[\r\n]+", log) if l.strip() and "it/s" not in l and "s/it" not in l
         and "Loading" not in l and "%|" not in l]
print("\n".join(lines[-8:]))
print(subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                     shell=True, capture_output=True, text=True).stdout)
