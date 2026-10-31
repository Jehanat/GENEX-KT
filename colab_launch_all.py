"""%run genex_kt/colab_launch_all.py - accuracy run (colab_accuracy.sh) then GENEX-KT v2 (colab_v2.sh)."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
log = open("/content/GENEX-KT_accuracy.log", "a")
p = subprocess.Popen(["bash", "-c", "bash colab_accuracy.sh; bash colab_v2.sh"], cwd=HERE, env=dict(os.environ),
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Started, pid", p.pid)
