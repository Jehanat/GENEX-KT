"""%run genex_kt/colab_launch_accuracy.py  - starts colab_accuracy.sh in the background."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
log = open("/content/GENEX-KT_accuracy.log", "a")
p = subprocess.Popen(["bash", os.path.join(HERE, "colab_accuracy.sh")], cwd=HERE, env=dict(os.environ),
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Started, pid", p.pid, "- progress: !grep -a -E 'TEST|===|CHECKPOINT|best val|rror' /content/GENEX-KT_accuracy.log | tail -20")
