"""%run genex_kt/colab_launch_v2.py  - starts colab_v2.sh in the background."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
log = open("/content/GENEX-KT_v2.log", "a")
p = subprocess.Popen(["bash", os.path.join(HERE, "colab_v2.sh")], cwd=HERE, env=dict(os.environ),
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Started, pid", p.pid, "- progress: !grep -a -E 'TEST|===|CHECKPOINT|best val|rror' /content/GENEX-KT_v2.log | tail -20")
