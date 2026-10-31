"""%run genex_kt/colab_launch_final.py  - starts colab_final.sh in the background."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
log = open("/content/GENEX-KT_final.log", "a")
p = subprocess.Popen(["bash", os.path.join(HERE, "colab_final.sh")], cwd=HERE, env=dict(os.environ),
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Started, pid", p.pid, "- progress: !grep -a -E 'TEST|===|CHECKPOINT|rror' /content/GENEX-KT_final.log")
