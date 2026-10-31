"""%run genex_kt/colab_launch_resume.py  - starts colab_resume.sh in the background."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
log = open("/content/GENEX-KT_resume.log", "a")
p = subprocess.Popen(["bash", os.path.join(HERE, "colab_resume.sh")], cwd=HERE, env=dict(os.environ),
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Started, pid", p.pid, "- progress: !tail -n 20 /content/GENEX-KT_resume.log")
