"""Download the three public datasets into data/raw/<dataset>/ (run on Colab).

    python -m genex.download assist2009 oulad ednet

Sources (as listed in the proposal):
  * ASSISTments 2009-2010 skill builder - mirror of the official CSV (pyBKT examples repo)
  * OULAD - Open University KMi (CC-BY 4.0), with the Figshare copy as a fallback
  * EdNet-KT1 + EdNet-Contents - the riiid/ednet README links (bit.ly -> Google Drive)
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

import requests

from .config import raw_dir

ASSIST_URL = "https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv"
# Figshare file 8606371 = anonymisedData.zip (the OU's own download link was retired)
OULAD_URLS = ["https://ndownloader.figshare.com/files/8606371",
              "https://analyse.kmi.open.ac.uk/open_dataset/download"]
EDNET_KT1 = "http://bit.ly/ednet_kt1"
EDNET_CONTENT = "http://bit.ly/ednet-content"


def resolve(url, hops=10):
    for _ in range(hops):
        r = requests.head(url, allow_redirects=False, timeout=30)
        if r.status_code in (301, 302, 303, 307, 308) and "location" in r.headers:
            url = requests.compat.urljoin(url, r.headers["location"])
        else:
            break
    return url


def fetch(url, dest):
    url = resolve(url)
    print(f"  {url} -> {dest}", flush=True)
    if "drive.google.com" in url or "docs.google.com" in url:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
        out = gdown.download(url, dest, fuzzy=True, quiet=False)
        if not out:
            raise RuntimeError("Google Drive refused the download (quota?). Download it "
                               f"in a browser from {url} and put it at {dest}.")
        return dest
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):   # decodes gzip transfer-encoding
                f.write(chunk)
    if os.path.getsize(dest) == 0:
        raise RuntimeError(f"empty download from {url}")
    return dest


def extract_csvs(zip_path, folder, wanted):
    """Pull the wanted CSVs out of a zip, searching nested zips too."""
    found = set()
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            base = os.path.basename(n)
            if base in wanted:
                with z.open(n) as src, open(os.path.join(folder, base), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                found.add(base)
            elif n.endswith(".zip"):
                inner = os.path.join(folder, "_inner.zip")
                with z.open(n) as src, open(inner, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                found |= extract_csvs(inner, folder, wanted - found)
                os.remove(inner)
    return found


def get_assist2009():
    d = raw_dir("assist2009")
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "as.csv")):
        fetch(ASSIST_URL, os.path.join(d, "as.csv"))


def get_oulad():
    d = raw_dir("oulad")
    os.makedirs(d, exist_ok=True)
    wanted = {"studentAssessment.csv", "assessments.csv", "courses.csv", "studentInfo.csv"}
    if all(os.path.exists(os.path.join(d, w)) for w in ["studentAssessment.csv", "assessments.csv"]):
        return
    for url in OULAD_URLS:
        try:
            z = fetch(url, os.path.join(d, "oulad.zip"))
            got = extract_csvs(z, d, wanted)
            os.remove(z)
            if {"studentAssessment.csv", "assessments.csv"} <= got:
                return
        except Exception as e:  # try the next mirror
            print(f"  failed: {e}")
    raise RuntimeError("OULAD download failed; download it from "
                       "https://analyse.kmi.open.ac.uk/open_dataset and unzip into " + d)


def get_ednet():
    d = raw_dir("ednet")
    os.makedirs(os.path.join(d, "contents"), exist_ok=True)
    if not os.path.exists(os.path.join(d, "KT1.zip")) and not os.path.isdir(os.path.join(d, "KT1")):
        fetch(EDNET_KT1, os.path.join(d, "KT1.zip"))
    if not os.path.exists(os.path.join(d, "contents", "questions.csv")):
        z = fetch(EDNET_CONTENT, os.path.join(d, "contents.zip"))
        if not extract_csvs(z, os.path.join(d, "contents"), {"questions.csv"}):
            raise RuntimeError("questions.csv not found in EdNet-Contents")
        os.remove(z)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["assist2009", "oulad", "ednet"]
    for ds in todo:
        print(f"== {ds} ==", flush=True)
        {"assist2009": get_assist2009, "oulad": get_oulad, "ednet": get_ednet}[ds]()
        print("  files:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(raw_dir(ds), "*"))))
