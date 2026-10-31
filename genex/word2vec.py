"""Concept vectors from pretrained word2vec (proposal: "knowledge concept embeddings
are initialised using pretrained word2vec representations where available").

Each concept name (e.g. "Equation Solving Two or Fewer Steps") becomes the mean of
the GoogleNews word2vec vectors of its words. Concepts without a usable name get NaN
and keep a random initialisation. Vectors are cached per dataset as
data/processed/<dataset>/concept_vectors.npy, so the 1.6 GB model is loaded once.
"""
import os
import re

import numpy as np

MODEL = "word2vec-google-news-300"
STOP = {"and", "or", "of", "the", "a", "an", "to", "with", "for", "in", "on", "from",
        "by", "than", "more", "fewer", "all", "unnamed", "skill", "tag"}


def tokens(name):
    name = re.sub(r"\(#\d+\)|#\d+", " ", name)           # drop id suffixes
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)     # split camelCase
    return [w for w in re.findall(r"[A-Za-z]+", name) if w.lower() not in STOP]


def build(names, kv):
    K = max(names)
    out = np.full((K, kv.vector_size), np.nan, dtype=np.float32)
    for k, name in names.items():
        vs = []
        for w in tokens(name):
            for cand in (w, w.lower(), w.capitalize()):
                if cand in kv:
                    vs.append(kv[cand])
                    break
        if vs:
            out[k - 1] = np.mean(vs, 0)
    return out


def concept_vectors(processed_dir, meta, log=print):
    path = os.path.join(processed_dir, "concept_vectors.npy")
    if os.path.exists(path):
        return np.load(path)
    try:
        import gensim.downloader as api
        log(f"[word2vec] loading {MODEL} (first time downloads ~1.6 GB)")
        kv = api.load(MODEL)
    except Exception as e:
        log(f"[word2vec] unavailable ({e}); concept embeddings stay random")
        return None
    vec = build(meta["skill_names"], kv)
    np.save(path, vec)
    log(f"[word2vec] vectors for {int((~np.isnan(vec).any(1)).sum())}/{len(vec)} concepts")
    return vec
