"""Phase 1 - Data collection and feature engineering.

Turns a raw interaction log into fixed-length windows with the four
features named in the proposal, plus a per-student temporal split:

  * skill id                 (integer index, 0 = padding)
  * correctness              (0/1)
  * response-time quantile   (bins fit on TRAIN rows only, 0 = unknown)
  * cumulative attempt count per skill (known before the student answers)

The question/item id is kept as well, because pyKT's AKT uses it for its
Rasch difficulty term.

Split: for every student, the first 80% of their interactions (in time
order) are train, the next 10% validation, the most recent 10% test.
Validation/test positions keep their earlier history as context, but only
train positions contribute to the training loss.

    python -m genex.data --dataset assist2009|ednet|oulad
"""
import argparse
import json
import os
import zipfile

import numpy as np
import pandas as pd

from .config import DATASET_SETTINGS, DATASETS, DataConfig, raw_dir

PAD = 0
SPLIT_CODES = {"pad": 0, "train": 1, "val": 2, "test": 3}
FEATS = ["skill", "item", "correct", "tbin", "attempts", "split", "lag"]
N_LAG_BINS = 13   # 0 = first attempt on the concept, 1..12 = 1 + floor(log2(gap))


# --------------------------------------------------------------------------
# Loaders: each returns a standard frame with columns
#   user, order, item_raw, skill_raw, skill_name, correct, response_ms, answer_type
# --------------------------------------------------------------------------
def load_assistments(path):
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df = df.dropna(subset=["skill_id"])                     # rows with no skill tag
    df = df[df["correct"].isin([0, 1])]                     # missing labels dropped
    # Multi-skill problems are repeated once per skill with the same order_id.
    # Keeping every copy leaks the answer into the next step, so keep one row.
    df = df.sort_values(["order_id", "skill_id"]).drop_duplicates("order_id", keep="first")
    return pd.DataFrame({
        "user": df["user_id"].astype(int),
        "order": df["order_id"].astype(np.int64),
        "item_raw": df["problem_id"].astype(int).astype(str),
        "skill_raw": df["skill_id"].astype(int).astype(str),
        "skill_name": df["skill_name"].fillna("Unknown skill").astype(str),
        "correct": df["correct"].astype(int),
        "response_ms": pd.to_numeric(df["ms_first_response"], errors="coerce"),
        "answer_type": df.get("answer_type", pd.Series("unknown", index=df.index)).fillna("unknown"),
    })


TOEIC_PARTS = {1: "Listening: Photographs", 2: "Listening: Question-Response",
               3: "Listening: Conversations", 4: "Listening: Talks",
               5: "Reading: Incomplete Sentences", 6: "Reading: Text Completion",
               7: "Reading: Comprehension"}


def load_ednet_kt1(kt1, questions_csv, n_students=100_000, seed=42):
    """EdNet-KT1: one CSV per user (u<id>.csv, in a folder or inside KT1.zip) plus
    contents/questions.csv. Samples n_students users at random (proposal risk plan:
    100K students). Concept = the question's first tag, named by its TOEIC part."""
    q = pd.read_csv(questions_csv)
    q = q[q["tags"].astype(str) != "-1"].copy()
    q["skill_raw"] = q["tags"].astype(str).str.split(";").str[0]
    part_of_tag = q.groupby("skill_raw")["part"].agg(lambda s: int(s.mode().iat[0]))
    q["skill_name"] = q["skill_raw"].map(
        lambda t: f"{TOEIC_PARTS.get(part_of_tag[t], 'Part ' + str(part_of_tag[t]))} · tag {t}")
    q = q.set_index("question_id")

    zf = zipfile.ZipFile(kt1) if str(kt1).endswith(".zip") else None
    names = ([n for n in zf.namelist() if n.endswith(".csv") and "/u" in "/" + n]
             if zf else [os.path.join(kt1, f) for f in os.listdir(kt1) if f.endswith(".csv")])
    rng = np.random.default_rng(seed)
    if len(names) > n_students:
        names = list(rng.choice(sorted(names), n_students, replace=False))
    frames = []
    for i, n in enumerate(names):
        u = pd.read_csv(zf.open(n) if zf else n)
        u["user"] = int(os.path.basename(n)[1:-4])
        frames.append(u)
        if (i + 1) % 20000 == 0:
            print(f"[ednet] read {i + 1}/{len(names)} users", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df.join(q[["correct_answer", "skill_raw", "skill_name"]], on="question_id")
    df = df.dropna(subset=["skill_raw", "user_answer", "elapsed_time", "timestamp"])
    return pd.DataFrame({
        "user": df["user"],
        "order": df["timestamp"].astype(np.int64),
        "item_raw": df["question_id"].astype(str),
        "skill_raw": df["skill_raw"],
        "skill_name": df["skill_name"],
        "correct": (df["user_answer"] == df["correct_answer"]).astype(int),
        "response_ms": df["elapsed_time"].astype(float),
        "answer_type": "multiple_choice",
    })


OULAD_TYPES = {"TMA": "Tutor-marked assessment", "CMA": "Computer-marked assessment",
               "Exam": "Final exam"}


def load_oulad(folder):
    """OULAD has no item-level responses, so it is adapted to KT at assessment
    level: each graded submission is one interaction; the concept is
    module x assessment type; correct = score >= 40 (the OU pass mark). Order is
    presentation start (B = February, J = October) + submission day. There is no
    response time, so the time-bin feature is 'unknown' for every row."""
    sa = pd.read_csv(os.path.join(folder, "studentAssessment.csv"))
    a = pd.read_csv(os.path.join(folder, "assessments.csv"))
    df = sa.merge(a, on="id_assessment", how="left")
    df = df[(df["is_banked"] == 0)].dropna(subset=["score", "date_submitted"])
    pres = df["code_presentation"].astype(str)
    start = pres.str[:4].astype(int) * 365 + np.where(pres.str[4] == "J", 273, 31)
    return pd.DataFrame({
        "user": df["id_student"].astype(int),
        "order": (start + df["date_submitted"].astype(int)).astype(np.int64),
        "item_raw": df["id_assessment"].astype(str),
        "skill_raw": df["code_module"] + "_" + df["assessment_type"],
        "skill_name": "Module " + df["code_module"] + " – " + df["assessment_type"].map(OULAD_TYPES),
        "correct": (df["score"] >= 40).astype(int),
        "response_ms": np.nan,
        "answer_type": df["assessment_type"],
    })


def load_generic(path):
    """Any CSV with columns user,order,skill_raw,skill_name,correct,response_ms."""
    df = pd.read_csv(path)
    df["answer_type"] = df.get("answer_type", "unknown")
    df["item_raw"] = df.get("item_raw", df["skill_raw"]).astype(str)
    return df


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def engineer(df, cfg: DataConfig):
    df = df.drop_duplicates().copy()
    df = df.sort_values(["user", "order"], kind="stable")
    counts = df.groupby("user")["user"].transform("size")
    df = df[counts >= cfg.min_interactions].copy()

    # skill encoding (1..K; 0 reserved for padding)
    skills = sorted(df["skill_raw"].astype(str).unique(), key=lambda s: (len(s), s))
    skill2idx = {s: i + 1 for i, s in enumerate(skills)}
    df["skill"] = df["skill_raw"].astype(str).map(skill2idx).astype(int)
    names = df.groupby("skill")["skill_name"].agg(lambda s: s.mode().iat[0]).to_dict()
    # concept names must be unique for the explainer: tag duplicates / unnamed with raw id
    raw_of = {v: k for k, v in skill2idx.items()}
    dup = pd.Series(names).duplicated(keep=False)
    for k in names:
        if names[k] in ("Unknown skill", "", "nan"):
            names[k] = f"Unnamed skill #{raw_of[k]}"
        elif dup[k]:
            names[k] = f"{names[k]} (#{raw_of[k]})"

    # item (question) encoding, 1..Q
    items = sorted(df["item_raw"].astype(str).unique())
    item2idx = {s: i + 1 for i, s in enumerate(items)}
    df["item"] = df["item_raw"].astype(str).map(item2idx).astype(int)

    # per-student temporal split: most recent 10% test, the 10% before it validation.
    # At least one interaction each, so short histories (OULAD) still get a test item.
    df["pos"] = df.groupby("user").cumcount()
    n = df.groupby("user")["user"].transform("size")
    _, va, te = cfg.split
    n_te = np.maximum(1, np.round(n * te)).astype(int)
    n_va = np.maximum(1, np.round(n * va)).astype(int)
    df["split"] = np.where(df["pos"] >= n - n_te, 3, np.where(df["pos"] >= n - n_te - n_va, 2, 1))

    # cumulative attempts on this skill BEFORE this interaction (no leakage)
    df["attempts"] = df.groupby(["user", "skill"]).cumcount().clip(upper=cfg.max_attempt_bucket)

    # forgetting signal: interactions since this student last practised the same concept
    # (computed from earlier rows only, so it is known before the answer; no leakage)
    gap = df["pos"] - df.groupby(["user", "skill"])["pos"].shift(1)
    lag = 1 + np.floor(np.log2(gap.clip(lower=1).fillna(1)))
    df["lag"] = np.where(gap.isna(), 0, np.minimum(lag, N_LAG_BINS - 1)).astype(int)

    # response-time quantile bins - edges fit on train rows only
    rt = pd.to_numeric(df["response_ms"], errors="coerce")
    rt = rt.where(rt > 0)
    train_rt = rt[df["split"] == 1].dropna()
    qs = np.linspace(0, 1, cfg.n_time_bins + 1)[1:-1]
    edges = np.unique(np.quantile(train_rt, qs)) if len(train_rt) else np.array([])
    tb = np.digitize(rt.fillna(-1), edges) + 1                # 1..n_bins
    df["tbin"] = np.where(rt.isna(), 0, tb).astype(int)       # 0 = unknown

    meta = {
        "dataset": cfg.dataset,
        "n_skills": len(skills),
        "n_items": len(items),
        "skill_names": {int(k): v for k, v in names.items()},
        "skill_raw": {int(v): k for k, v in skill2idx.items()},
        "time_bin_edges_ms": edges.tolist(),
        "n_time_bins": cfg.n_time_bins,
        "max_attempt_bucket": cfg.max_attempt_bucket,
        "n_lag_bins": N_LAG_BINS,
        "max_len": cfg.max_len,
        "min_interactions": cfg.min_interactions,
        "n_students": int(df["user"].nunique()),
        "n_interactions": int(len(df)),
        "split_counts": {k: int((df["split"] == v).sum()) for k, v in
                         {"train": 1, "val": 2, "test": 3}.items()},
    }
    return df, meta


def to_windows(df, max_len):
    """Chop each student's sequence into consecutive windows of max_len."""
    out = {c: [] for c in FEATS}
    users, starts = [], []
    for user, g in df.groupby("user", sort=True):
        arr = {c: g[c].to_numpy() for c in FEATS}
        for s in range(0, len(g), max_len):
            for c in FEATS:
                seg = arr[c][s:s + max_len]
                padded = np.zeros(max_len, dtype=np.int64)
                padded[:len(seg)] = seg
                out[c].append(padded)
            users.append(user)
            starts.append(s)
    res = {c: np.stack(v) for c, v in out.items()}
    res["user"] = np.array(users)
    res["start"] = np.array(starts)
    return res


def load_raw(dataset, **kw):
    r = kw.get("raw_dir") or raw_dir(dataset)
    if dataset == "assist2009":
        return load_assistments(kw.get("raw_path") or os.path.join(r, "as.csv"))
    if dataset == "ednet":
        kt1 = os.path.join(r, "KT1.zip") if os.path.exists(os.path.join(r, "KT1.zip")) \
            else os.path.join(r, "KT1")
        return load_ednet_kt1(kt1, os.path.join(r, "contents", "questions.csv"),
                              kw.get("n_students") or DATASET_SETTINGS["ednet"]["n_students"])
    if dataset == "oulad":
        return load_oulad(r)
    return load_generic(kw["raw_path"])


def preprocess(dataset="assist2009", cfg=None, **kw):
    cfg = cfg or DataConfig.for_dataset(dataset)
    raw = load_raw(dataset, **kw)
    df, meta = engineer(raw, cfg)
    win = to_windows(df, cfg.max_len)
    os.makedirs(cfg.processed_dir, exist_ok=True)
    np.savez_compressed(os.path.join(cfg.processed_dir, "windows.npz"), **win)
    df.drop(columns=["skill_name"]).to_csv(
        os.path.join(cfg.processed_dir, "interactions.csv.gz"), index=False)
    meta["n_windows"] = int(len(win["user"]))
    with open(os.path.join(cfg.processed_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return df, win, meta


def load_processed(processed_dir=None):
    processed_dir = processed_dir or DataConfig().processed_dir
    win = dict(np.load(os.path.join(processed_dir, "windows.npz")))
    with open(os.path.join(processed_dir, "meta.json")) as f:
        meta = json.load(f)
    meta["skill_names"] = {int(k): v for k, v in meta["skill_names"].items()}
    if "item" not in win:          # older processed files
        win["item"] = win["skill"].copy()
        meta.setdefault("n_items", meta["n_skills"])
    return win, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 1: preprocess a KT dataset")
    ap.add_argument("--dataset", default="assist2009", choices=DATASETS + ["generic"])
    ap.add_argument("--raw_dir")
    ap.add_argument("--raw_path")
    ap.add_argument("--n_students", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    c = DataConfig.for_dataset(a.dataset)
    if a.out:
        c.processed_dir = a.out
    _, _, m = preprocess(a.dataset, c, raw_dir=a.raw_dir, raw_path=a.raw_path,
                         n_students=a.n_students)
    print(json.dumps({k: v for k, v in m.items() if k not in ("skill_names", "skill_raw")}, indent=2))
