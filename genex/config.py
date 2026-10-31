"""Central configuration for GENEX-KT. Every script reads defaults from here;
most values can also be overridden on the command line.

Three datasets, as in the proposal, each modelled independently:
    assist2009  ASSISTments 2009-2010 skill builder
    ednet       EdNet-KT1 (100K-student subsample, per the proposal's risk plan)
    oulad       Open University Learning Analytics Dataset (assessment-level KT)

The active dataset comes from --dataset on the command line or the GENEX_DATASET
environment variable (default assist2009). Data lives in data/processed/<dataset>/
and results in outputs/<dataset>/.
"""
from dataclasses import dataclass, asdict
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_BASE = os.environ.get("GENEX_OUT_BASE", os.path.join(ROOT, "outputs"))
DATASETS = ["assist2009", "ednet", "oulad"]
DATASET = os.environ.get("GENEX_DATASET", "assist2009")

# per-dataset settings that differ from the defaults
DATASET_SETTINGS = {
    "assist2009": {"min_interactions": 10, "word2vec": True},
    "ednet": {"min_interactions": 10, "word2vec": False, "n_students": 100_000},
    # OULAD students have ~5-15 assessments; the 200-step cap never binds
    "oulad": {"min_interactions": 5, "word2vec": False},
}


def raw_dir(dataset=DATASET):
    return os.path.join(DATA_DIR, "raw", dataset)


def processed_dir(dataset=DATASET):
    return os.path.join(DATA_DIR, "processed", dataset)


def out_root(dataset=DATASET):
    # GENEX_OUT_DIR (single folder) is kept for the demo app
    return os.environ.get("GENEX_OUT_DIR", os.path.join(OUT_BASE, dataset))


OUT_DIR = out_root()


@dataclass
class DataConfig:
    dataset: str = DATASET
    raw_path: str = os.path.join(DATA_DIR, "raw", "assist2009", "as.csv")
    processed_dir: str = processed_dir()
    max_len: int = 200            # proposal: sequences capped at 200 timesteps
    min_interactions: int = 10     # students with fewer interactions are dropped
    split: tuple = (0.8, 0.1, 0.1)  # per-student temporal split (train/val/test)
    n_time_bins: int = 10          # response-time quantile bins
    max_attempt_bucket: int = 15   # cumulative attempts per skill are capped here
    seed: int = 42

    @classmethod
    def for_dataset(cls, dataset, **kw):
        s = DATASET_SETTINGS.get(dataset, {})
        return cls(dataset=dataset, processed_dir=processed_dir(dataset),
                   min_interactions=s.get("min_interactions", 10), **kw)


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    # diffusion
    diffusion_steps: int = 100     # T in the forward process
    lambda_diff: float = 0.5       # weight of the denoising (MSE) loss
    lambda_noisy_bce: float = 0.5  # weight of BCE on the noised pass
    mc_samples: int = 8            # Monte-Carlo samples for the uncertainty estimate
    mc_noise_step: int = 10        # noise level used when sampling at inference
    use_word2vec: bool = True      # word2vec concept-embedding init where names exist


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    patience: int = 5
    seed: int = 42
    device: str = "auto"


@dataclass
class ExplainConfig:
    backend: str = "template"      # template | claude | hf | openai
    openai_model: str = "gpt-4o"
    # Claude replaces GPT-4o as the hosted model (documented deviation from the proposal)
    claude_model: str = os.environ.get("GENEX_CLAUDE_MODEL", "claude-sonnet-5-5")
    # local open-source model, as proposed; override e.g. GENEX_HF_MODEL=Qwen/Qwen2.5-7B-Instruct
    hf_model: str = os.environ.get("GENEX_HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    hf_4bit: bool = os.environ.get("GENEX_HF_4BIT", "0") == "1"
    n_students: int = 100           # proposal: 100 explanations per dataset
    prompt_mode: str = "grounded"   # grounded (KT model classifies, LLM explains) | free
    use_shap: bool = True           # add SHAP evidence (which practised concepts drive a gap)
    shap_nsamples: int = 256
    gap_threshold: float = 0.6      # mastery below this counts as a gap
    mastered_threshold: float = 0.75
    min_attempts: int = 3


def save_config(path, **cfgs):
    with open(path, "w") as f:
        json.dump({k: asdict(v) for k, v in cfgs.items()}, f, indent=2)


def get_device(name="auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
