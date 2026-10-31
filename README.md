# GENEX-KT: Explainable and Generative Knowledge Tracing

This is the code for the LJMU MSc thesis *GENEX-KT: An Explainable and Generative Knowledge
Tracing Framework for Personalised Adaptive Learning*. The whole pipeline is
open source and runs on a free Google Colab T4 GPU. It uses no paid API.

## Run everything on Colab (`GENEX-KT_Experiments.ipynb`)

1. Runtime → Change runtime type → **T4 GPU**.
2. Upload `GENEX-KT_Code.zip` to the Files pane, then run:
   ```
   !unzip -q -o GENEX-KT_Code.zip
   %run genex_kt/colab_launch.py          # starts the full run in the background
   %run /content/status.py                # progress check (upload status.py first)
   ```
3. When the log ends with `ALL DONE`, download `GENEX-KT_Results_<date>.zip` from
   `/content/genex_kt/`. A second zip, `GENEX-KT_Results_<date>_models.zip`, holds the
   trained checkpoints.

The run is resumable. After a disconnect, run `colab_launch.py` again and every step
whose output already exists is skipped.

**Optional LLaMA 3:** accept Meta's licence for `meta-llama/Meta-Llama-3-8B-Instruct`
on Hugging Face, then add your token as the Colab secret `HF_TOKEN`. The launcher then
uses LLaMA-3 + Qwen2.5. Without the token it uses Qwen2.5 + Phi-3.5.

**Second session / resume:** `%run genex_kt/colab_launch_resume.py` runs `colab_resume.sh`
(EdNet + the Rasch ablation, one job at a time) and `colab_followup.sh` finishes the remaining
EdNet explanations. `python make_mini.py` packs a ~100 KB copy of every result the thesis uses.
Run one job at a time: parallel jobs exhausted the Colab VM's RAM in the first session.

**Exploratory ablation:** `python ablation_rasch.py oulad assist2009` trains GENEX-KT + Rasch
item difficulty (`genex_rasch`) with each dataset's tuned hyper-parameters.

## Final results (test AUC)

| Dataset | BKT | DKT | SAKT | AKT | GENEX-KT | GENEX-KT + Rasch | GENEX-KT v2 |
|---|---|---|---|---|---|---|---|
| ASSISTments 2009–10 | 0.726 | 0.751 | 0.707 | **0.775** | 0.746 | 0.767 | 0.734 |
| EdNet-KT1 (100K sample) | 0.623 | 0.672 | 0.665 | **0.767** | 0.675 | 0.762 | 0.673 |
| OULAD | 0.790 | 0.808 | 0.806 | **0.846** | 0.813 | 0.814 | 0.745 |

Main run, seed 42. Over three seeds (42, 43, 44), GENEX-KT + Rasch reaches 0.765 ± 0.002 on
ASSISTments and 0.821 ± 0.007 on OULAD, against 0.744 ± 0.002 and 0.790 ± 0.022 for the base
GENEX-KT. The base model is unstable on OULAD. Grounded Qwen2.5-7B diagnoses are fully faithful
for 70–91% of students (69–94% with the Rasch backbone). Every number in the thesis comes from
the files in `results/`.

### Accuracy extensions and repeated runs

| Script | What it does |
|---|---|
| `colab_full.sh`, `colab_launch_all.py` | Final sequential plan: Rasch and v2 tuning, EdNet variants, explanations on the best variant, EdNet DeLong baselines, extra seeds |
| `train_set.py --seed 43` | Retrains a model set with another seed (written to `outputs/<ds>_seed43/`) |
| `explain_best.py` | Reruns explanations, SHAP and pathways with the variant of highest validation AUC (`outputs/<ds>_best/`) |
| `ensemble.py` | Averages the GENEX-KT variants and runs DeLong tests against the best member and AKT |
| `make_mini.py` | Packs every result the thesis uses into a ~160 KB archive |

Long Colab runs should save progress to Google Drive, because free sessions reset. Each stage
skips work whose output already exists, so a reset run can simply be restarted.

## Repository layout

- `genex/`: the package (data, models, training, tuning, explanation, interventions, evaluation)
- `results/`: compact results of the final run (metrics, DeLong tests, explanation and pathway
  summaries, EDA, tuned hyper-parameters), including `<ds>_seed43/44` and `<ds>_best`
- Raw data, processed data and model checkpoints are not included. `genex/download.py` fetches
  the public datasets.

## Pipeline: phase → file

| Phase | Files | Output (`outputs/<dataset>/…`) |
|---|---|---|
| 1 Data | `genex/download.py`, `genex/data.py`, `genex/eda.py` | `data/processed/<ds>/`, `eda/` |
| 2 Models | `genex/models.py` (GENEX-KT diffusion backbone, pyKT DKT/SAKT/AKT), `genex/bkt.py`, `genex/train.py`, `genex/tune.py` (Optuna), `genex/word2vec.py` | `<model>/metrics.json`, `tuning/` |
| 3 Explanations | `genex/shap_attr.py` (concept-level Kernel SHAP), `genex/explain.py` (grounded/free prompts, faithfulness check) | `explanations/` |
| 4 Interventions | `genex/intervene.py` (prerequisite DAG, 5-concept pathway, rule baseline, validation) | `interventions/` |
| Evaluation | `genex/evaluate.py` (DeLong, SHAP summary, expert sheets, Krippendorff α, Wilcoxon, Kendall W) | `report/`, `outputs/cross_dataset/` |
| Orchestration | `run_all.py`, `colab_pipeline.sh`, `colab_launch.py`, `status.py` | `GENEX-KT_Results_<date>.zip` |
| Demo | `app.py` (Gradio) | web app |

The datasets are ASSISTments 2009–2010, EdNet-KT1 (a 100K-student sample) and OULAD. Each
dataset is modelled independently.

## Open-source models

| Role | Model | Licence |
|---|---|---|
| Explanation + feedback | Qwen/Qwen2.5-7B-Instruct (4-bit) | Apache 2.0 |
| Explanation + feedback | microsoft/Phi-3.5-mini-instruct (4-bit) | MIT |
| Optional | meta-llama/Meta-Llama-3-8B-Instruct | Llama 3 Community Licence |
| Concept embeddings | word2vec GoogleNews-300 (gensim) | Apache 2.0 |
| Baselines | pyKT DKT / SAKT / AKT | MIT |

## Local commands

```bash
pip install -r requirements.txt
python -m genex.download assist2009
python run_all.py --datasets assist2009 --explainers template   # CPU, no LLM
python run_all.py --datasets assist2009 --explainers qwen25     # GPU
python -m genex.evaluate cross                                  # cross-dataset report
python app.py                                                   # demo at http://127.0.0.1:7860
```

## Expert evaluation

`outputs/<ds>/report/expert_sheet_explanations.csv` holds 100 blinded explanations per dataset.
`outputs/<ds>/interventions/expert_sheet_interventions_<llm>.csv` holds blinded A/B plans.
The `*_KEY.csv` files hold the answer keys; keep them away from the raters. Once the
sheets are filled in, analyse them with:

```bash
python -m genex.evaluate ratings --explanations rater1.csv rater2.csv rater3.csv
python -m genex.evaluate ratings --interventions r1.csv r2.csv r3.csv \
    --key outputs/assist2009/interventions/expert_sheet_interventions_qwen25_KEY.csv
```
