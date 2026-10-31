"""GENEX-KT demo app (Gradio).

    pip install gradio plotly
    python app.py                      # opens http://127.0.0.1:7860

Pick a student from ASSISTments 2009 (or upload your own answer history) and the
trained GENEX-KT backbone produces, live:
  * a mastery profile with uncertainty for every concept practised,
  * a 3-part diagnosis written by the chosen explainer, with its faithfulness check,
  * a personalised learning plan, shown next to the rule-based baseline.

Explainers: "template" (offline, always available), "claude" (needs
ANTHROPIC_API_KEY), "openai" (needs OPENAI_API_KEY), "hf" (local open-source LLM;
set GENEX_ENABLE_HF=1, needs a GPU). On Hugging Face Spaces, add API keys as
Space secrets, never in code.
"""
import os

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from genex.config import OUT_DIR, DataConfig, ExplainConfig, get_device
from genex.data import load_processed
from genex.explain import (BACKENDS, SYSTEM_PROMPT, build_profiles, faithfulness,
                           fit_projection, prompt_for, render_text)
from genex.intervene import genex_plan, mine_prerequisites, rule_baseline
from genex.train import load_model

# status palette (good / warning / critical) - always paired with a text label
COLORS = {"Mastered": "#0ca30c", "Developing": "#fab219", "Gap": "#d03b3b"}

# --------------------------------------------------------------------------
# load once at start-up
# --------------------------------------------------------------------------
PROCESSED = DataConfig().processed_dir
_, META = load_processed(PROCESSED)
NAMES = META["skill_names"]
NAME2ID = {v.lower(): k for k, v in NAMES.items()}
DF = pd.read_csv(os.path.join(PROCESSED, "interactions.csv.gz"))
DEVICE = get_device()
MODEL = load_model(os.path.join(OUT_DIR, "genex"), META, DEVICE)
ECFG = ExplainConfig()
ANSWER_TYPES = DF.groupby("skill").answer_type.agg(lambda s: s.mode().iat[0]).to_dict()

_graph_path = os.path.join(OUT_DIR, "interventions", "prerequisite_graph.csv")
GRAPH = (pd.read_csv(_graph_path)[["pre", "post", "precedence", "lift", "support"]]
         if os.path.exists(_graph_path) else mine_prerequisites(DF))

USERS = sorted(DF.user.unique().tolist())
# default student: a long, mixed history so the first screen shows every element
_sz = DF.groupby("user").agg(n=("pos", "size"), acc=("correct", "mean"), k=("skill", "nunique"))
DEFAULT_USER = int(_sz[_sz.n.between(250, 450) & _sz.acc.between(0.6, 0.75)].k.idxmax())
_ref = np.random.default_rng(0).choice(USERS, 300, replace=False).tolist()
PCA = fit_projection(build_profiles(MODEL, DF, META, _ref, DEVICE, ECFG))

AVAILABLE = ["template"]
if os.environ.get("ANTHROPIC_API_KEY"):
    AVAILABLE.append("claude")
if os.environ.get("OPENAI_API_KEY"):
    AVAILABLE.append("openai")
if os.environ.get("GENEX_ENABLE_HF") == "1":
    AVAILABLE.append("hf")
_LLMS = {}


def llm(name):
    if name not in _LLMS:
        _LLMS[name] = BACKENDS[name](ECFG)
    return _LLMS[name]


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def band(m):
    if m >= ECFG.mastered_threshold:
        return "Mastered"
    return "Gap" if m < ECFG.gap_threshold else "Developing"


def mastery_chart(stats, max_rows=25):
    """All gaps and developing concepts, topped up with the most-practised mastered ones."""
    s = stats.copy()
    s["band"] = s.mastery.map(band)
    weak = s[s.band != "Mastered"].nsmallest(max_rows, "mastery")
    strong = s[s.band == "Mastered"].nlargest(max(0, max_rows - len(weak)), "attempts")
    s = pd.concat([weak, strong]).sort_values("mastery")
    fig = go.Figure()
    for b in ["Gap", "Developing", "Mastered"]:
        d = s[s.band == b]
        fig.add_bar(
            y=d.concept, x=d.mastery, orientation="h", name=b, marker_color=COLORS[b],
            error_x=dict(type="data", array=2 * d.unc, color="#555", thickness=1, width=3),
            customdata=np.stack([d.attempts, d.acc * 100, d.recent * 100, d.unc], -1),
            hovertemplate="<b>%{y}</b><br>Mastery %{x:.2f} (± %{customdata[3]:.2f})<br>"
                          "Attempts %{customdata[0]:.0f} · accuracy %{customdata[1]:.0f}%"
                          " · last 5: %{customdata[2]:.0f}%<extra>" + b + "</extra>")
    for x, lbl in [(ECFG.gap_threshold, "gap"), (ECFG.mastered_threshold, "mastered")]:
        fig.add_vline(x=x, line=dict(color="#888", dash="dot", width=1),
                      annotation_text=lbl, annotation_position="top")
    fig.update_layout(
        height=max(320, 26 * len(s) + 110), margin=dict(l=10, r=20, t=40, b=40),
        xaxis=dict(range=[0, 1], title="Predicted mastery (P correct on next item)",
                   gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(automargin=True), bargap=0.25, legend=dict(orientation="h", y=1.08, x=0),
        # mid-grey ink stays readable on both the light and the dark Gradio theme
        font=dict(color="#8c8c8c", size=13),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(tickfont=dict(color="#8c8c8c"), title_font=dict(color="#8c8c8c"),
                     zeroline=False, linecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(tickfont=dict(color="#8c8c8c"))
    return fig


def plan_md(p, title):
    if not p["pathway"]:
        return f"### {title}\n_No concepts recommended._\n\n**Feedback:** {p['feedback']}"
    lines = [f"### {title}", "**Learning pathway**"]
    lines += [f"{i}. {c}" + (f" — mastery {m:.2f}" if m is not None else "")
              for i, (c, m) in enumerate(zip(p["pathway"],
                                             p.get("pathway_mastery", [None] * 5)), 1)]
    lines.append("\n**Practice**")
    lines += [f"- {e['n_items']} {e['difficulty']} *{e['exercise_type']}* items on "
              f"**{e['concept']}** — {e['target_misconception']}" for e in p["exercises"]]
    lines.append(f"\n**Feedback to the student:** {p['feedback']}")
    return "\n".join(lines)


def analyse(g, backend, label):
    if g.empty:
        raise gr.Error("No interactions to analyse.")
    p = build_profiles(MODEL, g, META, [g.user.iat[0]], DEVICE, ECFG)[0]
    p["descriptor"] = PCA.transform(p["state"][None])[0]
    user_prompt, table = prompt_for(p, ECFG)
    try:
        expl = llm(backend)(SYSTEM_PROMPT, user_prompt, table)
    except Exception as e:  # keep the demo alive if an API call fails
        gr.Warning(f"{backend} explainer failed ({e}); using the offline template.")
        backend, expl = "template", llm("template")(SYSTEM_PROMPT, user_prompt, table)
    f = faithfulness(expl, table, ECFG)
    rec = {"explanation": expl, "stats": p["stats"].to_dict("records"),
           "mastery": {k: float(v) for k, v in zip(range(1, META["n_skills"] + 1),
                                                    p["mastery_all"])}}
    gp = genex_plan(rec, GRAPH, NAMES, ANSWER_TYPES, ECFG)
    bp = rule_baseline(rec, NAMES)

    st = p["stats"]
    summary = (f"### {label}\n**{p['n_int']}** answers on **{len(st)}** concepts · "
               f"overall accuracy **{p['acc']:.0%}** · last 20 answers **{p['acc_recent']:.0%}** · "
               f"**{(st.mastery >= ECFG.mastered_threshold).sum()}** mastered, "
               f"**{(st.mastery < ECFG.gap_threshold).sum()}** gaps")
    badge = ("✅ **Faithful** — every concept named agrees with the model."
             if f["pass"] and f["claims"] else
             "⚠️ **Check needed** — " + "; ".join(f["issues"]))
    diag = (f"### Diagnosis ({backend})\n{badge}\n\n"
            + render_text(expl).replace("\n  - ", "\n- ").replace("Knowledge gaps:", "\n**Knowledge gaps**")
            .replace("Demonstrated mastery:", "**Demonstrated mastery:**")
            .replace("Review first:", "\n**Review first:**"))
    table_out = st.sort_values("mastery")[["concept", "mastery", "unc", "attempts", "acc",
                                           "recent", "trend"]].round(2)
    table_out.columns = ["Concept", "Mastery", "Uncertainty", "Attempts", "Accuracy",
                         "Last 5", "Trend"]
    return (summary, mastery_chart(st), diag, plan_md(gp, "GENEX-KT plan"),
            plan_md(bp, "Rule-based baseline (accuracy < 60%)"), table_out, user_prompt)


def run_student(user_id, backend):
    if not user_id:
        raise gr.Error("Choose a student.")
    g = DF[DF.user == int(user_id)].sort_values("pos")
    return analyse(g, backend, f"Student {user_id} (ASSISTments 2009)")


def run_upload(file, backend):
    if file is None:
        raise gr.Error("Upload a CSV first.")
    up = pd.read_csv(file.name if hasattr(file, "name") else file)
    cols = {c.lower(): c for c in up.columns}
    if "skill" not in cols or "correct" not in cols:
        raise gr.Error("The CSV needs columns 'skill' (concept name or id) and 'correct' (0/1).")
    sk = up[cols["skill"]].astype(str).str.strip()
    ids = sk.map(lambda s: int(s) if s.isdigit() and int(s) in NAMES else NAME2ID.get(s.lower()))
    bad = sorted(set(sk[ids.isna()]))
    if bad:
        raise gr.Error("Unknown concepts: " + ", ".join(bad[:8]) + " — see the 'Concepts' tab.")
    g = pd.DataFrame({"user": -1, "pos": np.arange(len(up)), "skill": ids.astype(int),
                      "correct": up[cols["correct"]].astype(int).clip(0, 1)})
    edges = np.array(META["time_bin_edges_ms"])
    if "response_ms" in cols:
        rt = pd.to_numeric(up[cols["response_ms"]], errors="coerce")
        g["tbin"] = np.where(rt > 0, np.digitize(rt.fillna(-1), edges) + 1, 0)
    else:
        g["tbin"] = 0
    g["attempts"] = g.groupby("skill").cumcount().clip(upper=META["max_attempt_bucket"])
    return analyse(g, backend, f"Uploaded history ({len(g)} answers)")


def random_student():
    return str(np.random.default_rng().choice(USERS))


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
EXAMPLE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_example.csv")
if not os.path.exists(EXAMPLE_CSV):
    # a mid-length history with a mix of strong and weak concepts
    sizes = DF.groupby("user").agg(n=("pos", "size"), k=("skill", "nunique"), acc=("correct", "mean"))
    pick = sizes[(sizes.n.between(100, 160)) & (sizes.acc.between(0.55, 0.7))].k.idxmax()
    ex = DF[DF.user == pick].sort_values("pos")
    pd.DataFrame({"skill": ex.skill.map(NAMES), "correct": ex.correct}).to_csv(EXAMPLE_CSV, index=False)

with gr.Blocks(title="GENEX-KT demo") as demo:
    gr.Markdown(
        "# GENEX-KT — explainable, generative knowledge tracing\n"
        "A diffusion-based knowledge-tracing model estimates what a learner has mastered; "
        "an explainer turns that into a teacher-readable diagnosis; a planner builds a "
        "prerequisite-aware learning pathway. *MSc research prototype — Liverpool John "
        "Moores University. Data: ASSISTments 2009 (public, anonymised).*")
    with gr.Row():
        backend = gr.Dropdown(AVAILABLE, value=AVAILABLE[-1] if "claude" in AVAILABLE
                              else "template", label="Explainer", scale=1)
    with gr.Tab("Explore a student"):
        with gr.Row():
            uid = gr.Dropdown([str(u) for u in USERS], value=str(DEFAULT_USER),
                              label="Student ID", filterable=True, scale=3)
            rnd = gr.Button("🎲 Random student", scale=1)
            go_s = gr.Button("Analyse", variant="primary", scale=1)
    with gr.Tab("Upload a history"):
        gr.Markdown("CSV in time order with columns **skill** (concept name or id) and "
                    "**correct** (0/1); optional **response_ms**. "
                    "Example file: `demo_example.csv`.")
        with gr.Row():
            f_in = gr.File(label="Answer history (.csv)", file_types=[".csv"], scale=3)
            go_u = gr.Button("Analyse upload", variant="primary", scale=1)
        gr.Examples([[EXAMPLE_CSV]], inputs=[f_in])
    with gr.Tab("Concepts"):
        gr.Dataframe(pd.DataFrame({"id": list(NAMES), "concept": list(NAMES.values())}),
                     label="Concept names accepted in uploads")
    with gr.Tab("About"):
        gr.Markdown(
            "**How it works.** The interaction history is embedded and passed through a "
            "causal transformer trained as a DDPM denoiser; its hidden state is the knowledge "
            "state. Re-noising and denoising it several times gives the uncertainty bars. "
            "In the default *grounded* mode the model decides which concepts are mastered "
            "(≥ 0.75) or gaps (< 0.60) and the explainer only explains and prioritises; the "
            "faithfulness check flags any claim that disagrees with the model.\n\n"
            "**Limits.** Research prototype; not validated with real classes. Mastery for "
            "concepts with very few attempts is uncertain. Recommendations need teacher review.")

    summary = gr.Markdown()
    gr.Markdown("**Mastery profile** — all gaps and developing concepts plus the most-practised "
                "mastered ones. Hover a bar for details; whiskers show ± 2 × uncertainty.")
    chart = gr.Plot(show_label=False)
    with gr.Row():
        diag = gr.Markdown()
    with gr.Row():
        plan_g = gr.Markdown()
        plan_b = gr.Markdown()
    with gr.Accordion("Concept table", open=False):
        tbl = gr.Dataframe()
    with gr.Accordion("Transparency: exact prompt sent to the explainer", open=False):
        prompt_box = gr.Code(language=None)

    outs = [summary, chart, diag, plan_g, plan_b, tbl, prompt_box]
    go_s.click(run_student, [uid, backend], outs)
    rnd.click(random_student, None, uid).then(run_student, [uid, backend], outs)
    go_u.click(run_upload, [f_in, backend], outs)
    demo.load(run_student, [uid, backend], outs)

if __name__ == "__main__":
    # Hugging Face Spaces sets GRADIO_SERVER_NAME=0.0.0.0; locally we stay on 127.0.0.1
    demo.launch(server_name=os.environ.get("HOST", os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")),
                server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860))),
                share=os.environ.get("GENEX_SHARE") == "1")
