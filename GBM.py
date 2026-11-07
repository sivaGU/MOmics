import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from docs import OVERVIEW, GUI_GUIDE, MODEL_ARCH

#path to the files required 
_HERE = Path(__file__).parent
_DATA = _HERE / "data" / "discovery"
_MODEL = _HERE / "models" / "MOmics_v11_locked_pipeline.pkl"

# Page Configuration 
st.set_page_config(page_title="MOmics", layout="wide", page_icon="🧬")

# CSS for the website
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #001f3f;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
        font-size: 12px;
    }
    header[data-testid="stHeader"] {
        background-color: #5dade2;
    }
    .stButton > button {
        background-color: #5dade2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 12px;
    }
    .stButton > button:hover { background-color: #3498db; }
    .stDownloadButton > button {
        background-color: #5dade2;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: 500;
        font-size: 12px;
    }
    .stDownloadButton > button:hover { background-color: #3498db; }
    [data-testid="stSidebar"] .stRadio > label { color: #ffffff; font-size: 12px; }
    .demo-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #5dade2;
        margin: 10px 0;
        font-size: 12px;
    }
    .demo-success { background-color: #d5f4e6; border-left-color: #27ae60; }
    .demo-warning { background-color: #fff3cd; border-left-color: #f39c12; }
    h2, h3, h4,
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3,
    [data-testid="stHeadingWithActionElements"] h4,
    .stSubheader { font-size: 16px !important; }
    p, li, label, .stMarkdown p, .stMarkdown li,
    [data-testid="stText"], [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"], .stDataFrame,
    .stSelectbox label, .stFileUploader label,
    .stNumberInput label, .stRadio label,
    .stExpander summary, .stAlert p,
    div[data-testid="stInfoBox"] p,
    div[data-testid="stSuccessBox"] p,
    div[data-testid="stWarningBox"] p,
    div[data-testid="stErrorBox"] p,
    .stTabs [data-baseweb="tab"] { font-size: 12px !important; }
    </style>
""", unsafe_allow_html=True)


# Loading pipline for the model

@st.cache_resource
def load_pipeline():
    return joblib.load(_MODEL)

pipe = load_pipeline()

PRUNED         = pipe["pruned_features"]
SUB_MODELS     = pipe["sub_models"]
FUSION_MODEL   = pipe["fusion_model"]
ZSCORE_PARAMS  = pipe["zscore_params"]
THRESHOLD      = pipe["youden_threshold"]
RNA_LOG1P      = pipe["rna_log1p_required"]
CALIBRATOR     = pipe["isotonic_calibrator"]
DISCOVERY      = pipe["discovery_metrics"]
SYMBOL_TO_ENSG = pipe["symbol_to_ensg"]

RNA_FEATURES  = PRUNED["rna"]
PROT_FEATURES = PRUNED["prot"]
MET_FEATURES  = PRUNED["met"]
ALL_FEATURES  = RNA_FEATURES + PROT_FEATURES + MET_FEATURES

QC_COLS = ["Features Used", "RNA Features", "Protein Features", "Metabolomics Features",
           "QC Status", "QC Notes"]
SCORE_COLS = ["Prediction", "Risk Score (Raw)", "Risk Score (Calibrated)",
              "Binary Call", "RNA Score", "Protein Score", "Metabolomics Score",
              "Sample ID"] + QC_COLS

LAYER_FEATURES = {"rna": RNA_FEATURES, "prot": PROT_FEATURES, "met": MET_FEATURES}
LAYER_NAMES = {"rna": "RNA", "prot": "Protein", "met": "Metabolomics"}
# Symbols present in more than one layer (e.g. BSN, PCLO, PTPRT, CIT)
SHARED_SYMBOLS = sorted(set(RNA_FEATURES) & set(PROT_FEATURES)
                        | set(RNA_FEATURES) & set(MET_FEATURES)
                        | set(PROT_FEATURES) & set(MET_FEATURES))

RUO_NOTICE = (
    "**Research use only.** Not for diagnostic or clinical decision-making. Scores come from a "
    "model trained on CPTAC-3 GBM and GTEx normal brain data and are only interpretable for "
    "inputs generated and processed the same way."
)
THRESHOLD_CAPTION = (
    f"Model call = GBM-like when the raw fusion score is ≥ {THRESHOLD:.4f} (Youden threshold). "
    "The calibrated probability is isotonic-mapped to the external validation cohort "
    "(~4.6% GBM prevalence) and is not a risk estimate for an individual. Research use only."
)

# Indices the full 109-sample used for the demo for condensed and better representation
DEMO_INDICES = [0, 13, 20, 104]



# The list of features the model relies on

def _build_importance_df():
    rows = []
    layer_map = {"rna": "RNA", "prot": "Protein", "met": "Metabolomics"}
    for layer_key, label in layer_map.items():
        feats = PRUNED[layer_key]
        model = SUB_MODELS[layer_key]
        for feat, imp in zip(feats, model.feature_importances_):
            rows.append({"Biomarker": f"{feat} ({label})", "Influence Score": float(imp), "Layer": label})
    return pd.DataFrame(rows).sort_values("Influence Score", ascending=False).reset_index(drop=True)

importance_df_display = _build_importance_df()

# Inference pipeline

def _zscore(value, mean, std):
    if std is None or np.isnan(std) or std == 0:
        return 0.0
    if mean is None or np.isnan(mean):
        return 0.0
    return (value - mean) / std


def _prepare_layer(layer_key, user_dict):
    """Z-score a layer dict. RNA gets log1p first. Returns (1, n) array or None."""
    if user_dict is None:
        return None
    features = PRUNED[layer_key]
    means    = ZSCORE_PARAMS[layer_key]["mean"]
    stds     = ZSCORE_PARAMS[layer_key]["std"]
    row = []
    for f in features:
        raw = user_dict.get(f, None)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            row.append(0.0)
        else:
            val = np.log1p(float(raw)) if (layer_key == "rna" and RNA_LOG1P) else float(raw)
            row.append(_zscore(val, means.get(f), stds.get(f)))
    return np.array(row, dtype=float).reshape(1, -1)


def score_sample(rna_dict, prot_dict, met_dict):
    """3-layer z-score → sub-models → fusion → isotonic calibration.
    None dict = layer not provided → NaN passed to fusion (XGBoost handles natively).
    Dict with some/all None values = layer provided, blanks filled with training mean (z=0).
    """
    layer_probs  = []
    layer_scores = {}
    for layer_key, user_dict, label in [
        ("rna",  rna_dict,  "RNA Score"),
        ("prot", prot_dict, "Protein Score"),
        ("met",  met_dict,  "Metabolomics Score"),
    ]:
        # Treat a dict where every value is None as a missing layer
        if user_dict is None or all(v is None for v in user_dict.values()):
            layer_probs.append(np.nan)
            layer_scores[label] = None
        else:
            arr = _prepare_layer(layer_key, user_dict)
            p = float(SUB_MODELS[layer_key].predict_proba(arr)[:, 1][0])
            layer_probs.append(p)
            layer_scores[label] = p

    fusion_input = np.array([layer_probs])
    raw_prob     = float(FUSION_MODEL.predict_proba(fusion_input)[:, 1][0])
    cal_prob     = float(CALIBRATOR.predict([raw_prob])[0])
    binary       = "GBM" if raw_prob >= THRESHOLD else "Non-GBM"

    return {
        "Prediction":              "High Risk" if raw_prob >= THRESHOLD else "Low Risk",
        "Risk Score (Raw)":        raw_prob,
        "Risk Score (Calibrated)": cal_prob,
        "Binary Call":             binary,
        **layer_scores,
    }


def map_upload_columns(columns):
    """Map uploaded columns to (layer, symbol). '<layer>_<SYMBOL>' always works;
    a bare symbol only if it belongs to one layer. Returns mapping + rejected columns."""
    mapping, ambiguous, unrecognized = {}, [], []
    for c in columns:
        name = str(c).strip()
        target = None
        for key, feats in LAYER_FEATURES.items():
            if name.startswith(f"{key}_") and name[len(key) + 1:] in feats:
                target = (key, name[len(key) + 1:])
                break
        if target is None:
            if name in SHARED_SYMBOLS:
                ambiguous.append(name)
                continue
            layers = [k for k, feats in LAYER_FEATURES.items() if name in feats]
            if not layers:
                unrecognized.append(name)
                continue
            target = (layers[0], name)
        if target not in mapping.values():
            mapping[c] = target
    return mapping, ambiguous, unrecognized


def process_dataframe(df, mapping=None):
    """Score a user-uploaded DataFrame (layer-prefixed or unambiguous bare columns)."""
    if mapping is None:
        mapping, _, _ = map_upload_columns(df.columns)
    numeric = df[list(mapping)].apply(pd.to_numeric, errors="coerce")

    results = []
    for i in range(len(df)):
        dicts = {"rna": {}, "prot": {}, "met": {}}
        for col, (key, f) in mapping.items():
            v = numeric.iloc[i][col]
            if pd.notna(v):
                dicts[key][f] = float(v)
        layers = {k: (d if d else None) for k, d in dicts.items()}

        result = score_sample(layers["rna"], layers["prot"], layers["met"])
        result.update(input_qc(layers["rna"], layers["prot"], layers["met"]))
        for key in ("rna", "prot", "met"):
            for f in LAYER_FEATURES[key]:
                result[f"{key}_{f}"] = dicts[key].get(f, np.nan)
        results.append(result)

    return pd.DataFrame(results)



# Input QC (front end): report what was actually supplied for every prediction

def _is_blank(v):
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def input_qc(rna_dict, prot_dict, met_dict):
    """Count supplied features per layer and build QC fields shown with each prediction.
    Mirrors score_sample's rules: a None / all-blank layer is 'not provided';
    blanks inside a provided layer are filled with the training mean (z = 0)."""
    used_total, notes, status = 0, [], "PASS"
    out = {}
    for key, d in (("rna", rna_dict), ("prot", prot_dict), ("met", met_dict)):
        feats, name = LAYER_FEATURES[key], LAYER_NAMES[key]
        n = len(feats)
        k = 0 if d is None else sum(not _is_blank(d.get(f)) for f in feats)
        out[f"{name} Features"] = f"{k}/{n}"
        used_total += k
        if k == 0:
            notes.append(f"{name} layer not provided; scored without it.")
            status = "REVIEW"
        elif k < n:
            notes.append(f"{n - k}/{n} {name} features missing; filled with the training mean (z = 0).")
            status = "REVIEW"
    if used_total == 0:
        notes.append("No features supplied — this score reflects the training mean only. Do not interpret.")
        status = "FAIL"
    elif used_total < 0.5 * len(ALL_FEATURES):
        notes.append("Fewer than half of the panel features were supplied; interpret with caution.")
        status = "FAIL"
    out.update({"Features Used": f"{used_total}/{len(ALL_FEATURES)}",
                "QC Status": status, "QC Notes": " | ".join(notes)})
    return out


def render_qc(row):
    head = (f"**QC: {row['QC Status']}** · Features used: **{row['Features Used']}** "
            f"(RNA {row['RNA Features']} · Protein {row['Protein Features']} · "
            f"Metabolomics {row['Metabolomics Features']})")
    notes = [n for n in str(row.get("QC Notes", "")).split(" | ") if n]
    body = head + ("\n\n" + "\n".join(f"- {n}" for n in notes) if notes else "")
    {"PASS": st.success, "REVIEW": st.warning}.get(row["QC Status"], st.error)(body)


def _call_label(binary):
    return "GBM-like" if binary == "GBM" else "Non-GBM-like"


# Visualizations


def _fmt(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.4f}"


def render_risk_charts(results, mode="manual", key_prefix=""):
    st.subheader("Model Output (research use only)")
    call_colors = {"GBM-like": "#EF553B", "Non-GBM-like": "#00CC96"}

    if mode == "manual":
        row = results.iloc[0]
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Model Call",             _call_label(row["Binary Call"]))
        c2.metric("Raw Fusion Score",       f"{row['Risk Score (Raw)']:.4f}")
        c3.metric("Calibrated Probability", f"{row['Risk Score (Calibrated)']:.4f}")
        c4.metric("RNA Score",              _fmt(row["RNA Score"]))
        c5.metric("Protein Score",          _fmt(row["Protein Score"]))
        c6.metric("Metabolomics Score",     _fmt(row["Metabolomics Score"]))
        c7.metric("Features Used",          row["Features Used"])
        render_qc(row)
        st.caption(THRESHOLD_CAPTION)

    else:
        plot_df = results.copy()
        plot_df["Model Call"] = plot_df["Binary Call"].map(_call_label)
        plot_df["Sample"] = (plot_df["Sample ID"].astype(str) if "Sample ID" in plot_df
                             else plot_df.index.astype(str))

        q = plot_df["QC Status"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("QC PASS",   int(q.get("PASS", 0)))
        c2.metric("QC REVIEW", int(q.get("REVIEW", 0)))
        c3.metric("QC FAIL",   int(q.get("FAIL", 0)))
        if q.get("REVIEW", 0) or q.get("FAIL", 0):
            st.warning("Some samples have missing features or layers. Check the Features Used and "
                       "QC Notes columns before interpreting their scores.")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(plot_df, x="Risk Score (Raw)", color="Model Call",
                               title="Raw Fusion Score Distribution",
                               color_discrete_map=call_colors, nbins=20)
            fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="gray",
                          annotation_text=f"Youden ({THRESHOLD:.3f})")
            fig.update_layout(xaxis_title="Raw Score", yaxis_title="Number of Samples")
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_hist")
        with col2:
            rs = plot_df.sort_values("Risk Score (Raw)", ascending=False)
            fig2 = px.bar(rs, x="Sample", y="Risk Score (Raw)", color="Model Call",
                          title="Raw Fusion Score by Sample", color_discrete_map=call_colors,
                          hover_data=["Features Used", "QC Status"])
            fig2.add_hline(y=THRESHOLD, line_dash="dash", line_color="gray",
                           annotation_text=f"Threshold ({THRESHOLD:.3f})")
            fig2.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_bar")

        st.divider()
        st.subheader("Results Table")
        display = plot_df[["Sample", "Model Call", "Risk Score (Raw)", "Risk Score (Calibrated)",
                           "RNA Score", "Protein Score", "Metabolomics Score",
                           "Features Used", "QC Status", "QC Notes"]].copy()
        for col in ["Risk Score (Raw)", "Risk Score (Calibrated)",
                    "RNA Score", "Protein Score", "Metabolomics Score"]:
            display[col] = display[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) and x is not None else "N/A"
            )
        st.dataframe(display, use_container_width=True, hide_index=True)
        out = results.copy()
        out.insert(0, "Intended Use", "Research use only - not for diagnostic use")
        st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="MOmics_results_research_use_only.csv", mime="text/csv",
                           key=f"{key_prefix}_dl")
        st.caption(THRESHOLD_CAPTION)


def render_dashboard(results, mode="manual", key_prefix="", patient_labels=None):
    render_risk_charts(results, mode=mode, key_prefix=key_prefix)

    if mode == "bulk":
        st.divider()
        st.subheader("Cohort Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        total = len(results)
        gbm_n = (results["Binary Call"] == "GBM").sum()
        c1.metric("Total Samples",   total)
        c2.metric("GBM-like Calls",  f"{gbm_n} ({gbm_n/total:.1%})")
        c3.metric("Mean Raw Score",  f"{results['Risk Score (Raw)'].mean():.4f}")
        c4.metric("Median Raw Score",f"{results['Risk Score (Raw)'].median():.4f}")

    st.divider()
    st.subheader("Individual Sample Analysis")
    fmt = (lambda i: f"Sample {i} — {patient_labels[i]}"
           if patient_labels and i < len(patient_labels) else f"Sample {i}")
    selected = st.selectbox("Select Sample", results.index,
                            format_func=fmt, key=f"{key_prefix}_select")
    row = results.iloc[selected]

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Model Call",             _call_label(row["Binary Call"]))
    c2.metric("Raw Score",              f"{row['Risk Score (Raw)']:.4f}")
    c3.metric("Calibrated",             f"{row['Risk Score (Calibrated)']:.4f}")
    c4.metric("RNA Score",              _fmt(row["RNA Score"]))
    c5.metric("Protein Score",          _fmt(row["Protein Score"]))
    c6.metric("Metabolomics Score",     _fmt(row["Metabolomics Score"]))
    c7.metric("Features Used",          row["Features Used"])
    render_qc(row)

    st.divider()
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.write("### Multi-Modal Signature")
        r_vals = [
            row["Protein Score"]      if pd.notna(row["Protein Score"])      else 0,
            row["RNA Score"]          if pd.notna(row["RNA Score"])          else 0,
            row["Metabolomics Score"] if pd.notna(row["Metabolomics Score"]) else 0,
        ]
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=r_vals, theta=["Proteins", "RNA", "Metabolites"],
            fill="toself", line_color="#5dade2"
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}_radar_{selected}")

    with col_r:
        st.write(f"### Biomarker Input Values (Sample {selected})")
        marker_cols = [c for c in results.columns if c not in SCORE_COLS]
        marker_vals = row[marker_cols].astype(float).dropna().sort_values(ascending=False)
        fig_top = px.bar(x=marker_vals.values, y=marker_vals.index, orientation="h",
                         color=marker_vals.values, color_continuous_scale="Viridis",
                         labels={"x": "Raw Value", "y": "Biomarker"})
        st.plotly_chart(fig_top, use_container_width=True, key=f"{key_prefix}_pbar_{selected}")

    st.divider()
    col_imp1, col_imp2 = st.columns(2)

    with col_imp1:
        st.write("#### Sample Biomarker Values")
        patient_vals = row[marker_cols].astype(float).dropna().sort_values(ascending=False)
        fig_pt = px.bar(x=patient_vals.values, y=patient_vals.index, orientation="h",
                        color=patient_vals.values, color_continuous_scale="Viridis",
                        title=f"Input Values — Sample {selected}")
        fig_pt.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_pt, use_container_width=True, key=f"{key_prefix}_ptop_{selected}")

    with col_imp2:
        st.write("#### Global XGBoost Importance")
        fig_gi = px.bar(importance_df_display,
                        x="Influence Score", y="Biomarker", color="Layer",
                        orientation="h", title="Feature Importance by Layer",
                        color_discrete_map={"RNA": "#5dade2", "Protein": "#e74c3c",
                                            "Metabolomics": "#27ae60"})
        fig_gi.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_gi, use_container_width=True, key=f"{key_prefix}_gimp_{selected}")

    with st.expander("View All Biomarker Values for This Sample"):
        all_markers = row[marker_cols].to_frame(name="Value").reset_index()
        all_markers.columns = ["Biomarker", "Value"]
        all_markers = all_markers.sort_values("Value", ascending=False)
        st.dataframe(all_markers, use_container_width=True, hide_index=True)

# Reference table
def build_reference_table():
    rows = []
    layer_label = {"rna": "RNA", "prot": "Protein", "met": "Metabolomics"}
    for layer_key, label in layer_label.items():
        feats = PRUNED[layer_key]
        means = ZSCORE_PARAMS[layer_key]["mean"]
        stds  = ZSCORE_PARAMS[layer_key]["std"]
        imps  = dict(zip(feats, SUB_MODELS[layer_key].feature_importances_))
        for f in feats:
            m = means.get(f, np.nan)
            s = stds.get(f, np.nan)
            rows.append({
                "Layer":          label,
                "Symbol":         f,
                "ENSG":           SYMBOL_TO_ENSG.get(f, "—"),
                "Training Mean":  round(m, 4) if not np.isnan(m) else "N/A",
                "Training Std":   round(s, 4) if not np.isnan(s) else "N/A",
                "XGB Importance": round(imps.get(f, 0.0), 4),
            })
    return pd.DataFrame(rows)



# Training files for demo section 
# Full cohort of the  99 GBM (CPTAC-3) + 10 GTEx normal brain = 109 samples.
# The demo walkthrough displays only DEMO_INDICES = [0, 13, 20, 104] because storage reasons.
# RNA and protein features overlap in symbol (BSN, PCLO, PTPRT, CIT) so
# they are stored in rna_/prot_/met_ prefixed columns to prevent mixing.

@st.cache_data
def load_demo_data():
    try:
        rna_raw  = pd.read_csv(_DATA / "rnaseq_washu_readcount.v4.0.tsv",
                               sep="\t").set_index("gene_name")
        prot_raw = pd.read_csv(_DATA / "proteome_mssm_per_gene_imputed.v4.0.tsv",
                               sep="\t").set_index("gene")
        met_raw  = pd.read_csv(_DATA / "metabolome_pnnl.v4.0.tsv",
                               sep="\t").set_index("Metabolite")
        manifest = pd.read_csv(_DATA / "all_subtypes.v5.1.tsv", sep="\t")
    except FileNotFoundError as e:
        st.error(f"Demo TSV not found: {e}. Ensure all TSV files are in the repo root.")
        st.stop()

    rna_cols  = set(c for c in rna_raw.columns  if c.startswith(("C3", "PT-")))
    prot_cols = set(c for c in prot_raw.columns if c.startswith(("C3", "PT-")))
    met_cols  = set(c for c in met_raw.columns  if c.startswith(("C3", "PT-")))
    label_map = dict(zip(manifest["case"], manifest["sample_type"]))

    all_samples = sorted(rna_cols | prot_cols | met_cols)
    rows = []
    for s in all_samples:
        true_label = "GBM" if label_map.get(s, "") == "tumor" else "Normal"
        row = {"Sample_ID": s, "True_Label": true_label}
        for f in RNA_FEATURES:
            row[f"rna_{f}"]  = float(rna_raw.loc[f, s])  if (s in rna_cols  and f in rna_raw.index)  else np.nan
        for f in PROT_FEATURES:
            row[f"prot_{f}"] = float(prot_raw.loc[f, s]) if (s in prot_cols and f in prot_raw.index) else np.nan
        for f in MET_FEATURES:
            row[f"met_{f}"]  = float(met_raw.loc[f, s])  if (s in met_cols  and f in met_raw.index)  else np.nan
        rows.append(row)

    df          = pd.DataFrame(rows)
    sample_ids  = df["Sample_ID"].tolist()
    true_labels = df["True_Label"].tolist()
    df_data     = df.drop(columns=["Sample_ID", "True_Label"])
    return df_data, sample_ids, true_labels


def _filter_demo(df_data, sample_ids, true_labels):
    """Slice full cohort to DEMO_INDICES only, reset index."""
    idx = DEMO_INDICES
    df_out      = df_data.iloc[idx].reset_index(drop=True)
    ids_out     = [sample_ids[i]   for i in idx]
    labels_out  = [true_labels[i]  for i in idx]
    return df_out, ids_out, labels_out


def process_demo_dataframe(df):
    """
    Score the demo DataFrame. Reads rna_/prot_/met_ prefixed columns so that
    overlapping gene symbols route to the correct layer. Missing layers pass
    None to the fusion model — not imputed.
    """
    results = []
    for _, row in df.iterrows():
        rna_dict  = {f: row[f"rna_{f}"]  for f in RNA_FEATURES  if pd.notna(row.get(f"rna_{f}"))}
        prot_dict = {f: row[f"prot_{f}"] for f in PROT_FEATURES if pd.notna(row.get(f"prot_{f}"))}
        met_dict  = {f: row[f"met_{f}"]  for f in MET_FEATURES  if pd.notna(row.get(f"met_{f}"))}

        rna_dict, prot_dict, met_dict = (rna_dict or None), (prot_dict or None), (met_dict or None)
        result = score_sample(rna_dict, prot_dict, met_dict)
        result.update(input_qc(rna_dict, prot_dict, met_dict))
        for f in RNA_FEATURES:
            result[f"rna_{f}"]  = row.get(f"rna_{f}",  np.nan)
        for f in PROT_FEATURES:
            result[f"prot_{f}"] = row.get(f"prot_{f}", np.nan)
        for f in MET_FEATURES:
            result[f"met_{f}"]  = row.get(f"met_{f}",  np.nan)
        results.append(result)

    return pd.DataFrame(results)



# Sidebar

st.sidebar.title("MOmics")
st.sidebar.markdown("**Research use only**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Home", "Documentation", "User Analysis", "Demo Walkthrough"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"")
st.sidebar.markdown(f"RNA features: {len(RNA_FEATURES)}")
st.sidebar.markdown(f"Protein features: {len(PROT_FEATURES)}")
st.sidebar.markdown(f"Metabolomics features: {len(MET_FEATURES)}")
st.sidebar.markdown(f"")
st.sidebar.markdown(f"")

st.title("MOmics | GBM Multi-Omics Research Tool")
st.warning(RUO_NOTICE, icon="⚠️")



# Home page

if page == "Home":
    try:
        from PIL import Image
        logo = Image.open(_HERE / "logo.png")
        st.image(logo, use_container_width=True)
    except Exception:
        st.info("Logo image not found. Please ensure 'logo.png' is in the root directory.")
    st.markdown("<h1 style='text-align: center;'>MOmics</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>GBM Multi-Omics Research Tool</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
**Intended use.** MOmics scores RNA, protein and metabolite profiles with a model trained on
CPTAC-3 GBM tumours and GTEx normal brain. It returns model scores for research purposes, not
diagnoses, and has not been validated for clinical use.

- Inputs must be generated and processed like the CPTAC-3 discovery data (RNA raw read counts;
  protein and metabolite log2 abundance). Values from other platforms or normalisations are
  z-scored against CPTAC-3 parameters and may produce meaningless scores.
- Missing features are filled with the training mean. Every prediction reports how many of the
  {len(ALL_FEATURES)} panel features were actually supplied, and is flagged when any are missing.
""")
    st.markdown("---")
    st.markdown('<div style="font-size: 28px; font-weight: 600; margin-bottom: 10px;">Contact Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size: 20px;"><b>For Contact:</b> Dr. Sivanesan Dakshanamurthy, PhD, MBA. '
        '<a href="mailto:sivanesan@innsciteai.com">sivanesan@innsciteai.com</a>; '
        '<a href="mailto:sd233@georgetown.edu">sd233@georgetown.edu</a></div>',
        unsafe_allow_html=True
    )



# Documentation

elif page == "Documentation":
    st.header("System Documentation")
    doc_tabs = st.tabs(["Overview", "GUI User Guide", "Model Architecture", "Feature Reference"])
    with doc_tabs[0]:
        st.markdown(OVERVIEW)
    with doc_tabs[1]:
        st.markdown(GUI_GUIDE)
    with doc_tabs[2]:
        st.markdown(MODEL_ARCH)
    with doc_tabs[3]:
        st.subheader("Feature Reference")
        st.write(
            "All v11 panel features with frozen z-score parameters used during inference. "
            "RNA inputs are log1p-transformed before z-scoring. "
            "Features with NaN std were dropped during training and are filled with 0."
        )
        ref_df = build_reference_table()
        layer_filter = st.selectbox("Filter by layer", ["All", "RNA", "Protein", "Metabolomics"])
        if layer_filter != "All":
            ref_df = ref_df[ref_df["Layer"] == layer_filter]
        st.dataframe(ref_df, use_container_width=True, hide_index=True)


# User analysis/main page

elif page == "User Analysis":
    st.header("User Analysis")
    analysis_tabs = st.tabs(["Manual Entry", "Bulk Data Upload", "Synthetic Example"])

    # Manual entry section
    with analysis_tabs[0]:
        st.subheader("Manual Sample Entry")
        st.info(
            "**RNA:** raw read counts (log1p applied internally; do not pre-normalize). "
            "**Protein / Metabolomics:** log2 abundance as in CPTAC-3. "
            "Leave a field **blank** if you don't have it (enter 0 only for a true zero). "
            "Blanks are filled with the training mean and reported with the result. "
            "Uncheck a layer to exclude it entirely."
        )

        user_inputs = {"rna": {}, "prot": {}, "met": {}}
        col_tog1, col_tog2, col_tog3 = st.columns(3)
        rna_enabled  = col_tog1.checkbox("Include RNA layer",          value=True, key="tog_rna")
        prot_enabled = col_tog2.checkbox("Include Protein layer",      value=True, key="tog_prot")
        met_enabled  = col_tog3.checkbox("Include Metabolomics layer", value=True, key="tog_met")

        st.write("#### RNA Biomarkers (raw read counts)")
        st.caption("Leave blank (None) to fill with training mean. Enter 0 only if the gene genuinely has zero counts.")
        rna_cols = st.columns(3)
        for i, feat in enumerate(RNA_FEATURES):
            with rna_cols[i % 3]:
                user_inputs["rna"][feat] = st.number_input(
                    feat, value=None, placeholder="blank = missing",
                    key=f"rna_{feat}",
                    help=f"ENSG: {SYMBOL_TO_ENSG.get(feat, '—')}. Raw read count.",
                    disabled=not rna_enabled
                )

        st.write("#### Protein Biomarkers (log2 abundance)")
        st.caption("Leave blank (None) to fill with training mean.")
        prot_cols = st.columns(4)
        for i, feat in enumerate(PROT_FEATURES):
            with prot_cols[i % 4]:
                user_inputs["prot"][feat] = st.number_input(
                    feat, value=None, placeholder="blank = missing",
                    key=f"prot_{feat}",
                    help=f"ENSG: {SYMBOL_TO_ENSG.get(feat, '—')}. Log2 CPTAC abundance.",
                    disabled=not prot_enabled
                )

        st.write("#### Metabolomics Biomarkers (log2 abundance)")
        st.caption("Leave blank (None) to fill with training mean.")
        met_cols = st.columns(3)
        for i, feat in enumerate(MET_FEATURES):
            with met_cols[i % 3]:
                user_inputs["met"][feat] = st.number_input(
                    feat, value=None, placeholder="blank = missing",
                    key=f"met_{feat}",
                    disabled=not met_enabled
                )

        enabled = {"rna": rna_enabled, "prot": prot_enabled, "met": met_enabled}
        supplied = {k: (sum(v is not None for v in user_inputs[k].values()) if enabled[k] else 0)
                    for k in user_inputs}
        st.caption("Supplied: " + " · ".join(
            f"{LAYER_NAMES[k]} {supplied[k]}/{len(LAYER_FEATURES[k])}" for k in user_inputs)
            + f" · Total {sum(supplied.values())}/{len(ALL_FEATURES)}")

        if st.button("Score Sample", key="btn_manual", type="primary"):
            rna_d  = user_inputs["rna"]  if rna_enabled  else None
            prot_d = user_inputs["prot"] if prot_enabled else None
            met_d  = user_inputs["met"]  if met_enabled  else None
            result = score_sample(rna_d, prot_d, met_d)
            result.update(input_qc(rna_d, prot_d, met_d))
            for key in ("rna", "prot", "met"):
                for f in LAYER_FEATURES[key]:
                    v = user_inputs[key].get(f) if enabled[key] else None
                    result[f"{key}_{f}"] = np.nan if v is None else float(v)
            m_results = pd.DataFrame([result])
            st.success("Analysis Complete!")
            st.divider()
            render_dashboard(m_results, mode="manual", key_prefix="man")

    #  Bulk upload section
    with analysis_tabs[1]:
        st.subheader("Bulk Data Processing")
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            st.write("### Download Template")
            template_cols = ["Sample_ID"] + [f"{k}_{f}" for k in ("rna", "prot", "met")
                                             for f in LAYER_FEATURES[k]]
            template_csv = pd.DataFrame(columns=template_cols).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV Template",
                data=template_csv,
                file_name="MOmics_v11_Template.csv",
                mime="text/csv",
            )
            st.caption(
                "Columns: `Sample_ID`, then `<layer>_<SYMBOL>` (e.g. `rna_BSN`, `prot_BSN`). "
                "Units: `rna_` = raw read counts; `prot_` / `met_` = log2 abundance. "
                f"Shared symbols ({', '.join(SHARED_SYMBOLS)}) must carry the layer prefix."
            )
        with col_t1:
            st.write("### Upload Sample Data")
            uploaded_file = st.file_uploader("Upload CSV", type="csv")

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                id_col = next((c for c in raw_df.columns if c in ("Sample ID", "Sample_ID")), None)
                ids = (raw_df[id_col].astype(str).tolist() if id_col
                       else [f"Sample {i + 1}" for i in range(len(raw_df))])
                feat_df = raw_df.drop(columns=[id_col]) if id_col else raw_df
                mapping, ambiguous, unrecognized = map_upload_columns(feat_df.columns)

                st.success(f"{len(raw_df)} sample(s) read. Recognised {len(mapping)} of "
                           f"{len(ALL_FEATURES)} panel columns.")
                if ambiguous:
                    st.error("Ignored columns that exist in more than one layer — add the "
                             f"`rna_` / `prot_` / `met_` prefix: {', '.join(ambiguous)}")
                if unrecognized:
                    st.warning(f"Ignored {len(unrecognized)} unrecognised column(s): "
                               f"{', '.join(unrecognized[:15])}{'...' if len(unrecognized) > 15 else ''}")
                if len(mapping) < len(ALL_FEATURES):
                    st.warning(f"{len(ALL_FEATURES) - len(mapping)} panel feature(s) are absent from "
                               "the file and will be filled with the training mean for every sample.")

                if not mapping:
                    st.error("No panel features recognised; nothing was scored.")
                else:
                    b_results = process_dataframe(feat_df, mapping)
                    b_results.insert(0, "Sample ID", ids)
                    st.divider()
                    st.subheader("Analysis Results")
                    render_dashboard(b_results, mode="bulk", key_prefix="blk", patient_labels=ids)
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Ensure columns match the template.")

    # Example section
    with analysis_tabs[2]:
        st.subheader("Synthetic Example")
        st.warning(
            "These two profiles are **synthetic**, not real samples: every feature is set +1.5 SD (A) "
            "or −0.5 SD (B) from the training mean, plus small noise. They only demonstrate the "
            "interface. See Demo Walkthrough for real CPTAC-3 / GTEx samples."
        )

        def _generate_mock_patients():
            rng = np.random.default_rng(42)
            rows = []
            for label, rna_z, prot_z, met_z in [
                ("Synthetic A (+1.5 SD)", +1.5, +1.5, +1.5),
                ("Synthetic B (−0.5 SD)", -0.5, -0.5, -0.5),
            ]:
                row = {"Sample ID": label}
                for f in RNA_FEATURES:
                    mean = ZSCORE_PARAMS["rna"]["mean"].get(f, 0.0)
                    std  = ZSCORE_PARAMS["rna"]["std"].get(f, 1.0) or 1.0
                    log1p_val = mean + rna_z * std + rng.normal(0, 0.05)
                    row[f"rna_{f}"] = float(np.expm1(max(log1p_val, 0.0)))
                for f in PROT_FEATURES:
                    mean = ZSCORE_PARAMS["prot"]["mean"].get(f, 0.0)
                    std  = ZSCORE_PARAMS["prot"]["std"].get(f, 1.0) or 1.0
                    row[f"prot_{f}"] = float(mean + prot_z * std + rng.normal(0, 0.05))
                for f in MET_FEATURES:
                    mean = ZSCORE_PARAMS["met"]["mean"].get(f, 0.0)
                    std  = ZSCORE_PARAMS["met"]["std"].get(f, 1.0) or 1.0
                    row[f"met_{f}"] = float(mean + met_z * std + rng.normal(0, 0.05))
                rows.append(row)
            return pd.DataFrame(rows)

        col_ex1, col_ex2 = st.columns([1, 2])
        with col_ex1:
            st.markdown("**Synthetic A:** all features +1.5 SD")
            st.markdown("**Synthetic B:** all features −0.5 SD")
            st.markdown("**RNA:** raw read counts | **Prot/Met:** log2 abundance")

            if st.button("Run Example Analysis", type="primary", key="btn_example"):
                try:
                    example_df = _generate_mock_patients()
                    score_df = example_df.drop(columns=["Sample ID"])
                    results = process_dataframe(score_df)
                    results.insert(0, "Sample ID", example_df["Sample ID"].values)
                    st.session_state.example_results = results
                    st.session_state.example_ran = True
                except Exception as e:
                    st.error(f"Error running example: {e}")

            if st.session_state.get("example_ran"):
                if st.button("Clear Results", key="btn_example_clear"):
                    st.session_state.pop("example_results", None)
                    st.session_state.pop("example_ran", None)
                    st.rerun()

        with col_ex2:
            with st.expander("Preview synthetic input data"):
                try:
                    preview = _generate_mock_patients()
                    st.dataframe(preview, use_container_width=True, hide_index=True)
                except Exception:
                    st.info("Click 'Run Example Analysis' to generate and preview the data.")
            with st.expander("v11 feature panel"):
                st.dataframe(build_reference_table(), use_container_width=True, hide_index=True)

        if st.session_state.get("example_ran") and "example_results" in st.session_state:
            st.divider()
            st.subheader("Example Results")
            res = st.session_state.example_results
            render_res = res.drop(columns=["Sample ID"], errors="ignore")
            patient_labels = res["Sample ID"].tolist() if "Sample ID" in res.columns else None
            render_dashboard(render_res, mode="bulk", key_prefix="ex",
                             patient_labels=patient_labels)

# Demo walkthrough
elif page == "Demo Walkthrough":
    st.header("Interactive Demo Workspace")
    st.markdown("""
    <div class="demo-box">
    <h3>Welcome to the Demo Workspace</h3>
    <p>This workspace uses real samples from the v11 training cohort — the same data the model was built on.
    Correct calls here show that the interface runs end to end, <strong>not</strong> how well the model
    performs on new data.</p>
    <ul>
        <li><strong>3 GBM tumor samples</strong> (CPTAC-3 cohort, C3L/C3N IDs) — all 3 layers available</li>
        <li><strong>1 GTEx normal brain sample</strong> (PT- ID) — RNA + Protein only (no metabolomics)</li>
        <li>Full z-score → sub-model → fusion → isotonic calibration pipeline</li>
        <li>Raw score (for the model call) and calibrated probability both shown, with features used and QC status</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Load full cohort (cached), then slice to the 4 selected demo patients
    _all_data, _all_ids, _all_labels = load_demo_data()
    demo_data, demo_sample_ids, demo_true_labels = _filter_demo(_all_data, _all_ids, _all_labels)
    demo_labels = [f"{sid} ({lbl})" for sid, lbl in zip(demo_sample_ids, demo_true_labels)]
    st.divider()

    demo_mode = st.radio(
        "**Choose Demo Mode:**",
        ["Try with Sample Patients", "Guided Tutorial", "Learn by Exploring"],
        horizontal=True
    )

    
    if demo_mode == "Try with Sample Patients":
        st.subheader("Interactive Analysis with Sample Data")
        st.markdown("""<div class="demo-box demo-success">
        <h4>Real Training Samples Loaded</h4>
        <p><strong>3 GBM tumor samples</strong> (CPTAC-3, all 3 layers) and
        <strong>1 GTEx normal brain sample</strong> (RNA + Protein only, no metabolomics).
        Click "Analyze Sample Patients" to run the full v11 pipeline.</p>
        </div>""", unsafe_allow_html=True)

        with st.expander("Preview Sample Patient Data"):
            preview_df = demo_data.copy()
            preview_df.insert(0, "Sample", demo_sample_ids)
            preview_df.insert(1, "True Label", demo_true_labels)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

        if st.button("Analyze Sample Patients", key="analyze_demo_patients", type="primary"):
            with st.spinner("Running v11 inference pipeline..."):
                st.session_state.demo_try_results = process_demo_dataframe(demo_data)

        if "demo_try_results" in st.session_state:
            st.markdown("---")
            st.success("Analysis Complete!")
            res = st.session_state.demo_try_results
            concordant = sum(
                (tl == "GBM") == (bc == "GBM")
                for tl, bc in zip(demo_true_labels, res["Binary Call"].tolist())
            )
            st.info(f"Agreement with true labels: **{concordant}/{len(demo_true_labels)}** (training samples — not a performance estimate)")
            render_dashboard(res, mode="bulk", key_prefix="demo", patient_labels=demo_labels)

    #  GUIDED TUTORIAL
    elif demo_mode == "Guided Tutorial":
        st.subheader("Step-by-Step Guided Tutorial")
        if "tutorial_step" not in st.session_state:
            st.session_state.tutorial_step = 0

        st.progress(st.session_state.tutorial_step / 4)
        st.write(f"**Progress:** Step {st.session_state.tutorial_step + 1} of 5")

        if st.session_state.tutorial_step == 0:
            st.markdown("""<div class="demo-box"><h3>Step 1: The Demo Dataset</h3>
            <p>This demo uses 4 real samples from the v11 training cohort:
            <strong>3 CPTAC-3 GBM tumor samples</strong> and
            <strong>1 GTEx normal brain sample</strong>.
            Layer availability varies — the normal sample has RNA + Protein only (no metabolomics).
            Missing layers pass NaN to the fusion model rather than being imputed.</p></div>""",
                        unsafe_allow_html=True)
            preview_df = demo_data.copy()
            preview_df.insert(0, "Sample", demo_sample_ids)
            preview_df.insert(1, "True Label", demo_true_labels)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            st.caption("NaN = that layer is not available for this sample. XGBoost fusion handles this natively.")
            if st.button("Next: Run Analysis", key="tutorial_next_0"):
                st.session_state.tutorial_step = 1
                st.rerun()

        elif st.session_state.tutorial_step == 1:
            st.markdown("""<div class="demo-box"><h3>Step 2: The Inference Pipeline</h3>
            <p>Inputs are z-scored with frozen training parameters → scored by per-layer XGBoost sub-models
            → fused by a meta-XGBoost → optionally calibrated by isotonic regression.
            Missing layers pass NaN to fusion — not imputed.</p></div>""",
                        unsafe_allow_html=True)
            if st.button("Process Sample Data", key="tutorial_analyze", type="primary"):
                with st.spinner("Running inference..."):
                    st.session_state.demo_results = process_demo_dataframe(demo_data)
                    st.session_state.tutorial_step = 2
                st.rerun()

        elif st.session_state.tutorial_step == 2:
            st.markdown(
                f"""<div class="demo-box demo-success"><h3>Step 3: Cohort Results</h3>
                <p>Raw scores cluster bimodally: GBM ~0.97, Non-GBM ~0.38.
                Youden threshold = {THRESHOLD:.4f}. The GTEx normal correctly scores Non-GBM
                even without a metabolomics layer.</p></div>""",
                unsafe_allow_html=True)
            if "demo_results" in st.session_state:
                res = st.session_state.demo_results
                concordant = sum(
                    (tl == "GBM") == (bc == "GBM")
                    for tl, bc in zip(demo_true_labels, res["Binary Call"].tolist())
                )
                st.info(f"Concordance with true labels: **{concordant}/{len(demo_true_labels)}**")
                render_risk_charts(res, mode="bulk", key_prefix="tutorial")
            if st.button("Next: Individual Patient", key="tutorial_next_2"):
                st.session_state.tutorial_step = 3
                st.rerun()

        elif st.session_state.tutorial_step == 3:
            st.markdown("""<div class="demo-box"><h3>Step 4: Individual Sample</h3>
            <p>Compare a GBM and the Normal sample side-by-side to see how the biomarker
            profiles differ across the panel.</p></div>""", unsafe_allow_html=True)
            if "demo_results" in st.session_state:
                sel = st.selectbox("Choose sample:", range(len(demo_labels)),
                                   format_func=lambda i: demo_labels[i],
                                   key="tutorial_patient_select")
                row = st.session_state.demo_results.iloc[sel]
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                c1.metric("True Label",  demo_true_labels[sel])
                c2.metric("Model Call",  row["Binary Call"])
                c3.metric("Raw",         f"{row['Risk Score (Raw)']:.4f}")
                c4.metric("Calibrated",  f"{row['Risk Score (Calibrated)']:.4f}")
                c5.metric("RNA",         _fmt(row["RNA Score"]))
                c6.metric("Protein",     _fmt(row["Protein Score"]))
                c7.metric("Met",         _fmt(row["Metabolomics Score"]))

                marker_cols = [c for c in st.session_state.demo_results.columns if c not in SCORE_COLS]
                vals = row[marker_cols].astype(float).dropna().sort_values(ascending=False)
                fig = px.bar(x=vals.values, y=vals.index, orientation="h",
                             title=f"Biomarker Values — {demo_labels[sel]}")
                st.plotly_chart(fig, use_container_width=True)
            if st.button("Next: Wrap Up", key="tutorial_next_3"):
                st.session_state.tutorial_step = 4
                st.rerun()

        elif st.session_state.tutorial_step == 4:
            st.markdown("""<div class="demo-box demo-success"><h3>Tutorial Complete!</h3>
            <p>You've walked through the full v11 pipeline: z-score normalization → sub-models →
            fusion → Youden threshold → isotonic calibration, on real CPTAC GBM and GTEx normal samples.</p>
            </div>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.info("Navigate to 'User Analysis' to score your own samples.")
            with c2:
                if st.button("🔄 Restart Tutorial", key="restart_tut"):
                    st.session_state.tutorial_step = 0
                    st.session_state.pop("demo_results", None)
                    st.rerun()

    # LEARN BY EXPLORING
    elif demo_mode == "Learn by Exploring":
        st.subheader("Free Exploration Mode")
        st.markdown("""<div class="demo-box"><h4>Explore at Your Own Pace</h4>
        <p>Full interface with 3 real CPTAC GBM samples and 1 GTEx normal brain sample.</p>
        </div>""", unsafe_allow_html=True)

        exp_tabs = st.tabs(["Sample Analysis", "Learning Resources", "Tips & Tricks"])

        with exp_tabs[0]:
            if st.button("Load & Analyze Sample Data", key="explore_analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    st.session_state.demo_explore_results = process_demo_dataframe(demo_data)
            if "demo_explore_results" in st.session_state:
                st.success("Sample data analyzed!")
                res = st.session_state.demo_explore_results
                concordant = sum(
                    (tl == "GBM") == (bc == "GBM")
                    for tl, bc in zip(demo_true_labels, res["Binary Call"].tolist())
                )
                st.info(f"Concordance with true labels: **{concordant}/{len(demo_true_labels)}**")
                st.divider()
                render_dashboard(res, mode="bulk", key_prefix="explore", patient_labels=demo_labels)

        with exp_tabs[1]:
            st.write("### Quick Reference")
            with st.expander("Understanding the Two Scores"):
                st.write(
                    f"- **Raw fusion score**: XGBoost output. Threshold = {THRESHOLD:.4f} (Youden). "
                    "Scores cluster near 0.38 (Non-GBM) or 0.97 (GBM).\n"
                    "- **Calibrated probability**: isotonic-mapped to external validation prevalence "
                    "(4.6% GBM). A raw score of ~0.97 → ~0.40 calibrated. This is expected, not a bug. "
                    "In a higher-prevalence clinical setting the posterior would be higher."
                )
            with st.expander("Input Format by Layer"):
                st.write(
                    f"- **RNA ({len(RNA_FEATURES)} features):** {', '.join(RNA_FEATURES)} — "
                    "**raw read counts**. log1p applied internally before z-scoring.\n"
                    f"- **Protein ({len(PROT_FEATURES)} features):** {', '.join(PROT_FEATURES)} — "
                    "**log2 abundance** (CPTAC reference-intensity normalized).\n"
                    f"- **Metabolomics ({len(MET_FEATURES)} features):** {', '.join(MET_FEATURES)} — "
                    "**log2 abundance**.\n\n"
                    "Do not pre-normalize inputs — z-scoring uses frozen training parameters."
                )
            with st.expander("Missing Layers"):
                st.write(
                    "Uncheck a layer in Manual Entry or simply omit its columns from a CSV. "
                    "The fusion model handles NaN sub-model probabilities natively via XGBoost's "
                    "missing-value mechanism — do not impute or average. "
                    "The GTEx normal sample in this demo has no metabolomics and still scores correctly."
                )

        with exp_tabs[2]:
            st.write("### Exploration Tips")
            st.info(
                "**Things to Try:**\n"
                "1. Compare a GBM sample (C3L/C3N) vs the Normal (PT-) in the individual patient selector\n"
                "2. Notice how the Normal sample scores ~0.38 raw even with only 2 layers available\n"
                "3. Compare raw vs calibrated scores — the gap reflects 4.6% GBM prevalence in the external set\n"
                "4. Look at the metabolomics score for the Normal sample (shows N/A) vs the GBM samples"
            )

    st.divider()
    if st.button("Reset Demo Workspace"):
        for key in [k for k in st.session_state if "demo" in k or "tutorial" in k]:
            del st.session_state[key]
        st.rerun()
