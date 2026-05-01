import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from docs import OVERVIEW, GUI_GUIDE, MODEL_ARCH

# --- Page Configuration ---
st.set_page_config(page_title="MOmics", layout="wide", page_icon="🧬")

# --- Custom CSS ---
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


# =============================================================================
# PIPELINE LOADING
# =============================================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("MOmics_v11_locked_pipeline.pkl")

pipe = load_pipeline()

# Unpack everything we need from the single artifact
PRUNED         = pipe["pruned_features"]       # {'rna': [...], 'prot': [...], 'met': [...]}
SUB_MODELS     = pipe["sub_models"]            # {'rna': XGB, 'prot': XGB, 'met': XGB}
FUSION_MODEL   = pipe["fusion_model"]          # XGBClassifier
ZSCORE_PARAMS  = pipe["zscore_params"]         # {'rna': {'mean': {f: μ}, 'std': {f: σ}}, ...}
THRESHOLD      = pipe["youden_threshold"]      # 0.9641
RNA_LOG1P      = pipe["rna_log1p_required"]    # True
CALIBRATOR     = pipe["isotonic_calibrator"]   # IsotonicRegression
DISCOVERY      = pipe["discovery_metrics"]
SYMBOL_TO_ENSG = pipe["symbol_to_ensg"]

RNA_FEATURES  = PRUNED["rna"]   # ['BSN', 'PCLO', 'PRKCE', 'PTPRT', 'CIT', 'MAPT']
PROT_FEATURES = PRUNED["prot"]  # ['PTPRT', 'CIT', 'PCLO', 'BSN']
MET_FEATURES  = PRUNED["met"]   # ['hypotaurine', 'creatinine', 'citricacid']
ALL_FEATURES  = RNA_FEATURES + PROT_FEATURES + MET_FEATURES

SCORE_COLS = ["Prediction", "Risk Score (Raw)", "Risk Score (Calibrated)",
              "Binary Call", "RNA Score", "Protein Score", "Metabolomics Score"]


# =============================================================================
# FEATURE IMPORTANCE DF
# =============================================================================
def _build_importance_df():
    rows = []
    layer_map = {"rna": "RNA", "prot": "Protein", "met": "Metabolomics"}
    for layer_key, label in layer_map.items():
        feats = PRUNED[layer_key]
        model = SUB_MODELS[layer_key]
        for feat, imp in zip(feats, model.feature_importances_):
            rows.append({"Biomarker": feat, "Influence Score": float(imp), "Layer": label})
    return pd.DataFrame(rows).sort_values("Influence Score", ascending=False).reset_index(drop=True)

importance_df_display = _build_importance_df()


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

def _zscore(value, mean, std):
    if std is None or np.isnan(std) or std == 0:
        return 0.0
    if mean is None or np.isnan(mean):
        return 0.0
    return (value - mean) / std


def _prepare_layer(layer_key, user_dict):
    """
    Z-score normalize a layer's input dict.
    RNA: log1p applied first. Missing features within a provided layer filled with 0 (z-score mean).
    Returns (1, n_features) array, or None if user_dict is None.
    """
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
    """
    Full inference: z-score → 3 sub-models → fusion → calibration.
    Each dict is {symbol: raw_value} or None if layer unavailable.
    Binary call uses raw probability vs Youden threshold.
    """
    layer_probs  = []
    layer_scores = {}

    for layer_key, user_dict, label in [
        ("rna",  rna_dict,  "RNA Score"),
        ("prot", prot_dict, "Protein Score"),
        ("met",  met_dict,  "Metabolomics Score"),
    ]:
        arr = _prepare_layer(layer_key, user_dict)
        if arr is None:
            layer_probs.append(np.nan)
            layer_scores[label] = None
        else:
            p = float(SUB_MODELS[layer_key].predict_proba(arr)[:, 1][0])
            layer_probs.append(p)
            layer_scores[label] = p

    fusion_input = np.array([layer_probs])  # NaN → XGBoost handles natively
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


def process_dataframe(df):
    """Score every row. Columns should be feature symbols; layer routing is automatic."""
    rna_set  = set(RNA_FEATURES)
    prot_set = set(PROT_FEATURES)
    met_set  = set(MET_FEATURES)

    results = []
    for _, row in df.iterrows():
        def layer_dict(feat_set):
            d = {f: row[f] for f in feat_set if f in row.index and pd.notna(row[f])}
            return d if d else None

        result = score_sample(
            rna_dict  = layer_dict(rna_set),
            prot_dict = layer_dict(prot_set),
            met_dict  = layer_dict(met_set),
        )
        for f in ALL_FEATURES:
            result[f] = float(row[f]) if f in row.index and pd.notna(row.get(f)) else np.nan
        results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def _fmt(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.4f}"


def render_risk_charts(results, mode="manual", key_prefix=""):
    st.subheader("Prediction & Risk Assessment")

    if mode == "manual":
        row = results.iloc[0]
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Binary Call",            row["Binary Call"])
        c2.metric("Prediction",             row["Prediction"])
        c3.metric("Raw Fusion Score",       f"{row['Risk Score (Raw)']:.4f}")
        c4.metric("Calibrated Probability", f"{row['Risk Score (Calibrated)']:.4f}")
        c5.metric("RNA Score",              _fmt(row["RNA Score"]))
        c6.metric("Protein Score",          _fmt(row["Protein Score"]))
        c7.metric("Metabolomics Score",     _fmt(row["Metabolomics Score"]))
        st.caption(
            f"**Threshold:** {THRESHOLD:.4f} (Youden, operates on raw score). "
            "Calibrated probability reflects ~4.6% GBM prevalence in the external validation set — "
            "a raw score of ~0.97 maps to ~0.40 calibrated. This is expected, not a bug. "
            "Use raw score for the binary GBM/Non-GBM call."
        )

    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(results, x="Risk Score (Raw)", color="Prediction",
                               title="Raw Fusion Score Distribution",
                               color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"},
                               nbins=20)
            fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="gray",
                          annotation_text=f"Youden ({THRESHOLD:.3f})")
            fig.update_layout(xaxis_title="Raw Score", yaxis_title="Number of Patients")
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_hist")
        with col2:
            rs = results.sort_values("Risk Score (Raw)", ascending=False).reset_index(drop=True)
            rs["Patient_ID"] = rs.index
            fig2 = px.bar(rs, x="Patient_ID", y="Risk Score (Raw)", color="Prediction",
                          title="Individual Patient Raw Scores",
                          color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"})
            fig2.add_hline(y=THRESHOLD, line_dash="dash", line_color="gray",
                           annotation_text=f"Threshold ({THRESHOLD:.3f})")
            fig2.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_bar")

        st.divider()
        st.subheader("Risk Score Table")
        display = results[["Prediction", "Binary Call",
                            "Risk Score (Raw)", "Risk Score (Calibrated)",
                            "RNA Score", "Protein Score", "Metabolomics Score"]].copy()
        display.insert(0, "Patient ID", display.index)
        for col in ["Risk Score (Raw)", "Risk Score (Calibrated)",
                    "RNA Score", "Protein Score", "Metabolomics Score"]:
            display[col] = display[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) and x is not None else "N/A"
            )
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            f"Threshold = {THRESHOLD:.4f} (Youden, raw score). "
            "Calibrated score is isotonic-mapped to the external validation set (~4.6% GBM prevalence). "
            "Use raw score for the binary call."
        )


def render_dashboard(results, mode="manual", key_prefix="", patient_labels=None):
    render_risk_charts(results, mode=mode, key_prefix=key_prefix)

    if mode == "bulk":
        st.divider()
        st.subheader("Cohort Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        total = len(results)
        gbm_n = (results["Binary Call"] == "GBM").sum()
        c1.metric("Total Patients",  total)
        c2.metric("GBM Calls",       f"{gbm_n} ({gbm_n/total:.1%})")
        c3.metric("Mean Raw Score",  f"{results['Risk Score (Raw)'].mean():.4f}")
        c4.metric("Median Raw Score",f"{results['Risk Score (Raw)'].median():.4f}")

    st.divider()
    st.subheader("Individual Patient Analysis")
    fmt = (lambda i: f"Patient {i} — {patient_labels[i]}"
           if patient_labels and i < len(patient_labels) else f"Patient {i}")
    selected = st.selectbox("Select Patient Record", results.index,
                            format_func=fmt, key=f"{key_prefix}_select")
    row = results.iloc[selected]

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Binary Call",            row["Binary Call"])
    c2.metric("Prediction",             row["Prediction"])
    c3.metric("Raw Score",              f"{row['Risk Score (Raw)']:.4f}")
    c4.metric("Calibrated",             f"{row['Risk Score (Calibrated)']:.4f}")
    c5.metric("RNA Score",              _fmt(row["RNA Score"]))
    c6.metric("Protein Score",          _fmt(row["Protein Score"]))
    c7.metric("Metabolomics Score",     _fmt(row["Metabolomics Score"]))

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
        st.write(f"### Biomarker Input Values (Patient {selected})")
        marker_cols = [c for c in results.columns if c not in SCORE_COLS]
        marker_vals = row[marker_cols].astype(float).dropna().sort_values(ascending=False)
        fig_top = px.bar(x=marker_vals.values, y=marker_vals.index, orientation="h",
                         color=marker_vals.values, color_continuous_scale="Viridis",
                         labels={"x": "Raw Value", "y": "Biomarker"})
        st.plotly_chart(fig_top, use_container_width=True, key=f"{key_prefix}_pbar_{selected}")

    st.divider()
    col_imp1, col_imp2 = st.columns(2)

    with col_imp1:
        st.write("#### Patient's Biomarker Values")
        patient_vals = row[marker_cols].astype(float).dropna().sort_values(ascending=False)
        fig_pt = px.bar(x=patient_vals.values, y=patient_vals.index, orientation="h",
                        color=patient_vals.values, color_continuous_scale="Viridis",
                        title=f"Input Values — Patient {selected}")
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

    with st.expander("View All Biomarker Values for This Patient"):
        all_markers = row[marker_cols].to_frame(name="Value").reset_index()
        all_markers.columns = ["Biomarker", "Value"]
        all_markers = all_markers.sort_values("Value", ascending=False)
        st.dataframe(all_markers, use_container_width=True, hide_index=True)


# =============================================================================
# REFERENCE TABLE
# =============================================================================
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


# =============================================================================
# DEMO DATA — built from real CPTAC GBM + GTEx normal training samples
# 5 GBM tumor (C3L/C3N, all 3 layers) + 5 GTEx normal (RNA+Prot, no met)
# =============================================================================
DEMO_CSV_PATH = "MOmics_v11_demo_data.csv"

@st.cache_data
def load_demo_data():
    try:
        df = pd.read_csv(DEMO_CSV_PATH)
        sample_ids  = df["Sample_ID"].tolist()
        true_labels = df["True_Label"].tolist() if "True_Label" in df.columns else ["Unknown"] * len(df)
        df_data = df.drop(columns=["Sample_ID", "True_Label"], errors="ignore")
        return df_data, sample_ids, true_labels
    except FileNotFoundError:
        st.error(f"Demo data file '{DEMO_CSV_PATH}' not found. Ensure it is in the repo root.")
        st.stop()


# =============================================================================
# SIDEBAR & NAVIGATION
# =============================================================================
st.sidebar.title("MOmics")
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

st.title("MOmics | GBM Clinical Diagnostic Suite")


# =============================================================================
# HOME PAGE
# =============================================================================
if page == "Home":
    try:
        from PIL import Image
        logo = Image.open("logo.png")
        st.image(logo, use_container_width=True)
    except Exception:
        st.info("Logo image not found. Please ensure 'logo.png' is in the root directory.")
    st.markdown("<h1 style='text-align: center;'>MOmics</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>GBM Clinical Diagnostic Suite</h3>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Discovery Cohort Performance")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("RNA OOF AUROC",   f"{DISCOVERY.get('rna_oof_auc', 0):.4f}")
    c2.metric("Prot OOF AUROC",  f"{DISCOVERY.get('prot_oof_auc', 0):.4f}")
    c3.metric("Met OOF AUROC",   f"{DISCOVERY.get('met_oof_auc', 0):.4f}")
    c4.metric("Fusion AUROC",    f"{DISCOVERY.get('fusion_auc', 0):.4f}")
    c5.metric("Sensitivity",     f"{DISCOVERY.get('fusion_sens', 0):.1%}")
    c6.metric("Specificity",     f"{DISCOVERY.get('fusion_spec', 0):.1%}")


# =============================================================================
# DOCUMENTATION PAGE
# =============================================================================
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


# =============================================================================
# USER ANALYSIS PAGE
# =============================================================================
elif page == "User Analysis":
    st.header("User Analysis")
    analysis_tabs = st.tabs(["Manual Patient Entry", "Bulk Data Upload", "Example Analysis"])

    # ── MANUAL ENTRY ─────────────────────────────────────────────────────────
    with analysis_tabs[0]:
        st.subheader("Manual Patient Entry")
        st.info(
            "**RNA:** enter raw read counts (log1p applied internally). "
            "**Protein & Metabolomics:** enter log2 abundances (CPTAC standard). "
            "Do not pre-normalize. Leave a field at 0.0 to fill with training mean. "
            "Uncheck a layer to exclude it — the fusion model handles missing layers natively."
        )

        user_inputs = {}
        col_tog1, col_tog2, col_tog3 = st.columns(3)
        rna_enabled  = col_tog1.checkbox("Include RNA layer",          value=True, key="tog_rna")
        prot_enabled = col_tog2.checkbox("Include Protein layer",      value=True, key="tog_prot")
        met_enabled  = col_tog3.checkbox("Include Metabolomics layer", value=True, key="tog_met")

        st.write("#### RNA Biomarkers (raw read counts)")
        rna_cols = st.columns(3)
        for i, feat in enumerate(RNA_FEATURES):
            with rna_cols[i % 3]:
                user_inputs[feat] = st.number_input(
                    feat, value=0.0, key=f"rna_{feat}",
                    help=f"ENSG: {SYMBOL_TO_ENSG.get(feat, '—')}",
                    disabled=not rna_enabled
                )

        st.write("#### Protein Biomarkers")
        prot_cols = st.columns(4)
        for i, feat in enumerate(PROT_FEATURES):
            with prot_cols[i % 4]:
                user_inputs[feat] = st.number_input(
                    feat, value=0.0, key=f"prot_{feat}",
                    help=f"ENSG: {SYMBOL_TO_ENSG.get(feat, '—')}",
                    disabled=not prot_enabled
                )

        st.write("#### Metabolomics Biomarkers ")
        met_cols = st.columns(3)
        for i, feat in enumerate(MET_FEATURES):
            with met_cols[i % 3]:
                user_inputs[feat] = st.number_input(
                    feat, value=0.0, key=f"met_{feat}",
                    disabled=not met_enabled
                )

        if st.button("Analyze Single Patient", key="btn_manual", type="primary"):
            rna_d  = {f: user_inputs[f] for f in RNA_FEATURES}  if rna_enabled  else None
            prot_d = {f: user_inputs[f] for f in PROT_FEATURES} if prot_enabled else None
            met_d  = {f: user_inputs[f] for f in MET_FEATURES}  if met_enabled  else None
            result = score_sample(rna_d, prot_d, met_d)
            for f in ALL_FEATURES:
                result[f] = user_inputs.get(f, np.nan)
            m_results = pd.DataFrame([result])
            st.success("Analysis Complete!")
            st.divider()
            render_dashboard(m_results, mode="manual", key_prefix="man")

    # ── BULK UPLOAD ───────────────────────────────────────────────────────────
    with analysis_tabs[1]:
        st.subheader("Bulk Data Processing")
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            st.write("### Download Template")
            layer_note = {f: ("raw counts" if f in RNA_FEATURES else "log2 abundance") for f in ALL_FEATURES}
            template_csv = pd.DataFrame([layer_note]).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV Template",
                data=template_csv,
                file_name="MOmics_v11_Patient_Template.csv",
                mime="text/csv",
                help="Row 1 shows expected value type. Replace it with patient data before uploading."
            )
        with col_t1:
            st.write("### Upload Patient Data")
            uploaded_file = st.file_uploader(
                "Upload patient CSV", type="csv",
                help="Columns = gene symbols. RNA = raw counts, Prot/Met = log2 abundance."
            )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                raw_df = raw_df.drop(
                    columns=[c for c in raw_df.columns if c in ("Sample ID", "Sample_ID")],
                    errors="ignore"
                )
                st.success(f"File uploaded — {len(raw_df)} patient(s) found.")
                b_results = process_dataframe(raw_df)
                st.divider()
                st.subheader("Analysis Results")
                render_dashboard(b_results, mode="bulk", key_prefix="blk")
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Ensure columns match feature symbols in the template.")

    # ── EXAMPLE ANALYSIS ──────────────────────────────────────────────────────
    with analysis_tabs[2]:
        st.subheader("Example Analysis")
        st.write(
            "Runs the full v11 inference pipeline on a pre-loaded CPTAC GBM patient sample. "
            "Input file (`momics_input.csv`) should have columns matching the v11 feature symbols."
        )
        col_ex1, col_ex2 = st.columns([1, 2])
        with col_ex1:
            st.markdown("**Input file:** `momics_input.csv`")
            st.markdown("**Sample:** CPTAC-3 GBM patient")
            st.markdown("**RNA:** raw read counts | **Prot/Met:** log2 abundance")

            if st.button("Run Example Analysis", type="primary", key="btn_example"):
                try:
                    example_df = pd.read_csv("momics_input.csv")
                    st.session_state.example_results = process_dataframe(example_df)
                    st.session_state.example_ran = True
                except FileNotFoundError:
                    st.error("`momics_input.csv` not found in the repo root.")
                except Exception as e:
                    st.error(f"Error running example: {e}")

            if st.session_state.get("example_ran"):
                if st.button("Clear Results", key="btn_example_clear"):
                    st.session_state.pop("example_results", None)
                    st.session_state.pop("example_ran", None)
                    st.rerun()

        with col_ex2:
            with st.expander("v11 feature panel"):
                st.dataframe(build_reference_table(), use_container_width=True, hide_index=True)

        if st.session_state.get("example_ran") and "example_results" in st.session_state:
            st.divider()
            st.subheader("Example Results")
            render_dashboard(st.session_state.example_results, mode="bulk", key_prefix="ex")


# =============================================================================
# DEMO WALKTHROUGH PAGE
# =============================================================================
elif page == "Demo Walkthrough":
    st.header("Interactive Demo Workspace")
    st.markdown("""
    <div class="demo-box">
    <h3>Welcome to the Demo Workspace</h3>
    <p>This workspace uses real samples from the v11 training cohort — the same data the model was built on.</p>
    <ul>
        <li><strong>5 GBM tumor samples</strong> (CPTAC-3 cohort, C3L/C3N IDs) — all 3 layers available</li>
        <li><strong>5 GTEx normal brain samples</strong> (PT- IDs) — RNA + Protein only (no metabolomics)</li>
        <li>Full z-score → sub-model → fusion → isotonic calibration pipeline</li>
        <li>Raw score (for binary call) and calibrated probability (for clinical context) both shown</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    demo_data, demo_sample_ids, demo_true_labels = load_demo_data()
    # Build display labels: "C3L-00365 (GBM)" etc.
    demo_labels = [f"{sid} ({lbl})" for sid, lbl in zip(demo_sample_ids, demo_true_labels)]
    st.divider()

    demo_mode = st.radio(
        "**Choose Demo Mode:**",
        ["Try with Sample Patients", "Guided Tutorial", "Learn by Exploring"],
        horizontal=True
    )

    # ── TRY WITH SAMPLE PATIENTS ─────────────────────────────────────────────
    if demo_mode == "Try with Sample Patients":
        st.subheader("Interactive Analysis with Sample Data")
        st.markdown("""<div class="demo-box demo-success">
        <h4>Real Patient Dataset Loaded</h4>
        <p><strong>5 GBM tumor samples</strong> (CPTAC-3, all 3 layers) and
        <strong>5 GTEx normal brain samples</strong> (RNA + Protein only, no metabolomics).
        Click "Analyze Sample Patients" to run the full v11 diagnostic pipeline.</p>
        </div>""", unsafe_allow_html=True)

        with st.expander("Preview Sample Patient Data"):
            preview_df = demo_data.copy()
            preview_df.insert(0, "Sample", demo_sample_ids)
            preview_df.insert(1, "True Label", demo_true_labels)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

        if st.button("Analyze Sample Patients", key="analyze_demo_patients", type="primary"):
            with st.spinner("Running v11 inference pipeline..."):
                st.session_state.demo_try_results = process_dataframe(demo_data)

        if "demo_try_results" in st.session_state:
            st.markdown("---")
            st.success("Analysis Complete!")
            # Show concordance with true labels
            res = st.session_state.demo_try_results
            concordant = sum(
                (tl == "GBM") == (bc == "GBM")
                for tl, bc in zip(demo_true_labels, res["Binary Call"].tolist())
            )
            st.info(f"Model concordance with true labels: **{concordant}/{len(demo_true_labels)}** correct")
            render_dashboard(res, mode="bulk", key_prefix="demo", patient_labels=demo_labels)

    # ── GUIDED TUTORIAL ───────────────────────────────────────────────────────
    elif demo_mode == "Guided Tutorial":
        st.subheader("Step-by-Step Guided Tutorial")
        if "tutorial_step" not in st.session_state:
            st.session_state.tutorial_step = 0

        st.progress(st.session_state.tutorial_step / 4)
        st.write(f"**Progress:** Step {st.session_state.tutorial_step + 1} of 5")

        if st.session_state.tutorial_step == 0:
            st.markdown("""<div class="demo-box"><h3>Step 1: The Demo Dataset</h3>
            <p>This demo uses real CPTAC-3 GBM tumor samples and GTEx normal brain samples —
            the same data used to train the v11 model. 5 GBM + 5 Normal, with a mix of complete
            (all 3 layers) and partial (RNA + Protein only) inputs.</p></div>""", unsafe_allow_html=True)
            preview_df = demo_data.copy()
            preview_df.insert(0, "Sample", demo_sample_ids)
            preview_df.insert(1, "True Label", demo_true_labels)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            st.caption("NaN = metabolomics layer not available for GTEx normals. XGBoost fusion handles this natively.")
            if st.button("Next: Run Analysis", key="tutorial_next_0"):
                st.session_state.tutorial_step = 1
                st.rerun()

        elif st.session_state.tutorial_step == 1:
            st.markdown("""<div class="demo-box"><h3>Step 2: The Inference Pipeline</h3>
            <p>Inputs are z-scored with frozen training parameters → scored by per-layer XGBoost sub-models
            → fused by a meta-XGBoost → optionally calibrated by isotonic regression.
            Missing layers (metabolomics for GTEx) pass NaN to fusion — not imputed.</p></div>""",
                        unsafe_allow_html=True)
            if st.button("Process Sample Data", key="tutorial_analyze", type="primary"):
                with st.spinner("Running inference..."):
                    st.session_state.demo_results = process_dataframe(demo_data)
                    st.session_state.tutorial_step = 2
                st.rerun()

        elif st.session_state.tutorial_step == 2:
            st.markdown(
                f"""<div class="demo-box demo-success"><h3>Step 3: Cohort Results</h3>
                <p>Raw scores cluster bimodally: GBM ~0.97, Non-GBM ~0.38.
                Youden threshold = {THRESHOLD:.4f}. GTEx normals correctly score Non-GBM
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
            <p>Compare a GBM and a Normal sample side-by-side to see how the biomarker
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

    # ── LEARN BY EXPLORING ────────────────────────────────────────────────────
    elif demo_mode == "Learn by Exploring":
        st.subheader("Free Exploration Mode")
        st.markdown("""<div class="demo-box"><h4>Explore at Your Own Pace</h4>
        <p>Full interface with real CPTAC GBM tumor + GTEx normal brain samples and the v11 pipeline.</p>
        </div>""", unsafe_allow_html=True)

        exp_tabs = st.tabs(["Sample Analysis", "Learning Resources", "Tips & Tricks"])

        with exp_tabs[0]:
            if st.button("Load & Analyze Sample Data", key="explore_analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    st.session_state.demo_explore_results = process_dataframe(demo_data)
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
                    "The GTEx normal samples in this demo have no metabolomics and still score correctly."
                )

        with exp_tabs[2]:
            st.write("### Exploration Tips")
            st.info(
                "**Things to Try:**\n"
                "1. Compare a GBM sample (C3L/C3N) vs a Normal (PT-) in the individual patient selector\n"
                "2. Notice how Normal samples score ~0.38 raw even with only 2 layers available\n"
                "3. Compare raw vs calibrated scores — the gap reflects 4.6% GBM prevalence in the external set\n"
                "4. Check the Feature Reference tab in Documentation for training z-score parameters"
            )

    st.divider()
    if st.button("Reset Demo Workspace"):
        for key in [k for k in st.session_state if "demo" in k or "tutorial" in k]:
            del st.session_state[key]
        st.rerun()
