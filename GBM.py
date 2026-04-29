import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from scipy.stats import rankdata
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
# ASSET LOADING
# =============================================================================
@st.cache_resource
def load_assets():
    with open('momics_rna_model.pkl', 'rb') as f:
        rna_pkg = pickle.load(f)
    with open('momics_prot_model.pkl', 'rb') as f:
        prot_pkg = pickle.load(f)
    with open('momics_met_model.pkl', 'rb') as f:
        met_pkg = pickle.load(f)
    with open('momics_fusion_model.pkl', 'rb') as f:
        fusion_pkg = pickle.load(f)
    with open('momics_feature_metadata.pkl', 'rb') as f:
        feature_metadata = pickle.load(f)
    with open('momics_reference_ranges.pkl', 'rb') as f:
        reference_ranges = pickle.load(f)
    return rna_pkg, prot_pkg, met_pkg, fusion_pkg, feature_metadata, reference_ranges

rna_pkg, prot_pkg, met_pkg, fusion_pkg, feature_metadata, reference_ranges = load_assets()


# =============================================================================
# FEATURE HELPERS
# =============================================================================

# Build lookup dicts from feature_metadata
# feature_metadata = {'rna': [{ensembl, symbol, name}, ...], 'prot': [...], 'met': [...]}
def _build_lookups():
    id_to_symbol = {}
    id_to_name   = {}
    symbol_to_id = {}
    for layer_key in ('rna', 'prot', 'met'):
        for entry in feature_metadata.get(layer_key, []):
            eid = entry['ensembl'] or entry['symbol']  # met has no ensembl
            sym = entry['symbol'] or eid
            name = entry['name'] or sym
            id_to_symbol[eid] = sym
            id_to_name[eid]   = name
            symbol_to_id[sym] = eid
    return id_to_symbol, id_to_name, symbol_to_id

ID_TO_SYMBOL, ID_TO_NAME, SYMBOL_TO_ID = _build_lookups()

def to_symbol(feat_id):
    return ID_TO_SYMBOL.get(str(feat_id), str(feat_id))

def to_name(feat_id):
    return ID_TO_NAME.get(str(feat_id), str(feat_id))

# All features ordered: RNA first, then PROT, then MET
ALL_FEATURES = rna_pkg['features'] + prot_pkg['features'] + met_pkg['features']
ALL_SYMBOLS  = rna_pkg['feature_symbols'] + prot_pkg['feature_symbols'] + met_pkg['feature_symbols']

# Combined importance DataFrame (for display)
def _build_importance_df():
    rows = []
    for pkg, layer in [(rna_pkg, 'RNA'), (prot_pkg, 'Protein'), (met_pkg, 'Metabolomics')]:
        syms = pkg.get('feature_symbols', pkg['features'])
        imps = pkg['model'].feature_importances_
        for sym, imp in zip(syms, imps):
            rows.append({'Biomarker': sym, 'Influence Score': float(imp), 'Layer': layer})
    return pd.DataFrame(rows).sort_values('Influence Score', ascending=False).reset_index(drop=True)

importance_df_display = _build_importance_df()


# =============================================================================
# PREDICTION PIPELINE
# =============================================================================

def _rank_transform(values_array):
    """Rank-transform within sample to [0, 1]."""
    arr = np.asarray(values_array, dtype=float)
    if len(arr) <= 1:
        return np.array([0.5])
    ranks = rankdata(arr, method='average')
    return (ranks - 1) / (len(ranks) - 1)


def _layer_prob(pkg, user_dict):
    """
    Compute a layer's GBM probability.
    user_dict: {feature_id: raw_value}  or None if layer not available.
    Missing features are filled with gbm_mean from reference_ranges.
    """
    if user_dict is None:
        return np.nan
    features = pkg['features']
    ref      = pkg['reference_ranges']
    raw = np.array([
        user_dict.get(f, ref[f]['gbm_mean']) for f in features
    ], dtype=float)
    ranked = _rank_transform(raw)
    return float(pkg['model'].predict_proba(ranked.reshape(1, -1))[:, 1][0])


def predict_patient(rna_dict, prot_dict, met_dict):
    """
    Run the full 3-layer → fusion pipeline for one patient.
    Each dict is {feature_id: raw_value} or None.
    Returns dict with all scores.
    """
    rna_prob  = _layer_prob(rna_pkg,  rna_dict)
    prot_prob = _layer_prob(prot_pkg, prot_dict)
    met_prob  = _layer_prob(met_pkg,  met_dict)

    fusion_input = np.array([[rna_prob, prot_prob, met_prob]])
    gbm_prob = float(fusion_pkg['model'].predict_proba(fusion_input)[:, 1][0])
    is_gbm   = gbm_prob >= fusion_pkg['decision_threshold_default']

    return {
        'Prediction':         'High Risk' if is_gbm else 'Low Risk',
        'Risk Score':          gbm_prob,
        'RNA Score':           rna_prob  if not np.isnan(rna_prob)  else None,
        'Protein Score':       prot_prob if not np.isnan(prot_prob) else None,
        'Metabolomics Score':  met_prob  if not np.isnan(met_prob)  else None,
    }


def process_dataframe(df):
    """
    Process a DataFrame where columns are feature IDs or gene symbols.
    Auto-detects and maps symbols → IDs.
    Returns a results DataFrame.
    """
    # Remap any symbol columns to feature IDs
    df = df.rename(columns=lambda c: SYMBOL_TO_ID.get(c, c))

    results = []
    for _, row in df.iterrows():
        rna_dict  = {f: row[f] for f in rna_pkg['features']  if f in row.index and pd.notna(row[f])}
        prot_dict = {f: row[f] for f in prot_pkg['features'] if f in row.index and pd.notna(row[f])}
        met_dict  = {f: row[f] for f in met_pkg['features']  if f in row.index and pd.notna(row[f])}

        result = predict_patient(
            rna_dict  if rna_dict  else None,
            prot_dict if prot_dict else None,
            met_dict  if met_dict  else None,
        )
        # Attach original feature values for downstream display (symbol-keyed)
        for f in ALL_FEATURES:
            result[to_symbol(f)] = float(row[f]) if f in row.index and pd.notna(row.get(f)) else np.nan
        results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

SCORE_COLS = ['Prediction', 'Risk Score', 'RNA Score', 'Protein Score', 'Metabolomics Score']


def render_risk_charts(results, mode="manual", key_prefix=""):
    st.subheader("Prediction & Risk Assessment")
    if mode == "manual":
        row = results.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Prediction",         row['Prediction'])
        c2.metric("Fusion Risk Score",  f"{row['Risk Score']:.2%}")
        c3.metric("RNA Score",          f"{row['RNA Score']:.2%}"          if row['RNA Score']          is not None else "N/A")
        c4.metric("Protein Score",      f"{row['Protein Score']:.2%}"      if row['Protein Score']      is not None else "N/A")
        c5.metric("Metabolomics Score", f"{row['Metabolomics Score']:.2%}" if row['Metabolomics Score'] is not None else "N/A")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(results, x="Risk Score", color="Prediction",
                               title="Risk Probability Distribution",
                               color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"},
                               nbins=20)
            fig.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Patients")
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_hist")
        with col2:
            rs = results.sort_values('Risk Score', ascending=False).reset_index(drop=True)
            rs['Patient_ID'] = rs.index
            fig2 = px.bar(rs, x='Patient_ID', y='Risk Score', color='Prediction',
                          title="Individual Patient Risk Scores",
                          color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"})
            fig2.add_hline(y=fusion_pkg['decision_threshold_default'], line_dash="dash",
                           line_color="gray",
                           annotation_text=f"Threshold ({fusion_pkg['decision_threshold_default']:.2f})")
            fig2.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_bar")

        st.divider()
        st.subheader("Risk Probability List")
        display = results[['Prediction', 'Risk Score', 'RNA Score', 'Protein Score', 'Metabolomics Score']].copy()
        display.insert(0, 'Patient ID', display.index)
        for col in ['Risk Score', 'RNA Score', 'Protein Score', 'Metabolomics Score']:
            display[col] = display[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) and x is not None else "N/A")
        st.dataframe(display, use_container_width=True, hide_index=True)


def render_dashboard(results, mode="manual", key_prefix="", patient_labels=None):
    render_risk_charts(results, mode=mode, key_prefix=key_prefix)

    if mode == "bulk":
        st.divider()
        st.subheader("Cohort Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        total     = len(results)
        high_risk = (results['Prediction'] == 'High Risk').sum()
        c1.metric("Total Patients",    total)
        c2.metric("High Risk",         f"{high_risk} ({high_risk/total:.1%})")
        c3.metric("Mean Risk Score",   f"{results['Risk Score'].mean():.2%}")
        c4.metric("Median Risk Score", f"{results['Risk Score'].median():.2%}")

    st.divider()
    st.subheader("Individual Patient Analysis")
    fmt = (lambda i: f"Patient {i} — {patient_labels[i]}" if patient_labels and i < len(patient_labels)
           else lambda i: f"Patient {i}")
    if callable(fmt):
        selected = st.selectbox("Select Patient Record", results.index,
                                format_func=fmt, key=f"{key_prefix}_select")
    else:
        selected = st.selectbox("Select Patient Record", results.index, key=f"{key_prefix}_select")

    row = results.iloc[selected]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Prediction",         row['Prediction'])
    c2.metric("Fusion Score",       f"{row['Risk Score']:.2%}")
    c3.metric("RNA Score",          f"{row['RNA Score']:.2%}"          if pd.notna(row['RNA Score'])          else "N/A")
    c4.metric("Protein Score",      f"{row['Protein Score']:.2%}"      if pd.notna(row['Protein Score'])      else "N/A")
    c5.metric("Metabolomics Score", f"{row['Metabolomics Score']:.2%}" if pd.notna(row['Metabolomics Score']) else "N/A")

    st.divider()
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.write("### Multi-Modal Signature")
        r_vals = [
            row['Protein Score']      if pd.notna(row['Protein Score'])      else 0,
            row['RNA Score']          if pd.notna(row['RNA Score'])          else 0,
            row['Metabolomics Score'] if pd.notna(row['Metabolomics Score']) else 0,
        ]
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=r_vals,
            theta=['Proteins', 'RNA', 'Metabolites'],
            fill='toself',
            line_color='#5dade2'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}_radar_{selected}")

    with col_r:
        st.write(f"### Top 20 Marker Levels (Patient {selected})")
        marker_cols = [c for c in results.columns if c not in SCORE_COLS]
        marker_vals = row[marker_cols].astype(float).dropna().sort_values(ascending=False).head(20)
        fig_top = px.bar(x=marker_vals.values, y=marker_vals.index, orientation='h',
                         color=marker_vals.values, color_continuous_scale='Viridis')
        fig_top.update_layout(yaxis_title="Biomarker", xaxis_title="Value")
        st.plotly_chart(fig_top, use_container_width=True, key=f"{key_prefix}_pbar_{selected}")

    st.divider()
    st.subheader(f"Biomarker Levels for Patient {selected}")
    col_imp1, col_imp2 = st.columns(2)

    with col_imp1:
        st.write("#### Patient's Top 15 Expressed Markers")
        marker_cols = [c for c in results.columns if c not in SCORE_COLS]
        patient_top = row[marker_cols].astype(float).dropna().sort_values(ascending=False).head(15)
        fig_pt = px.bar(x=patient_top.values, y=patient_top.index, orientation='h',
                        color=patient_top.values, color_continuous_scale='Viridis',
                        title=f"Highest Values — Patient {selected}")
        fig_pt.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_pt, use_container_width=True, key=f"{key_prefix}_ptop_{selected}")

    with col_imp2:
        st.write("#### Global Model Importance (Top 15)")
        fig_gi = px.bar(importance_df_display.head(15),
                        x='Influence Score', y='Biomarker', color='Layer',
                        orientation='h', title="Most Influential Markers Globally",
                        color_discrete_map={'RNA': '#5dade2', 'Protein': '#e74c3c', 'Metabolomics': '#27ae60'})
        fig_gi.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_gi, use_container_width=True, key=f"{key_prefix}_gimp_{selected}")

    with st.expander("View All Biomarker Values for This Patient"):
        marker_cols = [c for c in results.columns if c not in SCORE_COLS]
        all_markers = row[marker_cols].to_frame(name='Value').reset_index()
        all_markers.columns = ['Biomarker', 'Value']
        all_markers = all_markers.sort_values('Value', ascending=False)
        st.dataframe(all_markers, use_container_width=True, hide_index=True)


# =============================================================================
# REFERENCE TABLE HELPER
# =============================================================================
def build_reference_table():
    rows = []
    for pkg, layer in [(rna_pkg, 'RNA'), (prot_pkg, 'Protein'), (met_pkg, 'Metabolomics')]:
        for feat, sym in zip(pkg['features'], pkg['feature_symbols']):
            rr = pkg['reference_ranges'][feat]
            rows.append({
                'Layer': layer,
                'Feature ID': feat,
                'Symbol': sym,
                'GBM Mean': round(rr.get('gbm_mean', np.nan), 4),
                'Healthy Mean': round(rr.get('healthy_mean', np.nan), 4),
                'Min': round(rr['min'], 4),
                'Max': round(rr['max'], 4),
            })
    return pd.DataFrame(rows)


# =============================================================================
# DEMO DATA
# =============================================================================
DEMO_CSV_PATH = 'TGCA_DEMO_DATA.csv'

@st.cache_data
def load_demo_data():
    try:
        df = pd.read_csv(DEMO_CSV_PATH)
        id_col = 'Sample ID' if 'Sample ID' in df.columns else 'Sample_ID'
        sample_ids = df[id_col].tolist() if id_col in df.columns else [f"Patient {i}" for i in range(len(df))]
        df_data = df.drop(columns=[id_col], errors='ignore')
        # Remap symbols to IDs
        df_data = df_data.rename(columns=lambda c: SYMBOL_TO_ID.get(c, c))
        return df_data, sample_ids
    except FileNotFoundError:
        st.error(f"Demo data file '{DEMO_CSV_PATH}' not found.")
        st.stop()


# =============================================================================
# SIDEBAR & NAVIGATION
# =============================================================================
st.sidebar.title("MOmics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Home", "Documentation", "User Analysis", "Demo Walkthrough"])

# Model info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"RNA features: {len(rna_pkg['features'])}")
st.sidebar.markdown(f"Protein features: {len(prot_pkg['features'])}")
st.sidebar.markdown(f"Metabolomics features: {len(met_pkg['features'])}")
st.sidebar.markdown(f"Fusion threshold: {fusion_pkg['decision_threshold_default']:.3f}")

st.title("MOmics | GBM Clinical Diagnostic Suite")


# =============================================================================
# HOME PAGE
# =============================================================================
if page == "Home":
    try:
        from PIL import Image
        logo = Image.open('logo.png')
        st.image(logo, use_container_width=True)
    except Exception:
        st.info("Logo image not found. Please ensure 'logo.png' is in the root directory.")
    st.markdown("<h1 style='text-align: center;'>MOmics</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>GBM Clinical Diagnostic Suite</h3>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Model Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RNA Sub-model AUROC",          f"{rna_pkg.get('training_auroc', 0):.3f}")
    col2.metric("Protein Sub-model AUROC",      f"{prot_pkg.get('training_auroc', 0):.3f}")
    col3.metric("Metabolomics Sub-model AUROC", f"{met_pkg.get('training_auroc', 0):.3f}")
    col4.metric("Fusion Model AUROC",           f"{fusion_pkg.get('training_auroc', 0):.3f}")


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
        st.subheader("Feature Reference Ranges")
        st.write("GBM mean and healthy mean rank values used as defaults when a feature is not provided.")
        ref_df = build_reference_table()
        layer_filter = st.selectbox("Filter by layer", ["All", "RNA", "Protein", "Metabolomics"])
        if layer_filter != "All":
            ref_df = ref_df[ref_df['Layer'] == layer_filter]
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
            "Enter raw laboratory values for each biomarker. "
            "Leave fields at 0.0 to use GBM reference mean for that feature. "
            "You may omit entire layers — the fusion model handles missing layers natively."
        )

        user_inputs = {}

        st.write("#### RNA Biomarkers")
        rna_cols = st.columns(4)
        for i, (feat, sym) in enumerate(zip(rna_pkg['features'], rna_pkg['feature_symbols'])):
            with rna_cols[i % 4]:
                user_inputs[feat] = st.number_input(
                    sym, value=0.0, key=f"rna_{feat}",
                    help=f"Feature ID: {feat}"
                )

        st.write("#### Protein Biomarkers")
        prot_cols = st.columns(3)
        for i, (feat, sym) in enumerate(zip(prot_pkg['features'], prot_pkg['feature_symbols'])):
            with prot_cols[i % 3]:
                user_inputs[feat] = st.number_input(
                    sym, value=0.0, key=f"prot_{feat}",
                    help=f"Feature ID: {feat}"
                )

        st.write("#### Metabolomics Biomarkers")
        met_cols = st.columns(4)
        for i, (feat, sym) in enumerate(zip(met_pkg['features'], met_pkg['feature_symbols'])):
            with met_cols[i % 4]:
                user_inputs[feat] = st.number_input(
                    sym, value=0.0, key=f"met_{feat}",
                    help=f"Metabolite: {feat}"
                )

        if st.button("Analyze Single Patient", key="btn_manual", type="primary"):
            rna_d  = {f: user_inputs[f] for f in rna_pkg['features']}
            prot_d = {f: user_inputs[f] for f in prot_pkg['features']}
            met_d  = {f: user_inputs[f] for f in met_pkg['features']}
            result = predict_patient(rna_d, prot_d, met_d)
            # Add feature values for display
            for f in ALL_FEATURES:
                result[to_symbol(f)] = user_inputs.get(f, np.nan)
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
            template_csv = pd.DataFrame(columns=ALL_SYMBOLS).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Template",
                data=template_csv,
                file_name="MOmics_Patient_Template.csv",
                mime="text/csv",
                help="Fill in raw biomarker values. Columns are gene symbols."
            )
        with col_t1:
            st.write("### Upload Patient Data")
            uploaded_file = st.file_uploader(
                "Upload filled MOmics CSV Template", type="csv",
                help="CSV with gene symbol or feature ID column headers"
            )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded — {len(raw_df)} patient(s) found.")

                # Drop sample ID columns if present
                raw_df = raw_df.drop(columns=[c for c in raw_df.columns
                                               if c in ('Sample ID', 'Sample_ID')], errors='ignore')

                b_results = process_dataframe(raw_df)
                st.divider()
                st.subheader("Analysis Results")
                render_dashboard(b_results, mode="bulk", key_prefix="blk")
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Ensure your CSV uses gene symbol or Ensembl ID column headers.")

    # ── EXAMPLE ANALYSIS ──────────────────────────────────────────────────────
    with analysis_tabs[2]:
        st.subheader("Example Analysis")
        st.write(
            "This tab runs a real analysis on a CPTAC GBM patient sample. "
            "Click **Run Example Analysis** to see the full results dashboard."
        )
        col_ex1, col_ex2 = st.columns([1, 2])
        with col_ex1:
            st.markdown("**Input file:** `momics_input.csv`")
            st.markdown("**Sample:** CPTAC-3 GBM patient")
            st.markdown("**Format:** Raw expression / abundance values")

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
            with st.expander("Preview: feature reference ranges"):
                ref_df = build_reference_table()
                st.dataframe(ref_df, use_container_width=True, hide_index=True)

        if st.session_state.get("example_ran") and "example_results" in st.session_state:
            st.divider()
            st.subheader("Example Results")
            render_dashboard(
                st.session_state.example_results,
                mode="bulk",
                key_prefix="ex"
            )


# =============================================================================
# DEMO WALKTHROUGH PAGE
# =============================================================================
elif page == "Demo Walkthrough":
    st.header("Interactive Demo Workspace")
    st.markdown("""
    <div class="demo-box">
    <h3>Welcome to the Demo Workspace</h3>
    <p>This workspace uses <strong>real GBM patient data</strong> from the CPTAC dataset.
    Explore the full analysis workflow with genuine patient biomarker profiles.</p>
    <ul>
        <li>Real CPTAC GBM patients</li>
        <li>RNA, Protein, and Metabolomics layers</li>
        <li>Full 3-layer → fusion pipeline</li>
        <li>Interactive per-patient biomarker visualizations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    demo_data, demo_sample_ids = load_demo_data()
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
        <p>Click "Analyze Sample Patients" to run the full 3-layer diagnostic pipeline.</p>
        </div>""", unsafe_allow_html=True)

        with st.expander("Preview Sample Patient Data"):
            preview_cols = [c for c in demo_data.columns if c in rna_pkg['features']][:8]
            if not preview_cols:
                preview_cols = demo_data.columns[:8].tolist()
            st.dataframe(demo_data[preview_cols], use_container_width=True)

        if st.button("Analyze Sample Patients", key="analyze_demo_patients", type="primary"):
            with st.spinner("Analyzing biomarkers..."):
                st.session_state.demo_try_results = process_dataframe(demo_data)

        if 'demo_try_results' in st.session_state:
            st.markdown("---")
            st.success("Analysis Complete!")
            render_dashboard(
                st.session_state.demo_try_results,
                mode="bulk",
                key_prefix="demo",
                patient_labels=demo_sample_ids
            )

    # ── GUIDED TUTORIAL ───────────────────────────────────────────────────────
    elif demo_mode == "Guided Tutorial":
        st.subheader("Step-by-Step Guided Tutorial")
        if 'tutorial_step' not in st.session_state:
            st.session_state.tutorial_step = 0

        st.progress(st.session_state.tutorial_step / 4)
        st.write(f"**Progress:** Step {st.session_state.tutorial_step + 1} of 5")

        if st.session_state.tutorial_step == 0:
            st.markdown("""<div class="demo-box"><h3>Step 1: Understanding the Sample Data</h3>
            <p>Let's look at the pre-loaded CPTAC GBM patient data and the three omics layers.</p></div>""",
                        unsafe_allow_html=True)
            ref_df = build_reference_table()
            st.write("**Model feature layers:**")
            st.dataframe(ref_df[['Layer', 'Symbol', 'GBM Mean', 'Healthy Mean']],
                         use_container_width=True, hide_index=True)
            if st.button("Next: Run Analysis", key="tutorial_next_0"):
                st.session_state.tutorial_step = 1
                st.rerun()

        elif st.session_state.tutorial_step == 1:
            st.markdown("""<div class="demo-box"><h3>Step 2: Running the Analysis</h3>
            <p>Process the sample patients through the 3-layer fusion pipeline.</p></div>""",
                        unsafe_allow_html=True)
            if st.button("Process Sample Data", key="tutorial_analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    st.session_state.demo_results = process_dataframe(demo_data)
                    st.session_state.tutorial_step = 2
                st.rerun()

        elif st.session_state.tutorial_step == 2:
            st.markdown("""<div class="demo-box demo-success"><h3>Step 3: Cohort Results</h3></div>""",
                        unsafe_allow_html=True)
            if 'demo_results' in st.session_state:
                render_risk_charts(st.session_state.demo_results, mode="bulk", key_prefix="tutorial")
            if st.button("Next: Individual Patient", key="tutorial_next_2"):
                st.session_state.tutorial_step = 3
                st.rerun()

        elif st.session_state.tutorial_step == 3:
            st.markdown("""<div class="demo-box"><h3>Step 4: Individual Patient</h3></div>""",
                        unsafe_allow_html=True)
            if 'demo_results' in st.session_state:
                sel = st.selectbox("Choose patient:", range(len(demo_sample_ids)),
                                   format_func=lambda i: f"Patient {i} — {demo_sample_ids[i]}",
                                   key="tutorial_patient_select")
                row = st.session_state.demo_results.iloc[sel]
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Prediction",    row['Prediction'])
                c2.metric("Fusion Score",  f"{row['Risk Score']:.1%}")
                c3.metric("RNA",           f"{row['RNA Score']:.1%}"          if pd.notna(row['RNA Score'])          else "N/A")
                c4.metric("Protein",       f"{row['Protein Score']:.1%}"      if pd.notna(row['Protein Score'])      else "N/A")
                c5.metric("Metabolomics",  f"{row['Metabolomics Score']:.1%}" if pd.notna(row['Metabolomics Score']) else "N/A")

                marker_cols = [c for c in st.session_state.demo_results.columns if c not in SCORE_COLS]
                top10 = row[marker_cols].astype(float).dropna().sort_values(ascending=False).head(10)
                fig = px.bar(x=top10.values, y=top10.index, orientation='h',
                             title=f"Top 10 Biomarkers — Patient {sel}")
                st.plotly_chart(fig, use_container_width=True)

            if st.button("Next: Wrap Up", key="tutorial_next_3"):
                st.session_state.tutorial_step = 4
                st.rerun()

        elif st.session_state.tutorial_step == 4:
            st.markdown("""<div class="demo-box demo-success"><h3>Tutorial Complete!</h3>
            <p>You've learned the full MOmics workflow: data layers, pipeline, cohort results, and individual analysis.</p>
            </div>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.info("Navigate to 'User Analysis' to work with your own data")
            with c2:
                if st.button("🔄 Restart Tutorial", key="restart_tut"):
                    st.session_state.tutorial_step = 0
                    st.session_state.pop('demo_results', None)
                    st.rerun()

    # ── LEARN BY EXPLORING ────────────────────────────────────────────────────
    elif demo_mode == "Learn by Exploring":
        st.subheader("Free Exploration Mode")
        st.markdown("""<div class="demo-box"><h4>Explore at Your Own Pace</h4>
        <p>Full interface with real CPTAC GBM data.</p></div>""", unsafe_allow_html=True)

        exp_tabs = st.tabs(["Sample Analysis", "Learning Resources", "Tips & Tricks"])

        with exp_tabs[0]:
            if st.button("Load & Analyze Sample Data", key="explore_analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    st.session_state.demo_explore_results = process_dataframe(demo_data)
            if 'demo_explore_results' in st.session_state:
                st.success("Sample data analyzed!")
                st.divider()
                render_dashboard(
                    st.session_state.demo_explore_results,
                    mode="bulk",
                    key_prefix="explore",
                    patient_labels=demo_sample_ids
                )

        with exp_tabs[1]:
            st.write("### Quick Reference")
            with st.expander("Understanding Risk Scores"):
                st.write("- **0–30%**: Very Low Risk\n- **30–50%**: Low Risk\n- **50–70%**: Moderate-High Risk\n- **70–100%**: Very High Risk")
            with st.expander("Omics Layers"):
                st.write(
                    f"- **RNA ({len(rna_pkg['features'])} features):** Gene expression from RNA-seq. "
                    f"AUROC = {rna_pkg.get('training_auroc', 0):.3f}\n"
                    f"- **Protein ({len(prot_pkg['features'])} features):** Protein abundance from mass spectrometry. "
                    f"AUROC = {prot_pkg.get('training_auroc', 0):.3f}\n"
                    f"- **Metabolomics ({len(met_pkg['features'])} features):** Metabolite concentrations. "
                    f"AUROC = {met_pkg.get('training_auroc', 0):.3f}\n"
                    f"- **Fusion AUROC = {fusion_pkg.get('training_auroc', 0):.3f}**"
                )
            with st.expander("Input Format"):
                st.write(
                    "The model expects **raw values** (counts, abundance, concentration). "
                    "The app automatically applies within-sample rank normalization before scoring. "
                    "Missing layers are handled natively — XGBoost treats them as NaN in the fusion step."
                )

        with exp_tabs[2]:
            st.write("### Exploration Tips")
            st.info(
                "**Things to Try:**\n"
                "1. Compare Low Risk vs High Risk patient radar charts — look for layer-level differences\n"
                "2. Check whether RNA, Protein, or Metabolomics score drives the fusion risk score\n"
                "3. Download the CSV template and try uploading your own data in Bulk Upload"
            )

    st.divider()
    if st.button("Reset Demo Workspace"):
        for key in [k for k in st.session_state if 'demo' in k or 'tutorial' in k]:
            del st.session_state[key]
        st.rerun()
