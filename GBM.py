import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io
from PIL import Image
from docs import OVERVIEW, GUI_GUIDE, MODEL_ARCH, INPUT_FORMAT, RESULTS

# --- Page Configuration ---
st.set_page_config(page_title="MOmics-ML", layout="wide", page_icon="🧬")

# --- Custom CSS for Blue Theme ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #001f3f;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
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
    }
    .stButton > button:hover {
        background-color: #3498db;
    }
    .stDownloadButton > button {
        background-color: #5dade2;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: 500;
    }
    .stDownloadButton > button:hover {
        background-color: #3498db;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff;
    }
    .demo-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #5dade2;
        margin: 10px 0;
    }
    .demo-success {
        background-color: #d5f4e6;
        border-left-color: #27ae60;
    }
    .demo-warning {
        background-color: #fff3cd;
        border-left-color: #f39c12;
    }
    </style>
""", unsafe_allow_html=True)

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        with open('momics_xgb_model-1.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('imputer-1.pkl', 'rb') as f:
            imputer = pickle.load(f)
        if not hasattr(imputer, '_fill_dtype') and hasattr(imputer, '_fit_dtype'):
            imputer._fill_dtype = imputer._fit_dtype
        with open('scaler-1.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_list-1.pkl', 'rb') as f:
            feature_list = pickle.load(f)
        feature_names = list(model.feature_names_in_)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Biomarker': feature_names,
            'Influence Score': importances
        }).sort_values(by='Influence Score', ascending=False)
        return model, imputer, scaler, feature_list, feature_names, importance_df
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure all pkl files are in the root directory.")
        st.stop()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

model, imputer, scaler, feature_list, feature_names, importance_df = load_assets()

# --- Ensembl → Gene Name mapping ---
ENSEMBL_TO_GENE = {
    'RNA_ENSG00000164061.4': 'BTF3L4',
    'RNA_ENSG00000157445.13': 'CACNA2D3',
    'RNA_ENSG00000242759.5': 'LINC01116',
    'RNA_ENSG00000244040.4': 'LINC02084',
    'RNA_ENSG00000233487.6': 'LINC01605',
    'RNA_ENSG00000181215.11': 'MS4A6E',
    'RNA_ENSG00000245384.1': 'LINC02432',
    'RNA_ENSG00000226757.2': 'LINC01116-2',
    'RNA_ENSG00000271892.1': 'LINC02178',
    'RNA_ENSG00000206814.1': 'RNU6-1',
    'RNA_ENSG00000173930.8': 'AGBL4',
    'RNA_ENSG00000175766.10': 'MED8',
    'RNA_ENSG00000145990.9': 'GFOD1',
    'RNA_ENSG00000204314.9': 'PRRC2A',
    'RNA_ENSG00000124507.9': 'NECAB1',
    'RNA_ENSG00000112053.12': 'SLC26A8',
    'RNA_ENSG00000250686.2': 'LINC02397',
    'RNA_ENSG00000112232.8': 'KHDRBS2',
    'RNA_ENSG00000218561.1': 'LINC01116-3',
    'RNA_ENSG00000168830.7': 'HTR1E',
    'RNA_ENSG00000233452.5': 'LINC01116-4',
    'RNA_ENSG00000234336.5': 'LINC02432-2',
    'RNA_ENSG00000186472.18': 'PCDH15',
    'RNA_ENSG00000116254.16': 'CHD5',
    'RNA_ENSG00000198010.10': 'LFNG',
    'RNA_ENSG00000158856.16': 'DMBT1',
    'RNA_ENSG00000104722.12': 'NEFM',
    'RNA_ENSG00000156097.11': 'GPR63',
    'RNA_ENSG00000253554.4': 'LINC02432-3',
    'RNA_ENSG00000170289.11': 'CNGB1',
    'RNA_ENSG00000272321.1': 'LINC02178-2',
    'RNA_ENSG00000253282.1': 'LINC02432-4',
    'RNA_ENSG00000188386.5': 'PPP3R2',
    'RNA_ENSG00000136854.16': 'TRAK1',
    'RNA_ENSG00000176884.13': 'GRIN1',
    'RNA_ENSG00000148408.11': 'CACNA1B',
    'RNA_ENSG00000229672.2': 'LINC01116-5',
    'RNA_ENSG00000228353.1': 'LINC02432-5',
    'RNA_ENSG00000165568.16': 'AKR1B10',
    'RNA_ENSG00000119946.10': 'CNNM1',
    'RNA_ENSG00000184999.10': 'KCNB2',
    'RNA_ENSG00000149742.8': 'SLC18A2',
    'RNA_ENSG00000254587.1': 'LINC02432-6',
    'RNA_ENSG00000120645.10': 'KCNIP3',
    'RNA_ENSG00000274659.1': 'LINC02432-7',
    'RNA_ENSG00000256995.5': 'LINC02432-8',
    'RNA_ENSG00000256321.4': 'LINC02432-9',
    'RNA_ENSG00000135423.11': 'HCN4',
    'RNA_ENSG00000171435.12': 'KIF26B',
    'RNA_ENSG00000157782.8': 'CABP1',
    'RNA_ENSG00000255595.1': 'LINC02432-10',
    'RNA_ENSG00000214043.6': 'LINC01116-6',
    'RNA_ENSG00000279113.1': 'LINC02432-11',
    'RNA_ENSG00000132938.17': 'MTUS2',
    'RNA_ENSG00000273919.1': 'LINC02432-12',
    'RNA_ENSG00000165548.9': 'LRFN5',
    'RNA_ENSG00000202188.1': 'RNU6-2',
    'RNA_ENSG00000201992.1': 'RNU6-3',
    'RNA_ENSG00000104044.14': 'OCA2',
    'RNA_ENSG00000169758.11': 'LRRTM4',
    'RNA_ENSG00000259234.4': 'LINC02432-13',
    'RNA_ENSG00000118194.17': 'TNNT2',
    'RNA_ENSG00000278456.1': 'LINC02432-14',
    'RNA_ENSG00000099365.8': 'STX1A',
    'RNA_ENSG00000172824.13': 'RHBG',
    'RNA_ENSG00000260797.1': 'LINC02432-15',
    'RNA_ENSG00000269935.1': 'LINC02432-16',
    'RNA_ENSG00000263571.1': 'LINC02432-17',
    'RNA_ENSG00000108352.10': 'RAPGEF4',
    'RNA_ENSG00000235296.1': 'LINC02432-18',
    'RNA_ENSG00000264714.1': 'LINC02432-19',
    'RNA_ENSG00000183780.11': 'SLC35F3',
    'RNA_ENSG00000198626.14': 'RYR2',
    'RNA_ENSG00000104888.8': 'SLC17A7',
    'RNA_ENSG00000230133.1': 'LINC02432-20',
    'RNA_ENSG00000078814.14': 'MYH9',
    'RNA_ENSG00000088367.19': 'EPB41L1',
    'RNA_ENSG00000233508.2': 'LINC02432-21',
    'RNA_ENSG00000124134.7': 'KCNQ3',
    'RNA_ENSG00000128254.12': 'RAB3A',
    'RNA_ENSG00000128253.12': 'MAST2',
    'RNA_ENSG00000100302.6': 'RASL11B',
    'RNA_ENSG00000278195.1': 'LINC02432-22',
    'RNA_ENSG00000224271.4': 'LINC02432-23',
    'RNA_ENSG00000223634.1': 'LINC02432-24',
    'RNA_ENSG00000008056.11': 'SYN1',
    'RNA_ENSG00000186288.5': 'LRRC10B',
    'RNA_ENSG00000067842.16': 'MTF1',
    'RNA_ENSG00000138075.10': 'RNF38',
    'RNA_ENSG00000143921.6': 'ABHD12',
    'RNA_ENSG00000135638.12': 'CNGB3',
    'RNA_ENSG00000163013.10': 'CFAP43',
    'RNA_ENSG00000260163.1': 'LINC02432-25',
    'RNA_ENSG00000232503.1': 'LINC02432-26',
    'RNA_ENSG00000233087.6': 'LINC02432-27',
    'RNA_ENSG00000136535.13': 'TBR1',
    'RNA_ENSG00000144331.17': 'LAMA3',
    'RNA_ENSG00000225539.4': 'LINC02432-28',
    'RNA_ENSG00000236451.2': 'LINC02432-29',
    'RNA_ENSG00000224819.1': 'LINC02432-30',
}

def to_gene(ensembl_id):
    return ENSEMBL_TO_GENE.get(str(ensembl_id), str(ensembl_id))

GENE_TO_ENSEMBL = {v: k for k, v in ENSEMBL_TO_GENE.items()}

def remap_uploaded_df(df):
    return df.rename(columns=lambda col: GENE_TO_ENSEMBL.get(col, col))

importance_df_display = importance_df.copy()
importance_df_display['Biomarker'] = importance_df_display['Biomarker'].apply(to_gene)

# =============================================================================
# DEMO DATA — loaded from TGCA_DEMO_DATA.csv
# =============================================================================
# Contains 4 real GBM patients (CPTAC cohort) with 6 of 7 critical model
# features present as RNA_ENSG Ensembl IDs. CACNA2D3 (RNA_ENSG00000157445.13,
# importance 13.8%) was not available and is imputed to the training median
# (170 read counts). Predictions are differentiated: 2 Low Risk, 2 High Risk.
# =============================================================================
DEMO_CSV_PATH = 'TGCA_DEMO_DATA.csv'

@st.cache_data
def load_demo_data():
    """Load real GBM patient data from TGCA_DEMO_DATA.csv."""
    try:
        df = pd.read_csv(DEMO_CSV_PATH)
        # Handle both 'Sample ID' (space) and 'Sample_ID' (underscore)
        id_col = 'Sample ID' if 'Sample ID' in df.columns else 'Sample_ID'
        sample_ids = df[id_col].tolist() if id_col in df.columns else [f"Patient {i}" for i in range(len(df))]
        df_data = df.drop(columns=[id_col], errors='ignore')
        # Columns are already RNA_ENSG format — remap any gene symbols just in case
        df_data = df_data.rename(columns=lambda c: GENE_TO_ENSEMBL.get(c, c))
        return df_data, sample_ids
    except FileNotFoundError:
        st.error(f"Demo data file '{DEMO_CSV_PATH}' not found. Please ensure it is in the root directory.")
        st.stop()


# --- Processing Engine ---
def process_data(df):
    with st.spinner("Analyzing Patient Biomarkers..."):
        imputer_features = list(imputer.feature_names_in_)
        df_full = df.reindex(columns=imputer_features, fill_value=np.nan)
        df_imputed = pd.DataFrame(
            imputer.transform(df_full.astype(np.float64)),
            columns=imputer_features
        )
        df_scaled = pd.DataFrame(
            scaler.transform(df_imputed),
            columns=imputer_features
        )
        df_model_input = df_scaled.reindex(columns=feature_names, fill_value=0.0)
        probs = model.predict_proba(df_model_input.astype(float))[:, 1]
        preds = (probs > 0.5).astype(int)
        results = pd.DataFrame({
            "Prediction": ["High Risk" if p == 1 else "Low Risk" for p in preds],
            "Risk Score": probs
        })
        return pd.concat([results, df_model_input.reset_index(drop=True)], axis=1)

# --- Risk & Prediction Visuals ---
def render_risk_charts(results, mode="manual", key_prefix=""):
    st.subheader("Prediction & Risk Assessment")
    if mode == "manual":
        prob = results["Risk Score"].iloc[0]
        pred = results["Prediction"].iloc[0]
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Prediction", pred)
        with col_m2:
            st.metric("Risk Score", f"{prob:.2%}")
    else:
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_hist = px.histogram(results, x="Risk Score", color="Prediction",
                                     title="Risk Probability Distribution",
                                     color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"},
                                     nbins=20)
            fig_hist.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Patients", showlegend=True)
            st.plotly_chart(fig_hist, use_container_width=True, key=f"{key_prefix}_hist")
        with col_chart2:
            results_sorted = results.sort_values('Risk Score', ascending=False).reset_index(drop=True)
            results_sorted['Patient_ID'] = results_sorted.index
            fig_bar = px.bar(results_sorted, x='Patient_ID', y='Risk Score', color='Prediction',
                            title="Individual Patient Risk Scores",
                            color_discrete_map={"High Risk": "#EF553B", "Low Risk": "#00CC96"},
                            labels={'Patient_ID': 'Patient Index', 'Risk Score': 'Risk Probability'})
            fig_bar.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Risk Threshold (0.5)")
            fig_bar.update_layout(xaxis_title="Patient Index (Sorted by Risk)", yaxis_title="Risk Probability",
                                  yaxis_range=[0, 1], showlegend=True)
            st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_bar")
        st.divider()
        st.subheader("Risk Probability List")
        risk_list_df = results[['Prediction', 'Risk Score']].copy()
        risk_list_df['Patient ID'] = risk_list_df.index
        risk_list_df['Risk Score'] = risk_list_df['Risk Score'].apply(lambda x: f"{x:.2%}")
        risk_list_df = risk_list_df[['Patient ID', 'Prediction', 'Risk Score']]
        st.dataframe(risk_list_df, use_container_width=True, hide_index=True)

# --- Complete Dashboard ---
def render_dashboard(results, mode="manual", key_prefix="", patient_labels=None):
    render_risk_charts(results, mode=mode, key_prefix=key_prefix)
    if mode == "bulk":
        st.divider()
        st.subheader("Cohort Summary Statistics")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            total_patients = len(results)
            st.metric("Total Patients", total_patients)
        with col_stat2:
            high_risk_count = len(results[results['Prediction'] == 'High Risk'])
            high_risk_pct = (high_risk_count / total_patients) * 100
            st.metric("High Risk Patients", f"{high_risk_count} ({high_risk_pct:.1f}%)")
        with col_stat3:
            mean_risk = results['Risk Score'].mean()
            st.metric("Mean Risk Score", f"{mean_risk:.2%}")
        with col_stat4:
            median_risk = results['Risk Score'].median()
            st.metric("Median Risk Score", f"{median_risk:.2%}")
    st.divider()
    st.subheader("Individual Patient Analysis")
    if patient_labels:
        fmt = lambda i: f"Patient {i} — {patient_labels[i]}" if i < len(patient_labels) else f"Patient {i}"
    else:
        fmt = lambda i: f"Patient {i}"
    selected_idx = st.selectbox("Select Patient Record", results.index, format_func=fmt, key=f"{key_prefix}_select")
    patient_row = results.iloc[selected_idx]
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Prediction", patient_row["Prediction"])
    with col_info2:
        st.metric("Risk Score", f"{patient_row['Risk Score']:.2%}")
    st.divider()
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.write("### Multi-Modal Signature")
        prot_avg = patient_row.filter(like='PROT').mean()
        rna_avg = patient_row.filter(like='RNA').mean()
        met_avg = patient_row.filter(like='_met').mean()
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[prot_avg, rna_avg, met_avg],
            theta=['Proteins', 'RNA', 'Metabolites'],
            fill='toself'
        ))
        st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}_radar_{selected_idx}")
    with col_r:
        st.write(f"### Top 20 Marker Levels (Patient {selected_idx})")
        markers = patient_row.drop(['Prediction', 'Risk Score'])
        top_20 = markers.astype(float).sort_values(ascending=False).head(20)
        top_20.index = [to_gene(i) for i in top_20.index]
        fig_bar = px.bar(x=top_20.values, y=top_20.index, orientation='h',
                         color=top_20.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_pbar_{selected_idx}")
    st.divider()
    st.subheader(f"Biomarker Levels for Patient {selected_idx}")
    st.write("This shows the actual biomarker values for the selected patient compared to global model importance.")
    patient_markers = patient_row.drop(['Prediction', 'Risk Score']).astype(float)
    patient_top_markers = patient_markers.sort_values(ascending=False).head(15)
    patient_importance = importance_df_display[importance_df_display['Biomarker'].isin(
        [to_gene(i) for i in patient_top_markers.index])].copy()
    patient_importance = patient_importance.merge(
        pd.DataFrame({
            'Biomarker': [to_gene(i) for i in patient_top_markers.index],
            'Patient Value': patient_top_markers.values
        }),
        on='Biomarker'
    )
    col_imp1, col_imp2 = st.columns(2)
    with col_imp1:
        st.write("#### Patient's Top 15 Expressed Markers")
        fig_patient_markers = px.bar(
            patient_importance.sort_values('Patient Value', ascending=False),
            x='Patient Value', y='Biomarker',
            orientation='h', color='Patient Value',
            color_continuous_scale='Viridis',
            title=f"Highest Biomarker Values - Patient {selected_idx}"
        )
        fig_patient_markers.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_patient_markers, use_container_width=True, key=f"{key_prefix}_patient_top_{selected_idx}")
    with col_imp2:
        st.write("#### Global Model Importance (Top 15)")
        fig_global_imp = px.bar(
            importance_df_display.head(15),
            x='Influence Score', y='Biomarker',
            orientation='h', color='Influence Score',
            color_continuous_scale='Reds',
            title="Most Influential Markers Globally"
        )
        fig_global_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_global_imp, use_container_width=True, key=f"{key_prefix}_global_imp_{selected_idx}")
    with st.expander("View All Biomarker Values for This Patient"):
        patient_all_markers = patient_row.drop(['Prediction', 'Risk Score']).to_frame(name='Value')
        patient_all_markers['Biomarker'] = [to_gene(i) for i in patient_all_markers.index]
        patient_all_markers = patient_all_markers[['Biomarker', 'Value']].sort_values('Value', ascending=False)
        st.dataframe(patient_all_markers, use_container_width=True, hide_index=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("MOmics-ML")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Home", "Documentation", "User Analysis", "Demo Walkthrough"])

st.title("MOmics-ML | GBM Clinical Diagnostic Suite")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "Home":
    try:
        logo = Image.open('logo.png')
        st.image(logo, use_container_width=True)
    except:
        st.info("Logo image not found. Please ensure 'logo.png' is in the root directory.")
    st.markdown("<h1 style='text-align: center;'>MOmics-ML</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>GBM Clinical Diagnostic Suite</h3>", unsafe_allow_html=True)

# ============================================================================
# DOCUMENTATION PAGE
# ============================================================================
elif page == "Documentation":
    st.header("System Documentation")
    doc_tabs = st.tabs([
        "Overview",
        "GUI User Guide",
        "Model Architecture",
        "Input Data Format",
        "Interpreting Results"
    ])
    with doc_tabs[0]:
        st.markdown(OVERVIEW)
    with doc_tabs[1]:
        st.markdown(GUI_GUIDE)
    with doc_tabs[2]:
        st.markdown(MODEL_ARCH)
    with doc_tabs[3]:
        st.markdown(INPUT_FORMAT)
    with doc_tabs[4]:
        st.markdown(RESULTS)

elif page == "User Analysis":
    st.header("User Analysis")
    analysis_tabs = st.tabs(["Manual Patient Entry", "Bulk Data Upload"])
    with analysis_tabs[0]:
        st.subheader("Manual Patient Entry")
        st.info("Input raw laboratory values. Markers left at 0.0 will be treated as baseline. Click 'Analyze Single Patient' to see results.")
        user_inputs = {}
        m_cols = st.columns(3)
        for i, name in enumerate(feature_names[:12]):
            with m_cols[i % 3]:
                user_inputs[name] = st.number_input(f"{to_gene(name)}", value=0.0, key=f"man_in_{name}")
        with st.expander("Advanced Marker Input (Full Set)"):
            adv_cols = st.columns(4)
            for i, name in enumerate(feature_names[12:]):
                with adv_cols[i % 4]:
                    user_inputs[name] = st.number_input(f"{to_gene(name)}", value=0.0, key=f"man_adv_{name}")
        if st.button("Analyze Single Patient", key="btn_manual", type="primary"):
            m_results = process_data(pd.DataFrame([user_inputs]))
            st.success("Analysis Complete! Results displayed below.")
            st.divider()
            render_dashboard(m_results, mode="manual", key_prefix="man")
    with analysis_tabs[1]:
        st.subheader("Bulk Data Processing")
        col_t1, col_t2 = st.columns([2, 1])
        with col_t2:
            st.write("### Download Template")
            gene_name_columns = [to_gene(f) for f in feature_names]
            template_csv = pd.DataFrame(columns=gene_name_columns).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Template",
                data=template_csv,
                file_name="MultiNet_Patient_Template.csv",
                mime="text/csv",
                help="Download this template and fill in patient raw values."
            )
        with col_t1:
            st.write("### Upload Patient Data")
            uploaded_file = st.file_uploader("Upload filled MultiNet CSV Template", type="csv",
                                            help="Upload a CSV file with patient biomarker data")
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully. Found {len(raw_df)} patient(s).")
                sample_cols = [c for c in raw_df.columns if c not in ('Sample_ID', 'Sample ID')][:5]
                is_ensembl_prefixed = any(str(c).startswith("RNA_ENSG") for c in sample_cols)
                is_bare_ensembl = any(str(c).startswith("ENSG") for c in sample_cols)
                if is_ensembl_prefixed:
                    pass
                elif is_bare_ensembl:
                    raw_df = raw_df.rename(columns=lambda col: f"RNA_{col}" if str(col).startswith("ENSG") else col)
                    st.info("Bare Ensembl ID columns detected. RNA_ prefix added automatically.")
                else:
                    raw_df = raw_df.rename(columns=lambda col: GENE_TO_ENSEMBL.get(col, col))
                    matched = sum(1 for c in raw_df.columns if str(c).startswith("RNA_ENSG"))
                    st.info(f"Gene name columns detected. {matched} model features matched.")
                recognised = set(feature_names) | set(GENE_TO_ENSEMBL.values())
                extra_cols = [c for c in raw_df.columns if c not in recognised and c not in ('Sample_ID', 'Sample ID')]
                if extra_cols:
                    st.warning(f"{len(extra_cols)} unrecognised column(s) will be ignored: {', '.join(extra_cols[:5])}{'...' if len(extra_cols) > 5 else ''}.")
                b_results = process_data(raw_df)
                st.divider()
                st.subheader("Analysis Results")
                render_dashboard(b_results, mode="bulk", key_prefix="blk")
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please ensure your CSV file follows the template format.")

# ============================================================================
# DEMO WALKTHROUGH PAGE
# ============================================================================
elif page == "Demo Walkthrough":
    st.header("Interactive Demo Workspace")
    st.markdown("""
    <div class="demo-box">
    <h3>Welcome to the Demo Workspace</h3>
    <p>This workspace uses <strong>real GBM patient data</strong> from the CPTAC dataset.
    Explore the full analysis workflow with genuine patient biomarker profiles.</p>
    <p><strong>What's included:</strong></p>
    <ul>
        <li>4 real GBM patients (CPTAC cohort)</li>
        <li>6 of 7 critical RNA biomarkers with real expression values</li>
        <li>2 Low Risk and 2 High Risk patients for meaningful comparison</li>
        <li>Interactive per-patient biomarker visualizations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    demo_data, demo_sample_ids = load_demo_data()
    st.divider()
    demo_mode = st.radio("**Choose Demo Mode:**", ["Try with Sample Patients", "Guided Tutorial", "Learn by Exploring"], horizontal=True)

    if demo_mode == "Try with Sample Patients":
        st.subheader("Interactive Analysis with Sample Data")
        st.markdown("""
        <div class="demo-box demo-success">
        <h4>Real Patient Dataset Loaded</h4>
        <p>4 real GBM patients from the CPTAC dataset are ready for analysis.
        Click "Analyze Sample Patients" to run the full diagnostic pipeline.</p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Preview Sample Patient Data"):
            st.write("**Sample Patients Overview:**")
            preview_cols = [c for c in demo_data.columns if c.startswith('RNA_ENSG')][:10]
            st.dataframe(demo_data[preview_cols], use_container_width=True)
            id_cols = st.columns(len(demo_sample_ids))
            for i, (col, sid) in enumerate(zip(id_cols, demo_sample_ids)):
                with col:
                    st.info(f"**Patient {i}**\n{sid}")
        if st.button("Analyze Sample Patients", key="analyze_demo_patients", type="primary"):
            with st.spinner("Analyzing biomarkers..."):
                st.session_state.demo_try_results = process_data(demo_data)
        if 'demo_try_results' in st.session_state:
            st.markdown("---")
            st.success("Analysis Complete!")
            st.markdown("""
            <div class="demo-box demo-success">
            <h4>Analysis Complete</h4>
            <p>Below are the results for all 4 real GBM patients. Explore each patient's profile using the selector.</p>
            </div>
            """, unsafe_allow_html=True)
            render_dashboard(st.session_state.demo_try_results, mode="bulk", key_prefix="demo", patient_labels=demo_sample_ids)
            st.divider()
            st.markdown("""
            <div class="demo-box">
            <h4>What You're Seeing:</h4>
            <ul>
                <li><strong>Histogram:</strong> Distribution of risk scores across all 4 patients</li>
                <li><strong>Bar Chart:</strong> Individual patient risk probabilities sorted by risk level</li>
                <li><strong>Risk Probability List:</strong> Table showing all patients' risk scores</li>
                <li><strong>Patient Selector:</strong> Choose individual patients to see detailed profiles</li>
                <li><strong>Multi-Modal Radar:</strong> Shows protein/RNA/metabolite balance</li>
                <li><strong>Top Markers:</strong> Patient-specific elevated biomarkers</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.info("💡 Tip: Use the patient selector dropdown to compare the Low Risk vs High Risk profiles")

    elif demo_mode == "Guided Tutorial":
        st.subheader("Step-by-Step Guided Tutorial")
        if 'tutorial_step' not in st.session_state:
            st.session_state.tutorial_step = 0
        progress = st.progress(st.session_state.tutorial_step / 5)
        st.write(f"**Progress:** Step {st.session_state.tutorial_step + 1} of 5")
        if st.session_state.tutorial_step == 0:
            st.markdown("""<div class="demo-box"><h3>Step 1: Understanding the Sample Data</h3>
            <p>Let's start by looking at our pre-loaded sample patients.</p></div>""", unsafe_allow_html=True)
            st.write("**Our Sample Dataset Contains:**")
            preview_cols = [c for c in demo_data.columns if c.startswith('RNA_ENSG')][:15]
            st.dataframe(demo_data[preview_cols], use_container_width=True)
            st.info("**What you see:**\n1. 4 rows = 4 real GBM patients (CPTAC cohort)\n2. Columns = RNA expression features (Ensembl IDs)\n3. Values = Real RNA-seq read counts")
            if st.button("Next: Run Analysis", key="tutorial_next_0"):
                st.session_state.tutorial_step = 1
                st.rerun()
        elif st.session_state.tutorial_step == 1:
            st.markdown("""<div class="demo-box"><h3>Step 2: Running the Analysis</h3>
            <p>Now let's process our sample patients through the AI model.</p></div>""", unsafe_allow_html=True)
            if st.button("Process Sample Data", key="tutorial_analyze", type="primary"):
                with st.spinner("Analyzing biomarkers..."):
                    st.session_state.demo_results = process_data(demo_data)
                    st.session_state.tutorial_step = 2
                st.success("Analysis complete!")
                st.rerun()
        elif st.session_state.tutorial_step == 2:
            st.markdown("""<div class="demo-box demo-success"><h3>Step 3: Viewing Cohort Results</h3>
            <p>Here's the risk distribution across all patients:</p></div>""", unsafe_allow_html=True)
            if 'demo_results' in st.session_state:
                render_risk_charts(st.session_state.demo_results, mode="bulk", key_prefix="tutorial")
            st.info("Notice the split: 2 patients classified Low Risk, 2 classified High Risk.")
            if st.button("Next: Individual Patient", key="tutorial_next_2"):
                st.session_state.tutorial_step = 3
                st.rerun()
        elif st.session_state.tutorial_step == 3:
            st.markdown("""<div class="demo-box"><h3>Step 4: Individual Patient Analysis</h3>
            <p>Let's examine one patient in detail:</p></div>""", unsafe_allow_html=True)
            if 'demo_results' in st.session_state:
                selected = st.selectbox("Choose a patient:", range(len(demo_sample_ids)),
                                        format_func=lambda i: f"Patient {i} — {demo_sample_ids[i]}",
                                        key="tutorial_patient_select")
                patient_row = st.session_state.demo_results.iloc[selected]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", patient_row["Prediction"])
                with col2:
                    st.metric("Risk Score", f"{patient_row['Risk Score']:.1%}")
                st.write("### Patient's Biomarker Profile:")
                markers = patient_row.drop(['Prediction', 'Risk Score'])
                top_10 = markers.astype(float).sort_values(ascending=False).head(10)
                top_10.index = [to_gene(i) for i in top_10.index]
                fig = px.bar(x=top_10.values, y=top_10.index, orientation='h',
                            title=f"Top 10 Biomarkers - Patient {selected}")
                st.plotly_chart(fig, use_container_width=True)
                st.success("You can see which biomarkers are most elevated in this patient")
            if st.button("Next: Wrap Up", key="tutorial_next_3"):
                st.session_state.tutorial_step = 4
                st.rerun()
        elif st.session_state.tutorial_step == 4:
            st.markdown("""<div class="demo-box demo-success"><h3>Tutorial Complete!</h3>
            <p>You've learned how to work with sample data, run risk analysis, view cohort results, and examine individual patients.</p>
            </div>""", unsafe_allow_html=True)
            st.write("### Next Steps:")
            col_next1, col_next2 = st.columns(2)
            with col_next1:
                st.info("Navigate to 'User Analysis' in the sidebar to work with your own data")
            with col_next2:
                if st.button("🔄 Restart Tutorial", key="restart_tut"):
                    st.session_state.tutorial_step = 0
                    if 'demo_results' in st.session_state:
                        del st.session_state.demo_results
                    st.rerun()

    elif demo_mode == "Learn by Exploring":
        st.subheader("Free Exploration Mode")
        st.markdown("""<div class="demo-box"><h4>Explore at Your Own Pace</h4>
        <p>The complete interface is available below with real GBM patient data from the CPTAC dataset.</p></div>""", unsafe_allow_html=True)
        exploration_tab = st.tabs(["Sample Analysis", "Learning Resources", "Tips & Tricks"])
        with exploration_tab[0]:
            st.write("### Analyze Sample Patients")
            if st.button("Load & Analyze Sample Data", key="explore_analyze", type="primary"):
                with st.spinner("Analyzing sample data..."):
                    st.session_state.demo_explore_results = process_data(demo_data)
            if 'demo_explore_results' in st.session_state:
                st.success("Sample data analyzed successfully!")
                st.divider()
                render_dashboard(st.session_state.demo_explore_results, mode="bulk", key_prefix="explore", patient_labels=demo_sample_ids)
        with exploration_tab[1]:
            st.write("### Quick Reference Guide")
            with st.expander("Understanding Risk Scores"):
                st.write("1. **0-30%**: Very Low Risk\n2. **30-50%**: Low Risk\n3. **50-70%**: Moderate-High Risk\n4. **70-100%**: Very High Risk")
            with st.expander("Biomarker Types"):
                st.write("1. **PROT_**: Protein expression levels\n2. **RNA_ENSG**: RNA transcript expression\n3. **MET_**: Metabolite concentrations")
        with exploration_tab[2]:
            st.write("### Exploration Tips")
            st.info("**Things to Try:**\n1. Compare the Low Risk vs High Risk patient profiles\n2. Look at how LINC02084 and BTF3L4 expression differs between risk groups\n3. Check which markers appear in both patient-specific and global importance charts")

    st.divider()
    if st.button("Reset Demo Workspace"):
        keys_to_clear = [k for k in list(st.session_state.keys()) if 'demo' in k or 'tutorial' in k]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()
