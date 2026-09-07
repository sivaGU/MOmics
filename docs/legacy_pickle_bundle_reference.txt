MOmics-ML GUI Integration Guide
================================
 
Files in this package:
- momics_rna_model.pkl
- momics_prot_model.pkl
- momics_met_model.pkl
- momics_fusion_model.pkl
- momics_feature_metadata.pkl
- momics_reference_ranges.pkl
 
LOADING IN STREAMLIT:
---------------------
import pickle
 
with open('momics_rna_model.pkl', 'rb') as f:
    rna_pkg = pickle.load(f)
with open('momics_prot_model.pkl', 'rb') as f:
    prot_pkg = pickle.load(f)
with open('momics_met_model.pkl', 'rb') as f:
    met_pkg = pickle.load(f)
with open('momics_fusion_model.pkl', 'rb') as f:
    fusion_pkg = pickle.load(f)
 
EACH PACKAGE CONTAINS:
- 'model'             : trained XGBoost classifier
- 'features'          : list of Ensembl IDs (RNA/PROT) or symbols (MET)
- 'feature_symbols'   : human-readable gene symbols for display
- 'feature_names'     : full gene names
- 'reference_ranges'  : per-feature {min,max,mean,gbm_mean,healthy_mean}
- 'feature_type'      : 'within_sample_rank' (input format)
 
PREDICTION PIPELINE:
--------------------
1. User enters values for each feature (in the GUI, allow gene-symbol search
   instead of raw Ensembl IDs - use 'feature_symbols' for display)
 
2. INPUT FORMAT: All features expect within-sample rank values in [0, 1].
   For users entering RAW expression / abundance values, the GUI must
   rank-transform within the user's sample before scoring:
 
       def to_rank(values_array):
           from scipy.stats import rankdata
           ranks = rankdata(values_array, method='average')
           return (ranks - 1) / (len(values_array) - 1)
 
   In practice, users typically only have a few feature values, not a full
   transcriptome. For partial input, ask the user to provide values relative
   to a typical sample (e.g. the median GBM training value -- shown as
   'gbm_mean' in reference_ranges).
 
3. PER-LAYER PREDICTION:
       import numpy as np
       rna_input  = np.array([[user_values_for_rna_features]])  # shape (1, 8)
       rna_prob   = rna_pkg['model'].predict_proba(rna_input)[:, 1][0]
       prot_input = np.array([[user_values_for_prot_features]]) # shape (1, 6)
       prot_prob  = prot_pkg['model'].predict_proba(prot_input)[:, 1][0]
       met_input  = np.array([[user_values_for_met_features]])  # shape (1, 4)
       met_prob   = met_pkg['model'].predict_proba(met_input)[:, 1][0]
 
4. FUSION:
       fusion_input = np.array([[rna_prob, prot_prob, met_prob]])
       gbm_prob = fusion_pkg['model'].predict_proba(fusion_input)[:, 1][0]
       is_gbm = gbm_prob >= fusion_pkg['decision_threshold_default']
 
5. HANDLING MISSING LAYERS:
   If user only provides RNA + PROT (no metabolomics), pass NaN for met_prob.
   XGBoost handles NaN natively in fusion - no need to fill.
       fusion_input = np.array([[rna_prob, prot_prob, np.nan]])
 
INTENDED USE / SCOPE:
---------------------
MOmics is trained and validated on CPTAC-processed multi-omic data. The model
expects input normalized to CPTAC scale (rank-transformed within sample).
Performance on data processed through other pipelines is not guaranteed.
For research use only - not for clinical diagnostic decisions.
