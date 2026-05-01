# MOmics-ML v11 — GUI integration

## Files

| File | Purpose |
|---|---|
| `MOmics_v11_locked_pipeline.pkl` | Single bundled artifact — load with `joblib.load(...)` and you have everything (sub-models, fusion model, feature lists, z-score parameters, threshold, symbol-to-ENSG map, isotonic calibrator). 1.96 MB. |
| `MOmics_v11_inference.py` | GUI-ready inference helper. Drop into the Streamlit codebase, import `load_pipeline` and `score_sample`. |

## Training data

These three files were the only inputs to model training:

| File (in `/mnt/user-data/uploads/`) | Role |
|---|---|
| `all_subtypes.v5.1.tsv` | Sample manifest. Defines tumor vs normal labels (109 samples: 99 GBM tumor + 10 GTEx-derived normal brain) |
| `rnaseq_washu_readcount.v4.0.tsv` | Raw RNA-seq read counts. RNA sub-model trained on 99 tumor + 9 normal |
| `proteome_mssm_per_gene_imputed.v4.0.tsv` | Log2 reference-intensity normalized protein abundances. Proteomic sub-model trained on 99 tumor + 10 normal |
| `metabolome_pnnl.v4.0.tsv` | Log2 metabolite abundances. Metabolomic sub-model trained on 75 tumor + 7 normal |

Plus one feature-selection input that defines the candidate panel (later pruned to 9 active features by gain importance):

| File | Role |
|---|---|
| `diablo_multiomics_ranked_features_FDR_CLEAN.csv` | DIABLO-ranked 25 candidate features (12 RNA + 8 prot + 5 met) |

Everything else (CGGA, BRCA, ccRCC, LUAD, PDAC) was external validation only — never seen during training.

## Pipeline contents

After `pipe = joblib.load("MOmics_v11_locked_pipeline.pkl")`, the dict has:

| Key | What it is |
|---|---|
| `version` | `"v11"` |
| `pruned_features` | Dict of `{layer: [feature_names]}` — 6 RNA, 4 prot, 3 met = 9 unique entities |
| `sub_models` | Dict of `{layer: XGBClassifier}` — three trained per-layer models |
| `fusion_model` | XGBClassifier trained on out-of-fold sub-model probabilities |
| `zscore_params` | Dict of `{layer: {"mean": {feature: μ}, "std": {feature: σ}}}` frozen from discovery |
| `youden_threshold` | 0.964 — operates on **raw** fusion probabilities |
| `symbol_to_ensg` | Gene symbol → ENSG mapping (for cross-platform feature alignment) |
| `rna_log1p_required` | True — RNA inputs are log1p'd before z-scoring |
| `isotonic_calibrator` | Sklearn IsotonicRegression fit on pooled external scores. Optional. |
| `calibration_info` | Metadata about how the calibrator was fit |
| `diablo_panel_full` | The 25-feature DIABLO panel before pruning (audit trail) |
| `discovery_metrics` | Discovery-cohort AUROC, PR-AUC, sensitivity, specificity at threshold |
| `balance_method` | `"synthetic"` — records the augmentation strategy used |

## Minimum integration

```python
from MOmics_v11_inference import load_pipeline, score_sample, get_required_features

pipe = load_pipeline("MOmics_v11_locked_pipeline.pkl")

# What features should the GUI ask for?
print(get_required_features(pipe))
# {'rna': ['BSN', 'PCLO', 'PRKCE', 'PTPRT', 'CIT', 'MAPT'],
#  'prot': ['PTPRT', 'CIT', 'PCLO', 'BSN'],
#  'met':  ['hypotaurine', 'creatinine', 'citricacid']}

# Score a sample
result = score_sample(
    pipe,
    rna_dict={"BSN": 23.4, "PCLO": 18.1, "PRKCE": 12.8,
              "PTPRT": 8.2, "CIT": 14.5, "MAPT": 31.0},
    prot_dict={"PTPRT": -1.1, "CIT": -0.4, "PCLO": -0.8, "BSN": -1.5},
    met_dict={"hypotaurine": 8.2, "creatinine": 5.1, "citricacid": 6.7},
)
# → {"P_GBM_raw": 0.97, "P_GBM_calibrated": 0.43, "binary_call": "GBM", ...}
```

`rna_dict` takes raw read counts (the helper applies log1p internally to match training).
`prot_dict` and `met_dict` take log2 abundances (matches CPTAC standard at the source).
Pass `None` for layers the user can't provide — the fusion model handles missing layers natively.

## On the calibrated probability — important

The bundle exposes both:

- **`P_GBM_raw`** — the model's confidence score. Values cluster bimodally near 0.38 (rejection) or 0.97 (confident GBM call). The 0.964 Youden threshold operates on this.
- **`P_GBM_calibrated`** — isotonic-mapped clinical probability. Fit on the 756-sample external set (35 GBM + 721 non-GBM, ≈4.6% prevalence).

The calibrated probability for a confident model call (raw ≈ 0.97) comes out around 0.40 — not 0.97. This is **not a bug**. Isotonic calibration learns from the prevalence of GBM in the data it was fit on. In our external set, only 4.6% of samples are actually GBM, so a confident model call corresponds to a posterior probability of ≈40% in that population.

**This means:**
- For binary clinical decisions: use **raw probability** with the Youden threshold (the default in `score_sample`). This is what gives 97.1% sensitivity and 88.5% specificity.
- For displaying a probability number: use calibrated, but show context. The honest framing is: "in the external evaluation set (4.6% GBM prevalence), samples with this score had a 40% posterior probability of GBM. In a higher-prevalence clinical population — e.g., neuro-oncology referral — the posterior would be higher."
- For research/diagnostic exploration where the clinician interprets the score themselves: just show raw.

If the GUI is deployed in a setting with substantially different GBM prevalence than 4.6%, the calibrator can be re-fit on a more representative cohort. The fitting is one line of code (see how it was done in the `MOmics_v11_paper_figures.py` calibration section).

## What the GUI should NOT do

- Don't apply any normalization to inputs before passing to `score_sample`. The helper does it (log1p for RNA, frozen z-score for all layers).
- Don't change the threshold based on calibrated scores. The Youden threshold of 0.964 is for raw scores.
- Don't average per-layer probabilities yourself. The fusion model is an XGBoost meta-classifier, not a mean (Section 3.3.5 of the paper explains why).
- Don't fall back to defaults if the user is missing entire layers. Pass `None` and let the fusion model handle it — that's the whole point of XGBoost's missing-value handling.
