# MOmics

A machine learning-driven multi-omics integration tool that identifies diagnostic biomarkers and therapeutic targets in glioblastoma (GBM). All R and Python code used for the *MOmics* study, plus the Streamlit GUI that serves the locked v11 model.

**GUI:** https://momics-gbm.streamlit.app/

## Features

The Streamlit app (`momics_app.py`) has four sections:

- **Home** — project overview and quick access
- **Documentation** — pipeline stages, feature alignment, and model architecture (served from `docs.py`)
- **User Analysis** — manual entry or CSV upload of patient RNA/protein/metabolite values, scored against the locked pipeline with risk charts and a downloadable PDF report
- **Demo Walkthrough** — a guided run through `gui_assets/demo_data/MOmics_GUI_demo_mixed.csv`

## Project Structure

```
MOmics/
├── momics_app.py                        # Main Streamlit GUI application
├── ml_pipeline.py                       # XGBoost training pipeline + paper-figure generation
├── MOmics_v11_inference.py              # Standalone GUI-integration inference helper (see docs/gui_integration_guide.md)
├── docs.py                              # Documentation strings shown in the app
├── single_omics_transcriptomics.R       # RNA differential analysis (DESeq2)
├── single_omics_metabolomics.R          # Metabolite differential analysis
├── single_omics_proteomics.R            # Proteomics differential analysis (placeholder, not yet implemented)
├── diabolo_integration.R                # Multi-omics DIABLO integration (placeholder, not yet implemented)
├── run_app.bat                          # Windows one-click launch
├── run_app.ps1                          # PowerShell launch
├── verify_setup.py                      # Environment + dependency check
├── requirements.txt
├── README.md
├── INSTALLATION.md
├── logo.png
│
├── data/
│   ├── discovery/                       # GBM training and discovery inputs (see data/discovery/README.md)
│   │   └── legacy/                      # Earlier-normalization exports, not used by ml_pipeline.py
│   ├── external_validation/
│   │   ├── BRCA/
│   │   ├── CCRCC/
│   │   ├── CGGA/
│   │   └── LUAD/
│   └── reference/                       # Feature-selection references (DIABLO ranked panel)
│
├── models/
│   └── MOmics_v11_locked_pipeline.pkl   # Bundled sub-models, fusion model, feature lists, calibrator
│
├── docs/
│   ├── gui_integration_guide.md         # How to wire MOmics_v11_inference.py into a GUI
│   └── legacy_pickle_bundle_reference.txt  # Older per-layer .pkl bundle format
│
├── results/                              # Analysis outputs (placeholders, see per-folder README.md)
│   ├── single_omics/{RNA,Proteomics,Metabolomics}/
│   ├── multi_omics/
│   ├── ml_model/{performance_metrics,confusion_matrices,PR_AUC_ROC_curves}/
│   ├── biomarker_reports/
│   └── figures/
│
├── features/                             # Placeholder for standalone feature-list exports
├── notebooks/                            # Placeholder for exploratory/training notebooks
├── scripts/                              # Placeholder for CLI wrappers around ml_pipeline.py
├── supplementary/
│   ├── Supplementary_Tables/
│   ├── Supplementary_Figures/
│   └── Supplementary_Code/
└── gui_assets/
    └── demo_data/
```

## Installation

```bash
pip install -r requirements.txt
streamlit run momics_app.py
```

Or use the provided scripts:
- Windows: double-click `run_app.bat`
- PowerShell: run `.\run_app.ps1`

Run `python verify_setup.py` to check your environment before launching. See `INSTALLATION.md` for platform-specific steps, virtual environments, and troubleshooting.

## Data Sources

- **Discovery cohort** — 109 CPTAC-processed samples (99 GBM tumor + 10 GTEx-derived normal brain) across RNA-seq, proteomics, and metabolomics. See `data/discovery/README.md` for a known gap: the metabolomics R script expects a `metabolome_sample_info.v4.0.tsv` file not yet checked into the repo.
- **External validation** — BRCA, CCRCC, LUAD, and CGGA cohorts, never seen during training.
- **Reference panel** — `data/reference/diablo_multiomics_ranked_features_FDR_CLEAN.csv`, the DIABLO-ranked 25-feature candidate panel (later pruned to 9 active features).

## Model

`models/MOmics_v11_locked_pipeline.pkl` bundles the three per-layer sub-models, the fusion model, frozen z-score parameters, the Youden threshold, and an isotonic calibrator. See `docs/gui_integration_guide.md` for the full integration contract (input formats, missing-layer handling, and why raw vs. calibrated probabilities are used differently).

`single_omics_proteomics.R` and `diabolo_integration.R` are placeholders for analyses described in the paper that aren't yet checked into this repo as standalone scripts.

## Intended Use / Scope

MOmics is trained and validated on CPTAC-processed multi-omic data, rank/z-score normalized within sample. Performance on data processed through other pipelines is not guaranteed. **For research use only — not for clinical diagnostic decisions.**

## Contact

Questions and bug reports: Dr. Sivanesan Dakshanamurthy — sd233@georgetown.edu
