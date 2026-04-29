
# MOmics

All R and Python codes used for MOmics: A machine learning-driven Multi-Omics Integration Identifies Diagnostic Biomarkers and Therapeutic Targets in Glioblastoma study. 
GUI Link: https://momics-gbm.streamlit.app/

Requirements: 
MOmics_GBM_Project Structure: 
## Project Structure
```
MOmics_GBM_Project/
├── momics_app.py                     # Main Streamlit GUI application
├── ml_pipeline.py                    # XGBoost training + prediction pipeline
├── feature_selection.py              # Feature selection + rank transformation
├── data_preprocessing.py             # Data cleaning, harmonization, normalization
├── diabolo_integration.R             # Multi-omics DIABLO analysis (R script)
├── single_omics_analysis.R           # RNA, protein, metabolite differential analysis
├── run_app.bat                       # Windows batch script to launch GUI
├── run_app.ps1                       # PowerShell script to launch GUI
├── verify_setup.py                   # Environment + dependency check
├── README.md                         # Project overview and instructions

├── data/
│   ├── discovery_cohort/
│   │   ├── transcriptomics/
│   │   ├── proteomics/
│   │   └── metabolomics/
│   ├── scalability_cohorts/
│   │   ├── LUAD/
│   │   ├── CCRCC/
│   │   ├── BRCA/
│   │   └── UCEC/
│   ├── external_validation/
│   └── reference_data/

├── results/
│   ├── single_omics/
│   │   ├── RNA/
│   │   ├── Proteomics/
│   │   └── Metabolomics/
│   ├── multi_omics/
│   ├── ml_model/
│   │   ├── performance_metrics/
│   │   ├── confusion_matrices/
│   │   └── PR_AUC_ROC_curves/
│   ├── biomarker_reports/
│   └── figures/

├── models/
│   ├── transcriptomics_model.pkl
│   ├── proteomics_model.pkl
│   ├── metabolomics_model.pkl
│   └── fusion_model.pkl

├── features/
│   ├── RNA_features.txt
│   ├── protein_features.txt
│   └── metabolite_features.txt

├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│   └── validation_analysis.ipynb

├── gui_assets/
│   ├── sample_input_files/
│   ├── demo_data/
│   └── screenshots/

├── scripts/
│   ├── run_training.py
│   ├── run_validation.py
│   ├── run_feature_selection.py
│   └── generate_reports.py

└── supplementary/
    ├── Supplementary_Tables/
    ├── Supplementary_Figures/
    └── Supplementary_Code/
```
Contant: 
Questions and bug reports, please contact: Dr. Sivanesan Dakshanamurthy: sd233@georgetown.edu
