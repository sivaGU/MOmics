# docs.py — Documentation strings for MOmics
# Edit this file to update documentation without touching app.py

OVERVIEW = """
### Purpose and Scope

MOmics is a clinical decision support tool designed for glioblastoma multiforme (GBM) patient risk stratification.
The system integrates multi-omics biomarker data — transcriptomics, proteomics, and metabolomics — to generate
probability-based risk assessments that can help clinicians identify high-risk patients who may benefit from
more aggressive monitoring or treatment strategies.

The tool is not intended to replace clinical judgement. Risk scores produced by MOmics are probabilistic
outputs derived from population-level training data and should be interpreted in the context of each patient's
full clinical picture.

---

### Analysis Pipeline

Patient data passes through the following stages in sequence:

**1. Data Ingestion**
Raw multi-omics measurements are provided either by manual entry (single patient) or CSV upload (cohort).
The system accepts values in three formats: gene symbol names (e.g. BTF3L4), bare Ensembl IDs
(e.g. ENSG00000164061.4), or prefixed Ensembl IDs (e.g. RNA_ENSG00000164061.4). Column format is
detected automatically and remapped internally before processing.

**2. Feature Alignment**
The input data is aligned against the full multi-omics feature space used during model training, spanning
RNA-seq transcriptomics, proteomics, and metabolomics. Any features present in the input but not in the
model space are silently ignored. Any features required by the model but absent from the input are filled
with NaN before imputation.

**3. Missing Value Imputation**
Missing values are replaced with per-feature medians derived from the training cohort. This means that
missing markers do not cause pipeline failure but do reduce prediction accuracy — particularly if the
missing features carry high model importance.

**4. Feature Scaling**
Each feature is normalised to zero mean and unit variance relative to the training cohort. This step
ensures that features measured on different scales (e.g. raw read counts vs. protein ratios) are
comparable when passed to the model.

**5. Risk Inference**
The XGBoost classifier outputs a probability score between 0 and 1 representing the likelihood that a
patient belongs to the high-risk class. A threshold of 0.5 is applied to assign the binary label
(High Risk / Low Risk). The continuous probability score is also reported to allow for finer clinical
interpretation.

**6. Visualisation**
Results are rendered as interactive charts and per-patient biomarker profiles. Individual patient
dashboards show the top expressed markers, a comparison against global model feature importance, and
a multi-modal radar summarising expression across omics layers.

---

### Cohort

The model was trained on GBM patient data from the Clinical Proteomic Tumor Analysis Consortium (CPTAC)
dataset. Training features include RNA-seq transcript counts quantified in Ensembl ID format, protein
expression values from mass spectrometry, and known metabolite concentrations from targeted metabolomics.
"""

GUI_GUIDE = """
### Navigating the Application

The application is divided into four pages accessible from the left sidebar:

- **Home** — Landing page with application branding.
- **Documentation** — This page. Full reference for the system, model, and data format.
- **User Analysis** — The primary workspace for running analyses on your own patient data.
- **Demo Walkthrough** — An interactive demo environment using real CPTAC GBM patients.

---

### User Analysis Page

The User Analysis page has two tabs: Manual Patient Entry and Bulk Data Upload.

**Manual Patient Entry**

This mode is designed for single-patient analysis. The top section displays the first 12 model features
as individual number input fields labelled with human-readable gene names. All fields default to 0.0.
The full set of model features is accessible by expanding the "Advanced Marker Input" section below
the top fields. Once values have been entered, clicking "Analyze Single Patient" runs the full pipeline
and renders the results dashboard directly below.

Note: Leaving a field at 0.0 is not equivalent to a missing value — it is treated as a literal
measurement of zero. If a marker was not measured, consider using the bulk upload mode instead, where
unmeasured columns can simply be omitted and will be handled by the imputer.

**Bulk Data Upload**

This mode processes multiple patients in a single run. To use it:

1. Click "Download CSV Template" to obtain a pre-formatted CSV with the correct column headers in
   gene-name format. Each column corresponds to one of the model features.
2. Fill in one patient per row. Each row should contain numeric expression values for the biomarkers
   that were measured. Columns for unmeasured biomarkers may be left empty or omitted entirely.
3. Upload the completed file using the file uploader. The system will detect the column format
   automatically (gene names, Ensembl IDs, or prefixed Ensembl IDs) and remap as needed.
4. Results for all patients are displayed simultaneously in the dashboard below the upload widget.

---

### Input Data Format

**Accepted Column Name Formats**

The application accepts three column naming conventions and detects the format automatically.
Prefixed Ensembl IDs are strongly recommended for best results. Gene symbol remapping relies on a
fixed reference table covering only the selected model features, which means any feature not in that
table will be silently dropped. Prefixed Ensembl IDs bypass this remapping step entirely and are
guaranteed to match the model's internal feature space exactly.

**Format 1: Gene Symbol Names**

Column headers are standard HGNC gene symbols, for example:

```
BTF3L4, CACNA2D3, LINC01116, MS4A6E, LINC02084, LINC01605, RNU6-1
```

This is the format used in the downloadable CSV template. Gene symbols are remapped internally to
their corresponding Ensembl IDs before processing. Use this format only when prefixed Ensembl IDs
are not available.

**Format 2: Prefixed Ensembl IDs (recommended — best results)**

Column headers use the full internal feature identifier with the data-type prefix:

```
RNA_ENSG00000164061.4, RNA_ENSG00000157445.13, RNA_ENSG00000242759.5
```

This format matches the model's internal feature names exactly and requires no remapping. All
features present in the file are recognised directly, and there is no risk of a feature being dropped
due to a missing or incorrect gene symbol mapping. This is the preferred format for all bulk uploads.

**Format 3: Bare Ensembl IDs**

Column headers contain the Ensembl ID without the RNA_ prefix:

```
ENSG00000164061.4, ENSG00000157445.13, ENSG00000242759.5
```

The application detects this format and prepends the RNA_ prefix automatically. Version suffixes
must be present and must match the training reference exactly. This format assumes all features are
RNA-seq features; proteomics and metabolomics features cannot be represented in this format and
should use Format 2 instead.

**CSV File Structure**

- Each row represents one patient.
- Each column represents one biomarker measurement.
- An optional Sample_ID column (or "Sample ID" with a space) may be included as the first column.
  This column is dropped before processing and is not passed to the model.
- All measurement values must be numeric.
- Column headers are required.
- Columns for unmeasured biomarkers may be omitted entirely rather than filled with zeros. Providing
  a zero where a marker was not measured is incorrect and will skew the result.

**Expression Value Units**

The model was trained on raw RNA-seq read counts as produced by a standard RNA-seq quantification
pipeline (gene-level counts, not TPM or FPKM). Submitting TPM-normalised or log-transformed values
will produce incorrect results. For proteomics features, values should be log2-transformed protein
expression ratios as produced by the CPTAC MSSM proteomics pipeline. For metabolomics features,
values should be corrected peak area intensities as produced by the CPTAC PNNL metabolomics pipeline.

**Common Errors**

- "Error processing file" — Most commonly caused by non-numeric values in data columns, a missing
  header row, or a file saved in Excel .xlsx format rather than CSV. Ensure the file is saved as CSV
  with UTF-8 encoding before uploading.
- All patients receiving identical risk scores — Indicates that none of the 7 high-importance features
  were present in the uploaded file and all critical features were imputed to the same training median.
  Verify that your column names match one of the three accepted formats.
- Unrecognised column warning — Columns that do not match any model feature and are not a Sample_ID
  column will trigger a warning. These columns are ignored and do not affect results.

---

### Interpreting Results

**Risk Score**

The risk score is the raw probability output of the XGBoost model, representing P(High Risk) on a
scale from 0 to 1. A score above 0.5 results in a High Risk classification; a score at or below 0.5
results in a Low Risk classification. The score should not be treated as a precise clinical probability
— it reflects the model's confidence relative to patterns observed in the CPTAC training cohort.

| Risk Score | Label | Interpretation |
|-----------|-------|---------------|
| 0.00 - 0.30 | Low Risk | Profile substantially dissimilar to high-risk training cases |
| 0.30 - 0.50 | Low Risk (borderline) | Profile weakly dissimilar; interpret with caution |
| 0.50 - 0.70 | High Risk (borderline) | Profile weakly similar to high-risk training cases |
| 0.70 - 1.00 | High Risk | Profile substantially similar to high-risk training cases |

**Risk Probability Distribution (Histogram)**

Shows the spread of risk scores across the uploaded cohort, colour-coded by classification. A narrow
distribution indicates the model is confident in its classifications; a broad distribution indicates
more heterogeneity or uncertainty.

**Individual Patient Risk Scores (Bar Chart)**

One bar per patient sorted from highest to lowest risk score. The dashed line at 0.5 marks the
classification threshold. Patients near the threshold warrant closer clinical review.

**Multi-Modal Signature (Radar Chart)**

Shows the average scaled expression level across the three omics layers for the selected patient:
Proteins, RNA, and Metabolites. Values are post-scaling z-scores relative to the training cohort mean.
If only RNA features are present in the uploaded data, the Proteins and Metabolites axes will read zero.

**Top 20 Marker Levels**

The 20 features with the highest scaled values for the selected patient. This identifies which
biomarkers are most elevated relative to the training cohort, not which are most important to the
model globally.

**Patient Markers vs Global Model Importance**

Two side-by-side charts for comparison. The left panel shows the 15 features most elevated in the
selected patient. The right panel shows the 15 features with the highest model importance globally.
Overlap between the two panels — markers that are both highly expressed in the patient and globally
important — provides the most clinically actionable signal.

**Limitations**

The model was trained and validated on data from a single cohort (CPTAC GBM). Performance on data
from different sequencing platforms or patient populations has not been evaluated. Predictions for
patients with more than 3 of the 7 high-importance features missing should be treated as unreliable.

---

### Demo Walkthrough Page

The demo page provides three modes of interaction using real CPTAC GBM patients:

**Try with Sample Patients** — Runs the full pipeline on four real patients with one button click.
Results persist while you interact with the patient selector dropdown.

**Guided Tutorial** — A five-step walkthrough covering data preview, analysis, cohort results,
and individual patient profiling. Progress is tracked with a progress bar.

**Learn by Exploring** — Opens the full dashboard with demo data pre-loaded for unguided exploration.

A "Reset Demo Workspace" button at the bottom clears all session state and returns the page to its
initial state without requiring a browser refresh.
"""

MODEL_ARCH = """
The MOmics predictive model was developed entirely within Python 3.10.12.
Data harmonization was performed to synchronize samples across the three omics layers
(transcriptomics, proteomics, and metabolomics) from the CPTAC Discovery Cohort. Missing data values
were inputted using a median strategy via the Scikit-learn library (version 1.6.1), followed by
Z-score standardization to normalize feature magnitudes. Pre-processing was required for utilizing
downstream machine learning libraries. We selected the Extreme Gradient Boosting (XGBoost) library
(version 2.0.3) as the classifier for MOmics. While Random Forest (RF) and Support Vector Machines
(SVM) are commonly used in genomic classification, XGBoost learns optimal directions for missing
values during tree construction. This capability allowed MOmics to handle sparse data from our
multi-omic integration. Furthermore, XGBoost incorporates L1 (Lasso) and L2 (Ridge) regularization
terms in its objective function. This mitigates against overfitting in datasets with high feature
amounts and low sample sizes.

---

### Feature Selection, Training, and Optimization

Raw transcriptomics, proteomics, and metabolomics datasets were preprocessed for feature selection.
Preprocessing included addressing external noise, data sparsity and missing values using the
SimpleImputer strategy. Z-score normalization using StandardScaler was used to ensure all features
were evaluated fairly. Primary feature selection was completed using SelectKBest algorithm with an
ANOVA F-value scoring function (f_classif). This process condensed the initial input space of about
70,000 molecular features to a multi-omics signature composed of the most highly weighted features.
To ensure model stability and compatibility with the MOmics GUI, the feature list was
lexicographically sorted and fixed.

The MOmics predictive model was constructed using the XGBClassifier from the Extreme Gradient
Boosting library. To reduce the risk of overfitting, we constrained the modeling with a maximum tree
depth of 3 and 150 boosting rounds (n_estimators). The model was optimized using a binary logistic
loss function with a learning rate of 0.1. To address the class imbalance in the discovery cohort,
with an approximate 10:1 tumor-to-normal ratio, we set the scale_pos_weight based on the ratio of
normal to tumor samples. This allowed the model to treat the small normal sample group with the same
importance as the larger tumor group, preventing bias towards tumor detection. Furthermore, we
utilized colsample_bytree with the parameter set to 0.3, forcing the model to randomly subset
features during tree construction. This parameterization allows the model to pull signatures from
all three omics layers.

Model performance was validated using a stratified 3-fold cross-validation. This ensured that each
fold maintained the proper ratio of tumor and normal samples. Performance was quantified using the
Area Under the Receiver Operating Characteristic (AUROC) and the Precision-Recall AUC (PR-AUC).
The PR-AUC was a definitive metric for evaluating diagnostic reliability, providing a more rigorous
assessment of the relationship between precision and recall.

While the primary model was trained on the multi-omic signature, a secondary XGBoost classifier was
trained on the individual omic layers. This secondary classifier was implemented to identify the top
drivers within each specific layer, ensuring that the final visualization displayed a balanced view
of the top transcriptomic, proteomic, and metabolic drivers.

---

### Comparative Benchmarking and Specificity Stress Testing

To quantify the value of the multi-omics strategy, we conducted a systematic ablation study. We
trained baseline classifiers using single-omics data (RNA-only, Protein-only, Metabolite-only) and
compared them against the integrated MOmics model. This benchmarking confirmed a lift in diagnostic
performance provided by multi-omics integration. The combination of biomolecular layers provided a
more accurate differentiation of tumor and normal phenotypes compared to a single RNA layer.

Clinical validity was assessed through a rigorous specificity stress test. The trained model was
applied to the external validation glioma cohort (CGGA) (n > 300). The learned feature space was
intersected with available external data to test MOmics's ability to identify GBM in an unseen
population. To ensure the identified biomarkers were specific to only GBM and no other cancer types,
the model was tested against a broader cancer population. MOmics was introduced to PDAC and ccRCC
to test its ability to differentiate against these test cancers and identify only GBM-related signals.
This validates the high specificity of our identified multi-omics signature.

---

### Rigor and Reproducibility

To ensure reproducible results, random seeds were fixed throughout the analysis: a consistent seed
was used for all data splitting, hyperparameter optimization, and bootstrap resampling. For
single-omics processing, rigor was maintained through standardized nomenclature and stringent feature
filtering, where noisy features were excluded based on fixed thresholds for low-expression genes,
protein abundance value, and metabolite presence. Differential expression analyses were performed
consistently across cohorts, with pathway enrichment standardized across all three omics layers using
raw p-values to ensure methodological consistency.

Multi-omics integration via DIABLO was performed using a correlation weight of 0.9, selected via
internal cross-validation to prioritize overlapping biological signals across the three omics layers,
with 20-repeat 5-fold cross-validation used to minimize classification error. The MOmics architecture
was constrained to a maximum tree depth of 3 and 150 boosting rounds to prevent overfitting on
high-dimensional inputs, with a fixed feature subsampling ratio to promote stable multi-modal
integration.

Leakage controls were implemented at multiple stages: (1) Z-score normalization parameters were
calculated exclusively on training folds to prevent distributional leakage, (2) the scale-penalty for
class variance was taken from the training distribution rather than post-hoc tuning, and (3) feature
selection was performed strictly within cross-validation loops to ensure identified biomarkers were
not informed by validation labels.

External evaluation on the CGGA, PDAC, and ccRCC cohorts was performed using fully locked protocols,
with no model retraining, hyperparameter recalibration, or decision-threshold adjustment. Model
weights and deterministically ordered feature signatures derived from the discovery cohort were
applied unchanged to all external and stress-test populations. Performance benchmarking was conducted
using an identical methodology for single-omics baselines to isolate the performance gains due to
multi-omic integration.

---

### Selected Features and Importance

Of the 100 features passed to the model, only 7 carry non-zero importance in the current trained
XGBoost model. These features account for 100% of the model's predictive signal:

| Gene Name | Ensembl ID | Feature Importance |
|-----------|-----------|-------------------|
| LINC02084 | RNA_ENSG00000244040.4 | 21.5% |
| BTF3L4 | RNA_ENSG00000164061.4 | 15.8% |
| RNU6-1 | RNA_ENSG00000206814.1 | 15.4% |
| MS4A6E | RNA_ENSG00000181215.11 | 13.9% |
| CACNA2D3 | RNA_ENSG00000157445.13 | 13.8% |
| LINC01605 | RNA_ENSG00000233487.6 | 10.3% |
| LINC01116 | RNA_ENSG00000242759.5 | 9.3% |

Four of the seven are long intergenic non-coding RNAs (LINCs) and one is a small nuclear RNA (RNU6-1).
All seven are transcriptomic features. Proteomics and metabolomics features, while present in the
preprocessing pipeline, carry zero importance in the current model.

Feature importance values are derived from the XGBoost gain metric, which measures the average
improvement in model accuracy contributed by each feature across all splits in which it appears.

---

### Risk Score Output

The model outputs a continuous probability P(High Risk) between 0 and 1. The binary label is assigned
using a fixed threshold of 0.5. The probability score carries more information than the binary label
and should be reported alongside it in clinical contexts.

The current model exhibits a tendency toward polarised outputs — scores cluster near 0.2–0.3 for
low-risk patients and near 0.8 for high-risk patients. This reflects the decision boundary structure
learned from the training data and is consistent with a model that has identified strong discriminating
features (particularly LINC02084 expression), rather than a calibration artefact.

---

### GUI

MOmics GUI is a free-access web application allowing users to interact with the model and generate
patient-specific diagnostic predictions. The GUI requires input data in a standard CSV or Excel format
using standardized feature names (e.g., Gene Symbols or HMDB IDs) labeled as 'prot,' 'rna,' and
'met.' Users must provide raw abundance scores. The GUI allows for both manual entry of single samples
and bulk analysis with the use of a downloadable template. To maintain model alignment, any
non-specified features are automatically filled with a value of 0.00. The GUI also contains
documentation, a user demo, and a user analysis to support the user.
"""
