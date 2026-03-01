# docs.py — Documentation strings for MOmics-ML
# Edit this file to update documentation without touching app.py

OVERVIEW = """
### Purpose and Scope

MOmics-ML is a clinical decision support tool designed for glioblastoma multiforme (GBM) patient risk stratification.
The system integrates multi-omics biomarker data — transcriptomics, proteomics, and metabolomics — to generate
probability-based risk assessments that can help clinicians identify high-risk patients who may benefit from
more aggressive monitoring or treatment strategies.

The tool is not intended to replace clinical judgement. Risk scores produced by MOmics-ML are probabilistic
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
The input data is aligned against the full multi-omics feature space used during model training, spanning RNA-seq transcriptomics, proteomics, and metabolomics. Any features present in the input but not in the model space are silently ignored. Any features required by the model but absent from the input are filled with NaN before imputation.

**3. Missing Value Imputation**
A SimpleImputer fitted on the training cohort replaces all NaN values with the per-feature median derived
from the training data. This means that missing markers do not cause pipeline failure but do reduce
prediction accuracy — particularly if the missing features carry high model importance.

**4. Feature Scaling**
A StandardScaler fitted on the training cohort normalises each feature to zero mean and unit variance.
This step is required because XGBoost is sensitive to the relative scale of input features when they
are used across split thresholds.

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

### Demo Walkthrough Page

The demo page provides three modes of interaction, all using the same pre-loaded real patient dataset:

**Try with Sample Patients**
Runs the full analysis pipeline on four real CPTAC GBM patients with a single button click. Results
are displayed immediately and persist while you interact with the patient selector dropdown. Use this
mode to quickly see what a complete analysis output looks like.

**Guided Tutorial**
A five-step walkthrough that introduces the data, runs the analysis, explains the cohort-level charts,
and then walks through an individual patient profile. Each step must be completed before advancing to
the next. Progress is tracked with a progress bar at the top of the section.

**Learn by Exploring**
Opens the full analysis dashboard with the demo data pre-loaded. This mode is unguided and intended
for users who want to explore the interface independently. A learning resources tab provides reference
definitions for risk score ranges and biomarker type prefixes.

---

### Resetting the Demo

A "Reset Demo Workspace" button at the bottom of the Demo Walkthrough page clears all session state
associated with the demo, including stored analysis results and tutorial progress. This resets the
page to its initial state without requiring a browser refresh.
"""

MODEL_ARCH = """
### Machine Learning Model

**Algorithm: XGBoost (Extreme Gradient Boosting)**

The core predictive model is an XGBoost binary classifier. XGBoost builds an ensemble of decision trees
in a sequential, gradient-boosted fashion. Each tree is trained to correct the residual errors of the
previous ensemble, and the final prediction is the sum of contributions from all trees passed through a
logistic function to produce a probability. XGBoost was selected for this application because it handles
high-dimensional sparse input efficiently, is robust to the presence of correlated features common in
multi-omics data, and provides interpretable feature importance scores.

---

### Feature Space

The model was trained on a feature space spanning three omics data types: RNA-seq transcriptomics
(RNA_ENSG prefix), proteomics from mass spectrometry (PROT_ prefix), and metabolomics (MET_ prefix).

At the inference stage, the model operates on a reduced set of 100 features selected during training.
Features outside this selected set are used only by the imputer and scaler and do not contribute to the
final prediction.

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
preprocessing pipeline, have zero importance in the current model version.

Feature importance values are derived from the XGBoost gain metric, which measures the average
improvement in model accuracy contributed by each feature across all splits in which it appears.

---

### Preprocessing Components

The model bundle consists of four serialised objects loaded at application startup:

**momics_xgb_model-1.pkl** — The trained XGBoost classifier. Stores the full ensemble of decision
trees, feature names, and hyperparameters. Used exclusively for inference.

**imputer-1.pkl** — A scikit-learn SimpleImputer fitted on the training cohort with strategy="median".
Replaces NaN values across the full multi-omics feature space with per-feature training medians before scaling.
Patients with many missing features will have their input dominated by training-set medians, which
reduces the reliability of their risk score.

**scaler-1.pkl** — A scikit-learn StandardScaler fitted on the training cohort. Applied after imputation
to normalise all features to zero mean and unit variance. The scaler was fitted on the imputed training
data, so it expects input that has already been passed through the imputer.

**feature_list-1.pkl** — The ordered list of feature names that define the model input space.
Used to align the scaled output of the scaler with the feature order expected by the XGBoost model.

---

### Risk Score Interpretation

The model outputs a continuous probability P(High Risk) between 0 and 1. The binary label is assigned
using a fixed threshold of 0.5. The probability score itself carries more information than the binary
label and should be reported alongside it in clinical contexts.

The current model exhibits a tendency toward polarised outputs — scores cluster near 0.2-0.3 for
low-risk patients and near 0.8 for high-risk patients. This reflects the decision boundary structure
learned from the training data and is consistent with a model that has identified strong discriminating
features (particularly LINC02084 expression) rather than a calibration artefact.
"""

INPUT_FORMAT = """
### Input Data Format Specification

---

### Accepted Column Name Formats

The application accepts three column naming conventions and detects the format automatically.
**Prefixed Ensembl IDs (Format 2) are strongly recommended for best results.** Gene symbol
remapping relies on a fixed reference table covering only the selected model features, which means
any feature not in that table will be silently dropped. Prefixed Ensembl IDs bypass this
remapping step entirely and are guaranteed to match the model's internal feature space exactly,
eliminating any risk of features being lost due to naming mismatches.

**Format 1: Gene Symbol Names**

Column headers are standard HGNC gene symbols, for example:

```
BTF3L4, CACNA2D3, LINC01116, MS4A6E, LINC02084, LINC01605, RNU6-1
```

This is the format used in the downloadable CSV template. Gene symbols are remapped internally to
their corresponding Ensembl IDs using a built-in reference table before processing. Only the
selected model features have entries in this reference table; any other gene symbol columns will be ignored.
Use this format only when prefixed Ensembl IDs are not available.

**Format 2: Prefixed Ensembl IDs (recommended — best results)**

Column headers use the full internal feature identifier with the data-type prefix:

```
RNA_ENSG00000164061.4, RNA_ENSG00000157445.13, RNA_ENSG00000242759.5
```

This format matches the model's internal feature names exactly and requires no remapping. All
features present in the file are recognised directly, version suffixes are matched precisely, and
there is no risk of a feature being dropped due to a missing or incorrect gene symbol mapping.
Files exported from the CPTAC pipeline or generated by the extract_demo_features.py utility will
be in this format. This is the preferred format for all bulk uploads.

**Format 3: Bare Ensembl IDs**

Column headers contain the Ensembl ID without the RNA_ prefix:

```
ENSG00000164061.4, ENSG00000157445.13, ENSG00000242759.5
```

The application detects this format and prepends the RNA_ prefix automatically. Version suffixes
(the .4, .13 portion) must be present and must match the training reference exactly. This format
assumes all features are RNA-seq features; proteomics and metabolomics features cannot be
represented in this format and should use Format 2 instead.

---

### CSV File Structure

Files must be in comma-separated values (CSV) format with UTF-8 encoding. The expected structure is:

- Each **row** represents one patient.
- Each **column** represents one biomarker measurement.
- An optional **Sample_ID** column (or "Sample ID" with a space) may be included as the first column
  to provide patient identifiers. This column is dropped before processing and is not passed to the model.
- All measurement values must be **numeric**. String values, percentage signs, or other non-numeric
  content in data columns will cause a processing error.
- **Column headers are required.** Files without a header row cannot be parsed correctly.
- Columns for biomarkers that were not measured may be **omitted entirely** rather than filled with
  zeros or empty strings. The imputer will handle missing features using training-set medians.
  Providing a zero where a marker was not measured is incorrect and will skew the result.

---

### Minimal Valid Example (Gene Symbol Format)

```
Sample_ID,BTF3L4,CACNA2D3,LINC01116,LINC02084,MS4A6E,LINC01605,RNU6-1
Patient_001,1707,607,403831,311900,92228,3669,111
Patient_002,975,57,106,10,41,2,2
```

A file containing only the 7 high-importance features is sufficient to produce a differentiated
risk score. All remaining 93 model features will be imputed to their training medians and will
not affect the prediction, since they carry zero importance in the current model.

---

### Minimal Valid Example (Prefixed Ensembl ID Format)

```
Sample_ID,RNA_ENSG00000164061.4,RNA_ENSG00000157445.13,RNA_ENSG00000242759.5,RNA_ENSG00000244040.4,RNA_ENSG00000181215.11,RNA_ENSG00000233487.6,RNA_ENSG00000206814.1
Patient_001,1707,607,403831,311900,92228,3669,111
Patient_002,975,57,106,10,41,2,2
```

---

### Expression Value Units

The model was trained on raw RNA-seq read counts as produced by a standard RNA-seq quantification
pipeline (gene-level counts, not TPM or FPKM). Submitting TPM-normalised or log-transformed values
will produce incorrect results because the imputer and scaler were fitted on raw count distributions.
If your data has been normalised, contact the model training team to obtain a version of the
preprocessing objects fitted on the same normalisation scheme.

For proteomics features (PROT_ prefix), values should be log2-transformed protein expression ratios
as produced by the CPTAC MSSM proteomics pipeline. For metabolomics features (MET_ prefix), values
should be corrected peak area intensities as produced by the CPTAC PNNL metabolomics pipeline.

---

### Common Errors

**"Error processing file"** — Most commonly caused by non-numeric values in data columns, a missing
header row, or a file saved in a format other than CSV (e.g. Excel .xlsx). Ensure the file is saved
as CSV with UTF-8 encoding before uploading.

**All patients receiving identical risk scores** — Indicates that none of the 7 high-importance features
were present in the uploaded file, and all critical features were imputed to the same training median.
Verify that your column names match one of the three accepted formats and that the 7 critical features
listed in the Model Architecture tab are present in your data.

**Unrecognised column warning** — Columns that do not match any of the model features and are not
a Sample_ID column will trigger a warning listing the unrecognised names. These columns are ignored
and do not affect results. This commonly occurs when a file includes clinical metadata columns
alongside expression data.
"""

RESULTS = """
### Interpreting Analysis Results

---

### Risk Score

The risk score is the raw probability output of the XGBoost model, representing P(High Risk) on a
scale from 0 to 1. A score above 0.5 results in a High Risk classification; a score at or below 0.5
results in a Low Risk classification.

The score should not be treated as a precise clinical probability. It reflects the model's confidence
relative to the patterns observed in the CPTAC training cohort. A score of 0.80, for example, means
the patient's biomarker profile is similar to profiles that were associated with high-risk outcomes
in the training data, not that there is an 80% clinical probability of a specific event.

Approximate interpretation ranges:

| Risk Score | Label | Interpretation |
|-----------|-------|---------------|
| 0.00 - 0.30 | Low Risk | Profile substantially dissimilar to high-risk training cases |
| 0.30 - 0.50 | Low Risk (borderline) | Profile weakly dissimilar; interpret with caution |
| 0.50 - 0.70 | High Risk (borderline) | Profile weakly similar to high-risk training cases |
| 0.70 - 1.00 | High Risk | Profile substantially similar to high-risk training cases |

---

### Risk Probability Distribution (Histogram)

The histogram shows the spread of risk scores across the full uploaded cohort. Each bar represents
the number of patients whose risk score falls within that range. Bars are colour-coded red for High
Risk patients and green for Low Risk patients. A narrow distribution clustered near 0.8 or 0.2
indicates the model is confident in its classifications; a broad distribution indicates more
heterogeneity in the cohort or more uncertainty in the predictions.

---

### Individual Patient Risk Scores (Bar Chart)

The bar chart displays one bar per patient, sorted from highest to lowest risk score. The dashed
horizontal line at 0.5 marks the classification threshold. Bars above the line are High Risk; bars
below are Low Risk. This chart is useful for identifying patients near the threshold who may warrant
closer review.

---

### Multi-Modal Signature (Radar Chart)

The radar chart shows the average scaled expression level across the three omics layers for the
selected patient: Proteins (PROT_ features), RNA (RNA_ENSG features), and Metabolites (MET_ features).
Values shown are post-scaling (z-scores relative to the training cohort mean), so a value above zero
indicates above-average expression and a value below zero indicates below-average expression. This
chart provides a high-level summary of which data modalities are elevated for that patient. If only
RNA features are present in the uploaded data, the Proteins and Metabolites axes will read zero.

---

### Top 20 Marker Levels

This horizontal bar chart shows the 20 features with the highest scaled values for the selected
patient. Features are displayed using gene names where a mapping exists, or their raw Ensembl ID
otherwise. This chart identifies which specific biomarkers are most elevated in the patient relative
to the training cohort, not which biomarkers are most important to the model globally. A marker can
appear in this chart without contributing to the risk score if it carries zero model importance.

---

### Patient's Top 15 Expressed Markers vs Global Model Importance

These two side-by-side charts enable comparison between patient-specific expression and model-level
importance:

**Left panel (Patient's Top 15 Expressed Markers):** The 15 features most elevated in the selected
patient, coloured by expression level. This is patient-specific.

**Right panel (Global Model Importance):** The 15 features with the highest XGBoost gain importance
across all patients. This reflects what the model has learned to rely on across the training cohort.

Overlap between the two panels — markers that are both highly expressed in the patient and globally
important to the model — provides the most clinically actionable signal. A patient with high LINC02084
and BTF3L4 expression appearing in both panels has a profile that strongly matches the patterns the
model associates with high-risk outcomes.

---

### Limitations

The model was trained and validated on data from a single cohort (CPTAC GBM). Performance on data
from different sequencing platforms, different RNA-seq quantification pipelines, or patient populations
with substantially different demographic characteristics has not been evaluated. The model should be
validated against local institutional data before being used to inform clinical decisions.

Predictions for patients with a large proportion of missing features (imputed values) are less reliable
than predictions for patients with complete data. If more than 3 of the 7 high-importance features are
missing for a given patient, the risk score for that patient should be treated as unreliable.
"""
