# data/discovery/

Discovery-cohort inputs used to train the locked v11 pipeline (see `docs/gui_integration_guide.md` for exactly which files feed which sub-model).

`metabolome_sample_info.v4.0.tsv` is the metabolomics sample manifest read by `Single Omics & Integration/single_omics_metabolomics.R`.

`mRNA_RSEM_UQ_log2_{Normal,Tumor}.cct.csv` and `proteomics_gene_level_MD_abundance_{normal,tumor}.cct.csv` are the PDAC external-validation inputs read directly from this folder by `ml_pipeline.py` (built via f-string, e.g. `f"mRNA_RSEM_UQ_log2_{kind}.cct.csv"` — not obvious from a filename grep, which is why these were briefly and incorrectly filed as unused legacy exports).
