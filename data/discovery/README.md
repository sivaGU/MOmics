# data/discovery/

Discovery-cohort inputs used to train the locked v11 pipeline (see `docs/gui_integration_guide.md` for exactly which files feed which sub-model).

**Known gap:** `single_omics_metabolomics.R` reads `metabolome_sample_info.v4.0.tsv` from this folder, but that file is not currently checked into the repo. The script will fail until it's added, or is pointed at `all_subtypes.v5.1.tsv` (the existing sample manifest) instead.

`../legacy/` holds earlier-normalization exports (`*.cct.csv`) that are not used by `ml_pipeline.py` or the locked model — kept for reference only.
