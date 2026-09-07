# data/discovery/

Discovery-cohort inputs used to train the locked v11 pipeline (see `docs/gui_integration_guide.md` for exactly which files feed which sub-model).

`metabolome_sample_info.v4.0.tsv` is the metabolomics sample manifest read by `single_omics_metabolomics.R`.

`../legacy/` holds earlier-normalization exports (`*.cct.csv`) that are not used by `ml_pipeline.py` or the locked model — kept for reference only.
