# Data Sources And Reproducibility Notes

This package separates small processed results from large public raw data.

Current GitHub repository:

https://github.com/OkKakao/SAXS-simulation

The repository is currently intended for private manuscript review. Before
journal submission or acceptance, the authors should decide whether to make the
repository public, create a public release only, and/or archive the release with
a permanent DOI.

## Public Data

- CXIDB ID57 is used for the main public nanoparticle retrieval validation.
- CXIDB ID187 is used for the GroEL-related modelling and supplementary
  demonstration.
- PDB 1SS8 is used for the GroEL structural model.

Large public-data downloads are not committed to this package. They should be
downloaded from the original public repositories when the analyses are rerun.

## Included Processed Data

The package includes compact CSV, JSON, PNG, NPZ, NPY, and HDF5 files when they
are needed to recreate a manuscript figure, table, or supplementary item. These
files are intentionally small enough for ordinary GitHub use.

## Regeneration Entry Points

- Main figure/table asset organization:
  `validation/organize_manuscript_visual_assets.py` in the original workspace.
- Curated validation scripts:
  `validation_package/scripts/`.
- Source simulator files:
  `software/saxs_simulator_source/`.

Some regeneration commands may need local Python environments and public raw
data paths that are not included in this repository package.

## Provisional Data Availability Wording

The manuscript draft, figure and table source data, processed validation
summaries, and scripts used to organize the reported analyses are currently
maintained in a private GitHub repository at
https://github.com/OkKakao/SAXS-simulation. A public archival release, with a
permanent DOI if required, will be prepared from the accepted or submitted
version of the repository. Large raw public datasets are not redistributed in
the repository. CXIDB ID57 and CXIDB ID187 data should be obtained from CXIDB,
and the GroEL structural model should be obtained from the Protein Data Bank
entry 1SS8.
