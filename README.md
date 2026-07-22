# SAXS-simulation

Figure, table, and validation-data package for:

Surface-Mesh Projected-Density Simulation for Low-q Single-Shot XFEL
Diffraction

## Paper

The paper DOI will be added here after publication.

## Overview

This repository contains the compact source data, generated figure/table assets,
and processed validation summaries needed to audit the results reported in the
paper and its Supporting Information.

This is not a general-purpose software release. The experimental-data
interpretation source tree used during method development is intentionally not
included.

## Contents

- `assets/main_manuscript/`: main-text figure and table folders. Each folder
  contains the current image plus the compact source data used to recreate or
  inspect the item.
- `assets/supplementary_information/`: Supporting Information figure and table
  folders with image and data subfolders.
- `validation_package/`: curated validation and benchmark outputs supporting
  the manuscript. This package intentionally excludes large regenerable
  intermediates.

## Data Policy

Large raw public-data files are not included. Public experimental data should be
retrieved from the original public repositories, including CXIDB ID57 and CXIDB
ID187. The GroEL structural model should be retrieved from Protein Data Bank
entry 1SS8.

Small processed arrays, summaries, and curated previews needed to audit the
figures, tables, and validation claims are included.

## License and citation

All source-code files are licensed under the BSD 3-Clause License. Unless
otherwise noted, data, figures, tables, documentation, and other non-code
materials created by the repository authors are licensed under the Creative
Commons Attribution 4.0 International License (CC BY 4.0). See `LICENSE`,
`LICENSE-CODE`, and `LICENSE-DATA` for details and third-party-material
exceptions.

If you use this repository in academic work, please cite it using the metadata
in `CITATION.cff` and cite the associated article once its bibliographic
information is available.
