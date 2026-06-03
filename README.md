# SAXS-simulation

Manuscript and validation package for:

Fast Surface-Mesh Projected-Density Simulation for Low-q Single-Shot XFEL
Diffraction: Validation, Benchmarking, and Public-Data Template Retrieval

## Overview

This repository package organizes the manuscript draft, figure/table assets,
supplementary assets, validation outputs, and source scripts used for the
manuscript.

The package is intended for private GitHub review before journal submission.

## Contents

- `manuscript/`: section-by-section text drafts, captions, reference planning,
  MathML equations, and the current Word manuscript draft.
- `assets/main_manuscript/`: main-manuscript figure/table folders. Each folder
  keeps current images plus source data needed to recreate or revise the item.
- `assets/supplementary_information/`: supplementary figure/table folders with
  image and data subfolders.
- `validation_package/`: curated validation and benchmark package supporting
  the manuscript. This package intentionally excludes large regenerable
  intermediates.
- `software/saxs_simulator_source/`: source files for the SAXS simulator used as
  the development base. Packaged executables and build folders are excluded.
- `docs/`: professor review guide, reproducibility guide, GitHub upload notes,
  data policy notes, and manuscript-format notes.

## Data Policy

Large raw public-data files are not included in this repository package. Public
experimental data should be cited and retrieved from the original public
repositories, including CXIDB ID57 and CXIDB ID187, rather than committed here.

Small processed arrays, summaries, and curated previews needed to reproduce the
figures and tables are included.

## Suggested GitHub Workflow

1. Create a private GitHub repository.
2. Upload the contents of this `github_upload_package/` directory as the
   repository root.
3. Keep manuscript revisions in branches or tags, for example
   `v0.1-internal-review`, `v1.0-submission`, and `v1.1-revision`.
4. Before public release, review author names, acknowledgements, funding,
   conflicts of interest, data availability, and any AI-tool disclosure.
5. If a public archived version is needed, create a GitHub release and connect
   it to Zenodo or another DOI-providing archive.

## Excluded From This Package

- Raw CXIDB data and large public-data downloads.
- PyInstaller `dist/` and build outputs.
- Python virtual environments and caches.
- Codex/PDF-review scratch folders.
- Copyrighted journal PDFs.
