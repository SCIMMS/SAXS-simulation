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

## Excluded From This Package

- Raw CXIDB data and large public-data downloads.
- Experimental-data interpretation source code used during development.
- Packaged executables, build outputs, virtual environments, and caches.
- Manuscript drafts and review notes.
