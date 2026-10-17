# SAXS-simulation

Figure, table, and validation-data package for:

Surface Mesh Projection for Low q Single Shot XFEL Diffraction

## Paper

The paper DOI will be added here after publication.

## Overview

This repository contains the compact source data, generated figure/table assets,
and processed validation summaries needed to audit the results reported in the
paper and its Supporting Information.

The current release is the [R19 mesh/CZT supplement](validation_package/revision_CZT_R19_2026_10_17/README.md), aligned with manuscript/SI R19. It makes both primary experimental examples use mesh projection and CZT, retains FFT only as a numerical control, and updates the GroEL controls, A/B comparison and figure/table sources. The [R18 base](validation_package/revision_2026_10_17/README.md) supplies the unchanged ID57 complex-projection/five-plane results, supplementary refinement evidence and exact processed inputs.

This is not a general-purpose software release. The revision includes the scientific forward/validation modules needed to inspect these experiments; the simulator GUI and unrelated development source tree remain excluded.

## Contents

- `assets/main_manuscript/`: main-text figure and table folders. Each folder
  contains the current image plus the compact source data used to recreate or
  inspect the item.
- `assets/supplementary_information/`: Supporting Information figure and table
  folders with image and data subfolders.
- `validation_package/`: curated validation and benchmark outputs supporting
  the manuscript. This package intentionally excludes large regenerable
  intermediates.

Current numbering is indexed in the [R19 main assets](assets/main_manuscript/revision_CZT_R19_2026_10_17/README.md) and [R19 SI assets](assets/supplementary_information/revision_CZT_R19_2026_10_17/README.md). Earlier versioned and unversioned folders remain historical. Use current definitions and manifests for present claims.

The optional current supplement ZIP is [`revision_CZT_R19_reproduction_supplement_2026-10-17.zip`](validation_package/revision_CZT_R19_2026_10_17/downloads/revision_CZT_R19_reproduction_supplement_2026-10-17.zip). Extract at the root of a full repository copy; the R18 base and original public section folders are required. [The package README](validation_package/revision_CZT_R19_2026_10_17/README.md) gives the archive filename, checksum record, dependency layout and the status of earlier local working archives. The article refers to this repository instead of fixing ZIP filenames in its Data availability statement.

To verify the current release without a GPU:

```sh
python validation_package/revision_CZT_R19_2026_10_17/scripts/verify_package.py
python validation_package/revision_CZT_R19_2026_10_17/scripts/verify_package.py --numerical
```

The optional numerical audit requires NumPy. Full scientific reruns require the compatible CUDA environment and inputs in the [reproduction guide](validation_package/revision_CZT_R19_2026_10_17/REPRODUCING.md). The new paired radial-control protocol shares each historical assigned state and differs from the earlier independently selected FFT-control library; its changed score gap must not be attributed to CZT.

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
