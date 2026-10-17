# SAXS-simulation

Figure, table, and validation-data package for:

Surface Mesh Projection for Low q Single Shot XFEL Diffraction

## Paper

The paper DOI will be added here after publication.

## Overview

This repository contains the compact source data, generated figure/table assets,
and processed validation summaries needed to audit the results reported in the
paper and its Supporting Information.

The current release is the [17 October 2026 revision validation package](validation_package/revision_2026_10_17/README.md), aligned with manuscript/SI revision R18. It adds the longitudinal complex-projection forward method, full five-plane sharp/rounded comparison, supplementary refinement controls, illumination tests, source snapshots and a lightweight saved-evidence audit.

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

Current numbering is indexed in the [R18 main assets](assets/main_manuscript/revision_2026_10_17/README.md) and [R18 SI assets](assets/supplementary_information/revision_2026_10_17/README.md). The original unversioned folders remain historical. Use the revision package's definitions and manifests for current claims; earlier ID57 implementation scores are not interchangeable with the corrected planar or five-plane scores.

To verify the current release without a GPU:

```sh
python validation_package/revision_2026_10_17/scripts/verify_package.py
python validation_package/revision_2026_10_17/scripts/verify_package.py --numerical
```

The optional numerical audit requires NumPy. Full scientific reruns require the inputs and CUDA environment described in the [reproduction guide](validation_package/revision_2026_10_17/REPRODUCING.md).

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
