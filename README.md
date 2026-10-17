# SAXS-simulation

Figure, table, and validation-data package for:

Surface Mesh Projection for Low q Single Shot XFEL Diffraction

## Paper

The paper DOI will be added here after publication.

## Overview

This repository contains the compact source data, generated figure/table assets,
and processed validation summaries needed to audit the results reported in the
paper and its Supporting Information.

This is not a general-purpose software release. The revision includes the scientific forward/validation modules needed to inspect these experiments; the simulator GUI and unrelated development source tree remain excluded.

## Revision

The current revised article is supported by these entry points:

- [Validation and reproduction guide](validation_package/revision/README.md)
- [Main figures and tables](assets/main_manuscript/revision/README.md): 6 figures and 2 tables
- [Supporting Information figures and tables](assets/supplementary_information/revision/README.md): 22 figures and 21 tables

Both primary experimental examples use mesh projection and CZT. The nanoparticle
comparison includes five longitudinal-frequency samples and physical Ewald
evaluation; the GroEL example uses the planar specialization, with FFT retained
as a numerical control. Optional differentiable refinement remains supplementary.

The current assets include the analysis-condition summary table, updated table
numbering, clarified plot definitions and high-resolution figures. These
presentation changes use the retained scientific results and processed inputs;
they do not introduce new fitting or a new performance benchmark. Earlier
snapshot directories remain available for numerical provenance. Use the
**Revision** links above for current figure and table numbering.

The optional current asset-and-guide archive is
[`revision_reproduction_supplement.zip`](validation_package/revision/downloads/revision_reproduction_supplement.zip).
Extract it at the root of a full repository copy. It supplements the retained
scientific evidence; the [package guide](validation_package/revision/README.md)
lists dependencies and archive checksums. The article points to this repository
instead of fixing an archive filename in its Data availability statement.

To verify the current package without a GPU:

```sh
python validation_package/revision/scripts/verify_package.py
python validation_package/revision/scripts/verify_package.py --numerical
```

The numerical audit requires NumPy. Full scientific reruns require the compatible
CUDA environment and inputs in the
[reproduction guide](validation_package/revision/REPRODUCING.md).

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
