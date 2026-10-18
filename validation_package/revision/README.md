# Revision validation package

This is the current entry point for the revised article's figures, tables and
validation evidence. It follows the current manuscript numbering: six main
figures, two main tables, 22 supplementary figures and 21 supplementary tables.

## Current figures and tables

- [Main assets](../../assets/main_manuscript/revision/README.md)
- [Supporting Information assets](../../assets/supplementary_information/revision/README.md)
- [Reproduction instructions](REPRODUCING.md)
- [Analysis conditions and evidence map](DATA_DICTIONARY.md)

Main Table 1 summarizes the forward route, fitting and selection objective,
evaluation pixels and kept fixed quantities for each analysis. Main Table 2 is the
existing fixed-morphology benchmark. The figures define coordinates, scales and
residuals explicitly; embedded PNGs meet a minimum effective resolution of
600 dpi at the manuscript's drawing sizes. Available vector sources and compact
plotting inputs accompany the images.

The presentation and documentation changes reuse the saved numerical results.
They do not add a new fitting run, a new geometry-changing benchmark or a new
structural-identification claim. Differentiable refinement remains an optional
supplementary example.

## Retained scientific evidence

The current package depends on two immutable scientific snapshots:

- [Nanoparticle, complex-projection and supplementary evidence](../revision_2026_10_17/README.md)
- [Consistent mesh/CZT GroEL calculations and numerical controls](../revision_CZT_R19_2026_10_17/README.md)

Keep these directories, the original `validation_package/section*` directories
and `validation_package/scripts/` in their repository-relative locations. The
current manifest pins the dependency manifests by SHA256. Earlier directory
names are provenance identifiers; they are not the current asset index.

The five-plane ID57 comparison reselects the entire 8,400-candidate library.
Rounded models have a higher test correlation score than variable-size sharp
models in 187 of 209 frames after averaging the two complementary split
directions within each frame. This supports sensitivity to the tested boundary
families, with parameter-bound limitations on radius interpretation.

The primary GroEL comparison freezes historical class, view and alignment and
uses planar mesh/CZT templates. Its radial control shares the assigned state.
The separate A/B comparison selects a view within each assembly on fitting
bands. Historical full-ROI assignments or alignment make the within-image
holdouts conditional; these comparisons do not establish assembly identity.
The control protocol differs from the earlier independently selected FFT
library, so a changed control gap must not be attributed to CZT alone.

## Verify and reproduce

```sh
python validation_package/revision/scripts/verify_package.py
python validation_package/revision/scripts/verify_package.py --numerical
python validation_package/revision/scripts/prepare_workspace.py NEW_WORKSPACE --portable
```

Integrity verification uses only the Python standard library. The optional
saved-result audit also requires NumPy. Neither command reruns GPU simulations.
The [reproduction guide](REPRODUCING.md) distinguishes saved-data plotting from
scientific reruns and records the environment requirements.

## Optional archive

The optional archive is
[`revision_reproduction_supplement.zip`](downloads/revision_reproduction_supplement.zip).
It contains this current guide and the current main/SI asset directories, with
repository-relative member paths. Extract it at the root of a complete repository
copy. It does not duplicate the scientific snapshots or original section inputs.
The archive's SHA256, byte size and member count are recorded in
[archive.json](provenance/archive.json).

`MANIFEST.json` identifies the canonical current files and dependency hashes.
The archive, its checksum record and the post-package audit are excluded from
that manifest to avoid circular hashes. Manifests and source indexes distinguish
current plot assets from their retained numerical evidence.

Code: BSD 3-Clause. Author-created data, figures and documentation: CC BY 4.0.
Upstream inputs retain their original terms. Manuscripts, reviewer correspondence,
raw CXI files and unrelated development files are excluded from this package.
