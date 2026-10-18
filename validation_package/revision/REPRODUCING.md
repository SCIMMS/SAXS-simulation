# Reproduce the revision evidence

Start with a complete repository clone. Preserve the scientific snapshot and
original section directories described in the [package guide](README.md).

## Verify saved results

```sh
python validation_package/revision/scripts/verify_package.py
python validation_package/revision/scripts/verify_package.py --numerical
```

The first command checks current files, both dependency manifests and retained
scientific files, current relative README links, asset counts and archive
members. The second additionally runs the saved numerical audits and rechecks
the 187/209 within-frame paired result. These checks do not execute GPU fitting.

## Redraw current main figures

The current main figure directories contain compact plotting inputs or links
to unchanged repository inputs, together with neutral-named redraw helpers.
Run all six helpers together, or use the individual script documented in each
figure's README:

```sh
python validation_package/revision/scripts/regenerate_main_figures.py --output-dir NEW_FIGURE_DIRECTORY
```

Redrawing from saved arrays does not repeat template selection or change the numerical cohorts.
NumPy, Matplotlib and Pillow are needed; any additional figure-specific
requirements are listed with the corresponding asset.

For supplementary figures, the asset source indexes identify vector/native
images and scientific source arrays. Full scientific regeneration uses the
retained modules described below; raster source preservation is identified
separately from a fresh numerical render.

## Rerun scientific calculations

```sh
python validation_package/revision/scripts/prepare_workspace.py NEW_WORKSPACE --portable
```

The destination must not exist. This delegates to the retained materializer,
which reconstructs the research-layout inputs and scientific modules without
overwriting a working directory. Run the scientific commands from the newly
materialized workspace, following the
[mesh/CZT scientific reproduction guide](../revision_CZT_R19_2026_10_17/REPRODUCING.md)
and its [command list](../revision_CZT_R19_2026_10_17/commands.json).

For ID57 numerical validation, the full 8,400-candidate Ewald comparison,
physical baselines and supplementary refinement, follow the separate
[base-evidence reproduction guide](../revision_2026_10_17/REPRODUCING.md) and
[ordered scientific commands](../revision_2026_10_17/commands.json).
The GroEL command list supplements these studies; it does not replace them.

NumPy, SciPy, h5py, Matplotlib, VTK, scikit-image, periodictable, psutil and a
compatible CuPy/CUDA environment are required for the full calculations. On
Windows CUDA runtime, NVRTC, cuFFT and cuBLAS DLLs must be discoverable; the
retained portable adapter supports `SAXS_CUDA_DLL_DIRS`. The optional
`--fresh-groel` argument is forwarded to the retained materializer. See the
linked guide for exact input coverage, tolerances and flags.

The processed 76-shot GroEL cohort, density grids, layer meshes and historical
assignments are already retained. Large raw public files and regenerable
template banks are excluded. Main Figure 5 therefore has compact fixed inputs
for its two displayed frames; reconstructing the complete candidate bank is a
separate scientific rerun.

New figure/table presentation does not change saved scientific results.
GPU/library differences can affect floating-point output; source identities,
saved arrays and recorded tolerances distinguish scientific reproduction from
byte-for-byte image rendering.
