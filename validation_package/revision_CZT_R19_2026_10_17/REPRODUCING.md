# Reproduce R19 scientific results

## Saved evidence, CPU audit

Start from a complete repository clone. Run `python validation_package/revision_CZT_R19_2026_10_17/scripts/verify_package.py --numerical` with NumPy installed. This checks the release and its exact R18 dependency manifest, all archive members, record coverage, paired summaries, A/B counts and exact embedded figure/table source identities recorded in the asset manifests. It does not execute GPU forward calculations.

## Separate research-layout workspace

```sh
python validation_package/revision_CZT_R19_2026_10_17/scripts/prepare_workspace.py NEW_WORKSPACE --portable
```

The destination must not exist. The materializer calls the retained R18 preparer, copies exact inputs/evidence and source modules, then overlays the four new source modules plus any explicitly recorded exact dependency overrides. `--portable` applies the base's documented CUDA discovery and absent-unpublished-DOCX provenance guards in the new copy only. The published source snapshots remain unchanged. Do not run simulations inside the evidence package.

On Windows the CUDA runtime, NVRTC, cuFFT and cuBLAS DLLs must be discoverable. A CuPy wheel alone may not supply all of them. The portable adapter discovers the environment's `nvidia/*/bin` directories; additional installed CUDA library directories can be supplied through `SAXS_CUDA_DLL_DIRS` (semicolon-separated on Windows). The original local run obtained cuBLAS from a separate installed simulator runtime; that binary runtime is not distributed. In the public-package replay, setting this environment variable to that existing cuBLAS directory resolved an initial DLL import failure; no scientific source or numerical setting changed.

To recompute both new GroEL studies from scratch, add `--fresh-groel` when preparing a NEW workspace. This moves the two copied result folders into `retained_R19_evidence/` in that new workspace before running anything; prior input meshes, density, curated observations and original assignment records remain in their research-layout paths. No existing destination is overwritten.

Use a Python environment with NumPy, SciPy, h5py, Matplotlib, VTK, scikit-image, periodictable, psutil and one compatible CuPy CUDA distribution. Consult the original scientific environment/audits and the base requirements; the requirements list is not a lockfile. Run each command in [commands.json](commands.json) from the materialized workspace root in the listed order. GPU output can vary at floating-point precision with CUDA/BLAS/library versions; saved numerical tolerances are in the audits.

The exact processed 76-shot HDF5, assignments, rotation library, smoothed density grid, twelve-layer meshes, finer meshes and legacy selected templates are inherited from the R18 base and original public section folders. Original raw CXIDB files and the full historical 429-template HDF5 are not needed for this fixed-assignment rerun. ID57 or upstream density regeneration still needs the source public files documented in the base reproduction guide.

1. `revision_groel_mesh_czt_matched.py all` generates the four path variants for all 34 assigned templates, fits the fixed 76-shot cohort and runs seven predetermined grid/decimation controls.
2. `revision_groel_mesh_czt_report.py audit` performs independent fitting checks, all-34 sparse boundary comparisons and CPU/GPU reference controls; `report` makes summaries and diagnostic plots.
3. `revision_groel_czt_primary.py all` generates same-assignment radial controls, all A/B predictions over 201 views and the separate A/B selections.
4. `revision_groel_czt_public_figures.py audit` checks all primary fitting records and selected-view ranks, plus predetermined A/B boundary references; `figures` makes the four public GroEL figure images.

When only regenerating public figures from retained arrays, preserve the copied outputs and run the last `figures` command. This script still imports the scientific CUDA modules and computes the synthetic S12 mesh/CZT preview. File integrity checks alone require only the standard Python library; numerical saved-result checks require NumPy.

Archive: `revision_CZT_R19_reproduction_supplement_2026-10-17.zip`; extract repository-relative paths at a full repository root. It is an optional transport of the supplement, not a replacement for its R18 and original-section dependencies. The original local R12/R13 editorial ZIP bundles are not required.
