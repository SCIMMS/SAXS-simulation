# Reproduction guide

## Two distinct levels

The lightweight audit checks the published bytes, figure/table mapping and recomputes statistics from retained records. It does not claim to repeat the scientific simulations. Full regeneration uses the research source snapshot, upstream inputs and a compatible CUDA environment.

Python 3.10 or later is sufficient for the standard-library package verifier. The saved-array numerical audit additionally requires NumPy. Full GPU studies were run on the recorded Windows/RTX 2070 SUPER environment. Historical benchmarks use a different Tesla T4 environment; see [environment records](provenance/environments.json). Install a CuPy build appropriate to the installed CUDA toolkit/driver. [Requirements](scripts/requirements.txt) list dependencies, not a universally valid lockfile.

## Prepare a separate workspace

From the repository root, choose a directory that does not already exist:

```sh
python validation_package/revision_2026_10_17/scripts/prepare_workspace.py ../saxs_revision_reproduction --portable
```

This copies the source snapshot, compact inputs and historical public evidence into the relative layout expected by the research scripts. It never modifies the repository evidence. `--portable` honors the installed CUDA setup and omits two absent unpublished DOCX files from preservation-only hash lists. It writes every adaptation with before/after hashes to `PUBLIC_PREPARATION.json`. It does not change geometry, forward calculations, objectives, bounds, masks or numerical data. Omit the option to materialize exact source bytes and adapt the original environment yourself.

On Windows the adapter discovers NVIDIA runtime DLL directories in the active Python environment. A complete compatible CUDA/CuPy installation, including cuBLAS, cuFFT and NVRTC, is still required. If these DLLs live outside the environment and system search path, set `SAXS_CUDA_DLL_DIRS` to their directory or semicolon-separated directories before running scientific commands. CUDA binaries are not bundled.

The preparation copies retained results and checkpoints for inspection. **For a fresh scientific rerun, first move that study's dated output directory aside in this new workspace.** Preserve the prerequisite studies needed by the next stage. Do not run a checkpoint-aware fitting script over the retained evidence and call that independent recomputation. Freshly generated source/config hashes will differ from historical manifests after environment adapters, path changes or different input archive bytes; historical manifests document the original run, not the new run.

## Required upstream and regenerated inputs

- **CXIDB 57**: obtain `normal-incidence-pattern.cxi` and its public geometry/metadata from [the dataset page](https://www.cxidb.org/id-57.html). Place them under `public_data/CXIDB_ID_57/`. The package retains the exact synthetic summary, orientation parameters and masks, but omits the 142 MB full synthetic pattern archive and raw CXI. The original scripts and historical package contain orientation-library generation and synthetic-pattern commands. Consult `python validation/section5_generate_id57_core_shell_test_images.py --help`; use the retained 240-view, 8° near-normal rotation library and parameters, rather than a newly sampled library. The expected directory is `public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240/`. The required archive keys are documented in the script; regenerating an archive does not reproduce its compressed-byte hash automatically.
- **Geometry-only complex-projection validation**: `data/inputs/id57_geometry_fixture.npz` preserves the exact valid mask and 240 rotation matrices without pattern arrays. Pass `--geometry-only` together with `--portable` to the workspace preparer to point *only* `revision_complex_projection_validation.py` at this fixture. Then its `controls`, `accuracy`, `benchmark` and `report` stages run without downloading ID57 observations or regenerating the pattern archive. Full ID57 fitting still requires the upstream CXI and full prerequisite studies. The adapter records the changed metadata input path; it does not change its values.
- **GroEL**: the compact, manually curated 76-shot processed HDF5, original class/alignment records, rotations and regenerated PDB-derived density grid are included at their original relative paths. This permits the fixed-cohort refinement and matched A/B studies without redistributing raw detector runs. Rebuilding the original curation/calibration pipeline requires upstream [CXIDB 187](https://www.cxidb.org/id-187.html) raw run and calibration files. The 76-shot sample is a fixed manual curation, not an automatic unbiased cohort rule. Cite [PDB 1SS8](https://www.rcsb.org/structure/1SS8) for the structural source.
- **Large caches**: the five-plane intensity bank (~5.7 GiB), measured ROI array, selected exact-boundary bank, earlier planar/exact libraries and large beam-control arrays are omitted. [Omission manifest](provenance/omitted_files.csv) records paths, sizes and original array hashes. Recreate them in scientific order. All 209 × 8,400 training-score matrices and selected-fit records are retained for auditing selection and cohort summaries without these caches.

## Scientific execution order

[commands.json](commands.json) groups the commands and their dependencies. Run them from the prepared workspace root with the scientific Python environment. These commands describe the original stages; no broad GPU refitting is implied by the public-package audit.

1. Original synthetic metadata and planar ID57 consistency, physical ellipse baseline and rounding bank establish the common masks and candidate parameters.
2. Cached-volume and independent-boundary Ewald checks establish numerical/physical error scopes.
3. Complex projections: controls, 240-view sharp/4/12 nm accuracy sweeps, isolated-process benchmarks for each K and geometry, then diagnostics/reporting.
4. Five-plane ID57: generate all candidates, score all three folds, select candidates/nuisance on training pixels, audit selected geometries using independent boundary integrals, then report. This stage may consume multiple GiB and substantial GPU time/disk space.
5. Supplementary differentiable fits: synthetic controls; planar ID57; fixed-class GroEL and cross-class controls; forward bridge, validation-only penalty selection and local-start sensitivity. Complete fitting and reporting are distinct steps in the source scripts.
6. Five-plane radius and nested-aspect controls depend on the five-plane input bank and saved training-selected seeds. Both expanded aspect ranges and failures/bounds are reported. No range is selected using held-out gains.
7. Beam-truncation and matched GroEL A/B studies use their own stated planar/control conventions. They are not evidence for a combined illumination-plus-five-plane fit.

Some scripts import the scientific GPU environment even in reporting mode. Use the standalone package verifier for an audit on a machine without CUDA. Several research scripts write figures and checkpoint records; run only in the separate prepared workspace.

## Provenance and publication transformations

Scientific `.py` snapshots retain their original bytes, including line endings, except for two reporting helpers whose editorial response-drafting text was removed for the public release. Numerical calculation, audit and plotting code is unchanged. `provenance/source_records.json` records every original path/hash, published hash and transformation. JSON records containing this workstation's absolute directory or external-drive label replace those labels with `${WORKSPACE}`/`${GROEL_RAW}`; values, fit parameters and scores are not altered. Historical in-record hashes remain the original-run provenance, even when the containing JSON's path labels were normalized. The release `MANIFEST.json` is the authority for published bytes.

Current figure PNGs are exact embedded images extracted from the final R18 main/SI documents; the documents themselves and referee correspondence are excluded. Current table CSVs preserve displayed values, Unicode units and rounding. Historical memory CSV column names ending `_MB` are left intact for provenance; the final figure and interpretation use MiB. Original unversioned assets and results remain historical, with explicit links to the current release.
