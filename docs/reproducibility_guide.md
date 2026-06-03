# Reproducibility Guide

This guide describes what data were used, which processed outputs are already
included in this repository, and which commands can be used to regenerate the
main validation and supplementary results.

The commands are written for Windows PowerShell from the repository root.
Paths can be adjusted for Linux/macOS by replacing `\` with `/` and activating
the virtual environment in the usual shell-specific way.

## Reproducibility Levels

There are three practical levels of reproduction.

1. **Audit included outputs.** This requires only the files already committed to
   this repository.
2. **Regenerate public/synthetic validation outputs.** This requires Python,
   the simulator runtime dependencies, and no private experimental data.
3. **Regenerate raw-data-dependent outputs.** This requires downloading public
   CXIDB/PDB files and, for full GroEL raw processing, the larger external raw
   CXIDB ID187 data products.

The manuscript results were organized to avoid private experimental data. Raw
public datasets are not redistributed in this repository.

## Repository Layout

- `manuscript/`: manuscript text, captions, reference planning, MathML
  equations, and provisional data-availability wording.
- `assets/main_manuscript/`: current figure/table images and source data for
  main-manuscript items.
- `assets/supplementary_information/`: current supplementary figure/table
  images and source data.
- `validation_package/scripts/`: scripts used to generate or summarize
  validation outputs.
- `validation_package/section3_public_synthetic/`: numerical validation outputs
  for analytical/synthetic objects.
- `validation_package/section4_performance/`: benchmark outputs.
- `validation_package/section5_id57/`: CXIDB ID57 nanoparticle validation
  outputs.
- `validation_package/section5_groel/`: GroEL modelling and supplementary
  comparison outputs.
- `software/saxs_simulator_source/`: source files for the SAXS simulator used as
  the development base.

## Data Used

| Manuscript role | Data source | Repository status |
| --- | --- | --- |
| Section 3 numerical validation | Analytical/synthetic sphere, ellipsoid, box/polyhedron, and CZT/q-grid tests | Processed CSV/JSON/NPZ outputs included under `validation_package/section3_public_synthetic/` |
| Section 4 performance benchmark | Public-safe synthetic morphologies: sphere, ellipsoid, two-droplet composite, icosahedron, Gaussian scalar-field object | Processed CSV/JSON outputs included under `validation_package/section4_performance/` |
| Section 5 ID57 public-data validation | CXIDB ID57 normal-incidence core-shell nanoparticle patterns: `normal-incidence-pattern.cxi` and `new.geom` | Raw CXIDB files not included; processed summaries and representative images included under `validation_package/section5_id57/` |
| Supplementary ID57 self-retrieval/control | Synthetic ID57 core-shell template library generated from public ID57 detector geometry | Full synthetic template NPZ omitted as regenerable; rotation library, summaries, retrieval results, and plots included |
| GroEL density model | PDB 1SS8 non-hydrogen atoms | Raw PDB not included in repository root; density/mesh summaries and previews included |
| GroEL public-shot demonstration | CXIDB ID187 GroEL-related data, including run-151 curated hits, `rotations.h5`, `histograms.h5`, `mask.h5`, and PDB-related files when fully rerun | Compact curated 76-shot HDF5 and processed outputs included; full raw PNCCD data and large intermediate template/mesh files omitted |

## Environment Setup

The exact validation environment can vary. The benchmark summary records Python
3.12.13, NumPy 2.0.2, SciPy 1.16.3, Open3D 0.19.0, CuPy 14.0.1, and a Tesla T4
GPU for the full Section 4 benchmark. CPU-only runs can audit many processed
outputs, but GPU timing values will differ.

Create a local environment:

```powershell
cd "C:\path\to\SAXS-simulation"
py -3.12 -m venv .venv_repro
.\.venv_repro\Scripts\python.exe -m pip install --upgrade pip
.\.venv_repro\Scripts\python.exe -m pip install -r validation_package\scripts\requirements_cxidb.txt
.\.venv_repro\Scripts\python.exe -m pip install scipy open3d
```

Optional GPU acceleration requires a CUDA-compatible GPU and a matching CuPy
build. The original benchmark used CuPy 14.0.1 and the `signed_raster_gpu`
projection backend.

## Quick Audit Of Included Results

These commands read the committed summary files and print the key manuscript
metrics without regenerating raw data.

```powershell
$py = ".\.venv_repro\Scripts\python.exe"

& $py -c "import json, pathlib; s=json.loads(pathlib.Path('validation_package/section5_id57/synthetic_self_retrieval/id57_synthetic_self_retrieval_summary.json').read_text()); print('ID57 self retrieval:', s['n_queries'], s['exact_index_accuracy'], s['score_log_corr']['median'])"

& $py -c "import json, pathlib; s=json.loads(pathlib.Path('validation_package/section5_id57/real_vs_random_control/id57_real_vs_random_control_summary.json').read_text()); d=s['delta_score_log_corr_real_minus_control']; print('ID57 real-vs-control:', s['n_paired'], d['median'], d['real_better_count'])"

& $py -c "import json, pathlib; s=json.loads(pathlib.Path('validation_package/section5_groel/real_vs_random_control/groel_real_vs_random_control_summary.json').read_text()); d=s['delta_logcorr_real_minus_control']; print('GroEL real-vs-control:', s['n_paired'], d['median'], d['real_better_count'])"

& $py -c "import json, pathlib; s=json.loads(pathlib.Path('validation_package/section5_groel/mesh_convergence/groel_mesh_convergence_summary.json').read_text()); print('GroEL mesh decimation:', s['decimation']['median_point_ratio'], s['decimation']['max_abs_relative_volume_error'])"
```

Expected values from the committed summaries:

- ID57 synthetic self-retrieval: 240 queries, exact-index accuracy 1.0, median
  log-correlation 0.966886.
- ID57 real-vs-radial-control: 209 paired shots, median score delta +0.283249,
  structured library better in 209/209 comparisons.
- GroEL real-vs-radial-control: 76 paired shots, median log-correlation delta
  +0.003249, protein-derived library better in 61/76 comparisons.
- GroEL mesh decimation: median point ratio 0.199897, maximum absolute relative
  volume error 0.001574.

## Section 3: Public/Synthetic Numerical Validation

The Section 3 script is a notebook-style script. It expects the main simulator
definitions to be loaded first, including `StructureGeometry`, `project_density`,
`signed_mesh_volume`, and `czt2_from_real_axes`.

In the original workflow this was run after executing the main simulator cell.
From a notebook or Python session where the simulator definitions are loaded:

```python
exec(open("validation_package/scripts/section3_data_acquisition_v3_public.py", encoding="utf-8").read())
run_all_section3_data_acquisition(
    out_root="reproduced/section3_validation_outputs_public",
    prefer_gpu=True,
    n_orientations=200,
)
```

Committed outputs to compare against:

- `validation_package/section3_public_synthetic/section3_validation_outputs_public/summaries/`
- `validation_package/section3_public_synthetic/section3_validation_outputs_public/json/`
- `validation_package/section3_public_synthetic/section3_validation_outputs_public/npz/`

No external experimental data are used in this section.

## Section 4: Performance Benchmark

The Section 4 script has the same notebook-style requirement: load the main
simulator definitions first, then run the benchmark functions.

Quick benchmark:

```python
exec(open("validation_package/scripts/section4_performance_benchmark_v2_public.py", encoding="utf-8").read())
results = run_all_section4_benchmarks(
    out_root="reproduced/section4_performance_outputs_public_quick",
    prefer_gpu=True,
    quick=True,
)
```

Full benchmark matching the manuscript scope:

```python
exec(open("validation_package/scripts/section4_performance_benchmark_v2_public.py", encoding="utf-8").read())
results = run_all_section4_benchmarks(
    out_root="reproduced/section4_performance_outputs_public_full",
    prefer_gpu=True,
    quick=False,
)
```

Committed outputs to compare against:

- `validation_package/section4_performance/section4_performance_outputs_public_full/summaries/`
- `validation_package/section4_performance/section4_performance_outputs_public_full/json/`

Benchmark numbers are hardware- and backend-dependent. The committed full
benchmark used the `signed_raster_gpu` backend, two warm-up runs, five measured
repeats, and a Tesla T4 GPU.

## Section 5: CXIDB ID57 Public Nanoparticle Validation

Required external files if regenerating from raw public data:

```text
public_data/CXIDB_ID_57/normal-incidence-pattern.cxi
public_data/CXIDB_ID_57/new.geom
```

The committed repository includes compact processed summaries, representative
retrieval images, and the 240-orientation rotation library. The full synthetic
pattern NPZ is omitted because it can be regenerated.

Probe the public CXI file:

```powershell
$py = ".\.venv_repro\Scripts\python.exe"
& $py validation_package\scripts\section5_cxidb_id57_probe.py `
  --data-dir public_data\CXIDB_ID_57 `
  --out-dir reproduced\section5_id57\analysis_preview
```

Generate the 240-orientation symmetry-aware rotation library:

```powershell
& $py validation_package\scripts\section5_make_id57_rotation_library.py `
  --out-dir reproduced\section5_id57\rotation_libraries `
  --count 240 `
  --tilt-max-deg 8.0 `
  --start-index 1 `
  --min-separation-deg 0.05 `
  --name id57_normal_incidence_n0240_tilt08
```

Generate the fixed core-shell synthetic template library:

```powershell
& $py validation_package\scripts\section5_generate_id57_core_shell_test_images.py `
  --data-dir public_data\CXIDB_ID_57 `
  --rotation-library reproduced\section5_id57\rotation_libraries\id57_normal_incidence_n0240_tilt08.npz `
  --out-dir reproduced\section5_id57\synthetic_library_summary `
  --count 240 `
  --outer-cube-side-nm 52.0 `
  --core-octa-vertex-radius-nm 26.2 `
  --rho-pd 1.0 `
  --rho-au 1.5 `
  --projection-grid 384 `
  --projection-extent-nm 144.0 `
  --depth-steps 256 `
  --depth-extent-nm 120.0 `
  --background-adu 8.0 `
  --save-individual-png-count 0 `
  --montage-count 240
```

Run synthetic noisy-to-clean self-retrieval:

```powershell
& $py validation_package\scripts\section5_id57_synthetic_self_retrieval.py `
  --synthetic-dir reproduced\section5_id57\synthetic_library_summary `
  --output-dir reproduced\section5_id57\synthetic_self_retrieval `
  --inner-radius-px 45.0 `
  --outer-radius-px 190.0
```

Score public ID57 frames against the structured library:

```powershell
& $py validation_package\scripts\section5_id57_retrieval.py `
  --data-dir public_data\CXIDB_ID_57 `
  --synthetic-dir reproduced\section5_id57\synthetic_library_summary `
  --out-dir reproduced\section5_id57\retrieval_real `
  --top-experimental-count 209 `
  --top-k 5 `
  --inner-radius-px 45.0 `
  --outer-radius-px 190.0 `
  --template-kind clean `
  --max-comparison-png 8
```

Create the radial-profile randomized control library and score it:

```powershell
& $py validation_package\scripts\section5_id57_make_random_control_npz.py `
  --input-dir reproduced\section5_id57\synthetic_library_summary `
  --output-dir reproduced\section5_id57\synthetic_radial_control `
  --seed 20260602 `
  --inner-radius-px 45.0 `
  --outer-radius-px 190.0

& $py validation_package\scripts\section5_id57_retrieval.py `
  --data-dir public_data\CXIDB_ID_57 `
  --synthetic-dir reproduced\section5_id57\synthetic_radial_control `
  --out-dir reproduced\section5_id57\retrieval_radial_control `
  --top-experimental-count 209 `
  --top-k 5 `
  --inner-radius-px 45.0 `
  --outer-radius-px 190.0 `
  --template-kind clean `
  --max-comparison-png 0
```

Compare structured templates against the control:

```powershell
& $py validation_package\scripts\section5_id57_compare_control_scores.py `
  --real-best-csv reproduced\section5_id57\retrieval_real\retrieval_best.csv `
  --control-best-csv reproduced\section5_id57\retrieval_radial_control\retrieval_best.csv `
  --output-dir reproduced\section5_id57\real_vs_random_control
```

Committed outputs to compare against:

- `validation_package/section5_id57/rotation_libraries/`
- `validation_package/section5_id57/synthetic_library_summary/`
- `validation_package/section5_id57/synthetic_self_retrieval/`
- `validation_package/section5_id57/retrieval_real/`
- `validation_package/section5_id57/retrieval_radial_control/`
- `validation_package/section5_id57/real_vs_random_control/`

## Supplementary GroEL Modelling And Control Comparison

Required public files for full regeneration:

```text
public_data/CXIDB_ID_187/1SS8.pdb
public_data/CXIDB_ID_187/rotations.h5
public_data/CXIDB_ID_187/histograms.h5
public_data/CXIDB_ID_187/mask.h5
```

The full raw PNCCD run files are not included. The repository includes the
curated downsampled 76-shot HDF5 used for the manuscript:

```text
validation_package/section5_groel/curated_run151_hits/hits_r0151_curated76_ds8.h5
```

For full raw GroEL processing, see:

```text
validation_package/scripts/GroEL_CXIDB_ID187_external_processing_checklist_EN.txt
```

Build the PDB-derived density grid and isosurface meshes:

```powershell
& $py validation_package\scripts\section5_build_1ss8_wk_density_meshes.py `
  --pdb public_data\CXIDB_ID_187\1SS8.pdb `
  --output-dir reproduced\section5_groel\protein_1ss8_wk_density_meshes_sigma6 `
  --spacing-A 1.0 `
  --smooth-sigma-A 6.0 `
  --clip-sigma 3.0 `
  --threshold-fractions 0.10,0.30,0.50,0.70,0.90 `
  --decimate-reduction 0.80 `
  --save-density `
  --save-meshes
```

Assemble C7/D7 GroEL mesh models:

```powershell
& $py validation_package\scripts\section5_assemble_1ss8_groel_models.py `
  --mesh-dir reproduced\section5_groel\protein_1ss8_wk_density_meshes_sigma6 `
  --output-dir reproduced\section5_groel\protein_1ss8_groel_models_sigma6 `
  --threshold-fractions 0.10,0.30,0.50,0.70,0.90 `
  --suffixes original quadric_r80
```

Validate z-projection consistency:

```powershell
& $py validation_package\scripts\section5_validate_1ss8_z_projection.py `
  --density-npz reproduced\section5_groel\protein_1ss8_wk_density_meshes_sigma6\1ss8_wk_density_grid.npz `
  --mesh-dir reproduced\section5_groel\protein_1ss8_wk_density_meshes_sigma6 `
  --output-dir reproduced\section5_groel\z_projection_validation `
  --threshold-fractions 0.10,0.30,0.50,0.70,0.90
```

Summarize mesh decimation and projection convergence:

```powershell
& $py validation_package\scripts\section5_groel_mesh_convergence_summary.py `
  --mesh-stats-csv reproduced\section5_groel\protein_1ss8_groel_models_sigma6\1ss8_groel_model_mesh_stats.csv `
  --projection-summary-json reproduced\section5_groel\z_projection_validation\1ss8_z_projection_validation_summary.json `
  --output-dir reproduced\section5_groel\mesh_convergence
```

Generate GroEL preview panels:

```powershell
& $py validation_package\scripts\section5_generate_groel_model_ab_preview.py `
  --density-npz reproduced\section5_groel\protein_1ss8_wk_density_meshes_sigma6\1ss8_wk_density_grid.npz `
  --rotations-h5 public_data\CXIDB_ID_187\rotations.h5 `
  --output-dir reproduced\section5_groel\polyprism_preview `
  --num-rotations 4 `
  --include-prism-models `
  --prism-radii-A 35,55 `
  --seed 187
```

Generate the D7 symmetry-reduced GroEL template library:

```powershell
& $py validation_package\scripts\section5_groel_generate_template_library.py `
  --density-npz reproduced\section5_groel\protein_1ss8_wk_density_meshes_sigma6\1ss8_wk_density_grid.npz `
  --rotations-h5 public_data\CXIDB_ID_187\rotations.h5 `
  --output-dir reproduced\section5_groel\template_library `
  --output-name groel_templates_B_prism35_55_symD7_sep10_step128.h5 `
  --models B `
  --include-prism-models `
  --prism-radii-A 35,55 `
  --rotation-start 0 `
  --rotation-stop 25680 `
  --rotation-step 128 `
  --symmetry D7 `
  --symmetry-min-separation-deg 10.0 `
  --image-size 128 `
  --downsample 8
```

Create the GroEL radial-profile randomized control library:

```powershell
& $py validation_package\scripts\section5_groel_make_random_control_library.py `
  --input-library reproduced\section5_groel\template_library\groel_templates_B_prism35_55_symD7_sep10_step128.h5 `
  --output-library reproduced\section5_groel\template_library_random_control\groel_templates_random_radial_control_n429_seed20260602.h5 `
  --seed 20260602 `
  --active-mask annular
```

Run retrieval against the curated public-shot file:

```powershell
& $py validation_package\scripts\section5_groel_retrieval.py `
  --pattern-file validation_package\section5_groel\curated_run151_hits\hits_r0151_curated76_ds8.h5 `
  --pattern-dataset patterns_ph `
  --mask-dataset mask `
  --template-files reproduced\section5_groel\template_library\groel_templates_B_prism35_55_symD7_sep10_step128.h5 `
  --output-dir reproduced\section5_groel\retrieval_symD7_curated76 `
  --top-pattern-count 76 `
  --top-k 8 `
  --shift-radius 6 `
  --max-comparison-png 16

& $py validation_package\scripts\section5_groel_retrieval.py `
  --pattern-file validation_package\section5_groel\curated_run151_hits\hits_r0151_curated76_ds8.h5 `
  --pattern-dataset patterns_ph `
  --mask-dataset mask `
  --template-files reproduced\section5_groel\template_library_random_control\groel_templates_random_radial_control_n429_seed20260602.h5 `
  --output-dir reproduced\section5_groel\retrieval_random_control_curated76 `
  --top-pattern-count 76 `
  --top-k 8 `
  --shift-radius 6 `
  --max-comparison-png 16
```

Compare GroEL protein-derived templates against the control:

```powershell
& $py validation_package\scripts\section5_groel_compare_control_scores.py `
  --real-best-csv reproduced\section5_groel\retrieval_symD7_curated76\groel_retrieval_best.csv `
  --control-best-csv reproduced\section5_groel\retrieval_random_control_curated76\groel_retrieval_best.csv `
  --output-dir reproduced\section5_groel\real_vs_random_control
```

Committed outputs to compare against:

- `validation_package/section5_groel/density_model_summary/`
- `validation_package/section5_groel/model_assembly_summary/`
- `validation_package/section5_groel/z_projection_validation/`
- `validation_package/section5_groel/mesh_convergence/`
- `validation_package/section5_groel/polyprism_preview/`
- `validation_package/section5_groel/template_library_summary/`
- `validation_package/section5_groel/retrieval_symD7_curated76/`
- `validation_package/section5_groel/retrieval_random_control_curated76/`
- `validation_package/section5_groel/real_vs_random_control/`

## Figure And Table Asset Regeneration

The current figure/table images and source data are organized under `assets/`.
The original workspace used an asset-organization script:

```powershell
.\.venv_repro\Scripts\python.exe validation\organize_manuscript_visual_assets.py
```

That script is referenced for provenance but is not the primary public
regeneration entry point in this repository package. For GitHub review, use the
committed `assets/main_manuscript/` and `assets/supplementary_information/`
folders as the source-of-truth figure/table asset package.

## Outputs Intentionally Omitted

The following outputs are intentionally not committed:

- Raw CXIDB downloads.
- Full ID57 synthetic pattern NPZ libraries.
- Full GroEL template HDF5 libraries.
- VTP mesh files and large mesh intermediates.
- PyInstaller `dist/` and build folders.
- Private experimental droplet data.

These files are either public raw data that should be obtained from original
repositories, large regenerable intermediates, or private/non-distributable
material outside the public manuscript validation scope.

