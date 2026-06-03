# Validation Supporting Package Manifest

Generated on 2026-06-02 from the local validation workspace.

This package keeps the material that is useful for the manuscript and possible
supporting information. It intentionally excludes private coned-droplet data and
large intermediate files that can be regenerated from the scripts.

## Structure

- `scripts/`: validation Python scripts, small requirements/checklist files, and
  utility modules used for public synthetic data, performance benchmarking,
  ID57 retrieval/control, and GroEL modeling/retrieval summaries.
- `section3_public_synthetic/`: public/synthetic shape validation outputs.
- `section4_performance/`: full public benchmark outputs.
- `section5_id57/`: CXIDB ID 57 nanoparticle validation.
- `section5_groel/`: GroEL protein-density modeling, curated run-151 hit
  preview, retrieval summaries, controls, and mesh convergence checks.

## ID57 Nanoparticle Validation

Use this as the main public-data retrieval validation.

- Synthetic library: 240 symmetry-aware orientations.
- Self-retrieval: 240/240 exact recovery.
- Symmetry-aware angular error: median 0 deg, max 0 deg.
- Real-vs-radial-control comparison: 209 paired shots.
- Real template median log-correlation: 0.724752.
- Radial-control median log-correlation: 0.443121.
- Median delta: +0.283249.
- Real template better in 209/209 pairs.

Key files:

- `section5_id57/synthetic_library_summary/`
- `section5_id57/synthetic_self_retrieval/`
- `section5_id57/retrieval_real/`
- `section5_id57/retrieval_radial_control/`
- `section5_id57/real_vs_random_control/`
- `section5_id57/rotation_libraries/`

Large omitted intermediates:

- full synthetic pattern NPZ files,
- all per-shot retrieval comparison PNGs,
- radial-control synthetic NPZ files.

These can be regenerated from the scripts and the public CXIDB ID 57 data.

## GroEL Protein-Density Modeling

Use this as a modeling and simulation demonstration, not as a claim of precise
orientation retrieval.

- PDB model: 1SS8, non-hydrogen atoms only.
- Non-H atom count: 27064.
- WK density grid generation time: about 51.8 s.
- Grid shape: 188 x 186 x 120.
- Expected electrons: 179402.484.
- Grid-integrated electrons: 179402.327.
- Template library: D7 symmetry-reduced orientations, 429 templates across
  model variants.

GroEL real-vs-radial-control retrieval is intentionally interpreted as limited:

- Curated run-151 shots: 76.
- Real median log-correlation: 0.610956.
- Radial-control median log-correlation: 0.605896.
- Median delta: +0.003249.

This small separation is consistent with the earlier concern that much of the
experimental image energy is explained by low-order/radial structure.

Key files:

- `section5_groel/groel_modeling_validation_note.md`
- `section5_groel/curated_run151_hits/`
- `section5_groel/retrieval_symD7_curated76/`
- `section5_groel/retrieval_random_control_curated76/`
- `section5_groel/real_vs_random_control/`
- `section5_groel/density_model_summary/`
- `section5_groel/model_assembly_summary/`
- `section5_groel/template_library_summary/`
- `section5_groel/mesh_convergence/`
- `section5_groel/z_projection_validation/`

Large omitted intermediates:

- VTP mesh files,
- full GroEL template HDF5 libraries,
- smoke-test template and retrieval outputs,
- old non-symmetry retrieval outputs.

## Mesh Decimation And Projection Checks

The mesh convergence summary supports using decimated meshes for practical
simulation.

- Median point ratio after decimation: 0.199897.
- Median cell ratio after decimation: 0.199993.
- Maximum absolute relative volume error: 0.001574.
- Component-count change: 0.
- Midpoint projection Pearson correlation: 0.999291 for original mesh and
  0.999289 for quadric-r80 mesh.

Key files:

- `section5_groel/mesh_convergence/`
- `section5_groel/z_projection_validation/`

## Recommended Manuscript Use

- Main validation: section 3 synthetic shapes, section 4 performance benchmark,
  and section 5 ID57 public nanoparticle retrieval/control.
- Supporting: ID57 self-retrieval, radial-control comparison, GroEL
  protein-density modeling, GroEL mesh convergence, and selected preview images.
- Avoid: private coned-droplet experimental data and GroEL over-interpretation
  as high-confidence orientation retrieval.
