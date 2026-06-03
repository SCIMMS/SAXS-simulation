# Validation Cleanup Report

Cleanup date: 2026-06-02

## Summary

- Created `validation_supporting_package/` as the retained supporting-data bundle.
- Package contents: 155 files, about 79.8 MB.
- Deleted 36 redundant output/intermediate directories.
- Approximate space reclaimed: 726 MB.

## Retained Outside The Package

The following large files/directories were kept because they are useful for
future reruns and are expensive or inconvenient to recreate locally:

- `public_data/CXIDB_ID_57/normal-incidence-pattern.cxi`
- `public_data/CXIDB_ID_57/new.geom`
- `public_data/CXIDB_ID_57/README`
- `public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240/`
- `public_data/CXIDB_ID_57/rotation_libraries/`
- `validation/*.py`
- small validation notes/checklists in `validation/`

## Removed Categories

- Older ID57 synthetic libraries and v1 retrieval outputs.
- ID57 radial-control synthetic NPZ/output directories after their summaries
  were copied into the package.
- Full per-shot ID57 retrieval PNG directories after selected comparison images,
  CSVs, and JSON summaries were copied into the package.
- GroEL VTP mesh, template HDF5, smoke-test, excluded-shot, and quick-test
  output directories after summaries/figures/CSVs were copied into the package.
- Duplicated section 3 and section 4 result directories after they were copied
  into the package.

## Current Intended Use

- Use `validation_supporting_package/` for manuscript/supporting-material
  drafting and figure/result lookup.
- Use retained `validation/*.py` scripts plus the retained ID57 raw/synthetic
  data if reruns are needed.
- Recreate GroEL heavy intermediates only if raw CXIDB ID 187 hit data is
  reintroduced from the external storage.
