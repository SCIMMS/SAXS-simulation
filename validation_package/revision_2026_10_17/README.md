# Revision validation package — 17 October 2026

This is the scientific evidence release aligned with revision R18 of *Surface Mesh Projection for Low q Single Shot XFEL Diffraction*. It extends the historical section 3–5 package without replacing it. Manuscript drafts, referee reports and correspondence are excluded.

## Start here

- [Current main figures and table](../../assets/main_manuscript/revision_2026_10_17/README.md): 6 figures and 1 table.
- [Current Supporting Information figures and tables](../../assets/supplementary_information/revision_2026_10_17/README.md): 22 figures and 21 tables.
- [Definitions and interpretation](DATA_DICTIONARY.md).
- [Reproduction and input requirements](REPRODUCING.md).
- [Ordered scientific commands](commands.json).
- [File integrity manifest](MANIFEST.json), [source provenance](provenance/source_records.json), and [omitted arrays](provenance/omitted_files.csv).
- [Publication audit](provenance/publication_audit.json): saved-result checks and the 24 functional controls rerun in a separately prepared workspace.

Run the lightweight audit from the repository root:

```sh
python validation_package/revision_2026_10_17/scripts/verify_package.py
python validation_package/revision_2026_10_17/scripts/verify_package.py --numerical
```

The first command uses the Python standard library. The second also requires NumPy and checks complete saved score matrices, per-frame summaries and exact-reference comparisons. Neither reruns GPU simulations or fitting. A fresh scientific rerun is described separately.

## Contents

`data/output/` retains the dated study paths: corrected planar controls and physical baseline; Ewald and cached-volume checks; complex projections at K=3,5,9; the full 8,400-candidate five-plane reselection; boundary-reference audits; differentiable synthetic/ID57/GroEL examples; regularization and initialization controls; aspect-bound sensitivity; matched GroEL A/B comparisons; and illumination/truncation tests. Successful, worsening, incomplete and bound-limited fits are retained.

`data/inputs/` supplies compact exact metadata, rotations, the fixed 76-shot processed GroEL sample, its historical assignments and the regenerated density grid. Upstream ID57 CXI, GroEL raw/calibration files, full template libraries, and large intensity caches are excluded. The three complete 209 × 8,400 training-score matrices are included so candidate selection can be inspected without those caches.

`scripts/source_tree/` is the scientific source snapshot and its local dependency closure. It contains the forward kernel and validation helpers, not the simulator GUI. Original source bytes are preserved except for removal of editorial response-drafting text from two reporting helpers, recorded in provenance. The separate workspace preparer documents optional environment/provenance-only adapters; numerical algorithms are unchanged. Research scripts and saved checkpoints are not a cross-platform installer.

## Principal evidence and limits

For all 209 ID57 frames, the five-plane rounded family exceeds the variable-size sharp family by mean withheld-band correlation differences 0.04516109 and 0.04719056 in the two complementary folds. Their equal-frame, two-fold mean is 0.04617583; 187/209 frames have a positive paired mean. Continuous-boundary reevaluation of the 701 distinct selected geometries gives 0.04617775. This audits the selected fits, not an exact-reference reranking of all 8,400 candidates.

The geometry-specific K=5 checks extend the physical Ewald comparison to the original nominal q range, about 0.267–1.128 nm⁻¹. They do not establish arbitrary high-q validity or atomic resolution. Rounding preferences and effective radii are conditional on the prescribed density, orientation library, aspect bounds and illumination assumptions. They are not confidence intervals for unique physical corner radii.

Differentiation remains an exploratory Supporting Information extension with a short Discussion introduction. GroEL holdouts are conditional on historical full-ROI class/orientation/alignment selection. Matched A/B assembly comparisons are a separate experiment. Illumination tests show that beam edges can generate streaks at unchanged morphology; the cause of any particular experimental streak remains unassigned. Combined five-plane illumination fitting was not validated.

The historical public ID57 values 0.724752/0.443121 describe the earlier implementation. The corrected planar structured/control medians used in R18 are approximately 0.661/0.392, with median paired difference 0.261. Different forward/scoring conventions are explicitly labeled; their absolute scores are not pooled into one ranking.

## Licensing and upstream attribution

Repository code is BSD 3-Clause; original figures, tables, documentation and author-created data use CC BY 4.0 under the repository licenses. Public experimental and structural inputs retain their upstream terms and attribution: [CXIDB 57](https://www.cxidb.org/id-57.html), [CXIDB 187](https://www.cxidb.org/id-187.html), and [PDB 1SS8](https://www.rcsb.org/structure/1SS8). Processing does not relicense third-party observations. Use `CITATION.cff` for repository attribution and cite the source datasets and associated article as appropriate.
