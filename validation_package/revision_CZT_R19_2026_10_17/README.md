# R19: consistent mesh/CZT primary comparisons

17 October 2026. This supplement updates the current main-text Figure 6 and SI Figures S12–S14 / Tables S12–S13. Primary ID57 results retain the R18 mesh/CZT evidence; primary GroEL results now use signed mesh projection and selected-window CZT. The previous FFT route is retained as a numerical control in SI S5.3. Exploratory direct-boundary GroEL refinement remains separately labelled in S6.

## Obtain the reproduction package

Clone or download the complete repository. Keep the [R18 base](../revision_2026_10_17/README.md), the original `validation_package/section*` folders and `validation_package/scripts/`, and this R19 supplement together in their repository-relative locations. The base and historical inputs are required by the materializer; this is a supplement, not a standalone replacement for the entire repository.

The optional supplement archive is [`revision_CZT_R19_reproduction_supplement_2026-10-17.zip`](downloads/revision_CZT_R19_reproduction_supplement_2026-10-17.zip). It contains this versioned supplement and the R19 main/SI asset folders, with repository-relative member paths. Extract it at the root of a full repository download. It does **not** contain the R18 base or the original section folders. Archive SHA256, size and member-count records are in [archive.json](provenance/archive.json). These filenames belong in the README, not in the article's Data availability statement.

Earlier local working archives `revision_evidence_R12.zip` and `revision_evidence_R13_supplement.zip` were editorial evidence bundles. They are not required for public reproduction and are not distributed here. Their public scientific content is represented by the versioned repository packages; do not search for those private working archives as dependencies.

## Scientific scope and changed control

- The 76 historical B/R35/R55 class/view/alignment assignments are frozen; only nonnegative signal and constant background are fitted. All 34 distinct assigned states are included.
- The current radial control permutes each assigned mesh/CZT template within one-pixel radial bins and shares its class/view/alignment. It replaces the earlier independently selected FFT-control library protocol. Changes in the control gap cannot be attributed to CZT accuracy or performance.
- Full-ROI median diagnostic log-correlations are 0.610636808 / 0.595620716; paired median difference is 0.005891568, positive in 74/76 shots. This paired median is not the difference between the two marginal medians.
- The separate A/B comparison reselects the lowest training RMS across the same 201 quaternion views per assembly, with historical alignment fixed. A/B lower-RMS counts are 38/38 on full ROI and 28/48 in both withheld-band directions.
- These are planar q_z=0, K=1 GroEL calculations. They do not establish GroEL Ewald accuracy, classification accuracy or assembly identity. Historical full-ROI selection makes within-image holdouts conditional.
- Changing only FFT to CZT on the same projection produces maximum relative intensity L2 differences below 1.6e-7. Point/FFT versus mesh/CZT also changes the density representation: mean full-ROI RMS changes by +0.0654%; there is no systematic fit gain attributable to CZT alone.

## Inspect or rerun

```sh
python validation_package/revision_CZT_R19_2026_10_17/scripts/verify_package.py
python validation_package/revision_CZT_R19_2026_10_17/scripts/verify_package.py --numerical
python validation_package/revision_CZT_R19_2026_10_17/scripts/prepare_workspace.py NEW_WORKSPACE --portable
```

See [REPRODUCING.md](REPRODUCING.md), [DATA_DICTIONARY.md](DATA_DICTIONARY.md) and [commands.json](commands.json). The numerical audit requires NumPy; scientific reruns require the compatible CUDA environment. Hash verification and saved-record audit do not rerun the GPU simulations. Input sources include CXIDB ID57/ID187 and PDB 1SS8. The exact processed GroEL cohort, density grids, meshes and historical orientation/assignment records used here are already present in the base repository package.

The [main assets](../../assets/main_manuscript/revision_CZT_R19_2026_10_17/README.md) and [SI assets](../../assets/supplementary_information/revision_CZT_R19_2026_10_17/README.md) follow the established per-figure images/data/README and editable-table CSV layout. Unchanged evidence links to the immutable R18 base. `MANIFEST.json` gives canonical published SHA256 checksums; source provenance records distinguish original source bytes from any normalized local path labels. Historical nested completion manifests retain their original-run hash scope.

The R18 prose description of the supplementary GroEL 2-pixel band origin was incorrect. R19 defines it as floor(r/2), globally from radius zero, then restricted to the 5–40 pixel ROI. Stored masks and scientific refinement results are unchanged. This differs from the primary 5-pixel split starting at radius 5.

Code: BSD 3-Clause. Author-created data, figures and documentation: CC BY 4.0. Original upstream inputs retain their source terms; see the repository licenses. Manuscripts, reviewer correspondence, raw CXI files and unrelated development files are excluded.
