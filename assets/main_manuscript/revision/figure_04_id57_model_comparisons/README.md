# Figure 4

ID57 model comparisons for all 209 frames. Each horizontal coordinate is a score difference: the model correlation score minus the comparator correlation score for the same frame and evaluation pixels. (a) Planar structured-minus-radial-control comparison on the full region of interest (ROI). (b) Planar fixed core-shell minus fitted ellipsoid on separate radial evaluation bands. (c) Five-plane physical-Ewald rounded minus variable-size sharp comparisons after regenerating and reselecting the complete candidate libraries. In (b,c), Fit even → evaluate odd means fitting on even-numbered radial bands and evaluating predictions on odd-numbered bands; the reverse direction exchanges these roles. These labels refer to band numbers, not frame numbers. Geometry, intensity scale and background are determined using fitting bands and kept fixed during evaluation. Positive differences favor the first model named in each panel. Panels show different matched comparisons; absolute scores across their calculation and scoring conventions are not treated as a common ranking. Complete results and direct parameter summaries are in Section S4.

## Files and resolution

- `images/figure_04.png`: exact current embedded figure, 3,900 × 1,300 pixels.
- Effective resolution at the article drawing size: 622.0 dpi or higher.
- `images/figure_04.svg`: vector/hybrid source.

Recreate this figure from the recorded scientific inputs with `python data/redraw.py --output-dir PATH`. The [current reproduction guide](../../../../validation_package/revision/REPRODUCING.md) describes the full six-figure command and environment.

## Scientific inputs

- [validation_package/revision_2026_10_17/data/output/revision_remaining_2026-10-09/01_consistency](../../../../validation_package/revision_2026_10_17/data/output/revision_remaining_2026-10-09/01_consistency)
- [validation_package/revision_2026_10_17/data/output/revision_remaining_2026-10-09/02_baseline](../../../../validation_package/revision_2026_10_17/data/output/revision_remaining_2026-10-09/02_baseline)
- [validation_package/revision_2026_10_17/data/output/id57_qz5_rounding_2026-10-16](../../../../validation_package/revision_2026_10_17/data/output/id57_qz5_rounding_2026-10-16)

`data/source_index.csv` uses repository-relative paths to retained numerical evidence. Scientific values are unchanged.
