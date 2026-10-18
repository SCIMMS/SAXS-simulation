# Figure S11

Figure S11 GroEL density-construction checks, distinct from the primary 12-layer calculation in S5.1. (a) Orthogonal mid-slices of the smoothed C7 density grid. (b) The C7 reference z projection and original/quadric-r80 mesh projections, using five thresholds at 10%, 30%, 50%, 70% and 90% of the density maximum with midpoint layer weights; residuals are mesh minus reference. (c) Retained A/B threshold-series point ratios, relative volume errors and connected-component counts. These construction checks assess density projection and decimation, rather than experimental structure recovery.

## Files and resolution

- `images/figure_S11.png`: exact current embedded figure, 3,600 × 4,296 pixels.
- Effective resolution at the article drawing size: 672.9 dpi or higher.
- `images/figure_S11.svg`: vector/hybrid source.

Replot the supplied scalar arrays and unchanged decimation CSV with `python data/redraw.py --output-dir PATH`. This plot-only command does not rerun VTK meshing. The retained scientific inputs and public construction scripts document the separate full reconstruction.

## Scientific inputs

- [assets/supplementary_information/figure_S02_groel_density_projection](../../../../assets/supplementary_information/figure_S02_groel_density_projection)

`data/source_index.csv` uses repository-relative paths to retained numerical evidence. Scientific values are unchanged.

## Construction and operational-model distinction

The supplied `data/projection_arrays.npz` contains the original C7 density mid-slices, direct density projection, original-mesh and quadric-r80 mesh-stencil projections, and threshold/weight arrays. The construction check uses five isodensity thresholds at 10%, 30%, 50%, 70% and 90% of the maximum density with midpoint weights. Its exact-reconstruction audit is in `data/reconstruction_audit.json`. These five-level stencil checks are separate from the operational twelve-layer mesh/CZT model used in the main GroEL comparison.

The compact redraw uses NumPy and Matplotlib. Full reconstruction additionally uses the retained original density grid, public mesh construction/projection scripts and VTK; it is not rerun by `data/redraw.py`.
