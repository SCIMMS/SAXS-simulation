# Figure 3

Longitudinal-plane validation and cost. (a,b) Mean and maximum physical-Ewald intensity relative L₂ errors against continuous-boundary references over 240 views per mesh and 5,750 distributed ROI samples per view; full-ROI controls are in SI. (c) Warm synchronized GPU times for planar and K=3,5,9 paths on RTX 2070 SUPER at N=768, M=401. Per-view host rotation, geometry transfer and intensity return are included; setup, first-use compilation and file I/O are excluded. Additional planes improve longitudinal accuracy and increase cost, with a residual plateau consistent with transverse discretization error.

## Files and resolution

- `images/figure_03.png`: exact current embedded figure, 6,300 × 1,920 pixels.
- Effective resolution at the article drawing size: 1016.1 dpi or higher.
- `images/figure_03.svg`: vector/hybrid source.

Recreate this figure from the recorded scientific inputs with `python data/redraw.py --output-dir PATH`. The [current reproduction guide](../../../../validation_package/revision/REPRODUCING.md) describes the full six-figure command and environment.

Panel identifiers are placed above the axes, separated from all plotted marks and the shared legend.

## Scientific inputs

- [validation_package/revision_2026_10_17/data/output/complex_projection_2026-10-15](../../../../validation_package/revision_2026_10_17/data/output/complex_projection_2026-10-15)

`data/source_index.csv` uses repository-relative paths to retained numerical evidence. Scientific values are unchanged.
