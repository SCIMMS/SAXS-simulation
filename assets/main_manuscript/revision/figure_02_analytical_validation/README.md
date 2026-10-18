# Figure 2

Planar numerical validation for the unit-radius sphere (a–c), rotated ellipsoid (d–f) and rectangular box (g–i). Columns show mesh/CZT, analytical reference and signed log residual. Transverse qx and qy span −25 to 25 ℓ⁻¹ on a 257×257 grid at qz=0; ℓ equals the sphere radius. Ellipsoid semi-axes are (1.0, 0.65, 0.42)ℓ and box half-lengths are (1.0, 0.72, 0.45)ℓ. Colors show log₁₀(I/I₀+10⁻⁸), with the analytical zero-frequency intensity I₀ shared within each row. Residuals subtract reference from mesh/CZT log intensity; symmetric row-specific limits are labelled without clipping. Relative L₂ errors use the original unnormalized arrays. Smooth rows include mesh faceting; the box tests remaining numerical errors, separately from physical Ewald curvature.

## Files and resolution

- `images/figure_02.png`: exact current embedded figure, 3,900 × 3,713 pixels.
- Effective resolution at the article drawing size: 622.0 dpi or higher.
- `images/figure_02.svg`: vector/hybrid source.

Recreate this figure from the recorded scientific inputs with `python data/redraw.py --output-dir PATH`. The [current reproduction guide](../../../../validation_package/revision/REPRODUCING.md) describes the full six-figure command and environment.

## Scientific inputs

- [assets/main_manuscript/figure_02_analytical_validation](../../../../assets/main_manuscript/figure_02_analytical_validation)

`data/source_index.csv` uses repository-relative paths to retained numerical evidence. Scientific values are unchanged.
