# Definitions used in this release

## Comparison scores and aggregation

**ROI** is the valid detector-pixel region after masking. ID57's primary comparison uses a radial annulus of 45–190 detector pixels. Its nominal planar transverse q range is about 0.267–1.128 nm⁻¹. Physical detector wavevectors are used for the five-plane Ewald calculation.

**Log-correlation** is the Pearson correlation between `log1p` measured intensity and `log1p` the signal/background-adjusted model on the stated pixels; higher is better. It is distinct from the least-squares loss used for supplementary refinement. Historical GroEL log-correlation is a diagnostic at the RMS-selected candidate.

**Paired mean difference** first subtracts the two models' scores for the same frame and evaluation fold, then averages these differences with equal frame weights. It is not a percent improvement. **Median paired difference** takes the median of these within-frame differences, which generally differs from subtracting the two model medians.

**Withheld-band mean** is the equal-frame mean of the scores (or explicitly named paired differences) evaluated on detector radial bands withheld from the corresponding training fit. In the ID57 two-fold library comparison, 10-pixel bands start at band 0 at radius 45 pixels. Even bands train and odd bands test, then the roles reverse. The geometry, orientation, signal and background selected/fitted on training pixels stay fixed for evaluation. The two-fold paired statistic averages the two test differences per frame before summarizing all 209 frames. Pixels in the same image are not independent particles or independent repeated measurements.

`selection_results.csv` identifies `fold`, `frame`, `family`, selected `candidate`, geometric parameters, `fit_logcorr` (training) and `test_logcorr` (evaluation). `full` fits and evaluates the same ROI; `even_to_odd` and `odd_to_even` are the complementary held-out folds. `paired_twofold_deltas.csv` gives one two-fold paired comparison per frame. `scores_<fold>.npy` stores float64 training scores with shape `(209, 8400)`, indexed by the rows of `candidate_parameters.csv`. The saved CPU-finalist stage checks the top eight and any near-tied candidates with the independent CPU fit calculation.

The three five-plane families share one candidate bank: fixed sharp has 240 orientations at scale 1; variable-size sharp has 240 × 5 scales; rounded has 240 × 5 × 7 radii. The radii include zero, so the rounded bank contains the sharp subfamilies. All 209 frames and negative held-out differences are retained.

## Complex projection and numerical error

**K** is the number of sampled longitudinal Fourier frequencies κ (q_z planes) per prescribed orientation. It is not the number of physical depth slices. Each complex projection integrates density with the factor `exp(−i κ z)`. K two-dimensional CZTs evaluate the selected transverse q window. Interpolation acts on complex amplitudes on the physical Ewald surface before modulus squared.

**Relative L2 error** is `norm(prediction − reference) / norm(reference)` on the stated amplitude or intensity samples. A percent column multiplies this ratio by 100. Averaging/maximizing over orientations, and sparse distributed-ROI versus full-ROI reference coverage, are separate reporting choices. K=5 accuracy is geometry, thickness, grid and q-range specific. It does not change the internal resolution of a coarse density boundary model.

**Memory** uses MiB = 2²⁰ bytes. Original historical CSV column names ending in `_MB` may actually use this binary divisor; R18 Figure S4 and its caption correctly display MiB. Live allocation, allocator-pool high-water and cached-volume storage measure different scopes. **Warm per-view timing** includes host rotation, geometry transfer and intensity return, while excluding setup, compilation and file I/O. The cached-volume 1,000-view benchmark includes its setup and uses different grids/precision. The results do not establish a universal speed ranking.

## Supplementary refinement and regularization

For actual data, the residual on N training pixels is

`r_i = (log(1 + mu_i(theta)) − log(1 + y_i)) / sqrt(N)`.

The appended penalty residual is `sqrt(lambda) * L * theta`; the least-squares solver minimizes half the squared norm of the combined residual, equivalently half of `mean(log1p residual²) + lambda * ||L theta||²`. Intensities are clipped nonnegative in the recorded dataset units. Signal and background are fitted. Test `log_mse` is the unpenalized mean squared log1p residual on full-resolution test pixels; lower is better. A cohort loss reduction is `100 * (mean(seed loss) − mean(fitted loss)) / mean(seed loss)`, not the mean of per-frame percentages.

- Planar ID57: 13 parameters; L has weights `(20,20,45,45)` on cos4/sin4/cos6/sin6 angular coefficients, and zero weights on scale, aspect, radius, rotation and nuisance variables. The selected common λ is 0.1. Ten-pixel bands start at 45 pixels: even bands train; bands 1 mod 4 validate; bands 3 mod 4 test.
- GroEL: four geometric-mode weights `(20,20,40,20)`. Prism families additionally penalize log prism-radius and log prism-length ratios with weight 10 each; these columns are zero for the density-only family. Scale/rotation/beam-offset/fluence/background columns are unpenalized. The validation-selected primary λ is 0; the separately labeled best strictly positive control uses 10⁻⁴. Two-pixel bands start at radius 5 pixels with the same even/1-mod-4/3-mod-4 division.
- Five-plane angular-prior controls use λ=0.1 with the angular-mode penalty specified in `revision_qz5_radius_controls.py`. No aspect or physical-radius confidence penalty is implied. Nested aspect ranges are prescribed comparisons, not test-selected ranges.
- Synthetic tests use Poisson deviance and a separately validation-selected penalty. They do not use the measured-data loss or imply a calibrated Poisson likelihood for processed GroEL counts.

**Effective radius** in the aspect sensitivity study is the mean of sphere-fit radii over the eight transformed corner caps of one fitted mesh. It differs from the pre-transform base radius and is not a unique local curvature. `aspect_bound`, `scale_bound`, `success` and other diagnostic columns are retained. The nested aspect ranges are `[0.9,1.1]`, `[0.8,1.25]` and `[2/3,1.5]` for the stated pre-rotation x/z and y/z scale ratios.

**GroEL classes** B/R35/R55 are the historical tetradecamer density model and its two phenomenological polyprism variants; R35 and R55 denote nominal prism radii 35 and 55 Å. Cross-class refinement compares those families, not the A heptamer. The separate matched A/B study uses 201 views per assembly before D7 reduction, historical fixed integer alignment and RMS candidate selection. Its 5-pixel radial-band folds differ from the refinement study's 2-pixel split.

## Illumination controls

The mask acts on incident **field amplitude** times the projected density before Fourier transformation. It is not an intensity-space detector mask. Sharp truncation can generate streaks; smoothing changes their concentration. The preserved tests are planar, with fixed geometry within each fold. They demonstrate a mechanism and sensitivity, without assigning the actual illumination condition or single-hit purity of a measured pattern.
