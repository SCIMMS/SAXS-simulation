# Revision evidence map

## Main tables

- Table 1: analysis-specific forward calculation, fitting and selection,
  evaluation and fixed conditions. Editable text is in the current asset's
  `data/table.csv`. It summarizes methods and is not a new numerical experiment.
- Table 2: the retained 1,000-view fixed-morphology benchmark. Its numerical
  values are unchanged from the scientific evidence snapshot.

## Main comparisons

- Figure 2: planar numerical checks against analytical sphere, ellipsoid and
  box references. Normalized log intensity and signed log residual are display
  quantities; relative L2 errors use the unnormalized arrays.
- Figure 3 and SI S2.7–S2.8: independent Ewald-reference checks and the measured
  extra cost of finite longitudinal-frequency sampling.
- Figure 4 and SI S4.1: distinct radial-control, physical ellipsoid and curved
  sharp/rounded comparisons. Their absolute scores are not a single shared
  ranking. Paired differences use the same frame and evaluation pixels.
- Figure 5: selected ID57 predictions and projected densities. The complete
  aggregate analysis retains all 209 frames.
- Figure 6 and SI S5.1–S5.2: planar mesh/CZT GroEL examples, same-assignment
  radial controls and separate A/B comparisons. SI S5.3 compares numerical paths.

## Evaluation definitions

For ID57 physical-model comparisons, nuisance fitting uses nonnegative signal
and constant background in linear intensity. Model selection uses training
log-correlation. Alternating 10-pixel radial bands provide training/evaluation
pixels. Geometry and fitted coefficients remain fixed on evaluation bands.
Pearson correlation is centered and scaled on each evaluation pixel set.

The 187/209 result averages rounded-minus-variable-size-sharp test correlation
differences over both complementary split directions within each frame before
counting positive frames. It does not count independent particle shots in the
two folds and is not a physical-radius confidence interval.

GroEL primary nuisance fitting minimizes linear-intensity RMS, with
log-correlation as a diagnostic. Primary folds use 5-pixel bands starting at
radius 5. Historical class/view/alignment are fixed in the primary assigned
comparison; A/B selects a view separately within each assembly. Historical
full-ROI alignment remains fixed in A/B holdouts.

## Supplementary scope

S6 contains optional geometric refinement, penalty definitions and validation
controls. It remains distinct from the primary forward-calculation contribution.
S7 contains separate planar illumination controls; they demonstrate possible
mechanisms without identifying the origin of a measured streak.

Per-item source indexes and the dependency manifests identify numerical CSV,
JSON and NPZ records. Source coordinates, masks, fitting metrics and uncertainty
limitations are documented in the retained scientific dictionaries and current
figure captions.
