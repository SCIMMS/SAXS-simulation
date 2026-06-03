# GroEL Modeling Validation Note

## Purpose

This GroEL case should be used as a protein-scale modeling and forward-simulation
demonstration, not as a stringent orientation-retrieval benchmark. The retrieved
single-shot patterns are dominated by low-q radial-envelope intensity, and a
radial-profile-preserving randomized control gives nearly identical correlation
scores. The useful validation claim is therefore computational and modeling
oriented: realistic protein electron-density-derived meshes can be generated,
assembled into GroEL models, and simulated rapidly enough for public-data
comparison.

## Electron-density construction

- Input structure: `1SS8.pdb`
- Atoms used: non-hydrogen only
- Atom count: `27064`
- Elements: `C=16765`, `N=4634`, `O=5525`, `S=140`
- Density model: Waasmaier-Kirfel-inspired Gaussian electron-density
  approximation
- Grid spacing: `1.0 A`
- Smoothing sigma: `6.0 A`
- Sigma clipping: `3.0 sigma`
- Grid shape: `188 x 186 x 120`
- Grid voxels: `4,196,160`
- Density storage: `16.0 MB` as float32
- Density construction time: `51.8 s`
- Expected electrons: `179402.484`
- Grid-integrated electrons: `179402.327`

This supports the claim that a full protein-scale non-hydrogen density model can
be constructed within a practical runtime and memory footprint.

## Isosurface and mesh approximation

Five isosurface levels were generated at `10%`, `30%`, `50%`, `70%`, and `90%`
of the density maximum. VTK Flying Edges handled the disconnected high-threshold
objects: at the `90%` threshold the heptamer mesh had `15` connected components
and the tetradecamer mesh had `30` connected components.

The midpoint layer-weight approximation gave a strong z-projection match to the
reference density:

- `voxel_midpoint`: Pearson `0.999291`, cosine `0.999526`,
  relative total error `-2.445%`, relative RMSE `3.201%`
- `mesh_stencil_original_midpoint`: Pearson `0.999291`, cosine `0.999526`,
  relative total error `-2.445%`, relative RMSE `3.201%`
- `mesh_stencil_quadric_r80_midpoint`: Pearson `0.999289`, cosine `0.999524`,
  relative total error `-2.493%`, relative RMSE `3.219%`

This indicates that the mesh/stencil representation and the 80% quadric
decimation preserve the projection-level density approximation very well.

## GroEL assembly

Two model families were assembled by mesh reuse rather than reconstructing
density:

- Model A: heptamer, `C7`
- Model B: tetradecamer, `D7`
- Model B was assembled by appending the BIOMT copy with transform
  `(x, -y, -z)`.

Representative tetradecamer mesh sizes:

- `thr050_original`: `194460` points, `388976` cells, volume
  `1,067,555.95 A^3`
- `thr050_quadric_r80`: `38868` points, `77792` cells, volume
  `1,067,383.05 A^3`

The `thr050` decimated tetradecamer keeps the volume within about `0.016%` of
the original mesh while reducing the mesh to about `20%` of the original points
and cells.

## Template generation

The current GroEL template library used:

- Model variants: dry tetradecamer, `R35` polyprism, `R55` polyprism
- Symmetry-aware sampling: `D7`
- Candidate rotations before symmetry thinning: `201`
- Retained representative rotations: `143`
- Total templates: `429`
- Detector image size: `128 x 128`
- Annular ROI: `5-40 px`
- Projection pixel size: `1.0 A`
- FFT size: `2048`

This is the main positive result for GroEL: a realistic protein-derived mesh
model can be taken through the forward-simulation/template-generation pipeline
at a useful scale.

## Retrieval/control interpretation

For the manually curated 76 GroEL shots:

- Real template median logcorr: `0.610956`
- Radial-profile random control median logcorr: `0.605896`
- Median delta logcorr: `+0.00325`
- Real template better by logcorr: `61/76`
- Real template median fit RMS: `1.8573`
- Random control median fit RMS: `1.8674`
- Real template better by RMS: `70/76`

The real library is consistently but only slightly better than the radial
control. This means the GroEL shots should not be presented as strong
orientation-retrieval validation. They are better framed as a limitation-aware
public-data demonstration: the experimental shots are dominated by central
low-q envelope information, while the real protein-derived templates provide a
small additional model-specific improvement.

## Suggested manuscript position

Use ID57 as the retrieval-oriented validation case. Use GroEL as the
protein-density modeling and forward-simulation scalability case. The GroEL
radial-control result should be included as a limitation check to avoid
overstating orientation specificity.
