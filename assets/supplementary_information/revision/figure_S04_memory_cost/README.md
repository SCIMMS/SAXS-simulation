# Figure S4

Figure S4 Memory-cost comparison between dense three-dimensional arrays and the projected-density representation. Dense float32 and complex64 voxel volumes scale as O(N³), whereas projected float32 and complex64 detector-plane arrays scale as O(N²). Memory is reported in MiB; legacy source CSV columns ending in _MB were calculated by dividing bytes by 1024². Estimated mesh-array costs for representative vertex and face counts are included to show that mesh storage remains small compared with dense three-dimensional voxel allocation at the tested grid sizes.

## Files and resolution

- `images/figure_S04.png`: exact current embedded figure, 3,763 × 2,606 pixels.
- Effective resolution at the article drawing size: 600.3 dpi or higher.
- `images/figure_S04.svg`: vector/hybrid source.

The SVG preserves vector labels, axes and linework; detector/density image panels, when present, retain their original raster arrays. It can be rasterized at the required final figure size without enlarging the previous whole-figure PNG. Retained numerical evidence is indexed below; the [reproduction guide](../../../../validation_package/revision/REPRODUCING.md) describes scientific regeneration.

## Scientific inputs

- [assets/supplementary_information/figure_S08_memory_cost_comparison](../../../../assets/supplementary_information/figure_S08_memory_cost_comparison)

`data/source_index.csv` uses repository-relative paths to retained numerical evidence. Scientific values are unchanged.
