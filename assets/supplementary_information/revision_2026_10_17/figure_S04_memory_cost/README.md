# Figure S4: memory cost

Memory-cost comparison between dense three-dimensional arrays and the projected-density representation. Dense float32 and complex64 voxel volumes scale as O(N³), whereas projected float32 and complex64 detector-plane arrays scale as O(N²). Memory is reported in MiB; legacy source CSV columns ending in _MB were calculated by dividing bytes by 1024². Estimated mesh-array costs for representative vertex and face counts are included to show that mesh storage remains small compared with dense three-dimensional voxel allocation at the tested grid sizes.

## Source data

- [assets/supplementary_information/figure_S08_memory_cost_comparison](../../figure_S08_memory_cost_comparison)

The PNG is the exact embedded image in the R18 manuscript/SI. The source index uses repository-relative paths. Numerical records are canonical in the validation package; they are linked rather than duplicated.
