# Table S5

Table S5 Fixed-morphology cached-volume benchmark. Total time includes setup. *Coarse FFT fails the 1% target and was audited at three references; †CPU accuracy here uses three references. Cubic cache and GPU-CZT variants were audited over 240 orientations. MiB denotes 1,048,576 bytes. Cache is retained complex-volume storage; pool is peak memory reserved by the allocator. Worst ROI L₂ is the maximum relative L₂ intensity error over the audited cases. Pool high-water excludes total driver and GPU FFT-library peak memory; precision and geometry discretization differ.

`data/table.csv` transcribes the final R18 native Word table, preserving displayed rounding, units and footnote symbols. Unrounded measurements and conditions are in the [validation package](../../../../validation_package/revision_2026_10_17/README.md). Tables remain editable CSV; there is no redundant raster table preview.
