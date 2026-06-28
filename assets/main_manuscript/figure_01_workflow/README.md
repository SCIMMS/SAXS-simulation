# Figure 1 - Computational workflow

Mesh-based graphical workflow for the projected-density simulation pipeline.

## Local data files
- `data/figure_01_workflow_steps.csv`

## Source references
- Generated from the manuscript visual-asset source at `manuscript_visual_assets/figure_01_computational_workflow/make_mesh_workflow_figure.py`.
- The schematic uses an ID57-style core-shell mesh with a Pd cube shell and Au octahedral core.

## Generated outputs
- `data\figure_01_workflow_steps.csv`
- `images\figure_01_workflow.png`
- `images\figure_01_workflow.pdf`

## Regeneration
Run the source script from the repository root:

```powershell
$env:MPLCONFIGDIR = ".runtime_cache\matplotlib"
.\.venv_cxidb\Scripts\python.exe manuscript_visual_assets\figure_01_computational_workflow\make_mesh_workflow_figure.py
```

Then copy `figure_01_workflow.png` and `figure_01_workflow.pdf` into this manuscript-order asset folder.
