# Manuscript Visual Assets

Each table or figure has its own folder with `data/`, `images/`, and `README.md`.
The `data/` folders contain copied source CSV/JSON/NPZ/PNG files needed to redesign the figure or table.
The `images/` folders contain current manuscript-ready PNG previews or copied source figures.

Current main-manuscript set after text reduction:
- Figures 1-8 remain in the main manuscript.
- Main tables are now Table 1, Table 2, and Table 3.
- Former main Table 2 has been moved to Supplementary Table S3.
- Former main Table 5 has been moved to Supplementary Table S1.
- Former main Figure 9 has been moved to Supplementary Figure S4.

The old folder names are retained so the existing regeneration script and copied source data remain traceable.

Regenerate everything with:

`.venv_cxidb\Scripts\python.exe validation\organize_manuscript_visual_assets.py`
