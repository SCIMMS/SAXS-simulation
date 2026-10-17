"""Redraw main Figure 5 from frozen public inputs (Python, NumPy, Matplotlib)."""
from pathlib import Path
import importlib.util
import sys
sys.dont_write_bytecode = True

helper = Path(__file__).resolve().parents[2] / "figure_01_workflow" / "data" / "regenerate_main_figures.py"
spec = importlib.util.spec_from_file_location("main_figure_plotting", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if __name__ == "__main__":
    module.main(default_figure=5)

