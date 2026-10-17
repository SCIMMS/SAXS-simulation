"""Redraw current main figures from compact saved public plotting inputs."""
from pathlib import Path
import argparse,subprocess,sys

REPO=Path(__file__).resolve().parents[3]
FIGURES=['figure_01_workflow','figure_02_analytical_validation','figure_03_complex_projection_validation','figure_04_id57_model_comparisons','figure_05_id57_five_plane_examples','figure_06_groel_representative_match']
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,help='Directory for regenerated PNG/SVG figures')
    args=parser.parse_args();out=Path(args.output_dir).resolve()
    for folder in FIGURES:
        helper=REPO/'assets/main_manuscript/revision'/folder/'data/redraw.py'
        if not helper.is_file():raise FileNotFoundError(helper)
        subprocess.run([sys.executable,str(helper),'--output-dir',str(out)],check=True)
if __name__=='__main__':main()
