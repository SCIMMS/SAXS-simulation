"""Prepare a separate scientific workspace using the retained materializer."""
from pathlib import Path
import argparse,subprocess,sys

REPO=Path(__file__).resolve().parents[3]
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Materialize the retained revision evidence in a new scientific workspace.')
    parser.add_argument('destination',help='New directory; it must not already exist')
    parser.add_argument('--portable',action='store_true',help='Apply documented environment adapters in the new copy')
    parser.add_argument('--fresh-groel',action='store_true',help='Keep saved GroEL outputs aside for a new scientific rerun')
    args=parser.parse_args()
    source=REPO/'validation_package/revision_CZT_R19_2026_10_17/scripts/prepare_workspace.py'
    command=[sys.executable,str(source),args.destination]
    if args.portable:command.append('--portable')
    if args.fresh_groel:command.append('--fresh-groel')
    raise SystemExit(subprocess.call(command))
