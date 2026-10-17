"""Materialize the base plus R19 supplement into a new research workspace."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
PACKAGE=Path(__file__).resolve().parents[1]
REPO=PACKAGE.parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('--portable',action='store_true');p.add_argument('--fresh-groel',action='store_true');a=p.parse_args()
 dest=a.destination.resolve()
 if dest.exists():p.error('Destination already exists; choose a NEW directory.')
 if REPO==dest or REPO in dest.parents:p.error('Choose a destination outside the repository.')
 cmd=[sys.executable,str(REPO/'validation_package/revision_2026_10_17/scripts/prepare_workspace.py'),str(dest)]
 if a.portable:cmd.append('--portable')
 subprocess.run(cmd,check=True)
 # Base legacy sections can contain LF-normalized copies of exact input CSVs.
 # Restore explicitly preserved input bytes after the legacy-section copy.
 shutil.copytree(REPO/'validation_package/revision_2026_10_17/data/inputs',dest,dirs_exist_ok=True)
 shutil.copytree(PACKAGE/'scripts/source_tree',dest,dirs_exist_ok=True)
 shutil.copytree(PACKAGE/'data/output',dest/'output',dirs_exist_ok=True)
 if a.fresh_groel:
  backup=dest/'retained_R19_evidence';backup.mkdir()
  for name in ['groel_mesh_czt_matched_2026-10-17','groel_czt_primary_2026-10-17']:
   shutil.move(str(dest/'output'/name),str(backup/name))
 (dest/'R19_PREPARATION.json').write_text(json.dumps(dict(revision='R19',portable=a.portable,fresh_groel=a.fresh_groel,scope='Materialization only; no simulation or fitting run.'),indent=2)+'\n',encoding='utf-8')
 print(dest)
if __name__=='__main__':main()
