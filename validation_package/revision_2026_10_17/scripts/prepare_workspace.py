"""Create a separate research-layout copy; never overwrite an existing target."""
from pathlib import Path
import argparse, hashlib, json, shutil

PACKAGE=Path(__file__).resolve().parents[1]
REPO=PACKAGE.parents[1]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--portable',action='store_true',help='Apply documented environment and absent-DOCX provenance guards in the new copy only')
    parser.add_argument('--geometry-only',action='store_true',help='Use exact compact mask/rotations for complex-validation commands only; requires --portable')
    a=parser.parse_args();dest=a.destination.resolve()
    if a.geometry_only and not a.portable:parser.error('--geometry-only requires --portable')
    if dest.exists():parser.error('Destination already exists; choose a NEW directory.')
    if PACKAGE==dest or PACKAGE in dest.parents:parser.error('Choose a directory outside the evidence package.')
    dest.mkdir(parents=True)
    shutil.copytree(PACKAGE/'scripts/source_tree',dest,dirs_exist_ok=True)
    shutil.copytree(PACKAGE/'data/output',dest/'output',dirs_exist_ok=True)
    shutil.copytree(PACKAGE/'data/inputs',dest,dirs_exist_ok=True)
    # Historical public evidence is needed by research-layout source paths.
    for name in ['section3_public_synthetic','section4_performance','section5_id57','section5_groel']:
        shutil.copytree(REPO/'validation_package'/name,dest/'validation_supporting_package'/name,dirs_exist_ok=True)
    for p in (REPO/'validation_package/scripts').glob('*.py'):
        target=dest/'validation'/p.name
        if not target.exists():shutil.copyfile(p,target)
    changes=[]
    if a.portable:
        path=dest/'validation/revision_common.py';original=path.read_text(encoding='utf-8-sig')
        start=original.index('_DLL_HANDLES=[]');end=original.index('sys.path.insert(0,str(ROOT/"uploaded_SAXS_simulator"))')
        replacement='''# Public packaging adapter: honor installed CUDA, without changing numerics.
import site
_DLL_HANDLES=[]
if os.name == "nt":
    _cuda_roots=[ROOT/".venv_cxidb/Lib/site-packages/nvidia"]
    _cuda_roots += [Path(p)/"nvidia" for p in site.getsitepackages()]
    _cuda_dirs=[]
    for _cuda_base in _cuda_roots:
        if (_cuda_base/"cuda_runtime").exists():
            os.environ.setdefault("CUDA_PATH",str(_cuda_base/"cuda_runtime"))
        _cuda_dirs.extend(sorted(_cuda_base.glob("*/bin")))
    _cuda_dirs += [Path(p) for p in os.environ.get("SAXS_CUDA_DLL_DIRS","").split(os.pathsep) if p]
    for p in dict.fromkeys(_cuda_dirs):
        if p.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(p)))
            os.environ["PATH"]=str(p)+os.pathsep+os.environ.get("PATH","")
'''
        before=digest(path);path.write_text(original[:start]+replacement+original[end:],encoding='utf-8',newline='\n')
        changes.append(dict(path=path.relative_to(dest).as_posix(),before_sha256=before,after_sha256=digest(path),reason='Environment discovery only; use installed CUDA and guard Windows DLL setup.'))
        # Two old experiments guarded a local unpublished manuscript in their
        # preservation-only hash lists. Omit only absent DOCX paths in this copy;
        # all numerical/data/source inputs continue to be required and hashed.
        for name in ['revision_beam_truncation.py','revision_si_penalty_selection.py']:
            path=dest/'validation'/name;before=digest(path);s=path.read_text(encoding='utf-8-sig')
            if name=='revision_beam_truncation.py':
                old='hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}'
                new='hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths if p.suffix != ".docx" or p.exists()}'
            else:
                old='return {str(p.relative_to(ROOT)): sha(p) for p in paths}'
                new='return {str(p.relative_to(ROOT)): sha(p) for p in paths if p.suffix != ".docx" or p.exists()}'
            assert old in s,name
            path.write_text(s.replace(old,new,1),encoding='utf-8',newline='\n')
            changes.append(dict(path=path.relative_to(dest).as_posix(),before_sha256=before,after_sha256=digest(path),reason='Skip an absent unpublished DOCX in preservation-only hash list; retain all scientific input checks.'))
        if a.geometry_only:
            path=dest/'validation/revision_complex_projection_validation.py';before=digest(path)
            s=path.read_text(encoding='utf-8-sig')
            s=s.replace("folder/'synthetic_id57_patterns.npz'", "ROOT/'id57_geometry_fixture.npz'")
            s=s.replace("ROOT/'public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240/synthetic_id57_patterns.npz'", "ROOT/'id57_geometry_fixture.npz'")
            path.write_text(s,encoding='utf-8',newline='\n')
            changes.append(dict(path=path.relative_to(dest).as_posix(),before_sha256=before,after_sha256=digest(path),reason='Use compact exact mask and rotation arrays in complex-validation inputs and provenance, instead of the omitted full pattern archive.'))
    (dest/'PUBLIC_PREPARATION.json').write_text(json.dumps(dict(mode='portable' if a.portable else 'exact_source_snapshot',changes=changes,scope='Workspace materialization only; no simulation or fitting has been run.'),indent=2)+'\n',encoding='utf-8')
    print(dest)

if __name__=='__main__':main()
