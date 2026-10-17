"""Check current revision assets and immutable saved scientific evidence."""
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZipFile
import argparse,ast,csv,hashlib,json,re,struct,subprocess,sys

PACKAGE=Path(__file__).resolve().parents[1]
REPO=PACKAGE.parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def safe(rel):
    p=(REPO/rel).resolve()
    assert REPO in p.parents,p
    return p
def run_audit(path,numerical=False):
    cmd=[sys.executable,str(path)]
    if numerical:cmd.append('--numerical')
    completed=subprocess.run(cmd,check=True,capture_output=True,text=True)
    return completed.stdout.strip()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--numerical',action='store_true')
    args=ap.parse_args();manifest=load(PACKAGE/'MANIFEST.json')
    assert manifest['edition']=='Revision'
    for entry in manifest['files']:
        p=safe(entry['path'])
        assert p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],entry['path']
        assert 'R20' not in p.name,p.name
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    deps=load(PACKAGE/'dependencies.json')['scientific_snapshots']
    for dep in deps:
        m=safe(dep['manifest_path']);assert sha(m)==dep['manifest_sha256'],dep['role']
    # The GroEL verifier also checks the exact base dependency and all its files.
    groel=next(d for d in deps if d['role']=='mesh_czt_groel_evidence')
    base=next(d for d in deps if d['role']=='base_evidence')
    retained_audits={'mesh_czt_groel_evidence':run_audit(safe(groel['manifest_path']).parent/'scripts/verify_package.py',args.numerical)}
    if args.numerical:
        retained_audits['base_evidence']=run_audit(safe(base['manifest_path']).parent/'scripts/verify_package.py',True)
    assets=[]
    for directory,total in [('assets/main_manuscript/revision',8),('assets/supplementary_information/revision',43)]:
        records=rows(REPO/directory/'asset_manifest.csv');assert len(records)==total,(directory,len(records))
        for row in records:
            folder=safe(row['repository_relative_folder']);assert (folder/'README.md').is_file()
            if 'Figure' in row['item']:
                p=list((folder/'images').glob('*.png'));assert len(p)==1,(folder,p)
                artifact=p[0]
            else:artifact=folder/'data/table.csv'
            assert sha(artifact)==row['image_or_table_sha256'],row['item']
            for source in row['source_paths'].split(';'):
                if source:assert safe(source).exists(),source
        assets+=records
    assert len(assets)==51
    resolution=load(PACKAGE/'provenance/figure_resolution.json')['figures']
    assert len(resolution)==28
    for row in resolution:
        p=safe(row['path']);dims=list(struct.unpack('>II',p.read_bytes()[16:24]))
        assert dims==row['pixels'] and sha(p)==row['sha256']
        assert min(dims[i]/row['drawing_inches'][i] for i in range(2))>=600
    readmes=[REPO/'README.md',REPO/'assets/main_manuscript/README.md',REPO/'assets/supplementary_information/README.md']
    readmes+=list(PACKAGE.glob('*.md'))
    for d in ['assets/main_manuscript/revision','assets/supplementary_information/revision']:readmes+=list((REPO/d).rglob('README.md'))
    links=0
    for readme in readmes:
        text=readme.read_text(encoding='utf-8-sig')
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            target=raw.strip().split(' "',1)[0].strip('<>')
            if re.match(r'^(?:https?://|mailto:|#)',target):continue
            rel=unquote(target.split('#',1)[0]);p=(readme.parent/rel).resolve()
            assert p.exists(),(readme.relative_to(REPO).as_posix(),target)
            links+=1
    archive_status='Optional archive absent; canonical files checked'
    metadata=PACKAGE/'provenance/archive.json'
    if metadata.exists():
        transport=load(metadata);archive=PACKAGE/'downloads'/transport['filename']
        assert archive.is_file() and sha(archive)==transport['sha256'] and archive.stat().st_size==transport['bytes']
        expected={e['path'] for e in manifest['files']}|{manifest['self_path']}
        with ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist())==len(expected)==transport['members']
            assert set(z.namelist())==expected
            for name in z.namelist():assert z.read(name)==safe(name).read_bytes(),name
        archive_status=f"Verified {transport['members']} archive members"
    result={'status':'pass','edition':'Revision','canonical_files':len(manifest['files']),'assets':len(assets),'main_figures':6,'main_tables':2,'si_figures':22,'si_tables':21,'relative_links_checked':links,'archive':archive_status,'scientific_dependencies':'verified','numerical':args.numerical}
    if args.numerical:
        import numpy as np
        bank=safe(base['manifest_path']).parent/'data/output/id57_qz5_rounding_2026-10-16'
        records=rows(bank/'selection_results.csv')
        lookup={(r['fold'],int(r['frame']),r['family']):float(r['test_logcorr']) for r in records}
        delta=np.array([np.mean([lookup[fold,i,'rounded']-lookup[fold,i,'variable_size_sharp'] for fold in ['even_to_odd','odd_to_even']]) for i in range(209)])
        assert int((delta>0).sum())==187
        result['within_frame_positive_count']={'positive':187,'frames':209,'two_complementary_test_directions_averaged':True}
    result['retained_audits']=retained_audits
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
