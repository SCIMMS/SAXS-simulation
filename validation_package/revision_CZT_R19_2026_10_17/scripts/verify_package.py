"""CPU integrity and saved-result audit; does not rerun GPU simulations."""
from pathlib import Path
from zipfile import ZipFile
import argparse,csv,hashlib,json,subprocess,sys
PACKAGE=Path(__file__).resolve().parents[1]
REPO=PACKAGE.parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=2e-12):assert abs(float(a)-float(b))<=tol,(a,b)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--numerical',action='store_true');a=parser.parse_args()
 manifest=load(PACKAGE/'MANIFEST.json');base=REPO/manifest['base_dependency']['manifest_path']
 assert sha(base)==manifest['base_dependency']['manifest_sha256'],'Base dependency manifest changed'
 for item in manifest['files']:
  p=(REPO/item['path']).resolve();assert REPO in p.parents
  assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
 subprocess.run([sys.executable,str(base.parent/'scripts/verify_package.py')],check=True)
 for item in load(PACKAGE/'provenance/inherited_imports.json'):
  assert sha(REPO/item['repository_path'])==item['source_sha256'],item['module']
 archive_status='Optional transport absent; canonical files verified'
 metadata=PACKAGE/'provenance/archive.json'
 if metadata.exists():
  archive=load(metadata);zpath=PACKAGE/'downloads'/archive['filename']
  if zpath.exists():
   assert sha(zpath)==archive['sha256'] and zpath.stat().st_size==archive['bytes']
   expected={i['path']:i['sha256'] for i in manifest['files']}
   with ZipFile(zpath) as z:
    assert z.testzip() is None and len(z.namelist())==archive['members']
    assert set(z.namelist())==set(expected)|{manifest['self_path']}
    for name in z.namelist():assert z.read(name)==(REPO/name).read_bytes(),name
   archive_status=f"Verified {archive['members']} archive members"
 result=dict(status='pass',revision='R19',files=len(manifest['files']),archive=archive_status,base_dependency='verified',numerical=False)
 if a.numerical:
  import numpy as np
  d=PACKAGE/'data/output/groel_czt_primary_2026-10-17'
  fixed=rows(d/'fixed_control_results.csv');paired=rows(d/'fixed_control_paired.csv');ab=rows(d/'ab_paired_results.csv')
  assert len(fixed)==len(ab)==456 and len(paired)==228
  fs=load(d/'fixed_control_summary.json')['results'];bs=load(d/'ab_summary.json')['results']
  for fold in ['full','even_to_odd','odd_to_even']:
   f={(int(r['pattern_index']),r['family']):r for r in fixed if r['fold']==fold}
   b={(int(r['pattern_index']),r['model']):r for r in ab if r['fold']==fold}
   assert len(f)==len(b)==152
   dc=np.array([float(f[i,'structured']['test_logcorr'])-float(f[i,'radial_control']['test_logcorr']) for i in range(76)])
   dr=np.array([float(f[i,'structured']['test_rms'])-float(f[i,'radial_control']['test_rms']) for i in range(76)])
   close(np.median(dc),fs[fold]['paired_median_logcorr']);close(np.mean(dc),fs[fold]['paired_mean_logcorr'])
   assert int((dc>0).sum())==fs[fold]['structured_higher_logcorr']
   assert int((dr<0).sum())==fs[fold]['structured_lower_RMS']
   for family,key in [('structured','structured'),('radial_control','control')]:
    close(np.median([float(f[i,family]['test_logcorr']) for i in range(76)]),fs[fold][key+'_median_logcorr'])
    close(np.median([float(f[i,family]['test_rms']) for i in range(76)]),fs[fold][key+'_median_RMS'])
   ba=np.array([float(b[i,'A']['test_rms'])-float(b[i,'B']['test_rms']) for i in range(76)])
   bc=np.array([float(b[i,'A']['test_logcorr'])-float(b[i,'B']['test_logcorr']) for i in range(76)])
   assert int((ba<0).sum())==bs[fold]['A_lower_RMS'] and int((ba>0).sum())==bs[fold]['B_lower_RMS']
   close(np.median(ba),bs[fold]['median_A_minus_B_RMS']);close(np.median(bc),bs[fold]['median_A_minus_B_logcorr'])
   for r in [r for r in paired if r['fold']==fold]:
    i=int(r['pattern_index']);close(r['paired_delta'],dc[i]);close(r['RMS_delta'],dr[i])
  for filename,families,source in [('pattern67_primary.npz',['structured','radial_control'],fixed),('pattern67_AB.npz',['A','B'],ab)]:
   with np.load(d/filename) as z:
    y=z['measured'][z['roi']]
    for family in families:
     p=z[family][z['roi']];record=next(r for r in source if int(r['pattern_index'])==67 and r['fold']=='full' and r.get('family',r.get('model'))==family)
     close(np.sqrt(np.mean((p-y)**2)),record['test_rms'])
     close(np.corrcoef(np.log1p(p),np.log1p(y))[0,1],record['test_logcorr'])
  with np.load(d/'ab_templates_mesh_czt.npz') as z:
   arrays={k:z[k] for k in z.files}
   assert any(v.shape==(2,201,128,128) for v in arrays.values())
  m=PACKAGE/'data/output/groel_mesh_czt_matched_2026-10-17';qa=rows(m/'template_qa.csv')
  assert len(qa)==34 and len(rows(m/'fit_metrics.csv'))==1216
  assert max(float(r['point_CZT_FFT_relative_L2']) for r in qa)<1.6e-7
  assert max(float(r['mesh_CZT_FFT_relative_L2']) for r in qa)<1.6e-7
  result.update(numerical=True,fixed_control_records=456,AB_records=456,matched_path_records=1216,unique_assigned_states=34,scope='Saved-record aggregate and example checks; GPU forward/reference runs are recorded separately in the scientific audits.')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
