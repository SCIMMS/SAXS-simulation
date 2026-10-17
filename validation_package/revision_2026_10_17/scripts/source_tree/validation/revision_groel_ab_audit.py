"""Independent reconstruction of every reported matched A/B metric."""
from revision_common import ROOT,np,dump,sha
import csv,json,h5py
from scipy.optimize import nnls
OUT=ROOT/'output/groel_ab_matched_2026-10-15'
with np.load(OUT/'templates.npz') as z:lib=z['templates'].astype(float);rot=z['rotation_indices']
with h5py.File(ROOT/'validation_supporting_package/section5_groel/curated_run151_hits/hits_r0151_curated76_ds8.h5') as h:
    images=h['patterns_ph'][:].astype(float);valid=h['mask'][:].astype(bool)
records=list(csv.DictReader((OUT/'paired_results.csv').open()))
summary=json.loads((OUT/'summary.json').read_text())
ygrid,xgrid=np.indices((128,128));radius=np.hypot(xgrid-63.5,ygrid-63.5)
ann=(radius>=5)&(radius<=40);even=(np.floor((radius-5)/5).astype(int)%2)==0
max_rms=0.;max_corr=0.;max_fit=0.;selection=[]
for row in records:
    i=int(row['pattern_index']);shift=(int(row['shift_y']),int(row['shift_x']))
    y=np.roll(images[i],shift,axis=(0,1));roi=np.roll(valid,shift,axis=(0,1))&ann
    train=roi if row['fold']=='full' else roi&(even if row['fold']=='even_to_odd' else ~even)
    test=roi if row['fold']=='full' else roi&~train
    j=0 if row['model']=='A' else 1;k=int(np.flatnonzero(rot==int(row['rotation_index']))[0])
    raw=lib[j,k];norm=float(raw[train].mean());coef,_=nnls(np.c_[raw[train]/norm,np.ones(train.sum())],y[train])
    p=coef[0]*raw/norm+coef[1]
    rms=float(np.sqrt(np.mean((p[test]-y[test])**2)))
    corr=float(np.corrcoef(np.log1p(y[test]),np.log1p(p[test]))[0,1])
    max_rms=max(max_rms,abs(rms-float(row['test_rms'])));max_corr=max(max_corr,abs(corr-float(row['test_logcorr'])))
    assert int(train.sum())==int(row['train_pixels']) and int(test.sum())==int(row['test_pixels'])
    if i in [0,25,67,75]:
        allrms=[]
        for template in lib[j]:
            norm=float(template[train].mean());a,_=nnls(np.c_[template[train]/norm,np.ones(train.sum())],y[train])
            allrms.append(float(np.sqrt(np.mean((a[0]*template[train]/norm+a[1]-y[train])**2))))
        diff=abs(min(allrms)-float(row['train_rms']));max_fit=max(max_fit,diff)
        selection.append({'pattern':i,'fold':row['fold'],'model':row['model'],'all_201_nnls_minimum_difference':diff})
assert len(records)==456 and len(selection)==24
assert max_rms<1e-10 and max_corr<1e-10 and max_fit<1e-10
for relative,expected in summary['source_hashes'].items():assert sha(ROOT/relative)==expected
dump(OUT/'independent_audit.json',{'status':'complete','raw_target_reconstructions':456,'predetermined_all_view_searches':24,
    'max_test_RMS_absolute_difference':max_rms,'max_test_logcorr_absolute_difference':max_corr,
    'max_minimum_train_RMS_difference':max_fit,'selection_checks':selection,'all_original_scientific_source_hashes_preserved':True})
print('AUDIT COMPLETE',max_rms,max_corr,max_fit,flush=True)
