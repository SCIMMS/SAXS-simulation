"""Matched finite-view A/B comparison, separate from frozen historical classes.

Fixed historical integer alignment makes withheld-band results conditional.
Both assemblies use all 201 pre-symmetry views and the same original FFT code.
"""
from revision_common import ROOT, np, dump, save_csv, sha
import csv, json, h5py, time
import section5_generate_groel_model_ab_preview as forward
from scipy.optimize import nnls

OUT=ROOT/'output/groel_ab_matched_2026-10-15'
OUT.mkdir(exist_ok=True,parents=True)
HITS=ROOT/'validation_supporting_package/section5_groel/curated_run151_hits/hits_r0151_curated76_ds8.h5'
DENSITY=ROOT/'tmp/revision_response_bundle_5_6_2026-10-09/groel_original_settings_density_grid.npz'
ASSIGN=ROOT/'validation_supporting_package/section5_groel/retrieval_symD7_curated76/groel_retrieval_best_with_frames.csv'
ROT=ROOT/'public_data/CXIDB_ID_187/rotations.h5'

def fit(y,x):
    # Normalize rows before solving to avoid the legacy ~1e-9 signal scale.
    norm=np.maximum(x.mean(axis=1),1e-30); z=x/norm[:,None]
    n=len(y); sx=z.sum(1); sxx=(z*z).sum(1); sy=y.sum(); xy=z@y
    den=n*sxx-sx*sx
    a=np.divide(n*xy-sx*sy,den,out=np.zeros(len(x)),where=den>1e-20); b=(sy-a*sx)/n
    bad=(a<0)|(b<0); ao=np.maximum(xy/sxx,0); bo=max(float(y.mean()),0)
    so=((y[None,:]-ao[:,None]*z)**2).sum(1); sb=((y-bo)**2).sum()
    a=np.where(bad,np.where(so<=sb,ao,0),a); b=np.where(bad,np.where(so<=sb,0,bo),b)
    mse=((a[:,None]*z+b[:,None]-y)**2).mean(1)
    return a/norm,b,mse

def score(y,p):
    a=np.log1p(np.maximum(y,0));b=np.log1p(np.maximum(p,0))
    return float(np.corrcoef(a,b)[0,1])

def main():
    sources=[HITS,DENSITY,ASSIGN,ROT,ROOT/'validation/section5_generate_groel_model_ab_preview.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    path=OUT/'templates.npz'
    if not path.exists():
        pts=forward.load_density_points(DENSITY,1e-4)
        with h5py.File(ROT) as h: quats=h['rotations'][::128].astype(np.float32)
        assert len(quats)==201
        qx,qy,_=forward.detector_q_grid(128,8,1200.,.15,75e-6)
        arrays=[]
        for model in ['A_heptamer_C7','B_tetradecamer_D7']:
            c,w,_=forward.make_model_points(pts,model); patterns=[]
            for i,q in enumerate(quats):
                proj,dx=forward.project_points(c,w,forward.quaternion_wxyz_to_matrix(q),1.,18.)
                patterns.append(forward.diffraction_from_projection(proj,dx,qx,qy,2048))
                if i%30==0: print('GENERATE',model,i,flush=True)
            arrays.append(np.stack(patterns))
        np.savez_compressed(path,templates=np.stack(arrays),quaternions=quats,rotation_indices=np.arange(201)*128)
    with np.load(path) as z: lib=z['templates'].astype(float); rindices=z['rotation_indices']
    with h5py.File(HITS) as h:
        images=h['patterns_ph'][:].astype(float);valid=h['mask'][:].astype(bool)
        h5attrs={k:str(v) for k,v in h.attrs.items()}
    assignments={int(r['pattern_index']):r for r in csv.DictReader(ASSIGN.open())}
    yy,xx=np.indices((128,128));rad=np.hypot(xx-63.5,yy-63.5)
    ann=(rad>=5)&(rad<=40);even=(np.floor((rad-5)/5).astype(int)%2)==0
    records=[];qa=[];predictions={}
    for i,image in enumerate(images):
        row=assignments[i];shift=(int(row['shift_y']),int(row['shift_x']))
        y=np.roll(image,shift,axis=(0,1)); mask=np.roll(valid,shift,axis=(0,1))&ann
        for fold,train,test in [('full',mask,mask),('even_to_odd',mask&even,mask&~even),('odd_to_even',mask&~even,mask&even)]:
            for j,model in enumerate(['A','B']):
                a,b,mse=fit(y[train],lib[j,:,train].T if lib[j,:,train].shape[0]!=201 else lib[j,:,train])
                k=int(np.argmin(mse)); p=a[k]*lib[j,k]+b[k]
                # Independent SciPy NNLS on every selected template.
                raw=lib[j,k,train];norm=max(float(raw.mean()),1e-30)
                coef,_=nnls(np.c_[raw/norm,np.ones(len(raw))],y[train])
                p2=coef[0]*raw/norm+coef[1]
                rel=float(np.linalg.norm(p2-p[train])/max(np.linalg.norm(p2),1e-30))
                qa.append(rel);assert rel<1e-10
                records.append({'pattern_index':i,'source_frame_index':int(row['source_frame_index']),
                    'fold':fold,'model':model,'rotation_index':int(rindices[k]),'signal':float(a[k]),'background':float(b[k]),
                    'train_rms':float(np.sqrt(mse[k])),'test_rms':float(np.sqrt(np.mean((p[test]-y[test])**2))),
                    'test_logcorr':score(y[test],p[test]),'train_pixels':int(train.sum()),'test_pixels':int(test.sum()),
                    'shift_y':shift[0],'shift_x':shift[1]})
                if i==67 and fold=='full':predictions[model]=p
        if i%15==0: print('FIT',i,flush=True)
    save_csv(OUT/'paired_results.csv',records)
    results={}
    for fold in ['full','even_to_odd','odd_to_even']:
        aa=[r for r in records if r['fold']==fold and r['model']=='A'];bb=[r for r in records if r['fold']==fold and r['model']=='B']
        rms=np.array([a['test_rms']-b['test_rms'] for a,b in zip(aa,bb)])
        corr=np.array([a['test_logcorr']-b['test_logcorr'] for a,b in zip(aa,bb)])
        results[fold]={'n':76,'A_lower_RMS':int((rms<0).sum()),'B_lower_RMS':int((rms>0).sum()),
          'median_A_minus_B_RMS':float(np.median(rms)),'mean_A_minus_B_RMS':float(rms.mean()),
          'median_A_minus_B_logcorr':float(np.median(corr)),'A_mean_logcorr':float(np.mean([r['test_logcorr'] for r in aa])),
          'B_mean_logcorr':float(np.mean([r['test_logcorr'] for r in bb]))}
    shot=[r for r in records if r['pattern_index']==67]
    np.savez_compressed(OUT/'pattern67_predictions.npz',measured=np.roll(images[67],(int(assignments[67]['shift_y']),int(assignments[67]['shift_x'])),axis=(0,1)),A=predictions['A'],B=predictions['B'])
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in sources}
    dump(OUT/'summary.json',{'status':'complete','source_hashes':hashes,'templates_sha256':sha(path),
      'source_h5_attributes':h5attrs,'views_per_assembly':201,'rotation_indices':'0:25680:128 before symmetry filtering',
      'backend':'Original density-point projection; complex-amplitude bilinear FFT sampling before modulus squared',
      'selection':'Training NNLS RMS; log-correlation diagnostic only; same frozen historical integer alignment',
      'holdout':'Two complementary 5-pixel radial-band folds; conditional on historical full-ROI alignment and curated sample',
      'independent_nnls_max_prediction_relative_L2':max(qa),'results':results,'pattern67':shot,
      'scope':'Finite candidate comparison, no calibrated assembly identity, no modification of original classes or refinement',
      'runtime_note':'No runtime performance claim from this experiment'})
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
