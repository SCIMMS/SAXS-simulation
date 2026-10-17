"""Independent selected-fit NNLS reconstruction and continuous Ewald audit.

The 8400-candidate selection is carried out by the five-plane renderer. This
audit holds each selected geometry fixed and refits nuisance on training only
using exact continuous boundary intensities. It does not relabel this as an
exhaustive exact-boundary search.
"""
from __future__ import annotations
from revision_id57_qz5_rounding import *
from revision_polyhedron_reference import polyhedron_amplitude
import scipy,platform


def corr(y,p):
    a=np.log1p(np.maximum(y,0));b=np.log1p(np.maximum(p,0))
    a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-20 else 0.


def exact_references():
    tic=time.perf_counter();config=json.loads((OUT/'library_config.json').read_text())
    assert config['source_hashes']==source_hashes()
    rows=list(csv.DictReader((OUT/'selection_results.csv').open()))
    assert len(rows)==1881
    ids=sorted({int(r['candidate']) for r in rows});lookup={idx:j for j,idx in enumerate(ids)}
    idpath=OUT/'exact_candidate_ids.json'
    if idpath.exists():assert json.loads(idpath.read_text())==ids
    else:dump(idpath,ids)
    with np.load(OUT/'data_masks.npz') as f:q=f['qphysical'];rot=f['rotation_matrices']
    file=OUT/'exact_curved_selected.npy'
    bank=np.lib.format.open_memmap(file,mode='r+' if file.exists() else 'w+',dtype='float64',shape=(len(ids),len(q)))
    checkpoint=OUT/'exact_reference_checkpoint.json'
    done=set(json.loads(checkpoint.read_text()) if checkpoint.exists() else [])
    last_radius=None;references={};meshes=None
    for j,idx in enumerate(ids):
        if idx in done:continue
        ri=idx//1200;si=idx%5;oi=idx%1200//5
        if ri!=last_radius:
            references.clear();gc.collect();cp.get_default_memory_pool().free_all_blocks()
            outer,core=id57_meshes(rounding=RADII[ri],segments=16)
            meshes=[(outer,1.),(core,.5)];last_radius=ri
        if si not in references:references[si]=MeshBoundaryReference(meshes,q*SCALES[si])
        bank[j]=references[si].intensity(rot[oi])
        done.add(idx)
        if j%20==0:
            bank.flush();dump(checkpoint,sorted(done))
            print('EXACT REFERENCE',len(done),'of',len(ids),'seconds',round(time.perf_counter()-tic,1),flush=True)
    bank.flush();dump(checkpoint,sorted(done))
    assert len(done)==len(ids)
    qa=[];qids=np.linspace(0,len(q)-1,73,dtype=int)
    for j in np.linspace(0,len(ids)-1,min(12,len(ids)),dtype=int):
        idx=ids[j];ri=idx//1200;si=idx%5;oi=idx%1200//5
        outer,core=id57_meshes(rounding=RADII[ri],segments=16)
        query=q[qids]*SCALES[si]@rot[oi]
        cpu=polyhedron_amplitude(query,outer)+.5*polyhedron_amplitude(query,core)
        err=rel_l2(bank[j,qids],abs(cpu)**2);assert err<1e-7,(idx,err)
        qa.append(dict(candidate=idx,points=73,independent_CPU_GPU_intensity_relative_L2=err))
    dump(OUT/'exact_reference_audit.json',dict(status='complete',unique_selected_candidates=len(ids),
        CPU_reference_checks=qa,reference_library_sha256=sha(file),seconds=time.perf_counter()-tic,
        source_hashes_preserved=config['source_hashes']==source_hashes()))


def reference_summary(rows,key):
    result={};paired=[]
    for fold in FOLDS:
        scores={family:np.array([float(r[key]) for r in sorted(
            [r for r in rows if r['fold']==fold and r['family']==family],key=lambda r:int(r['frame']))])
            for family in FAMILIES}
        assert all(len(x)==209 for x in scores.values())
        delta=scores['rounded']-scores['variable_size_sharp']
        if fold!='full':paired.append(delta)
        result[fold]=dict(n=209,mean_scores={name:float(a.mean()) for name,a in scores.items()},
            mean_rounding_over_size_delta=float(delta.mean()),median_rounding_delta=float(np.median(delta)),
            positive=int((delta>1e-8).sum()),negative=int((delta< -1e-8).sum()),tied=int((abs(delta)<=1e-8).sum()))
    perframe=np.mean(paired,axis=0);rng=np.random.default_rng(20261016)
    boot=rng.choice(perframe,size=(10000,209),replace=True).mean(axis=1)
    result['paired_twofold_mean']=dict(mean_delta=float(perframe.mean()),median_delta=float(np.median(perframe)),
        positive_frames=int((perframe>1e-8).sum()),negative_frames=int((perframe< -1e-8).sum()),
        tied_frames=int((abs(perframe)<=1e-8).sum()),
        descriptive_frame_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist())
    return result


def reconstruct():
    tic=time.perf_counter();config=json.loads((OUT/'library_config.json').read_text())
    assert config['source_hashes']==source_hashes()
    # Rebuild masks from the original configuration and reread each source frame.
    cfg,axis,roi,rotations,q,radius,even=load_geometry_inputs()
    target=np.load(OUT/'measured.npy',mmap_mode='r')
    with h5py.File(ROOT/'public_data/CXIDB_ID_57/normal-incidence-pattern.cxi') as f:
        ds=f[idforward.DATA_LINK_PATH]
        for frame in range(209):assert np.array_equal(np.maximum(np.asarray(ds[frame],dtype=np.float32)[roi],0),target[frame])
    with np.load(OUT/'data_masks.npz') as f:
        assert np.array_equal(even,f['even']) and np.array_equal(roi,f['mask'])
        assert np.array_equal(q,f['qphysical']) and np.array_equal(rotations,f['rotation_matrices'])
    ids=json.loads((OUT/'exact_candidate_ids.json').read_text());lookup={idx:j for j,idx in enumerate(ids)}
    raw=np.load(OUT/'qz5_library.npy',mmap_mode='r');exact=np.load(OUT/'exact_curved_selected.npy',mmap_mode='r')
    rows=list(csv.DictReader((OUT/'selection_results.csv').open()))
    assert len(rows)==1881 and len({(r['fold'],r['frame'],r['family']) for r in rows})==1881
    model_errors=[]
    for idx in ids:
        j=lookup[idx]
        model_errors.append(dict(candidate=idx,base_radius_nm=RADII[idx//1200],scale=SCALES[idx%5],
            intensity_relative_L2=rel_l2(raw[idx],exact[j])))
    save_csv(OUT/'selected_numerical_errors.csv',model_errors)
    fits=[]
    for count,r in enumerate(rows):
        frame=int(r['frame']);idx=int(r['candidate']);train=fitmask_for(r['fold'],even)
        test=train if r['fold']=='full' else ~train
        values={}
        for name,x in [('qz5',raw[idx]),('exact',exact[lookup[idx]])]:
            mean=float(x[train].mean())
            coef,_=nnls(np.c_[x[train]/mean,np.ones(train.sum())],target[frame,train])
            pred=coef[0]*x/mean+coef[1]
            values[name+'_train_logcorr']=corr(target[frame,train],pred[train])
            values[name+'_test_logcorr']=corr(target[frame,test],pred[test])
            values[name+'_signal_normalized']=float(coef[0]);values[name+'_background']=float(coef[1])
        trainerr=abs(values['qz5_train_logcorr']-float(r['fit_logcorr']))
        testerr=abs(values['qz5_test_logcorr']-float(r['test_logcorr']))
        assert trainerr<1e-8 and testerr<1e-8,(count,trainerr,testerr)
        fits.append(dict(fold=r['fold'],frame=frame,family=r['family'],candidate=idx,**values,
                         qz5_train_score_reconstruction_error=trainerr,qz5_test_score_reconstruction_error=testerr))
        if count%150==0:
            save_csv(OUT/'selected_exact_reference_fits.csv',fits)
            print('NNLS AUDIT',count,'of1881','seconds',round(time.perf_counter()-tic,1),flush=True)
    save_csv(OUT/'selected_exact_reference_fits.csv',fits)
    summary=reference_summary(fits,'exact_test_logcorr')
    audit=dict(status='complete',source_frames_and_masks_independently_reconstructed=True,
        selected_qz5_fits_reconstructed_with_SciPy_NNLS=len(rows),selected_exact_reference_fits_with_training_only_NNLS=len(rows),
        max_qz5_train_logcorr_reconstruction_error=max(r['qz5_train_score_reconstruction_error'] for r in fits),
        max_qz5_test_logcorr_reconstruction_error=max(r['qz5_test_score_reconstruction_error'] for r in fits),
        selected_unique_candidates=len(ids),selected_intensity_L2=dict(
            mean=float(np.mean([r['intensity_relative_L2'] for r in model_errors])),
            max=max(r['intensity_relative_L2'] for r in model_errors),
            above_1_percent=sum(r['intensity_relative_L2']>.01 for r in model_errors)),
        exact_reference_results=summary,
        scope='Frozen geometries newly selected by the full 8400-candidate five-plane search; exact boundary nuisance is refitted only on the original training pixels. No exhaustive exact-boundary reselection.',
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,cupy=cp.__version__),
        source_hashes_preserved=config['source_hashes']==source_hashes(),seconds=time.perf_counter()-tic)
    dump(OUT/'independent_result_audit.json',audit)
    print('INDEPENDENT AUDIT COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='references':exact_references()
    elif sys.argv[1]=='reconstruct':reconstruct()
    elif sys.argv[1]=='all':exact_references();reconstruct()
    else:raise ValueError(sys.argv[1])
