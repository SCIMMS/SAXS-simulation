"""Full wide-ROI sharp/rounded reselection using five longitudinal FT planes.

All 8400 geometries are regenerated. Every candidate is scored on training
pixels only with float64 arithmetic. Held-out nuisance coefficients are frozen.
The three families share one library: fixed sharp 240, size-sharp 1200, all 8400.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,csv,json,time,gc
from pathlib import Path
from revision_common import ROOT,np,h5py,sm,sha,dump,save_csv,id57_meshes,rel_l2,idforward
from revision_complex_projection import ComplexProjectionRenderer
from revision_gpu_logcorr_exact import ExactLogCorrelation
from revision_gpu_boundary_general import MeshBoundaryReference
from revision_models import fit_rows,evaluate_fixed
from scipy.optimize import nnls
import cupy as cp

OUT=ROOT/'output/id57_qz5_rounding_2026-10-16'
OLD=ROOT/'output/revision_remaining_2026-10-09'
RADII=[0.,.5,1.,2.,4.,8.,12.]
SCALES=[.9,.95,1.,1.05,1.1]
FOLDS=['full','even_to_odd','odd_to_even']
FAMILIES={'fixed_sharp':np.arange(240)*5+2,'variable_size_sharp':np.arange(1200),'rounded':np.arange(8400)}


def source_hashes():
    files=[ROOT/'uploaded_SAXS_simulator/saxs_module.py']
    files += [ROOT/'validation'/f for f in ['revision_common.py','revision_resolution_validation.py',
        'revision_gpu_fourier.py','revision_complex_projection.py','revision_gpu_logcorr_exact.py',
        'revision_models.py','revision_gpu_boundary_general.py']]
    files += [ROOT/'public_data/CXIDB_ID_57/normal-incidence-pattern.cxi',
        ROOT/'public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240/synthetic_id57_patterns.npz',
        ROOT/'public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240/synthetic_id57_summary.json',
        OLD/'02_baseline/evaluation_masks.npz',OLD/'03_rounding/candidate_parameters.csv',
        OLD/'06_information/information_summary.json']
    files += list((ROOT/'output/revision_manuscript_R12_2026-10-15').glob('*.docx'))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def load_geometry_inputs():
    folder=ROOT/'public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240'
    cfg=json.loads((folder/'synthetic_id57_summary.json').read_text())
    with np.load(folder/'synthetic_id57_patterns.npz') as f:
        rotations=f['rotation_matrices'].astype(float);valid=f['valid_mask'].astype(bool)
    yy,xx=np.indices((401,401));radius=np.hypot(xx-200,yy-200)
    roi=valid&(radius>=45)&(radius<=190)
    even=(np.floor((radius-45)/10).astype(int)%2)==0
    with np.load(OLD/'02_baseline/evaluation_masks.npz') as f:
        assert np.array_equal(roi,f['roi'])
        assert np.array_equal(roi&even,f['train']) and np.array_equal(roi&~even,f['test'])
    k=2*np.pi/cfg['wavelength_nm'];step=k*cfg['pixel_size_m']/cfg['detector_distance_m']
    axis=(np.arange(401)-200)*step;qx,qy=np.meshgrid(axis,axis)
    factor=np.sqrt(1+(qx/k)**2+(qy/k)**2)
    physical=np.column_stack([(qx/factor)[roi],(qy/factor)[roi],(k*(1/factor-1))[roi]])
    assert roi.sum()==91567 and rotations.shape==(240,3,3)
    return cfg,axis,roi,rotations,physical,radius[roi],even[roi]


def initialize_data():
    cfg,axis,roi,rotations,physical,radius,even=load_geometry_inputs()
    masks=dict(mask=roi,qaxis=axis,rotation_matrices=rotations,qphysical=physical,radius_px=radius,even=even)
    path=OUT/'data_masks.npz'
    if path.exists():
        with np.load(path) as f:
            for key,value in masks.items():assert np.array_equal(f[key],value),key
    else:np.savez_compressed(path,**masks)
    ypath=OUT/'measured.npy'
    if not ypath.exists():
        y=np.lib.format.open_memmap(ypath,mode='w+',dtype='float64',shape=(209,int(roi.sum())))
        with h5py.File(ROOT/'public_data/CXIDB_ID_57/normal-incidence-pattern.cxi') as f:
            dataset=f[idforward.DATA_LINK_PATH];assert len(dataset)==209
            for frame in range(209):y[frame]=np.maximum(np.asarray(dataset[frame],dtype=np.float32)[roi],0)
        y.flush()
    return cfg,axis,roi,rotations,physical,radius,even


def generate():
    OUT.mkdir(exist_ok=True,parents=True);tic=time.perf_counter()
    cfg,axis,roi,rotations,physical,radius,even=initialize_data();hashes=source_hashes()
    parameters=list(csv.DictReader((OLD/'03_rounding/candidate_parameters.csv').open()))
    for idx,row in enumerate(parameters):
        assert int(row['candidate'])==idx and float(row['radius_base_nm'])==RADII[idx//1200]
        assert float(row['scale'])==SCALES[idx%5] and int(row['orientation_index'])==idx%1200//5
    config=dict(projection_pixels=768,extent_nm=144.,rounding_segments=16,qz_planes=5,
        radii_base_nm=RADII,isotropic_scales=SCALES,orientations=240,candidate_count=8400,
        n_frames=209,roi_pixels=int(roi.sum()),inner_radius_px=45,outer_radius_px=190,
        q_nominal_nm_inv=[float(radius.min()*(axis[201]-axis[200])),float(radius.max()*(axis[201]-axis[200]))],
        physical_qz_band_nm_inv=[float(physical[:,2].min()),0.],
        scaling='Uniform object scale s is evaluated at all three components s*qphysical; common s^6 intensity factor omitted because free fluence absorbs it.',
        sampling='Five Chebyshev-Lobatto q_z planes per orientation/scale, complex barycentric interpolation; four-point lateral Lagrange interpolation.',
        density_model='Rounded Pd outer cube plus +0.5 contrast in the fixed clipped octahedral core.',
        split='Original complementary 10-pixel radial bands, original masks preserved exactly.',
        precision='Complex64 projection accumulation, complex128 CZT/interpolation, float64 saved intensity and fit/score arithmetic.',
        source_hashes=hashes)
    config_path=OUT/'library_config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config
    else:dump(config_path,config)
    checkpoint=OUT/'generation_checkpoint.json'
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(done=[],qa=[],diagnostics=[])
    done=set(state['done']);qa=state['qa'];diagnostics=state['diagnostics']
    path=OUT/'qz5_library.npy'
    raw=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float64',shape=(8400,int(roi.sum())))
    sampled=(np.indices(roi.shape)[0]%4==0)&(np.indices(roi.shape)[1]%4==0)
    sub=sampled[roi]
    for ri,r in enumerate(RADII):
        outer,core=id57_meshes(rounding=r,segments=16);meshes=[(outer,1.),(core,.5)]
        renderers=[ComplexProjectionRenderer(meshes,axis*s,physical*s,roi,planes=5) for s in SCALES]
        mass=sm.signed_mesh_volume(outer.vertices,outer.faces)+.5*sm.signed_mesh_volume(core.vertices,core.faces)
        for oi,rotation in enumerate(rotations):
            group=ri*240+oi
            if group in done:continue
            zero,balance=renderers[2].projector.project(rotation,[0.],True)
            mass_error=float((cp.sum(zero.real,dtype=cp.float64)*(144/768)**2/mass-1).get())
            bad=int(cp.count_nonzero(balance).get())
            assert abs(mass_error)<1e-3 and bad==0,(r,oi,mass_error,bad)
            diagnostics.append(dict(group=group,radius_nm=r,orientation=oi,mass_relative_error=mass_error,unpaired_pixels=bad))
            for si,renderer in enumerate(renderers):
                idx=ri*1200+oi*5+si
                raw[idx]=renderer.intensity(rotation)
                if oi==0:
                    exact=MeshBoundaryReference(meshes,physical[sub]*SCALES[si]).intensity(rotation)
                    err=rel_l2(raw[idx,sub],exact)
                    assert err<.02,(idx,err)
                    qa.append(dict(candidate=idx,radius_nm=r,scale=SCALES[si],points=int(sub.sum()),
                                   independent_boundary_intensity_relative_L2=err))
            done.add(group)
            if oi%20==0:
                raw.flush();dump(checkpoint,dict(done=sorted(done),qa=qa,diagnostics=diagnostics))
                print('GENERATE',r,oi,'completed',len(done),'of1680','seconds',round(time.perf_counter()-tic,1),flush=True)
        raw.flush();dump(checkpoint,dict(done=sorted(done),qa=qa,diagnostics=diagnostics))
        del renderers;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    assert len(done)==1680
    for start in range(0,8400,64):
        block=np.asarray(raw[start:start+64]);assert np.isfinite(block).all() and (block>=0).all()
    save_csv(OUT/'candidate_parameters.csv',parameters);save_csv(OUT/'projection_diagnostics.csv',diagnostics)
    assert hashes==source_hashes()
    dump(OUT/'generation_audit.json',dict(status='complete',candidates=8400,groups=len(done),qa=qa,
        source_hashes_preserved=True,library_sha256=sha(path),seconds=time.perf_counter()-tic))
    print('GENERATION COMPLETE',time.perf_counter()-tic,flush=True)


def fitmask_for(fold,even):
    return np.ones(len(even),bool) if fold=='full' else (even if fold=='even_to_odd' else ~even)


def score(chunk=64):
    tic=time.perf_counter();config=json.loads((OUT/'library_config.json').read_text())
    assert config['source_hashes']==source_hashes()
    raw=np.load(OUT/'qz5_library.npy',mmap_mode='r');target=np.load(OUT/'measured.npy',mmap_mode='r')
    with np.load(OUT/'data_masks.npz') as f:even=f['even']
    checkpoint=OUT/'scoring_checkpoint.json'
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'chunks':{},'qa':[]}
    qa=state['qa']
    for fold in FOLDS:
        train=fitmask_for(fold,even)
        scorer=ExactLogCorrelation(np.asarray(target[:,train]),frame_batch=16)
        path=OUT/f'scores_{fold}.npy'
        bank=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float64',shape=(209,8400))
        begin=int(state['chunks'].get(fold,0))
        for start in range(begin,8400,chunk):
            stop=min(8400,start+chunk)
            block=np.ascontiguousarray(raw[start:stop][:,train])
            values=scorer.fit(block);assert np.isfinite(values).all()
            # Two predetermined frames: all 8400 candidates independently
            # rescored on CPU, in every fold, rather than checking only winners.
            for frame in [0,208]:
                cpu=fit_rows(target[frame,train],block)[0]
                error=float(np.max(np.abs(cpu-values[frame])))
                assert error<1e-8,(fold,start,frame,error)
                qa.append(dict(fold=fold,start=start,stop=stop,frame=frame,max_GPU_CPU_score_difference=error))
            bank[:,start:stop]=values;bank.flush();state['chunks'][fold]=stop
            dump(checkpoint,state)
            if start%(chunk*4)==0:
                print('SCORE',fold,stop,'of8400','pixels',int(train.sum()),'seconds',round(time.perf_counter()-tic,1),flush=True)
        del scorer;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    dump(OUT/'scoring_audit.json',dict(status='complete',complete_candidate_scores=209*8400*3,
        CPU_all_candidate_frame_fold_checks=6,checked_candidate_frame_fold_pairs=8400*6,
        max_GPU_CPU_score_difference=max(r['max_GPU_CPU_score_difference'] for r in qa),
        source_hashes_preserved=config['source_hashes']==source_hashes(),seconds=time.perf_counter()-tic))


def select():
    tic=time.perf_counter();config=json.loads((OUT/'library_config.json').read_text())
    assert config['source_hashes']==source_hashes()
    raw=np.load(OUT/'qz5_library.npy',mmap_mode='r');target=np.load(OUT/'measured.npy',mmap_mode='r')
    with np.load(OUT/'data_masks.npz') as f:even=f['even']
    rows=[];profiles=[]
    for fold in FOLDS:
        train=fitmask_for(fold,even);test=train if fold=='full' else ~train
        bank=np.load(OUT/f'scores_{fold}.npy',mmap_mode='r')
        assert np.isfinite(bank).all()
        for frame,image in enumerate(target):
            scores=bank[frame]
            for family,candidates in FAMILIES.items():
                ranked=candidates[np.argsort(scores[candidates])[::-1]]
                ids=np.union1d(ranked[:8],candidates[scores[candidates]>=scores[ranked[0]]-1e-7]).astype(int)
                cs=[];aa=[];bb=[];means=[]
                for start in range(0,len(ids),32):
                    chosen=ids[start:start+32];xx=np.asarray(raw[chosen][:,train])
                    c,a,b,_,_=fit_rows(image[train],xx)
                    cs.extend(c);aa.extend(a);bb.extend(b);means.extend(xx.mean(axis=1))
                cs=np.array(cs);j=int(np.argmax(cs));idx=int(ids[j]);a=float(aa[j]);b=float(bb[j]);mean=float(means[j])
                delta=float(np.max(np.abs(cs-scores[ids])));assert delta<1e-8,(fold,frame,family,delta)
                ev,pred=evaluate_fixed(image[test],raw[idx,test],a,b,mean)
                coeff,_=nnls(np.c_[raw[idx,train]/mean,np.ones(train.sum())],image[train])
                independent=coeff[0]*raw[idx,test]/mean+coeff[1]
                diff=rel_l2(pred,independent);assert diff<1e-9,(fold,frame,family,diff)
                ri=idx//1200;oi=idx%1200//5;si=idx%5
                rows.append(dict(fold=fold,frame=frame,family=family,candidate=idx,
                    base_radius_nm=RADII[ri],scale=SCALES[si],side_nm=52*SCALES[si],
                    actual_rounding_radius_nm=RADII[ri]*SCALES[si],orientation_index=oi,
                    fit_logcorr=float(cs[j]),test_logcorr=ev,signal_normalized=a,background=b,
                    train_template_mean=mean,fit_pixels=int(train.sum()),test_pixels=int(test.sum()),
                    radius_at_upper_bound=ri==6,scale_at_bound=si in [0,4],CPU_finalists=len(ids),
                    max_shortlist_GPU_CPU_difference=delta,independent_nnls_prediction_relative_L2=diff))
            for radius,value in zip(RADII,scores.reshape(7,240,5).max(axis=(1,2))):
                profiles.append(dict(fold=fold,frame=frame,base_radius_nm=radius,fit_profile_score=float(value)))
            if frame%25==0:
                save_csv(OUT/'selection_results.csv',rows)
                print('SELECT',fold,frame,'seconds',round(time.perf_counter()-tic,1),flush=True)
    save_csv(OUT/'selection_results.csv',rows);save_csv(OUT/'radius_profiles.csv',profiles)
    summary=summarize_selection(rows,'test_logcorr')
    dump(OUT/'selection_summary.json',dict(status='selection_complete_exact_boundary_audit_pending',
        results=summary,n_selected_fits=len(rows),candidates=8400,frames=209,
        original_planar_wide_ROI=json.loads((OLD/'06_information/information_summary.json').read_text())['conditional_grid'],
        max_shortlist_GPU_CPU_difference=max(r['max_shortlist_GPU_CPU_difference'] for r in rows),
        max_independent_nnls_prediction_relative_L2=max(r['independent_nnls_prediction_relative_L2'] for r in rows),
        source_hashes_preserved=config['source_hashes']==source_hashes(),seconds=time.perf_counter()-tic))
    print('SELECTION COMPLETE',json.dumps(summary),flush=True)


def summarize_selection(rows,key):
    result={};paired=[]
    for fold in FOLDS:
        arrays={name:sorted([r for r in rows if r['fold']==fold and r['family']==name],key=lambda r:int(r['frame']))
                for name in FAMILIES}
        scores={name:np.array([float(r[key]) for r in rr]) for name,rr in arrays.items()}
        assert all(len(s)==209 for s in scores.values())
        delta=scores['rounded']-scores['variable_size_sharp']
        if fold=='full':assert delta.min()>-1e-8
        else:paired.append(delta)
        result[fold]=dict(n=209,mean_scores={name:float(s.mean()) for name,s in scores.items()},
            mean_rounding_over_size_delta=float(delta.mean()),median_rounding_delta=float(np.median(delta)),
            rounding_positive=int((delta>1e-8).sum()),rounding_negative=int((delta< -1e-8).sum()),
            rounding_tied=int((np.abs(delta)<=1e-8).sum()),
            mean_size_over_fixed_delta=float((scores['variable_size_sharp']-scores['fixed_sharp']).mean()),
            radius_upper_bound_count=sum(float(r['base_radius_nm'])==12 for r in arrays['rounded']),
            radius_zero_count=sum(float(r['base_radius_nm'])==0 for r in arrays['rounded']),
            scale_bound_count=sum(float(r['scale']) in [.9,1.1] for r in arrays['rounded']))
    perframe=np.mean(paired,axis=0);rng=np.random.default_rng(20261016)
    boot=rng.choice(perframe,size=(10000,209),replace=True).mean(axis=1)
    result['paired_twofold_mean']=dict(n_frames=209,mean_delta=float(perframe.mean()),
        median_delta=float(np.median(perframe)),positive_frames=int((perframe>1e-8).sum()),
        negative_frames=int((perframe< -1e-8).sum()),tied_frames=int((np.abs(perframe)<=1e-8).sum()),
        descriptive_frame_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generate','score','select','all'])
    parser.add_argument('--chunk',type=int,default=64);args=parser.parse_args()
    if args.stage in ['generate','all']:generate()
    if args.stage in ['score','all']:score(args.chunk)
    if args.stage in ['select','all']:select()
