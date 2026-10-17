"""Frozen GroEL class/orientation comparison: point/mesh x FFT/CZT.

Lengths and projected density use A and electrons/A^2 throughout, matching the
historical FFT library. Detector qx/qy, qz=0, alignment and masks are unchanged.
No geometry, orientation, class or alignment optimization is performed.
"""
from __future__ import annotations
import argparse,csv,json,time,platform
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from revision_common import ROOT,np,sm,sha,dump,save_csv
import section5_generate_groel_model_ab_preview as legacy
from scipy.optimize import nnls
import h5py
import cupy as cp
from cupyx.scipy.signal import CZT
from revision_gpu_boundary_general import MeshBoundaryReference

OUT=ROOT/'output/groel_mesh_czt_matched_2026-10-17'
PRIOR=ROOT/'output/differentiable_groel_2026-10-11'
SUP=ROOT/'validation_supporting_package/section5_groel'
ASSIGN=SUP/'retrieval_symD7_curated76/groel_retrieval_best_with_frames.csv'
HITS=SUP/'curated_run151_hits/hits_r0151_curated76_ds8.h5'
DENSITY=ROOT/'tmp/revision_response_bundle_5_6_2026-10-09/groel_original_settings_density_grid.npz'
ROUTES=['point_fft','point_czt','mesh_fft','mesh_czt']

def assignments():
    rows={int(r['pattern_index']):r for r in csv.DictReader(ASSIGN.open(encoding='utf-8'))}
    assert sorted(rows)==list(range(76))
    return rows

def unique_templates():
    result={}
    for i,row in assignments().items():
        k=int(row['template_index'])
        if k in result:
            old=result[k]
            assert all(row[x]==old[x] for x in ['model_name','rotation_index','qw','qx','qy','qz'])
        else:result[k]=row
    assert len(result)==34
    return dict(sorted(result.items()))

def protected():
    paths=[ASSIGN,HITS,DENSITY,Path(__file__),ROOT/'validation/revision_common.py',
        ROOT/'validation/section5_generate_groel_model_ab_preview.py',ROOT/'uploaded_SAXS_simulator/saxs_module.py',
        ROOT/'validation/revision_gpu_boundary_general.py',ROOT/'validation/revision_gpu_surface_reference.py']
    paths += [PRIOR/(m+'.npz') for m in sorted({r['model_name'] for r in assignments().values()})]
    paths += [PRIOR/'protein_boundary.npz',PRIOR/'finer_protein_boundary.npz']
    paths += sorted((PRIOR/'legacy_baselines').glob('template_*.npz'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def detector():return legacy.detector_q_grid(128,8,1200.,.15,75e-6)[:2]

def q_lattice(nfft=2048,dx=1.):
    qx,qy=detector()
    complete=np.fft.fftshift(2*np.pi*np.fft.fftfreq(nfft,d=dx))
    labels=complete.astype(np.float32)
    low=float(min(qx.min(),qy.min()));high=float(max(qx.max(),qy.max()))
    start=int(np.searchsorted(labels,low,side='right')-1)
    stop=int(np.searchsorted(labels,high,side='right')+1)
    assert 0<=start<stop<len(complete)
    return complete[start:stop],labels[start:stop]

def czt_amplitude(projection,dx=1.,nfft=2048,physical_origin=False):
    """Compute only the exact frequency-lattice subset needed by the detector.

    The original even-sized FFT treats its first projection pixel as -N*dx/2,
    while histogram/raster centers are (-N/2+1/2)*dx. Preserve the original
    half-pixel translation before amplitude interpolation. At changed spatial
    dx, use the same fixed -0.5 A translation relative to physical centers.
    """
    q,labels=q_lattice(nfft,1.)
    n=len(projection);u0=-n*dx/2+dx/2-.5
    if physical_origin:u0+=.5
    plan=CZT(n=n,m=len(q),w=np.exp(-1j*(q[1]-q[0])*dx),a=np.exp(1j*q[0]*dx))
    phase=cp.asarray(dx*np.exp(-1j*q*u0))
    p=cp.asarray(projection,dtype=cp.complex128)
    amp=plan(plan(p,axis=1)*phase[None,:],axis=0)*phase[:,None]
    return cp.asnumpy(amp),labels

def czt_intensity(projection,dx=1.,nfft=2048):
    amp,labels=czt_amplitude(projection,dx,nfft)
    qx,qy=detector()
    out=legacy.sample_complex_bilinear(amp,labels,qx,qy)
    return (np.abs(out)**2).astype(np.float32)

def rel(a,b):return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),1e-30))

def model_points(points,model):
    c,w,center=legacy.make_model_points(points,'B_tetradecamer_D7')
    if 'polyprism' in model:
        c,w,_=legacy.add_polygonal_prism_to_model_b(points,c,w,center,
            35. if 'R35_' in model else 55.,135.,14,2.,.267,.2)
    return c,w

def mesh_parts(model,finer=False):
    with np.load(PRIOR/(model+'.npz')) as z:
        vertices=z['vertices']*10.;faces=z['faces'];contrast=z['contrast']/1000.
    parts=[]
    for value in np.unique(contrast):
        ff=faces[contrast==value]
        if finer and len(ff)==23848:
            with np.load(PRIOR/'finer_protein_boundary.npz') as z:
                vv=z['vertices']*10.;ff=z['faces'];value=float(z['contrast'][0])/1000.
        else:vv=vertices
        normals=sm.compute_face_normals(vv,ff)
        parts.append((SimpleNamespace(vertices=vv,faces=ff,face_normals=normals),float(value)))
    return parts

def mesh_projection(parts,rotation,extent,n):
    p=np.zeros((n,n),float);diags=[]
    for mesh,contrast in parts:
        oriented=SimpleNamespace(vertices=mesh.vertices@rotation.T,faces=mesh.faces,face_normals=mesh.face_normals@rotation.T)
        assert np.max(np.abs(oriented.vertices[:,:2]))<extent/2
        image,d=sm._signed_thickness_from_rasterization_gpu(oriented,
            u=np.array([1.,0,0]),v=np.array([0.,1,0]),n=np.array([0.,0,1]),
            uv_bounds=(-extent/2,extent/2,-extent/2,extent/2),image_shape=(n,n))
        p+=contrast*image;diags.append(d)
    return p.astype(np.float32),diags

def generate():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'templates').mkdir(exist_ok=True)
    hashes=protected();points=legacy.load_density_points(DENSITY,1e-4);qx,qy=detector()
    config=dict(status='running',source_hashes=hashes,frames=76,unique_templates=34,
        class_orientation_alignment='Frozen original CSV; exact recorded quaternion, no normalization beyond legacy quaternion routine',
        point_density='Original sigma6 PDB-derived density, 1e-4 cutoff; original retained prism point occupancy and weights',
        mesh='Existing unrefined 12-level layered density meshes and fixed class-specific prism boundaries; no fitted deformations',
        projection_pixel_A=1.,projection_margin_A=18.,legacy_fft_size=2048,
        detector=dict(shape=[128,128],downsample=8,energy_eV=1200.,distance_m=.15,native_pixel_m=75e-6),
        q_condition='Identical physical transverse qx/qy labels as original FFT; qz=0 for every route. No Ewald-correction difference.',
        transform='CZT evaluates a cropped subset of the exact original padded-FFT lattice. Same float32 frequency labels and complex bilinear detector interpolation.',
        centering='Legacy half-pixel Fourier translation explicitly preserved before interpolation.',
        fitting='Frozen original signal/background plus matched nonnegative signal/constant-background NNLS on full or training pixels only',
        folds='5-pixel radial bands starting at radius 5; complementary even/odd; annulus 5-40 pixels',
        caveat='Historical classes/orientations/integer alignment selected using full ROI; all held-out comparisons conditional',
        performance='Generation wall times are recorded for bookkeeping; no comparative timing claim')
    dump(OUT/'experiment_config.json',config)
    qa=[];records=[];start=time.perf_counter();cache={}
    q64,labels=q_lattice();ann=np.hypot(*np.meshgrid(np.arange(128)-63.5,np.arange(128)-63.5));ann=(ann>=5)&(ann<=40)
    for k,row in unique_templates().items():
        path=OUT/f'templates/template_{k:03d}.npz'
        if row['model_name'] not in cache:
            cache[row['model_name']]=(model_points(points,row['model_name']),mesh_parts(row['model_name']))
        (coords,weights),parts=cache[row['model_name']]
        rotation=legacy.quaternion_wxyz_to_matrix([float(row[q]) for q in ['qw','qx','qy','qz']])
        projection,dx=legacy.project_points(coords,weights,rotation,1.,18.)
        n=len(projection);extent=float(n);t0=time.perf_counter()
        pfft=legacy.diffraction_from_projection(projection,dx,qx,qy,2048)
        pczt=czt_intensity(projection)
        mp,diag=mesh_projection(parts,rotation,extent,n)
        mfft=legacy.diffraction_from_projection(mp,1.,qx,qy,2048)
        mczt=czt_intensity(mp)
        patterns=np.stack([pfft,pczt,mfft,mczt])
        assert np.isfinite(patterns).all() and (patterns>=0).all()
        for a,b in [(pczt,pfft),(mczt,mfft)]:assert rel(a,b)<2e-6,rel(a,b)
        with np.load(PRIOR/f'legacy_baselines/template_{k}.npz') as z:
            old=z['intensity'];old_error=rel(pfft,old);assert old_error<1e-12,(k,old_error)
        mesh_charge=sum(c*sm.signed_mesh_volume(m.vertices,m.faces) for m,c in parts)
        qa.append(dict(template_index=k,model_name=row['model_name'],rotation_index=int(row['rotation_index']),
            projection_n=n,legacy_reproduction_relative_L2=old_error,
            point_CZT_FFT_relative_L2=rel(pczt,pfft),mesh_CZT_FFT_relative_L2=rel(mczt,mfft),
            mesh_point_raw_intensity_relative_L2=rel(mczt[ann],pfft[ann]),
            mesh_point_mean_normalized_relative_L2=rel(mczt[ann]/mczt[ann].mean(),pfft[ann]/pfft[ann].mean()),
            point_projection_charge_e=float(projection.sum(dtype=float)),point_charge_e=float(weights.sum(dtype=float)),
            mesh_exact_charge_e=mesh_charge,mesh_projection_charge_e=float(mp.sum(dtype=float)),
            mesh_to_point_charge_relative_difference=mesh_charge/weights.sum(dtype=float)-1,
            mesh_projection_charge_relative_error=float(mp.sum(dtype=float)/mesh_charge-1),
            bad_balance_pixels=sum(d['bad_balance_pixels'] for d in diag),
            negative_clamped_pixels=sum(d['negative_pixels_before_clamp'] for d in diag),
            global_sign_flips=sum(d['used_global_sign_flip'] for d in diag),generation_seconds=time.perf_counter()-t0))
        np.savez_compressed(path,patterns=patterns,routes=np.array(ROUTES),point_projection=projection,mesh_projection=mp,
            rotation=rotation,extent_A=extent,template_index=k)
        records.append(dict(template_index=k,class_name=row['model_name'],rotation_index=int(row['rotation_index']),source_quaternion=[float(row[q]) for q in ['qw','qx','qy','qz']],faces=sum(len(m.faces) for m,c in parts)))
        print('GENERATE',k,row['model_name'],n,'point transform',qa[-1]['point_CZT_FFT_relative_L2'],'mesh/point',qa[-1]['mesh_point_mean_normalized_relative_L2'],flush=True)
    save_csv(OUT/'template_qa.csv',qa);save_csv(OUT/'fixed_templates.csv',records)
    assert hashes==protected()
    config.update(status='complete',generation_seconds=time.perf_counter()-start)
    dump(OUT/'experiment_config.json',config)

def logcorr(a,b):return float(np.corrcoef(np.log1p(a),np.log1p(b))[0,1])

def nuisance(y,x,mask):
    norm=float(x[mask].mean());coef,_=nnls(np.c_[x[mask]/norm,np.ones(mask.sum())],y[mask])
    return float(coef[0]/norm),float(coef[1])

def evaluate():
    cfg=json.loads((OUT/'experiment_config.json').read_text());assert cfg['source_hashes']==protected()
    source=assignments();data={}
    with h5py.File(HITS) as h:
        images=h['patterns_ph'][:].astype(float);valid=h['mask'][:].astype(bool);frames=h['selected_frame_indices'][:]
    yy,xx=np.indices((128,128));radius=np.hypot(xx-63.5,yy-63.5);ann=(radius>=5)&(radius<=40)
    even=(np.floor((radius-5)/5).astype(int)%2)==0
    records=[];original_errors=[];original_corr_errors=[];examples={min(i for i,r in source.items() if r['model_name']==m) for m in {r['model_name'] for r in source.values()}}|{67}
    for i,row in source.items():
        assert frames[i]==int(row['source_frame_index'])
        shift=(int(row['shift_y']),int(row['shift_x']));y=np.roll(images[i],shift,axis=(0,1))
        roi=np.roll(valid,shift,axis=(0,1))&ann;assert roi.sum()==int(row['roi_pixels'])
        k=int(row['template_index'])
        if k not in data:
            with np.load(OUT/f'templates/template_{k:03d}.npz') as z:data[k]=z['patterns'].astype(float)
        predictions={}
        folds=[('frozen_original',roi,roi),('full',roi,roi),('even_to_odd',roi&even,roi&~even),('odd_to_even',roi&~even,roi&even)]
        for fold,train,test in folds:
            for j,route in enumerate(ROUTES):
                raw=data[k][j]
                a,b=(float(row['signal']),float(row['background_scale'])) if fold=='frozen_original' else nuisance(y,raw,train)
                p=a*raw+b
                rms=float(np.sqrt(np.mean((p[test]-y[test])**2)))
                records.append(dict(pattern_index=i,source_frame_index=int(frames[i]),template_index=k,class_name=row['model_name'],rotation_index=int(row['rotation_index']),
                    shift_y=shift[0],shift_x=shift[1],route=route,fold=fold,signal=a,background=b,
                    train_pixels=int(train.sum()),test_pixels=int(test.sum()),test_rms=rms,test_logcorr=logcorr(y[test],p[test]),
                    test_log_mse=float(np.mean((np.log1p(p[test])-np.log1p(y[test]))**2)),
                    prediction_FFT_relative_L2=rel(p[test],(float(row['signal'])*data[k][0]+float(row['background_scale']))[test]) if fold=='frozen_original' else 0.))
                if fold=='frozen_original' and route=='point_fft':
                    error=rms/float(row['fit_rms'])-1;original_errors.append(error);assert abs(error)<1e-10,(i,error)
                    corr_error=logcorr(y[test],p[test])-float(row['logcorr'])
                    original_corr_errors.append(corr_error)
                    # Archived scores used their original float32 log transform;
                    # this independent calculation uses float64 throughout.
                    assert abs(corr_error)<1e-8,(i,corr_error)
                if fold=='full':predictions[route]=p
        if i in examples:
            np.savez_compressed(OUT/f'example_pattern_{i:03d}.npz',measured=y,roi=roi,**predictions)
    save_csv(OUT/'fit_metrics.csv',records)
    paired=[];summaries=[]
    for fold in ['frozen_original','full','even_to_odd','odd_to_even']:
        by={(int(r['pattern_index']),r['route']):r for r in records if r['fold']==fold}
        for i in range(76):
            a=by[i,'point_fft'];b=by[i,'mesh_czt']
            paired.append(dict(pattern_index=i,class_name=a['class_name'],fold=fold,
                FFT_rms=a['test_rms'],mesh_CZT_rms=b['test_rms'],RMS_change=b['test_rms']-a['test_rms'],
                FFT_logcorr=a['test_logcorr'],mesh_CZT_logcorr=b['test_logcorr'],logcorr_change=b['test_logcorr']-a['test_logcorr'],
                FFT_log_mse=a['test_log_mse'],mesh_CZT_log_mse=b['test_log_mse']))
        for scope in ['all_76',*sorted({r['model_name'] for r in source.values()})]:
            ids=[i for i in range(76) if scope=='all_76' or source[i]['model_name']==scope]
            for route in ROUTES:
                a=[by[i,route] for i in ids];fft=[by[i,'point_fft'] for i in ids]
                delta=np.array([r['test_rms']-b['test_rms'] for r,b in zip(a,fft)])
                summaries.append(dict(fold=fold,scope=scope,route=route,n=len(ids),
                    mean_RMS=float(np.mean([r['test_rms'] for r in a])),median_RMS=float(np.median([r['test_rms'] for r in a])),
                    mean_logcorr=float(np.mean([r['test_logcorr'] for r in a])),median_logcorr=float(np.median([r['test_logcorr'] for r in a])),
                    mean_log_MSE=float(np.mean([r['test_log_mse'] for r in a])),mean_RMS_change_vs_FFT=float(delta.mean()),
                    mean_logcorr_change_vs_FFT=float(np.mean([r['test_logcorr']-b['test_logcorr'] for r,b in zip(a,fft)])),
                    lower_RMS_than_FFT=int((delta<-1e-10).sum()),higher_RMS_than_FFT=int((delta>1e-10).sum())))
    save_csv(OUT/'paired_differences.csv',paired);save_csv(OUT/'summary.csv',summaries)
    twofold=[]
    for i in range(76):
        rr=[r for r in paired if r['pattern_index']==i and r['fold'] in ['even_to_odd','odd_to_even']]
        twofold.append(dict(pattern_index=i,class_name=rr[0]['class_name'],RMS_change=float(np.mean([r['RMS_change'] for r in rr])),
            logcorr_change=float(np.mean([r['logcorr_change'] for r in rr]))))
    save_csv(OUT/'paired_twofold.csv',twofold)
    dump(OUT/'summary.json',dict(status='comparison_complete_numerical_controls_pending',frames=76,templates=34,
        class_counts=dict(Counter(r['model_name'] for r in source.values())),
        original_RMS_reproduction_max_relative_difference=max(abs(x) for x in original_errors),
        original_logcorr_reproduction_max_absolute_difference=max(abs(x) for x in original_corr_errors),
        results=summaries,twofold=dict(mean_RMS_change=float(np.mean([r['RMS_change'] for r in twofold])),
            mean_logcorr_change=float(np.mean([r['logcorr_change'] for r in twofold])),lower_RMS=sum(r['RMS_change']<0 for r in twofold)),
        example_patterns=sorted(examples),geometry_refitted=False,source_hashes_unchanged=protected()==cfg['source_hashes']))
    print('EVALUATE',json.dumps([r for r in summaries if r['scope']=='all_76' and r['route'] in ['point_fft','mesh_czt']]),flush=True)

def controls():
    cfg=json.loads((OUT/'experiment_config.json').read_text());assert cfg['source_hashes']==protected()
    templates=unique_templates();chosen=set()
    for model in {r['model_name'] for r in templates.values()}:
        ids=sorted([k for k,r in templates.items() if r['model_name']==model],key=lambda k:int(templates[k]['rotation_index']))
        chosen|={ids[0],ids[-1]}
    chosen.add(int(assignments()[67]['template_index']))
    yy,xx=np.indices((128,128));rad=np.hypot(xx-63.5,yy-63.5);ann=(rad>=5)&(rad<=40)
    sample=ann&(xx%4==0)&(yy%4==0);qx,qy=detector();q=np.c_[qx[sample],qy[sample],np.zeros(sample.sum())]
    records=[]
    for k in sorted(chosen):
        row=templates[k]
        with np.load(OUT/f'templates/template_{k:03d}.npz') as z:
            rotation=z['rotation'];extent=float(z['extent_A']);original=z['patterns'][3];projection=z['mesh_projection']
        parts=mesh_parts(row['model_name']);exact=MeshBoundaryReference(parts,q).intensity(rotation)
        for nfft in [2048,8192]:
            got=czt_intensity(projection,nfft=nfft)
            records.append(dict(template_index=k,model_name=row['model_name'],control='q_lattice',dx_A=1.,nfft_equivalent=nfft,faces=sum(len(m.faces) for m,c in parts),
                intensity_reference_relative_L2=rel(got[sample],exact),difference_from_primary_relative_L2=rel(got[ann],original[ann])))
        for dx in [.5,.25]:
            n=int(round(extent/dx));p,diag=mesh_projection(parts,rotation,extent,n);got=czt_intensity(p,dx,8192)
            records.append(dict(template_index=k,model_name=row['model_name'],control='spatial_grid',dx_A=dx,nfft_equivalent=8192,faces=sum(len(m.faces) for m,c in parts),
                intensity_reference_relative_L2=rel(got[sample],exact),difference_from_primary_relative_L2=rel(got[ann],original[ann])))
        finer=mesh_parts(row['model_name'],finer=True)
        fp,diag=mesh_projection(finer,rotation,extent,int(extent*2));fine=czt_intensity(fp,.5,8192)
        ref_fine=MeshBoundaryReference(finer,q).intensity(rotation)
        records.append(dict(template_index=k,model_name=row['model_name'],control='mesh_decimation',dx_A=.5,nfft_equivalent=8192,faces=sum(len(m.faces) for m,c in finer),
            intensity_reference_relative_L2=rel(fine[sample],ref_fine),difference_from_primary_relative_L2=rel(fine[ann],original[ann])))
        save_csv(OUT/'numerical_controls.csv',records)
        print('CONTROL',k,[(r['control'],r['dx_A'],r['intensity_reference_relative_L2']) for r in records if r['template_index']==k],flush=True)
    dump(OUT/'numerical_controls.json',dict(predetermined_templates=sorted(chosen),sampled_pixels=int(sample.sum()),
        reference='Independent continuous triangle boundary integral at identical planar detector q; no mesh raster or FFT/CZT interpolation in the reference',
        mesh_scope='Spatial convergence tests use identical 12-level meshes. Finer-decimation control retains 12 density levels; it is not a density-level convergence test.',
        records=records,source_hashes_unchanged=cfg['source_hashes']==protected()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generate','evaluate','controls','all'])
    args=parser.parse_args()
    if args.stage in ['generate','all']:generate()
    if args.stage in ['evaluate','all']:evaluate()
    if args.stage in ['controls','all']:controls()
