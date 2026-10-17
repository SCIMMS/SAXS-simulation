"""Five-plane differentiable radius estimation, training-only real-data fitting."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,csv,time,gc
from pathlib import Path
from revision_common import ROOT,np,sha,dump,save_csv,rel_l2
from revision_qz5_differentiable import RadiusFamily,FivePlaneDifferentiable,NAMES,LOW,HIGH
from revision_gpu_boundary_general import MeshBoundaryReference
from scipy.optimize import least_squares,nnls
import cupy as cp

OUT=ROOT/'output/qz5_differentiable_radius_2026-10-16'
BANK=ROOT/'output/id57_qz5_rounding_2026-10-16'
CASES=[0,29,59,89,119,149,179,208]


def context():
    with np.load(BANK/'data_masks.npz') as f:data={k:f[k] for k in f.files}
    y=np.load(BANK/'measured.npy',mmap_mode='r')
    yy,xx=np.indices(data['mask'].shape)
    data['sample']=((xx%4==0)&(yy%4==0))[data['mask']]
    band=np.floor((data['radius_px']-45)/10).astype(int)
    data['train']=band%2==0;data['validation']=band%4==1;data['test']=band%4==3
    assert np.array_equal(data['train'],data['even'])
    assert np.all(data['train'].astype(int)+data['validation']+data['test']==1)
    seeds={int(r['frame']):r for r in csv.DictReader((BANK/'selection_results.csv').open())
           if r['fold']=='even_to_odd' and r['family']=='rounded'}
    return data,y,seeds


def protected_hashes():
    files=[ROOT/'validation'/s for s in ['revision_complex_projection.py','revision_differentiable_real.py',
        'revision_differentiable_synthetic.py','revision_polyhedron_reference.py','revision_gpu_boundary_general.py']]
    files += [BANK/'library_config.json',BANK/'selection_results.csv',BANK/'data_masks.npz',BANK/'measured.npy']
    files += list((ROOT/'output/revision_manuscript_R12_2026-10-15').glob('*.docx'))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def theta_from_seed(seed):
    return np.array([np.log(float(seed['scale'])),float(seed['base_radius_nm']),0,0,0,
        -18.,np.log(max(float(seed['background']),.01))])


def make_forward(family,data,grid=384,quad=4,full=False,planes=5):
    sub=np.ones(len(data['qphysical']),bool) if full else data['sample']
    axis=data['qaxis'] if full else data['qaxis'][::4]
    return FivePlaneDifferentiable(family,axis,data['qphysical'][sub],n=grid,quad=quad,
        planes=planes,qz_low=data['qphysical'][:,2].min())


def qa():
    OUT.mkdir(exist_ok=True,parents=True);data,y,seeds=context();records=[]
    hashes=protected_hashes();tic=time.perf_counter()
    for grid,quad in [(384,4),(512,4),(768,4)]:
        family=RadiusFamily();forward=make_forward(family,data,grid,quad)
        errors=[];gradients=[]
        for case,(oi,radius) in enumerate([(0,4.),(69,16.),(138,24.)]):
            family.rotation=data['rotation_matrices'][oi]
            theta=np.array([np.log(1.02),radius,.013,-.017,.021,-18.,3.])
            amp=forward.amplitude(theta)
            truth=MeshBoundaryReference(family.boundaries(theta),forward.qphysical).intensity(np.eye(3))
            error=rel_l2(abs(amp[0])**2,truth);errors.append(error)
            if case==1:
                for j in range(5):
                    step=1e-5 if j==1 else 1e-7
                    plus=theta.copy();minus=theta.copy();plus[j]+=step;minus[j]-=step
                    finite=(forward.amplitude(plus,False)[0]-forward.amplitude(minus,False)[0])/(2*step)
                    gradients.append(dict(parameter=NAMES[j],relative_L2=rel_l2(amp[j+1],finite)))
        timings=[]
        for _ in range(3):
            start=time.perf_counter();forward.amplitude(theta);cp.cuda.Device().synchronize()
            timings.append(time.perf_counter()-start)
        record=dict(grid=grid,quad=quad,faces=len(family.faces),
            forward_relative_L2=errors,gradient_errors=gradients,
            median_amplitude_Jacobian_seconds=float(np.median(timings)),
            passes=max(errors)<.003 and max(r['relative_L2'] for r in gradients)<.003)
        records.append(record);dump(OUT/'numerical_qa_progress.json',records)
        print('QA',json.dumps(record),flush=True)
        del forward,family;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    eligible=[r for r in records if r['passes']]
    assert eligible,'No numerical configuration passed; improve quadrature/grid before fitting.'
    best=min(eligible,key=lambda r:r['median_amplitude_Jacobian_seconds'])
    assert hashes==protected_hashes()
    dump(OUT/'numerical_qa.json',dict(status='complete',records=records,selected=best,
        forward_gate=.003,gradient_gate=.003,source_hashes=hashes,seconds=time.perf_counter()-tic,
        selection='Smallest measured cost among configurations passing independent forward and finite-difference gates; no measured-data held-out scores used.'))


def metric(mu,y,mask):
    a=np.log1p(mu[mask]);b=np.log1p(y[mask]);da=a-a.mean();db=b-b.mean()
    den=np.linalg.norm(da)*np.linalg.norm(db)
    return dict(log_mse=float(np.mean((a-b)**2)),logcorr=float(da@db/den) if den>1e-20 else 0.,
                adu_relative_L2=rel_l2(mu[mask],y[mask]))


def initialize_nuisance(forward,y,train,theta):
    theta=theta.copy();raw=abs(forward.amplitude(theta,False)[0])**2
    normalizer=raw[train].mean()
    coefficient,_=nnls(np.c_[raw[train]/normalizer,np.ones(train.sum())],y[train])
    theta[-2:]=np.log([max(coefficient[0]/normalizer,1e-17),max(coefficient[1],1e-4)])
    return np.clip(theta,LOW+1e-8,HIGH-1e-8)


def optimize(forward,y,train,start,indices,max_nfev=80,fixed_radius=None,bounds=None,L=None,lam=0.):
    lower,upper=(LOW,HIGH) if bounds is None else bounds
    start=np.clip(start,lower+1e-8,upper-1e-8);idx=np.asarray(indices,int)
    if fixed_radius is not None:start[1]=fixed_radius
    geometry=[j for j in indices if j<len(start)-2];cache={};history=[];norm=np.sqrt(train.sum())
    def expand(z):
        theta=start.copy();theta[idx]=z;return theta
    def evaluate(z):
        if 'z' not in cache or not np.array_equal(z,cache['z']):
            theta=expand(z);mu,jac=forward.model(theta,geometry_indices=geometry)
            r=(np.log1p(mu[train])-np.log1p(y[train]))/norm
            J=jac[:,train].T/(1+mu[train,None])/norm
            if L is not None and lam:
                r=np.r_[r,np.sqrt(lam)*(L@theta)];J=np.vstack([J,np.sqrt(lam)*L])
            cache.update(z=z.copy(),r=r,J=J[:,idx]);history.append(float(r@r))
        return cache
    tic=time.perf_counter()
    opt=least_squares(lambda z:evaluate(z)['r'],start[idx],jac=lambda z:evaluate(z)['J'],
        bounds=(lower[idx],upper[idx]),x_scale='jac',max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    theta=expand(opt.x)
    radius_index=1 if len(theta)==7 else 3
    return dict(theta=theta,success=bool(opt.success),status=int(opt.status),nfev=opt.nfev,
        cost=float(opt.cost),optimality=float(opt.optimality),seconds=time.perf_counter()-tic,
        history=history,radius_base_nm=float(theta[radius_index]),radius_nm=float(theta[radius_index]*np.exp(theta[0])),
        side_nm=float(52*np.exp(theta[0])),radius_ratio_to_half_side=float(theta[radius_index]/26),
        bound_parameter_names=[(NAMES[j] if len(theta)==7 else str(j)) for j in idx
                               if min(theta[j]-lower[j],upper[j]-theta[j])<1e-4])


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    numerical=json.loads((OUT/'numerical_qa.json').read_text());assert numerical['status']=='complete'
    best=numerical['selected'];data,images,seeds=context()
    config=dict(status='prepared',grid=best['grid'],quad=best['quad'],flat=6,arc=12,core_subdivisions=2,
        planes=5,extent_nm=144.,parameter_names=NAMES,bounds=dict(lower=LOW,upper=HIGH),
        cohort=209,training_stride=4,
        seed='New five-plane 8400-candidate rounded-family winner, selected on original even radial bands only.',
        starts=['training-selected radius','base radius 18 nm','base radius24.5 nm'],
        preflight_revision='Independent near-sphere synthetic target exposed local minima for4/12/22nm starts;18/24.5nm controls added before any measured-data optimization.',
        start_selection='Minimum training log-MSE; validation/test never select starts.',
        split='Even 10-pixel radial bands train; odd 1 mod4 validation; odd 3 mod4 test, full detector for evaluation.',
        objective='Mean squared log1p ADU residual on sampled training pixels; free nonnegative signal/background.',
        primary_shape='Isotropically scaled, rounded cube plus fixed clipped-octahedral core. Radius is a common Euclidean corner/edge radius before faceting.',
        radius_definition='Physical r_nm = radius_base_nm*exp(log_scale); normalized rounding = radius_base_nm/26.',
        primary_regularization='No angular/aspect shape modes in the primary interpretable-radius family; no radius penalty.',
        natural_radius_limit='0 <= base radius <= 26 nm at sphere limit; optimization capped at25.5 to retain nondegenerate face patches and fixed core containment.',
        sensitivity_cases=CASES,profile_base_radii_nm=[.25,4.,8.,12.,16.,20.,24.,25.5],
        profile_description='Reoptimize scale, three local rotation angles and signal/background at fixed radius on training only; descriptors, not calibrated confidence intervals.',
        source_hashes=protected_hashes())
    path=OUT/'experiment_config.json'
    if path.exists():
        from revision_common import safe
        if json.loads(path.read_text())!=safe(config):
            assert not list((OUT/'fits').glob('frame_*.json')),'Never change a running measured-data configuration.'
            archive=OUT/'experiment_config_preflight_v0.json'
            if not archive.exists():archive.write_bytes(path.read_bytes())
            dump(path,config)
    else:dump(path,config)
    return config


def synthetic():
    config=prepare();data,_,_=context();sub=data['sample'];train=data['train'][sub]
    family=RadiusFamily();forward=make_forward(family,data,config['grid'],config['quad'])
    truth_family=RadiusFamily(flat=12,arc=24,core_subdivisions=3)
    rows=[]
    for oi,radius in [(0,4.),(69,16.),(138,24.)]:
        family.rotation=data['rotation_matrices'][oi];truth_family.rotation=family.rotation
        true=np.array([np.log(.97),radius,.02,-.015,.03,-18.,np.log(15.)])
        truth=MeshBoundaryReference(truth_family.boundaries(true),forward.qphysical).intensity(np.eye(3))
        a=20./truth[train].mean();y=a*truth+15.;truth_radius=radius*.97
        fits=[]
        for rstart in [4.,12.,18.,24.5]:
            start=np.array([np.log(1.02),rstart,0,0,0,-18.,np.log(15.)])
            start=initialize_nuisance(forward,y,train,start)
            fit=optimize(forward,y,train,start,list(range(7)),max_nfev=100)
            fit['start_radius_base_nm']=rstart;fits.append(fit)
        chosen=min(fits,key=lambda r:r['cost'])
        error=abs(chosen['radius_nm']-truth_radius)
        assert error<.30,(radius,error)
        row=dict(true_base_radius_nm=radius,true_radius_nm=truth_radius,
            recovered_radius_nm=chosen['radius_nm'],absolute_radius_error_nm=error,
            chosen_start_base_radius_nm=chosen['start_radius_base_nm'],
            recovered_scale=float(np.exp(chosen['theta'][0])),fits=fits)
        rows.append(row);dump(OUT/'synthetic_recovery_progress.json',rows)
        print('SYNTHETIC radius',radius,'recovered physical',chosen['radius_nm'],'error',error,flush=True)
    dump(OUT/'synthetic_recovery.json',dict(status='complete',cases=rows,
        maximum_radius_error_nm=max(r['absolute_radius_error_nm'] for r in rows),
        target='Noise-free independent continuous boundary integral on finer arc24/core-subdivision3 meshes; fitted family arc12/core-subdivision2.',
        selection='Minimum training objective across all starts; no held-out selection.',source_hashes_preserved=config['source_hashes']==protected_hashes()))


def run(frames):
    config=prepare();data,images,seeds=context();sub=data['sample'];train=data['train'][sub]
    family=RadiusFamily();forward=make_forward(family,data,config['grid'],config['quad'])
    full=make_forward(family,data,config['grid'],config['quad'],full=True)
    (OUT/'fits').mkdir(exist_ok=True)
    for frame in frames:
        path=OUT/f'fits/frame_{frame:03d}.json'
        if path.exists():continue
        tic=time.perf_counter();seed=seeds[frame]
        family.rotation=data['rotation_matrices'][int(seed['orientation_index'])]
        y=np.asarray(images[frame,sub]);base=theta_from_seed(seed)
        initial=initialize_nuisance(forward,y,train,base)
        coarse=optimize(forward,y,train,initial,[5,6],max_nfev=50)
        fits=[]
        for label,rstart in [('seed',float(seed['base_radius_nm'])),('round18',18.),('round24p5',24.5)]:
            start=base.copy();start[1]=rstart
            start=initialize_nuisance(forward,y,train,start)
            nu=optimize(forward,y,train,start,[5,6],max_nfev=50)
            fit=optimize(forward,y,train,np.asarray(nu['theta']),list(range(7)),max_nfev=80)
            if not fit['success']:
                previous=fit
                fit=optimize(forward,y,train,np.asarray(previous['theta']),list(range(7)),max_nfev=120)
                fit['continuation_of']=dict(nfev=previous['nfev'],cost=previous['cost'],seconds=previous['seconds'])
            fit['label']=label;fit['initial_radius_base_nm']=rstart;fits.append(fit)
        chosen=min(fits,key=lambda r:r['cost'])
        predictions={}
        for label,fit in [('coarse',coarse)]+[(r['label'],r) for r in fits]:
            theta=np.asarray(fit['theta']);mu=full.model(theta,geometry_indices=[])[0]
            fit['metrics']={name:metric(mu,images[frame],data[name]) for name in ['train','validation','test']}
            fit['training_sample_log_mse']=2*fit['cost']
            predictions[label]=mu
        source=dict(frame=frame,seed=seed,rotation=family.rotation,coarse=coarse,
            starts=fits,chosen_start=chosen['label'],selection='minimum training objective only',
            chosen_radius_nm=chosen['radius_nm'],chosen_radius_base_nm=chosen['radius_base_nm'],
            start_radius_range_nm=[min(f['radius_nm'] for f in fits),max(f['radius_nm'] for f in fits)],
            seconds=time.perf_counter()-tic)
        dump(path,source)
        # Predetermined examples only; all other predictions are reproducible
        # from parameters without storing another large detector library.
        if frame in CASES:np.savez_compressed(path.with_suffix('.npz'),**predictions)
        print('FIT',frame,'radius_nm',round(chosen['radius_nm'],4),'base',round(chosen['radius_base_nm'],4),
            'start_spread',round(source['start_radius_range_nm'][1]-source['start_radius_range_nm'][0],4),
            'test_MSE',round(chosen['metrics']['test']['log_mse'],5),
            'success',chosen['success'],'seconds',round(source['seconds'],2),flush=True)
    assert config['source_hashes']==protected_hashes()


def profiles():
    config=prepare();data,images,seeds=context();sub=data['sample'];train=data['train'][sub]
    family=RadiusFamily();forward=make_forward(family,data,config['grid'],config['quad'])
    rows=[]
    for frame in CASES:
        source=json.loads((OUT/f'fits/frame_{frame:03d}.json').read_text())
        family.rotation=np.asarray(source['rotation'])
        best=next(f for f in source['starts'] if f['label']==source['chosen_start'])
        y=np.asarray(images[frame,sub]);start=np.array(best['theta'])
        for radius in config['profile_base_radii_nm']:
            t=start.copy();t[1]=radius;t=initialize_nuisance(forward,y,train,t)
            fit=optimize(forward,y,train,t,[0,2,3,4,5,6],fixed_radius=radius,max_nfev=80)
            rows.append(dict(frame=frame,fixed_radius_base_nm=radius,
                radius_nm=fit['radius_nm'],scale=float(np.exp(fit['theta'][0])),
                training_log_mse=2*fit['cost'],success=fit['success'],nfev=fit['nfev']))
        save_csv(OUT/'radius_profiles.csv',rows)
        # A deliberately smaller third start checks local minima without using
        # radius profiles or held-out loss to pick the primary cohort result.
        t=start.copy();t[1]=4.;t[0]=theta_from_seed(seeds[frame])[0]
        t[2:5]=[.02,-.02,.025];t=initialize_nuisance(forward,y,train,t)
        third=optimize(forward,y,train,t,list(range(7)),max_nfev=120)
        dump(OUT/f'fits/third_start_{frame:03d}.json',dict(frame=frame,fit=third,
            primary_radius_nm=best['radius_nm'],primary_training_log_mse=2*best['cost']))
        print('PROFILE',frame,'done',flush=True)
    dump(OUT/'profile_audit.json',dict(status='complete',cases=CASES,profile_points=len(rows),
        selection='Predetermined evenly spaced/end-point frames; scale/orientation/nuisance reoptimized at fixed radius on training only.',
        uncertainty='Training loss sensitivity curves, not calibrated statistical confidence intervals.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['qa','synthetic','pilot','all','profiles']);args=parser.parse_args()
    if args.stage=='qa':qa()
    elif args.stage=='synthetic':synthetic()
    elif args.stage=='pilot':run(CASES)
    elif args.stage=='all':run(range(209))
    elif args.stage=='profiles':profiles()
