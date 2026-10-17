"""Nested aspect-bound sensitivity of five-plane differential refinement.

Original results and manuscripts remain immutable. The two bound expansions,
fixed lambda and warm-start rule are declared before fitting either cohort.
"""
from __future__ import annotations
from revision_qz5_radius_controls import *
from revision_qz5_radius_report import distribution

SOURCE=OUT
OUT=ROOT/'output/qz5_aspect_sensitivity_2026-10-16'
RANGES={'medium':(.8,1.25),'wide':(2/3,1.5)}
GRID=640
EXTENT=240.


def hashes():
    result=protected_hashes()
    files=[ROOT/'validation'/s for s in ['revision_qz5_differentiable.py','revision_qz5_radius_fit.py',
        'revision_qz5_radius_controls.py','revision_qz5_radius_report.py']]
    files += [SOURCE/s for s in ['experiment_config.json','summary.json','completion_manifest.json']]
    files += sorted((SOURCE/'regularized').glob('frame_*.json'))
    files += sorted((SOURCE/'fits').glob('frame_*.json'))
    result.update({str(p.relative_to(ROOT)):sha(p) for p in files})
    return result


def settings(label):
    lower,upper,L=shape_settings();a,b=RANGES[label]
    lower[1:3]=np.log(a);upper[1:3]=np.log(b)
    return lower,upper,L


def forwards(family,data,quad,full=False,planes=5):
    sub=np.ones(len(data['qphysical']),bool) if full else data['sample']
    axis=data['qaxis'] if full else data['qaxis'][::2]
    return FivePlaneDifferentiable(family,axis,data['qphysical'][sub],n=GRID,
        quad=quad,extent=EXTENT,planes=planes,qz_low=data['qphysical'][:,2].min())


def qa():
    OUT.mkdir(parents=True,exist_ok=True);data,_,_=context();records=[]
    family=ShapeFamily();small=FivePlaneDifferentiable(family,data['qaxis'][::2],
        data['qphysical'][data['sample']],n=512,quad=4,extent=192.,qz_low=data['qphysical'][:,2].min())
    large=forwards(family,data,4)
    t=expand_primary(np.array([np.log(1.02),12,.013,-.017,.021,-18.,3.]))
    t[1:3]=np.log([1.09,.91]);t[7:11]=[.006,-.003,.002,.001]
    bridge=rel_l2(abs(large.amplitude(t,False)[0])**2,abs(small.amplitude(t,False)[0])**2)
    assert bridge<1e-7,bridge
    del small,large;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    for quad in [4,6]:
        fwd=forwards(family,data,quad);errors=[];extent_checks=[]
        probes=[(0,.8,1.25,8.,1.02),(69,2/3,1.5,16.,1.15),
            (138,1.5,1.5,.1,1.5),(208,1.5,2/3,24.,1.)]
        for oi,ax,ay,r,s in probes:
            family.rotation=data['rotation_matrices'][oi]
            t=expand_primary(np.array([np.log(s),r,.013,-.017,.021,-18.,3.]))
            t[1:3]=np.log([ax,ay]);t[7:11]=[.006,-.003,.002,.001]
            raw=abs(fwd.amplitude(t,False)[0])**2
            exact=MeshBoundaryReference(family.boundaries(t),fwd.qphysical).intensity(np.eye(3))
            errors.append(rel_l2(raw,exact));extent_checks.append(float(np.abs(family.mesh(t)[0][:,:2]).max()))
        amp=fwd.amplitude(t);grad=[]
        for j in range(11):
            h=1e-5 if j==3 else 1e-7;plus=t.copy();minus=t.copy();plus[j]+=h;minus[j]-=h
            finite=(fwd.amplitude(plus,False)[0]-fwd.amplitude(minus,False)[0])/(2*h)
            grad.append(rel_l2(amp[j+1],finite))
        rec=dict(quad=quad,intensity_relative_L2=errors,gradient_relative_L2=grad,
            probe_max_transverse_nm=extent_checks,passes=max(errors)<.001 and max(grad)<.003)
        records.append(rec);print('ASPECT QA',json.dumps(rec),flush=True)
        dump(OUT/'numerical_qa_progress.json',records)
        if rec['passes']:break
        del fwd;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    assert records[-1]['passes'],'Improve numerical settings before measured-data expansion.'
    bound=26*np.sqrt(3)*1.5*1.5*(1+.1*np.sqrt(2))
    assert bound<EXTENT/2-2*EXTENT/GRID
    dump(OUT/'numerical_qa.json',dict(status='complete',grid=GRID,extent_nm=EXTENT,du_nm=EXTENT/GRID,
        chosen_quad=records[-1]['quad'],records=records,old_grid_bridge_intensity_L2=bridge,
        worst_bound_vertex_norm_nm=float(bound),FoV_half_width_nm=EXTENT/2,
        rationale='Same0.375nm gridding interval, integer-aligned larger field of view; bound covers all feasible scale/aspect/angular parameters.'))
    config=dict(status='prepared',frames=209,aspect_ranges=RANGES,lambda_fixed=.1,grid=GRID,extent_nm=EXTENT,
        quad=records[-1]['quad'],planes=5,training_transform_axis_stride=2,training_pixel_stride=4,
        aspect_definition='Pre-rotation x/z and y/z affine scale ratios, relative to z-axis global scale.',
        unchanged='Radius, scale, local-rotation, angular-mode, fluence/background bounds; mesh/core/density; angular Tikhonov; original sampled training and full evaluation masks.',
        numerical_change='Increase quadrature4 to selected passing order and transform-axis sampling101 to201 points to cover extreme expanded geometries. Pixel selection is unchanged; separately refit narrow-bound controls to quantify numerical sensitivity.',
        initialization='Medium starts at saved narrow optimum; wide starts at medium training optimum. Both nested ranges are reported, without held-out selection.',
        selection='One local continuation for each bound expansion, with up to160 additional evaluations for incomplete convergence. Minimum penalized training objective retained, including the source optimum.',
        validation='All full-ROI metrics recalculated; eight predetermined exact-boundary controls per range; within-pattern exploratory radial-band holdout.',
        source_hashes=hashes())
    path=OUT/'experiment_config.json'
    if path.exists():
        from revision_common import safe
        assert json.loads(path.read_text())==safe(config),'Do not mutate an established configuration.'
    else:dump(path,config)


def run(label):
    config=json.loads((OUT/'experiment_config.json').read_text());data,images,_=context()
    assert config['source_hashes']==hashes();lower,upper,L=settings(label)
    family=ShapeFamily();fwd=forwards(family,data,config['quad']);full=forwards(family,data,config['quad'],True)
    sub=data['sample'];train=data['train'][sub];dest=OUT/label;dest.mkdir(exist_ok=True)
    for frame in range(209):
        path=dest/f'frame_{frame:03d}.json'
        if path.exists():continue
        tic=time.perf_counter()
        source_path=SOURCE/f'regularized/frame_{frame:03d}.json' if label=='medium' else OUT/f'medium/frame_{frame:03d}.json'
        source=json.loads(source_path.read_text());initial=np.asarray(source['fit']['theta'])
        family.rotation=np.asarray(source['rotation'])
        mu0=fwd.model(initial,geometry_indices=[])[0]
        r=np.log1p(mu0[train])-np.log1p(images[frame,sub][train])
        initial_cost=.5*(np.mean(r*r)+.1*np.sum((L@initial)**2))
        fit=optimize(fwd,images[frame,sub],train,initial,list(range(13)),bounds=(lower,upper),L=L,lam=.1,max_nfev=100)
        if not fit['success']:
            prior=fit
            extra=optimize(fwd,images[frame,sub],train,np.asarray(fit['theta']),list(range(13)),
                bounds=(lower,upper),L=L,lam=.1,max_nfev=160)
            fit=extra if extra['cost']<=prior['cost'] else prior
            fit['continuation_of']=dict(cost=prior['cost'],nfev=prior['nfev'])
        fallback=False
        if fit['cost']>initial_cost+1e-10:
            fallback=True;fit=dict(source['fit']);fit['cost']=float(initial_cost)
        assert fit['cost']<=initial_cost+1e-10
        theta=np.asarray(fit['theta']);mu=full.model(theta,geometry_indices=[])[0]
        fit['metrics']={k:metric(mu,images[frame],data[k]) for k in ['train','validation','test']}
        cap=corner_radii(family,theta);v=family.mesh(theta)[0];assert np.max(np.abs(v[:,:2]))<119.25
        dump(path,dict(frame=frame,range_label=label,rotation=family.rotation,fit=fit,corner_sphere_fits=cap,
            initial_penalized_cost=initial_cost,retained_initial=fallback,source_result=str(source_path.relative_to(ROOT)),
            maximum_transverse_extent_nm=float(np.max(np.abs(v[:,:2]))),seconds=time.perf_counter()-tic))
        print(label,frame,'R',round(cap['mean_radius_nm'],4),'aspects',np.round(np.exp(theta[1:3]),4).tolist(),
            'test',round(fit['metrics']['test']['log_mse'],5),'s',round(time.perf_counter()-tic,2),flush=True)
    assert config['source_hashes']==hashes()
    dump(OUT/f'{label}_complete.json',dict(status='complete',frames=209,source_hashes_preserved=True))


if __name__=='__main__':
    import sys
    if sys.argv[1]=='qa':qa()
    elif sys.argv[1] in RANGES:run(sys.argv[1])
    else:raise ValueError(sys.argv[1])
