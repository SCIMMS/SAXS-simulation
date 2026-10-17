"""Independent radius-fit physics, discretization and shape-prior controls."""
from __future__ import annotations
from revision_qz5_radius_fit import *
from revision_differentiable_real import rotations
from revision_complex_projection import ComplexProjectionRenderer


class ShapeFamily(RadiusFamily):
    """Primary topology with two aspect ratios and four known smooth modes.

    r*scale is now a nominal family parameter, not a common Euclidean curvature
    radius: anisotropy and angular deformation change local surface curvatures.
    """
    def __init__(self,rotation=np.eye(3)):
        super().__init__(rotation);self.geometry_count=11
        ref=self.h+10*self.r;phi=np.arctan2(ref[:,1],ref[:,0])
        rho2=(ref[:,0]**2+ref[:,1]**2)/np.sum(ref**2,axis=1)
        self.basis=np.column_stack([rho2*f(k*phi) for k in (4,6) for f in (np.cos,np.sin)])

    def mesh(self,t):
        base=self.h+t[3]*self.r;deform=1+self.basis@t[7:11]
        axes=np.exp([t[0]+t[1],t[0]+t[2],t[0]])
        v=base*deform[:,None]*axes;dv=np.zeros((11,len(v),3))
        dv[0]=v;dv[1,:,0]=v[:,0];dv[2,:,1]=v[:,1];dv[3]=self.r*deform[:,None]*axes
        for j in range(4):dv[7+j]=base*self.basis[:,j,None]*axes
        local,dR=rotations(t[4:7]);R=local@self.rotation
        for j in range(3):dv[4+j]=v@(dR[j]@self.rotation).T
        v=v@R.T
        for j in [0,1,2,3,7,8,9,10]:dv[j]=dv[j]@R.T
        tri=v[self.faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
        normal=np.cross(e1,e2)*self.contrast[:,None];dt=dv[:,self.faces]
        dn=(np.cross(dt[:,:,1]-dt[:,:,0],e2)+np.cross(e1,dt[:,:,2]-dt[:,:,0]))*self.contrast[None,:,None]
        volume=np.sum(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))*self.contrast)/6
        return v,dv,normal,dn,volume


def expand_primary(t):
    out=np.zeros(13);out[0]=t[0];out[3]=t[1];out[4:7]=t[2:5];out[11:13]=t[5:7]
    return out


def corner_radii(family,theta):
    """Sphere-fit descriptor of each outer corner cap, invariant to rigid pose.

    It recovers R exactly for the isotropic rounded family. For a deformed cap,
    it is an effective geometric descriptor, not every local principal radius.
    """
    import itertools
    vertices=family.mesh(theta)[0];result=[]
    for signs in itertools.product([-1.,1.],repeat=3):
        mask=np.all(np.abs(family.h-26*np.asarray(signs))<1e-8,axis=1)
        v=vertices[mask];assert len(v)>10
        center=v.mean(axis=0);w=v-center
        solution=np.linalg.lstsq(np.c_[2*w,np.ones(len(w))],np.sum(w*w,axis=1),rcond=None)[0]
        radius=np.sqrt(solution[3]+solution[:3]@solution[:3])
        residual=np.linalg.norm(w-solution[:3],axis=1)-radius
        result.append(dict(signs=signs,vertices=len(v),radius_nm=float(radius),
            relative_surface_RMS=float(np.sqrt(np.mean(residual**2))/radius)))
    return dict(corners=result,mean_radius_nm=float(np.mean([r['radius_nm'] for r in result])),
        median_radius_nm=float(np.median([r['radius_nm'] for r in result])),
        min_radius_nm=min(r['radius_nm'] for r in result),max_radius_nm=max(r['radius_nm'] for r in result),
        max_relative_surface_RMS=max(r['relative_surface_RMS'] for r in result))


def shape_settings():
    lower=np.r_[np.log(.65),np.log([.9,.9]),.05,[-.18,-.18,-.26],np.full(4,-.05),-40.,-10.]
    upper=np.r_[np.log(1.5),np.log([1.1,1.1]),25.5,[.18,.18,.26],np.full(4,.05),5.,10.]
    L=np.zeros((4,13));L[np.arange(4),np.arange(7,11)]=np.array([1,1,2.25,2.25])/.05
    return lower,upper,L


def regularized_cohort():
    config=prepare();data,images,_=context();lower,upper,L=shape_settings()
    family=ShapeFamily()
    forward=FivePlaneDifferentiable(family,data['qaxis'][::4],data['qphysical'][data['sample']],
        n=512,quad=config['quad'],extent=192.,qz_low=data['qphysical'][:,2].min())
    full=FivePlaneDifferentiable(family,data['qaxis'],data['qphysical'],n=512,
        quad=config['quad'],extent=192.,qz_low=data['qphysical'][:,2].min())
    settings=dict(frames=209,lam=.1,lower=lower,upper=upper,L=L,grid=512,extent_nm=192.,quad=config['quad'],
        initialization='Training-selected primary three-start optimum, then the original additional aspect/angular fields at zero.',
        regularization='Angular Tikhonov lambda0.1 inherited from the prior differential experiment, not retuned here; aspect ratios bounded0.9–1.1.',
        extension_reason='Primary isotropic model worsened validation loss; additional shape freedoms reduced mean validation loss on eight predetermined controls.',
        effective_radius='Fit a least-squares sphere to each of eight outer corner caps; report their mean and cap-to-sphere residuals. Under anisotropy these are descriptors, not a shared local principal curvature.',
        scope='Exploratory within-pattern prediction; no new independent experimental validation cohort.',
        source_hashes=protected_hashes())
    (OUT/'regularized').mkdir(exist_ok=True);dump(OUT/'regularized_cohort_config.json',settings)
    # Validate the effective radius estimator on a scaled/rotated ideal cap.
    # Use an exactly orthogonal base pose for the strict geometric identity.
    # Archived float32 poses have ~6e-8 orthogonality error (about2e-7 nm
    # effect on this descriptor), so they cannot meet a1e-9 nm identity gate.
    probe=RadiusFamily(np.eye(3));t=np.array([np.log(1.07),13.,.03,-.02,.04,-18,3.])
    check=corner_radii(probe,t)
    assert abs(check['mean_radius_nm']-13*1.07)<1e-9
    archived=corner_radii(RadiusFamily(data['rotation_matrices'][0]),t)
    assert abs(archived['mean_radius_nm']-13*1.07)<1e-5
    dump(OUT/'corner_radius_descriptor_qa.json',dict(status='complete',known_radius_nm=13*1.07,estimated=check,
        archived_pose_estimated=archived,archived_pose_orthogonality_error=float(np.linalg.norm(
            data['rotation_matrices'][0].T@data['rotation_matrices'][0]-np.eye(3)))))
    for frame in range(209):
        path=OUT/f'regularized/frame_{frame:03d}.json'
        if path.exists():continue
        tic=time.perf_counter();source=json.loads((OUT/f'fits/frame_{frame:03d}.json').read_text())
        chosen=next(r for r in source['starts'] if r['label']==source['chosen_start'])
        family.rotation=np.asarray(source['rotation']);initial=expand_primary(np.asarray(chosen['theta']))
        existing=OUT/f'fits/shape_frame_{frame:03d}_lambda0.1.json'
        if existing.exists():fit=json.loads(existing.read_text());reused=True
        else:
            fit=optimize(forward,images[frame,data['sample']],data['train'][data['sample']],initial,
                list(range(13)),bounds=(lower,upper),L=L,lam=.1,max_nfev=100);reused=False
            if not fit['success']:
                prior=fit
                fit=optimize(forward,images[frame,data['sample']],data['train'][data['sample']],np.asarray(prior['theta']),
                    list(range(13)),bounds=(lower,upper),L=L,lam=.1,max_nfev=160)
                fit['continuation_of']=dict(cost=prior['cost'],nfev=prior['nfev'])
        theta=np.asarray(fit['theta']);mu=full.model(theta,geometry_indices=[])[0]
        fit['metrics']={name:metric(mu,images[frame],data[name]) for name in ['train','validation','test']}
        cap=corner_radii(family,theta)
        dump(path,dict(frame=frame,rotation=family.rotation,fit=fit,corner_sphere_fits=cap,
            primary_radius_nm=chosen['radius_nm'],primary_metrics=chosen['metrics'],
            coarse_metrics=source['coarse']['metrics'],reused_control_fit=reused,seconds=time.perf_counter()-tic))
        print('REGULARIZED',frame,'effective_R',round(cap['mean_radius_nm'],4),'nominal_R',round(fit['radius_nm'],4),
            'test_MSE',round(fit['metrics']['test']['log_mse'],5),'seconds',round(time.perf_counter()-tic,2),flush=True)
    assert settings['source_hashes']==protected_hashes()
    dump(OUT/'regularized_cohort_complete.json',dict(status='complete',frames=209,lambda_fixed=.1,source_hashes_preserved=True))


def physics():
    config=prepare();data,images,_=context();rows=[];all_checks=[]
    for frame in CASES:
        source=json.loads((OUT/f'fits/frame_{frame:03d}.json').read_text())
        chosen=next(r for r in source['starts'] if r['label']==source['chosen_start'])
        t=np.asarray(chosen['theta']);family=RadiusFamily(np.asarray(source['rotation']))
        full=make_forward(family,data,config['grid'],config['quad'],full=True)
        raw=abs(full.amplitude(t,False)[0])**2
        exact=MeshBoundaryReference(family.boundaries(t),data['qphysical']).intensity(np.eye(3))
        e=rel_l2(raw,exact);assert e<.003,(frame,e)
        nine=make_forward(family,data,config['grid'],config['quad'],full=True,planes=9)
        raw9=abs(nine.amplitude(t,False)[0])**2;zerr=rel_l2(raw,raw9)
        assert zerr<.001,(frame,zerr)
        ray=ComplexProjectionRenderer(family.boundaries(t),data['qaxis'],data['qphysical'],data['mask'],n=768)
        rayraw=ray.intensity(np.eye(3));rayerr=rel_l2(rayraw,exact)
        signal,bg=np.exp(t[-2:]);mu_exact=signal*exact+bg
        rows.append(dict(frame=frame,primary_radius_nm=chosen['radius_nm'],
            surface_vs_exact_intensity_L2=e,five_vs_nine_intensity_L2=zerr,
            ray5_vs_exact_intensity_L2=rayerr,
            approximate_test_log_mse=chosen['metrics']['test']['log_mse'],
            exact_frozen_test_log_mse=metric(mu_exact,images[frame],data['test'])['log_mse']))
        # Refine on a finer representation, at unchanged training pixels and
        # starting geometry, to measure numerical radius sensitivity directly.
        fine=RadiusFamily(family.rotation,flat=12,arc=24,core_subdivisions=3)
        fine_forward=make_forward(fine,data,grid=768,quad=4)
        train=data['train'][data['sample']];y=images[frame,data['sample']]
        finefit=optimize(fine_forward,y,train,t,list(range(7)),max_nfev=100)
        rows[-1].update(finer_refit_radius_nm=finefit['radius_nm'],
            finer_radius_shift_nm=finefit['radius_nm']-chosen['radius_nm'],
            finer_refit_success=finefit['success'])
        all_checks.append(dict(frame=frame,finer_fit=finefit))
        save_csv(OUT/'physical_and_discretization_controls.csv',rows)
        dump(OUT/'finer_mesh_fits.json',all_checks)
        print('PHYSICS',frame,'FT error',e,'radius shift',rows[-1]['finer_radius_shift_nm'],flush=True)
        del full,nine,ray,fine_forward;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    dump(OUT/'physical_controls_summary.json',dict(status='complete',cases=CASES,rows=rows,
        max_surface_intensity_L2=max(r['surface_vs_exact_intensity_L2'] for r in rows),
        max_five_nine_intensity_L2=max(r['five_vs_nine_intensity_L2'] for r in rows),
        max_absolute_finer_radius_shift_nm=max(abs(r['finer_radius_shift_nm']) for r in rows),
        scope='Predetermined cases; exact references freeze fitted parameters, finer-model refits use original sampled training pixels only.'))


def shape_prior():
    config=prepare();data,images,_=context();rows=[]
    # Identical du=0.375 nm, but a larger field of view accommodates the extra
    # deformations under these explicitly local aspect/mode bounds.
    lower=np.r_[np.log(.65),np.log([.9,.9]),.05,[-.18,-.18,-.26],np.full(4,-.05),-40.,-10.]
    upper=np.r_[np.log(1.5),np.log([1.1,1.1]),25.5,[.18,.18,.26],np.full(4,.05),5.,10.]
    L=np.zeros((4,13));L[np.arange(4),np.arange(7,11)]=np.array([1,1,2.25,2.25])/.05
    family=ShapeFamily()
    forward=FivePlaneDifferentiable(family,data['qaxis'][::4],data['qphysical'][data['sample']],
        n=512,quad=config['quad'],extent=192.,qz_low=data['qphysical'][:,2].min())
    full=FivePlaneDifferentiable(family,data['qaxis'],data['qphysical'],n=512,
        quad=config['quad'],extent=192.,qz_low=data['qphysical'][:,2].min())
    settings=dict(cases=CASES,lambdas=[0.,.1],lower=lower,upper=upper,L=L,
        penalty_source='lambda=0.1 and original angular Tikhonov weights inherited from prior real-data differential experiment; no retuning on these test pixels.',
        radius='Nominal pre-deformation family parameter, not a common physical curvature radius.',
        selection='Diagnostic controls, each fit on training only. Both lambda variants reported; no test-based selection.')
    dump(OUT/'shape_prior_control_config.json',settings)
    for frame in CASES:
        source=json.loads((OUT/f'fits/frame_{frame:03d}.json').read_text())
        chosen=next(r for r in source['starts'] if r['label']==source['chosen_start'])
        family.rotation=np.asarray(source['rotation']);t=expand_primary(np.asarray(chosen['theta']))
        y=images[frame,data['sample']];train=data['train'][data['sample']]
        primary=RadiusFamily(family.rotation);primary_forward=make_forward(primary,data,config['grid'],config['quad'])
        bridge=rel_l2(abs(forward.amplitude(t,False)[0])**2,
            abs(primary_forward.amplitude(np.asarray(chosen['theta']),False)[0])**2)
        assert bridge<1e-7,(frame,bridge)
        # Full augmented Jacobian, including all four mode fields.
        if frame==CASES[0]:
            probe=t.copy();probe[1:3]=[.02,-.015];probe[7:11]=[.01,-.008,.006,.004]
            amp=forward.amplitude(probe);derivatives=[]
            for j in range(11):
                h=1e-5 if j==3 else 1e-7;plus=probe.copy();minus=probe.copy();plus[j]+=h;minus[j]-=h
                finite=(forward.amplitude(plus,False)[0]-forward.amplitude(minus,False)[0])/(2*h)
                err=rel_l2(amp[j+1],finite);derivatives.append(err)
            assert max(derivatives)<.003,derivatives
            dump(OUT/'shape_prior_gradient_qa.json',dict(relative_L2=derivatives,gate=.003,
                parameter_order='Original thirteen-parameter differential model; first eleven geometry derivatives.'))
        for lam in settings['lambdas']:
            fit=optimize(forward,y,train,t,list(range(13)),bounds=(lower,upper),L=L,lam=lam,max_nfev=100)
            theta=np.asarray(fit['theta']);mu=full.model(theta,geometry_indices=[])[0]
            fit['metrics']={name:metric(mu,images[frame],data[name]) for name in ['train','validation','test']}
            fit['lambda']=lam
            # Confirm that every quadrature point is safely inside the FoV.
            v=family.mesh(theta)[0];assert np.max(np.abs(v[:,:2]))<95.5
            dump(OUT/f'fits/shape_frame_{frame:03d}_lambda{lam:g}.json',fit)
            rows.append(dict(frame=frame,lam=lam,primary_radius_nm=chosen['radius_nm'],
                nominal_radius_nm=fit['radius_nm'],radius_base_nm=fit['radius_base_nm'],
                aspect_x=float(np.exp(theta[1])),aspect_y=float(np.exp(theta[2])),
                angular_mode_norm=float(np.linalg.norm(theta[7:11])),success=fit['success'],
                primary_test_log_mse=chosen['metrics']['test']['log_mse'],test_log_mse=fit['metrics']['test']['log_mse'],
                zero_mode_bridge_intensity_L2=bridge))
            save_csv(OUT/'shape_prior_sensitivity.csv',rows)
        print('SHAPE PRIOR',frame,'complete',flush=True)
    dump(OUT/'shape_prior_control_summary.json',dict(status='complete',config=settings,rows=rows))


if __name__=='__main__':
    import sys
    if sys.argv[1]=='physics':physics()
    elif sys.argv[1]=='shape':shape_prior()
    elif sys.argv[1]=='cohort':regularized_cohort()
    else:raise ValueError(sys.argv[1])
