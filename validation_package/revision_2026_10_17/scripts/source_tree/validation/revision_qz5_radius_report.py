"""Audited conditional radius estimates and exportable research figures."""
from __future__ import annotations
from revision_qz5_radius_fit import *


def distribution(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),
        p10=float(np.quantile(a,.1)),p25=float(np.quantile(a,.25)),median=float(np.median(a)),
        p75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)))


def collect():
    config=prepare();data,images,_=context();rows=[];starts=[];rr=[]
    for frame in range(209):
        source=json.loads((OUT/f'fits/frame_{frame:03d}.json').read_text());rr.append(source)
        chosen=min(source['starts'],key=lambda r:r['cost'])
        assert chosen['label']==source['chosen_start']
        t=np.asarray(chosen['theta']);physical=t[1]*np.exp(t[0]);side=52*np.exp(t[0])
        assert abs(physical-chosen['radius_nm'])<1e-10
        assert abs(physical/(side/2)-t[1]/26)<1e-12
        entry=dict(frame=frame,chosen_start=chosen['label'],radius_nm=physical,radius_base_nm=t[1],
            side_nm=side,scale=float(np.exp(t[0])),radius_to_half_side=t[1]/26,success=chosen['success'],
            seed_base_radius_nm=float(source['seed']['base_radius_nm']),seed_scale=float(source['seed']['scale']),
            base_radius_at_upper_bound=t[1]>25.4999,base_radius_at_lower_bound=t[1]<.0501,
            scale_at_bound=abs(t[0]-LOW[0])<1e-4 or abs(t[0]-HIGH[0])<1e-4,
            base_radius_above_previous_grid=t[1]>12+1e-6,
            all_start_radius_span_nm=source['start_radius_range_nm'][1]-source['start_radius_range_nm'][0],
            training_sample_log_mse=2*chosen['cost'],
            **{f'{stage}_{split}_{key}':fit['metrics'][split][key]
               for stage,fit in [('coarse',source['coarse']),('refined',chosen)]
               for split in ['train','validation','test'] for key in ['log_mse','logcorr']})
        for gap in [.001,.005,.01]:
            near=[f for f in source['starts'] if 2*(f['cost']-chosen['cost'])<=gap]
            entry[f'near_training_gap_{gap:g}_radius_span_nm']=max(f['radius_nm'] for f in near)-min(f['radius_nm'] for f in near)
        rows.append(entry)
        for fit in source['starts']:
            starts.append(dict(frame=frame,label=fit['label'],radius_nm=fit['radius_nm'],
                radius_base_nm=fit['radius_base_nm'],training_sample_log_mse=2*fit['cost'],success=fit['success'],
                selected=fit['label']==chosen['label'],test_log_mse=fit['metrics']['test']['log_mse']))
    save_csv(OUT/'radius_estimates.csv',rows);save_csv(OUT/'all_initializations.csv',starts)
    metrics={}
    for split in ['train','validation','test']:
        base=np.array([r[f'coarse_{split}_log_mse'] for r in rows])
        after=np.array([r[f'refined_{split}_log_mse'] for r in rows]);delta=base-after
        rng=np.random.default_rng(20261016);boot=rng.choice(delta,size=(5000,len(delta)),replace=True).mean(axis=1)
        metrics[split]=dict(coarse_mean_log_mse=float(base.mean()),refined_mean_log_mse=float(after.mean()),
            relative_mean_loss_reduction_percent=float(100*(1-after.mean()/base.mean())),
            improved_frames=int((delta>0).sum()),worsened_frames=int((delta<0).sum()),
            mean_loss_reduction=float(delta.mean()),descriptive_frame_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
            coarse_mean_logcorr=float(np.mean([r[f'coarse_{split}_logcorr'] for r in rows])),
            refined_mean_logcorr=float(np.mean([r[f'refined_{split}_logcorr'] for r in rows])))
    result=dict(status='complete',frames=209,
        physical_radius_nm=distribution([r['radius_nm'] for r in rows]),
        base_radius_nm=distribution([r['radius_base_nm'] for r in rows]),
        side_nm=distribution([r['side_nm'] for r in rows]),
        radius_to_half_side=distribution([r['radius_to_half_side'] for r in rows]),
        upper_radius_bound_frames=sum(r['base_radius_at_upper_bound'] for r in rows),
        lower_radius_bound_frames=sum(r['base_radius_at_lower_bound'] for r in rows),
        scale_bound_frames=sum(r['scale_at_bound'] for r in rows),
        base_radius_above_previous_grid_frames=sum(r['base_radius_above_previous_grid'] for r in rows),
        converged_frames=sum(r['success'] for r in rows),metrics=metrics,
        chosen_start_counts={name:sum(r['chosen_start']==name for r in rows) for name in ['seed','round18','round24p5']},
        start_sensitivity={str(gap):dict(
            descriptor='Radius span among sampled starts within the stated absolute training log-MSE gap; not a confidence interval.',
            frames_with_radius_span_over_1_nm=sum(r[f'near_training_gap_{gap:g}_radius_span_nm']>1 for r in rows),
            max_radius_span_nm=max(r[f'near_training_gap_{gap:g}_radius_span_nm'] for r in rows)) for gap in [.001,.005,.01]})
    dump(OUT/'primary_summary.json',result)
    return rows,result


def audit():
    config=prepare();data,images,_=context();sub=data['sample'];train=data['train'][sub]
    family=RadiusFamily();fwd=make_forward(family,data,config['grid'],config['quad'])
    full=make_forward(family,data,config['grid'],config['quad'],full=True)
    errors=[];loss_errors=[]
    for frame in range(209):
        source=json.loads((OUT/f'fits/frame_{frame:03d}.json').read_text())
        family.rotation=np.asarray(source['rotation'])
        chosen=min(source['starts'],key=lambda r:r['cost']);assert chosen['label']==source['chosen_start']
        for label,fit in [('coarse',source['coarse']),('refined',chosen)]:
            t=np.asarray(fit['theta']);sample=fwd.model(t,geometry_indices=[])[0]
            r=np.log1p(sample[train])-np.log1p(images[frame,sub][train])
            err=abs(float(np.mean(r*r))-2*fit['cost']);assert err<1e-8,(frame,label,err)
            loss_errors.append(err)
            mu=full.model(t,geometry_indices=[])[0]
            for split in ['train','validation','test']:
                independent=metric(mu,images[frame],data[split])
                for key in ['log_mse','logcorr']:
                    e=abs(independent[key]-fit['metrics'][split][key]);assert e<1e-8,(frame,label,split,key,e)
                    errors.append(e)
        if frame%40==0:print('COHORT AUDIT',frame,'of209',flush=True)
    assert config['source_hashes']==protected_hashes()
    result=dict(status='complete',frames=209,coarse_and_selected_predictions_recomputed=418,
        score_comparisons=len(errors),max_score_difference=max(errors),
        max_training_objective_difference=max(loss_errors),all_start_choices_training_only=True,
        source_hashes_preserved=True)
    dump(OUT/'cohort_result_audit.json',result)


def collect_regularized():
    from revision_qz5_radius_controls import shape_settings
    lower,upper,L=shape_settings();rows=[]
    for frame in range(209):
        source=json.loads((OUT/f'regularized/frame_{frame:03d}.json').read_text())
        fit=source['fit'];t=np.asarray(fit['theta']);cap=source['corner_sphere_fits']
        rows.append(dict(frame=frame,effective_corner_radius_nm=cap['mean_radius_nm'],primary_radius_nm=source['primary_radius_nm'],
            min_corner_radius_nm=cap['min_radius_nm'],max_corner_radius_nm=cap['max_radius_nm'],
            max_cap_sphere_relative_RMS=cap['max_relative_surface_RMS'],nominal_radius_nm=fit['radius_nm'],
            radius_base_nm=t[3],scale=np.exp(t[0]),aspect_x=np.exp(t[1]),aspect_y=np.exp(t[2]),
            angular_mode_norm=np.linalg.norm(t[7:11]),success=fit['success'],
            radius_bound=min(t[3]-lower[3],upper[3]-t[3])<1e-4,
            aspect_bound=bool(np.any(np.minimum(t[1:3]-lower[1:3],upper[1:3]-t[1:3])<1e-4)),
            angle_bound=bool(np.any(np.minimum(t[4:7]-lower[4:7],upper[4:7]-t[4:7])<1e-4)),
            **{f'{stage}_{split}_{key}':fit0[split][key]
               for stage,fit0 in [('coarse',source['coarse_metrics']),('primary',source['primary_metrics']),('regularized',fit['metrics'])]
               for split in ['train','validation','test'] for key in ['log_mse','logcorr']}))
    save_csv(OUT/'regularized_radius_estimates.csv',rows);metrics={}
    for split in ['train','validation','test']:
        metrics[split]={}
        for base in ['coarse','primary']:
            before=np.array([r[f'{base}_{split}_log_mse'] for r in rows])
            after=np.array([r[f'regularized_{split}_log_mse'] for r in rows]);delta=before-after
            boot=np.random.default_rng(20261016).choice(delta,size=(5000,209),replace=True).mean(axis=1)
            metrics[split][f'versus_{base}']=dict(before_mean_log_mse=float(before.mean()),
                regularized_mean_log_mse=float(after.mean()),relative_mean_loss_reduction_percent=float(100*(1-after.mean()/before.mean())),
                improved_frames=int((delta>0).sum()),worsened_frames=int((delta<0).sum()),
                mean_loss_reduction=float(delta.mean()),descriptive_frame_bootstrap_95_interval=np.quantile(boot,[.025,.975]),
                before_mean_logcorr=float(np.mean([r[f'{base}_{split}_logcorr'] for r in rows])),
                regularized_mean_logcorr=float(np.mean([r[f'regularized_{split}_logcorr'] for r in rows])))
    result=dict(status='complete',frames=209,lambda_fixed=.1,
        effective_corner_radius_nm=distribution([r['effective_corner_radius_nm'] for r in rows]),
        nominal_family_radius_nm=distribution([r['nominal_radius_nm'] for r in rows]),
        maximum_cap_sphere_relative_RMS=distribution([r['max_cap_sphere_relative_RMS'] for r in rows]),
        within_pattern_corner_radius_span_nm=distribution([r['max_corner_radius_nm']-r['min_corner_radius_nm'] for r in rows]),
        effective_minus_primary_radius_nm=distribution([r['effective_corner_radius_nm']-r['primary_radius_nm'] for r in rows]),
        converged_frames=sum(r['success'] for r in rows),radius_bound_frames=sum(r['radius_bound'] for r in rows),
        aspect_bound_frames=sum(r['aspect_bound'] for r in rows),angle_bound_frames=sum(r['angle_bound'] for r in rows),metrics=metrics,
        interpretation='Mean of eight unweighted vertex-sphere fits per outer corner cap. Describes this model, not a common local principal radius or a model-independent structural measurement.',
        selection='One local augmented fit initialized from the three-start primary training winner. Fixed inherited lambda0.1; all parameters fit on training only.',
        limitation='Full-cohort extension followed an eight-case validation diagnostic after primary test scores had been examined; exploratory, not a fresh confirmatory test.')
    dump(OUT/'regularized_summary.json',result)
    return rows,result


def audit_regularized():
    from revision_qz5_radius_controls import ShapeFamily,corner_radii,shape_settings
    config=prepare();data,images,_=context();lower,upper,L=shape_settings()
    family=ShapeFamily();sub=data['sample'];train=data['train'][sub]
    fwd=FivePlaneDifferentiable(family,data['qaxis'][::4],data['qphysical'][sub],n=512,quad=4,extent=192.,qz_low=data['qphysical'][:,2].min())
    full=FivePlaneDifferentiable(family,data['qaxis'],data['qphysical'],n=512,quad=4,extent=192.,qz_low=data['qphysical'][:,2].min())
    errors=[];cost_errors=[];cap_errors=[];exact_rows=[]
    for frame in range(209):
        source=json.loads((OUT/f'regularized/frame_{frame:03d}.json').read_text());fit=source['fit']
        family.rotation=np.asarray(source['rotation']);t=np.asarray(fit['theta'])
        assert np.all(t>=lower-1e-10) and np.all(t<=upper+1e-10)
        v=family.mesh(t)[0];assert np.max(np.abs(v[:,:2]))<95.5
        sample=fwd.model(t,geometry_indices=[])[0]
        residual=np.log1p(sample[train])-np.log1p(images[frame,sub][train])
        cost=np.mean(residual**2)+.1*np.sum((L@t)**2)
        cost_errors.append(abs(cost-2*fit['cost']));assert cost_errors[-1]<1e-8
        mu=full.model(t,geometry_indices=[])[0]
        for split in ['train','validation','test']:
            m=metric(mu,images[frame],data[split])
            for key in ['log_mse','logcorr']:
                errors.append(abs(m[key]-fit['metrics'][split][key]));assert errors[-1]<1e-8
        cap=corner_radii(family,t)
        cap_errors.append(abs(cap['mean_radius_nm']-source['corner_sphere_fits']['mean_radius_nm']));assert cap_errors[-1]<1e-9
        if frame in CASES:
            raw=abs(full.amplitude(t,False)[0])**2
            exact=MeshBoundaryReference(family.boundaries(t),data['qphysical']).intensity(np.eye(3))
            e=rel_l2(raw,exact);assert e<.003,(frame,e)
            exactmu=np.exp(t[-2])*exact+np.exp(t[-1])
            exact_rows.append(dict(frame=frame,surface_vs_exact_intensity_L2=e,
                approximate_test_log_mse=fit['metrics']['test']['log_mse'],
                exact_frozen_test_log_mse=metric(exactmu,images[frame],data['test'])['log_mse']))
            save_csv(OUT/'regularized_exact_controls.csv',exact_rows)
        if frame%40==0:print('REGULARIZED AUDIT',frame,'of209',flush=True)
    assert config['source_hashes']==protected_hashes()
    dump(OUT/'regularized_result_audit.json',dict(status='complete',frames=209,full_predictions_recomputed=209,
        max_score_difference=max(errors),max_training_penalized_objective_difference=max(cost_errors),
        max_corner_descriptor_difference_nm=max(cap_errors),all_geometry_within_bounds_and_FoV=True,
        independent_deformed_exact_controls=exact_rows,max_exact_intensity_L2=max(r['surface_vs_exact_intensity_L2'] for r in exact_rows),
        source_hashes_preserved=True))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows,primary=collect();regularized_rows,regularized=collect_regularized();config=prepare()
    data,images,_=context();qnorm=np.linalg.norm(data['qphysical'],axis=1)
    primary['status']='complete'
    controls=json.loads((OUT/'physical_controls_summary.json').read_text())
    shape=json.loads((OUT/'shape_prior_control_summary.json').read_text())
    profiles=list(csv.DictReader((OUT/'radius_profiles.csv').open()))
    profile_summary=[]
    for frame in CASES:
        rr=[r for r in profiles if int(r['frame'])==frame]
        source=json.loads((OUT/f'fits/frame_{frame:03d}.json').read_text())
        selected=next(r for r in source['starts'] if r['label']==source['chosen_start'])
        minimum=min(2*selected['cost'],min(float(r['training_log_mse']) for r in rr))
        near=[float(r['fixed_radius_base_nm']) for r in rr if float(r['training_log_mse'])<=minimum+.005]
        profile_summary.append(dict(frame=frame,
            profiled_training_minimum=minimum,primary_training_loss=2*selected['cost'],
            profile_below_primary_by=2*selected['cost']-minimum,
            tested_base_radii_within_0p005_loss=near,
            tested_base_radius_span_within_0p005_loss_nm=max(near)-min(near) if near else 0.))
    shape_summary={}
    for lam in [0.,.1]:
        rr=[r for r in shape['rows'] if r['lam']==lam]
        shifts=np.array([r['nominal_radius_nm']-r['primary_radius_nm'] for r in rr])
        shape_summary[str(lam)]=dict(cases=len(rr),nominal_radius_shift_nm=distribution(shifts),
            primary_mean_test_log_mse=float(np.mean([r['primary_test_log_mse'] for r in rr])),
            augmented_mean_test_log_mse=float(np.mean([r['test_log_mse'] for r in rr])),
            converged=sum(r['success'] for r in rr),
            caveat='The augmented model has varying local curvature; nominal r*scale is a shape-family coordinate, not a common Euclidean radius.')
    dump(OUT/'chart_contract.json',dict(surface='scientific PNG/PDF, Matplotlib',
        primary=dict(families=['distribution','relationship'],grain='209 training-selected conditional pattern fits',
            axes=['physical radius R=r0*s in nm','side L=52*s in nm'],palette='blue plus neutral median/sphere references',
            claim='Shows pattern-level fitted radii and model-imposed geometric limits, not uncertainty of one common particle radius.'),
        profiles=dict(family='small-multiple sensitivity curves',grain='eight predetermined patterns',
            x='fixed base rounding radius r0 in nm',y='training log-MSE above the lowest tested/primary solution',
            semantics='Scale, local orientation and nuisance refitted; no confidence threshold or held-out selection.'),
        regularized=dict(families=['distribution','grouped comparison'],grain='209 pattern fits, fixed lambda0.1',
            radius='Mean of eight unweighted corner-cap sphere fits; model descriptor in nm.',
            prediction='Mean per-pattern log1p ADU MSE, same train/validation/test masks across stages; zero-based axis.',
            interpretation='Exploratory within-pattern validation; no physical radius confidence interval.')))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'text.color':'#222222','axes.labelcolor':'#222222',
        'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(12,5.2));values=np.array([r['radius_nm'] for r in rows])
    axes[0].hist(values,bins=18,color='#b7d4ec',edgecolor='#444444',linewidth=.7)
    axes[0].axvline(np.median(values),color='#276b9e',lw=1.5,label=f'Median {np.median(values):.2f} nm')
    axes[0].set_xlabel('Physical rounding radius R = r₀s (nm)');axes[0].set_ylabel('Number of patterns');axes[0].legend(frameon=False)
    side=np.array([r['side_nm'] for r in rows]);bound=np.array([r['base_radius_at_upper_bound'] for r in rows])
    axes[1].scatter(side[~bound],values[~bound],s=19,color='#276b9e',alpha=.65,edgecolors='none',label='Interior radius')
    axes[1].scatter(side[bound],values[bound],s=24,facecolors='none',edgecolors='#276b9e',label='Upper radius bound')
    xx=np.linspace(side.min()-1,side.max()+1,50);axes[1].plot(xx,xx/2,color='#555555',ls='--',lw=1,label='Sphere limit R = L/2')
    axes[1].set_xlabel('Outer side length L = 52s (nm)');axes[1].set_ylabel('Physical radius R (nm)');axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Conditional rounding-radius estimates with five-plane refinement',x=.07,ha='left',fontsize=15,y=.98)
    fig.text(.07,.89,'209 patterns; isotropic rounded-cube/core-shell family; three starts selected only by training log-MSE',fontsize=10)
    d=primary['physical_radius_nm'];fig.text(.07,.025,f'Median {d["median"]:.2f} nm; IQR {d["p25"]:.2f}–{d["p75"]:.2f} nm. Spread combines pattern variation, noise, model mismatch and fit uncertainty.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,top=.79,bottom=.18,wspace=.28)
    fig.savefig(OUT/'radius_summary.png',dpi=180);fig.savefig(OUT/'radius_summary.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(13,7),sharex=True)
    for ax,frame in zip(axes.ravel(),CASES):
        rr=sorted([r for r in profiles if int(r['frame'])==frame],key=lambda r:float(r['fixed_radius_base_nm']))
        one=next(r for r in profile_summary if r['frame']==frame)
        ax.plot([float(r['fixed_radius_base_nm']) for r in rr],
            [float(r['training_log_mse'])-one['profiled_training_minimum'] for r in rr],
            marker='o',ms=3,color='#276b9e',lw=1.1)
        main=rows[frame];ax.axvline(main['radius_base_nm'],color='#555555',ls='--',lw=.9)
        ax.set_title(f'Pattern {frame}: R = {main["radius_nm"]:.1f} nm',fontsize=10)
        ax.set_ylim(bottom=0);ax.grid(axis='y',color='#e5e7eb',lw=.6)
    for ax in axes[1]:ax.set_xlabel('Fixed base radius r₀ (nm)')
    for ax in axes[:,0]:ax.set_ylabel('Δ training log-MSE')
    fig.suptitle('Training-loss sensitivity to rounding',x=.07,ha='left',fontsize=15,y=.98)
    fig.text(.07,.91,'Eight predetermined patterns; size/orientation/signal/background reoptimized at each fixed r₀; dashed: primary r₀',fontsize=10)
    fig.text(.07,.02,'These are sensitivity profiles on a finite grid, not calibrated confidence intervals. The physical radius is R = r₀s.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,top=.83,bottom=.12,wspace=.29,hspace=.38)
    fig.savefig(OUT/'radius_profiles.png',dpi=180);fig.savefig(OUT/'radius_profiles.pdf');plt.close(fig)
    # Same cohort and objective for all three model stages; zero-based loss axis.
    fig,axes=plt.subplots(1,2,figsize=(12,5.4))
    vals=np.array([r['effective_corner_radius_nm'] for r in regularized_rows]);d=regularized['effective_corner_radius_nm']
    axes[0].hist(vals,bins=18,color='#b7d4ec',edgecolor='#444444',linewidth=.7)
    axes[0].axvline(d['median'],color='#276b9e',lw=1.5,label=f'Median {d["median"]:.2f} nm')
    axes[0].set_xlabel('Effective corner radius (nm)');axes[0].set_ylabel('Number of patterns');axes[0].legend(frameon=False)
    stages=['coarse','primary','regularized'];colors=['#666666','#a1bed4','#276b9e']
    xx=np.arange(3);width=.24
    for j,(stage,label) in enumerate(zip(stages,['Grid selection','Isotropic refinement','Shape + Tikhonov'])):
        values=[np.mean([r[f'{stage}_{sp}_log_mse'] for r in regularized_rows]) for sp in ['train','validation','test']]
        bars=axes[1].bar(xx+(j-1)*width,values,width=width,color=colors[j],edgecolor='#444444',linewidth=.6,label=label)
        axes[1].bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    axes[1].set_xticks(xx,['Training','Validation','Test']);axes[1].set_ylabel('Mean log1p ADU MSE (lower is better)')
    axes[1].set_ylim(0,max(ax.get_height() for ax in axes[1].patches)*1.32);axes[1].legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Regularized five-plane refinement: radius and prediction',x=.07,ha='left',fontsize=15,y=.98)
    fig.text(.07,.89,'209 patterns; fixed angular Tikhonov λ = 0.1; radius = mean of sphere fits to eight deformed corner caps',fontsize=10)
    fig.text(.07,.025,f'Effective radius median {d["median"]:.2f} nm; IQR {d["p25"]:.2f}–{d["p75"]:.2f} nm. Exploratory within-pattern holdouts; radius is model dependent.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,top=.79,bottom=.18,wspace=.28)
    fig.savefig(OUT/'regularized_radius_summary.png',dpi=180);fig.savefig(OUT/'regularized_radius_summary.pdf');plt.close(fig)
    scripts=[ROOT/'validation'/s for s in ['revision_qz5_differentiable.py','revision_qz5_radius_fit.py',
        'revision_qz5_radius_controls.py','revision_qz5_radius_report.py']]
    summary=dict(status='complete',primary=primary,regularized=regularized,config=config,
        data_domain=dict(source='CXIDB57 existing209-pattern cohort and five-plane bank',
            q_magnitude_nm_inverse=[float(qnorm.min()),float(qnorm.max())],
            qz_nm_inverse=[float(data['qphysical'][:,2].min()),float(data['qphysical'][:,2].max())],
            full_ROI_pixels=len(qnorm),sampled_fitting_pixels=int(np.sum(data['train']&data['sample'])),
            full_evaluation_pixels={k:int(data[k].sum()) for k in ['train','validation','test']},
            detector_ADU='Existing nonnegative-clipped ADU data, log1p least squares; no calibrated photon-noise likelihood.'),
        numerical_qa=json.loads((OUT/'numerical_qa.json').read_text()),
        synthetic_recovery=json.loads((OUT/'synthetic_recovery.json').read_text()),
        cohort_result_audit=json.loads((OUT/'cohort_result_audit.json').read_text()),
        regularized_result_audit=json.loads((OUT/'regularized_result_audit.json').read_text()),
        physical_controls=controls,profile_sensitivity=profile_summary,shape_prior_controls=shape_summary,
        uncertainty=['Reported quantiles describe fitted radii across different patterns; they are not confidence intervals for one common radius.',
            'Three-start local fitting is not a proof of global optimality. Profiles and additional starts probe only predetermined cases.',
            'Unknown ADU noise, density/background/illumination mismatch and fixed core-shell assumptions can bias physical interpretation.',
            'A single Euclidean radius is defined only for the primary isotropic rounded family. Added shape freedoms have a nominal family coordinate and varying local curvatures.',
            'Validation/test bands were not used to choose initializations, radius bounds or the synthetic/numerical settings. These are within-pattern exploratory holdouts, not a new independent experiment.',
            'The primary isotropic model worsened test loss. Added shape freedoms and inherited regularization were extended after validation diagnostics, with primary test scores already visible; this is exploratory.',
            'The regularized radius is a cap-sphere descriptor. Frequent aspect bounds, residual non-sphericity, initialization and structural assumptions limit its physical interpretation.'],
        source_hashes_preserved=config['source_hashes']==protected_hashes(),
        implementation_sources={str(p.relative_to(ROOT)):sha(p) for p in scripts},
        reproduce=['.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_fit.py qa',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_fit.py synthetic',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_fit.py all',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_fit.py profiles',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_controls.py physics',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_controls.py shape',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_controls.py cohort',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_report.py audit',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_report.py audit_regularized',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_radius_report.py report'])
    assert summary['source_hashes_preserved']
    dump(OUT/'summary.json',summary)
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='completion_manifest.json')
    fitfiles=sorted((OUT/'fits').glob('*'))+sorted((OUT/'regularized').glob('*'))
    dump(OUT/'completion_manifest.json',dict(status='complete',frames=209,primary_starts_per_frame=3,
        source_hashes_preserved=True,implementation_sources=summary['implementation_sources'],
        output_files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files+fitfiles}))
    print('REPORT COMPLETE:209 patterns; primary and regularized results, independent audits and figures saved.',flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='collect':collect()
    elif sys.argv[1]=='audit':audit()
    elif sys.argv[1]=='audit_regularized':audit_regularized()
    elif sys.argv[1]=='report':report()
    else:raise ValueError(sys.argv[1])
