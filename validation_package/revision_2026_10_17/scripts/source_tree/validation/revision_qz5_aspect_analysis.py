"""Audited bound-sensitivity comparisons and Ewald/rounding evidence bridge."""
from __future__ import annotations
import revision_qz5_aspect_sensitivity as a
from revision_common import ROOT,np,dump,save_csv,sha
import json,csv,gc

OUT=a.OUT
LABELS=['narrow','medium','wide']


def read_fit(label,frame):
    root=a.SOURCE/'regularized' if label=='narrow' else OUT/label
    return json.loads((root/f'frame_{frame:03d}.json').read_text())


def audit():
    config=json.loads((OUT/'experiment_config.json').read_text());data,images,_=a.context()
    family=a.ShapeFamily();fwd=a.forwards(family,data,config['quad']);full=a.forwards(family,data,config['quad'],True)
    sub=data['sample'];train=data['train'][sub];rows=[];exact_rows=[];numeric=[];errors=[];cost_errors=[]
    for frame in range(209):
        old=json.loads((a.SOURCE/f'fits/frame_{frame:03d}.json').read_text())
        family.rotation=np.asarray(old['rotation']);coarse=a.expand_primary(np.asarray(old['coarse']['theta']))
        cmu=full.model(coarse,geometry_indices=[])[0]
        coarse_metrics={k:a.metric(cmu,images[frame],data[k]) for k in ['train','validation','test']}
        for label in LABELS:
            source=read_fit(label,frame);fit=source['fit'];t=np.asarray(fit['theta'])
            lower,upper,L=a.shape_settings() if label=='narrow' else a.settings(label)
            assert np.all(t>=lower-1e-9) and np.all(t<=upper+1e-9)
            mu=full.model(t,geometry_indices=[])[0];m={k:a.metric(mu,images[frame],data[k]) for k in ['train','validation','test']}
            sample=fwd.model(t,geometry_indices=[])[0]
            residual=np.log1p(sample[train])-np.log1p(images[frame,sub][train])
            objective=float(np.mean(residual**2)+.1*np.sum((L@t)**2))
            if label!='narrow':
                for split in ['train','validation','test']:
                    for key in ['log_mse','logcorr']:
                        e=abs(m[split][key]-fit['metrics'][split][key]);errors.append(e);assert e<1e-8
                e=abs(objective-2*fit['cost']);cost_errors.append(e);assert e<1e-8
                assert fit['cost']<=source['initial_penalized_cost']+1e-10
            cap=a.corner_radii(family,t);v=family.mesh(t)[0];span=np.ptp(v,axis=0)
            assert np.max(np.abs(v[:,:2]))<119.25
            row=dict(frame=frame,range_label=label,radius_nm=cap['mean_radius_nm'],
                nominal_radius_nm=fit['radius_nm'],radius_base_nm=t[3],scale=np.exp(t[0]),
                aspect_x=np.exp(t[1]),aspect_y=np.exp(t[2]),equivalent_xy_scale=np.exp(t[0]+(t[1]+t[2])/2),
                lab_span_x_nm=span[0],lab_span_y_nm=span[1],lab_span_z_nm=span[2],
                max_cap_sphere_relative_RMS=cap['max_relative_surface_RMS'],
                within_pattern_radius_span_nm=cap['max_radius_nm']-cap['min_radius_nm'],
                angular_mode_norm=np.linalg.norm(t[7:11]),success=fit['success'],training_penalized_objective=objective,
                aspect_bound=bool(np.any(np.minimum(t[1:3]-lower[1:3],upper[1:3]-t[1:3])<1e-4)),
                aspect_x_bound=min(t[1]-lower[1],upper[1]-t[1])<1e-4,
                aspect_y_bound=min(t[2]-lower[2],upper[2]-t[2])<1e-4,
                scale_bound=min(t[0]-lower[0],upper[0]-t[0])<1e-4,
                radius_bound=min(t[3]-lower[3],upper[3]-t[3])<1e-4,
                angle_bound=bool(np.any(np.minimum(t[4:7]-lower[4:7],upper[4:7]-t[4:7])<1e-4)),
                **{f'{sp}_{key}':m[sp][key] for sp in ['train','validation','test'] for key in ['log_mse','logcorr']},
                **{f'coarse_{sp}_{key}':coarse_metrics[sp][key] for sp in ['train','validation','test'] for key in ['log_mse','logcorr']},
                old_numerics_test_log_mse=fit['metrics']['test']['log_mse'])
            rows.append(row)
            if frame in a.CASES:
                raw=abs(full.amplitude(t,False)[0])**2
                exact=a.MeshBoundaryReference(family.boundaries(t),data['qphysical']).intensity(np.eye(3))
                err=a.rel_l2(raw,exact);assert err<.003,(label,frame,err)
                exactmu=np.exp(t[-2])*exact+np.exp(t[-1])
                er=dict(frame=frame,range_label=label,intensity_relative_L2=err,
                    five_nine_intensity_relative_L2=None,
                    approximate_test_log_mse=m['test']['log_mse'],
                    exact_frozen_test_log_mse=a.metric(exactmu,images[frame],data['test'])['log_mse'])
                if label=='wide':
                    nine=a.forwards(family,data,config['quad'],True,planes=9)
                    er['five_nine_intensity_relative_L2']=a.rel_l2(raw,abs(nine.amplitude(t,False)[0])**2)
                    assert er['five_nine_intensity_relative_L2']<.001
                    del nine;gc.collect()
                exact_rows.append(er);save_csv(OUT/'exact_boundary_controls.csv',exact_rows)
            if label=='narrow' and frame in a.CASES:
                nfit=a.optimize(fwd,images[frame,sub],train,t,list(range(13)),bounds=(lower,upper),L=L,lam=.1,max_nfev=100)
                nt=np.asarray(nfit['theta']);ncap=a.corner_radii(family,nt)
                numeric.append(dict(frame=frame,source_radius_nm=cap['mean_radius_nm'],
                    new_numerics_refitted_radius_nm=ncap['mean_radius_nm'],
                    radius_change_nm=ncap['mean_radius_nm']-cap['mean_radius_nm'],success=nfit['success']))
                dump(OUT/f'narrow_numerical_controls/frame_{frame:03d}.json',dict(fit=nfit,corner_sphere_fits=ncap))
        if frame%30==0:
            print('ASPECT AUDIT',frame,'of209',flush=True);save_csv(OUT/'paired_results.csv',rows)
    save_csv(OUT/'paired_results.csv',rows);save_csv(OUT/'narrow_numerical_controls.csv',numeric)
    assert config['source_hashes']==a.hashes()
    dump(OUT/'result_audit.json',dict(status='complete',frames=209,coarse_and_three_family_predictions_recomputed=836,
        max_expansion_score_difference=max(errors),max_penalized_objective_difference=max(cost_errors),
        all_expanded_training_objectives_nonincreasing=True,independent_exact_cases_per_family=8,
        max_exact_intensity_relative_L2=max(r['intensity_relative_L2'] for r in exact_rows),
        max_five_nine_relative_L2=max(r['five_nine_intensity_relative_L2'] for r in exact_rows if r['range_label']=='wide'),
        narrow_numerical_refit_controls=numeric,max_absolute_narrow_numerical_radius_change_nm=max(abs(r['radius_change_nm']) for r in numeric),
        source_hashes_preserved=True))


def collect():
    rows=list(csv.DictReader((OUT/'paired_results.csv').open()));config=json.loads((OUT/'experiment_config.json').read_text())
    groups={label:sorted([r for r in rows if r['range_label']==label],key=lambda r:int(r['frame'])) for label in LABELS}
    for rr in groups.values():assert [int(r['frame']) for r in rr]==list(range(209))
    summary={}
    for label,rr in groups.items():
        number=lambda key:np.array([float(r[key]) for r in rr])
        rad=number('radius_nm');old=np.array([float(r['radius_nm']) for r in groups['narrow']]);change=rad-old
        metrics={}
        for sp in ['train','validation','test']:
            after=number(f'{sp}_log_mse');metrics[sp]={}
            for base in ['coarse','narrow']:
                before=number(f'coarse_{sp}_log_mse') if base=='coarse' else np.array([float(r[f'{sp}_log_mse']) for r in groups['narrow']])
                delta=before-after;boot=np.random.default_rng(20261016).choice(delta,size=(5000,209),replace=True).mean(axis=1)
                metrics[sp][f'versus_{base}']=dict(before_mean_log_mse=float(before.mean()),after_mean_log_mse=float(after.mean()),
                    loss_reduction_percent=float(100*(1-after.mean()/before.mean())),improved_frames=int((delta>1e-12).sum()),
                    worsened_frames=int((delta< -1e-12).sum()),mean_loss_reduction=float(delta.mean()),
                    descriptive_frame_bootstrap_95_interval=np.quantile(boot,[.025,.975]))
            metrics[sp]['mean_logcorr']=float(number(f'{sp}_logcorr').mean())
        summary[label]=dict(aspect_range=[.9,1.1] if label=='narrow' else a.RANGES[label],radius_nm=a.distribution(rad),
            signed_radius_change_from_narrow_nm=a.distribution(change),absolute_radius_change_from_narrow_nm=a.distribution(abs(change)),
            radius_changes_over_1_nm=int((abs(change)>1).sum()),radius_changes_over_2_nm=int((abs(change)>2).sum()),
            radius_changes_over_5_nm=int((abs(change)>5).sum()),
            geometry_distributions={key:a.distribution(number(key)) for key in ['aspect_x','aspect_y','scale','equivalent_xy_scale',
                'lab_span_x_nm','lab_span_y_nm','lab_span_z_nm','max_cap_sphere_relative_RMS']},
            bound_counts={key:sum(r[key]=='True' for r in rr) for key in ['aspect_bound','aspect_x_bound','aspect_y_bound','scale_bound','radius_bound','angle_bound']},
            converged_frames=sum(r['success']=='True' for r in rr),metrics=metrics)
        changes={}
        for key in ['lab_span_x_nm','lab_span_y_nm','lab_span_z_nm','equivalent_xy_scale']:
            baseline=np.array([float(r[key]) for r in groups['narrow']])
            changes[key]=a.distribution(100*(number(key)/baseline-1))
        dz=number('lab_span_z_nm')-np.array([float(r['lab_span_z_nm']) for r in groups['narrow']])
        summary[label]['paired_geometry_change_percent']=changes
        summary[label]['correlation_radius_change_with_lab_depth_change']=float(np.corrcoef(change,dz)[0,1]) if label!='narrow' else None
    result=dict(status='complete',frames=209,config=config,results=summary,audit=json.loads((OUT/'result_audit.json').read_text()),
        radius_definition='Mean of least-squares sphere radii over eight deformed corner caps. Effective3D descriptor; not a common local curvature radius.',
        interpretation_limits=['Radius quantiles summarize pattern fits, not confidence intervals for one common radius.',
            'Both aspect expansions are shown; neither is selected on held-out loss. This adaptive follow-up uses the existing209 patterns and old radial-band split, not fresh confirmatory data.',
            'Nested local continuations establish feasible fits, not global optima or unique3D shapes.',
            'The inherited Tikhonov term penalizes four angular shape modes. Aspect ratios have hard bounds but no additional quadratic penalty.',
            'Sphere approximation to deformed caps becomes less exact as anisotropy increases; cap RMS is reported.',
            'Held-out gains versus the rounded grid seed combine several freedoms; they cannot be attributed specifically to rounding.'],
        reproduce=['.venv_cxidb/Scripts/python.exe validation/revision_qz5_aspect_sensitivity.py qa',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_aspect_sensitivity.py medium',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_aspect_sensitivity.py wide',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_aspect_analysis.py audit',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_aspect_analysis.py depth',
            '.venv_cxidb/Scripts/python.exe validation/revision_qz5_aspect_analysis.py report'])
    if (OUT/'depth_diagnostic_summary.json').exists():result['depth_diagnostic']=json.loads((OUT/'depth_diagnostic_summary.json').read_text())
    dump(OUT/'summary.json',result);return groups,result


def depth_diagnostic():
    """Freeze xy and vary lab-frame depth; fit only signal/background on train.

    The affine identity is F_c(qx,qy,qz)=c*F(qx,qy,c*qz). At qz=0,
    the change is only c, fully absorbed by free signal. This deliberately
    probes a3D ambiguity, without reoptimizing or selecting morphology.
    """
    from types import SimpleNamespace
    from scipy.optimize import least_squares
    data,images,_=a.context();config=json.loads((OUT/'experiment_config.json').read_text())
    ratios=[.65,.8,1.,1.25,1.5];rows=[]
    dump(OUT/'depth_diagnostic_config.json',dict(cases=a.CASES,lab_z_stretch=ratios,
        geometry='Frozen widest-bound fit; stretch lab-frame z only, preserve all xy coordinates.',
        nuisance='Signal and constant background refitted only on original sampled training pixels for each displayed factor.',
        selection='Report every depth factor; no held-out selection or new structural estimate.',
        identity='F_c(qx,qy,qz)=c F(qx,qy,c qz); the qz0 intensity multiplier c^2 is absorbed exactly by free fluence.'))
    for frame in a.CASES:
        source=read_fit('wide',frame);t=np.asarray(source['fit']['theta']);family=a.ShapeFamily(np.asarray(source['rotation']))
        qp=data['qphysical'].copy();qp[:,2]=0
        plane=a.FivePlaneDifferentiable(family,data['qaxis'],qp,n=a.GRID,quad=config['quad'],extent=a.EXTENT,
            qz_low=data['qphysical'][:,2].min()*max(ratios))
        rawplane=abs(plane.amplitude(t,False)[0])**2
        train=data['train']&data['sample'];y=images[frame];norm=np.sqrt(train.sum())
        def photometric(raw,start):
            def fun(z):
                signal,bg=np.exp(z);mu=signal*raw[train]+bg
                return (np.log1p(mu)-np.log1p(y[train]))/norm
            def jac(z):
                signal,bg=np.exp(z);mu=signal*raw[train]+bg
                return np.column_stack([signal*raw[train],np.full(train.sum(),bg)])/(1+mu[:,None])/norm
            fit=least_squares(fun,start,jac=jac,bounds=([-40.,-10.],[5.,10.]),ftol=1e-11,xtol=1e-11,gtol=1e-11)
            return np.exp(fit.x[0])*raw+np.exp(fit.x[1]),fit.x
        baseline_plane,plane_t=photometric(rawplane,t[-2:])
        for factor in ratios:
            q=data['qphysical'].copy();q[:,2]*=factor
            curved=a.FivePlaneDifferentiable(family,data['qaxis'],q,n=a.GRID,quad=config['quad'],extent=a.EXTENT,
                qz_low=data['qphysical'][:,2].min()*max(ratios))
            raw=factor**2*abs(curved.amplitude(t,False)[0])**2
            mu,nu=photometric(raw,t[-2:]-[2*np.log(factor),0])
            planar_mu=np.exp(plane_t[0]-2*np.log(factor))*(factor**2*rawplane)+np.exp(plane_t[1])
            assert a.rel_l2(planar_mu,baseline_plane)<1e-12
            v=family.mesh(t)[0].copy();v[:,2]*=factor
            descriptor=a.corner_radii(SimpleNamespace(h=family.h,mesh=lambda _: (v,)),t)
            row=dict(frame=frame,depth_factor=factor,effective_radius_nm=descriptor['mean_radius_nm'],
                lab_z_span_nm=np.ptp(v[:,2]),
                planar_prediction_invariance_relative_L2=a.rel_l2(planar_mu,baseline_plane),
                affine_identity_independent_reference_L2='',
                **{f'{sp}_log_mse':a.metric(mu,y,data[sp])['log_mse'] for sp in ['train','validation','test']},
                sampled_train_log_mse=a.metric(mu,y,train)['log_mse'])
            if factor==1.5:
                exact=factor**2*a.MeshBoundaryReference(family.boundaries(t),q).intensity(np.eye(3))
                row['exact_intensity_relative_L2']=a.rel_l2(raw,exact);assert row['exact_intensity_relative_L2']<.003
                if frame==a.CASES[0]:
                    transformed=[]
                    for mesh,contrast in family.boundaries(t):
                        vv=mesh.vertices.copy();vv[:,2]*=factor;tri=vv[mesh.faces]
                        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
                        transformed.append((SimpleNamespace(vertices=vv,faces=mesh.faces,face_normals=normal),contrast))
                    explicit=a.MeshBoundaryReference(transformed,data['qphysical']).intensity(np.eye(3))
                    row['affine_identity_independent_reference_L2']=a.rel_l2(exact,explicit)
                    assert row['affine_identity_independent_reference_L2']<1e-9
            else:row['exact_intensity_relative_L2']=''
            rows.append(row);del curved
        save_csv(OUT/'depth_diagnostic.csv',rows)
        del plane;gc.collect();a.cp.get_default_memory_pool().free_all_blocks()
        print('DEPTH DIAGNOSTIC',frame,'complete',flush=True)
    summary=[]
    for frame in a.CASES:
        rr=[r for r in rows if r['frame']==frame];base=next(r for r in rr if r['depth_factor']==1.)
        summary.append(dict(frame=frame,radius_range_nm=[min(r['effective_radius_nm'] for r in rr),max(r['effective_radius_nm'] for r in rr)],
            sampled_training_loss_range=[min(r['sampled_train_log_mse'] for r in rr),max(r['sampled_train_log_mse'] for r in rr)],
            max_absolute_training_loss_change=float(max(abs(r['sampled_train_log_mse']-base['sampled_train_log_mse']) for r in rr)),
            baseline_training_loss=base['sampled_train_log_mse']))
    dump(OUT/'depth_diagnostic_summary.json',dict(status='complete',cases=summary,
        max_planar_prediction_difference=max(r['planar_prediction_invariance_relative_L2'] for r in rows),
        max_exact_intensity_relative_L2=max(float(r['exact_intensity_relative_L2']) for r in rows if r['exact_intensity_relative_L2']!=''),
        affine_identity_independent_reference_L2=next(r['affine_identity_independent_reference_L2'] for r in rows if r['affine_identity_independent_reference_L2']!=''),
        scope='Fixed-xy diagnostic, no geometry refit. Ewald sensitivity can break planar depth invariance, but its presence does not establish depth identifiability in noisy measured data.'))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups,summary=collect();colors=['#777777','#a1bed4','#276b9e'];names=['0.9–1.1','0.8–1.25','2/3–1.5']
    dump(OUT/'chart_contract.json',dict(surface='Standalone scientific PNG/PDF',cohort='209 paired pattern fits',
        panels=['Radius distributions by declared aspect bounds, median/IQR via boxplot; no confidence intervals.',
            'Paired narrow/wide radius scatter; identical scales, identity reference.',
            'Mean heldout log1p ADU MSE by bounds; zero-based grouped bar, same numerics and masks.'],
        palette='Blue plus neutral; explicit labels and open/filled marks.'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(15,5.5));vals=[np.array([float(r['radius_nm']) for r in groups[k]]) for k in LABELS]
    box=axes[0].boxplot(vals,tick_labels=names,patch_artist=True,widths=.5,
        medianprops={'color':'#222222','linewidth':1.4},flierprops={'marker':'o','markersize':3,'markerfacecolor':'white','markeredgecolor':'#666666'})
    for patch,color in zip(box['boxes'],colors):patch.set_facecolor(color);patch.set_edgecolor('#444444')
    axes[0].set_ylim(bottom=max(0,min(v.min() for v in vals)-2),top=max(v.max() for v in vals)+5)
    axes[0].set_xlabel('Allowed x/z and y/z aspect ratios');axes[0].set_ylabel('Effective corner radius (nm)')
    for j,k in enumerate(LABELS):axes[0].text(j+1,axes[0].get_ylim()[1]-.7,f'n={summary["results"][k]["bound_counts"]["aspect_bound"]} bound',ha='center',va='top',fontsize=8)
    axes[1].scatter(vals[0],vals[2],s=19,color=colors[-1],alpha=.65,edgecolors='none')
    low=min(vals[0].min(),vals[2].min())-1;high=max(vals[0].max(),vals[2].max())+1
    axes[1].plot([low,high],[low,high],color='#555555',ls='--',lw=1);axes[1].set_xlim(low,high);axes[1].set_ylim(low,high)
    axes[1].set_xlabel('Radius with 0.9–1.1 bounds (nm)');axes[1].set_ylabel('Radius with 2/3–1.5 bounds (nm)');axes[1].set_aspect('equal','box')
    x=np.arange(2);w=.24
    for j,(label,color,name) in enumerate(zip(LABELS,colors,names)):
        y=[summary['results'][label]['metrics'][sp]['versus_coarse']['after_mean_log_mse'] for sp in ['validation','test']]
        bars=axes[2].bar(x+(j-1)*w,y,width=w,color=color,edgecolor='#444444',linewidth=.6,label=name)
        axes[2].bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    axes[2].set_xticks(x,['Validation','Test']);axes[2].set_ylabel('Mean log1p ADU MSE (lower is better)');axes[2].set_ylim(0,.65);axes[2].legend(frameon=False,fontsize=9)
    fig.suptitle('Five-plane refinement: sensitivity to aspect-ratio bounds',x=.065,ha='left',fontsize=16,y=.98)
    fig.text(.065,.88,'209 patterns; fixed shape-mode Tikhonov λ=0.1; same training pixels and held-out bands; both expansions reported',fontsize=11)
    fig.text(.065,.025,'Radius is a model-dependent sphere-fit descriptor. Boxes show pattern-level median/IQR, not structural confidence intervals. “Bound” = either aspect at a limit.',fontsize=9)
    fig.subplots_adjust(left=.065,right=.99,top=.78,bottom=.18,wspace=.35)
    fig.savefig(OUT/'aspect_sensitivity.png',dpi=180);fig.savefig(OUT/'aspect_sensitivity.pdf');plt.close(fig)
    if (OUT/'depth_diagnostic.csv').exists():
        depth=list(csv.DictReader((OUT/'depth_diagnostic.csv').open()))
        fig,axes=plt.subplots(2,4,figsize=(13,7),sharex=True)
        for ax,frame in zip(axes.ravel(),a.CASES):
            rr=[r for r in depth if int(r['frame'])==frame];rr.sort(key=lambda r:float(r['depth_factor']))
            base=next(float(r['sampled_train_log_mse']) for r in rr if float(r['depth_factor'])==1.)
            ax.plot([float(r['depth_factor']) for r in rr],[float(r['sampled_train_log_mse'])-base for r in rr],
                color='#276b9e',marker='o',markersize=3,lw=1.1)
            ax.axhline(0,color='#555555',ls='--',lw=.8);ax.axvline(1,color='#999999',ls=':',lw=.8)
            radius=[float(r['effective_radius_nm']) for r in rr]
            ax.set_title(f'Pattern {frame}: R {min(radius):.1f}–{max(radius):.1f} nm',fontsize=10)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));ax.grid(axis='y',color='#e5e7eb',lw=.6)
        for ax in axes[1]:ax.set_xlabel('Lab-frame depth multiplier')
        for ax in axes[:,0]:ax.set_ylabel('Δ sampled training log-MSE')
        fig.suptitle('Ewald depth sensitivity with fixed transverse geometry',x=.07,ha='left',fontsize=15,y=.98)
        fig.text(.07,.91,'Eight predetermined fits; only signal/background refitted on training; Δ relative to depth factor1 in each panel',fontsize=10)
        fig.text(.07,.02,'For planar qz=0, changing depth is absorbed exactly by fluence. Curved sampling breaks this invariance; these are sensitivity curves, not confidence intervals.',fontsize=9)
        fig.subplots_adjust(left=.07,right=.985,top=.81,bottom=.12,wspace=.32,hspace=.42)
        fig.savefig(OUT/'depth_sensitivity.png',dpi=180);fig.savefig(OUT/'depth_sensitivity.pdf');plt.close(fig)
    write_interpretation(summary)


def write_interpretation(summary):
    bank_path=ROOT/'output/id57_qz5_rounding_2026-10-16/summary.json'
    bank=json.loads(bank_path.read_text());paired=bank['selection_details']['paired_twofold_mean']
    text=['Aspect-ratio sensitivity and the Ewald/rounding interpretation','',
        '실행 결과: 같은209개 패턴, 같은 sampled training 및 full held-out mask, angular Tikhonov lambda0.1을 사용했다.',
        'x/z, y/z aspect bound만 단계적으로 확대했다. 극단 형상에서 수치 정확도를 유지하기 위해 quadrature6, transform201-point training axis와 더 큰 동일 du 계산 영역을 사용했다.',
        '기존 좁은 범위의 최적 형상과 grid seed도 새 수치 설정에서 재평가했다. 좁은 범위8개 refit으로 수치 설정 자체의 반경 민감도를 별도 확인했다.','']
    for label in LABELS:
        s=summary['results'][label];r=s['radius_nm'];m=s['metrics']['test']['versus_coarse']
        text.append(f'{label}: aspect{s["aspect_range"]}; R median{r["median"]:.4f}nm, IQR{r["p25"]:.4f}–{r["p75"]:.4f}nm; aspect-bound{s["bound_counts"]["aspect_bound"]}/209; test MSE{m["after_mean_log_mse"]:.6f}; grid seed 대비 감소{m["loss_reduction_percent"]:.3f}%.')
    s=summary['results']['wide'];change=s['absolute_radius_change_from_narrow_nm']
    text.extend(['',f'넓은 범위에서 개별 반경의 절대 변화: median{change["median"]:.4f}nm, p90{change["p90"]:.4f}nm, max{change["max"]:.4f}nm. >1nm 변화{s["radius_changes_over_1_nm"]}개, >2nm 변화{s["radius_changes_over_2_nm"]}개.',
        f'넓은 범위에서도 scale bound에{s["bound_counts"]["scale_bound"]}개가 도달했다. 좁은 범위 대비 test loss의 추가 감소는{s["metrics"]["test"]["versus_narrow"]["loss_reduction_percent"]:.3f}%로 작고, paired mean loss reduction의 descriptive frame bootstrap interval은0을 포함한다.',
        f'반경 변화와 lab-depth 변화의 상관계수는{s["correlation_radius_change_with_lab_depth_change"]:.4f}이다. 모델의 x/y lab span 변화는 최대 약1.5%였으나 z span은 약-21%에서+34%까지 변했다. 이는 실제 깊이 측정값이 아니라 fit의 prior 민감도다.',
        '유효 반경은 변형된8개 corner cap에 구를 맞춘 반경의 평균이다. anisotropy가 있으면 한 점의 두 principal curvature나 모든 corner의 공통 실제 반경과 같지 않다.',
        '패턴별 반경 분포는 한 입자의 구조 신뢰구간이 아니다. 같은 자료로 진행한 adaptive follow-up이며 새로운 독립 실험 검증이 아니다.','',
        'Ewald 검증과 rounding 해석을 연결하는 권장 논리',
        '1. 기존 planar kernel의 구현 오차, planar 물리 근사의 오차, morphology 선택의 증거를 구분한다. Planar 구현이 수치적으로 정확해도 qz=0 물리 근사가 정확해지지는 않는다.',
        '2. 기존 fixed-mesh240-orientation Ewald 검증은 ROI 평균5.4%, 최대14%의 intensity L2 차이를 보였다. 약0.416nm^-1의5% worst-orientation cutoff는 그 모델/ROI/기준에만 해당한다. 이것을 rounded/anisotropic 모든 형상의 보편적 q 제한으로 일반화하지 않는다.',
        '3. 넓은 q에서 rounding을 해석할 근거는 곡률을 무시해도 된다는 주장 대신, 물리 Ewald wavevector에서 sharp/rounded,size,orientation 후보8400개 전체를 재생성하고 training-only로 다시 선택한 paired comparison이다.',
        f'그 비교의 twofold mean held-out log-correlation 차이는 rounded minus variable-size sharp={paired["mean_delta"]:.8f}; positive{paired["positive_frames"]}/209. 새로 선택된701개 고유 형상의 독립 경계 적분에서도 평균 차이0.04617775로 재현되었다.',
        '이는 해당 후보군 안에서의 rounded-boundary 선호를 뒷받침한다. 실제 입자가 유일하게 그 반경/3D 형상을 갖는다는 의미는 아니다. 이번 differential gain은 aspect/방향/크기/rounding/모드 변화를 함께 포함하므로 rounding만의 기여로 부르지 않는다.',
        '4. Ewald가 포함되었다는 사실과3D 반경이 식별 가능하다는 사실은 다르다. xy를 고정하고 lab z를 c배 하면 F_c(qx,qy,qz)=c F(qx,qy,c qz)이다. qz0에서는 자유 fluence가 변화를 정확히 흡수한다. Ewald sampling은 이 불변성을 깨지만, 실제 자료가 깊이를 얼마나 구속하는지는 별도 sensitivity 문제이다.',
        '5. Intensity L2의 백분율과 log-correlation 차이 또는 log-MSE 감소율은 서로 다른 지표다. 숫자 크기를 직접 비교해서 curvature를 무시해도 된다고 주장하지 않는다.','',
        '원고/SI/답변서의 범위',
        '본문3.3에는 planar 근사의 한계와 별도 broad-q curved control의 존재를 명확히 쓴다. 본문4.1의 rounding 해석은 Ewald를 포함한 재비교를 근거로 삼고 후보군에 조건부인 결론을 유지한다.',
        '5평면 구성/보간/독립 경계 검증/전체 재선택은 SI의 physical-geometry control로 둔다. 원래 planar runtime benchmark를 이 확장의 성능으로 표시하지 않는다.',
        'Differential form은 합의한 대로 Discussion의 짧은 소개와 SI 검증으로 유지한다. 범위 확대, 반경 분포, bound와 held-out 결과를 함께 제시하고, 정밀 반경 측정이라는 주장을 추가하지 않는다.',
        '다른 baseline이나 모든 public-data 결과까지 이번 sharp/rounded curvature control로 검증되었다고 확장하지 않는다.','',
        '제안 영문 핵심 문장:',
        'To assess whether the broad-q preference for rounded boundaries could arise from neglecting Ewald curvature, we regenerated the complete sharp/rounded size-orientation library at the physical detector wavevectors using five longitudinally weighted complex projections and repeated training-only selection. The held-out advantage of the rounded family persisted and was reproduced with independent continuous-boundary evaluations of the selected geometries. We interpret this result as conditional support for the boundary family, while aspect-bound sensitivity and the supplementary differentiable fits limit inference of a unique three-dimensional rounding radius.','',
        'Evidence:',str(bank_path),str(OUT/'summary.json'),str(OUT/'paired_results.csv'),str(OUT/'result_audit.json')])
    if 'depth_diagnostic' in summary:text.extend(['', 'Fixed-xy depth diagnostic: '+str(OUT/'depth_diagnostic_summary.json')])
    (OUT/'ewald_rounding_interpretation.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')


def manifest():
    assert json.loads((OUT/'experiment_config.json').read_text())['source_hashes']==a.hashes()
    scripts=[ROOT/'validation'/s for s in ['revision_qz5_aspect_sensitivity.py','revision_qz5_aspect_analysis.py']]
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='completion_manifest.json')
    dump(OUT/'completion_manifest.json',dict(status='complete',source_hashes_preserved=True,
        implementation_sources={str(p.relative_to(ROOT)):sha(p) for p in scripts},
        output_files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}))


if __name__=='__main__':
    import sys
    if sys.argv[1]=='audit':audit()
    elif sys.argv[1]=='report':plot();manifest()
    elif sys.argv[1]=='depth':depth_diagnostic()
    else:raise ValueError(sys.argv[1])
