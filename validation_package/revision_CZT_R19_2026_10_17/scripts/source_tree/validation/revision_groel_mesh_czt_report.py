"""Independent saved-array audit and publication-style plots of matched paths."""
from __future__ import annotations
import csv,json,sys,platform,importlib.metadata
from pathlib import Path
import revision_groel_mesh_czt_matched as m
from revision_polyhedron_reference import polyhedron_amplitude
np=m.np;OUT=m.OUT

def rows(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
def independent_fit(y,x,mask):
    y=y[mask];x=x[mask];norm=float(np.mean(x));z=x/norm
    denom=float(np.sum((z-z.mean())**2))
    a=float(np.sum((z-z.mean())*(y-y.mean()))/denom);b=float(y.mean()-a*z.mean())
    opts=[(max(float(z@y/(z@z)),0.),0.),(0.,max(float(y.mean()),0.))]
    if a>=0 and b>=0:opts.append((a,b))
    a,b=min(opts,key=lambda ab:float(np.mean((ab[0]*z+ab[1]-y)**2)))
    return a/norm,b

def independent_corr(y,p):
    y=np.log1p(y);p=np.log1p(p);y-=y.mean();p-=p.mean()
    return float(y@p/np.sqrt((y@y)*(p@p)))

def audit():
    cfg=json.loads((OUT/'experiment_config.json').read_text());assert cfg['source_hashes']==m.protected()
    source=m.assignments();fit=rows('fit_metrics.csv');assert len(fit)==76*4*4
    assert len({(r['pattern_index'],r['route'],r['fold']) for r in fit})==len(fit)
    by={(int(r['pattern_index']),r['route'],r['fold']):r for r in fit}
    with m.h5py.File(m.HITS) as h:images=h['patterns_ph'][:].astype(float);valid=h['mask'][:].astype(bool)
    yy,xx=np.indices((128,128));radius=np.hypot(xx-63.5,yy-63.5);ann=(radius>=5)&(radius<=40)
    even=(np.floor((radius-5)/5).astype(int)%2)==0
    maximum=dict(rms=0.,logcorr=0.,log_mse=0.,prediction_relative_L2=0.)
    cache={}
    for i,row in source.items():
        shift=(int(row['shift_y']),int(row['shift_x']));y=np.roll(images[i],shift,axis=(0,1));roi=np.roll(valid,shift,axis=(0,1))&ann
        k=int(row['template_index'])
        if k not in cache:
            with np.load(OUT/f'templates/template_{k:03d}.npz') as z:cache[k]=z['patterns'].astype(float)
        for fold,train,test in [('frozen_original',roi,roi),('full',roi,roi),('even_to_odd',roi&even,roi&~even),('odd_to_even',roi&~even,roi&even)]:
            for j,route in enumerate(m.ROUTES):
                record=by[i,route,fold];x=cache[k][j]
                assert record['class_name']==row['model_name'] and int(record['rotation_index'])==int(row['rotation_index'])
                assert int(record['shift_y'])==shift[0] and int(record['shift_x'])==shift[1]
                a,b=(float(row['signal']),float(row['background_scale'])) if fold=='frozen_original' else independent_fit(y,x,train)
                p=a*x+b;record_prediction=float(record['signal'])*x+float(record['background'])
                maximum['prediction_relative_L2']=max(maximum['prediction_relative_L2'],m.rel(p[test],record_prediction[test]))
                calculations=dict(rms=float(np.sqrt(np.mean((p[test]-y[test])**2))),
                    logcorr=independent_corr(y[test],p[test]),log_mse=float(np.mean((np.log1p(p[test])-np.log1p(y[test]))**2)))
                for key,value in calculations.items():maximum[key]=max(maximum[key],abs(value-float(record['test_'+key])))
                assert int(record['test_pixels'])==test.sum() and int(record['train_pixels'])==train.sum()
    assert max(maximum.values())<1e-10,maximum
    qx,qy=m.detector();sample=ann&(xx%4==0)&(yy%4==0);q=np.c_[qx[sample],qy[sample],np.zeros(sample.sum())]
    numerical=[];cpu=[];seen=set()
    for k,row in m.unique_templates().items():
        with np.load(OUT/f'templates/template_{k:03d}.npz') as z:
            projection=z['mesh_projection'];rotation=z['rotation'];primary=z['patterns'][3]
        parts=m.mesh_parts(row['model_name']);ref=m.MeshBoundaryReference(parts,q).intensity(rotation)
        dense=m.czt_intensity(projection,nfft=8192)
        numerical.append(dict(template_index=k,class_name=row['model_name'],q_samples=int(sample.sum()),
            primary_error_relative_L2=m.rel(primary[sample],ref),dense_q_error_relative_L2=m.rel(dense[sample],ref)))
        if row['model_name'] not in seen:
            probe=q[np.linspace(0,len(q)-1,11).astype(int)]
            gpu=m.MeshBoundaryReference(parts,probe).intensity(rotation)
            amp=sum(c*polyhedron_amplitude(probe@rotation,mesh) for mesh,c in parts)
            error=m.rel(gpu,np.abs(amp)**2);assert error<1e-7,error
            cpu.append(dict(template_index=k,class_name=row['model_name'],CPU_GPU_reference_relative_L2=error,points=len(probe)))
            seen.add(row['model_name'])
        print('AUDIT_REFERENCE',k,numerical[-1]['primary_error_relative_L2'],numerical[-1]['dense_q_error_relative_L2'],flush=True)
    m.save_csv(OUT/'all_template_reference_audit.csv',numerical)
    qa=rows('template_qa.csv')
    result=dict(status='pass',all_frames=76,all_templates=34,saved_fit_records_checked=len(fit),independent_calculation_max_difference=maximum,
        source_hashes_unchanged=m.protected()==cfg['source_hashes'],CPU_reference_crosschecks=cpu,
        original_source_RMS_error=json.loads((OUT/'summary.json').read_text())['original_RMS_reproduction_max_relative_difference'],
        point_CZT_FFT_max_relative_L2=max(float(r['point_CZT_FFT_relative_L2']) for r in qa),
        mesh_CZT_FFT_max_relative_L2=max(float(r['mesh_CZT_FFT_relative_L2']) for r in qa),
        primary_sparse_reference_max_relative_L2=max(r['primary_error_relative_L2'] for r in numerical),
        dense_q_sparse_reference_max_relative_L2=max(r['dense_q_error_relative_L2'] for r in numerical),
        raster_diagnostics=dict(total_bad_balance_pixels=sum(int(r['bad_balance_pixels']) for r in qa),
            max_bad_balance_pixels_per_template=max(int(r['bad_balance_pixels']) for r in qa),
            negative_clamped_pixels=sum(int(r['negative_clamped_pixels']) for r in qa),all_cases_retained=True),
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,fft_amplitude_dtype=str(np.fft.fft2(np.zeros((8,8),np.float32)).dtype),
            packages={name:importlib.metadata.version(name) for name in ['scipy','h5py','matplotlib','cupy-cuda12x','vtk']}))
    m.dump(OUT/'independent_audit.json',result)
    print('INDEPENDENT_AUDIT',json.dumps(result),flush=True)

def figures_and_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['status']=='pass'
    summary=json.loads((OUT/'summary.json').read_text());table=rows('summary.csv');qa=rows('template_qa.csv');paired=rows('paired_differences.csv');two=rows('paired_twofold.csv')
    models=['B_tetradecamer_D7','B_tetradecamer_D7_polyprism_R35_N14','B_tetradecamer_D7_polyprism_R55_N14']
    names=['B','R35','R55'];markers=['o','s','^'];colors=['#2B6CB0','#B7791F','#444444']
    m.dump(OUT/'chart_contract.json',dict(question='With fixed class, quaternion, alignment and detector conditions, how much changes when switching point/FFT to mesh/CZT?',
        surface='Standalone static scientific PNG and SVG; reproducible Matplotlib code',
        cohort_plot=dict(grain='34 unique templates for intensity differences; 76 fixed experimental frames for residual comparisons',panels=['Class-wise normalized template intensity error','Paired full-ROI RMS change versus baseline RMS','Histogram of per-frame two-fold mean correlation change']),
        examples=dict(selection='Earliest pattern index in each historical class, plus historical illustration 67; no score-based selection',patterns=summary['example_patterns']),
        palette='Blue and gold plus neutral, three distinct markers; zero lines and labels supplement color',
        caveats='Historical selections used full ROI; fixed-class conditional comparison, no new assignment or Ewald correction.'))
    fig,axs=plt.subplots(1,3,figsize=(11.8,3.7))
    for j,(model,label,marker,color) in enumerate(zip(models,names,markers,colors)):
        subset=[r for r in qa if r['model_name']==model]
        values=np.array([float(r['mesh_point_mean_normalized_relative_L2'])*100 for r in subset])
        offsets=np.linspace(-.13,.13,len(values))
        axs[0].scatter(j+offsets,values,marker=marker,s=25,color=color,label=label,alpha=.85)
        rs=[r for r in paired if r['fold']=='full' and r['class_name']==model]
        axs[1].scatter([float(r['FFT_rms']) for r in rs],[float(r['RMS_change']) for r in rs],marker=marker,s=25,color=color,label=label,alpha=.8)
    axs[0].set_xticks(range(3),names);axs[0].set_ylim(bottom=0);axs[0].set_ylabel('Mean-normalized intensity relative L2 / %');axs[0].set_xlabel('Fixed template family (34 templates)')
    axs[1].axhline(0,color='.4',lw=.8);axs[1].set_xlabel('Point / FFT full-ROI RMS / processed count');axs[1].set_ylabel('RMS change / processed count');axs[1].legend(frameon=False,fontsize=8)
    values=np.array([float(r['logcorr_change']) for r in two]);counts,edges,_=axs[2].hist(values,bins=18,color='#2B6CB0',edgecolor='white')
    assert counts.sum()==76
    axs[2].axvline(0,color='.4',lw=.8);axs[2].set_xlabel('Two-fold mean log-correlation change');axs[2].set_ylabel('Frames')
    for ax,label in zip(axs,['(a)','(b)','(c)']):ax.text(.02,.86 if label=='(a)' else .98,label,transform=ax.transAxes,va='top',fontweight='bold');ax.grid(alpha=.12)
    fig.suptitle('Fixed GroEL templates: matched computation paths',y=1.01,fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'matched_paths_summary.png',dpi=240,bbox_inches='tight');fig.savefig(OUT/'matched_paths_summary.svg',bbox_inches='tight');plt.close(fig)
    ids=summary['example_patterns'];fig,axs=plt.subplots(len(ids),4,figsize=(11.5,2.65*len(ids)),squeeze=False)
    for row,i in enumerate(ids):
        with np.load(OUT/f'example_pattern_{i:03d}.npz') as z:
            roi=z['roi'];images=[np.log1p(z[k]) for k in ['measured','point_fft','mesh_czt']];difference=images[2]-images[1]
        vmax=float(np.quantile(images[0][roi],.995));lim=float(np.max(np.abs(difference[roi])))
        for col,a in enumerate([*images,difference]):
            shown=np.where(roi,a,np.nan)
            im=axs[row,col].imshow(shown,origin='lower',extent=[-64,64,-64,64],cmap='RdBu_r' if col==3 else 'viridis',vmin=-lim if col==3 else 0,vmax=lim if col==3 else vmax)
            fig.colorbar(im,ax=axs[row,col],shrink=.75,pad=.025)
            axs[row,col].set_xlim(-42,42);axs[row,col].set_ylim(-42,42)
            axs[row,col].set_xlabel('Detector x / pixel');axs[row,col].set_ylabel('Detector y / pixel' if col==0 else '')
            axs[row,col].set_title((f'Pattern {i}: ' if col==0 else '')+['Measured log1p','Point / FFT','Mesh / CZT','Mesh minus FFT (log1p)'][col],fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'fixed_examples.png',dpi=210,bbox_inches='tight');plt.close(fig)
    def result(fold,route):return next(r for r in table if r['fold']==fold and r['route']==route and r['scope']=='all_76')
    full_a=result('full','point_fft');full_b=result('full','mesh_czt');frozen=result('frozen_original','mesh_czt')
    qa_shapes=np.array([float(r['mesh_point_mean_normalized_relative_L2']) for r in qa])
    charge=np.array([float(r['mesh_to_point_charge_relative_difference']) for r in qa])
    worst=sorted([r for r in paired if r['fold']=='full'],key=lambda r:float(r['RMS_change']),reverse=True)[:5]
    report=f'''Fixed GroEL class/orientation path comparison — 17 October 2026

Assessment: Share with caveats. All requested 76 frames and 34 unique selected templates were evaluated. Class, recorded quaternion, template identity, integer detector shift, mask, detector qx/qy, annulus and scoring objective remain fixed. No geometry refinement, template reselection or class switching was performed.

The 2 x 2 calculation compares point projection/FFT, identical point projection/CZT, mesh projection/FFT and identical mesh projection/CZT. The CZT evaluates the same cropped frequency lattice as the original 2,048-padded FFT, followed by the same bilinear COMPLEX-amplitude interpolation and modulus squared. The original half-pixel Fourier translation is preserved before interpolation. All routes use q_z=0; this comparison does not add Ewald curvature. Density/coordinates remain in electrons and Angstrom units.

Transform-only maximum relative intensity L2: point {audit['point_CZT_FFT_max_relative_L2']:.9g}; mesh {audit['mesh_CZT_FFT_max_relative_L2']:.9g}. The historical NumPy FFT amplitude dtype is {audit['environment']['fft_amplitude_dtype']}; the CZT accumulation is complex128. These small differences are numerical precision effects, not a changed prediction model. Original FFT templates are reproduced exactly; all 76 original RMS values agree to relative {summary['original_RMS_reproduction_max_relative_difference']:.4g}. Archived log-correlation differs by at most {summary['original_logcorr_reproduction_max_absolute_difference']:.4g} under independent float64 recomputation.

Mesh representation: the existing UNREFINED 12-level density meshes are used, with the same class-specific prism occupancy construction. The integrated-charge relative difference (mesh minus point, divided by point charge) ranges from {100*charge.min():+.3f}% to {100*charge.max():+.3f}%. After mean normalization, the mesh/point intensity difference over the unmasked 5-40 pixel annulus is {100*qa_shapes.min():.3f}% to {100*qa_shapes.max():.3f}% across 34 templates. This combines density layering, surface extraction/decimation and point-versus-continuous projection; it is not an FFT-versus-CZT error.

With original signal/background frozen, mean RMS changes from {float(full_a['mean_RMS']):.9f} to {float(frozen['mean_RMS']):.9f}. With only nonnegative signal and constant background independently refitted on the same full ROI, mean RMS changes from {float(full_a['mean_RMS']):.9f} to {float(full_b['mean_RMS']):.9f} ({100*(float(full_b['mean_RMS'])/float(full_a['mean_RMS'])-1):+.6f}%). Median RMS changes from {float(full_a['median_RMS']):.9f} to {float(full_b['median_RMS']):.9f}; mesh/CZT gives lower RMS for {full_b['lower_RMS_than_FFT']}/76 and higher RMS for {full_b['higher_RMS_than_FFT']}/76. The mean rises despite more individual improvements because the worsening cases have larger magnitudes. Median log-correlation changes from {float(full_a['median_logcorr']):.9f} to {float(full_b['median_logcorr']):.9f}; mean paired correlation difference is {float(full_b['mean_logcorr_change_vs_FFT']):+.9f}.

Two complementary 5-pixel radial-band folds (starting at radius 5 pixels) fit ONLY signal/background on training bands and freeze them on evaluation bands. Per-frame two-fold mean RMS change: {summary['twofold']['mean_RMS_change']:+.9f}, with lower RMS in {summary['twofold']['lower_RMS']}/76 frames. Mean paired two-fold correlation change: {summary['twofold']['mean_logcorr_change']:+.9f}. These are within-image conditional tests because historical class/orientation/alignment selection used the full ROI. The mean is equally weighted over frames; the two folds are first averaged within each frame.

Independent numerical controls: all 34 meshes were checked at distributed detector samples against a continuous triangular-boundary reference, without raster/FFT/CZT in the reference. Maximum primary-grid relative intensity L2 was {100*audit['primary_sparse_reference_max_relative_L2']:.5f}%. Densifying only the transverse q lattice fourfold reduced the maximum to {100*audit['dense_q_sparse_reference_max_relative_L2']:.5f}% at the same 1 A projection sampling. Seven predetermined cases additionally compare 0.5/0.25 A spatial grids and finer surface decimation. These controls verify the numerical mesh forward evaluation, not convergence to a unique physical protein density or convergence in density-layer count. Full-ROI reference coverage was not used for the boundary integral.

Raster diagnostics retain {audit['raster_diagnostics']['total_bad_balance_pixels']} unbalanced pixel occurrences across all unique templates (maximum {audit['raster_diagnostics']['max_bad_balance_pixels_per_template']} per template) and {audit['raster_diagnostics']['negative_clamped_pixels']} negative pixel occurrences clamped by the original production kernel. All templates remain in the comparison; the all-template reference audit includes the flagged cases. This is the production raster behavior, not a new silent correction.

Largest full-ROI worsening cases (all retained):
'''+''.join(f"  Pattern {r['pattern_index']}, {r['class_name']}: RMS change {float(r['RMS_change']):+.9f}; correlation change {float(r['logcorr_change']):+.9f}\n" for r in worst)+'''
Conclusion: the mesh/CZT route supports a closely comparable fixed-assignment GroEL prediction. Changing FFT to CZT alone leaves the predictions effectively unchanged. The residual differences arise mainly from density/mesh/projection representation and are small in experimental fit metrics. There is no systematic fit improvement attributable to CZT in these fixed assignments. No conclusion about classification, orientation recovery, assembly identity, radial-control separation or an Ewald-aware mesh library is established by this experiment.

Sources and reproducibility: experiment_config.json binds the original HDF5, assignment CSV, density grid, meshes and scientific source hashes. fit_metrics.csv contains every route/fold/frame; paired_differences.csv and paired_twofold.csv retain positive and negative differences; template_qa.csv and all_template_reference_audit.csv give all-template numerical diagnostics. independent_audit.json recomputes all 1,216 fitting records using an independent analytical two-parameter nonnegative least-squares solution and explicit centered-dot correlation. Numerical algorithms and all earlier evidence remain unchanged.

Commands from the workspace root:
  python validation/revision_groel_mesh_czt_matched.py all
  python validation/revision_groel_mesh_czt_report.py audit
  python validation/revision_groel_mesh_czt_report.py report
'''
    (OUT/'analysis_report.txt').write_text(report,encoding='utf-8')
    summary.update(status='complete',independent_audit='pass',numerical_controls='complete',interpretation='Closely comparable fixed-assignment predictions; transform-only differences negligible; no systematic experimental-fit improvement from CZT alone.')
    m.dump(OUT/'summary.json',summary)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='completion_manifest.json']
    sources=[Path(m.__file__),Path(__file__)]
    m.dump(OUT/'completion_manifest.json',dict(status='complete',source_hashes_unchanged=m.protected()==json.loads((OUT/'experiment_config.json').read_text())['source_hashes'],
        records=76*4*4,unique_templates=34,independent_audit=audit,
        source_scripts={p.relative_to(m.ROOT).as_posix():m.sha(p) for p in sources},
        files={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=m.sha(p)) for p in files},
        visual_qa='Pending inspection of saved PNGs; tracked separately after rendering'))
    print('REPORT complete',OUT,flush=True)

if __name__=='__main__':
    if sys.argv[1]=='audit':audit()
    elif sys.argv[1]=='report':figures_and_report()
    else:raise ValueError(sys.argv[1])
