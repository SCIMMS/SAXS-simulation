"""Audit and figures for the CZT-only GroEL primary comparisons."""
import json,csv
from pathlib import Path
import revision_groel_czt_primary as p
import revision_groel_mesh_czt_matched as m
from revision_groel_mesh_czt_report import independent_fit,independent_corr
np=m.np;OUT=p.OUT
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)

def rows(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8')))

def audit():
    images,valid,ann,even=p.observations();source=m.assignments();records=rows('fixed_control_results.csv')
    assert len(records)==76*3*2
    rng=np.random.default_rng(20260602);bins=p.radius_bin_map((128,128),1.);cache={}
    for k in range(429):
        path=m.OUT/f'templates/template_{k:03d}.npz'
        if path.exists():
            with np.load(path) as z:raw=z['patterns'][3].astype(np.float32)
        else:raw=np.zeros((128,128),np.float32)
        control=p.radial_scramble_template(raw,bins,ann,rng)
        if path.exists():cache[k]=dict(structured=raw.astype(float),radial_control=control.astype(float))
    max_rms=0.;max_corr=0.
    for r in records:
        i=int(r['pattern_index']);shift=(int(r['shift_y']),int(r['shift_x']));y=np.roll(images[i],shift,axis=(0,1));roi=np.roll(valid,shift,axis=(0,1))&ann
        train=roi if r['fold']=='full' else roi&(even if r['fold']=='even_to_odd' else ~even)
        test=roi if r['fold']=='full' else roi&~train;x=cache[int(r['template_index'])][r['family']]
        a,b=independent_fit(y,x,train);prediction=a*x+b
        rms=float(np.sqrt(np.mean((prediction[test]-y[test])**2)));corr=independent_corr(y[test],prediction[test])
        max_rms=max(max_rms,abs(rms-float(r['test_rms'])));max_corr=max(max_corr,abs(corr-float(r['test_logcorr'])))
        assert r['model_name']==source[i]['model_name'] and r['rotation_index']==source[i]['rotation_index']
    ab=rows('ab_paired_results.csv');assert len(ab)==456
    with np.load(OUT/'ab_templates_mesh_czt.npz') as z:lib=z['templates'].astype(float);indices=z['rotation_indices']
    max_rank_gap=0.
    for r in ab:
        i=int(r['pattern_index']);shift=(int(r['shift_y']),int(r['shift_x']));y=np.roll(images[i],shift,axis=(0,1));roi=np.roll(valid,shift,axis=(0,1))&ann
        train=roi if r['fold']=='full' else roi&(even if r['fold']=='even_to_odd' else ~even)
        test=roi if r['fold']=='full' else roi&~train;j=0 if r['model']=='A' else 1;k=int(np.flatnonzero(indices==int(r['rotation_index']))[0])
        a,b=independent_fit(y,lib[j,k],train);prediction=a*lib[j,k]+b
        rms=float(np.sqrt(np.mean((prediction[test]-y[test])**2)));corr=independent_corr(y[test],prediction[test])
        max_rms=max(max_rms,abs(rms-float(r['test_rms'])));max_corr=max(max_corr,abs(corr-float(r['test_logcorr'])))
        _,_,allmse=p.batch_fit(y[train],np.stack([x[train] for x in lib[j]]))
        gap=float(np.mean((prediction[train]-y[train])**2)-np.min(allmse));max_rank_gap=max(max_rank_gap,abs(gap))
    assert max_rms<1e-10 and max_corr<1e-10 and max_rank_gap<1e-9
    qx,qy=m.detector();yy,xx=np.indices(ann.shape);sample=ann&(xx%8==0)&(yy%8==0);q=np.c_[qx[sample],qy[sample],np.zeros(sample.sum())]
    with m.h5py.File(m.ROOT/'public_data/CXIDB_ID_187/rotations.h5') as h:quats=h['rotations'][::128].astype(np.float32)
    numerical=[]
    for j,model in enumerate(['A','B']):
        parts=p.assembly_parts(model)
        for k in sorted({0,1,50,100,150,200,18 if model=='A' else 23}):
            rotation=m.legacy.quaternion_wxyz_to_matrix(quats[k]);ref=m.MeshBoundaryReference(parts,q).intensity(rotation)
            error=m.rel(lib[j,k][sample],ref);assert error<.01,(model,k,error)
            numerical.append(dict(model=model,view=k,q_samples=int(sample.sum()),intensity_relative_L2=error))
    m.save_csv(OUT/'ab_independent_reference.csv',numerical)
    m.dump(OUT/'independent_audit.json',dict(status='pass',fixed_records=456,AB_records=456,
        max_independent_RMS_difference=max_rms,max_independent_logcorr_difference=max_corr,max_AB_training_MSE_rank_gap=max_rank_gap,
        AB_planar_boundary_reference_controls=numerical,source_hashes_unchanged=p.source_hashes()==json.loads((OUT/'ab_summary.json').read_text())['source_hashes']))
    print('AUDIT complete',max_rms,max_corr,flush=True)

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    # Main Figure 6 keeps the original square four-panel footprint.
    with np.load(OUT/'pattern67_primary.npz') as z:
        y=z['measured'];roi=z['roi'];prediction=z['structured'];projection=z['projection']
    fig,axs=plt.subplots(2,2,figsize=(8.36,7.6));vmax=float(np.quantile(np.log1p(y[roi]),.995));res=np.log1p(y)-np.log1p(prediction);lim=float(np.quantile(np.abs(res[roi]),.995))
    for ax,a,title,cmap,vmin,vmax2 in [(axs[0,0],np.log1p(y),'Measured log1p intensity','viridis',0,vmax),(axs[0,1],np.log1p(prediction),'Mesh/CZT log1p intensity','viridis',0,vmax),(axs[1,1],res,'Measured minus model log1p','RdBu_r',-lim,lim)]:
        im=ax.imshow(np.where(roi,a,np.nan),origin='lower',extent=[-64,64,-64,64],cmap=cmap,vmin=vmin,vmax=vmax2)
        ax.set_xlim(-42,42);ax.set_ylim(-42,42);ax.set_xlabel('Detector x / pixel');ax.set_ylabel('Detector y / pixel');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.8,pad=.03)
    n=len(projection);im=axs[1,0].imshow(projection,origin='lower',extent=[-n/2,n/2,-n/2,n/2],cmap='magma',vmin=0)
    axs[1,0].set_xlabel('u / Å');axs[1,0].set_ylabel('v / Å');axs[1,0].set_title('Projected mesh density / e Å⁻²');fig.colorbar(im,ax=axs[1,0],shrink=.8,pad=.03)
    fig.tight_layout();fig.savefig(FIG/'main_figure6_mesh_czt.png',dpi=300);plt.close(fig)
    # Paired controls at the same fixed class/view/registration.
    rr=[r for r in rows('fixed_control_paired.csv') if r['fold']=='full'];a=np.array([float(r['structured_logcorr']) for r in rr]);b=np.array([float(r['control_logcorr']) for r in rr]);delta=a-b
    fig,axs=plt.subplots(1,3,figsize=(11.5,3.41));bins=np.linspace(min(a.min(),b.min()),max(a.max(),b.max()),18)
    axs[0].hist(b,bins=bins,histtype='step',color='#B7791F',label='Paired radial control',lw=1.5);axs[0].hist(a,bins=bins,histtype='step',color='#2B6CB0',label='Mesh/CZT',lw=1.5);axs[0].legend(frameon=False,fontsize=8);axs[0].set_xlabel('Diagnostic log-correlation');axs[0].set_ylabel('Shots')
    lo=min(a.min(),b.min());hi=max(a.max(),b.max());axs[1].scatter(b,a,s=15,color='#2B6CB0');axs[1].plot([lo,hi],[lo,hi],color='.4',lw=.8);axs[1].set_xlabel('Paired radial control');axs[1].set_ylabel('Mesh/CZT');axs[1].set_aspect('equal',adjustable='box')
    axs[2].hist(delta,bins=18,color='#2B6CB0',edgecolor='white');axs[2].axvline(0,color='.4',lw=.8);axs[2].set_xlabel('Per-shot correlation difference');axs[2].set_ylabel('Shots');fig.tight_layout();fig.savefig(FIG/'si_figure13_mesh_czt_controls.png',dpi=240);plt.close(fig)
    # Matched A/B predictions for the same previously questioned shot.
    with np.load(OUT/'pattern67_AB.npz') as z:y=z['measured'];roi=z['roi'];a=z['A'];b=z['B']
    fig,axs=plt.subplots(1,4,figsize=(11.5,3.57));top=float(np.quantile(np.log1p(y[roi]),.995));lim=float(np.quantile(np.abs((np.log1p(y)-np.log1p(b))[roi]),.995))
    for col,(values,title) in enumerate([(np.log1p(y),'Measured'),(np.log1p(a),'A heptamer CZT'),(np.log1p(b),'B tetradecamer CZT'),(np.log1p(y)-np.log1p(b),'B log1p residual')]):
        im=axs[col].imshow(np.where(roi,values,np.nan),origin='lower',extent=[-64,64,-64,64],cmap='RdBu_r' if col==3 else 'viridis',vmin=-lim if col==3 else 0,vmax=lim if col==3 else top)
        axs[col].set_xlim(-42,42);axs[col].set_ylim(-42,42);axs[col].set_xlabel('Detector x / pixel');axs[col].set_title(title);fig.colorbar(im,ax=axs[col],shrink=.75,pad=.02)
    axs[0].set_ylabel('Detector y / pixel');fig.tight_layout();fig.savefig(FIG/'si_figure14_mesh_czt_AB.png',dpi=240);plt.close(fig)
    # Four synthetic model previews use the same recorded quaternion at index 0.
    with m.h5py.File(m.ROOT/'public_data/CXIDB_ID_187/rotations.h5') as h:q=h['rotations'][0]
    rotation=m.legacy.quaternion_wxyz_to_matrix(q)
    fig,axs=plt.subplots(2,4,figsize=(11.5,6.29));models=[('A','A'),('B','B'),('R35','B_tetradecamer_D7_polyprism_R35_N14'),('R55','B_tetradecamer_D7_polyprism_R55_N14')]
    for col,(label,model) in enumerate(models):
        parts=p.assembly_parts(model) if model in ['A','B'] else m.mesh_parts(model)
        n=2*int(np.ceil(max(np.linalg.norm(a.vertices,axis=1).max() for a,c in parts)+18));proj,_=m.mesh_projection(parts,rotation,float(n),n);intensity=m.czt_intensity(proj)
        im=axs[0,col].imshow(np.log1p(intensity/max(float(intensity.max()),1e-30)*100),origin='lower',cmap='viridis',vmin=0,vmax=np.log1p(100));axs[0,col].set_title(label+' synthetic mesh/CZT');fig.colorbar(im,ax=axs[0,col],shrink=.65,pad=.03)
        im=axs[1,col].imshow(proj,origin='lower',extent=[-n/2,n/2,-n/2,n/2],cmap='magma',vmin=0);fig.colorbar(im,ax=axs[1,col],shrink=.65,pad=.03);axs[1,col].set_xlabel('u / Å')
    axs[1,0].set_ylabel('v / Å');fig.tight_layout();fig.savefig(FIG/'si_figure12_mesh_czt_previews.png',dpi=240);plt.close(fig)
    m.dump(OUT/'figure_sources.json',dict(main_figure6='pattern67_primary.npz; log1p measured/prediction, density from exact fixed layered mesh',si_figure12='CZT-only previews, quaternion0, each intensity normalized to its own maximum for display',si_figure13='fixed_control_paired.csv; full ROI; fixed matched control protocol',si_figure14='pattern67_AB.npz and ab_paired_results.csv'))
    print('FIGURES complete',flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='audit':audit()
    elif sys.argv[1]=='figures':figures()
    else:raise ValueError(sys.argv[1])
