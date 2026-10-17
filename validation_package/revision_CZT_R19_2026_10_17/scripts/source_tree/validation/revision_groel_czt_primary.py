"""CZT-only primary GroEL results, fixed historical assignments and new A/B fits.

Radial controls are paired to the SAME frozen class/view and alignment; they
are not the older independently selected FFT-control library results.
"""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
from types import SimpleNamespace
import revision_groel_mesh_czt_matched as m
from revision_groel_ab_matched import fit as batch_fit,score
from section5_groel_make_random_control_library import radius_bin_map,radial_scramble_template
np=m.np;ROOT=m.ROOT
OUT=ROOT/'output/groel_czt_primary_2026-10-17'
OUT.mkdir(parents=True,exist_ok=True)

def source_hashes():
    paths=[Path(__file__),Path(m.__file__),m.HITS,m.ASSIGN,m.DENSITY,
        ROOT/'public_data/CXIDB_ID_187/rotations.h5',
        ROOT/'validation/revision_groel_ab_matched.py',
        ROOT/'validation/section5_groel_make_random_control_library.py',
        m.PRIOR/'experiment_config.json']
    paths += sorted((m.OUT/'templates').glob('template_*.npz'))
    return {p.relative_to(ROOT).as_posix():m.sha(p) for p in paths}

def observations():
    with m.h5py.File(m.HITS) as h:
        images=h['patterns_ph'][:].astype(float);valid=h['mask'][:].astype(bool)
    yy,xx=np.indices((128,128));rad=np.hypot(xx-63.5,yy-63.5)
    ann=(rad>=5)&(rad<=40);even=(np.floor((rad-5)/5).astype(int)%2)==0
    return images,valid,ann,even

def fixed_and_controls():
    hashes=source_hashes();images,valid,ann,even=observations();source=m.assignments()
    # Advance the generator in original global template-index order so a
    # selected template's control is independent of hit order/repetition.
    rng=np.random.default_rng(20260602);bins=radius_bin_map((128,128),1.);bank={};qa=[]
    for k in range(429):
        path=m.OUT/f'templates/template_{k:03d}.npz'
        if path.exists():
            with np.load(path) as z:raw=z['patterns'][3];projection=z['mesh_projection']
        else:raw=np.zeros((128,128),np.float32)
        control=radial_scramble_template(raw,bins,ann,rng)
        if path.exists():
            for b in np.unique(bins[ann]):
                mask=ann&(bins==b);assert np.array_equal(np.sort(raw[mask]),np.sort(control[mask]))
            bank[k]=(raw.astype(float),control.astype(float),projection)
            qa.append(dict(template_index=k,radial_bin_permutations_exact=True,total_intensity_exact=float(raw[ann].sum(dtype=float))==float(control[ann].sum(dtype=float))))
    records=[];predictions={};examples={67}
    for i,row in source.items():
        shift=(int(row['shift_y']),int(row['shift_x']));y=np.roll(images[i],shift,axis=(0,1));roi=np.roll(valid,shift,axis=(0,1))&ann
        raw,control,projection=bank[int(row['template_index'])]
        for fold,train,test in [('full',roi,roi),('even_to_odd',roi&even,roi&~even),('odd_to_even',roi&~even,roi&even)]:
            for name,x in [('structured',raw),('radial_control',control)]:
                a,b=m.nuisance(y,x,train);p=a*x+b
                records.append(dict(pattern_index=i,source_frame_index=int(row['source_frame_index']),model_name=row['model_name'],template_index=int(row['template_index']),rotation_index=int(row['rotation_index']),
                    shift_y=shift[0],shift_x=shift[1],fold=fold,family=name,signal=a,background=b,
                    train_rms=float(np.sqrt(np.mean((p[train]-y[train])**2))),test_rms=float(np.sqrt(np.mean((p[test]-y[test])**2))),test_logcorr=m.logcorr(y[test],p[test]),train_pixels=int(train.sum()),test_pixels=int(test.sum())))
                if i in examples and fold=='full':predictions[name]=p
        if i in examples:
            np.savez_compressed(OUT/f'pattern{i}_primary.npz',measured=y,roi=roi,projection=projection,**predictions)
    m.save_csv(OUT/'fixed_control_results.csv',records);m.save_csv(OUT/'radial_permutation_audit.csv',qa)
    summaries={};paired=[]
    for fold in ['full','even_to_odd','odd_to_even']:
        by={(int(r['pattern_index']),r['family']):r for r in records if r['fold']==fold}
        delta=[];rms_delta=[]
        for i in range(76):
            a,b=by[i,'structured'],by[i,'radial_control'];d=a['test_logcorr']-b['test_logcorr'];dr=a['test_rms']-b['test_rms']
            delta.append(d);rms_delta.append(dr)
            paired.append(dict(pattern_index=i,fold=fold,structured_logcorr=a['test_logcorr'],control_logcorr=b['test_logcorr'],paired_delta=d,structured_RMS=a['test_rms'],control_RMS=b['test_rms'],RMS_delta=dr))
        summaries[fold]=dict(n=76,structured_median_logcorr=float(np.median([by[i,'structured']['test_logcorr'] for i in range(76)])),
            control_median_logcorr=float(np.median([by[i,'radial_control']['test_logcorr'] for i in range(76)])),
            paired_median_logcorr=float(np.median(delta)),paired_mean_logcorr=float(np.mean(delta)),structured_higher_logcorr=int((np.array(delta)>0).sum()),
            structured_median_RMS=float(np.median([by[i,'structured']['test_rms'] for i in range(76)])),control_median_RMS=float(np.median([by[i,'radial_control']['test_rms'] for i in range(76)])),
            paired_mean_RMS=float(np.mean(rms_delta)),structured_lower_RMS=int((np.array(rms_delta)<0).sum()))
    m.save_csv(OUT/'fixed_control_paired.csv',paired)
    assert hashes==source_hashes()
    m.dump(OUT/'fixed_control_summary.json',dict(status='complete',source_hashes=hashes,results=summaries,
        protocol='Fixed historical structured class/view/integer alignment for all 76; its own mesh/CZT radial permutation shares that identity and alignment; only nonnegative signal and constant background refit.',
        radial_control='Seed 20260602, one-pixel radius bins on annulus 5-40; RNG advanced over original 429 global template slots. No independent control-library view/alignment selection.',
        change_from_historical='This is a matched fixed-assignment control, not a reproduction of independently RMS-selected historical FFT-control fits. The old 0.611/0.606 and 61/76 statistics are superseded for the primary comparison.',
        hypothesis_grid='Historical source pool 3*143=429 states; only the 34 distinct assigned states are used in this fixed-assignment experiment.',
        q_condition='Planar q_z=0; original transverse detector mapping, 1 A raster and selected CZT q lattice.',
        pattern67=[r for r in records if r['pattern_index']==67 and r['fold']=='full']))
    print('FIXED_CONTROL',json.dumps(summaries),flush=True)

def assembly_parts(model):
    if model=='B':return m.mesh_parts('B_tetradecamer_D7')
    points=m.legacy.load_density_points(m.DENSITY,1e-4)
    _,_,center_a=m.legacy.make_model_points(points,'A_heptamer_C7')
    _,_,center_b=m.legacy.make_model_points(points,'B_tetradecamer_D7')
    with np.load(m.PRIOR/'protein_boundary.npz') as z:v=z['vertices']*10.;f=z['faces'];c=z['contrast']/1000.
    cfg=json.loads((m.PRIOR/'experiment_config.json').read_text());keep_v=np.zeros(len(v),bool);offset=0
    for layer in cfg['mesh_rows']:
        n=layer['vertices_heptamer'];keep_v[offset:offset+n]=True;offset+=2*n
    assert offset==len(v)
    selected=np.all(keep_v[f],axis=1);ff=f[selected];used=np.unique(ff);mapping=np.full(len(v),-1,int);mapping[used]=np.arange(len(used))
    vv=v[used]+center_b-center_a;ff=mapping[ff]
    assert len(ff)*2==len(f) and len(np.unique(c[selected]))==1
    mesh=SimpleNamespace(vertices=vv,faces=ff,face_normals=m.sm.compute_face_normals(vv,ff))
    return [(mesh,float(c[selected][0]))]

def generate_ab():
    hashes=source_hashes();qx,qy=m.detector()
    with m.h5py.File(ROOT/'public_data/CXIDB_ID_187/rotations.h5') as h:quats=h['rotations'][::128].astype(np.float32)
    assert len(quats)==201
    arrays=[];diags=[];t0=time.perf_counter()
    for model in ['A','B']:
        parts=assembly_parts(model);radius=max(np.linalg.norm(p.vertices,axis=1).max() for p,c in parts)
        n=2*int(np.ceil(radius+18));patterns=[]
        for j,q in enumerate(quats):
            rotation=m.legacy.quaternion_wxyz_to_matrix(q);p,d=m.mesh_projection(parts,rotation,float(n),n);x=m.czt_intensity(p)
            patterns.append(x)
            diags.append(dict(model=model,view=j,rotation_index=j*128,projection_n=n,charge_e=float(p.sum(dtype=float)),bad_balance_pixels=sum(a['bad_balance_pixels'] for a in d)))
            if j%30==0:print('AB_CZT_GENERATE',model,j,flush=True)
        arrays.append(np.stack(patterns))
    np.savez_compressed(OUT/'ab_templates_mesh_czt.npz',templates=np.stack(arrays),quaternions=quats,rotation_indices=np.arange(201)*128)
    m.save_csv(OUT/'ab_projection_diagnostics.csv',diags)
    assert hashes==source_hashes()
    m.dump(OUT/'ab_generation.json',dict(status='complete',source_hashes=hashes,views_per_assembly=201,seconds=time.perf_counter()-t0,
        forward='Closed 12-layer density boundaries -> original signed raster at 1 A -> selected-window complex128 CZT -> common complex-amplitude bilinear detector interpolation -> intensity; planar q_z=0',
        grid='Even projection box encloses the rotation-independent vertex radius plus 18 A; same globally aligned 1 A pixel centers as the original variable boxes.'))

def evaluate_ab():
    hashes=source_hashes();images,valid,ann,even=observations();source=m.assignments()
    with np.load(OUT/'ab_templates_mesh_czt.npz') as z:lib=z['templates'].astype(float);indices=z['rotation_indices']
    records=[];max_error=0.;predictions={}
    for i,row in source.items():
        shift=(int(row['shift_y']),int(row['shift_x']));y=np.roll(images[i],shift,axis=(0,1));roi=np.roll(valid,shift,axis=(0,1))&ann
        for fold,train,test in [('full',roi,roi),('even_to_odd',roi&even,roi&~even),('odd_to_even',roi&~even,roi&even)]:
            for j,model in enumerate(['A','B']):
                xx=np.stack([x[train] for x in lib[j]]);a,b,mse=batch_fit(y[train],xx);k=int(np.argmin(mse));p=a[k]*lib[j,k]+b[k]
                aa,bb=m.nuisance(y,lib[j,k],train);ind=aa*lib[j,k]+bb;error=m.rel(ind[test],p[test]);max_error=max(max_error,error);assert error<1e-10
                records.append(dict(pattern_index=i,source_frame_index=int(row['source_frame_index']),fold=fold,model=model,rotation_index=int(indices[k]),
                    signal=float(a[k]),background=float(b[k]),train_rms=float(np.sqrt(mse[k])),test_rms=float(np.sqrt(np.mean((p[test]-y[test])**2))),
                    test_logcorr=m.logcorr(y[test],p[test]),train_pixels=int(train.sum()),test_pixels=int(test.sum()),shift_y=shift[0],shift_x=shift[1]))
                if i==67 and fold=='full':predictions[model]=p
    m.save_csv(OUT/'ab_paired_results.csv',records);summary={}
    for fold in ['full','even_to_odd','odd_to_even']:
        by={(int(r['pattern_index']),r['model']):r for r in records if r['fold']==fold}
        dr=np.array([by[i,'A']['test_rms']-by[i,'B']['test_rms'] for i in range(76)])
        dc=np.array([by[i,'A']['test_logcorr']-by[i,'B']['test_logcorr'] for i in range(76)])
        summary[fold]=dict(n=76,A_lower_RMS=int((dr<0).sum()),B_lower_RMS=int((dr>0).sum()),median_A_minus_B_RMS=float(np.median(dr)),median_A_minus_B_logcorr=float(np.median(dc)))
    row=source[67];shift=(int(row['shift_y']),int(row['shift_x']))
    np.savez_compressed(OUT/'pattern67_AB.npz',measured=np.roll(images[67],shift,axis=(0,1)),roi=np.roll(valid,shift,axis=(0,1))&ann,**predictions)
    assert hashes==source_hashes()
    m.dump(OUT/'ab_summary.json',dict(status='complete',source_hashes=hashes,results=summary,pattern67=[r for r in records if r['pattern_index']==67],
        views_per_assembly=201,records=len(records),independent_NNLS_max_prediction_relative_L2=max_error,
        protocol='Separate A/B experiment reselects each assembly view on fitting pixels across the same 201 pre-symmetry quaternions; historical integer alignment fixed. Main B/R35/R55 assignments are unchanged.',
        scope='CZT-only planar forward comparison; no calibrated assembly identity or GroEL Ewald validation.'))
    print('AB_CZT_RESULTS',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fixed','generate_ab','evaluate_ab','all']);a=p.parse_args()
    if a.stage in ['fixed','all']:fixed_and_controls()
    if a.stage in ['generate_ab','all']:generate_ab()
    if a.stage in ['evaluate_ab','all']:evaluate_ab()
