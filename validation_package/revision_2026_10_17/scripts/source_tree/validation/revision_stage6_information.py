"""Conditional grid coverage, size/rounding information, selected curved checks."""
from revision_common import *
from revision_models import fit_rows
from revision_gpu_boundary_general import MeshBoundaryReference
from revision_polyhedron_reference import polyhedron_amplitude
import cupy as cp,time
stage=OUT/'06_information';stage.mkdir(exist_ok=True);cfg,data,exp=id57_data();data.pop('clean_patterns',None)
radii=np.array([0,.5,1,2,4,8,12]);scales=np.array([.9,.95,1,1.05,1.1]);nori=240;ns=5
exp=np.maximum(exp,0)
roi=data['roi'];params=list(csv.DictReader((OUT/'03_rounding/candidate_parameters.csv').open()))
rows=list(csv.DictReader((OUT/'03_rounding/id57_rounding_results.csv').open()))
with np.load(OUT/'02_baseline/evaluation_masks.npz') as f:train=f['train'];test=f['test']
coverage=[];profiles=[];mode_results={};score_banks={}
for mode,key,evkey in [('full','full','full'),('heldout','heldout_train','heldout_eval'),('heldout_reverse','reverse_train','reverse_eval')]:
    bank=np.load(OUT/f'03_rounding/{key}_scores.npy',mmap_mode='r');ev=np.load(OUT/f'03_rounding/{evkey}_scores.npy',mmap_mode='r')
    score_banks[mode]=bank
    size_delta=[];radius_delta=[];coarse_loss=[]
    for frame in range(209):
      grid=np.asarray(bank[frame]).reshape(7,240,5)
      fixed=grid[0,:,2];sharp=grid[0].ravel();allv=grid.ravel()
      fixedi=int(np.argmax(fixed))*5+2;sharpi=int(np.argmax(sharp));alli=int(np.argmax(allv))
      coarse=grid[:,::2,:];c=np.unravel_index(np.argmax(coarse),coarse.shape);coarsei=c[0]*1200+c[1]*2*5+c[2]
      size_delta.append(float(ev[frame,sharpi]-ev[frame,fixedi]));radius_delta.append(float(ev[frame,alli]-ev[frame,sharpi]))
      coarse_loss.append(float(bank[frame,alli]-bank[frame,coarsei]))
      rp=grid.max(axis=(1,2));sp=grid.max(axis=(0,1))
      rnear=radii[rp>=rp.max()-.01];snear=52*scales[sp>=sp.max()-.01]
      coverage.append({'frame':frame,'mode':mode,'size_only_eval_delta':size_delta[-1],
        'rounding_over_variable_size_eval_delta':radius_delta[-1],'240_vs120_fit_score_gain':coarse_loss[-1],
        'radius_profile_width_within_0p01_nm':float(np.ptp(rnear)),
        'side_profile_width_within_0p01_nm':float(np.ptp(snear)),
        'radius_best_base_nm':float(radii[np.argmax(rp)]),'side_best_nm':float(52*scales[np.argmax(sp)])})
      for ri,r in enumerate(radii):profiles.append({'frame':frame,'mode':mode,'radius_nm':float(r),'profile_score':float(rp[ri])})
    mode_results[mode]={'n':209,'mean_size_only_delta':float(np.mean(size_delta)),
      'mean_rounding_over_size_delta':float(np.mean(radius_delta)),
      'rounding_over_size_positive_count':int((np.array(radius_delta)>0).sum()),
      'mean_240_vs120_fit_gain':float(np.mean(coarse_loss)),'p95_240_vs120_fit_gain':float(np.quantile(coarse_loss,.95)),
      'max_240_vs120_fit_gain':float(max(coarse_loss)),
      'median_radius_profile_width_within_0p01_nm':float(np.median([c['radius_profile_width_within_0p01_nm'] for c in coverage if c['mode']==mode]))}
save_csv(stage/'conditional_information.csv',coverage);save_csv(stage/'radius_profiles.csv',profiles)
# Independent full Ewald evaluation of eight pre-specified experimental cases.
qx,qy=np.meshgrid(data['qaxis'],data['qaxis']);k=2*np.pi/cfg['wavelength_nm'];norm=np.sqrt(1+(qx/k)**2+(qy/k)**2)
flat=np.column_stack([qx[roi],qy[roi],np.zeros(roi.sum())]);curved=np.column_stack([(qx/norm)[roi],(qy/norm)[roi],(k*(1/norm-1))[roi]])
checks=[];qa=[]
for frame in [0,1,11,129,67,176,54,58]:
    row=next(r for r in rows if int(r['frame'])==frame and r['mode']=='full')
    radius=float(row['radius_base_nm']);scale=float(row['scale']);oi=int(row['orientation_index']);rot=data['rotation_matrices'][oi].astype(float)
    outer,core=id57_meshes(rounding=radius,segments=16);weighted=[(outer,1.),(core,.5)]
    pf=MeshBoundaryReference(weighted,flat*scale).intensity(rot)*scale**6
    pc=MeshBoundaryReference(weighted,curved*scale).intensity(rot)*scale**6
    scores,a,b,x,pred=fit_rows(exp[frame][roi],np.stack([pf,pc]))
    # Independent CPU check on 73 points covering entire ROI, for generic boundary kernel.
    subset=np.linspace(0,len(flat)-1,73,dtype=int);query=(curved[subset]*scale)@rot
    cpu=sum(c*polyhedron_amplitude(query,m) for m,c in weighted)
    qa.append({'frame':frame,'gpu_cpu_exact_boundary_intensity_l2':rel_l2(pc[subset],np.abs(cpu)**2*scale**6)})
    checks.append({'frame':frame,'radius_base_nm':radius,'scale':scale,'orientation':oi,
      'planar_exact_score':float(scores[0]),'curved_exact_score':float(scores[1]),
      'curved_minus_planar_score':float(scores[1]-scores[0]),'physical_intensity_l2':rel_l2(pf,pc)})
    if frame in [0,1,11,129]:np.savez_compressed(stage/f'curved_rounded_fit_frame_{frame}.npz',fitted_curved=pred[1],fitted_planar=pred[0],roi=roi)
    print('Selected rounded curved check',frame,'scores',scores.tolist(),flush=True)
save_csv(stage/'selected_rounded_ewald.csv',checks)
dump(stage/'information_summary.json',{'conditional_grid':mode_results,'selected_curved_checks':checks,'boundary_qa':qa,
 'cautions':['Score profiles and a0.01 score drop are sensitivity descriptors, not confidence intervals or a posterior.',
 '240 versus120 comparison is only within the original near-normal orientation range and selected morphology grid.',
 'No coverage of all SO(3), all3D speckles, or unknown morphologies is established.',
 'Free fluence prevents determining absolute density or volume from fitted amplitude.',
 'Selected curved checks refit scale/background at a fixed planar-selected geometry; no global curve-aware rounded retrieval is claimed.',
 'Grid-edge winners require expanding the parameter grid before making a structural radius or size claim.'],
 'high_q_scope':'A boundary mesh encodes piecewise density contrast, not atomistic positions or q-dependent atomic scattering factors; atomic coordinates plus atomic form factors are the relevant high-q alternative.'})
