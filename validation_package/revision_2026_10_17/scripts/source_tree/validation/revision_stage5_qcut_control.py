"""Actual physical ROI cutoff sweep and curved-reference radial controls."""
from revision_common import *
from revision_gpu_surface_reference import *
from revision_scoring import legacy_shortlist
stage=OUT/"05_ewald";cfg,data,exp=id57_data();data.pop("clean_patterns",None)
mask=data["roi"];qx,qy=np.meshgrid(data["qaxis"],data["qaxis"]);k=2*np.pi/cfg["wavelength_nm"]
sx=qx/k;sy=qy/k;norm=np.sqrt(1+sx*sx+sy*sy)
curved=np.stack([k*sx/norm,k*sy/norm,k*(1/norm-1)],axis=-1)[mask]
flat=np.stack([qx,qy,np.zeros_like(qx)],axis=-1)[mask]
transverse=curved.copy();transverse[:,2]=0
core=id57_meshes()[1]
refs=[CoreShellReference(core,q) for q in [curved,flat,transverse]]
radial=np.hypot(qx,qy)[mask];cutoffs=np.arange(50,191,10)*(data["qaxis"][201]-data["qaxis"][200])
rows=[];banks=[[],[],[]]
for ti,r in enumerate(data["rotation_matrices"]):
    v=[ref.intensity(r.astype(float)) for ref in refs]
    for bank,a in zip(banks,v):bank.append(a.astype(np.float32))
    for cutoff in cutoffs:
        m=radial<=cutoff
        rows.append({"template":ti,"qmax_nm_inv":cutoff,"roi_pixels":int(m.sum()),
          "curvature_only_l2":rel_l2(v[2][m],v[0][m]),"combined_l2":rel_l2(v[1][m],v[0][m])})
save_csv(stage/"id57_ewald_cutoff_metrics.csv",rows)
aggregates=[]
for cutoff in cutoffs:
    selected=[r for r in rows if r["qmax_nm_inv"]==cutoff]
    x=np.array([r["combined_l2"] for r in selected])
    aggregates.append({"qmax_nm_inv":cutoff,"roi_pixels":selected[0]["roi_pixels"],"mean_l2":float(x.mean()),
       "median_l2":float(np.median(x)),"p95_l2":float(np.quantile(x,.95)),"max_l2":float(x.max())})
save_csv(stage/"id57_ewald_cutoff_summary.csv",aggregates)
limits={}
for tol in [.01,.05,.10]:
    eligible=[a["qmax_nm_inv"] for a in aggregates if a["max_l2"]<=tol]
    limits[str(tol)]=max(eligible) if eligible else None
np.savez_compressed(stage/"id57_exact_roi_physical_intensities.npz",
    curved=np.stack(banks[0]),planar=np.stack(banks[1]),transverse=np.stack(banks[2]),qradial=radial,roi=mask)
dump(stage/"id57_ewald_validity_cutoffs.json",{"cutoffs":aggregates,"largest_sampled_qmax_under_worst_orientation_tolerance":limits,
 "definition":"Intensity L2 across original ROI inner radius45px and each selected outer cutoff; maximum over all240 orientations.",
 "caution":"Thresholds are reporting conventions, not universal physical or detector-noise limits; no extrapolation below the tested cutoff."})
with np.load(stage/"id57_exact_ewald_library.npz") as f:lib=f["clean_patterns"]
random=control.scramble_stack(lib,control.build_radius_bins(mask.shape,1),mask,np.random.default_rng(20260602))
gpu=cp.asarray(random[:,mask]);controls=[]
original=list(csv.DictReader((stage/"id57_exact_ewald_retrieval.csv").open()))
for frame,image in enumerate(exp):
    ti,s,a,b,r,_=legacy_shortlist(image,random,mask,gpu)
    controls.append({"frame":frame,"structured_score":float(original[frame]["score"]),"control_score":s,
        "delta":float(original[frame]["score"])-s})
save_csv(stage/"id57_exact_ewald_paired_control.csv",controls)
dump(stage/"id57_exact_ewald_control_summary.json",{"n":209,"positive_count":sum(r["delta"]>0 for r in controls),
 "mean_delta":float(np.mean([r["delta"] for r in controls])),"mean_control_score":float(np.mean([r["control_score"] for r in controls]))})
print("Physical q limits",json.dumps(safe(limits)),"control positives",sum(r["delta"]>0 for r in controls),flush=True)
