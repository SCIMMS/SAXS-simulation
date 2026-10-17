"""Actual ID57 physical-geometry test over the original 240 orientations."""
from revision_common import *
from revision_polyhedron_reference import *
stage=OUT/"05_ewald";stage.mkdir(exist_ok=True)
cfg,data,exp=id57_data();del exp;data.pop("clean_patterns",None)
k=2*np.pi/cfg["wavelength_nm"];meshes=id57_meshes()
probe=np.random.default_rng(57).normal(size=(1000,3))
probe=np.vstack([probe,np.array([[0,0,0],[.3,0,0],[.3,.3,0],[.3,.3,.3],[1e-8,.3,.3]])])
closed=52**3*np.prod(np.sinc(probe*26/np.pi),axis=1)
numerical=polyhedron_amplitude(probe,meshes[0])
qa={"polyhedron_exact_box_amplitude_relative_l2":rel_l2(numerical,closed)}
rows=[]; planar_bank=[]; curved_bank=[]
def queries(step):
    qx,qy=np.meshgrid(data["qaxis"][::step],data["qaxis"][::step])
    sx=qx/k;sy=qy/k;norm=np.sqrt(1+sx*sx+sy*sy)
    curved=np.stack([k*sx/norm,k*sy/norm,k*(1/norm-1)],axis=-1)
    transverse=curved.copy();transverse[...,2]=0
    small=np.stack([qx,qy,np.zeros_like(qx)],axis=-1)
    mask=data["roi"][::step,::step]
    return mask,curved[mask],transverse[mask],small[mask],np.hypot(qx,qy)[mask]
mask,qcur,qtrans,qsmall,qr=queries(4)
for ti,rotation in enumerate(data["rotation_matrices"]):
    r=rotation.astype(float)
    true=abs(core_shell_amplitude(qcur,r,meshes))**2
    trans=abs(core_shell_amplitude(qtrans,r,meshes))**2
    small=abs(core_shell_amplitude(qsmall,r,meshes))**2
    planar_bank.append(small);curved_bank.append(true)
    ring=qr>.9
    row={"template":ti,"sampled_roi_pixels":int(mask.sum()),
      "curvature_only_roi_l2":rel_l2(trans,true),"combined_roi_l2":rel_l2(small,true),
      "outer_q_gt_0p9_curvature_l2":rel_l2(trans[ring],true[ring])}
    rows.append(row)
    if ti%40==0:print("Actual core-shell Ewald",ti,flush=True)
for ti in [0,144,238]:
    mask,cur,trans,small,qr=queries(1);r=data["rotation_matrices"][ti].astype(float)
    amplitude=core_shell_amplitude(small,r,meshes)
    true=abs(core_shell_amplitude(cur,r,meshes))**2
    p,_=id57_projection(r,768,144,meshes,"cpu")
    mesh_i=czt_intensity(p,144,data["qaxis"])
    qa[f"template_{ti}"]={"full_roi_combined_ewald_l2":rel_l2(abs(amplitude)**2,true),
        "mesh_czt_exact_polyhedron_planar_intensity_l2":rel_l2(mesh_i[mask],abs(amplitude)**2),
        "mesh_czt_vs_curved_exact_l2":rel_l2(mesh_i[mask],true)}
save_csv(stage/"id57_core_shell_ewald_all_orientations.csv",rows)
np.savez_compressed(stage/"id57_core_shell_exact_ewald_roi_sampled.npz",planar_intensities=np.stack(planar_bank),curved_intensities=np.stack(curved_bank),qradial=queries(4)[-1],roi_sampled=queries(4)[0])
dump(stage/"id57_core_shell_ewald_summary.json",{"n_orientations":240,"q_convention":"2pi/lambda; exact physical detector directions",
 "qa":qa,"metrics":{key:{"min":min(r[key] for r in rows),"max":max(r[key] for r in rows),"mean":float(np.mean([r[key] for r in rows]))}
  for key in ["curvature_only_roi_l2","combined_roi_l2","outer_q_gt_0p9_curvature_l2"]},
 "sampling":"every fourth detector pixel for 240-orientation sweep; all valid ROI pixels for three source orientations",
 "reference":"exact box amplitude plus exact clipped-octahedron boundary amplitude; confluent limits replace low-order quadrature near repeated triangle phases",
 "caution":"Numerical physical-geometry error, not detector-noise significance or a universal q limit."})
print(json.dumps(safe(qa)),flush=True)
