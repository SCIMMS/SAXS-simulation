from revision_common import *
from revision_polyhedron_reference import *
from revision_gpu_surface_reference import *
from revision_scoring import legacy_shortlist
import time
stage=OUT/"05_ewald";start=time.perf_counter()
cfg,data,exp=id57_data();data.pop("clean_patterns",None);meshes=id57_meshes();k=2*np.pi/cfg["wavelength_nm"]
qx,qy=np.meshgrid(data["qaxis"],data["qaxis"]);sx=qx/k;sy=qy/k;norm=np.sqrt(1+sx*sx+sy*sy)
q=np.stack([k*sx/norm,k*sy/norm,k*(1/norm-1)],axis=-1)
reference=CoreShellReference(meshes[1],q)
qa=[];library=[]
for ti,r in enumerate(data["rotation_matrices"]):
    raw=reference.intensity(r.astype(float)).reshape(qx.shape)
    library.append(clean_id57(raw,cfg,data["valid_mask"]))
    if ti in [0,144,238]:
        exact=abs(core_shell_amplitude(q.reshape(-1,3),r.astype(float),meshes))**2
        qa.append({"template":ti,"gpu_cpu_exact_reference_intensity_l2":rel_l2(raw.ravel(),exact)})
    if ti%40==0:print("Exact Ewald library",ti,"elapsed",round(time.perf_counter()-start,1),flush=True)
library=np.stack(library)
np.savez_compressed(stage/"id57_exact_ewald_library.npz",clean_patterns=library,rotation_matrices=data["rotation_matrices"],
    valid_mask=data["valid_mask"],roi=data["roi"])
rows=[];gpu=cp.asarray(library[:,data["roi"]])
for frame,image in enumerate(exp):
    ti,s,a,b,r,_=legacy_shortlist(image,library,data["roi"],gpu)
    rows.append({"frame":frame,"best_template":ti,"score":s,"scale":a,"background":b,"residual_rms":r})
save_csv(stage/"id57_exact_ewald_retrieval.csv",rows)
original=list(csv.DictReader((OUT/"01_consistency/id57_retrieval_comparison.csv").open()))
flat={int(r["frame"]):r for r in original if r["branch"]=="mesh_czt"}
comparison=[{"frame":r["frame"],"planar_score":float(flat[r["frame"]]["score"]),"curved_score":r["score"],
 "curved_minus_planar":r["score"]-float(flat[r["frame"]]["score"]),
 "same_best":r["best_template"]==int(flat[r["frame"]]["best_template"])} for r in rows]
save_csv(stage/"id57_exact_ewald_score_stability.csv",comparison)
dump(stage/"id57_exact_ewald_retrieval_summary.json",{"n":209,"qa":qa,"mean_curved_score":float(np.mean([r["score"] for r in rows])),
 "mean_curved_minus_planar":float(np.mean([r["curved_minus_planar"] for r in comparison])),
 "same_best_count":sum(r["same_best"] for r in comparison),"elapsed_s":time.perf_counter()-start,
 "scope":"independent low-face-count analytical 3D boundary reference; not a replacement for the generic GPU raster/CZT production kernel"})
print("Ewald reference fitting",json.dumps(safe(qa)),flush=True)
