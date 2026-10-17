"""Exact historical scores after GPU screening; complete 209-frame cohort."""
from revision_common import *
from revision_scoring import *
import time
cfg,data,exp=id57_data();stage=OUT/"01_consistency";start=time.perf_counter()
with np.load(stage/"id57_mesh_czt_library_n768.npz") as f:clean=f["clean_patterns"]
result=[];all_scores={};qa=[]
roi=data["roi"];bins=control.build_radius_bins(roi.shape,1)
for name,lib in [("original_fft",data["clean_patterns"]),("mesh_czt",clean)]:
    matrices=[]
    gpu=cp.asarray(lib[:,roi])
    records=[]
    for frame,image in enumerate(exp):
        ti,s,a,b,r,approx=legacy_shortlist(image,lib,roi,gpu)
        matrices.append(approx)
        records.append({"frame":frame,"branch":name,"best_template":ti,"score":s,"scale":a,"background":b,"residual_rms":r})
        if frame in [0,1,11]:
            exact=idscore.fit_scale_background_and_score(image,lib,roi)[0]
            qa.append({"branch":name,"frame":frame,"gpu_cpu_max_absolute_score_error":float(np.max(np.abs(exact-approx))),
             "same_best":int(np.argmax(exact))==ti})
        if frame%40==0:print("Retrieval",name,frame,flush=True)
    all_scores[name]=np.stack(matrices);del gpu
    rand=control.scramble_stack(lib,bins,roi,np.random.default_rng(20260602));gpu=cp.asarray(rand[:,roi])
    for frame,image in enumerate(exp):
        ti,s,a,b,r,_=legacy_shortlist(image,rand,roi,gpu)
        rec=records[frame];rec.update(control_best_template=ti,control_score=s,paired_delta=rec["score"]-s)
        result.append(rec)
        if frame%40==0:print("Control",name,frame,flush=True)
    del rand,gpu;cp.get_default_memory_pool().free_all_blocks()
save_csv(stage/"id57_retrieval_comparison.csv",result)
ranks=[]
for frame in range(len(exp)):
    a=all_scores["original_fft"][frame];b=all_scores["mesh_czt"][frame]
    rows=[r for r in result if r["frame"]==frame]
    original=next(r for r in rows if r["branch"]=="original_fft")
    current=next(r for r in rows if r["branch"]=="mesh_czt")
    ranks.append({"frame":frame,"original_best":original["best_template"],"mesh_czt_best":current["best_template"],
        "same_best":original["best_template"]==current["best_template"],
        "score_correlation":float(np.corrcoef(a,b)[0,1]),"best_score_difference":current["score"]-original["score"]})
save_csv(stage/"id57_ranking_comparison.csv",ranks)
summary={"scope":"all 209 ID57 experimental frames; no score-based exclusions","n_frames":len(exp),"orientation_count":240,
 "projection_grid":768,"extent_nm":144,"roi_pixels":int(roi.sum()),"same_best_count":sum(r["same_best"] for r in ranks),
 "qa_gpu_shortlist":qa,"backend":"unaltered production GPU raster and CPU complex128 SciPy CZT",
 "score":"original ID57 score, GPU shortlist followed by exact CPU fitting of at least top 10",
 "seed_radial_control":20260602,"elapsed_s":time.perf_counter()-start,"paired_summaries":{}}
for branch in ["original_fft","mesh_czt"]:
    rows=[r for r in result if r["branch"]==branch]
    delta=np.array([r["paired_delta"] for r in rows])
    summary["paired_summaries"][branch]={"n":len(rows),"mean_score":float(np.mean([r["score"] for r in rows])),
      "mean_control_score":float(np.mean([r["control_score"] for r in rows])),"mean_delta":float(delta.mean()),
      "min_delta":float(delta.min()),"max_delta":float(delta.max()),"positive_count":int((delta>0).sum())}
for ti in [0,144,238]:
    p,_=id57_projection(data["rotation_matrices"][ti].astype(float),768,144)
    axis=(np.arange(768)-(768-1)/2)*144/768;q=data["qaxis"][::25]
    a=sm.czt2_from_real_axes(p,axis,axis,q,q);e=np.exp(-1j*q[:,None]*axis[None,:]);direct=(e@p@e.T)*(144/768)**2
    summary.setdefault("czt_direct_dft_qa",[]).append({"template":ti,"relative_l2":rel_l2(a,direct)})
dump(stage/"id57_full_summary.json",summary)
np.savez_compressed(stage/"id57_all_scores.npz",**all_scores)
print(json.dumps(safe(summary),indent=2),flush=True)
