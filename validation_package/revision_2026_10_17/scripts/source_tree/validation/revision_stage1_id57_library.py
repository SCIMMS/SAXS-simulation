"""Stage 1 complete ID57 library and paired retrieval at unchanged ROI."""
from revision_common import *
import argparse,time
cfg,data,exp=id57_data()
stage=OUT/"01_consistency"
start=time.perf_counter()
path=stage/"id57_mesh_czt_library_n768.npz"
if path.exists():
    with np.load(path) as f: clean=f["clean_patterns"]; diagnostic_rows=list(csv.DictReader((stage/"id57_library_diagnostics.csv").open()))
else:
    clean=[];diagnostic_rows=[];meshes=id57_meshes()
    for ti,rot in enumerate(data["rotation_matrices"]):
        p,diag=id57_projection(rot.astype(float),768,144,meshes)
        raw=czt_intensity(p,144,data["qaxis"])
        clean.append(clean_id57(raw,cfg,data["valid_mask"]))
        diagnostic_rows.append({"template":ti,"mass_nm3":float(p.sum()*(144/768)**2),
          "bad_balance_pixels":diag["outer"]["bad_balance_pixels"]+diag["core"]["bad_balance_pixels"],
          "negative_pixels":diag["outer"]["negative_pixels_before_clamp"]+diag["core"]["negative_pixels_before_clamp"]})
        if ti%40==0:print("Library",ti,"elapsed",time.perf_counter()-start,flush=True)
    clean=np.stack(clean)
    np.savez_compressed(path,clean_patterns=clean,valid_mask=data["valid_mask"],roi=data["roi"],
      rotation_matrices=data["rotation_matrices"],qaxis_nm_inv=data["qaxis"],grid=768,extent_nm=144.)
    save_csv(stage/"id57_library_diagnostics.csv",diagnostic_rows)
roi=data["roi"]
score_matrices={}; result=[]
bins=control.build_radius_bins(roi.shape,1)
for name,library in [("original_fft",data["clean_patterns"]),("mesh_czt",clean)]:
    scores=[];records=[]
    for frame,image in enumerate(exp):
        s,sc,bg,rms=idscore.fit_scale_background_and_score(image,library,roi)
        scores.append(s)
        order=np.argsort(s)[::-1]
        ti=int(order[0])
        records.append({"frame":frame,"branch":name,"best_template":ti,"score":s[ti],
         "scale":sc[ti],"background":bg[ti],"residual_rms":rms[ti],"score_gap_second":s[ti]-s[order[1]]})
    score_matrices[name]=np.stack(scores)
    random=control.scramble_stack(library,bins,roi,np.random.default_rng(20260602))
    for frame,image in enumerate(exp):
        s,sc,bg,rms=idscore.fit_scale_background_and_score(image,random,roi)
        ti=int(np.argmax(s))
        rec=records[frame];rec.update({"control_best_template":ti,"control_score":s[ti],
           "paired_delta":rec["score"]-s[ti]})
        result.append(rec)
    print(name, "mean score",np.mean([r["score"] for r in records]),flush=True)
    del random
for frame in range(len(exp)):
    o=score_matrices["original_fft"][frame];n=score_matrices["mesh_czt"][frame]
    a=np.argsort(o)[::-1];b=np.argsort(n)[::-1]
    print("Frame",frame,"original",int(a[0]),float(o[a[0]]),"CZT",int(b[0]),float(n[b[0]]),flush=True)
save_csv(stage/"id57_retrieval_comparison.csv",result)
comparisons=[]
for frame in range(len(exp)):
    a=score_matrices["original_fft"][frame];b=score_matrices["mesh_czt"][frame]
    comparisons.append({"frame":frame,"original_best":int(np.argmax(a)),"mesh_czt_best":int(np.argmax(b)),
       "same_best":int(np.argmax(a))==int(np.argmax(b)),"score_correlation":float(np.corrcoef(a,b)[0,1]),
       "score_difference_best":float(b.max()-a.max())})
save_csv(stage/"id57_ranking_comparison.csv",comparisons)
qa=[]
for ti in [0,144,238]:
    p,_=id57_projection(data["rotation_matrices"][ti].astype(float),768,144)
    axis=(np.arange(768)-(768-1)/2)*144/768
    q=data["qaxis"][::25]
    a=sm.czt2_from_real_axes(p,axis,axis,q,q)
    e=np.exp(-1j*q[:,None]*axis[None,:])
    direct=(e@p@e.T)*(144/768)**2
    qa.append({"template":ti,"czt_direct_dft_relative_l2":rel_l2(a,direct)})
summary={"scope":"all 12 ID57 experimental frames; publication subset 0 through 7 reported separately",
 "original_sources_unchanged":True,"orientation_count":240,"projection_grid":768,"extent_nm":144,
 "roi_pixels":int(roi.sum()),"ranks":comparisons,"czt_direct_dft_qa":qa,
 "legacy_production_function":"uploaded_SAXS_simulator/saxs_module.py czt2_from_real_axes; complex128 CPU SciPy",
 "projection_function":"unchanged signed GPU raster kernel",
 "seed_radial_control":20260602,"elapsed_s":time.perf_counter()-start,
 "paired_summaries":{}}
for branch in ["original_fft","mesh_czt"]:
    for limit in [8,len(exp)]:
        rows=[r for r in result if r["branch"]==branch and r["frame"]<limit]
        delta=np.array([r["paired_delta"] for r in rows])
        summary["paired_summaries"][f"{branch}_{limit}_frames"]={"n":len(rows),"mean_score":float(np.mean([r["score"] for r in rows])),
           "mean_control_score":float(np.mean([r["control_score"] for r in rows])),"mean_delta":float(delta.mean()),
           "min_delta":float(delta.min()),"max_delta":float(delta.max()),"positive_count":int((delta>0).sum())}
dump(stage/"id57_full_summary.json",summary)
np.savez_compressed(stage/"id57_all_scores.npz",**score_matrices)
print(json.dumps(safe(summary["paired_summaries"]),indent=2),flush=True)
