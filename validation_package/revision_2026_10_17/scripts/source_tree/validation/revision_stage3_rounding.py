"""Stage 3 rounded-core-shell/size grid, same 209 frames and two withheld-q folds."""
from revision_common import *
from revision_models import fit_rows,evaluate_fixed
from revision_gpu_fourier import GPUCZT2
from revision_batch_fit import BatchLibrary
import cupy as cp
from functools import lru_cache
import time
stage=OUT/"03_rounding";stage.mkdir(exist_ok=True);start=time.perf_counter()
cfg,data,exp=id57_data();data.pop("clean_patterns",None)
exp=np.maximum(exp,0)
roi=data["roi"]
with np.load(OUT/"02_baseline/evaluation_masks.npz") as f:train=f["train"];test=f["test"]
radii=[0.,.5,1.,2.,4.,8.,12.];scales=[.9,.95,1.,1.05,1.1];nr=len(radii);ns=len(scales);nori=240
parameters=[{"radius_base_nm":r,"scale":s,"side_nm":52*s,"rounding_radius_nm":r*s,"orientation_index":oi,
 "candidate":ri*nori*ns+oi*ns+si} for ri,r in enumerate(radii) for oi in range(nori) for si,s in enumerate(scales)]
save_csv(stage/"candidate_parameters.csv",parameters)
plans=[GPUCZT2(768,144,data["qaxis"]*s) for s in scales]
geometry=[id57_meshes(rounding=r,segments=16) for r in radii]
n_candidates=len(parameters);score_files={}
for name in ["full","heldout_train","heldout_eval","reverse_train","reverse_eval"]:
    p=stage/f"{name}_scores.npy"
    score_files[name]=np.lib.format.open_memmap(p,mode="r+" if p.exists() else "w+",dtype="float32",shape=(209,n_candidates))
checkpoint=stage/"rounding_group_checkpoint.json"
done=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
qa=[]
for ri,r in enumerate(radii):
    if ri in done:continue
    library=np.empty((nori*ns,401,401),dtype=np.float32)
    for oi,rotation in enumerate(data["rotation_matrices"]):
        projection,diag=id57_projection(rotation.astype(float),768,144,geometry[ri],"gpu")
        expected_mass=sm.signed_mesh_volume(geometry[ri][0].vertices,geometry[ri][0].faces)+.5*sm.signed_mesh_volume(geometry[ri][1].vertices,geometry[ri][1].faces)
        mass_error=float(projection.sum()*(144/768)**2/expected_mass-1)
        assert abs(mass_error)<1e-3, ("Mesh mass drift",r,oi,mass_error)
        diagnostic={"radius":r,"orientation":oi,"mass_relative_error":mass_error,
          "bad_balance_pixels":diag["outer"]["bad_balance_pixels"]+diag["core"]["bad_balance_pixels"]}
        with (stage/"projection_diagnostics.jsonl").open("a",encoding="utf-8") as df:df.write(json.dumps(safe(diagnostic))+"\n")
        for si,plan in enumerate(plans):
            raw=plan.intensity(projection)
            library[oi*ns+si]=raw.astype(np.float32)
        if oi==0:
            cpu=czt_intensity(projection,144,data["qaxis"])
            qa.append({"radius_base_nm":r,"gpu_cpu_czt_intensity_relative_l2":rel_l2(library[2],cpu)})
        if oi%80==0:print("Rounded generation radius",r,"orientation",oi,"elapsed",round(time.perf_counter()-start,1),flush=True)
    batch={name:BatchLibrary(cp.asarray(library[:,m])) for name,m in [("full",roi),("train",train),("test",test)]}
    sl=slice(ri*nori*ns,(ri+1)*nori*ns)
    for frame,image in enumerate(exp):
        full,_,_=batch["full"].fit(image[roi])
        tr,a,b=batch["train"].fit(image[train]);te=batch["test"].evaluate(image[test],a,b,batch["train"].mean)
        rev,a,b=batch["test"].fit(image[test]);ev=batch["train"].evaluate(image[train],a,b,batch["test"].mean)
        for name,values in [("full",full),("heldout_train",tr),("heldout_eval",te),("reverse_train",rev),("reverse_eval",ev)]:
            score_files[name][frame,sl]=values
        if frame%70==0:print("Rounded scoring radius",r,"frame",frame,"elapsed",round(time.perf_counter()-start,1),flush=True)
    del library,batch;cp.get_default_memory_pool().free_all_blocks()
    for a in score_files.values():a.flush()
    done.append(ri);dump(checkpoint,done);dump(stage/"rounding_gpu_czt_qa.json",qa)
@lru_cache(maxsize=24)
def projection_cached(ri,oi):
    return id57_projection(data["rotation_matrices"][oi].astype(float),768,144,geometry[ri],"gpu")[0]
def raw_candidate(index):
    ri=index//(nori*ns);part=index%(nori*ns);oi=part//ns;si=part%ns
    return plans[si].intensity(projection_cached(ri,oi))
rows=[]
for frame,image in enumerate(exp):
    for mode,fitmask,evalmask,key in [("full",roi,roi,"full"),("heldout",train,test,"heldout_train"),
        ("heldout_reverse",test,train,"reverse_train")]:
        scores=np.asarray(score_files[key][frame]);order=np.argsort(scores)[::-1]
        count=8 if scores[order[0]]-scores[order[8]]>1e-4 else 24
        ids=order[:count];raws=[raw_candidate(int(i)) for i in ids]
        exact,a,b,x,pred=fit_rows(image[fitmask],np.stack([raw[fitmask] for raw in raws]))
        best=int(np.argmax(exact));index=int(ids[best]);raw=raws[best]
        ev,fitted=evaluate_fixed(image[evalmask],raw[evalmask],a[best],b[best],float(raw[fitmask].mean()))
        par=parameters[index]
        row={"frame":frame,"mode":mode,**par,"fit_score":float(exact[best]),"eval_score":ev,
          "signal_scale_normalized":a[best],"background":b[best],"gpu_cpu_winner_score_error":float(scores[index]-exact[best]),
          "same_approximate_best":index==int(order[0]),"score_gap_second":float(scores[order[0]]-scores[order[1]]),
          "on_radius_upper_bound":par["radius_base_nm"]==12.,"on_scale_bound":par["scale"] in [.9,1.1]}
        rows.append(row)
        if mode=="full" and frame in [0,1,11,129,67,176,54,58]:
            np.savez_compressed(stage/f"rounded_fit_frame_{frame}.npz",experimental=image,roi=roi,
              fitted=(a[best]*raw/float(raw[fitmask].mean())+b[best]).astype(np.float32),
              projection=projection_cached(index//(nori*ns),par["orientation_index"]),scale=par["scale"])
    if frame%25==0:print("Rounded exact refinement",frame,"elapsed",round(time.perf_counter()-start,1),flush=True)
    if frame%10==0:save_csv(stage/"id57_rounding_results.csv",rows)
save_csv(stage/"id57_rounding_results.csv",rows)
baseline=json.loads((OUT/"02_baseline/baseline_checkpoint.json").read_text())
baseline_lookup={(r["frame"],r["mode"]):r for r in baseline}
summary={}
rng=np.random.default_rng(20261009)
for mode in ["full","heldout","heldout_reverse"]:
    selected=[r for r in rows if r["mode"]==mode]
    d=np.array([r["eval_score"]-baseline_lookup[r["frame"],mode]["mesh_eval_score"] for r in selected])
    e=np.array([r["eval_score"]-baseline_lookup[r["frame"],mode]["baseline_eval_score"] for r in selected])
    boot=rng.choice(d,size=(10000,len(d)),replace=True).mean(axis=1)
    summary[mode]={"n":len(selected),"mean_rounded_score":float(np.mean([r["eval_score"] for r in selected])),
       "mean_rounded_minus_fixed_sharp":float(d.mean()),"mean_rounded_minus_ellipsoid":float(e.mean()),
       "rounded_better_than_sharp_count":int((d>0).sum()),"rounded_better_than_ellipsoid_count":int((e>0).sum()),
       "frame_bootstrap_delta_95_interval":np.quantile(boot,[.025,.975]).tolist(),
       "radius_upper_bound_count":sum(r["on_radius_upper_bound"] for r in selected),
       "scale_bound_count":sum(r["on_scale_bound"] for r in selected)}
dump(stage/"id57_rounding_summary.json",{"results":summary,"radii_base_nm":radii,"scales":scales,"orientation_count":240,
 "candidate_count":n_candidates,"source_transform":"research GPU complex128 CZT with production CPU coordinate convention; independently compared",
 "density_model":"Pd rounded outer cube plus 0.5 density contrast in clipped octahedral core; isotropic scaling of both boundaries",
 "nuisance_fit":"same exact active-set nonnegative signal/background as the physical baseline",
 "validation":"same two radial-block folds and complete 209-frame cohort; no score-based exclusion",
 "cautions":["Parameters are conditional on the selected core-shell family, planar approximation and finite library.",
  "Upper-bound winners and near-tied orientations do not establish precise rounding radii.",
  "Curve-aware re-evaluation of selected candidates and q-cutoff sensitivity are required before interpreting edge rounding as a structural measurement."],
 "max_gpu_cpu_winner_score_error":max(abs(r["gpu_cpu_winner_score_error"]) for r in rows),"elapsed_s":time.perf_counter()-start})
print("Rounding complete",json.dumps(safe(summary)),flush=True)
