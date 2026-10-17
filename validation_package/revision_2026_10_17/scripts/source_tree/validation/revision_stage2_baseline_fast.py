"""Stage 2 all 209 frames. GPU global/local search, exact CPU final fitting."""
from revision_common import *
from revision_models import *
from scipy.optimize import differential_evolution,minimize
import time
stage=OUT/"02_baseline";stage.mkdir(exist_ok=True);start=time.perf_counter()
cfg,data,exp=id57_data();data.pop("clean_patterns",None)
roi=data["roi"];qx,qy=np.meshgrid(data["qaxis"],data["qaxis"])
yy,xx=np.indices(roi.shape);r=np.hypot(xx-200,yy-200)
train=roi&(np.floor((r-45)/10).astype(int)%2==0);test=roi&~train
np.savez_compressed(stage/"evaluation_masks.npz",roi=roi,train=train,test=test)
with np.load(OUT/"01_consistency/id57_mesh_czt_library_n768.npz") as f:
    mesh=np.maximum(f["clean_patterns"]-8,0)
bounds=[(10.,60.),(.25,1.),(0.,np.pi)]
mesh_gpu={k:cp.asarray(mesh[:,m]) for k,m in [("full",roi),("heldout",train),("heldout_reverse",test)]}
all_rows=[];qa=[];selected={0,1,11,129,67,176,54,58}
checkpoint=stage/"baseline_checkpoint.json"
if checkpoint.exists():
    all_rows=json.loads(checkpoint.read_text());done={(int(x["frame"]),x["mode"]) for x in all_rows}
else:done=set()
for frame,image in enumerate(exp):
    y=np.maximum(image,0)
    for mode,fitmask in [("full",roi),("heldout",train),("heldout_reverse",test)]:
        if (frame,mode) in done:continue
        ids=np.flatnonzero(fitmask);sid=ids[np.linspace(0,len(ids)-1,min(6000,len(ids))).astype(int)]
        objective=GPUEllipsoid(qx.ravel()[sid],qy.ravel()[sid],y.ravel()[sid])
        starts=[];de_records=[]
        for seed in [57000+frame,157000+frame]:
            opt=differential_evolution(objective,bounds,rng=np.random.default_rng(seed),
                maxiter=110,popsize=12,polish=False,vectorized=True,updating="deferred",tol=1e-6)
            starts.append(opt.x);de_records.append({"seed":seed,"score":-float(opt.fun),"success":bool(opt.success),"iterations":opt.nit})
        full_gpu=GPUEllipsoid(qx[fitmask],qy[fitmask],y[fitmask])
        def full_objective(p):return float(full_gpu(p)[0])
        opts=[minimize(full_objective,p,method="Nelder-Mead",bounds=bounds,
            options={"maxiter":120,"xatol":2e-5,"fatol":3e-7}) for p in starts]
        exact=[]
        for opt in opts:
            base=ellipse_raw(opt.x,qx[fitmask],qy[fitmask])
            exact.append(float(fit_rows(y[fitmask],base)[0][0]))
        best=opts[int(np.argmax(exact))];p=best.x
        base=ellipse_raw(p,qx[fitmask],qy[fitmask]);bs,ba,bb,bn,bfit=fit_rows(y[fitmask],base)
        approximate=GPUScorer(y[fitmask]).scores(mesh_gpu[mode])
        top=np.argsort(approximate)[::-1][:10]
        ms,ma,mb,mn,mfit=fit_rows(y[fitmask],mesh[top][:,fitmask]);j=int(np.argmax(ms));ti=int(top[j])
        evalmask=roi if mode=="full" else (test if mode=="heldout" else train)
        bscore,bpred=evaluate_fixed(y[evalmask],ellipse_raw(p,qx[evalmask],qy[evalmask]),ba[0],bb[0],float(base.mean()))
        mscore,mpred=evaluate_fixed(y[evalmask],mesh[ti,evalmask],ma[j],mb[j],float(mesh[ti,fitmask].mean()))
        row={"frame":frame,"mode":mode,"ellipsoid_major_semi_axis_nm":p[0],"ellipsoid_minor_semi_axis_nm":p[0]*p[1],
            "projected_aspect_ratio":p[1],"in_plane_angle_deg":np.degrees(p[2]),"baseline_fit_score":float(bs[0]),
            "mesh_fit_score":float(ms[j]),"baseline_eval_score":bscore,"mesh_eval_score":mscore,"mesh_minus_baseline":mscore-bscore,
            "mesh_best_template":ti,"baseline_background":bb[0],"mesh_background":mb[j],
            "fit_pixels":int(fitmask.sum()),"eval_pixels":int(evalmask.sum()),
            "at_parameter_bound":bool(p[0]<10.1 or p[0]>59.9 or p[1]<.251 or p[1]>.999),
            "final_candidates":2,"final_cpu_score_range":max(exact)-min(exact),
            "local_search_success":bool(best.success),"gpu_cpu_best_score_difference":float(-best.fun-bs[0])}
        all_rows.append(row);dump(checkpoint,all_rows);save_csv(stage/"id57_baseline_results.csv",all_rows)
        if frame in [0,1,11]:
            exact_mesh=fit_rows(y[fitmask],mesh[:,fitmask])[0]
            qa.append({"frame":frame,"mode":mode,"gpu_cpu_max_score_error":float(np.max(np.abs(approximate-exact_mesh))),
                "same_best":int(np.argmax(exact_mesh))==ti})
            dump(stage/"baseline_qa.json",qa)
        if mode=="full" and frame in selected:
            bfull=ellipse_raw(p,qx.ravel(),qy.ravel()).reshape(roi.shape)
            np.savez_compressed(stage/f"fit_frame_{frame}.npz",experimental=image,
               baseline_fit=(ba[0]*bfull/base.mean()+bb[0]).astype(np.float32),
               mesh_fit=(ma[j]*mesh[ti]/mesh[ti,fitmask].mean()+mb[j]).astype(np.float32),roi=roi)
        if frame%10==0:print("Baseline",frame,mode,"scores",round(bscore,4),round(mscore,4),"elapsed",round(time.perf_counter()-start,1),flush=True)
rng=np.random.default_rng(20261009);summary={}
for mode in ["full","heldout","heldout_reverse"]:
    rows=[r for r in all_rows if r["mode"]==mode];delta=np.array([r["mesh_minus_baseline"] for r in rows])
    boots=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
    summary[mode]={"n":len(rows),"mean_delta":float(delta.mean()),"median_delta":float(np.median(delta)),
        "bootstrap_frame_mean_95_interval":np.quantile(boots,[.025,.975]).tolist(),"positive_count":int((delta>0).sum()),
        "mean_mesh_score":float(np.mean([r["mesh_eval_score"] for r in rows])),
        "mean_baseline_score":float(np.mean([r["baseline_eval_score"] for r in rows])),
        "parameter_bound_count":sum(r["at_parameter_bound"] for r in rows),
        "nonconverged_local_count":sum(not r["local_search_success"] for r in rows)}
byframe={i:np.mean([r["mesh_minus_baseline"] for r in all_rows if r["frame"]==i and r["mode"]!="full"]) for i in range(len(exp))}
delta=np.array(list(byframe.values()));boots=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
summary["twofold_heldout"]={"n_frames":209,"mean_delta":float(delta.mean()),"bootstrap_frame_mean_95_interval":np.quantile(boots,[.025,.975]).tolist(),"positive_count":int((delta>0).sum())}
dump(stage/"id57_baseline_summary.json",{"results":summary,"bounds":bounds,"n_frames":209,
 "nuisance_fit":"same exact active-set nonnegative linear signal plus constant background for both models",
 "legacy_difference":"Original ID57 independently clipped unconstrained coefficients; stage 1 preserves it. This physical-baseline analysis uses exact nonnegative coefficients and signal-only templates.",
 "holdout":"alternating 10-pixel radial blocks, training-only parameter and nuisance estimation; withheld pixels evaluated without refitting",
 "identifiability":"Projected ellipse axes/angle only; beam-direction extent and unique 3D ellipsoid/spheroid orientation are not determined.",
 "noise_model":"ADU; no unsupported Poisson photon likelihood","optimizer":"two seeded DE searches on 6000 training pixels, full-fit-mask local search, exact CPU final coefficients/scores",
 "elapsed_s":time.perf_counter()-start})
print("Baseline complete",json.dumps(safe(summary)),flush=True)
