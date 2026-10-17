"""Beam-amplitude multiplication after projection: controlled and fixed-seed tests.

No existing manuscript, geometry, classifications or experimental arrays are edited.
The sharp edge is a mechanism control, not a measured CXIDB probe profile.
"""
from revision_common import *
from revision_gpu_fourier import GPUCZT2
from revision_models import fit_rows,evaluate_fixed
from scipy.special import ndtr,ndtri
import argparse,time,platform
import cupy as cp

DEST=ROOT/"output/beam_truncation_2026-10-13"
DEST.mkdir(parents=True,exist_ok=True)

def axes(n,extent=144.):
    a=(np.arange(n)-(n-1)/2)*extent/n
    return a,np.meshgrid(a,a)

def field(par,X,Y):
    if par["kind"]=="uniform":return np.ones_like(X)
    w=par["fwhm_nm"]
    # w is the INTENSITY FWHM; apply its square-root AMPLITUDE before FT.
    g=np.exp(-2*np.log(2)*(((X-par.get("bx_nm",0))/w)**2+((Y-par.get("by_nm",0))/w)**2))
    if par["kind"]=="gaussian":return g
    theta=np.deg2rad(par["normal_deg"]);u=X*np.cos(theta)+Y*np.sin(theta)
    cut=par["cut_nm"];width=par["amplitude_edge_10_90_nm"]
    m=(u<=cut).astype(float) if width==0 else ndtr((cut-u)/(width/(2*ndtri(.9))))
    return g*m

def spec(name,kind,**kw):return {"name":name,"kind":kind,**kw}

def profiles():
    p=[spec("uniform","uniform")]
    for w in [100.,200.]:
        p.append(spec(f"gaussian{int(w)}_center","gaussian",fwhm_nm=w,bx_nm=0.,by_nm=0.))
        for angle in [0,45,90,135]:
            t=np.deg2rad(angle)
            p.append(spec(f"gaussian{int(w)}_offset_{angle}","gaussian",fwhm_nm=w,bx_nm=w/2*np.cos(t),by_nm=w/2*np.sin(t)))
    for angle in [0,30,60,90,120,150]:
        for cut in [-12.,0.,12.]:
            for width in [0.,2.,8.]:
                p.append(spec(f"edge_n{angle}_cut{cut:g}_w{width:g}","edge",fwhm_nm=200.,normal_deg=float(angle),cut_nm=cut,amplitude_edge_10_90_nm=width))
    assert len(p)==65
    return p

def diagnostics(P,E,I,qaxis,normal=35.):
    qx,qy=np.meshgrid(qaxis,qaxis);r=np.hypot(qx,qy)
    phi=np.mod(np.rad2deg(np.arctan2(qy,qx)),180.)
    ann=(r>=.44)&(r<=.89)
    delta=np.abs((phi-normal+90)%180-90)
    sector=ann&(delta<=6)
    mass=float(P.sum())
    return {"sector_normal_deg":normal,"sector_half_width_deg":6.,"annular_intensity_sum":float(I[ann].sum()),
        "normal_sector_intensity_sum":float(I[sector].sum()),"normal_sector_fraction":float(I[sector].sum()/I[ann].sum()),
        "amplitude_weighted_mass_fraction":float((P*E).sum()/mass),
        "mass_weighted_intensity_fraction":float((P*E*E).sum()/mass)}

def projection(case,n):
    _,(X,Y)=axes(n)
    if case=="smooth_sphere":
        return 2*np.sqrt(np.maximum(26.**2-X*X-Y*Y,0)),{}
    t=np.deg2rad(25.);R=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1.]])
    return id57_projection(R,n,144.,id57_meshes(rounding=8. if case=="rounded_core_shell" else 0.),"gpu")

def run():
    started=time.perf_counter();cfg,data,exp=id57_data();data.pop("clean_patterns",None)
    q=data["qaxis"];n=768;extent=144.;a,(X,Y)=axes(n);plan=GPUCZT2(n,extent,q)
    source_paths=[ROOT/"uploaded_SAXS_simulator/saxs_module.py",ROOT/"validation/revision_common.py",ROOT/"validation/revision_gpu_fourier.py",ROOT/"validation/revision_models.py",
      ROOT/"public_data/CXIDB_ID_57/README",ROOT/"public_data/CXIDB_ID_57/normal-incidence-pattern.cxi",OUT/"03_rounding/id57_rounding_results.csv",OUT/"02_baseline/evaluation_masks.npz",
      ROOT/"output/revision_manuscript_2026-10-13/M_Kim_JSR_R10_revision.docx"]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    source_info={"original_table_focus":"approximately100nm cross-sectiondiameter; FWHMnotexplicit","depositor_readme_focus":"approximately200nm; widthdefinitionnotexplicit",
      "profile_status":"No shot-resolved probe field, beam position or measured sharp edge in selectedCXI file.",
      "urls":["https://www.nature.com/articles/sdata201748.pdf","https://www.cxidb.org/id-57.html","https://journals.iucr.org/s/issues/2015/03/00/yi5010/index.html"],
      "interpretation":"100/200nm intensityFWHM Gaussians are sensitivity assumptions, not resolved measurements. The edge is a hypothetical mechanism control."}
    dump(DEST/"source_conditions.json",source_info);dump(DEST/"source_hashes.json",hashes)
    allprofiles=profiles();dump(DEST/"profile_parameters.json",allprofiles)
    controlled=[spec("uniform","uniform"),spec("gaussian200_offset","gaussian",fwhm_nm=200.,bx_nm=100.,by_nm=0.),
       spec("gaussian100_offset","gaussian",fwhm_nm=100.,bx_nm=50.,by_nm=0.),
       *[spec(f"edge35_w{w:g}","edge",fwhm_nm=200.,normal_deg=35.,cut_nm=0.,amplitude_edge_10_90_nm=w) for w in [0.,2.,8.,16.]],
       spec("edge0_w0","edge",fwhm_nm=200.,normal_deg=0.,cut_nm=0.,amplitude_edge_10_90_nm=0.)]
    dump(DEST/"controlled_parameters.json",controlled)
    rows=[];checks=[]
    for case in ["smooth_sphere","sharp_core_shell","rounded_core_shell"]:
        P,diag=projection(case,n);arrays={"projection":P,"axis_nm":a,"qaxis_nm_inv":q}
        for par in controlled:
            E=field(par,X,Y);I=plan.intensity(P*E)
            normal=par.get("normal_deg",35.)
            rows.append({"case":case,"profile":par["name"],**diagnostics(P,E,I,q,normal)})
            arrays[par["name"]+"_amplitude_mask"]=E;arrays[par["name"]+"_intensity"]=I
            assert np.isfinite(I).all() and I.min()>=0 and E.min()>=0 and E.max()<=1+1e-12
            if par["name"] in ["uniform","edge35_w0","edge35_w8"]:
                cpu=czt_intensity(P*E,extent,q)
                checks.append({"case":case,"profile":par["name"],"gpu_cpu_intensity_rel_l2":rel_l2(I,cpu)})
        np.savez_compressed(DEST/(case+".npz"),**arrays)
        print("Controlled",case,"complete",round(time.perf_counter()-started,1),flush=True)
    save_csv(DEST/"controlled_metrics.csv",rows)

    # Direct summation on the same masked discrete input is independent of CZT.
    P,_=projection("smooth_sphere",n);E=field(controlled[3],X,Y)
    A=cp.asnumpy(plan.amplitude(P*E));direct=[]
    for iy,ix in [(200,200),(230,250),(281,261),(341,327),(119,80)]:
        exact=np.einsum("y,yx,x->",np.exp(-1j*q[iy]*a),P*E,np.exp(-1j*q[ix]*a))*((extent/n)**2)
        direct.append({"iy":iy,"ix":ix,"abs_error_over_q0":float(abs(A[iy,ix]-exact)/abs(A[200,200]))})
    q0_rel=abs(A[200,200]-(P*E).sum()*(extent/n)**2)/abs(A[200,200])

    convergence=[]
    for nn in [384,768,1536]:
        pp,_=projection("smooth_sphere",nn);aa,(xx,yy)=axes(nn);pl=GPUCZT2(nn,extent,q)
        for par in [controlled[0],controlled[3],controlled[5]]:
            ee=field(par,xx,yy);ii=pl.intensity(pp*ee)
            convergence.append({"grid":nn,"profile":par["name"],**diagnostics(pp,ee,ii,q)})
            np.savez_compressed(DEST/f"convergence_{nn}_{par['name']}.npz",intensity=ii,projection=pp,amplitude_mask=ee)
    save_csv(DEST/"grid_metrics.csv",convergence)

    # Full complex Fourier-plane identity. This does not interpolate a3Dcache.
    cache=cp.fft.fft2(cp.asarray(P,dtype=cp.complex128));recovered=cp.fft.ifft2(cache)
    oracle=cp.asnumpy(cp.abs(plan.amplitude(recovered*cp.asarray(E)))**2)
    ifft_identity=rel_l2(oracle,np.abs(A)**2)
    nn=16;aa,(xx,yy)=axes(nn,80.);pp=2*np.sqrt(np.maximum(26.**2-xx*xx-yy*yy,0));ee=field(controlled[3],xx,yy)
    FP=np.fft.fft2(pp);FE=np.fft.fft2(ee);conv=np.zeros_like(FP);idx=np.arange(nn)
    for k in range(nn):
        for l in range(nn):conv[k,l]=np.sum(FP*FE[(k-idx[:,None])%nn,(l-idx[None,:])%nn])/(nn*nn)
    convolution_identity=rel_l2(conv,np.fft.fft2(pp*ee))

    # Locked historical TRAINING-selected geometry; probe choice and nuisance
    # coefficients use training bands only. Test bands never select a profile.
    with np.load(OUT/"02_baseline/evaluation_masks.npz") as f:train=f["train"].astype(bool);test=f["test"].astype(bool)
    prior=list(csv.DictReader((OUT/"03_rounding/id57_rounding_results.csv").open(encoding="utf-8")))
    selections=[];candidate_rows=[];baseline_errors=[]
    for frame in [0,1]:
        for mode,tr,te in [("heldout",train,test),("heldout_reverse",test,train)]:
            old=next(r for r in prior if int(r["frame"])==frame and r["mode"]==mode)
            s=float(old["scale"]);rot=data["rotation_matrices"][int(old["orientation_index"])].astype(float)
            pp,_=id57_projection(rot,n,extent,id57_meshes(rounding=float(old["radius_base_nm"])),"gpu")
            pl=GPUCZT2(n,extent,q*s);raw=[]
            for par in allprofiles:raw.append(pl.intensity(pp*field(par,X*s,Y*s))*s**6)
            raw=np.asarray(raw);y=np.maximum(exp[frame],0).astype(float)
            scores,aa,bb,_,_=fit_rows(y[tr],raw[:,tr]);means=raw[:,tr].mean(axis=1)
            evalscores=[]
            for i,par in enumerate(allprofiles):
                ev,pred=evaluate_fixed(y[te],raw[i,te],aa[i],bb[i],means[i]);evalscores.append(ev)
                candidate_rows.append({"frame":frame,"fold":mode,"profile_index":i,"profile":par["name"],"kind":par["kind"],"train_score":float(scores[i]),"test_score":ev,
                  "signal_normalized":float(aa[i]),"background":float(bb[i]),"train_raw_mean":float(means[i])})
            baseline_errors.append(abs(evalscores[0]-float(old["eval_score"])))
            assert baseline_errors[-1]<2e-6,("Prior baseline changed",frame,mode,baseline_errors[-1])
            arr={"experimental":y,"projection":pp,"physical_axis_nm":a*s,"qaxis_nm_inv":q,"train":tr,"test":te,"roi":data["roi"],"raw_intensities":raw}
            for family in ["uniform","gaussian","edge"]:
                ids=[i for i,par in enumerate(allprofiles) if par["kind"] in ["uniform",family]]
                best=max(ids,key=lambda i:scores[i]);par=allprofiles[best]
                pred=aa[best]*raw[best]/means[best]+bb[best]
                arr[family+"_fitted"]=pred;arr[family+"_amplitude_mask"]=field(par,X*s,Y*s)
                selections.append({"frame":frame,"fold":mode,"family":family,"selected_profile":par["name"],"selected_index":best,
                   "geometry_fixed":True,"historical_radius_base_nm":float(old["radius_base_nm"]),"historical_scale":s,"historical_orientation_index":int(old["orientation_index"]),
                   "train_score":float(scores[best]),"test_score":evalscores[best],"test_delta_vs_uniform":evalscores[best]-evalscores[0],
                   "signal_normalized":float(aa[best]),"background":float(bb[best]),"train_raw_mean":float(means[best])})
            np.savez_compressed(DEST/f"pilot_frame{frame}_{mode}.npz",**arr)
            print("Fixed-geometry pilot",frame,mode,"complete",round(time.perf_counter()-started,1),flush=True)
    save_csv(DEST/"pilot_all_candidates.csv",candidate_rows);save_csv(DEST/"pilot_selections.csv",selections)

    # GPU-resident same2D-transform tail only. Plane alreadycached, projection
    # or3Dgeneration/interpolation, setup, I/O and beam construction excluded.
    benchmarks=[]
    for nn in [384,768,1536]:
        pp,_=projection("smooth_sphere",nn);_,(xx,yy)=axes(nn);ee=field(controlled[3],xx,yy)
        pg=cp.asarray(pp,dtype=cp.complex128);eg=cp.asarray(ee);fg=cp.fft.fft2(pg);pl=GPUCZT2(nn,extent,q)
        def action(kind):
            if kind=="projection_mask_czt":return pl.amplitude(pg*eg)
            if kind=="cached_plane_ifft_mask_czt":return pl.amplitude(cp.fft.ifft2(fg)*eg)
            return cp.fft.ifft2(fg)
        for kind in ["projection_mask_czt","cached_plane_ifft_mask_czt","ifft_only"]:
            for _ in range(3):action(kind)
            cp.cuda.Stream.null.synchronize();times=[]
            for _ in range(12):
                start,end=cp.cuda.Event(),cp.cuda.Event();start.record();res=action(kind);end.record();end.synchronize();times.append(cp.cuda.get_elapsed_time(start,end))
            benchmarks.append({"grid":nn,"operation":kind,"median_gpu_ms":float(np.median(times)),"min_gpu_ms":float(min(times)),"max_gpu_ms":float(max(times)),"repetitions":12})
        cp.get_default_memory_pool().free_all_blocks()
    save_csv(DEST/"same_plane_tail_timings.csv",benchmarks)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    qa={"status":"numerical_checks_passed_pending_saved_array_audit_and_visual_review","source_hashes_preserved":True,"controlled_cases":3,"controlled_profiles":8,"pilot_fixed_geometries":4,"pilot_profiles_each":65,
       "gpu_cpu_checks":checks,"direct_dft_checks":direct,"q0_relative_error":float(q0_rel),"full_plane_ifft_mask_czt_intensity_rel_l2":ifft_identity,
       "direct_circular_convolution_amplitude_rel_l2":convolution_identity,"prior_uniform_baseline_max_score_difference":max(baseline_errors),
       "mask_semantics":"Field amplitude, never diffraction intensitymask. No addednoise, saturation, detectorPSF, complexprobephase or jet model.",
       "benchmark_scope":"GPU-resident fullcomplex2Dplane versus projection tail; NOT anend-to-end3DCZT/cache/interpolation comparison.",
       "environment":{"python":sys.version,"numpy":np.__version__,"cupy":cp.__version__,"platform":platform.platform()},"elapsed_seconds":time.perf_counter()-started}
    assert max(c["gpu_cpu_intensity_rel_l2"] for c in checks)<1e-10
    assert max(c["abs_error_over_q0"] for c in direct)<1e-10
    assert q0_rel<1e-10 and ifft_identity<1e-10 and convolution_identity<1e-10
    dump(DEST/"validation_audit.json",qa)
    print(json.dumps(qa),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["run"]);args=ap.parse_args();run()
