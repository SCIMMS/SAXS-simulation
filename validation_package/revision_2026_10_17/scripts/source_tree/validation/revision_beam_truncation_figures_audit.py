"""CPU saved-array verification, scientific figures, and qualified interpretation."""
from pathlib import Path
import csv,json,hashlib,os,sys
import numpy as np
from scipy.optimize import nnls
from scipy.signal import CZT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"output/beam_truncation_2026-10-13";FIG=OUT/"figures";FIG.mkdir(exist_ok=True)
def read(name):return json.loads((OUT/name).read_text(encoding="utf-8"))
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
def rows(name):return list(csv.DictReader((OUT/name).open(encoding="utf-8")))
def hx(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(y,p):
    a=np.log1p(np.maximum(y,0));b=np.log1p(np.maximum(p,0))
    return float(np.corrcoef(a,b)[0,1])
def cpu_amp(p,extent,q):
    n=p.shape[0];du=extent/n;u0=-(n-1)*du/2;dq=q[1]-q[0]
    plan=CZT(n,len(q),w=np.exp(-1j*dq*du),a=np.exp(1j*q[0]*du))
    phase=du*np.exp(-1j*q*u0)
    return plan(plan(p,axis=1)*phase[None,:],axis=0)*phase[:,None]
profiles=read("profile_parameters.json");controlled=read("controlled_parameters.json")
metrics=rows("controlled_metrics.csv");allcand=rows("pilot_all_candidates.csv");selected=rows("pilot_selections.csv")
max_metric_error=0.;max_score_error=0.;max_selected_cpu_error=0.
for case in ["smooth_sphere","sharp_core_shell","rounded_core_shell"]:
    with np.load(OUT/(case+".npz")) as z:
        P=z["projection"];q=z["qaxis_nm_inv"];qx,qy=np.meshgrid(q,q);r=np.hypot(qx,qy);phi=np.mod(np.rad2deg(np.arctan2(qy,qx)),180)
        for par in controlled:
            old=next(row for row in metrics if row["case"]==case and row["profile"]==par["name"])
            I=z[par["name"]+"_intensity"];E=z[par["name"]+"_amplitude_mask"];ann=(r>=.44)&(r<=.89)
            sector=ann&(np.abs((phi-float(old["sector_normal_deg"])+90)%180-90)<=6)
            vals={"annular_intensity_sum":I[ann].sum(),"normal_sector_intensity_sum":I[sector].sum(),"normal_sector_fraction":I[sector].sum()/I[ann].sum(),
                "amplitude_weighted_mass_fraction":np.sum(P*E)/np.sum(P),"mass_weighted_intensity_fraction":np.sum(P*E*E)/np.sum(P)}
            for key,val in vals.items():max_metric_error=max(max_metric_error,abs(float(old[key])-val)/max(abs(val),1e-30))
            assert np.isfinite(I).all() and I.min()>=0 and np.all((E>=0)&(E<=1+1e-12))
for frame in [0,1]:
    for fold in ["heldout","heldout_reverse"]:
        with np.load(OUT/f"pilot_frame{frame}_{fold}.npz") as z:
            y=z["experimental"];tr=z["train"];te=z["test"];raw=z["raw_intensities"]
            assert not (tr&te).any() and raw.shape[0]==65
            new=[]
            for i in range(65):
                means=raw[i,tr].mean();x=raw[i,tr]/means
                (a,b),_=nnls(np.column_stack([x,np.ones(len(x))]),y[tr])
                train_score=score(y[tr],a*x+b);test_score=score(y[te],a*raw[i,te]/means+b);new.append(train_score)
                old=next(r for r in allcand if int(r["frame"])==frame and r["fold"]==fold and int(r["profile_index"])==i)
                max_score_error=max(max_score_error,abs(train_score-float(old["train_score"])),abs(test_score-float(old["test_score"])))
            for family in ["uniform","gaussian","edge"]:
                old=next(r for r in selected if int(r["frame"])==frame and r["fold"]==fold and r["family"]==family)
                ids=[i for i,p in enumerate(profiles) if p["kind"] in ["uniform",family]]
                assert max(ids,key=lambda i:new[i])==int(old["selected_index"])
                assert abs(score(y[te],z[family+"_fitted"][te])-float(old["test_score"]))<1e-12
                s=float(old["historical_scale"]);E=z[family+"_amplitude_mask"]
                I=np.abs(cpu_amp(z["projection"]*E,144.,z["qaxis_nm_inv"]*s))**2*s**6
                ref=raw[int(old["selected_index"])]
                max_selected_cpu_error=max(max_selected_cpu_error,float(np.linalg.norm(I-ref)/np.linalg.norm(ref)))
assert max_metric_error<1e-12 and max_score_error<1e-10 and max_selected_cpu_error<1e-10
for p,h in read("source_hashes.json").items():assert hx(ROOT/p)==h

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
shown=["uniform","gaussian200_offset","edge0_w0","edge35_w0","edge35_w2","edge35_w8"]
titles=["Uniform field","Gaussian 200 nm\noffset 100 nm","Sharp edge\nnormal 0°","Sharp edge\nnormal 35°","Smooth edge 35°\n2 nm transition","Smooth edge 35°\n8 nm transition"]
with np.load(OUT/"smooth_sphere.npz") as z:
    P=z["projection"];a=z["axis_nm"];q=z["qaxis_nm_inv"];qx,qy=np.meshgrid(q,q);rr=np.hypot(qx,qy)
    roi=(rr>=.267)&(rr<=1.128);dx=a[1]-a[0];I0=(P.sum()*dx*dx)**2
    fig,axs=plt.subplots(2,6,figsize=(16.0,6.7),layout="constrained")
    for j,name in enumerate(shown):
        E=z[name+"_amplitude_mask"];I=z[name+"_intensity"]
        im0=axs[0,j].imshow(P*E,origin="lower",extent=[a[0],a[-1],a[0],a[-1]],cmap="viridis",vmin=0,vmax=52)
        axs[0,j].set(xlim=(-35,35),ylim=(-35,35),title=titles[j],xlabel="x (nm)")
        d=np.where(roi,np.log10(np.maximum(I/I0,1e-12)),np.nan)
        im1=axs[1,j].imshow(d,origin="lower",extent=[q[0],q[-1],q[0],q[-1]],cmap="magma",vmin=-8,vmax=-2)
        axs[1,j].set(xlim=(-1.13,1.13),ylim=(-1.13,1.13),xlabel="qₓ (nm⁻¹)")
        val=next(r for r in metrics if r["case"]=="smooth_sphere" and r["profile"]==name)
        axs[1,j].set_title(f"Normal-sector share {100*float(val['normal_sector_fraction']):.1f}%",fontsize=9,pad=6)
        if j==0:axs[0,j].set_ylabel("Projected density × field\ny (nm)");axs[1,j].set_ylabel("Diffraction intensity\nqᵧ (nm⁻¹)")
    fig.colorbar(im0,ax=axs[0,:],shrink=.82,pad=.01,label="Density contrast × nm")
    fig.colorbar(im1,ax=axs[1,:],shrink=.82,pad=.01,label="log₁₀[I / uniform I(0)]")
    fig.suptitle("Fixed smooth sphere: illumination boundary controls",fontsize=16,fontweight="bold")
    fig.supxlabel("Sphere diameter 52 nm; planar CZT, grid 768. Common incident-field scale; no dose rescaling.\nSector: edge-normal ±6° and opposite, q = 0.44–0.89 nm⁻¹. Display annulus: q = 0.267–1.128 nm⁻¹.\nAll edge controls include a centered 200 nm intensity-FWHM Gaussian; widths describe amplitude 10–90% transitions.",fontsize=9)
    fig.savefig(FIG/"smooth_sphere_edge_controls.png",dpi=220);fig.savefig(FIG/"smooth_sphere_edge_controls.svg");plt.close(fig)

fig,axs=plt.subplots(2,5,figsize=(13.5,6.5),layout="constrained")
names=["uniform","gaussian200_offset","edge35_w0","edge35_w8","edge0_w0"]
for row,case in enumerate(["sharp_core_shell","rounded_core_shell"]):
    with np.load(OUT/(case+".npz")) as z:
        q=z["qaxis_nm_inv"];qx,qy=np.meshgrid(q,q);r=np.hypot(qx,qy);roi=(r>=.267)&(r<=1.128);P=z["projection"];a=z["axis_nm"];I0=(P.sum()*(a[1]-a[0])**2)**2
        for col,name in enumerate(names):
            im=axs[row,col].imshow(np.where(roi,np.log10(np.maximum(z[name+"_intensity"]/I0,1e-12)),np.nan),origin="lower",extent=[q[0],q[-1],q[0],q[-1]],cmap="magma",vmin=-8,vmax=-2)
            axs[row,col].set(xlabel="qₓ (nm⁻¹)",xticks=[-1,0,1],yticks=[-1,0,1])
            if row==0:axs[row,col].set_title(dict(zip(shown,titles))[name],fontsize=10)
            if col==0:axs[row,col].set_ylabel(("Sharp" if row==0 else "Rounded, r = 8 nm")+"\nqᵧ (nm⁻¹)")
fig.colorbar(im,ax=axs,shrink=.7,pad=.01,label="log₁₀[I / row's uniform I(0)]")
fig.suptitle("Fixed core–shell morphology and orientation: illumination controls",fontsize=15,fontweight="bold")
fig.supxlabel("Outer side 52 nm; core contrast excess 0.5; morphology roll 25°. Beam-boundary orientation changes independently.\nRows use their own uniform q=0 reference; compare illumination within each row. No probe-phase variation, noise or liquid-jet model.",fontsize=9)
fig.savefig(FIG/"core_shell_illumination_controls.png",dpi=220);fig.savefig(FIG/"core_shell_illumination_controls.svg");plt.close(fig)

fig,axs=plt.subplots(2,4,figsize=(12.6,7.2),layout="constrained")
for row,frame in enumerate([0,1]):
    with np.load(OUT/f"pilot_frame{frame}_heldout.npz") as z:
        for col,(key,title) in enumerate([("experimental","Measured"),("uniform_fitted","Uniform"),("gaussian_fitted","Selected Gaussian"),("edge_fitted","Selected edge")]):
            vals=np.where(z["roi"],np.log1p(np.maximum(z[key],0)),np.nan)
            im=axs[row,col].imshow(vals,origin="lower",cmap="viridis",vmin=0,vmax=6)
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
            if col==0:axs[row,col].set_ylabel(f"Frame {frame}")
            if row==0:axs[row,col].set_title(title)
            if col>0:
                family=["uniform","gaussian","edge"][col-1]
                r=next(r for r in selected if int(r["frame"])==frame and r["fold"]=="heldout" and r["family"]==family)
                axs[row,col].text(.5,-.04,f"Test score {float(r['test_score']):.3f}",transform=axs[row,col].transAxes,ha="center",fontsize=10)
fig.colorbar(im,ax=axs,shrink=.7,pad=.01,label="log₁ₚ intensity (ADU)")
fig.suptitle("ID57 two-frame pilot: fixed training-selected morphology",fontsize=15,fontweight="bold")
fig.supxlabel("First radial-band fold displayed; both folds retained in CSV. Illumination choice and signal/background use training only.\nThe 65-profile sensitivity grid is hypothetical; improved prediction does not identify the measured probe or streak source.",fontsize=9)
fig.savefig(FIG/"id57_fixed_geometry_beam_pilot.png",dpi=220);fig.savefig(FIG/"id57_fixed_geometry_beam_pilot.svg");plt.close(fig)

# Compact scientific summary for conversation, retaining the full plates above.
with np.load(OUT/"smooth_sphere.npz") as z:
    P=z["projection"];a=z["axis_nm"];q=z["qaxis_nm_inv"];qx,qy=np.meshgrid(q,q);rr=np.hypot(qx,qy);roi=(rr>=.267)&(rr<=1.128);I0=(P.sum()*(a[1]-a[0])**2)**2
    fig,axs=plt.subplots(2,3,figsize=(9.4,6.8),layout="constrained")
    for j,(name,title) in enumerate([("uniform","Uniform illumination"),("edge35_w0","Sharp illumination edge"),("edge35_w8","Edge smoothed over 8 nm")]):
        E=z[name+"_amplitude_mask"];I=z[name+"_intensity"]
        im0=axs[0,j].imshow(P*E,origin="lower",extent=[a[0],a[-1],a[0],a[-1]],cmap="viridis",vmin=0,vmax=52)
        axs[0,j].set(xlim=(-35,35),ylim=(-35,35),title=title,xlabel="x (nm)",xticks=[-25,0,25],yticks=[-25,0,25])
        im1=axs[1,j].imshow(np.where(roi,np.log10(np.maximum(I/I0,1e-12)),np.nan),origin="lower",extent=[q[0],q[-1],q[0],q[-1]],cmap="magma",vmin=-8,vmax=-2)
        val=next(r for r in metrics if r["case"]=="smooth_sphere" and r["profile"]==name)
        axs[1,j].set(title=f"Sector share: {100*float(val['normal_sector_fraction']):.1f}%",xlabel="qₓ (nm⁻¹)",xticks=[-1,0,1],yticks=[-1,0,1])
        if j==0:axs[0,j].set_ylabel("Projected density × field\ny (nm)");axs[1,j].set_ylabel("Diffraction intensity\nqᵧ (nm⁻¹)")
    fig.colorbar(im0,ax=axs[0,:],shrink=.83,pad=.01,label="Density contrast × nm")
    fig.colorbar(im1,ax=axs[1,:],shrink=.83,pad=.01,label="log₁₀[I / uniform I(0)]")
    fig.suptitle("Same smooth sphere, different illumination",fontsize=16,fontweight="bold")
    fig.supxlabel("Diameter 52 nm; sharp/smooth edges have normal 35° and centered 200 nm Gaussian envelope.\nSector ±6° and opposite, q = 0.44–0.89 nm⁻¹; fixed incident-field scale.\nHypothetical illumination controls, not a measured CXIDB probe.",fontsize=9)
    fig.savefig(FIG/"edge_mechanism_summary.png",dpi=220);fig.savefig(FIG/"edge_mechanism_summary.svg");plt.close(fig)

sector={r["profile"]:float(r["normal_sector_fraction"]) for r in metrics if r["case"]=="smooth_sphere"}
deltas={fam:[float(r["test_delta_vs_uniform"]) for r in selected if r["family"]==fam] for fam in ["gaussian","edge"]}
summary={"status":"numerical_and_saved_array_audits_passed_pending_visual_review","sphere_normal_sector_fraction":sector,
    "gaussian_test_deltas_four_folds":deltas["gaussian"],"edge_test_deltas_four_folds":deltas["edge"],
    "n_independent_experimental_frames":2,"folds_per_frame":2,"candidate_selection":"Training logcorrelationonly; no testrescue or shape/orientationrefit",
    "mechanism_conclusion":"A sharp illumination edge producesa strongdirectionalstreak evenforasmoothsphere. Smoothing suppressesit. OffsetGaussians do not producesimilarnewthin streaksinthiscontrol.",
    "actual_data_conclusion":"OffsetGaussian improvedbothfoldsofthese2fixedgeometryframes; edgefamilygainwasmixed. Thisdoesnotidentifytheactualprobe, excludejetbackground, or establishsinglehitpurity.",
    "numerical_audit":{"controlled_metric_max_relative_difference":float(max_metric_error),"pilot_all260_train_test_NNLS_score_max_difference":max_score_error,"selected12_cpu_gpu_intensity_max_relative_l2":max_selected_cpu_error},
    "workflow_conclusion":"Projection-space amplitude multiplication isdirect. FullcomplexFourierplane IFFT-mask-CZT agreesnumerically; Fourierconvolution isalso valid. No intrinsicimpossibility or end-to-end3DCZT speedclaim."}
dump("summary.json",summary)
dump("chart_contract.json",{"family":"Scientific matrix/controlled comparisons","source":"saved projections, field masks, detector intensities and completefixed-geometrypilot records","units":"nm,nm^-1, relativeintensity andADU","normalization":"Sameuniformq0referencewithineachmorphology; actualdata commonADUloglimits","scope":"Hypotheticalfields; noactualprobemeasurement ornewclassassignment","uncertainty":"Twoexperimentalframes, twofoldseach; nopopulationconfidenceinterval orstructuralidentifiability"})
audit=read("validation_audit.json");audit["saved_array_independent_audit"]=summary["numerical_audit"];audit["status"]=summary["status"];dump("validation_audit.json",audit)
print(json.dumps(summary),flush=True)
