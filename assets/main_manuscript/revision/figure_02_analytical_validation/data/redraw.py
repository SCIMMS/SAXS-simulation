"""Redraw Figure 2 from retained analytical-validation arrays, without rerunning fits.

Only the display transform changes: normalized log10 intensities use a common
epsilon and the signed log residual has an explicitly defined scale.
Run: python data/redraw.py --output-dir PATH. Requires NumPy, Matplotlib and Pillow.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

import argparse
_parser=argparse.ArgumentParser(description="Redraw the current scientific figure from retained public inputs.")
_parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[1]/"images")
_args=_parser.parse_args()
OUT=_args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"assets").is_dir() and (p/"validation_package").is_dir())
SOURCE = ROOT / "assets/main_manuscript/figure_02_analytical_validation/data"
BASE = ROOT / "validation_package/section3_public_synthetic/section3_validation_outputs_public/npz"
EPSILON = 1e-8
CASES = [
    ("Sphere", "sphere_analytical_validation.npz", 0.4),
    ("Ellipsoid", "ellipsoid_analytical_validation.npz", 0.5),
    ("Box", "box_polyhedron_validation.npz", 0.06),
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def basis(ray, roll):
    # Independent elementary Gram-Schmidt/Rodrigues evaluation of the documented
    # detector basis; no simulator or validation helper is imported.
    n = np.asarray(ray, dtype=float)
    n /= np.linalg.norm(n)
    reference = np.array([0., 0., 1.]) if abs(n[2]) < .9 else np.array([1., 0., 0.])
    u = np.cross(reference, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    c, s = math.cos(roll), math.sin(roll)
    return c*u + s*v, c*v - s*u

def sphere_factor(t):
    ans = np.ones_like(t)
    nz = t != 0
    ans[nz] = 3 * (np.sin(t[nz]) - t[nz]*np.cos(t[nz])) / t[nz]**3
    return ans

def independent_reference(label, arrays):
    q = arrays["q"]
    x, y = np.meshgrid(q, q)
    if label == "Sphere":
        return (4*math.pi/3) * sphere_factor(np.hypot(x, y))
    if label == "Ellipsoid":
        u, v = basis([.25, .15, 1.], -.21)
        qlocal = (x[...,None]*u + y[...,None]*v) @ arrays["rotation"]
        axes = np.array([1., .65, .42])
        return (4*math.pi*np.prod(axes)/3) * sphere_factor(np.linalg.norm(qlocal*axes, axis=-1))
    u, v = basis([.2, -.3, 1.], .13)
    qlocal = (x[...,None]*u + y[...,None]*v) @ arrays["rotation"]
    half = np.array([1., .72, .45])
    return np.prod(2*half*np.sinc(qlocal*half/math.pi), axis=-1)

contract = {
    "question": "How closely does mesh projection plus planar CZT reproduce the known analytical intensities?",
    "takeaway": "The stored synthetic planar cases agree quantitatively; the figure does not validate neglect of Ewald curvature.",
    "surface": "Static manuscript Figure 2; same aspect ratio as the retained article figure",
    "family": "Matrix / comparative heatmaps, three rows by three columns",
    "observations": "Three cases, 257 by 257 q samples per case; all samples retained",
    "palette": "Explicit sequential blue-white intensity map and blue-white-orange signed residual map; signed zero and scales are labelled",
    "footprint_in": [6.5, 6.5*876/920],
    "dpi": 600,
    "intensity_transform": "log10(I/I0 + 1e-8), where I0 is the analytical I_ref at q=(0,0), shared by simulated/reference panels within a row",
    "residual": "log10((I_CZT/I0+1e-8)/(I_ref/I0+1e-8)); row-specific symmetric full-data limits, no clipping",
    "q_units": "inverse common reference length ell; ell equals the sphere radius",
    "q_domain": {"qx": [-25,25], "qy": [-25,25], "samples_per_axis":257, "plane":"qz=0"},
    "normalization_warning": "I0 normalization is for visualization only; relative L2 metrics use original unnormalized intensities",
}
(OUT/"figure2_chart_contract.json").write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size":8.0, "axes.labelsize":8.1,
    "xtick.labelsize":7.8,"ytick.labelsize":7.8,"axes.linewidth":.6,
    "svg.fonttype":"none", "pdf.fonttype":42,
    "text.color":"#222222", "axes.labelcolor":"#222222",
})
fig = plt.figure(figsize=contract["footprint_in"],dpi=600,facecolor="white")
intensity_cmap = LinearSegmentedColormap.from_list("intensity_blue",["#102d4f","#3479aa","#a5d1e6","#ffffff"])
residual_cmap = LinearSegmentedColormap.from_list("signed_blue_orange",["#2166ac","#f7f7f7","#b35806"])
W,H=contract["footprint_in"]
panel_w=.215
panel_h=panel_w*W/H
ys=[.702,.431,.160]
xs=[.105,.370,.681]
audit=[]
summary=[]
image_axes=[]
for row,(label,filename,reslim) in enumerate(CASES):
    path=SOURCE/filename
    with np.load(path) as npz:
        z={k:npz[k] for k in npz.files}
    q=z["q"]
    assert np.array_equal(q,np.linspace(-25.,25.,257))
    assert sha(path)==sha(BASE/filename)
    assert z["I_czt"].shape==(257,257)
    assert np.array_equal(z["I_czt"],np.abs(z["A_czt"])**2)
    assert np.array_equal(z["I_ref"],np.abs(z["A_ref"])**2)
    amp_ref=independent_reference(label,z)
    ref_diff=float(np.linalg.norm(amp_ref-z["A_ref"])/np.linalg.norm(z["A_ref"]))
    assert ref_diff<1e-13
    i0=float(z["I_ref"][128,128])
    lc=np.log10(z["I_czt"]/i0+EPSILON)
    lr=np.log10(z["I_ref"]/i0+EPSILON)
    residual=lc-lr
    metric=float(np.linalg.norm(z["I_czt"]-z["I_ref"])/np.linalg.norm(z["I_ref"]))
    assert float(np.max(np.abs(residual)))<reslim
    matrices=[lc,lr,residual]
    handles=[]
    for col,matrix in enumerate(matrices):
        ax=fig.add_axes([xs[col],ys[row],panel_w,panel_h])
        image_axes.append(ax)
        halfstep=float(q[1]-q[0])/2
        norm=Normalize(-8,0) if col<2 else Normalize(-reslim,reslim)
        im=ax.imshow(matrix,extent=[q[0]-halfstep,q[-1]+halfstep,q[0]-halfstep,q[-1]+halfstep],origin="lower",interpolation="nearest",cmap=intensity_cmap if col<2 else residual_cmap,norm=norm)
        handles.append(im)
        ax.set_xlim(-25,25)
        ax.set_ylim(-25,25)
        ax.set_xticks([-25,0,25]);ax.set_yticks([-25,0,25])
        ax.tick_params(direction="out",length=2.5,width=.6,pad=2,labelbottom=row==2,labelleft=col==0)
        if row==2: ax.set_xlabel(r"$q_x\;(\ell^{-1})$",labelpad=2)
        if col==0: ax.set_ylabel(r"$q_y\;(\ell^{-1})$",labelpad=2)
        ax.text(.025,.97,f"({chr(97+row*3+col)})",transform=ax.transAxes,va="top",ha="left",fontsize=8.1,fontweight="bold",bbox=dict(facecolor="white",alpha=.92,edgecolor="none",pad=1.0))
    cax=fig.add_axes([.602,ys[row],.012,panel_h])
    cb=fig.colorbar(handles[0],cax=cax,ticks=[-8,-4,0])
    cb.ax.tick_params(labelsize=7.6,length=2,width=.5,pad=2)
    cb.outline.set_linewidth(.5)
    cax=fig.add_axes([.913,ys[row],.012,panel_h])
    cb=fig.colorbar(handles[2],cax=cax,ticks=[-reslim,0,reslim])
    cb.ax.tick_params(labelsize=7.6,length=2,width=.5,pad=2)
    cb.outline.set_linewidth(.5)
    fig.text(.105,ys[row]+panel_h+.012,label,fontsize=8.7,fontweight="bold",ha="left",va="bottom")
    exp=int(math.floor(math.log10(metric)))
    mantissa=metric/10**exp
    fig.text(xs[2],ys[row]+panel_h+.012,rf"Intensity relative $L_2$: {mantissa:.2f}$\times10^{{{exp}}}$",fontsize=7.4,ha="left",va="bottom")
    rec={
        "case":label,"source_repository_relative":str(path.relative_to(ROOT)).replace("\\","/"),
        "source_sha256":sha(path),"retained_base_identical":True,
        "analytical_I0":i0,"intensity_relative_L2":metric,
        "independent_closed_form_amplitude_relative_L2":ref_diff,
        "residual_min":float(residual.min()),"residual_max":float(residual.max()),
        "residual_color_limits":[-reslim,reslim],"residual_clipped_pixels":int(np.count_nonzero(abs(residual)>reslim)),
        "intensity_normalization":"I0=I_ref(qx=qy=0), common to simulated/reference of this row",
    }
    audit.append(rec)
    summary.append({k:rec[k] for k in ["case","analytical_I0","intensity_relative_L2","independent_closed_form_amplitude_relative_L2","residual_min","residual_max","residual_clipped_pixels"]})

for x,text in [(xs[0]+panel_w/2,"Mesh + CZT"),(xs[1]+panel_w/2,"Analytical reference"),(xs[2]+panel_w/2,"Signed log residual")]:
    fig.text(x,.984,text,fontsize=9.1,fontweight="bold",ha="center",va="top")
fig.text(.105,.081,r"Intensity colors: $\log_{10}(I/I_0+10^{-8})$; $I_0=I_{\rm ref}(0,0)$ per object.",fontsize=7.5,ha="left",va="center")
fig.text(.105,.046,r"Residual: $\Delta\log_{10}I=\log_{10}[(I_{\rm CZT}/I_0+10^{-8})/(I_{\rm ref}/I_0+10^{-8})]$.",fontsize=7.5,ha="left",va="center")
fig.text(.105,.011,r"$q_z=0$; $\ell$ is the common reference length (sphere radius). Residual scales differ by row.",fontsize=7.3,ha="left",va="bottom")
fig.canvas.draw()
# The output pixel frame is intentionally fixed; no tight bbox changes footprint.
png=OUT/"figure_02.png"
svg=OUT/"figure_02.svg"
pdf=OUT/"figure_02.pdf"
fig.savefig(png,dpi=600,facecolor="white")
fig.savefig(svg,facecolor="white")
fig.savefig(pdf,facecolor="white")
plt.close(fig)
with (OUT/"figure2_numeric_summary.csv").open("w",encoding="utf-8",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
caption=(
    "Analytical validation of the planar projected-density kernel. Rows show the unit-radius sphere (a–c), "
    "rotated ellipsoid with semi-axes (1.0, 0.65, 0.42)ℓ (d–f), and exactly polyhedral rectangular box with half-lengths "
    "(1.0, 0.72, 0.45)ℓ (g–i). Columns show mesh+CZT intensity, independently evaluated analytical intensity, and their "
    "signed log residual. Both transverse axes contain 257 samples from −25 to 25 ℓ⁻¹ at q_z=0; ℓ is the common reference "
    "length, equal to the sphere radius. Within each row, I₀=I_ref(0,0) is shared by the simulated and reference panels. "
    "Intensity colors show log₁₀(I/I₀+10⁻⁸), and the residual is Δlog₁₀I=log₁₀[(I_CZT/I₀+10⁻⁸)/(I_ref/I₀+10⁻⁸)]. "
    "Residual color limits are symmetric and labelled separately for each row; no residual values are clipped. "
    "The displayed intensity relative L₂ errors use the original unnormalized arrays over the full 257×257 grid. "
    "Smooth-object comparisons include finite-mesh approximation; the box isolates the remaining projection and transform errors. "
    "This figure tests the numerical planar kernel, separately from physical Ewald curvature."
)
(OUT/"figure2_caption.txt").write_text(caption+"\n",encoding="utf-8")
with Image.open(png) as image:
    size=image.size;dpi=image.info.get("dpi")
    # Review preview is derived solely from this final rendered plot, not another
    # data rendering. It makes final-size visual inspection possible.
    preview=image.copy();preview.thumbnail((1420,1420))
    preview.save(OUT/"figure_02_preview.png")
provenance={"status":"numeric_checks_pass;_visual_QA_pending","chart_contract":contract,"cases":audit,
    "files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in [png,svg,pdf]],
    "png_dimensions_px":list(size),"png_dpi":dpi,"source_generation_script":"validation_package/scripts/section3_data_acquisition_v3_public.py",
    "source_generation_script_sha256":sha(ROOT/"validation_package/scripts/section3_data_acquisition_v3_public.py"),
    "note":"The plotting script imports no simulator; closed-form references and stored amplitude/intensity identities were checked independently. Source arrays are unchanged."}
(OUT/"figure2_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"dimensions_px":size,"dpi":dpi,"checks":summary,"files":[p.name for p in [png,svg,pdf]]},indent=2))
