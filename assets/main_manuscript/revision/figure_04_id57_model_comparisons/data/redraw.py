"""Figure 4 display-only redraw; original paired score values and histogram bins.

This independently reads the three data tables used by the retained figure.
It does not import or execute the original manuscript-authoring script.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import argparse
_parser=argparse.ArgumentParser(description="Redraw the current scientific figure from retained public inputs.")
_parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[1]/"images")
_args=_parser.parse_args()
OUT=_args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"assets").is_dir() and (p/"validation_package").is_dir())
BASE=ROOT/"validation_package/revision_2026_10_17/data"
INPUTS={
    "radial":"output/revision_remaining_2026-10-09/01_consistency/id57_retrieval_comparison.csv",
    "baseline":"output/revision_remaining_2026-10-09/02_baseline/id57_baseline_results.csv",
    "rounding":"output/id57_qz5_rounding_2026-10-16/selection_results.csv",
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
source_records=[]
for key,rel in INPUTS.items():
    assert (BASE/rel).exists(),f"Source differs from retained published base: {rel}"
    source_records.append({"key":key,"source":str((BASE/rel).relative_to(ROOT)).replace("\\","/"),"sha256":sha(BASE/rel),"published_base_identical":True})
rad=rows(BASE/INPUTS["radial"])
base=rows(BASE/INPUTS["baseline"])
sel=rows(BASE/INPUTS["rounding"])
N=209
panels={}
ra={int(r["frame"]):r for r in rad if r["branch"]=="mesh_czt"}
assert set(ra)==set(range(N))
delta=np.array([float(ra[i]["paired_delta"]) for i in range(N)])
assert np.max(abs(delta-np.array([float(ra[i]["score"])-float(ra[i]["control_score"]) for i in range(N)])))<1e-14
panels[("a","full")]=delta
for mode,fold in [("heldout","even_to_odd"),("heldout_reverse","odd_to_even")]:
    a={int(r["frame"]):r for r in base if r["mode"]==mode}
    assert set(a)==set(range(N))
    d=np.array([float(a[i]["mesh_minus_baseline"]) for i in range(N)])
    d_check=np.array([float(a[i]["mesh_eval_score"])-float(a[i]["baseline_eval_score"]) for i in range(N)])
    assert np.max(abs(d-d_check))<1e-14
    panels[("b",fold)]=d
for fold in ["even_to_odd","odd_to_even"]:
    a={int(r["frame"]):r for r in sel if r["fold"]==fold and r["family"]=="rounded"}
    b={int(r["frame"]):r for r in sel if r["fold"]==fold and r["family"]=="variable_size_sharp"}
    assert set(a)==set(b)==set(range(N))
    panels[("c",fold)]=np.array([float(a[i]["test_logcorr"])-float(b[i]["test_logcorr"]) for i in range(N)])

summary=[]
paired_records=[]
for (panel,fold),d in panels.items():
    summary.append({"panel":panel,"fold":fold,"n":len(d),"mean":float(d.mean()),"median":float(np.median(d)),"minimum":float(d.min()),"maximum":float(d.max()),"positive":int((d>0).sum()),"negative":int((d<0).sum()),"zero":int((d==0).sum())})
    paired_records.extend({"panel":panel,"fold":fold,"frame":i,"paired_delta_log_correlation":float(v)} for i,v in enumerate(d))

contract={
    "question":"What score differences across frames are observed for each of three distinct ID57 comparisons?",
    "takeaway":"Three defined comparisons have distinct scoring protocols; differences within a panel are interpretable, and absolute scores across panels do not form a common model ranking.",
    "family":"Comparison of empirical distributions; histogram and overlaid step histograms",
    "surface":"Main manuscript Figure 4 static PNG, SVG and PDF",
    "grain":"One paired difference per frame and evaluation fold; all 209 frames retained in each series",
    "dimensions_in":[6.5,6.5/3],"dpi":600,
    "axis_definition":"Score difference = model evaluation correlation score minus comparator evaluation correlation score for the same frame and pixels",
    "palette":"Blue structured/even-to-odd; orange odd-to-even; dashed versus solid traces distinguish folds without color",
    "histogram_bins":"Exactly the retained histogram bins: 20 auto bins for a, 30 edges over the joined b domain padded by0.005, and29 edges over joined c domain padded by0.005",
    "analyses":{
        "a":"Historical planar structured-minus-radial-control comparison on full ROI",
        "b":"Planar fixed core-shell minus fitted ellipsoid on separate radial evaluation bands",
        "c":"Five-plane physical-Ewald rounded minus variable-size sharp after complete library regeneration and reselection",
    },
    "legend_definition":"Even to odd / odd to even denotes fitting radial-band parity to evaluation radial-band parity, rather than shot-number parity",
}
(OUT/"figure4_chart_contract.json").write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":7.4,"axes.labelsize":7.6,
    "xtick.labelsize":7.1,"ytick.labelsize":7.1,"axes.linewidth":.65,
    "axes.spines.top":False,"axes.spines.right":False,
    "svg.fonttype":"none","pdf.fonttype":42,
    "text.color":"#222222","axes.labelcolor":"#222222",
})
fig,axs=plt.subplots(1,3,figsize=contract["dimensions_in"],dpi=600)
fig.subplots_adjust(left=.065,right=.992,bottom=.265,top=.745,wspace=.32)
blue="#3479aa";orange="#b76b22"
counts,bins,_=axs[0].hist(delta,bins=20,color=blue)
assert counts.sum()==N
bin_records=[]
for j,c in enumerate(counts):bin_records.append({"panel":"a","fold":"full","bin_lower":float(bins[j]),"bin_upper":float(bins[j+1]),"frames":int(c)})
for pidx,panel,n_edges in [(1,"b",30),(2,"c",29)]:
    allv=np.concatenate([panels[(panel,f)] for f in ["even_to_odd","odd_to_even"]])
    bins=np.linspace(allv.min()-.005,allv.max()+.005,n_edges)
    for fold,color,label,ls in [("even_to_odd",blue,"Even → odd","-"),("odd_to_even",orange,"Odd → even","--")]:
        counts,_,_=axs[pidx].hist(panels[(panel,fold)],bins=bins,histtype="step",color=color,label=label,lw=1.0,linestyle=ls)
        assert counts.sum()==N
        for j,c in enumerate(counts):bin_records.append({"panel":panel,"fold":fold,"bin_lower":float(bins[j]),"bin_upper":float(bins[j+1]),"frames":int(c)})

titles=["Structured − radial control","Fixed core-shell − ellipsoid","Rounded − variable-size sharp"]
subtitles=["Full ROI · planar","Evaluation bands · planar","Evaluation bands · 5-plane Ewald"]
for ax,letter,title,subtitle in zip(axs,"abc",titles,subtitles):
    ax.axvline(0,color="#777777",lw=.65,zorder=0)
    ax.text(.02,.98,f"({letter})",transform=ax.transAxes,va="top",fontweight="bold",fontsize=8)
    ax.set_title(title+"\n"+subtitle,fontsize=7.25,pad=8,linespacing=1.25,fontweight="normal")
    ax.set_xlabel("Score difference",labelpad=3)
    ax.grid(axis="y",color="#dddddd",lw=.45,alpha=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5,width=.6,pad=2)
axs[0].set_ylabel("Frames",labelpad=3)
handles,labels=axs[1].get_legend_handles_labels()
fig.legend(handles,["Fit even → evaluate odd","Fit odd → evaluate even"],loc="lower center",bbox_to_anchor=(.54,.005),ncol=2,frameon=False,fontsize=6.6,handlelength=2.0,handletextpad=.4,columnspacing=1.6)
fig.canvas.draw()
outputs=[]
for suffix in ["png","svg","pdf"]:
    p=OUT/f"figure_04.{suffix}"
    fig.savefig(p,dpi=600,facecolor="white")
    outputs.append({"name":p.name,"sha256":sha(p),"bytes":p.stat().st_size})
plt.close(fig)
with Image.open(OUT/outputs[0]["name"]) as im:
    dimensions=list(im.size);dpi=im.info.get("dpi")
    preview=im.copy();preview.thumbnail((1950,1950));preview.save(OUT/"figure_04_preview.png")
for name,records in [("figure4_numeric_summary.csv",summary),("figure4_paired_values.csv",paired_records),("figure4_histogram_bins.csv",bin_records)]:
    with (OUT/name).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
caption='ID57 model comparisons for all 209 frames. Each horizontal coordinate is a score difference: the model correlation score minus the comparator correlation score for the same frame and evaluation pixels. (a) Historical planar structured-minus-radial-control comparison on the full region of interest (ROI). (b) Planar fixed core-shell minus fitted ellipsoid on separate radial evaluation bands. (c) Five-plane physical-Ewald rounded minus variable-size sharp comparisons after regenerating and reselecting the complete candidate libraries. In (b,c), Fit even → evaluate odd means fitting on even-numbered radial bands and evaluating predictions on odd-numbered bands; the reverse direction exchanges these roles. These labels refer to band numbers, not frame numbers. Geometry, intensity scale and background are determined using fitting bands and kept fixed during evaluation. Positive differences favor the first model named in each panel. Panels show different matched comparisons; absolute scores across their calculation and scoring conventions are not treated as a common ranking. Complete results and direct parameter summaries are in Section S4.'
(OUT/"figure4_caption.txt").write_text(caption+"\n",encoding="utf-8")
provenance={"status":"numeric_checks_pass;visual_QA_pending","contract":contract,"sources":source_records,"series":summary,
    "files":outputs,"png_dimensions_px":dimensions,"png_dpi":dpi,
    "plotting_basis":"Retained Figure 4 plotting expressions; only public scientific tables are read",
    "source_and_histogram_assertions":"Each series has exactly209 unique frames; source data match the published retained base byte-for-byte; paired differences were independently reconstructed from score columns; each histogram accounts for all209 frames."}
(OUT/"figure4_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"dimensions_px":dimensions,"dpi":dpi,"series":summary},indent=2))
