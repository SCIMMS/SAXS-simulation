"""Figure S19 from saved numerical results; display terminology only."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'assets').is_dir() and (p/'validation_package').is_dir())
import argparse
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
OUT=p.parse_args().output_dir;OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'validation_package/revision_2026_10_17/data/output/qz5_aspect_sensitivity_2026-10-16'
LABELS=['narrow','medium','wide']
rows=list(csv.DictReader((SRC/'paired_results.csv').open()))
groups={k:sorted([r for r in rows if r['range_label']==k],key=lambda r:int(r['frame'])) for k in LABELS}
for rr in groups.values():assert [int(r['frame']) for r in rr]==list(range(209))
summary=json.loads((SRC/'summary.json').read_text())
colors=['#777777','#a1bed4','#276b9e'];names=['0.9–1.1','0.8–1.25','2/3–1.5']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
fig,axes=plt.subplots(1,3,figsize=(15,5.5));vals=[np.array([float(r['radius_nm']) for r in groups[k]]) for k in LABELS]
box=axes[0].boxplot(vals,tick_labels=names,patch_artist=True,widths=.5,
    medianprops={'color':'#222222','linewidth':1.4},flierprops={'marker':'o','markersize':3,'markerfacecolor':'white','markeredgecolor':'#666666'})
for patch,color in zip(box['boxes'],colors):patch.set_facecolor(color);patch.set_edgecolor('#444444')
axes[0].set_ylim(bottom=max(0,min(v.min() for v in vals)-2),top=max(v.max() for v in vals)+5)
axes[0].set_xlabel('Allowed x/z and y/z aspect ratios');axes[0].set_ylabel('Effective corner radius (nm)')
for j,k in enumerate(LABELS):axes[0].text(j+1,axes[0].get_ylim()[1]-.7,f'n={summary["results"][k]["bound_counts"]["aspect_bound"]} bound',ha='center',va='top',fontsize=8)
axes[1].scatter(vals[0],vals[2],s=19,color=colors[-1],alpha=.65,edgecolors='none')
low=min(vals[0].min(),vals[2].min())-1;high=max(vals[0].max(),vals[2].max())+1
axes[1].plot([low,high],[low,high],color='#555555',ls='--',lw=1);axes[1].set_xlim(low,high);axes[1].set_ylim(low,high)
axes[1].set_xlabel('Radius with 0.9–1.1 bounds (nm)');axes[1].set_ylabel('Radius with 2/3–1.5 bounds (nm)');axes[1].set_aspect('equal','box')
x=np.arange(2);w=.24
for j,(label,color,name) in enumerate(zip(LABELS,colors,names)):
    y=[summary['results'][label]['metrics'][sp]['versus_coarse']['after_mean_log_mse'] for sp in ['validation','test']]
    bars=axes[2].bar(x+(j-1)*w,y,width=w,color=color,edgecolor='#444444',linewidth=.6,label=name)
    axes[2].bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
axes[2].set_xticks(x,['Penalty selection','Test']);axes[2].set_ylabel('Mean log1p ADU MSE (lower is better)');axes[2].set_ylim(0,.65);axes[2].legend(frameon=False,fontsize=9)
fig.suptitle('Five-plane refinement: sensitivity to aspect-ratio bounds',x=.065,ha='left',fontsize=16,y=.98)
fig.text(.065,.88,'209 patterns; fixed shape-mode Tikhonov λ=0.1; same fitting pixels and separate evaluation bands; both expansions reported',fontsize=11)
fig.text(.065,.025,'Radius is a model-dependent sphere-fit descriptor. Boxes show pattern-level median/IQR, not structural confidence intervals. “Bound” = either aspect at a limit.',fontsize=9)
fig.subplots_adjust(left=.065,right=.99,top=.78,bottom=.18,wspace=.35)
fig.savefig(OUT/'figure_S19.png',dpi=300);fig.savefig(OUT/'figure_S19.svg');plt.close(fig)
