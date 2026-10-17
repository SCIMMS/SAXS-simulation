"""Replot supplied five-level GroEL construction arrays and unchanged decimation CSV.

This reproduces the display only. Full VTK reconstruction is documented by the
retained density/construction evidence; it is separate from the12-layer model.
"""
from pathlib import Path
import argparse,csv
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'images')
args=parser.parse_args();OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'assets').is_dir() and (p/'validation_package').is_dir())
HERE=Path(__file__).resolve().parent
with np.load(HERE/'projection_arrays.npz') as z:arrays={k:z[k] for k in z.files}
direct,original,decimated=[arrays[k] for k in ['direct','original','decimated']]
with (ROOT/'validation_package/section5_groel/mesh_convergence/groel_mesh_decimation_convergence.csv').open(encoding='utf-8',newline='') as f:curves=list(csv.DictReader(f))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8,
    'axes.labelsize':7.5,'legend.fontsize':6,'xtick.labelsize':7,'ytick.labelsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig=plt.figure(figsize=(6,7.16))
outer=fig.add_gridspec(3,1,height_ratios=[1.65,3.32,1.48],hspace=.38,
    left=.13,right=.88,top=.965,bottom=.075)

# (a) Original orthogonal C7 density mid-slices, each with its original log scale.
ga=outer[0].subgridspec(1,4,width_ratios=[1,1,1,.045],wspace=.14)
axes_a=[]
for col,(title,arr) in enumerate([('x-mid',arrays['mid_x']),
                                ('y-mid',arrays['mid_y']),
                                ('z-mid',arrays['mid_z'])]):
    ax=fig.add_subplot(ga[0,col]);axes_a.append(ax)
    im=ax.imshow(np.log10(np.maximum(arr.T,1e-9)),origin='lower',cmap='viridis')
    ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
cax=fig.add_subplot(ga[0,3]);fig.colorbar(im,cax=cax,label=r'$\log_{10}[\rho/(e\,\mathrm{\AA}^{-3})]$')

# (b) Same C7 density projection and five-level mesh-stencil midpoint estimates.
gb=outer[1].subgridspec(2,4,width_ratios=[1,1,1,.045],wspace=.14,hspace=.20)
vmax=float(np.percentile(direct[direct>0],99.5))
limit=max(float(np.max(np.abs(original-direct))),float(np.max(np.abs(decimated-direct))))
axes_b=[]
for col,(title,values) in enumerate([('Direct density',direct),('Original mesh',original),('Quadric-r80 mesh',decimated)]):
    ax=fig.add_subplot(gb[0,col]);axes_b.append(ax)
    im=ax.imshow(values.T,origin='lower',cmap='viridis',vmax=vmax)
    ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
cax=fig.add_subplot(gb[0,3]);fig.colorbar(im,cax=cax,label=r'Projected density / $e\,\mathrm{\AA}^{-2}$')
ax=fig.add_subplot(gb[1,0]);ax.axis('off')
for col,(title,values) in enumerate([('Original − direct',original),('Quadric-r80 − direct',decimated)],1):
    ax=fig.add_subplot(gb[1,col]);im=ax.imshow((values-direct).T,origin='lower',cmap='coolwarm',vmin=-limit,vmax=limit)
    ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
cax=fig.add_subplot(gb[1,3]);fig.colorbar(im,cax=cax,label=r'Residual / $e\,\mathrm{\AA}^{-2}$')

# (c) The retained original C7/D7 scalar decimation results, without recomputation.
gc=outer[2].subgridspec(1,3,wspace=.47);axes_c=[fig.add_subplot(gc[0,j]) for j in range(3)]
for model,color,label in [('A_heptamer_C7','#2b6cb0','A (C7)'),('B_tetradecamer_D7','#d97927','B (D7)')]:
    data=[r for r in curves if r['model']==model]
    x=[float(r['threshold_fraction']) for r in data]
    axes_c[0].plot(x,[float(r['point_ratio_quadric_over_original']) for r in data],'o-',color=color,label=label,ms=3,lw=1)
    axes_c[1].plot(x,[float(r['relative_volume_error_quadric_minus_original']) for r in data],'o-',color=color,ms=3,lw=1)
    axes_c[2].plot(x,[int(r['components_original']) for r in data],'o-',color=color,label=label+' original',ms=3,lw=1)
    axes_c[2].plot(x,[int(r['components_quadric_r80']) for r in data],'x--',color=color,label=label+' r80',ms=3,lw=1)
for ax in axes_c:
    ax.set_xlabel('Threshold fraction');ax.set_xticks([.1,.3,.5,.7,.9]);ax.grid(alpha=.15)
axes_c[0].set_ylabel('Point ratio\nDecimated / original');axes_c[0].set_ylim(0,.25);axes_c[0].legend(frameon=False,loc='lower left')
axes_c[1].set_ylabel('Relative volume error');axes_c[1].axhline(0,c='.5',lw=.6)
axes_c[1].ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
axes_c[2].set_ylabel('Connected components');axes_c[2].legend(frameon=False,loc='upper left',fontsize=5.5)
for ax,label in [(axes_a[0],'(a)'),(axes_b[0],'(b)'),(axes_c[0],'(c)')]:
    pos=ax.get_position();fig.text(.015,pos.y1+.012,label,fontweight='bold',fontsize=9)

png=OUT/'figure_S11.png';svg=OUT/'figure_S11.svg'
fig.savefig(png,dpi=600);fig.savefig(svg,dpi=600)
plt.close(fig)
