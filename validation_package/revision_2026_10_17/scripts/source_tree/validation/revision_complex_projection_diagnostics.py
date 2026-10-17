"""Residual-error audit, source preservation, and exportable scientific figures."""
from __future__ import annotations
import csv,json,time
from pathlib import Path
from revision_complex_projection_validation import *


def residuals():
    tic=time.perf_counter();cfg,axis,roi,rotations,qcur,qflat,sampled,radial=inputs()
    records=[];closure=[]
    for radius in RADII:
        rows=list(csv.DictReader((OUT/f'accuracy_radius{radius:g}.csv').open()))
        worst=max((r for r in rows if r['method']=='qz5_barycentric'),
                  key=lambda r:float(r['intensity_L2_roi']))
        oi=int(worst['orientation']);meshes=geometry(radius)
        truth=IndependentBoundaryAmplitude(meshes,qcur).amplitude(rotations[oi])
        for n in [384,768,1536]:
            renderer=ComplexProjectionRenderer(meshes,axis,qcur,roi,n=n,planes=5)
            amplitude=cp.asnumpy(renderer.amplitude(rotations[oi]))
            records.append(dict(radius_nm=radius,orientation=oi,n=n,qz_planes=5,
                intensity_relative_L2=rel_l2(abs(amplitude)**2,abs(truth)**2),
                amplitude_relative_L2=rel_l2(amplitude,truth)))
            del renderer;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        # Check closed-ray balance for every domain separately over all 240 views.
        for domain,(mesh,contrast) in enumerate(meshes):
            projector=ComplexMeshProjector([(mesh,1.)])
            max_bad=0;worst_ids=[]
            for index,r in enumerate(rotations):
                p,balance=projector.project(r,[0.],True)
                bad=int(cp.count_nonzero(balance).get())
                if bad>max_bad:max_bad=bad;worst_ids=[index]
                elif bad and bad==max_bad:worst_ids.append(index)
            closure.append(dict(radius_nm=radius,domain=domain,orientations=240,
                                max_unpaired_pixels=max_bad,worst_orientation_ids=worst_ids))
            assert max_bad==0,closure[-1]
        print('RESIDUAL',radius,'worst orientation',oi,'seconds',round(time.perf_counter()-tic,1),flush=True)
    save_csv(OUT/'spatial_resolution_residuals.csv',records)
    dump(OUT/'residual_diagnostics.json',dict(status='complete',rows=records,
        per_domain_crossing_balance=closure,seconds=time.perf_counter()-tic,
        selection='Worst sampled q_z=5 orientation of each model, then all 91567 valid ROI pixels.',
        interpretation='Vary spatial raster resolution at fixed q_z plane count to test the source of the 5/9-plane error plateau.'))


def make_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True,parents=True)
    controls_record=json.loads((OUT/'controls.json').read_text())
    accuracy_record=json.loads((OUT/'accuracy_summary.json').read_text())
    residual_record=json.loads((OUT/'residual_diagnostics.json').read_text())
    assert protected_hashes()==controls_record['source_hashes']==accuracy_record['config']['source_hashes']
    cfg,axis,roi,rotations,qcur,qflat,sampled,radial=inputs()
    records=[]
    for radius in RADII:
        for method in METHODS:
            r=json.loads((OUT/f'benchmark_radius{radius:g}_{method}.json').read_text())
            assert r['status']=='complete' and r['returned_roi_pixels']==int(roi.sum())
            records.append(r)
    comparison=[]
    for radius in RADII:
        acc=accuracy_record['models'][str(radius)]
        base=next(r for r in records if r['radius_nm']==radius and r['method']=='legacy_planar')
        resident=next(r for r in records if r['radius_nm']==radius and r['method']=='resident_planar')
        for method in METHODS:
            r=next(r for r in records if r['radius_nm']==radius and r['method']==method)
            a=acc[method+'_barycentric' if method.startswith('qz') else method]['intensity_L2_roi']
            comparison.append(dict(radius_nm=radius,method=method,
                sweep_mean_intensity_relative_L2_percent=100*a['mean'],
                sweep_p95_intensity_relative_L2_percent=100*a['p95'],
                sweep_max_intensity_relative_L2_percent=100*a['max'],
                median_ms_per_orientation=r['median_ms_per_orientation'],
                time_ratio_to_legacy=r['median_ms_per_orientation']/base['median_ms_per_orientation'],
                time_ratio_to_resident_planar=r['median_ms_per_orientation']/resident['median_ms_per_orientation'],
                cupy_live_allocation_peak_MiB=r['cupy_live_allocation_peak_MiB'],
                cupy_reserved_high_water_MiB=r['cupy_reserved_high_water_MiB'],
                setup_s=r['setup_s'],first_output_s=r['first_output_s']))
    save_csv(OUT/'comparison_summary.csv',comparison)
    # Chart contract: three model facets, identical log error axes, orientation
    # distributions rather than connecting sparse method categories as a trend.
    contract=dict(surface='exportable standalone scientific PNG/PDF figures, Matplotlib',
        accuracy=dict(question='How does Ewald intensity error change with 3/5/9 q_z planes?',
            family='distribution',variant='faceted box plots, mean diamonds, shared logarithmic percent scale',
            grain='one method/model/orientation; 240 orientations per model; 4-pixel detector sampling',
            palette='one blue root for complex projections, neutral existing baseline',
            interpretation='Box whiskers are 5th/95th percentiles, dots include every remaining outlier.'),
        cost=dict(question='What synchronized end-to-end runtime and allocated GPU memory does each method need?',
            family='comparison',variant='grouped bars for three geometries, zero-based axes',
            grain='one method/model; three fresh-process timing batches of 48 orientations',
            palette='blue/gold/olive with hatches for model identity',
            uncertainty='Runtime bars show the median; whiskers show min/max of three batch averages.',
            memory='Peak live CuPy allocation from a separate instrumented pass; excludes CUDA driver/context.'))
    dump(OUT/'chart_contract.json',contract)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':'#222222','text.color':'#222222',
        'axes.edgecolor':'#777777','figure.facecolor':'white','savefig.facecolor':'white'})
    labels=['Existing\nplanar','3 planes','5 planes','9 planes']
    show_methods=['legacy_planar','qz3_barycentric','qz5_barycentric','qz9_barycentric']
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.8),sharey=True)
    for ax,radius in zip(axes,RADII):
        rows=list(csv.DictReader((OUT/f'accuracy_radius{radius:g}.csv').open()))
        values=[100*np.array([float(r['intensity_L2_roi']) for r in rows if r['method']==m]) for m in show_methods]
        box=ax.boxplot(values,patch_artist=True,whis=(5,95),showmeans=True,
            meanprops=dict(marker='D',markersize=4,markerfacecolor='#222222',markeredgecolor='#222222'),
            flierprops=dict(marker='.',markersize=3,markerfacecolor='#333333',markeredgecolor='none'),
            medianprops=dict(color='#222222',linewidth=1.2))
        for j,patch in enumerate(box['boxes']):
            patch.set_facecolor('#d1d5db' if j==0 else '#b7d4ec');patch.set_edgecolor('#444444')
        ax.set_yscale('log');ax.set_xticks(range(1,5),labels)
        ax.set_title('Sharp' if radius==0 else f'Rounded: r = {radius:g} nm')
        ax.grid(axis='y',which='major',color='#e5e7eb',linewidth=.7);ax.set_axisbelow(True)
    axes[0].set_ylabel('ROI intensity relative L2 error (%)')
    fig.suptitle('Accuracy against continuous boundary integration',x=.065,ha='left',fontsize=15,y=.98)
    fig.text(.065,.89,'240 orientations per model; q = 0.267–1.128 nm⁻¹; primary q_z interpolation: barycentric',fontsize=10)
    fig.text(.065,.025,'Every fourth detector pixel in both axes; dense checks saved separately. Diamonds: means; boxes: median/IQR; whiskers: 5–95%.',fontsize=8.6)
    fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.18,wspace=.17)
    fig.savefig(OUT/'accuracy.png',dpi=180);fig.savefig(OUT/'accuracy.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11.5,7.7))
    x=np.arange(len(METHODS));width=.24
    colors=['#276b9e','#d0a13b','#7a8b49'];hatches=['','///','..']
    for j,radius in enumerate(RADII):
        selected=[next(r for r in records if r['radius_nm']==radius and r['method']==m) for m in METHODS]
        times=np.array([r['median_ms_per_orientation'] for r in selected])
        lows=np.array([min(r['batch_seconds'])*1000/len(r['timed_orientation_ids']) for r in selected])
        highs=np.array([max(r['batch_seconds'])*1000/len(r['timed_orientation_ids']) for r in selected])
        label='Sharp' if radius==0 else f'Rounded r={radius:g} nm'
        axes[0].bar(x+(j-1)*width,times,width,color=colors[j],edgecolor='#333333',linewidth=.5,
            hatch=hatches[j],label=label,yerr=np.array([times-lows,highs-times]),capsize=2,error_kw={'linewidth':.8})
        axes[1].bar(x+(j-1)*width,[r['cupy_live_allocation_peak_MiB'] for r in selected],width,
            color=colors[j],edgecolor='#333333',linewidth=.5,hatch=hatches[j])
    names=['Existing planar','Resident planar','3 q_z planes','5 q_z planes','9 q_z planes']
    for ax in axes:
        ax.set_xticks(x,names);ax.set_ylim(bottom=0);ax.grid(axis='y',color='#e5e7eb',linewidth=.7);ax.set_axisbelow(True)
    axes[0].set_ylim(0,max(max(r['batch_seconds'])*1000/len(r['timed_orientation_ids']) for r in records)*1.25)
    axes[0].set_ylabel('Time per orientation (ms)');axes[1].set_ylabel('Peak live CuPy allocation (MiB)')
    axes[0].legend(frameon=False,ncol=3,loc='upper left')
    fig.suptitle('Execution time and GPU allocation',x=.09,ha='left',fontsize=15,y=.98)
    fig.text(.09,.925,'RTX 2070 SUPER; 768² raster; 401² CZT grid; 91,567 returned ROI intensities',fontsize=10)
    fig.text(.09,.025,'Three 48-view batches per method/model, each in an isolated process. Error bars: batch min/max. Memory excludes CUDA driver/context.',fontsize=8.6)
    fig.subplots_adjust(left=.09,right=.985,top=.87,bottom=.105,hspace=.29)
    fig.savefig(OUT/'cost.png',dpi=180);fig.savefig(OUT/'cost.pdf');plt.close(fig)
    # Independent recomputation of aggregates and archived example errors.
    audit=[];five_vs_nine=[]
    for radius in RADII:
        rows=list(csv.DictReader((OUT/f'accuracy_radius{radius:g}.csv').open()))
        assert len(rows)==240*10
        for method in show_methods:
            values=np.array([float(r['intensity_L2_roi']) for r in rows if r['method']==method])
            original=accuracy_record['models'][str(radius)][method]['intensity_L2_roi']
            assert abs(values.mean()-original['mean'])<1e-15
            assert abs(values.max()-original['max'])<1e-15
        for oi in [0,144,238]:
            with np.load(OUT/f'example_radius{radius:g}_orientation{oi}.npz') as f:
                five_vs_nine.append(dict(radius_nm=radius,orientation=oi,
                    intensity_difference_over_reference_L2=float(np.linalg.norm(
                        abs(f['qz5_barycentric'])**2-abs(f['qz9_barycentric'])**2)/
                        np.linalg.norm(abs(f['truth'])**2))))
                for method in ['legacy_planar','resident_planar','qz3_barycentric','qz5_barycentric','qz9_barycentric']:
                    got=np.linalg.norm(abs(f[method])**2-abs(f['truth'])**2)/np.linalg.norm(abs(f['truth'])**2)
                    ref=next(r for r in json.loads((OUT/f'accuracy_radius{radius:g}.json').read_text())['dense_rows']
                             if r['orientation']==oi and r['method']==method)['intensity_L2_roi']
                    assert abs(got-ref)<1e-14
        audit.append(dict(radius_nm=radius,rows=len(rows),examples_recomputed=3*5))
    assert protected_hashes()==controls_record['source_hashes']
    tilt=np.degrees(np.arccos(np.clip(rotations[:,2,2],-1,1)))
    summary=dict(status='complete',question='Orientation-specific complex projections: accuracy, speed and memory for 3/5/9 q_z planes.',
        config=accuracy_record['config'],comparison=comparison,controls=controls_record['checks'],
        residual_diagnostics=residual_record,aggregate_and_example_audit=audit,
        five_vs_nine_dense_intensity_difference=five_vs_nine,
        orientation_tilt_degrees=dict(min=float(tilt.min()),max=float(tilt.max())),
        source_hashes_preserved=True,
        implementation_sources={str(p.relative_to(ROOT)):sha(p) for p in
            [Path(__file__),ROOT/'validation/revision_complex_projection.py',
             ROOT/'validation/revision_complex_projection_validation.py']},
        limitations=['Accuracy sweep uses every fourth detector pixel, plus full-ROI checks at predefined and worst sampled orientations.',
            'These are synthetic forward-model errors for fixed piecewise-constant meshes, not a noise-aware structural identification test.',
            'The existing 240-view orientation bank and two rounding radii do not establish accuracy for arbitrary morphology, thickness, wavelength or q range.',
            'Five/nine-plane residuals include spatial raster and lateral interpolation errors; K alone is not an accuracy guarantee.',
            'GPU memory is instrumented CuPy allocation, not total process GPU memory.',
            'New kernel uses robust float64 edge tests; current real-valued production baseline is retained unchanged.',
            'No manuscript, SI, response letter or measured-data template selection has been changed.'],
        reproduce=[".venv_cxidb/Scripts/python.exe validation/revision_complex_projection_validation.py controls",
                   ".venv_cxidb/Scripts/python.exe validation/revision_complex_projection_validation.py accuracy",
                   ".venv_cxidb/Scripts/python.exe validation/revision_complex_projection_validation.py benchmark --radius 0 --method qz5",
                   "Repeat benchmark for radii 0,4,12 and methods legacy_planar,resident_planar,qz3,qz5,qz9, each in a fresh process.",
                   ".venv_cxidb/Scripts/python.exe validation/revision_complex_projection_diagnostics.py residuals",
                   ".venv_cxidb/Scripts/python.exe validation/revision_complex_projection_diagnostics.py report"])
    dump(OUT/'results_summary.json',summary)
    # A compact evidence manifest binds the readable outputs to the exact code.
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='completion_manifest.json')
    dump(OUT/'completion_manifest.json',dict(status='complete',source_hashes_preserved=True,
        functional_control_count=len(controls_record['checks']),benchmark_variants=len(records),
        sweep_model_orientation_pairs=3*240,full_ROI_check_model_orientation_pairs=sum(
            len(json.loads((OUT/f'accuracy_radius{r:g}.json').read_text())['dense_orientation_ids']) for r in RADII),
        output_files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
        implementation_sources=summary['implementation_sources']))
    print('REPORT complete',len(comparison),'comparison rows',flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='residuals':residuals()
    elif sys.argv[1]=='report':make_report()
    else:raise ValueError(sys.argv[1])
