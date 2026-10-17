"""Aggregate and visualize the full five-plane wide-q reselection experiment."""
from __future__ import annotations
from revision_id57_qz5_rounding import *
from revision_id57_qz5_rounding_audit import reference_summary


def boot(values):
    a=np.asarray(values,float);rng=np.random.default_rng(20261016)
    samples=rng.choice(a,size=(10000,len(a)),replace=True).mean(axis=1)
    return dict(mean=float(a.mean()),median=float(np.median(a)),positive=int((a>1e-8).sum()),
        negative=int((a< -1e-8).sum()),tied=int((abs(a)<=1e-8).sum()),
        descriptive_frame_bootstrap_95_interval=np.quantile(samples,[.025,.975]).tolist())


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config=json.loads((OUT/'library_config.json').read_text())
    generation=json.loads((OUT/'generation_audit.json').read_text())
    scoring=json.loads((OUT/'scoring_audit.json').read_text())
    selected=json.loads((OUT/'selection_summary.json').read_text())
    audit=json.loads((OUT/'independent_result_audit.json').read_text())
    reference=json.loads((OUT/'exact_reference_audit.json').read_text())
    rows=list(csv.DictReader((OUT/'selection_results.csv').open()))
    exact=list(csv.DictReader((OUT/'selected_exact_reference_fits.csv').open()))
    old=list(csv.DictReader((OLD/'06_information/conditional_information.csv').open()))
    lookup={(int(r['frame']),r['fold'],r['family']):r for r in rows}
    exlookup={(int(r['frame']),r['fold'],r['family']):r for r in exact}
    oldlookup={(int(r['frame']),r['mode']):r for r in old}
    mapping={'full':'full','even_to_odd':'heldout','odd_to_even':'heldout_reverse'}
    comparisons=[]
    for fold in FOLDS:
        for frame in range(209):
            sharp=lookup[frame,fold,'variable_size_sharp'];rounded=lookup[frame,fold,'rounded']
            e_sharp=exlookup[frame,fold,'variable_size_sharp'];e_round=exlookup[frame,fold,'rounded']
            comparisons.append(dict(fold=fold,frame=frame,
                original_planar_delta=float(oldlookup[frame,mapping[fold]]['rounding_over_variable_size_eval_delta']),
                qz5_delta=float(rounded['test_logcorr'])-float(sharp['test_logcorr']),
                exact_frozen_geometry_delta=float(e_round['exact_test_logcorr'])-float(e_sharp['exact_test_logcorr']),
                sharp_test_score=float(sharp['test_logcorr']),rounded_test_score=float(rounded['test_logcorr']),
                rounded_base_radius_nm=float(rounded['base_radius_nm']),rounded_scale=float(rounded['scale']),
                rounded_candidate=int(rounded['candidate']),sharp_candidate=int(sharp['candidate'])))
    save_csv(OUT/'paired_comparison_by_fold.csv',comparisons)
    paired=[]
    for frame in range(209):
        rr=[r for r in comparisons if r['frame']==frame and r['fold']!='full']
        assert len(rr)==2
        paired.append(dict(frame=frame,**{key:float(np.mean([r[key] for r in rr])) for key in
            ['original_planar_delta','qz5_delta','exact_frozen_geometry_delta']}))
    save_csv(OUT/'paired_twofold_deltas.csv',paired)
    results={}
    for fold in FOLDS+['paired_twofold_mean']:
        rr=paired if fold=='paired_twofold_mean' else [r for r in comparisons if r['fold']==fold]
        results[fold]={key:boot([r[key] for r in rr]) for key in
            ['original_planar_delta','qz5_delta','exact_frozen_geometry_delta']}
        results[fold]['change_in_delta_qz5_minus_original']=boot([r['qz5_delta']-r['original_planar_delta'] for r in rr])
    for fold in FOLDS:
        assert abs(results[fold]['qz5_delta']['mean']-selected['results'][fold]['mean_rounding_over_size_delta'])<1e-14
        assert abs(results[fold]['exact_frozen_geometry_delta']['mean']-
                   audit['exact_reference_results'][fold]['mean_rounding_over_size_delta'])<1e-14
    assert abs(results['paired_twofold_mean']['qz5_delta']['mean']-
               selected['results']['paired_twofold_mean']['mean_delta'])<1e-14
    # Scientific chart contract: paired observations, identical score definitions,
    # a visible zero, and uncertainty labelled as descriptive frame resampling.
    dump(OUT/'chart_contract.json',dict(surface='exportable scientific PNG/PDF, Matplotlib',
        figures=[dict(name='heldout_comparison',families=['relationship','distribution'],
          grain='209 frames, each averaged across the two complementary radial-band holdouts',
          question='Does the incremental held-out gain from rounding persist after full Ewald-aware reselection?',
          fields=['original_planar_delta','qz5_delta','exact_frozen_geometry_delta'],
          palette='one blue root plus neutral zero/identity/mean lines',
          limitations='Within-frame radial prediction; frame bootstrap is descriptive, not calibrated structure confidence.'),
          dict(name='radius_selections',family='comparison',grain='one training-selected radius per frame/fold',
          fields=['fold','base_radius_nm','count'],palette='blue/gold/olive with hatch distinctions',
          question='How often do finite-grid rounding selections reach the largest sampled radius?')]))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'text.color':'#222222','axes.labelcolor':'#222222',
        'axes.edgecolor':'#777777','figure.facecolor':'white','savefig.facecolor':'white'})
    x=np.array([r['original_planar_delta'] for r in paired]);y=np.array([r['qz5_delta'] for r in paired])
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.2))
    ax=axes[0];ax.scatter(x,y,s=18,color='#276b9e',alpha=.65,edgecolors='none')
    low=min(x.min(),y.min(),0)-.015;high=max(x.max(),y.max())+.015
    ax.plot([low,high],[low,high],color='#555555',lw=1,ls='--',label='Equal gain')
    ax.axhline(0,color='#aaaaaa',lw=.7);ax.axvline(0,color='#aaaaaa',lw=.7)
    ax.set_xlim(low,high);ax.set_ylim(low,high);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Original planar: rounded − size-adjusted sharp')
    ax.set_ylabel('Five-plane: rounded − size-adjusted sharp');ax.set_title('Paired held-out gain per frame')
    ax.legend(frameon=False,loc='upper left')
    ax=axes[1];ax.hist(y,bins=18,color='#b7d4ec',edgecolor='#444444',linewidth=.7)
    ax.axvline(0,color='#444444',ls='--',lw=1)
    ax.axvline(y.mean(),color='#276b9e',lw=1.5,label=f'Mean {y.mean():+.4f}')
    ax.set_xlabel('Five-plane held-out correlation gain');ax.set_ylabel('Number of frames')
    ax.set_title('Distribution across all 209 frames');ax.legend(frameon=False)
    fig.suptitle('Rounding comparison after full five-plane reselection',x=.07,ha='left',fontsize=15,y=.98)
    fig.text(.07,.90,'All 8,400 candidates regenerated; q = 0.267–1.128 nm⁻¹; training-only selection and signal/background fitting',fontsize=10)
    qstats=results['paired_twofold_mean']['qz5_delta'];ci=qstats['descriptive_frame_bootstrap_95_interval']
    fig.text(.07,.025,f'Two-fold mean per frame. Positive: {qstats["positive"]}/209; mean gain: {qstats["mean"]:+.4f}; descriptive frame-bootstrap 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}].',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,top=.79,bottom=.18,wspace=.32)
    fig.savefig(OUT/'heldout_comparison.png',dpi=180);fig.savefig(OUT/'heldout_comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.5,4.8));positions=np.arange(7);width=.24
    colors=['#276b9e','#d0a13b','#7a8b49'];hatches=['','///','..']
    labels=['Full ROI','Even bands → odd bands','Odd bands → even bands']
    counts={}
    for j,fold in enumerate(FOLDS):
        rr=[r for r in rows if r['fold']==fold and r['family']=='rounded']
        values=[sum(float(r['base_radius_nm'])==rad for r in rr) for rad in RADII]
        counts[fold]=values
        ax.bar(positions+(j-1)*width,values,width,color=colors[j],edgecolor='#333333',
               linewidth=.5,hatch=hatches[j],label=labels[j])
    ax.set_xticks(positions,[f'{r:g}' for r in RADII]);ax.set_ylim(bottom=0)
    ax.set_xlabel('Selected base rounding radius (nm)');ax.set_ylabel('Number of frames')
    ax.legend(frameon=False,ncol=3,loc='upper left')
    ax.set_ylim(0,max(max(a) for a in counts.values())*1.25)
    ax.grid(axis='y',color='#e5e7eb',linewidth=.7);ax.set_axisbelow(True)
    fig.suptitle('Training-selected rounding radii',x=.09,ha='left',fontsize=15,y=.98)
    fig.text(.09,.89,'209 frames per fold; radius × isotropic scale determines the physical mesh radius',fontsize=10)
    fig.text(.09,.025,'12 nm is the upper grid boundary. These finite-grid selections do not establish a calibrated physical rounding radius.',fontsize=9)
    fig.subplots_adjust(left=.09,right=.985,top=.80,bottom=.18)
    fig.savefig(OUT/'radius_selections.png',dpi=180);fig.savefig(OUT/'radius_selections.pdf');plt.close(fig)
    assert config['source_hashes']==source_hashes()
    implementation=[ROOT/'validation'/s for s in ['revision_complex_projection.py','revision_gpu_logcorr_exact.py',
        'revision_id57_qz5_rounding.py','revision_id57_qz5_rounding_audit.py','revision_id57_qz5_rounding_report.py']]
    summary=dict(status='complete',question='Does rounding retain held-out gains after regenerating and reselecting the full wide-q library with five-plane Ewald evaluation?',
        config=config,results=results,selection_details=selected['results'],
        generation_audit=generation,scoring_audit=scoring,independent_result_audit=audit,
        exact_reference_audit=reference,selected_radius_counts=counts,
        source_hashes_preserved=True,implementation_sources={str(p.relative_to(ROOT)):sha(p) for p in implementation},
        limitations=['Held-out pixels are complementary radial bands within each measured pattern, not independently acquired particles or a new experimental dataset.',
          'All candidate geometry, size, rounding, orientation and linear nuisance coefficients are selected/fitted on training pixels only.',
          'The main search exhaustively scores all 8400 five-plane candidates; continuous boundary checks freeze newly selected geometries and are not an exhaustive exact-reference search.',
          'Free fluence/background and the fixed core-shell density family make this a conditional predictive comparison, not unique morphology or a precise rounding-radius determination.',
          'Upper-bound radius/scale selections remain finite-grid limits; no boundary extension was chosen using test scores.',
          'Original planar results used the historical float32 search/CPU-finalist pipeline; the new pipeline scores every candidate with float64 arithmetic.',
          'Bootstrap intervals describe resampling of the retained frame cohort, not calibrated structural uncertainty.',
          'The measured-data noise model, detector masks, original orientation bank and ROI remain the same; no beam-truncation or differential refinement was fitted.'],
        reproduce=['.venv_cxidb/Scripts/python.exe validation/revision_id57_qz5_rounding.py all',
          '.venv_cxidb/Scripts/python.exe validation/revision_id57_qz5_rounding_audit.py all',
          '.venv_cxidb/Scripts/python.exe validation/revision_id57_qz5_rounding_report.py'])
    dump(OUT/'summary.json',summary)
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='completion_manifest.json')
    signatures={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    assert signatures['qz5_library.npy']['sha256']==generation['library_sha256']
    assert signatures['exact_curved_selected.npy']['sha256']==reference['reference_library_sha256']
    dump(OUT/'completion_manifest.json',dict(status='complete',source_hashes_preserved=True,
        candidate_count=8400,frames=209,folds=3,selected_fits=1881,
        unique_exact_reference_candidates=audit['selected_unique_candidates'],
        implementation_sources=summary['implementation_sources'],
        output_files=signatures))
    print('REPORT COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':main()
