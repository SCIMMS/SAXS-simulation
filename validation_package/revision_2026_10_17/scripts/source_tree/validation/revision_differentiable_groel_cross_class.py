"""Cross-family GroEL refinement at a common historical local initialization.

Inputs and fixed-class results are read-only. Class candidates are selected on
validation bands; test losses are evaluated only after that selection.
"""
from __future__ import annotations
import argparse, json, time
from collections import Counter
from pathlib import Path
import revision_differentiable_groel as g
from revision_common import ROOT, dump, save_csv, sha, np

OUT=ROOT/'output/differentiable_groel_cross_class_2026-10-11'
PARENT=g.OUT
MODELS=['B_tetradecamer_D7','B_tetradecamer_D7_polyprism_R35_N14','B_tetradecamer_D7_polyprism_R55_N14']
SHORT=dict(zip(MODELS,['GroEL','R35','R55']))
LAMBDAS=[0.,.0001]

def path_for(i,model):return OUT/f'fits/pattern_{i:03d}_{SHORT[model]}.json'

def state(status='running',**extra):
    rr=[json.loads(p.read_text()) for p in (OUT/'fits').glob('*.json')] if (OUT/'fits').exists() else []
    dump(OUT/'work_state.json',{'status':status,'candidate_pairs_saved':len(rr),
      'candidate_pairs_expected':228,'historical_classes_preserved':True,
      'zero_penalty_saved':sum('0.0' in r['fits'] for r in rr),
      'positive_control_saved':sum('0.0001' in r['fits'] for r in rr),
      'class_selection':'Validation log MSE only; test not used',**extra})

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((PARENT/'validation_manifest.json').read_text())
    assert previous['validation_status']=='complete'
    assert all(sha(ROOT/p)==h for p,h in previous['sources'].items())
    assert all(sha(PARENT/p)==h for p,h in previous['outputs'].items())
    prior=json.loads((PARENT/'experiment_config.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in prior['source_hashes'].items())
    selection=json.loads((PARENT/'lambda_selection.json').read_text())
    assert selection['lambda']==0 and selection['positive_control_lambda']==.0001
    cfg={
      'date':'2026-10-11 Asia/Seoul','n_hits':76,'n_candidate_pairs':228,
      'models':MODELS,'short_labels':SHORT,'original_class_counts':dict(Counter(r['model_name'] for r in g.seeds().values())),
      'initialization':'Same historical quaternion and integer alignment for all three candidate families of each hit; no new orientation-library search',
      'class_assignment':'Original labels preserved. New labels are validation-selected candidate families, not structural ground truth.',
      'mask_and_objective':'Identical detector bands, masked pixels, training stride, curved Ewald vectors, log1p objective and bounds as the fixed-class experiment',
      'class_selection':'Minimum full validation-band mean squared log1p-count error among the three refined candidates; deterministic model-order tie break',
      'test_used_for_class_selection':False,
      'penalties':LAMBDAS,'penalty_selection':'Inherited from the preceding fixed-class calibration; no cross-class retuning',
      'calibration':selection['calibration'],'fit_budget':90,'rescue_budget':270,
      'reused_parent_fits':'The original family candidates reuse their converged fixed-class fits exactly',
      'candidate_dof':{'GroEL':14,'R35':16,'R55':16,'bulk_all':10},
      'cautions':[
        'The common seed was historically chosen using the full ROI and original class; the comparison is conditional and may favor that local initialization.',
        'No new global orientation search or class-specific historical seed is used.',
        'Different candidate families have different refinement dimensions; validation selection controls the comparison but does not establish unique structures.',
        'R35 and R55 extra-density boundaries are phenomenological and the two prism families are not nested in this implementation.',
        'Log-count prediction loss is descriptive; no noise variance or measured SNR is available.',
        'The inherited penalty was tuned in the original families and is not optimized for cross-class model selection.'
      ]}
    dump(OUT/'experiment_config.json',cfg)
    dump(OUT/'preservation_snapshot.json',{
      'parent_manifest_hash':sha(PARENT/'validation_manifest.json'),
      'parent_source_hashes':previous['sources'],'parent_output_hashes':previous['outputs'],
      'original_input_hashes':prior['source_hashes']})
    for i in range(76):
        original=json.loads((PARENT/f'fits/pattern_{i:03d}.json').read_text())
        p=path_for(i,original['model_name'])
        if not p.exists():
            dump(p,{'pattern_index':i,'source_frame_index':original['source_frame_index'],
              'original_class':original['model_name'],'candidate_class':original['model_name'],
              'model_name':original['model_name'],'rotation':original['rotation'],'legacy_seed':original['legacy_seed'],
              'coarse':original['coarse'],'bulk':original['bulk'],
              'fits':{str(lam):original['fits'][str(lam)] for lam in LAMBDAS},
              'provenance':'Exact reuse of converged original-family fit'})
    state()
    print('PREPARED cross-class experiment; 76 original-family pairs reused',flush=True)

def data_for(i,model):
    row,img,masks,pix,subs,original=g.data_for(i)
    family=g.GroELFamily(model);family.rotation=original.rotation.copy()
    return row,img,masks,pix,subs,family

def fit_pair(i,model,budget=90):
    row,img,masks,pix,subs,family=data_for(i,model)
    train=subs['train']&(pix[:,0]%2==0)&(pix[:,1]%2==0)
    y=img[masks['roi']];fwd=g.BoundaryForward(family,pix[train]);full=g.BoundaryForward(family,pix)
    path=path_for(i,model)
    extra=[6,7,8,9]+([10,11] if family.prism.any() else [])
    if path.exists():record=json.loads(path.read_text())
    else:
        t=np.zeros(16);t[-2]=-20;t[-1]=0
        raw=np.abs(fwd.amplitude(t,False)[0])**2;normalizer=raw.max()
        a,_=g.nnls(np.column_stack([raw/normalizer,np.ones(len(raw))]),y[train])
        t[-2]=np.log(max(a[0]/normalizer,1e-14));t[-1]=np.log(max(a[1],1e-4))
        coarse=g.optimize(fwd,y[train],t,[14,15],budget=budget)
        bulk=g.optimize(fwd,y[train],np.asarray(coarse['theta']),g.BULK,budget=budget)
        for fit in [coarse,bulk]:fit['metrics']=g.metrics(full.predict(np.asarray(fit['theta'])),y,subs)
        record={'pattern_index':i,'source_frame_index':int(row['source_frame_index']),
          'original_class':row['model_name'],'candidate_class':model,'model_name':model,
          'legacy_seed':row,'rotation':family.rotation,'coarse':coarse,'bulk':bulk,'fits':{},
          'provenance':'New alternate-family fit at shared historical quaternion and integer alignment'}
        dump(path,record)
    for lam in LAMBDAS:
        if str(lam) in record['fits']:continue
        fit=g.optimize(fwd,y[train],np.asarray(record['bulk']['theta']),g.BULK+extra,lam,budget)
        fit['metrics']=g.metrics(full.predict(fit['theta']),y,subs)
        record['fits'][str(lam)]=fit;dump(path,record)
        print('FIT',i,SHORT[model],'lambda',lam,'converged',fit['success'],'nfev',fit['nfev'],
          'seconds',round(fit['seconds'],1),flush=True)
    g.cp.get_default_memory_pool().free_all_blocks()

def all_fits(ids=range(76),budget=90):
    tic=time.perf_counter()
    for i in ids:
        for model in MODELS:
            p=path_for(i,model)
            if p.exists() and all(str(lam) in json.loads(p.read_text())['fits'] for lam in LAMBDAS):continue
            fit_pair(i,model,budget)
        state(elapsed_current_pass_seconds=time.perf_counter()-tic)
        print('HIT COMPLETE',i,'elapsed',round(time.perf_counter()-tic,1),flush=True)

def retry():
    changes=[]
    for path in sorted((OUT/'fits').glob('*.json')):
        record=json.loads(path.read_text())
        if record['provenance'].startswith('Exact reuse'):continue
        i=record['pattern_index'];model=record['candidate_class']
        row,img,masks,pix,subs,family=data_for(i,model)
        train=subs['train']&(pix[:,0]%2==0)&(pix[:,1]%2==0);y=img[masks['roi']]
        fwd=g.BoundaryForward(family,pix[train]);full=g.BoundaryForward(family,pix)
        extra=[6,7,8,9]+([10,11] if family.prism.any() else [])
        for parent,key,active,lam in [(record,'coarse',[14,15],0.),(record,'bulk',g.BULK,0.)]+[
          (record['fits'],str(lam),g.BULK+extra,lam) for lam in LAMBDAS]:
            if parent[key]['success']:continue
            backup=OUT/f'development/before_retry_{path.stem}_{key}.json'
            if not backup.exists():dump(backup,parent[key])
            after=g.optimize(fwd,y[train],np.asarray(parent[key]['theta']),active,lam,budget=270)
            after['metrics']=g.metrics(full.predict(after['theta']),y,subs)
            parent[key]=after;dump(path,record)
            changes.append({'pattern':i,'candidate':model,'stage':key,'success':after['success'],'nfev':after['nfev']})
            print('RETRY',i,SHORT[model],key,after['success'],flush=True)
        g.cp.get_default_memory_pool().free_all_blocks()
    dump(OUT/'retry_summary.json',changes);state()

def records():
    return [json.loads(path_for(i,m).read_text()) for i in range(76) for m in MODELS]

def stages(r):return {'coarse':r['coarse'],'bulk':r['bulk'],'refined':r['fits']['0.0'],'regularized':r['fits']['0.0001']}

def summarize():
    rr=records();cfg=json.loads((OUT/'experiment_config.json').read_text());metrics=[];parameters=[]
    for r in rr:
        for stage,fit in stages(r).items():
            metrics.append({'pattern':r['pattern_index'],'source_frame_index':r['source_frame_index'],
              'original_class':r['original_class'],'candidate_class':r['candidate_class'],'stage':stage,
              'success':fit['success'],'bound_hits':fit['bound_hits'],'nfev':fit['nfev'],
              **{f'{split}_{key}':value for split,v in fit['metrics'].items() for key,value in v.items()}})
            parameters.append({'pattern':r['pattern_index'],'original_class':r['original_class'],
              'candidate_class':r['candidate_class'],'stage':stage,
              **dict(zip(g.NAMES,fit['theta']))})
    save_csv(OUT/'fit_metrics.csv',metrics);save_csv(OUT/'parameter_metrics.csv',parameters)
    by={i:{r['candidate_class']:r for r in rr if r['pattern_index']==i} for i in range(76)}
    selections=[];groups=[];transitions=[];matrices={}
    for stage in ['coarse','bulk','refined','regularized']:
        decisions=[];matrix=np.zeros((3,3),int)
        for i in range(76):
            candidates=by[i];original=g.seeds()[i]['model_name']
            ranked=sorted(MODELS,key=lambda m:stages(candidates[m])[stage]['metrics']['validation']['log_mse'])
            chosen=ranked[0];new=stages(candidates[chosen])[stage];old=stages(candidates[original])[stage]
            validation_gap=stages(candidates[ranked[1]])[stage]['metrics']['validation']['log_mse']-new['metrics']['validation']['log_mse']
            decisions.append({'pattern':i,'stage':stage,'original_class':original,'selected_class':chosen,
              'changed':chosen!=original,'original_validation_loss':old['metrics']['validation']['log_mse'],
              'selected_validation_loss':new['metrics']['validation']['log_mse'],
              'validation_margin_to_runner_up':validation_gap,
              'original_test_loss':old['metrics']['test']['log_mse'],'selected_test_loss':new['metrics']['test']['log_mse'],
              'test_loss_change':new['metrics']['test']['log_mse']-old['metrics']['test']['log_mse'],
              'selected_bound_hits':new['bound_hits']})
            matrix[MODELS.index(original),MODELS.index(chosen)]+=1
        selections+=decisions;matrices[stage]=matrix.tolist()
        scopes={'all_76':range(76),'outside_calibration_64':[i for i in range(76) if i not in cfg['calibration']]}
        scopes.update({m:[i for i in range(76) if g.seeds()[i]['model_name']==m] for m in MODELS})
        for scope,ids in scopes.items():
            d=[r for r in decisions if r['pattern'] in ids]
            old=np.array([r['original_test_loss'] for r in d]);new=np.array([r['selected_test_loss'] for r in d])
            groups.append({'scope':scope,'stage':stage,'n':len(d),'class_changes':sum(r['changed'] for r in d),
              'original_mean_test_loss':float(old.mean()),'selected_mean_test_loss':float(new.mean()),
              'reduction_vs_original_family_same_stage_percent':float(100*(1-new.mean()/old.mean())),
              'test_improved':int(np.sum(new<old)),'test_worsened':int(np.sum(new>old)),'test_unchanged':int(np.sum(new==old)),
              'selected_with_bound_hits':sum(r['selected_bound_hits']>0 for r in d),
              'selected_class_counts':dict(Counter(r['selected_class'] for r in d)),
              'median_validation_margin':float(np.median([r['validation_margin_to_runner_up'] for r in d]))})
        for original in MODELS:
            for selected in MODELS:
                d=[r for r in decisions if r['original_class']==original and r['selected_class']==selected]
                transitions.append({'stage':stage,'original_class':original,'selected_class':selected,'n':len(d),
                  'mean_test_loss_change':float(np.mean([r['test_loss_change'] for r in d])) if d else None})
    save_csv(OUT/'class_selection_by_hit.csv',selections);save_csv(OUT/'class_transitions.csv',transitions)
    selected_by_stage={stage:{r['pattern']:r for r in selections if r['stage']==stage} for stage in ['coarse','bulk','refined','regularized']}
    stage_preference_changes={f'{before}_to_{after}':sum(selected_by_stage[before][i]['selected_class']!=selected_by_stage[after][i]['selected_class'] for i in range(76))
      for before,after in [('coarse','bulk'),('coarse','refined'),('bulk','refined'),('refined','regularized')]}
    result={'models':MODELS,'class_order':[SHORT[m] for m in MODELS],'groups':groups,'transition_matrices':matrices,
      'all_candidate_fits_converged':all(f['success'] for r in rr for f in stages(r).values()),
      'test_used_for_selection':False,'original_classes_overwritten':False,
      'candidate_preference_changes_between_stages':stage_preference_changes,
      'candidate_pairs':len(rr),'new_candidate_pairs':sum(not r['provenance'].startswith('Exact reuse') for r in rr),
      'positive_control_label_agreement_with_zero_penalty':sum(a['selected_class']==b['selected_class'] for a,b in zip(
        [r for r in selections if r['stage']=='refined'],[r for r in selections if r['stage']=='regularized']))}
    dump(OUT/'summary.json',result)
    for r in groups:
        if r['scope']=='all_76':print('SUMMARY',json.dumps(r),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pilot','all','retry','summarize']);p.add_argument('--budget',type=int,default=90)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='pilot':all_fits([0,1,2],a.budget)
    elif a.stage=='all':all_fits(budget=a.budget)
    elif a.stage=='retry':retry()
    else:summarize()
