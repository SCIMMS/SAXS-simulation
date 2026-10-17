"""Conditional coherent double-hit diagnostic; no confirmed hit classification."""
from revision_common import *
from revision_models import fit_rows,evaluate_fixed,GPUScorer
from scipy.optimize import differential_evolution,minimize
import cupy as cp,time
stage=OUT/'03_rounding';cfg,data,exp=id57_data();data.pop('clean_patterns',None)
with np.load(OUT/'01_consistency/id57_mesh_czt_library_n768.npz') as f:lib=f['clean_patterns'].astype(np.float32)
rawlib=np.maximum(lib-cfg['background_adu'],0)
exp=np.maximum(exp,0)
with np.load(OUT/'02_baseline/evaluation_masks.npz') as f:train=f['train'];test=f['test']
roi=data['roi'];qx,qy=np.meshgrid(data['qaxis'],data['qaxis']);rng=np.random.default_rng(20261010)
frames=[0,1,11,129];rows=[];start=time.perf_counter()
for mode,fm,em in [('full',roi,roi),('heldout',train,test),('heldout_reverse',test,train)]:
  allraw=cp.asarray(rawlib[:,fm]);coordinates=np.column_stack([qx[fm],qy[fm]])
  for frame in frames:
    y=exp[frame][fm];scores=GPUScorer(y).scores(allraw);ids=np.argsort(scores)[-12:]
    exact,a,b,x,pred=fit_rows(y,rawlib[ids][:,fm]);best=int(np.argmax(exact));ti=int(ids[best]);signal=rawlib[ti]
    single_score,_=evaluate_fixed(exp[frame][em],signal[em],a[best],b[best],float(signal[fm].mean()))
    subset=np.sort(rng.choice(len(y),min(6000,len(y)),replace=False))
    gx=cp.asarray(coordinates[subset,0],dtype=cp.float32);gy=cp.asarray(coordinates[subset,1],dtype=cp.float32)
    template=cp.asarray(signal[fm][subset],dtype=cp.float32);scorer=GPUScorer(y[subset])
    def objective(p):
      gp=cp.asarray(np.asarray(p).reshape(3,-1),dtype=cp.float32)
      phase=gp[0,:,None]*(gx[None,:]*cp.cos(gp[1,:,None])+gy[None,:]*cp.sin(gp[1,:,None]))
      bank=template[None,:]*(1+gp[2,:,None]*cp.cos(phase))
      return -scorer.scores(bank)
    candidates=[]
    for seed in [20261010+frame,20261020+frame]:
      result=differential_evolution(objective,[(80,450),(0,np.pi),(0,1)],rng=seed,popsize=12,maxiter=100,
          vectorized=True,updating='deferred',polish=False,tol=2e-4)
      candidates.append(result.x)
    def raw(p,mask):
      d,phi,c=p
      return signal[mask]*(1+c*np.cos(d*(qx[mask]*np.cos(phi)+qy[mask]*np.sin(phi))))
    def local(p):
      if not(80<=p[0]<=450 and 0<=p[2]<=1):return 2
      return -fit_rows(y,raw(p,fm))[0][0]
    refined=[minimize(local,v,method='Nelder-Mead',options={'maxiter':160,'xatol':1e-4,'fatol':1e-6}) for v in candidates]
    winner=min(refined,key=lambda z:z.fun);par=winner.x
    fit,a2,b2,_,_=fit_rows(y,raw(par,fm))
    ev,pairfit=evaluate_fixed(exp[frame][em],raw(par,em),a2[0],b2[0],float(raw(par,fm).mean()))
    rows.append({'frame':frame,'mode':mode,'template':ti,'single_eval_score':single_score,'pair_eval_score':ev,
      'pair_minus_single':ev-single_score,'separation_nm':par[0],'direction_rad':par[1]%np.pi,'contrast':par[2],
      'fit_score':float(fit[0]),'optimizer_success':bool(winner.success)})
    if mode=='full':
      fullraw=signal*(1+par[2]*np.cos(par[0]*(qx*np.cos(par[1])+qy*np.sin(par[1]))))
      np.savez_compressed(stage/f'pair_diagnostic_frame_{frame}.npz',experimental=exp[frame],roi=roi,
        pair_fitted=(a2[0]*fullraw/float(raw(par,fm).mean())+b2[0]).astype(np.float32),
        single_fitted=(a[best]*signal/float(signal[fm].mean())+b[best]).astype(np.float32))
    save_csv(stage/'double_hit_diagnostic.csv',rows)
    print('Pair diagnostic',frame,mode,'delta',round(ev-single_score,4),'elapsed',round(time.perf_counter()-start,1),flush=True)
  del allraw;cp.get_default_memory_pool().free_all_blocks()
dump(stage/'double_hit_diagnostic_summary.json',{'frames':frames,'rows':rows,
 'model':'Two coherent identical oriented particles: Ipair=I_single*(1+c*cos(qx dx+qy dy)), 0<=c<=1.',
 'cautions':['Orientation fixed to best single-particle template chosen using training pixels only.',
 'Restricted diagnostic search; not complete joint orientation/double-particle inversion or externally labeled hit classification.',
 'Interference score improvement alone does not establish physical double hits; no real-data photon likelihood assumed.',
 'Frames0 and1 are the reviewer-discussed examples,11 and129 are comparison cases, not a purity-selected cohort.',
 'Double-hit features and streaks are reported as single-particle model inadequacy; shell versus background origin remains unresolved.']})
