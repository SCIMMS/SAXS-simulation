"""Separate curvature-induced assignment changes from raster approximation."""
from revision_common import *
from revision_models import GPUScorer,fit_rows
import cupy as cp
stage=OUT/'05_ewald';cfg,data,exp=id57_data();data.pop('clean_patterns',None)
with np.load(stage/'id57_exact_roi_physical_intensities.npz') as z:bank={key:z[key] for key in ['planar','curved']}
gpu={key:cp.asarray(value) for key,value in bank.items()};rows=[]
for frame,image in enumerate(exp):
    y=np.maximum(image[data['roi']],0);found={}
    for key in ['planar','curved']:
      approximate=GPUScorer(y).scores(gpu[key]);ids=np.argsort(approximate)[-12:]
      exact,*_=fit_rows(y,bank[key][ids]);j=int(np.argmax(exact));found[key]=(int(ids[j]),float(exact[j]))
    rows.append({'frame':frame,'exact_planar_template':found['planar'][0],'exact_curved_template':found['curved'][0],
      'exact_planar_score':found['planar'][1],'exact_curved_score':found['curved'][1],
      'same_exact_best':found['planar'][0]==found['curved'][0],
      'curved_minus_planar_score':found['curved'][1]-found['planar'][1]})
save_csv(stage/'continuous_planar_curved_retrieval.csv',rows)
s={'n':209,'same_best_count':sum(r['same_exact_best'] for r in rows),
 'mean_planar_score':float(np.mean([r['exact_planar_score'] for r in rows])),
 'mean_curved_score':float(np.mean([r['exact_curved_score'] for r in rows])),
 'mean_curved_minus_planar':float(np.mean([r['curved_minus_planar_score'] for r in rows])),
 'definition':'Independent continuous polyhedral intensity banks, same exact nonnegative nuisance coefficients and masks; no raster-error difference.'}
dump(stage/'continuous_planar_curved_summary.json',s);print(json.dumps(s),flush=True)
