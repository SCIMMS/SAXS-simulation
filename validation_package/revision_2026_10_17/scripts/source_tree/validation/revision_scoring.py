"""GPU candidate scoring with exact original-CPU refinement of shortlisted templates."""
from revision_common import *
import cupy as cp
def legacy_gpu_scores(experimental,x):
    y=cp.maximum(cp.asarray(experimental,dtype=cp.float32),0)
    n=float(x.shape[1])
    sx=cp.sum(x,axis=1,dtype=cp.float64)
    sxx=cp.sum(x*x,axis=1,dtype=cp.float64)
    sy=cp.sum(y,dtype=cp.float64)
    xy=cp.sum(x*y[None,:],axis=1,dtype=cp.float64)
    den=n*sxx-sx*sx
    a=cp.where(cp.abs(den)>1e-12,(n*xy-sx*sy)/den,0)
    b=(sy-a*sx)/n
    a=cp.maximum(a,0).astype(cp.float32);b=cp.maximum(b,0).astype(cp.float32)
    yl=cp.log1p(y);yl=(yl-yl.mean())/cp.maximum(yl.std(),1e-12)
    fl=cp.log1p(cp.maximum(a[:,None]*x+b[:,None],0))
    fl=(fl-fl.mean(axis=1,keepdims=True))/cp.maximum(fl.std(axis=1,keepdims=True),1e-12)
    return cp.asnumpy(cp.mean(fl*yl,axis=1,dtype=cp.float64))
def legacy_shortlist(image,library,roi,x_gpu):
    approx=legacy_gpu_scores(image[roi],x_gpu)
    order=np.argsort(approx)[::-1]
    n=10 if approx[order[0]]-approx[order[10]]>1e-4 else min(50,len(order))
    ids=order[:n]
    s,a,b,r=idscore.fit_scale_background_and_score(image,library[ids],roi)
    ti=int(ids[np.argmax(s)]);j=int(np.argmax(s))
    return ti,float(s[j]),float(a[j]),float(b[j]),float(r[j]),approx
