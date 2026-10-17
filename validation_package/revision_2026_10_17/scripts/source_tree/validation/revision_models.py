"""Projected ellipsoid model and shared exact nonnegative nuisance fit."""
from revision_common import *
import cupy as cp

def sphere_factor(t):
    t=np.asarray(t,float);small=np.abs(t)<1e-5
    return np.where(small,1-t*t/10+t**4/280,3*(np.sin(t)-t*np.cos(t))/np.maximum(t,1e-30)**3)

def ellipse_raw(params,qxv,qyv):
    p=np.asarray(params,float)
    one=p.ndim==1
    if one:p=p[:,None]
    a=p[0,:,None];b=a*p[1,:,None];phi=p[2,:,None]
    qu=qxv[None,:]*np.cos(phi)+qyv[None,:]*np.sin(phi)
    qv=-qxv[None,:]*np.sin(phi)+qyv[None,:]*np.cos(phi)
    value=sphere_factor(np.sqrt((a*qu)**2+(b*qv)**2))**2
    return value[0] if one else value

def fit_rows(y,x):
    x=np.atleast_2d(x).astype(float)
    x=x/np.maximum(x.mean(axis=1,keepdims=True),1e-30)
    y=np.maximum(np.asarray(y,float),0);n=len(y)
    sx=x.sum(axis=1);sxx=(x*x).sum(axis=1);sy=y.sum();xy=x@y
    den=n*sxx-sx*sx
    a=np.divide(n*xy-sx*sy,den,out=np.zeros(len(x)),where=den>1e-20)
    b=(sy-a*sx)/n
    bad=(a<0)|(b<0)
    ao=np.maximum(np.divide(xy,sxx,out=np.zeros(len(x)),where=sxx>1e-20),0)
    sse_o=((y[None,:]-ao[:,None]*x)**2).sum(axis=1)
    bo=max(float(y.mean()),0);sse_b=float(((y-bo)**2).sum())
    use_o=bad&(sse_o<=sse_b);use_b=bad&~use_o
    a=np.where(use_o,ao,np.where(use_b,0,a));b=np.where(use_o,0,np.where(use_b,bo,b))
    fitted=a[:,None]*x+b[:,None]
    yl=idscore.zscore(np.log1p(y))
    fl=np.log1p(np.maximum(fitted,0))
    fl=(fl-fl.mean(axis=1,keepdims=True))/np.maximum(fl.std(axis=1,keepdims=True),1e-12)
    scores=(fl*yl).mean(axis=1)
    return scores,a,b,x,fitted

def evaluate_fixed(y,x,a,b,train_mean):
    fitted=a*x/train_mean+b
    score=float(np.mean(idscore.zscore(np.log1p(np.maximum(y,0)))*idscore.zscore(np.log1p(np.maximum(fitted,0)))))
    return score,fitted
class GPUScorer:
    """Approximate GPU score for search; final candidate scores use float64 CPU."""
    def __init__(self,y):
        self.y=cp.asarray(np.maximum(y,0),dtype=cp.float32)
        self.n=len(y);self.sy=cp.sum(self.y,dtype=cp.float64);self.yy=cp.sum(self.y*self.y,dtype=cp.float64)
        self.yl=cp.log1p(self.y)
        self.yl=(self.yl-self.yl.mean())/cp.maximum(self.yl.std(),1e-12)
    def scores(self,raw):
        x=cp.atleast_2d(raw)
        x=x/cp.maximum(x.mean(axis=1,keepdims=True),1e-30)
        sx=cp.sum(x,axis=1,dtype=cp.float64);sxx=cp.sum(x*x,axis=1,dtype=cp.float64)
        xy=cp.sum(x*self.y[None,:],axis=1,dtype=cp.float64);den=self.n*sxx-sx*sx
        a=cp.where(den>1e-20,(self.n*xy-sx*self.sy)/den,0);b=(self.sy-a*sx)/self.n
        bad=(a<0)|(b<0);ao=cp.maximum(xy/cp.maximum(sxx,1e-30),0);bo=cp.maximum(self.sy/self.n,0)
        use_o=bad&((self.yy-2*ao*xy+ao*ao*sxx)<=(self.yy-2*bo*self.sy+bo*bo*self.n))
        a=cp.where(use_o,ao,cp.where(bad,0,a)).astype(cp.float32)
        b=cp.where(use_o,0,cp.where(bad,bo,b)).astype(cp.float32)
        fl=cp.log1p(cp.maximum(a[:,None]*x+b[:,None],0))
        fl=(fl-fl.mean(axis=1,keepdims=True))/cp.maximum(fl.std(axis=1,keepdims=True),1e-12)
        return cp.asnumpy(cp.mean(fl*self.yl,axis=1,dtype=cp.float64))
class GPUEllipsoid:
    def __init__(self,qx,qy,y):
        self.qx=cp.asarray(qx,dtype=cp.float32);self.qy=cp.asarray(qy,dtype=cp.float32)
        self.scorer=GPUScorer(y)
    def __call__(self,p):
        p=cp.asarray(np.asarray(p).reshape(3,-1),dtype=cp.float32)
        a=p[0,:,None];b=a*p[1,:,None];phi=p[2,:,None]
        qu=self.qx[None,:]*cp.cos(phi)+self.qy[None,:]*cp.sin(phi)
        qv=-self.qx[None,:]*cp.sin(phi)+self.qy[None,:]*cp.cos(phi)
        t=cp.sqrt((a*qu)**2+(b*qv)**2)
        factor=3*(cp.sin(t)-t*cp.cos(t))/cp.maximum(t,1e-15)**3
        raw=factor*factor
        return -self.scorer.scores(raw)
