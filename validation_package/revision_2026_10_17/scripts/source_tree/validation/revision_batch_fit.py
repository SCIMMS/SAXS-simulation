"""Batched exact active-set nuisance fit on GPU; CPU final scores remain definitive."""
from revision_common import *
import cupy as cp
class BatchLibrary:
    def __init__(self,raw):
        self.mean=cp.mean(raw,axis=1,dtype=cp.float64)
        self.x=(raw/cp.maximum(self.mean[:,None],1e-30)).astype(cp.float32)
        self.n=raw.shape[1];self.sx=cp.sum(self.x,axis=1,dtype=cp.float64)
        self.sxx=cp.sum(self.x*self.x,axis=1,dtype=cp.float64)
    def score_fitted(self,y,fitted):
        y=cp.asarray(np.maximum(y,0),dtype=cp.float32)
        yl=cp.log1p(y);yl=(yl-yl.mean())/cp.maximum(yl.std(),1e-12)
        fl=cp.log1p(cp.maximum(fitted,0))
        fl=(fl-fl.mean(axis=1,keepdims=True))/cp.maximum(fl.std(axis=1,keepdims=True),1e-12)
        return cp.asnumpy(cp.mean(fl*yl,axis=1,dtype=cp.float64))
    def fit(self,y):
        cy=cp.asarray(np.maximum(y,0),dtype=cp.float32)
        sy=cp.sum(cy,dtype=cp.float64);yy=cp.sum(cy*cy,dtype=cp.float64)
        xy=cp.sum(self.x*cy[None,:],axis=1,dtype=cp.float64)
        den=self.n*self.sxx-self.sx*self.sx
        a=cp.where(den>1e-20,(self.n*xy-self.sx*sy)/den,0)
        b=(sy-a*self.sx)/self.n
        ao=cp.maximum(xy/cp.maximum(self.sxx,1e-30),0);bo=cp.maximum(sy/self.n,0)
        bad=(a<0)|(b<0)
        only=bad&((yy-2*ao*xy+ao*ao*self.sxx)<=(yy-2*bo*sy+bo*bo*self.n))
        a=cp.where(only,ao,cp.where(bad,0,a)).astype(cp.float32)
        b=cp.where(only,0,cp.where(bad,bo,b)).astype(cp.float32)
        return self.score_fitted(y,a[:,None]*self.x+b[:,None]),a,b
    def evaluate(self,y,a,b,train_mean):
        return self.score_fitted(y,(a*self.mean/train_mean)[:,None]*self.x+b[:,None])
