"""Memory-bounded float64 all-candidate nuisance fitting and log-correlation.

Fits nonnegative signal plus constant background by exact two-variable active
sets, then ranks by Pearson correlation of log1p intensities. A fused reduction
avoids materializing frame x candidate x detector arrays. No test pixels enter.
"""
from revision_common import np
import cupy as cp

_SCORE=r'''
extern "C" __global__ void logcorr(
 const double* x,const double* yl,const double* yl_sum,
 const double* a,const double* b,const int N,const int F,const int P,
 const int first_frame,double* scores){
 int i=blockIdx.x,f=first_frame+blockIdx.y,t=threadIdx.x;
 double aa=a[i*F+f],bb=b[i*F+f],anchor=log1p(aa+bb);
 double s=0.,ss=0.,sy=0.;
 for(int p=t;p<P;p+=blockDim.x){
   double z=log1p(fmax(aa*x[i*P+p]+bb,0.))-anchor;
   s+=z;ss+=z*z;sy+=z*yl[f*P+p];
 }
 __shared__ double sums[3*256];
 sums[t]=s;sums[256+t]=ss;sums[512+t]=sy;__syncthreads();
 for(int step=128;step>0;step/=2){
   if(t<step){sums[t]+=sums[t+step];sums[256+t]+=sums[256+t+step];sums[512+t]+=sums[512+t+step];}
   __syncthreads();
 }
 if(t==0){
   double mean=sums[0]/P,var=fmax(sums[256]-sums[0]*mean,0.)/P;
   double denom=P*fmax(sqrt(var),1.e-12);
   scores[f*N+i]=(sums[512]-mean*yl_sum[f])/denom;
 }
}
'''


class ExactLogCorrelation:
    def __init__(self,y,frame_batch=16):
        self.y=cp.asarray(np.maximum(y,0),dtype=cp.float64,order='C')
        self.F,self.P=self.y.shape;self.sy=self.y.sum(axis=1)
        self.yl=cp.log1p(self.y)
        self.yl-=self.yl.mean(axis=1,keepdims=True)
        self.yl/=cp.maximum(self.yl.std(axis=1,keepdims=True),1e-12)
        self.yl_sum=self.yl.sum(axis=1)
        self.frame_batch=frame_batch
        self.kernel=cp.RawKernel(_SCORE,'logcorr',options=('--std=c++11',))

    def fit(self,raw,return_coefficients=False):
        x=cp.asarray(raw,dtype=cp.float64,order='C')
        means=x.mean(axis=1);x=x/cp.maximum(means[:,None],1e-30)
        N=len(x);sx=x.sum(axis=1);sxx=(x*x).sum(axis=1)
        xy=x@self.y.T;den=self.P*sxx-sx*sx
        a=cp.where(den[:,None]>1e-20,
            (self.P*xy-sx[:,None]*self.sy[None,:])/cp.maximum(den[:,None],1e-30),0.)
        b=(self.sy[None,:]-a*sx[:,None])/self.P
        ao=cp.maximum(xy/cp.maximum(sxx[:,None],1e-30),0.)
        bo=cp.maximum(self.sy/self.P,0.)
        bad=(a<0)|(b<0)
        only=bad&((-2*ao*xy+ao*ao*sxx[:,None])<=
                  (-2*bo*self.sy+bo*bo*self.P)[None,:])
        a=cp.where(only,ao,cp.where(bad,0.,a))
        b=cp.where(only,0.,cp.where(bad,bo[None,:],b))
        scores=cp.empty((self.F,N),dtype=cp.float64)
        for first in range(0,self.F,self.frame_batch):
            self.kernel((N,min(self.frame_batch,self.F-first)),(256,),
                (x,self.yl,self.yl_sum,a,b,np.int32(N),np.int32(self.F),np.int32(self.P),np.int32(first),scores))
        result=cp.asnumpy(scores)
        if return_coefficients:return result,cp.asnumpy(a.T),cp.asnumpy(b.T),cp.asnumpy(means)
        return result
