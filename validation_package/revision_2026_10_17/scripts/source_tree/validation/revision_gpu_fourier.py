"""Reusable GPU CZT matching the production CPU coordinate convention."""
from revision_common import *
import cupy as cp
from cupyx.scipy.signal import CZT
class GPUCZT2:
    def __init__(self,n,extent,qaxis):
        self.n=n;self.extent=extent
        du=extent/n;u0=-(n-1)*du/2;q=np.asarray(qaxis,float);dq=q[1]-q[0]
        self.plan=CZT(n=n,m=len(q),w=np.exp(-1j*dq*du),a=np.exp(1j*q[0]*du))
        self.phase=cp.asarray(du*np.exp(-1j*q*u0),dtype=cp.complex128)
    def amplitude(self,projection):
        p=cp.asarray(projection,dtype=cp.complex128)
        a=self.plan(p,axis=1)*self.phase[None,:]
        return self.plan(a,axis=0)*self.phase[:,None]
    def intensity(self,projection):
        return cp.asnumpy(cp.abs(self.amplitude(projection))**2)
