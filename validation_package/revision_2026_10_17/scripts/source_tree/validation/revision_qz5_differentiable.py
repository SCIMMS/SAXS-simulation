"""Differentiable five-plane surface-gradient CZT for interpretable rounding.

For q_perp != 0, F(qx,qy,kappa) = (qx*Nx + qy*Ny)/(-i*q_perp^2),
where Nc is the 2D FT of the projected oriented surface element weighted by
exp(-i*kappa*z). This follows from transverse divergence even when kappa != 0.
The Jacobian includes moving quadrature points, surface elements, and depth
phase: d(N exp(-ikz)) = (dN - ik N dz) exp(-ikz).
"""
from __future__ import annotations
from types import SimpleNamespace
from revision_common import np,sm,id57_meshes
from revision_differentiable_real import cube_coefficients,rotations
from revision_differentiable_synthetic import old
from revision_complex_projection import longitudinal_nodes,longitudinal_weights,lateral_weights,_INTERPOLATE
from revision_gpu_fourier import GPUCZT2
import cupy as cp

NAMES=['log_scale','radius_base_nm','delta_rx','delta_ry','delta_rz','log_signal','log_background']
LOW=np.array([np.log(.65),.05,-.18,-.18,-.26,-40.,-10.])
HIGH=np.array([np.log(1.5),25.5,.18,.18,.26,5.,10.])


def subdivide(vertices,faces,times):
    vertices=vertices.copy();faces=faces.copy()
    for _ in range(times):
        points=list(vertices);mid={};new=[]
        def middle(a,b):
            key=tuple(sorted([int(a),int(b)]))
            if key not in mid:
                mid[key]=len(points);points.append((vertices[a]+vertices[b])/2)
            return mid[key]
        for a,b,c in faces:
            ab,bc,ca=middle(a,b),middle(b,c),middle(c,a)
            new.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
        vertices=np.asarray(points);faces=np.asarray(new)
    return vertices,faces


class RadiusFamily:
    """Rounded cube + fixed relative core; isotropic scale and rigid orientation.

    Radius r is a Euclidean edge/corner rounding radius before global scaling.
    Its physical radius is r*exp(log_scale); side is 52*exp(log_scale).
    No anisotropy or free angular modes confound this primary radius definition.
    """
    def __init__(self,rotation=np.eye(3),flat=6,arc=12,core_subdivisions=2):
        self.geometry_count=5
        vh,vr,of=cube_coefficients(flat,arc);_,core=id57_meshes()
        cv,cf=subdivide(core.vertices,core.faces,core_subdivisions)
        self.h=np.r_[vh*26.,cv];self.r=np.r_[vr,np.zeros_like(cv)]
        self.faces=np.r_[of,cf+len(vh)].astype(np.int32)
        self.contrast=np.r_[np.ones(len(of)),np.full(len(cf),.5)]
        self.rotation=np.asarray(rotation);self.flat=flat;self.arc=arc

    def mesh(self,t):
        scale=np.exp(t[0]);v=(self.h+t[1]*self.r)*scale
        dv=np.zeros((5,len(v),3));dv[0]=v;dv[1]=self.r*scale
        local,dR=rotations(t[2:5]);R=local@self.rotation
        for j in range(3):dv[2+j]=v@(dR[j]@self.rotation).T
        v=v@R.T;dv[:2]=dv[:2]@R.T
        tri=v[self.faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
        normal=np.cross(e1,e2)*self.contrast[:,None]
        dt=dv[:,self.faces]
        dn=(np.cross(dt[:,:,1]-dt[:,:,0],e2)+np.cross(e1,dt[:,:,2]-dt[:,:,0]))*self.contrast[None,:,None]
        volume=np.sum(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))*self.contrast)/6
        return v,dv,normal,dn,volume

    def boundaries(self,t):
        v,_,normal,_,_=self.mesh(t);normal=normal/self.contrast[:,None]
        length=np.linalg.norm(normal,axis=1);active=length>1e-15
        return [(SimpleNamespace(vertices=v,faces=self.faces[active&(self.contrast==c)],
            face_normals=normal[active&(self.contrast==c)]/length[active&(self.contrast==c),None]),c) for c in [1.,.5]]


_SPLAT=r'''
extern "C" __global__ void phase_splat(
 const double* v,const int* f,const double* normal,const double* dv,const double* dn,
 const double* nodes,const double* kappas,int nv,int nf,int ns,int D,int K,
 int n,double du,double u0,double* grids){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nf*ns*D)return;
 int p=t/(nf*ns),rem=t%(nf*ns),face=rem/ns,s=rem%ns;
 int i=f[3*face],j=f[3*face+1],k=f[3*face+2];
 double b0=nodes[4*s],b1=nodes[4*s+1],b2=nodes[4*s+2],w=nodes[4*s+3]/(du*du);
 double x=(b0*v[3*i]+b1*v[3*j]+b2*v[3*k]-u0)/du;
 double y=(b0*v[3*i+1]+b1*v[3*j+1]+b2*v[3*k+1]-u0)/du;
 double z=b0*v[3*i+2]+b1*v[3*j+2]+b2*v[3*k+2];
 int ix=(int)floor(x),iy=(int)floor(y);double tx=x-ix,ty=y-iy;
 double dx=0.,dy=0.,dz=0.;
 if(p){const double* d=dv+(p-1)*nv*3;
   dx=(b0*d[3*i]+b1*d[3*j]+b2*d[3*k])/du;
   dy=(b0*d[3*i+1]+b1*d[3*j+1]+b2*d[3*k+1])/du;
   dz=b0*d[3*i+2]+b1*d[3*j+2]+b2*d[3*k+2];}
 for(int band=0;band<K;band++){
   double kz=kappas[band],sn,cs;sincos(kz*z,&sn,&cs);
   for(int oy=0;oy<2;oy++)for(int ox=0;ox<2;ox++){
     int xx=ix+ox,yy=iy+oy;if(xx<0||xx>=n||yy<0||yy>=n)continue;
     double wx=ox?tx:1.-tx,wy=oy?ty:1.-ty,cic=wx*wy;
     double dc=(ox?dx:-dx)*wy+wx*(oy?dy:-dy);
     for(int c=0;c<2;c++){
       double N=normal[3*face+c],base=N*cic,real,imag;
       if(p){double change=dn[((p-1)*nf+face)*3+c]*cic+N*dc;
         real=w*(change*cs-kz*dz*base*sn);imag=w*(-change*sn-kz*dz*base*cs);
       }else{real=w*base*cs;imag=-w*base*sn;}
       long long pos=2*((((long long)p*K+band)*2+c)*n*n+yy*n+xx);
       atomicAdd(grids+pos,real);atomicAdd(grids+pos+1,imag);
     }
   }
 }
}
'''


class FivePlaneDifferentiable:
    def __init__(self,family,qaxis,qphysical,n=384,quad=4,extent=144.,planes=5,qz_low=None):
        self.family=family;self.qaxis=np.asarray(qaxis);self.qphysical=np.asarray(qphysical)
        self.n=int(n);self.quad=int(quad);self.extent=float(extent);self.K=planes
        self.du=extent/n;self.plan=GPUCZT2(n,extent,qaxis)
        low=float(qphysical[:,2].min()) if qz_low is None else float(qz_low)
        self.kappas=longitudinal_nodes(low,0.,planes);self.kg=cp.asarray(self.kappas)
        ix,wx=lateral_weights(qaxis,qphysical[:,0]);iy,wy=lateral_weights(qaxis,qphysical[:,1])
        self.interp_arrays=[cp.asarray(a) for a in [ix,iy,wx,wy,
            longitudinal_weights(self.kappas,qphysical[:,2])]]
        self.interpolate_kernel=cp.RawKernel(_INTERPOLATE,'interpolate_surface',options=('--std=c++11',))
        qx,qy=np.meshgrid(qaxis,qaxis);q2=qx*qx+qy*qy
        self.qx=cp.asarray(qx);self.qy=cp.asarray(qy)
        self.den=cp.asarray(np.where(q2>0,-1j*q2,1.))
        self.cic=cp.asarray(np.sinc(qx*self.du/(2*np.pi))**2*np.sinc(qy*self.du/(2*np.pi))**2)
        self.origin=cp.asarray(q2==0)
        # Only nonzero transverse scattering vectors are queried by the ID57 annulus.
        assert np.min(np.linalg.norm(qphysical[:,:2],axis=1))>3*(qaxis[1]-qaxis[0])
        s,t,w=old.duffy_nodes(quad)
        self.nodes=cp.asarray(np.column_stack([(1-s)*(1-t),s,(1-s)*t,w]).ravel())
        self.ns=len(w);self.faces=cp.asarray(family.faces)
        self.kernel=cp.RawKernel(_SPLAT,'phase_splat',options=('--std=c++11',))
        self.calls=0

    def amplitude(self,theta,derivatives=True,geometry_indices=None):
        v,dv,normal,dn,volume=self.family.mesh(theta)
        indices=list(range(len(dv))) if geometry_indices is None else list(geometry_indices)
        if not derivatives:indices=[]
        dv=dv[indices];dn=dn[indices];D=1+len(indices)
        grids=cp.zeros((D,self.K,2,self.n,self.n),dtype=cp.complex128)
        arrays=[cp.asarray(a,dtype=cp.float64) for a in (v,normal,dv,dn)]
        total=len(normal)*self.ns*D
        self.kernel(((total+255)//256,),(256,),
            (arrays[0],self.faces,arrays[1],arrays[2],arrays[3],self.nodes,self.kg,
             np.int32(len(v)),np.int32(len(normal)),np.int32(self.ns),np.int32(D),np.int32(self.K),
             np.int32(self.n),np.float64(self.du),np.float64(-(self.n-1)*self.du/2),grids))
        # Batched transforms of all planes and derivative fields.
        ft=self.plan.plan(grids,axis=-1)*self.plan.phase
        del grids
        ft=self.plan.plan(ft,axis=-2)*self.plan.phase[:,None]
        bands=(self.qx*ft[:,:,0]+self.qy*ft[:,:,1])/self.den/self.cic
        # Origin values are excluded from interpolation by the annular guard.
        bands[:,:,self.origin]=0.
        out=cp.empty((D,len(self.qphysical)),dtype=cp.complex128)
        for j in range(D):
            self.interpolate_kernel(((len(self.qphysical)+127)//128,),(128,),
                (bands[j],*self.interp_arrays,np.int32(len(self.qphysical)),
                 np.int32(len(self.qaxis)),np.int32(self.K),out[j]))
        self.calls+=1
        return cp.asnumpy(out)

    def model(self,theta,geometry_indices=None):
        indices=list(range(getattr(self.family,'geometry_count',5))) if geometry_indices is None else list(geometry_indices)
        amp=self.amplitude(theta,derivatives=bool(indices),geometry_indices=indices)
        scale,bg=np.exp(theta[-2:]);signal=scale*abs(amp[0])**2;mu=signal+bg
        J=np.zeros((len(theta),len(mu)))
        for j,index in enumerate(indices):J[index]=2*scale*np.real(np.conj(amp[0])*amp[j+1])
        J[-2]=signal;J[-1]=bg
        return mu,J
