"""Orientation-specific longitudinal Fourier projections of closed density meshes.

Research extension, kept separate from the manuscript's production simulator.
For a closed, outward-oriented constant-density domain, each signed ray crossing
at z contributes sign * (1-exp(-i*kappa*z))/(i*kappa).  The kappa=0 limit is
the existing signed-depth projection.  No voxel grid or ray-depth sorting is used.
"""
from __future__ import annotations

from revision_common import np, sm
from revision_gpu_fourier import GPUCZT2
import cupy as cp

_RASTER = r'''
extern "C" {
__device__ bool top_left(double ax,double ay,double bx,double by){
    double dy=by-ay,dx=bx-ax;return dy<0. || (dy==0. && dx>0.);
}
// Evaluate a shared edge in a canonical direction, so adjacent triangles see
// exactly opposite signed edge values. Normalize triangle winding before the
// top-left test; use no triangle-dependent barycentric epsilon.
__device__ double edge(double ax,double ay,double bx,double by,double x,double y){
    if(ax>bx || (ax==bx && ay>by))return -((ax-bx)*(y-by)-(ay-by)*(x-bx));
    return (bx-ax)*(y-ay)-(by-ay)*(x-ax);
}
__device__ bool accept(double t,bool inclusive){return t>0. || (t==0. && inclusive);}
__device__ float sinc1(float x){return fabsf(x)<1.e-4f?1.f-x*x/6.f:sinf(x)/x;}
__global__ void partial_projection(
 const double* xyz,const int* faces,const double* sd,const float* contrast,
 const float* kappas,const int nf,const int K,const int N,const float extent,
 float* output,float* balance){
 int f=blockIdx.x;if(f>=nf)return;
 double normal=sd[f];if(fabs(normal)<=1.e-12)return;
 float sign=normal>0.f?1.f:-1.f,weight=sign*contrast[f];
 int a=faces[3*f],b=faces[3*f+1],c=faces[3*f+2];
 double du=(double)extent/N,origin=-extent*.5+.5*du;
 double x0=(xyz[3*a]-origin)/du,y0=(xyz[3*a+1]-origin)/du,z0=xyz[3*a+2];
 double x1=(xyz[3*b]-origin)/du,y1=(xyz[3*b+1]-origin)/du,z1=xyz[3*b+2];
 double x2=(xyz[3*c]-origin)/du,y2=(xyz[3*c+1]-origin)/du,z2=xyz[3*c+2];
 int xmin=max(0,(int)floor(fmin(x0,fmin(x1,x2))-1.));
 int xmax=min(N-1,(int)ceil(fmax(x0,fmax(x1,x2))+1.));
 int ymin=max(0,(int)floor(fmin(y0,fmin(y1,y2))-1.));
 int ymax=min(N-1,(int)ceil(fmax(y0,fmax(y1,y2))+1.));
 if(xmax<xmin||ymax<ymin)return;
 double den=edge(x0,y0,x1,y1,x2,y2);if(fabs(den)<=1.e-20)return;
 double orient=den>0.?1.:-1.;
 bool e0=orient>0?top_left(x1,y1,x2,y2):top_left(x2,y2,x1,y1);
 bool e1=orient>0?top_left(x2,y2,x0,y0):top_left(x0,y0,x2,y2);
 bool e2=orient>0?top_left(x0,y0,x1,y1):top_left(x1,y1,x0,y0);
 int nx=xmax-xmin+1,total=nx*(ymax-ymin+1),pixels=N*N;
 for(int t=threadIdx.x;t<total;t+=blockDim.x){
   int y=ymin+t/nx,x=xmin+t%nx;
   double t0=edge(x1,y1,x2,y2,x,y),t1=edge(x2,y2,x0,y0,x,y),t2=edge(x0,y0,x1,y1,x,y);
   if(!(accept(orient*t0,e0)&&accept(orient*t1,e1)&&accept(orient*t2,e2)))continue;
   float z=(float)((t0*z0+t1*z1+t2*z2)/den);int p=y*N+x;
   // Stable at kappa=0: sin(k*z)/k and -(1-cos(k*z))/k.
   for(int j=0;j<K;j++){
     float k=kappas[j],u=k*z,h=sinc1(.5f*u);
     int pos=2*(j*pixels+p);
     atomicAdd(output+pos,weight*z*sinc1(u));
     atomicAdd(output+pos+1,-weight*.5f*k*z*z*h*h);
   }
   // Every closed domain should have zero signed crossing balance along each ray.
   if(balance)atomicAdd(balance+p,sign);
 }
}
}
'''

_INTERPOLATE = r'''
extern "C" __global__ void interpolate_surface(
 const double* planes,const int* ix,const int* iy,
 const double* wx,const double* wy,const double* wk,
 const int count,const int M,const int K,double* out){
 int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=count)return;
 double re=0.,im=0.;
 for(int j=0;j<K;j++){
   double wz=wk[j*count+p];if(wz==0.)continue;
   double r=0.,s=0.;
   for(int a=0;a<4;a++)for(int b=0;b<4;b++){
     double w=wy[4*p+a]*wx[4*p+b];if(w==0.)continue;
     int pos=2*((j*M+iy[p]+a-1)*M+ix[p]+b-1);
     r+=w*planes[pos];s+=w*planes[pos+1];
   }
   re+=wz*r;im+=wz*s;
 }
 out[2*p]=re;out[2*p+1]=im;
}
'''


def longitudinal_nodes(low, high, count):
    """Nested Chebyshev-Lobatto nodes, including both band endpoints."""
    if count < 2:
        return np.array([0.], dtype=float)
    x=np.cos(np.linspace(np.pi,0,count))
    result=(low+high)/2+(high-low)*x/2
    result[0]=low;result[-1]=high
    return result


def longitudinal_weights(nodes, targets, interpolation='barycentric'):
    nodes=np.asarray(nodes,float);targets=np.asarray(targets,float)
    if len(nodes)==1:
        return np.ones((1,len(targets)))
    if np.min(targets)<nodes[0]-1e-12 or np.max(targets)>nodes[-1]+1e-12:
        raise ValueError('Target q_z lies outside the sampled band.')
    if interpolation=='linear':
        j=np.clip(np.searchsorted(nodes,targets)-1,0,len(nodes)-2)
        t=(targets-nodes[j])/(nodes[j+1]-nodes[j])
        out=np.zeros((len(nodes),len(targets)))
        out[j,np.arange(len(targets))]=1-t
        out[j+1,np.arange(len(targets))]=t
        return out
    if interpolation!='barycentric':
        raise ValueError(interpolation)
    # General barycentric weights also permit non-Lobatto nodes in controls.
    delta=nodes[:,None]-nodes[None,:];np.fill_diagonal(delta,1.)
    weights=1/np.prod(delta,axis=1)
    d=targets[None,:]-nodes[:,None]
    exact=np.abs(d)<1e-14
    safe=np.where(exact,1.,d)
    out=weights[:,None]/safe
    out/=np.sum(out,axis=0,keepdims=True)
    hit=exact.any(axis=0)
    out[:,hit]=exact[:,hit].astype(float)
    return out


def lateral_weights(axis, targets):
    t=(np.asarray(targets)-axis[0])/(axis[1]-axis[0])
    i=np.floor(t).astype(np.int32);a=t-i
    if (i<1).any() or (i+2>=len(axis)).any():
        raise ValueError('Four-point lateral interpolation requires a one-pixel guard band.')
    weights=np.column_stack([-a*(a-1)*(a-2)/6,
        (a+1)*(a-1)*(a-2)/2,-(a+1)*a*(a-2)/2,(a+1)*a*(a-1)/6])
    return i,weights


class ComplexMeshProjector:
    """Resident mesh geometry; all kappa planes share triangle coverage work."""
    def __init__(self,meshes,n=768,extent=144.):
        self.n=int(n);self.extent=float(extent)
        vertices=[];normals=[];faces=[];contrasts=[];offset=0
        for mesh,contrast in meshes:
            volume=sm.signed_mesh_volume(mesh.vertices,mesh.faces)
            if volume<=0:
                raise ValueError('Each density domain must be closed and outward-oriented.')
            vertices.append(mesh.vertices);normals.append(mesh.face_normals)
            faces.append(mesh.faces+offset);offset+=len(mesh.vertices)
            contrasts.append(np.full(len(mesh.faces),contrast))
        self.vertices=np.concatenate(vertices).astype(np.float64)
        self.normals=np.concatenate(normals).astype(np.float64)
        self.faces=cp.asarray(np.concatenate(faces),dtype=cp.int32)
        self.contrasts=cp.asarray(np.concatenate(contrasts),dtype=cp.float32)
        self.nf=len(self.normals)
        self.kernel=cp.RawKernel(_RASTER,'partial_projection',options=('--std=c++11',))

    def project(self,rotation,kappas,diagnostics=False):
        # Double-precision edge tests avoid unpaired crossings and eliminate
        # the production real-raster's negative-value clamp in the complex path.
        rotation=np.asarray(rotation,dtype=np.float64)
        xyz=cp.asarray(self.vertices@rotation.T,dtype=cp.float64)
        sd=cp.asarray(self.normals@rotation[2],dtype=cp.float64)
        ks=cp.asarray(kappas,dtype=cp.float32)
        out=cp.zeros((len(kappas),self.n,self.n),dtype=cp.complex64)
        balance=cp.zeros((self.n,self.n),dtype=cp.float32) if diagnostics else None
        self.kernel((self.nf,),(128,),
            (xyz,self.faces,sd,self.contrasts,ks,np.int32(self.nf),np.int32(len(kappas)),
             np.int32(self.n),np.float32(self.extent),out,balance if diagnostics else np.uint64(0)))
        return (out,balance) if diagnostics else out


class ComplexProjectionRenderer:
    """Complex projection stack -> 2D CZTs -> complex Ewald-surface interpolation.

    Raster edge geometry uses float64; partial projections accumulate complex64.
    CZTs and interpolation use complex128.
    Barycentric interpolation in q_z and four-point Lagrange interpolation in
    physical q_x/q_y precede the intensity calculation. No intensity interpolation.
    """
    def __init__(self,meshes,qaxis,qphysical,roi,n=768,extent=144.,planes=5):
        self.projector=ComplexMeshProjector(meshes,n,extent)
        self.plan=GPUCZT2(n,extent,qaxis)
        self.qaxis=np.asarray(qaxis);self.roi=np.asarray(roi,bool)
        self.qphysical=np.asarray(qphysical,float);self.count=len(qphysical)
        self.kappas=longitudinal_nodes(float(qphysical[:,2].min()),0.,planes)
        self.ix,self.wx=lateral_weights(self.qaxis,qphysical[:,0])
        self.iy,self.wy=lateral_weights(self.qaxis,qphysical[:,1])
        self.ixg=cp.asarray(self.ix);self.iyg=cp.asarray(self.iy)
        self.wxg=cp.asarray(self.wx);self.wyg=cp.asarray(self.wy)
        self.weights={mode:cp.asarray(longitudinal_weights(self.kappas,qphysical[:,2],mode))
                      for mode in ['barycentric','linear']}
        self.roig=cp.asarray(self.roi)
        self.kernel=cp.RawKernel(_INTERPOLATE,'interpolate_surface',options=('--std=c++11',))

    def transform_planes(self,rotation):
        p=self.projector.project(rotation,self.kappas)
        f=cp.empty((len(self.kappas),len(self.qaxis),len(self.qaxis)),dtype=cp.complex128)
        for j in range(len(self.kappas)):
            f[j]=self.plan.amplitude(p[j])
        return f

    def interpolate(self,f,mode='barycentric'):
        out=cp.empty(self.count,dtype=cp.complex128)
        self.kernel(((self.count+127)//128,),(128,),
            (f,self.ixg,self.iyg,self.wxg,self.wyg,self.weights[mode],np.int32(self.count),
             np.int32(len(self.qaxis)),np.int32(len(self.kappas)),out))
        return out

    def amplitude(self,rotation,mode='barycentric'):
        return self.interpolate(self.transform_planes(rotation),mode)

    def intensity(self,rotation,mode='barycentric'):
        a=self.amplitude(rotation,mode)
        return cp.asnumpy(a.real*a.real+a.imag*a.imag)

    def planar_amplitude(self,rotation,physical_transverse=False):
        p=self.projector.project(rotation,[0.])[0]
        f=self.plan.amplitude(p)
        if not physical_transverse:
            return f[self.roig]
        # Reuse the same lateral interpolator with one q_z plane and unit weights.
        out=cp.empty(self.count,dtype=cp.complex128)
        ones=cp.ones((1,self.count),dtype=cp.float64)
        self.kernel(((self.count+127)//128,),(128,),
            (f,self.ixg,self.iyg,self.wxg,self.wyg,ones,np.int32(self.count),
             np.int32(len(self.qaxis)),np.int32(1),out))
        return out
