"""GPU evaluation of an independent exact core-shell 3D boundary reference."""
from revision_common import *
import cupy as cp
_SRC=r"""
extern "C" __global__ void core_shell_reference(
const double* qx,const double* qy,const double* qz,const double* R,
const double* triangles,const int nf,const int nq,const double core_volume,double* out){
 int t=blockDim.x*blockIdx.x+threadIdx.x;if(t>=nq)return;
 double x=qx[t]*R[0]+qy[t]*R[3]+qz[t]*R[6];
 double y=qx[t]*R[1]+qy[t]*R[4]+qz[t]*R[7];
 double z=qx[t]*R[2]+qy[t]*R[5]+qz[t]*R[8];
 double q2=x*x+y*y+z*z;
 if(q2<1e-20){double a=52.*52.*52.+.5*core_volume;out[t]=a*a;return;}
 double real=0.,imag=0.;
 for(int f=0;f<nf;f++){
   const double* v=triangles+13*f;
   double a=x*v[0]+y*v[1]+z*v[2], b=x*v[3]+y*v[4]+z*v[5],c=x*v[6]+y*v[7]+z*v[8];
   double dab=fabs(a-b),dac=fabs(a-c),dbc=fabs(b-c),vr,vi;
   if(fmin(dab,fmin(dac,dbc))<1e-5){
     double m,d;
     if(dab<=dac&&dab<=dbc){m=.5*(a+b);d=c-m;}
     else if(dac<=dbc){m=.5*(a+c);d=b-m;}
     else{m=.5*(b+c);d=a-m;}
     double tr,ti;
     if(fabs(d)<1e-3){double d2=d*d;tr=.5-d2/24.+d2*d2/720.;ti=-d/6.+d*d2/120.;}
     else{double sd,cd;sincos(d,&sd,&cd);tr=(1.-cd)/(d*d);ti=(sd-d)/(d*d);}
     double sm,cm;sincos(m,&sm,&cm);vr=cm*tr+sm*ti;vi=cm*ti-sm*tr;
   }else{
     double sa,ca,sb,cb,sc,cc;sincos(a,&sa,&ca);sincos(b,&sb,&cb);sincos(c,&sc,&cc);
     double da=(a-b)*(a-c),db=(b-a)*(b-c),dc=(c-a)*(c-b);
     vr=-ca/da-cb/db-cc/dc;vi=sa/da+sb/db+sc/dc;
   }
   double factor=(x*v[9]+y*v[10]+z*v[11])*v[12]/q2;
   real-=factor*vi;imag+=factor*vr;
 }
 double u=26*x,v=26*y,w=26*z;
 double sx=fabs(u)<1e-8?1.-u*u/6.:sin(u)/u;
 double sy=fabs(v)<1e-8?1.-v*v/6.:sin(v)/v;
 double sz=fabs(w)<1e-8?1.-w*w/6.:sin(w)/w;
 double ar=52.*52.*52.*sx*sy*sz+.5*real,ai=.5*imag;out[t]=ar*ar+ai*ai;
}
"""
class CoreShellReference:
    def __init__(self,core,q):
        vertices=core.vertices[core.faces]
        jac=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)
        table=np.column_stack([vertices.reshape(-1,9),core.face_normals,jac])
        self.table=cp.asarray(table,dtype=cp.float64)
        self.volume=sm.signed_mesh_volume(core.vertices,core.faces);self.nf=len(core.faces)
        q=np.asarray(q).reshape(-1,3)
        self.q=[cp.asarray(q[:,i],dtype=cp.float64) for i in range(3)];self.n=len(q)
        self.kernel=cp.RawKernel(_SRC,"core_shell_reference",options=("--std=c++11",))
    def intensity(self,rotation):
        out=cp.empty(self.n,dtype=cp.float64);r=cp.asarray(np.asarray(rotation).ravel(),dtype=cp.float64)
        self.kernel(((self.n+127)//128,),(128,),(*self.q,r,self.table,np.int32(self.nf),np.int32(self.n),np.float64(self.volume),out))
        return cp.asnumpy(out)
