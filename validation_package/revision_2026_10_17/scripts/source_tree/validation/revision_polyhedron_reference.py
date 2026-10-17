"""Exact triangular-polyhedron Fourier amplitude, with confluent divided differences."""
from revision_common import *
def triangle_integrals(q,tri):
    z=-1j*(q@tri.T)
    d01=z[:,0]-z[:,1];d02=z[:,0]-z[:,2];d12=z[:,1]-z[:,2]
    distances=np.stack([np.abs(d01),np.abs(d02),np.abs(d12)],axis=1)
    pair=np.argmin(distances,axis=1);near=np.min(distances,axis=1)<1e-5
    with np.errstate(divide="ignore",invalid="ignore"):
        value=np.exp(z[:,0])/(d01*d02)-np.exp(z[:,1])/(d01*d12)+np.exp(z[:,2])/(d02*d12)
    for label,(i,j,k) in enumerate([(0,1,2),(0,2,1),(1,2,0)]):
        mask=near&(pair==label)
        if not mask.any():continue
        mean=(z[mask,i]+z[mask,j])/2;d=z[mask,k]-mean
        small=np.abs(d)<1e-3
        term=np.empty_like(d)
        term[small]=.5+d[small]/6+d[small]**2/24+d[small]**3/120+d[small]**4/720
        term[~small]=(np.expm1(d[~small])-d[~small])/(d[~small]**2)
        value[mask]=np.exp(mean)*term
    jac=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))
    return value*jac
def polyhedron_amplitude(q,mesh):
    q=np.asarray(q,float);q2=np.sum(q*q,axis=1);zero=q2<1e-20
    out=np.zeros(len(q),complex)
    for tri,normal in zip(mesh.vertices[mesh.faces],mesh.face_normals):
        surface=triangle_integrals(q,tri)
        out+=1j*np.divide(q@normal,q2,out=np.zeros(len(q)),where=~zero)*surface
    out[zero]=abs(sm.signed_mesh_volume(mesh.vertices,mesh.faces))
    return out
def core_shell_amplitude(q,rotation,meshes):
    local=q@rotation
    outer,core=meshes
    h=26.
    # The exact box formula avoids cancellation in the outer surface sum.
    box=52.**3*np.prod(np.sinc(local*h/np.pi),axis=1)
    return box+.5*polyhedron_amplitude(local,core)
