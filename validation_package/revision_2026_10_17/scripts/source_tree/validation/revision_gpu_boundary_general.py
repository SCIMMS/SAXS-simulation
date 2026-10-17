"""Generic exact triangle boundary reference for selected rounded-mesh Ewald checks."""
from revision_gpu_surface_reference import *
from revision_gpu_surface_reference import _SRC
from revision_polyhedron_reference import polyhedron_amplitude
_GENERIC=_SRC.replace('core_shell_reference','generic_reference').replace('52.*52.*52.+.5*core_volume','core_volume')
start=_GENERIC.index(' double u=26*x')
_GENERIC=_GENERIC[:start]+' out[t]=real*real+imag*imag;\n}\n'
class MeshBoundaryReference(CoreShellReference):
    def __init__(self,meshes,q):
        tables=[];volume=0
        for mesh,contrast in meshes:
            vertices=mesh.vertices[mesh.faces]
            jac=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)*contrast
            tables.append(np.column_stack([vertices.reshape(-1,9),mesh.face_normals,jac]))
            volume+=contrast*sm.signed_mesh_volume(mesh.vertices,mesh.faces)
        table=np.concatenate(tables)
        self.table=cp.asarray(table,dtype=cp.float64);self.volume=volume;self.nf=len(table)
        q=np.asarray(q).reshape(-1,3);self.q=[cp.asarray(q[:,i],dtype=cp.float64) for i in range(3)];self.n=len(q)
        self.kernel=cp.RawKernel(_GENERIC,'generic_reference',options=('--std=c++11',))
    def intensity(self,rotation):
        chunks=[]
        for start in range(0,self.n,4096):
            count=min(4096,self.n-start);out=cp.empty(count,dtype=cp.float64)
            r=cp.asarray(np.asarray(rotation).ravel(),dtype=cp.float64)
            self.kernel(((count+127)//128,),(128,),(*(v[start:start+count] for v in self.q),r,self.table,
              np.int32(self.nf),np.int32(count),np.float64(self.volume),out))
            chunks.append(cp.asnumpy(out))
        return np.concatenate(chunks)
