"""Fixed legacy GroEL classes; density-boundary refinement on physical Ewald q.

Original data, classes, production code and previous experiments are read-only.
Lengths are nm. Derived photon counts are compared using descriptive log loss.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from types import SimpleNamespace
from revision_common import ROOT, dump, save_csv, sha, np
from revision_differentiable_real import rotations
from revision_differentiable_synthetic import old
from revision_gpu_boundary_general import MeshBoundaryReference
import section5_generate_groel_model_ab_preview as legacy
import cupy as cp
import h5py
from scipy.optimize import least_squares, nnls

OUT=ROOT/'output/differentiable_groel_2026-10-11'
OUT.mkdir(parents=True,exist_ok=True)
SUP=ROOT/'validation_supporting_package/section5_groel'
HITS=SUP/'curated_run151_hits/hits_r0151_curated76_ds8.h5'
CLASSES=SUP/'retrieval_symD7_curated76/groel_retrieval_best_with_frames.csv'
DENSITY=ROOT/'tmp/revision_response_bundle_5_6_2026-10-09/groel_original_settings_density_grid.npz'
NAMES=['log_scale_x','log_scale_y','log_scale_z','delta_rx','delta_ry','delta_rz',
 'mode_x2','mode_r2','mode_x4','mode_yz','log_prism_radius','log_prism_length',
 'beam_offset_x_px','beam_offset_y_px','log_signal','log_background']
GEOM=[0,1,2,6,7,8,9,10,11];QPARAM=[3,4,5,12,13]
LOW=np.r_[np.log([.8]*3),[-.22]*3,[-.06]*4,np.log([.85,.85]),[-1.5]*2,-35.,-12.]
HIGH=np.r_[np.log([1.2]*3),[.22]*3,[.06]*4,np.log([1.15,1.15]),[1.5]*2,5.,5.]
BULK=[0,1,2,3,4,5,12,13,14,15]
LAMBDAS=[0.,.0001,.001,.01,.1,1.]

def seeds():
    return {int(r['pattern_index']):r for r in csv.DictReader(CLASSES.open())}

def mesh_arrays(volume,origin,spacing,level,reduction=.995):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
    image=vtk.vtkImageData();image.SetDimensions(*volume.shape)
    image.SetOrigin(*origin);image.SetSpacing(spacing,spacing,spacing)
    image.GetPointData().SetScalars(numpy_to_vtk(volume.ravel(order='F'),deep=True))
    extract=vtk.vtkFlyingEdges3D();extract.SetInputData(image);extract.SetValue(0,level);extract.Update()
    for target in sorted(set([reduction,.99,.98,.95,.90,0.]),reverse=True):
        if target>reduction:continue
        dec=vtk.vtkQuadricDecimation();dec.SetInputConnection(extract.GetOutputPort());dec.SetTargetReduction(target);dec.Update()
        poly=dec.GetOutput();v=vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
        f=vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1,3).astype(np.int32)
        # Oppositely oriented duplicate faces cancel exactly in the integral.
        _,inv,counts=np.unique(np.sort(f,axis=1),axis=0,return_inverse=True,return_counts=True)
        keep=np.ones(len(f),bool);ok=True
        for key in np.flatnonzero(counts>1):
            ids=np.flatnonzero(inv==key)
            if len(ids)!=2:ok=False;break
            a,b=f[ids]
            if not any(np.array_equal(a,np.roll(b[::-1],j)) for j in range(3)):ok=False;break
            keep[ids]=False
        if not ok:continue
        f=f[keep]
        edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True)
        if np.all(counts==2):break
    else:raise ValueError('No closed manifold approximation found')
    tri=v[f];signed=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
    if signed<0:f=f[:,[0,2,1]]
    return v,f

def prepare():
    rows=seeds();assert sorted(rows)==list(range(76))
    with h5py.File(HITS) as h:
        images=h['patterns_ph'][:].astype(float);valid=h['mask'][:].astype(bool)
        frame=h['selected_frame_indices'][:];lit=h['selected_lit_pixels'][:]
    assert np.isfinite(images).all() and (images>=0).all()
    for i,r in rows.items():
        assert frame[i]==int(r['source_frame_index']) and lit[i]==int(r['selected_lit_pixels'])
    points=legacy.load_density_points(DENSITY,1e-4)
    coords,weights,center=legacy.make_model_points(points,'B_tetradecamer_D7')
    all_v=[];all_f=[];all_c=[];all_labels=[];mesh_info=[]
    levels=(np.arange(12)+.5)/12;increment=points.density_max/12*1000 # e/nm^3
    for frac in levels:
        v,f=mesh_arrays(points.density,points.origin_A,points.spacing_A,frac*points.density_max)
        for copy in range(2):
            vc=v.copy()
            if copy:vc[:,1:]*=-1
            vc=(vc-center)/10
            offset=sum(len(a) for a in all_v);all_v.append(vc);all_f.append(f+offset)
            all_c.append(np.full(len(f),increment));all_labels.append(np.zeros(len(vc),np.int32))
        mesh_info.append({'fraction':frac,'vertices_heptamer':len(v),'faces_heptamer':len(f)})
        print('MESH',round(frac,4),len(f),flush=True)
    protein_v=np.concatenate(all_v);protein_f=np.concatenate(all_f);protein_c=np.concatenate(all_c)
    np.savez_compressed(OUT/'protein_boundary.npz',vertices=protein_v,faces=protein_f,contrast=protein_c)
    volumes={};charges={};prism_meta={}
    for model in sorted({r['model_name'] for r in rows.values()}):
        v=[protein_v];f=[protein_f];c=[protein_c];labels=[np.zeros(len(protein_v),np.int32)]
        if 'polyprism' in model:
            radius=35. if 'R35_' in model else 55.
            full,pw=legacy.make_polygonal_prism_points(radius,135.,14,2.,.267)
            keep=legacy.exclude_prism_protein_overlap(points,full,center,.2)
            axes=[np.unique(full[:,j]) for j in range(3)]
            # Preserve the original retained 2-A voxel centers, including overlap exclusions.
            binary=np.zeros(tuple(len(a)+2 for a in axes),np.float32)
            ix=np.rint((full[keep]-np.array([a[0] for a in axes]))/2).astype(int)+1
            binary[tuple(ix.T)]=1
            pv,pf=mesh_arrays(binary,np.array([a[0] for a in axes])-2,2.,.5,reduction=.90)
            pv/=10;f.append(pf+len(protein_v));v.append(pv);c.append(np.full(len(pf),267.));labels.append(np.ones(len(pv),np.int32))
            prism_meta[model]={'radius_A':radius,'length_A':135.,'retained_points':int(keep.sum()),
              'legacy_prism_charge_e':float(pw[keep].sum(dtype=float)),'faces':len(pf),
              'exclusion':'Original occupancy exclusion retained, moves with the prism boundary during prism-only scaling'}
        vv=np.concatenate(v);ff=np.concatenate(f);cc=np.concatenate(c);ll=np.concatenate(labels)
        tri=vv[ff];charge=np.sum(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))*cc)/6
        charges[model]=charge
        np.savez_compressed(OUT/(model+'.npz'),vertices=vv,faces=ff,contrast=cc,prism_vertex=ll)
    radius=np.hypot(*np.meshgrid(np.arange(128)-63.5,np.arange(128)-63.5))
    roi=(radius>=5)&(radius<=40)
    band=np.floor(radius/2).astype(int)
    train=roi&(band%2==0);validation=roi&(band%4==1);test=roi&(band%4==3)
    np.savez_compressed(OUT/'detector_bands.npz',roi=roi,train=train,validation=validation,test=test,valid=valid)
    assignments=[{'pattern_index':i,'source_frame_index':int(rows[i]['source_frame_index']),
      'model_name':rows[i]['model_name'],'template_index':int(rows[i]['template_index']),
      'rotation_index':int(rows[i]['rotation_index']),'shift_x':int(rows[i]['shift_x']),'shift_y':int(rows[i]['shift_y'])} for i in range(76)]
    save_csv(OUT/'fixed_class_assignments.csv',assignments)
    calibration=[]
    for model in sorted({r['model_name'] for r in rows.values()}):
        ids=sorted(i for i,r in rows.items() if r['model_name']==model)
        calibration.extend(ids[j] for j in np.linspace(0,len(ids)-1,4).astype(int))
    sources=[HITS,CLASSES,DENSITY,ROOT/'validation/section5_generate_groel_model_ab_preview.py',
      SUP/'template_library_summary/groel_templates_B_prism35_55_symD7_sep10_step128.summary.json',
      SUP/'groel_modeling_validation_note.md',SUP/'real_vs_random_control/groel_real_vs_random_control_summary.json',
      ROOT/'output/differentiable_real_2026-10-11/validation_manifest.json']
    cfg={'date':'2026-10-11','n':76,'class_assignment':'Frozen legacy best-with-frames CSV; no classification or class switching',
      'legacy_selection_caveat':'Class, seed orientation and integer alignment were selected using the full legacy ROI. New test bands are conditional evaluation, not an end-to-end independent test of selection.',
      'parameter_names':NAMES,'bounds_lower':LOW,'bounds_upper':HIGH,'bulk_indices':BULK,
      'refinement':'Four smooth even polynomial modes; prism-only radial/axial scales active only for the two fixed prism classes',
      'prior':'Fixed density differences, topology, biological assembly copies; shared smooth deformation of all boundaries; fixed class-specific prism seed',
      'model_prism_caution':'Prism/overlap boundary adjustments are phenomenological nuisance geometry and not identified protein conformational changes',
      'mesh_levels':levels,'mesh_reduction':.995,'mesh_rows':mesh_info,'mesh_charge_e':charges,'prism_meta':prism_meta,
      'mesh_processing':'Closed edge incidence checked; exactly cancelling opposite duplicate triangles removed; reduction relaxed per level if closure check fails',
      'legacy_control_context':'Radial-profile-preserving random controls had similar log correlations; this refinement is not a test of unique orientation recovery.',
      'protein_voxel_charge_e':float(weights.sum(dtype=float)),'density_center_A':center,
      'objective':'Mean squared log1p residual in pipeline-derived photon counts + lambda ||L theta||^2; descriptive, not a calibrated Poisson likelihood',
      'detector':{'image_size':128,'energy_eV':1200.,'distance_m':.15,'native_pixel_m':75e-6,'ds':8},
      'pixel_integration':'Detector bin center approximation; 2x2 bin integration assessed separately',
      'bands':'2-pixel radial bands: even=train; odd 1 mod4=validation; odd 3 mod4=test',
      'fit_stride':2,'evaluation_stride':1,'quadrature_order':2,'calibration':sorted(calibration),'lambda_candidates':LAMBDAS,
      'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},'E_drive_read_only':True}
    dump(OUT/'experiment_config.json',cfg)
    print('PREPARED',json.dumps({'charges':charges,'calibration':sorted(calibration),'faces_protein':len(protein_f)}),flush=True)

class GroELFamily:
    def __init__(self,model):
        a=np.load(OUT/(model+'.npz'));self.base=a['vertices'];self.faces=a['faces'];self.contrast=a['contrast'];self.prism=a['prism_vertex'].astype(bool)
        self.model=model;self.rotation=np.eye(3)
        v=self.base;axes=np.max(np.abs(v),axis=0);u=v[:,0]/axes[0];r2=(v[:,1]**2+v[:,2]**2)/(axes[1]**2+axes[2]**2)
        self.basis=np.column_stack([u*u-.35,r2-.35,u**4-.2,(v[:,1]**2-v[:,2]**2)/(axes[1]**2+axes[2]**2)])
    def geometry(self,t):
        shape=1+self.basis@t[6:10]
        prism_axes=np.exp([t[11],t[10],t[10]])
        mult=np.ones_like(self.base);mult[self.prism]=prism_axes
        v=self.base*mult*shape[:,None]*np.exp(t[:3])
        dv=np.zeros((len(GEOM),len(v),3))
        for j in range(3):dv[j,:,j]=v[:,j]
        for j in range(4):dv[3+j]=self.base*mult*self.basis[:,j,None]*np.exp(t[:3])
        dv[7,self.prism,1:]=v[self.prism,1:];dv[8,self.prism,0]=v[self.prism,0]
        tri=v[self.faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
        normal=np.cross(e1,e2)*self.contrast[:,None]
        dt=dv[:,self.faces];dn=(np.cross(dt[:,:,1]-dt[:,:,0],e2)+np.cross(e1,dt[:,:,2]-dt[:,:,0]))*self.contrast[None,:,None]
        return v,dv,normal,dn
    def boundaries(self,t):
        v,_,_,_=self.geometry(t);R=rotations(t[3:6])[0]@self.rotation;v=v@R.T;meshes=[]
        for c in np.unique(self.contrast):
            faces=self.faces[self.contrast==c];tri=v[faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(n,axis=1)
            active=length>1e-15;faces=faces[active];n=n[active];length=length[active]
            meshes.append((SimpleNamespace(vertices=v,faces=faces,face_normals=n/length[:,None]),c))
        return meshes

KERNEL=r'''
extern "C" __global__ void evaluate(const double* v,const int* faces,const double* normal,
 const double* dv,const double* dn,const double* nodes,const double* q,const double* dq,
 const int nv,const int nf,const int ns,const int nq,const int nb,const int ng,
 const int nd,const int chunk,double* out){
 int iq=blockIdx.x*blockDim.x+threadIdx.x;int tile=blockIdx.y;if(iq>=nq)return;
 double qr=q[3*iq],qs=q[3*iq+1],qt=q[3*iq+2];
 double q2=qr*qr+qs*qs+qt*qt;
 double ar[20],ai[20];for(int p=0;p<=ng+nd;p++){ar[p]=0.;ai[p]=0.;}
 for(int f=tile*chunk;f<min(nf,(tile+1)*chunk);f++){
  int a=faces[3*f],b=faces[3*f+1],c=faces[3*f+2];
  double nx=normal[3*f],ny=normal[3*f+1],nz=normal[3*f+2];double dot=qr*nx+qs*ny+qt*nz;
  for(int j=0;j<ns;j++){
   double ba=nodes[4*j],bb=nodes[4*j+1],bc=nodes[4*j+2],w=nodes[4*j+3];
   double x=ba*v[3*a]+bb*v[3*b]+bc*v[3*c];
   double y=ba*v[3*a+1]+bb*v[3*b+1]+bc*v[3*c+1];
   double z=ba*v[3*a+2]+bb*v[3*b+2]+bc*v[3*c+2];
   double s,co;sincos(qr*x+qs*y+qt*z,&s,&co);
   ar[0]+=w*dot*s/q2;ai[0]+=w*dot*co/q2;
   for(int p=0;p<ng+nd;p++){
    double dd,dp,den=0.;
    if(p<ng){const double* d=dv+p*nv*3;const double* n=dn+(p*nf+f)*3;
     dd=qr*n[0]+qs*n[1]+qt*n[2];
     dp=qr*(ba*d[3*a]+bb*d[3*b]+bc*d[3*c])+qs*(ba*d[3*a+1]+bb*d[3*b+1]+bc*d[3*c+1])+qt*(ba*d[3*a+2]+bb*d[3*b+2]+bc*d[3*c+2]);
    }else{const double* d=dq+((p-ng)*nq+iq)*3;
     dd=d[0]*nx+d[1]*ny+d[2]*nz;dp=d[0]*x+d[1]*y+d[2]*z;
     den=2*(qr*d[0]+qs*d[1]+qt*d[2])/q2;
    }
    ar[p+1]+=w*((dd-den*dot)*s+dot*dp*co)/q2;
    ai[p+1]+=w*((dd-den*dot)*co-dot*dp*s)/q2;
   }
  }
 }
 for(int p=0;p<=ng+nd;p++){long i=((long)p*nq+iq)*nb+tile;out[2*i]=ar[p];out[2*i+1]=ai[p];}
}
'''

def wavevectors(t,pixels,rotation,curved=True):
    local,dlocal=rotations(t[3:6]);R=local@rotation
    a=(pixels[:,0]-63.5-t[12])*.0006/.15;b=(pixels[:,1]-63.5-t[13])*.0006/.15
    k=2*np.pi/(1239.841984/1200);s=np.column_stack([a,b,np.ones(len(a))]);norm=np.linalg.norm(s,axis=1)
    q=k*s/norm[:,None];q[:,2]-=k
    derivatives=[]
    for axis in [0,1]:
        e=np.eye(3)[axis];derivatives.append(-k*.0006/.15*(e/norm[:,None]-s*s[:,axis,None]/norm[:,None]**3))
    if not curved:q[:,2]=0;derivatives=[d*np.array([1,1,0]) for d in derivatives]
    dq=[q@(d@rotation) for d in dlocal]+[d@R for d in derivatives]
    return q@R,np.asarray(dq),q

class BoundaryForward:
    def __init__(self,family,pixels,quad=2,curved=True):
        self.family=family;self.pixels=np.asarray(pixels,float);self.curved=curved;self.quad=quad
        s,t,w=old.duffy_nodes(quad);self.nodes=cp.asarray(np.column_stack([(1-s)*(1-t),s,(1-s)*t,w]))
        assert family.faces.min()>=0 and family.faces.max()<len(family.base)
        self.faces=cp.asarray(family.faces,dtype=cp.int32);self.kernel=cp.RawKernel(KERNEL,'evaluate',options=('--std=c++11',))
    def amplitude(self,t,derivatives=True,indices=None):
        v,dv,n,dn=self.family.geometry(t);q,dq,_=wavevectors(t,self.pixels,self.family.rotation,self.curved)
        if not derivatives:dv=dv[:0];dn=dn[:0];dq=dq[:0]
        elif indices is not None:
            gi=[j for j,k in enumerate(GEOM) if k in indices];qi=[j for j,k in enumerate(QPARAM) if k in indices]
            dv=dv[gi];dn=dn[gi];dq=dq[qi]
        chunk=64;nb=(len(n)+chunk-1)//chunk;nq=len(q);ng=len(dv);nd=len(dq)
        out=cp.empty((ng+nd+1,nq,nb,2),cp.float64)
        a=[cp.asarray(x,dtype=cp.float64) for x in [v,n,dv,dn,q,dq]]
        self.kernel(((nq+63)//64,nb),(64,),(a[0],self.faces,a[1],a[2],a[3],self.nodes,a[4],a[5],
          np.int32(len(v)),np.int32(len(n)),np.int32(len(self.nodes)),np.int32(nq),np.int32(nb),np.int32(ng),np.int32(nd),np.int32(chunk),out))
        summed=cp.asnumpy(out.sum(axis=2));return summed[...,0]+1j*summed[...,1]
    def model(self,t,indices=None):
        amp=self.amplitude(t,indices=indices);scale,bg=np.exp(t[-2:]);signal=scale*np.abs(amp[0])**2;mu=signal+bg
        jac=np.zeros((len(t),len(mu)));geomder=2*scale*np.real(np.conj(amp[0])[None]*amp[1:])
        active=GEOM+QPARAM if indices is None else [j for j in GEOM+QPARAM if j in indices]
        jac[active]=geomder;jac[-2]=signal;jac[-1]=bg
        return mu,jac
    def predict(self,t):
        return np.exp(t[-2])*np.abs(self.amplitude(t,False)[0])**2+np.exp(t[-1])

def data_for(i):
    row=seeds()[i]
    with h5py.File(HITS) as h:img=h['patterns_ph'][i].astype(float);mask=h['mask'][:]
    shift=[int(row['shift_y']),int(row['shift_x'])]
    img=np.roll(img,shift,axis=(0,1));mask=np.roll(mask,shift,axis=(0,1))
    # Exclude wrapped rows/columns; keep split fixed in aligned detector coordinates.
    for axis,s in enumerate(shift):
        if s:
            index=[slice(None),slice(None)];index[axis]=slice(0,s) if s>0 else slice(s,None);mask[tuple(index)]=False
    d=np.load(OUT/'detector_bands.npz');masks={k:d[k]&mask for k in ['roi','train','validation','test']}
    yy,xx=np.indices(img.shape);pix=np.column_stack([xx[masks['roi']],yy[masks['roi']]])
    fam=GroELFamily(row['model_name']);fam.rotation=legacy.quaternion_wxyz_to_matrix([float(row[k]) for k in ['qw','qx','qy','qz']]).astype(float)
    # Normalize stored float32 rotation through its unit quaternion, then polar factor.
    u,_,vt=np.linalg.svd(fam.rotation);fam.rotation=u@vt
    subs={k:masks[k][masks['roi']] for k in ['train','validation','test']}
    return row,img,masks,pix,subs,fam

def Lmatrix(prism):
    L=np.zeros((6,len(NAMES)));L[np.arange(4),np.arange(6,10)]=np.array([1.,1.,2.,1.])/.05
    if prism:L[4,10]=1/.1;L[5,11]=1/.1
    return L

def optimize(fwd,y,start,indices,lam=0.,budget=90):
    idx=np.asarray(indices);start=np.clip(start,LOW+1e-8,HIGH-1e-8);L=Lmatrix(fwd.family.prism.any());norm=np.sqrt(len(y));cache={};history=[]
    def expand(z):
        t=start.copy();t[idx]=z;return t
    def evaluate(z):
        if 'z' not in cache or not np.array_equal(z,cache['z']):
            t=expand(z);mu,j=fwd.model(t,idx);r=(np.log1p(mu)-np.log1p(y))/norm
            jac=j.T/(1+mu[:,None])/norm
            cache.update(z=z.copy(),r=np.r_[r,np.sqrt(lam)*L@t],j=np.vstack([jac,np.sqrt(lam)*L])[:,idx]);history.append(float(r@r))
        return cache
    tic=time.perf_counter()
    fit=least_squares(lambda z:evaluate(z)['r'],start[idx],jac=lambda z:evaluate(z)['j'],bounds=(LOW[idx],HIGH[idx]),
      x_scale='jac',max_nfev=budget,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    t=expand(fit.x)
    return {'theta':t,'success':bool(fit.success),'status':int(fit.status),'nfev':fit.nfev,'cost':float(fit.cost),
      'optimality':float(fit.optimality),'seconds':time.perf_counter()-tic,'bound_hits':int(np.sum(np.minimum(t-LOW,HIGH-t)[idx]<1e-4)),'history':history}

def metrics(pred,y,subs):
    a=np.log1p(pred);b=np.log1p(y)
    return {k:{'log_mse':float(np.mean((a[m]-b[m])**2)),'logcorr':float(np.corrcoef(a[m],b[m])[0,1]),
      'raw_rel_l2':float(np.linalg.norm((pred-y)[m])/np.linalg.norm(y[m])),'pixels':int(m.sum())} for k,m in subs.items()}

def fit_frame(i,lambdas,budget=90):
    row,img,masks,pix,subs,fam=data_for(i);train=subs['train']&(pix[:,0]%2==0)&(pix[:,1]%2==0)
    fwd=BoundaryForward(fam,pix[train]);full=BoundaryForward(fam,pix);y=img[masks['roi']]
    path=OUT/f'fits/pattern_{i:03d}.json'
    if path.exists():record=json.loads(path.read_text())
    else:
        t=np.zeros(16);t[-2]=-20;t[-1]=0
        raw=np.abs(fwd.amplitude(t,False)[0])**2;norm=raw.max()
        a,_=nnls(np.column_stack([raw/norm,np.ones(len(raw))]),y[train])
        t[-2]=np.log(max(a[0]/norm,1e-14));t[-1]=np.log(max(a[1],1e-4))
        coarse=optimize(fwd,y[train],t,[14,15],budget=budget)
        bulk=optimize(fwd,y[train],np.asarray(coarse['theta']),BULK,budget=budget)
        for fit in [coarse,bulk]:fit['metrics']=metrics(full.predict(np.asarray(fit['theta'])),y,subs)
        record={'pattern_index':i,'source_frame_index':int(row['source_frame_index']),'model_name':row['model_name'],
          'legacy_seed':row,'rotation':fam.rotation,'coarse':coarse,'bulk':bulk,'fits':{}}
        dump(path,record)
    extra=[6,7,8,9]+([10,11] if fam.prism.any() else [])
    for lam in lambdas:
        if str(lam) in record['fits']:continue
        fit=optimize(fwd,y[train],np.asarray(record['bulk']['theta']),BULK+extra,lam,budget)
        fit['metrics']=metrics(full.predict(fit['theta']),y,subs);record['fits'][str(lam)]=fit;dump(path,record)
        print('FIT',i,row['model_name'],lam,fit['success'],fit['nfev'],round(fit['metrics']['test']['log_mse'],5),round(fit['seconds'],1),flush=True)
    cp.get_default_memory_pool().free_all_blocks()

def qa():
    cfg=json.loads((OUT/'experiment_config.json').read_text());rows=seeds();result=[]
    for model in sorted({r['model_name'] for r in rows.values()}):
        i=min(i for i,r in rows.items() if r['model_name']==model);_,img,masks,pix,subs,fam=data_for(i)
        pix=pix[np.linspace(0,len(pix)-1,97).astype(int)];fwd=BoundaryForward(fam,pix)
        t=np.zeros(16);t[:3]=[.01,-.01,.02];t[3:6]=[.015,-.02,.01];t[6:10]=[.004,-.005,.002,.003];t[10:12]=[.01,-.01];t[12:14]=[.1,-.12];t[-2:]=[-20.,-.3]
        tic=time.perf_counter();amp=fwd.amplitude(t);sec=time.perf_counter()-tic
        q=wavevectors(t,pix,fam.rotation)[2]
        exact=MeshBoundaryReference(fam.boundaries(t),q).intensity(np.eye(3))
        rel=np.linalg.norm(np.abs(amp[0])**2-exact)/np.linalg.norm(exact)
        errs={}
        for j in range(14):
            if j in [10,11] and not fam.prism.any():continue
            dt=np.zeros(16);dt[j]=1e-5
            fd=(fwd.amplitude(t+dt,False)[0]-fwd.amplitude(t-dt,False)[0])/(2e-5)
            # Amplitude rows use GEOM then QPARAM ordering.
            analytic=amp[1+(GEOM+QPARAM).index(j)]
            errs[NAMES[j]]=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(fd),np.linalg.norm(analytic),1e-12))
        result.append({'model_name':model,'pattern':i,'forward_relative_l2':rel,'amplitude_gradient_relative_l2':errs,
          'benchmark_points':len(pix),'benchmark_seconds':sec,'faces':len(fam.faces)})
        print('QA',model,rel,max(errs.values()),sec,flush=True)
    dump(OUT/'numerical_qa.json',result)
    assert max(r['forward_relative_l2'] for r in result)<.001
    assert max(max(r['amplitude_gradient_relative_l2'].values()) for r in result)<.001

def select():
    cfg=json.loads((OUT/'experiment_config.json').read_text());items=[]
    for lam in cfg['lambda_candidates']:
        rr=[json.loads((OUT/f'fits/pattern_{i:03d}.json').read_text())['fits'][str(lam)] for i in cfg['calibration']]
        assert all(r['success'] for r in rr),'Unconverged calibration'
        items.append({'lambda':lam,'validation_mean':float(np.mean([r['metrics']['validation']['log_mse'] for r in rr]))})
    chosen=min(items,key=lambda r:r['validation_mean'])['lambda']
    positive=min((r for r in items if r['lambda']>0),key=lambda r:r['validation_mean'])['lambda']
    dump(OUT/'lambda_selection.json',{'lambda':chosen,'candidates':items,'calibration':cfg['calibration'],
      'positive_control_lambda':positive,'positive_control_note':'Best strictly positive penalty for the requested regularization comparison; primary penalty selection still includes zero',
      'test_used_for_penalty_selection':False,'selection_scope':'Fixed existing classes; one common lambda weighted equally per calibration frame'})
    print('SELECT',json.dumps({'lambda':chosen,'candidates':items}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','qa','pilot','calibrate','select','all']);parser.add_argument('--budget',type=int,default=90)
    a=parser.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='qa':qa()
    elif a.stage=='select':select()
    else:
        cfg=json.loads((OUT/'experiment_config.json').read_text())
        if a.stage=='pilot':ids=[0];lams=[0.,.1]
        elif a.stage=='calibrate':ids=cfg['calibration'];lams=cfg['lambda_candidates']
        else:
            ids=range(76);selection=json.loads((OUT/'lambda_selection.json').read_text())
            lams=sorted(set([0.,selection['lambda'],selection['positive_control_lambda']]))
        for i in ids:fit_frame(i,lams,a.budget)
