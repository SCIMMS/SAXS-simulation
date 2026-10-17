"""Real ID57 core-shell refinement. Original paper/simulator are read-only.

Training-only legacy seeds; interleaved radial validation/test bands; ADU
log-residual fitting, with no claim of calibrated Poisson measurement noise.
"""
from __future__ import annotations
import argparse, csv, time
from types import SimpleNamespace
from revision_common import ROOT, OUT as PRIOR, dump, save_csv, sha, np, id57_data, id57_meshes
from revision_differentiable_synthetic import DifferentiableForward, old
from revision_gpu_boundary_general import MeshBoundaryReference
from scipy.optimize import least_squares, nnls

OUT=ROOT/'output/differentiable_real_2026-10-11'
OUT.mkdir(parents=True,exist_ok=True)
NAMES=['log_scale','log_aspect_x','log_aspect_y','radius_base_nm',
       'delta_rx','delta_ry','delta_rz','cos4','sin4','cos6','sin6',
       'log_signal','log_background']
LOW=np.r_[np.log([.65,.7,.7]),0.,[-.18,-.18,-.26],np.full(4,-.08),-40.,-10.]
HIGH=np.r_[np.log([1.5,1.4,1.4]),22.,[.18,.18,.26],np.full(4,.08),5.,10.]
LAMBDAS=[0.,.0001,.001,.01,.1,1.]

def cube_coefficients(flat=10,arc=10):
    """Affine h/r vertices with fixed, watertight rounded-cube topology."""
    import itertools
    def raw(h,r):
        b=h-r;points=[];triangles=[]
        angles=np.linspace(0,np.pi/2,arc+1);line=np.linspace(-1,1,flat+1)
        def patch(a):
            offset=len(points);ny,nx,_=a.shape;points.extend(a.reshape(-1,3))
            for j in range(ny-1):
                for i in range(nx-1):
                    k=offset+j*nx+i;triangles.extend([[k,k+1,k+nx+1],[k,k+nx+1,k+nx]])
        for axis in range(3):
            rest=[j for j in range(3) if j!=axis]
            for sign in [-1,1]:
                a=np.zeros((flat+1,flat+1,3));a[...,axis]=sign*h
                a[...,rest[0]]=b*line[None,:];a[...,rest[1]]=b*line[:,None];patch(a)
            for signs in itertools.product([-1,1],repeat=2):
                a=np.zeros((flat+1,arc+1,3));a[...,axis]=b*line[:,None]
                a[...,rest[0]]=signs[0]*(b+r*np.cos(angles))[None,:]
                a[...,rest[1]]=signs[1]*(b+r*np.sin(angles))[None,:];patch(a)
        theta,phi=np.meshgrid(angles,angles,indexing='ij')
        sphere=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=-1)
        for signs in itertools.product([-1,1],repeat=3):patch((b+r*sphere)*np.asarray(signs))
        return np.asarray(points),np.asarray(triangles)
    v,faces=raw(26.,10.);vh,_=raw(1.,0.);vr,_=raw(0.,1.)
    mapping={};ids=[];unique=[]
    for i,p in enumerate(v):
        key=tuple(np.round(p,9))
        if key not in mapping:mapping[key]=len(unique);unique.append(i)
        ids.append(mapping[key])
    faces=np.asarray(ids)[faces];faces=faces[np.array([len(set(f))==3 for f in faces])]
    v=v[unique];t=v[faces]
    inward=np.einsum('ij,ij->i',np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),t.mean(axis=1))<0
    faces[inward]=faces[inward][:,[0,2,1]]
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True);assert np.all(counts==2)
    return vh[unique],vr[unique],faces

def rotations(angles):
    matrices=[];derivatives=[]
    for axis,a in enumerate(angles):
        c,s=np.cos(a),np.sin(a)
        if axis==0:R=np.array([[1,0,0],[0,c,-s],[0,s,c]]);D=np.array([[0,0,0],[0,-s,-c],[0,c,-s]])
        elif axis==1:R=np.array([[c,0,s],[0,1,0],[-s,0,c]]);D=np.array([[-s,0,c],[0,0,0],[-c,0,-s]])
        else:R=np.array([[c,-s,0],[s,c,0],[0,0,1]]);D=np.array([[-s,-c,0],[c,-s,0],[0,0,0]])
        matrices.append(R);derivatives.append(D)
    x,y,z=matrices;dx,dy,dz=derivatives
    return z@y@x,[z@y@dx,z@dy@x,dz@y@x]

class CoreShellFamily:
    def __init__(self,rotation=np.eye(3),flat=10,arc=10):
        vh,vr,of=cube_coefficients(flat,arc);_,core=id57_meshes()
        # Large flat core triangles need spatial subdivision before quadrature.
        cv=core.vertices.copy();cf=core.faces.copy()
        for _ in range(3):
            points=list(cv);mid={};new=[]
            def middle(a,b):
                key=tuple(sorted([int(a),int(b)]))
                if key not in mid:mid[key]=len(points);points.append((cv[a]+cv[b])/2)
                return mid[key]
            for a,b,c in cf:
                ab,bc,ca=middle(a,b),middle(b,c),middle(c,a)
                new.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
            cv=np.asarray(points);cf=np.asarray(new)
        self.nouter=len(vh);self.h=np.r_[vh*26.,cv]
        self.r=np.r_[vr,np.zeros_like(cv)]
        self.faces=np.r_[of,cf+len(vh)]
        self.contrast=np.r_[np.ones(len(of)),np.full(len(cf),.5)]
        self.rotation=rotation
        reference=self.h+10.*self.r
        phi=np.arctan2(reference[:,1],reference[:,0]);rho2=(reference[:,0]**2+reference[:,1]**2)/np.sum(reference**2,axis=1)
        self.basis=np.column_stack([rho2*f(k*phi) for k in (4,6) for f in (np.cos,np.sin)])
    def mesh(self,t):
        v0=self.h+t[3]*self.r;deform=1+self.basis@t[7:11]
        axes=np.exp([t[0]+t[1],t[0]+t[2],t[0]])
        v=v0*deform[:,None]*axes;dv=np.zeros((11,len(v),3))
        dv[0]=v;dv[1,:,0]=v[:,0];dv[2,:,1]=v[:,1]
        dv[3]=self.r*deform[:,None]*axes
        for j in range(4):dv[7+j]=v0*self.basis[:,j,None]*axes
        local,dR=rotations(t[4:7]);R=local@self.rotation
        for j in range(3):dv[4+j]=v@(dR[j]@self.rotation).T
        v=v@R.T
        for j in [0,1,2,3,7,8,9,10]:dv[j]=dv[j]@R.T
        tri=v[self.faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
        normal=np.cross(e1,e2)*self.contrast[:,None]
        dt=dv[:,self.faces]
        dn=(np.cross(dt[:,:,1]-dt[:,:,0],e2)+np.cross(e1,dt[:,:,2]-dt[:,:,0]))*self.contrast[None,:,None]
        volume=np.sum(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))*self.contrast)/6
        dvolume=[]
        for d in dt:
            term=np.einsum('ij,ij->i',d[:,0],np.cross(tri[:,1],tri[:,2]))
            term+=np.einsum('ij,ij->i',tri[:,0],np.cross(d[:,1],tri[:,2])+np.cross(tri[:,1],d[:,2]))
            dvolume.append(np.sum(term*self.contrast)/6)
        return v,dv,normal,dn,volume,np.asarray(dvolume)
    def boundaries(self,t):
        v,_,norm,_,_,_=self.mesh(t);norm=norm/self.contrast[:,None]
        length=np.linalg.norm(norm,axis=1);active=length>1e-15
        meshes=[]
        for contrast in [1.,.5]:
            sel=active&(self.contrast==contrast)
            meshes.append((SimpleNamespace(vertices=v,faces=self.faces[sel],face_normals=norm[sel]/length[sel,None]),contrast))
        return meshes

def forward(family,qaxis,n=384,quad=4):
    import cupy as cp
    from cupyx.scipy.signal import CZT
    fwd=DifferentiableForward(family,n=n,quad=quad,qsize=len(qaxis),qmax=float(qaxis[-1]))
    fwd.extent=192.;fwd.du=fwd.extent/n
    fwd.plan=CZT(n=n,m=len(qaxis),w=np.exp(-1j*(qaxis[1]-qaxis[0])*fwd.du),a=np.exp(1j*qaxis[0]*fwd.du))
    fwd.phase=cp.asarray(fwd.du*np.exp(-1j*qaxis*(-(n-1)*fwd.du/2)))
    fwd.cic=cp.asarray(np.sinc(fwd.qx*fwd.du/(2*np.pi))**2*np.sinc(fwd.qy*fwd.du/(2*np.pi))**2)
    return fwd

def metric(mu,y,mask):
    a=np.log1p(mu[mask]);b=np.log1p(y[mask]);r=a-b
    return {'log_mse':float(np.mean(r*r)),
      'logcorr':float(np.corrcoef(a,b)[0,1]),
      'adu_rel_l2':float(np.linalg.norm((mu-y)[mask])/np.linalg.norm(y[mask]))}

def penalty():
    L=np.zeros((4,len(NAMES)))
    L[np.arange(4),np.arange(7,11)]=np.repeat([1.,(6/4)**2],2)/.05
    return L

def optimize(fwd,y,mask,start,indices,lam=0.,max_nfev=65):
    start=np.clip(start,LOW+1e-8,HIGH-1e-8);idx=np.asarray(indices);L=penalty();cache={};history=[]
    norm=np.sqrt(mask.sum())
    def expand(z):
        t=start.copy();t[idx]=z;return t
    def evaluate(z):
        if 'z' not in cache or not np.array_equal(z,cache['z']):
            t=expand(z);mu,jac=fwd.model(t);r=(np.log1p(mu[mask])-np.log1p(y[mask]))/norm
            J=jac[:,mask].T/(1+mu[mask,None])/norm
            cache.update(z=z.copy(),r=np.r_[r,np.sqrt(lam)*(L@t)],J=np.vstack([J,np.sqrt(lam)*L])[:,idx])
            history.append(float(r@r))
        return cache
    tic=time.perf_counter()
    opt=least_squares(lambda z:evaluate(z)['r'],start[idx],jac=lambda z:evaluate(z)['J'],
      bounds=(LOW[idx],HIGH[idx]),x_scale='jac',max_nfev=max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    t=expand(opt.x)
    return {'theta':t,'success':bool(opt.success),'status':int(opt.status),'nfev':opt.nfev,
      'optimality':float(opt.optimality),'seconds':time.perf_counter()-tic,'cost':float(opt.cost),
      'bound_hits':int(np.sum(np.minimum(t-LOW,HIGH-t)[idx]<1e-4)),'history':history}

def context(stride=4):
    cfg,data,exp=id57_data();data.pop('clean_patterns',None)
    assert exp.shape==(209,401,401) and np.isfinite(exp).all()
    q=data['qaxis'];yy,xx=np.indices(data['roi'].shape);rad=np.hypot(xx-200,yy-200)
    band=np.floor((rad-45)/10).astype(int)
    roi=data['roi'];train=roi&(band%2==0);validation=roi&(band%4==1);test=roi&(band%4==3)
    masks={'roi':roi,'train':train,'validation':validation,'test':test}
    with np.load(PRIOR/'02_baseline/evaluation_masks.npz') as f:assert np.array_equal(train,f['train'])
    assert not np.any(train&validation|train&test|validation&test)
    assert np.array_equal(train|validation|test,roi)
    seeds={int(r['frame']):r for r in csv.DictReader((PRIOR/'03_rounding/id57_rounding_results.csv').open()) if r['mode']=='heldout'}
    return cfg,data,np.maximum(exp,0).astype(float),masks,seeds

def start_theta(seed):
    return np.r_[np.log(float(seed['scale'])),0.,0.,float(seed['radius_base_nm']),np.zeros(7),-18.,np.log(max(float(seed['background']),.1))]

def numerical_qa(fwd,t,masks,cfg):
    amp=fwd.amplitude(t);mask=masks['roi'];q=np.column_stack([fwd.qx[mask],fwd.qy[mask],np.zeros(mask.sum())])
    exact=MeshBoundaryReference(fwd.family.boundaries(t),q).intensity(np.eye(3))
    approximate=np.abs(amp[0][mask])**2
    errs={}
    for j,name in enumerate(NAMES[:-2]):
        h=1e-5 if j==3 else 1e-6;plus=t.copy();minus=t.copy();plus[j]+=h;minus[j]-=h
        fd=(fwd.amplitude(plus,False)[0]-fwd.amplitude(minus,False)[0])/(2*h)
        errs[name]=float(np.linalg.norm((amp[j+1]-fd)[mask])/max(np.linalg.norm(fd[mask]),1e-20))
    result={'faces':len(fwd.family.faces),'grid':fwd.n,'quad_nodes':fwd.ns,
      'forward_relative_l2':float(np.linalg.norm(approximate-exact)/np.linalg.norm(exact)),
      'gradient_relative_l2':errs,'forward_gate':.003,'gradient_gate':.003}
    import json
    history=json.loads((OUT/'numerical_qa_history.json').read_text()) if (OUT/'numerical_qa_history.json').exists() else []
    history.append(result);dump(OUT/'numerical_qa_history.json',history)
    dump(OUT/'numerical_qa.json',result);print('QA',result,flush=True)
    assert result['forward_relative_l2']<.003 and max(errs.values())<.003
    return result

def run(frames,grid=384,stride=4,calibration=False,max_nfev=65):
    cfg,data,images,fullmask,seeds=context(stride)
    masks={k:v[::stride,::stride] for k,v in fullmask.items()}
    qaxis=data['qaxis'][::stride];fam=CoreShellFamily();fwd=forward(fam,qaxis,grid)
    np.savez_compressed(OUT/'detector_masks.npz',qaxis=data['qaxis'],stride=stride,**fullmask)
    config={'dataset':'CXIDB ID57 measured normal-incidence-pattern.cxi','cohort':209,
      'units':'nm, nm^-1, detector ADU','objective':'mean squared log1p ADU residual + lambda ||L theta||^2',
      'seed':'legacy rounded core-shell, selected only on even 10-pixel radial bands',
      'split':'even bands train; odd bands 1 mod 4 validation, 3 mod 4 test; q 0.267 to 1.128 nm^-1',
      'sampling_stride':stride,'q_axis_points':len(qaxis),'grid':grid,'parameter_names':NAMES,
      'density':'outer Pd contrast 1; clipped Au octahedral core excess contrast 0.5; core size ratio fixed',
      'geometry':'fixed topology rounded cube; global scale/aspects/local 3D rotation; four smooth angular deformation modes applied to both boundaries',
      'penalty':'Tikhonov angular curvature on cos/sin k=4,6; 5% reference amplitude; bulk/nuisance unpenalized',
      'noise':'experimental ADU; negatives clipped as in earlier comparison; no calibrated photon-noise claim',
      'lambda_grid':LAMBDAS,'calibration_frames':list(range(0,209,9)),
      'selection':'one common lambda minimizing average validation log MSE on evenly spaced calibration frames; lambda zero included',
      'evaluation':'full-resolution validation/test with fixed training nuisance parameters; no refitting on held-out pixels',
      'bounds':{'lower':LOW,'upper':HIGH},'scope':'conditional local mesh refinement, planar forward model; no unique 3D reconstruction or measured rounding radius claim'}
    dump(OUT/'experiment_config.json',config)
    qa=__import__('json').loads((OUT/'numerical_qa.json').read_text()) if (OUT/'numerical_qa.json').exists() else {}
    if not qa or qa['forward_relative_l2']>=.003 or max(qa['gradient_relative_l2'].values())>=.003 or qa['faces']!=len(fam.faces) or qa['grid']!=grid:
        t=start_theta(seeds[0]);t[7:11]=[.01,-.01,.008,.005];numerical_qa(fwd,t,masks,cfg)
    fullfwd=forward(fam,data['qaxis'],grid)
    if calibration:lambdas=LAMBDAS
    else:
        selection=__import__('json').loads((OUT/'lambda_selection.json').read_text());lambdas=sorted(set([0.,selection['lambda']]))
    for frame in frames:
        path=OUT/f'fits/frame_{frame:03d}.json'
        if path.exists():
            existing=__import__('json').loads(path.read_text());have=existing['fits'].keys()
            if all(str(lam) in have for lam in lambdas):continue
        else:existing=None
        fam.rotation=data['rotation_matrices'][int(seeds[frame]['orientation_index'])].astype(float)
        y=images[frame,::stride,::stride];start=start_theta(seeds[frame]);raw=np.abs(fwd.amplitude(start,False)[0])**2
        normalizer=float(raw[masks['train']].max());nuisance,_=nnls(np.c_[raw[masks['train']]/normalizer,np.ones(masks['train'].sum())],y[masks['train']])
        start[-2:]=np.log([max(nuisance[0]/normalizer,1e-17),max(nuisance[1],.001)])
        if existing:
            coarse=existing['coarse'];bulk=existing['bulk'];fits=existing['fits']
            bulk['theta']=np.asarray(bulk['theta'])
        else:
            coarse=optimize(fwd,y,masks['train'],start,[11,12],0,max_nfev)
            bulk=optimize(fwd,y,masks['train'],np.asarray(coarse['theta']),list(range(7))+[11,12],0,max_nfev)
            fits={}
            for record in [coarse,bulk]:
                pred=full_predict(fullfwd,np.asarray(record['theta']))
                record['metrics']={k:metric(pred,images[frame],v) for k,v in fullmask.items() if k!='roi'}
        for lam in lambdas:
            if str(lam) in fits:continue
            fit=optimize(fwd,y,masks['train'],np.asarray(bulk['theta']),list(range(13)),lam,max_nfev)
            pred=full_predict(fullfwd,fit['theta']);fit['metrics']={k:metric(pred,images[frame],v) for k,v in fullmask.items() if k!='roi'}
            fits[str(lam)]=fit
        dump(path,{'frame':frame,'seed':seeds[frame],'rotation':fam.rotation,'coarse':coarse,'bulk':bulk,'fits':fits})
        print('FIT',frame,[(k,round(v['metrics']['test']['log_mse'],5),v['success'],v['nfev']) for k,v in fits.items()],flush=True)

def full_predict(fwd,t):
    return np.exp(t[-2])*np.abs(fwd.amplitude(t,False)[0])**2+np.exp(t[-1])

def select():
    import json
    frames=list(range(0,209,9));records=[json.loads((OUT/f'fits/frame_{i:03d}.json').read_text()) for i in frames]
    rows=[{'lambda':lam,'validation_log_mse_mean':np.mean([r['fits'][str(lam)]['metrics']['validation']['log_mse'] for r in records]),
       'validation_logcorr_mean':np.mean([r['fits'][str(lam)]['metrics']['validation']['logcorr'] for r in records])} for lam in LAMBDAS]
    selected=min(rows,key=lambda r:r['validation_log_mse_mean'])
    dump(OUT/'lambda_selection.json',{'lambda':selected['lambda'],'calibration_frames':frames,'candidates':rows,'test_pixels_used':False})
    print('SELECT',selected,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['qa','pilot','calibrate','select','all'],default='pilot');ap.add_argument('--grid',type=int,default=384);ap.add_argument('--max-nfev',type=int,default=65)
    args=ap.parse_args()
    if args.stage=='select':select()
    elif args.stage=='qa':
        cfg,data,images,masks,seeds=context();fam=CoreShellFamily(data['rotation_matrices'][int(seeds[0]['orientation_index'])]);fwd=forward(fam,data['qaxis'][::4],args.grid)
        t=start_theta(seeds[0]);t[7:11]=[.01,-.01,.008,.005];numerical_qa(fwd,t,{k:v[::4,::4] for k,v in masks.items()},cfg)
    else:run([0,9,18] if args.stage=='pilot' else list(range(0,209,9)) if args.stage=='calibrate' else list(range(209)),args.grid,calibration=args.stage in ['pilot','calibrate'],max_nfev=args.max_nfev)
