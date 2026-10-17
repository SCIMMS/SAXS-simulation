"""Fixed-topology differentiable mesh + projected-gradient CZT synthetic study.

Production simulator and earlier evidence are never edited. Lengths are nm.
An independent exact triangle boundary integral generates observations.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, time
from pathlib import Path
from types import SimpleNamespace
from revision_common import ROOT, dump, save_csv, sha, sm, np
import cupy as cp
from cupyx.scipy.signal import CZT
from scipy.optimize import least_squares
from scipy.optimize import nnls
from revision_gpu_boundary_general import MeshBoundaryReference

OUT = ROOT / 'output/differentiable_synthetic_2026-10-11'
OUT.mkdir(parents=True, exist_ok=True)
OLD = ROOT / 'deliverables/projected_gradient_mesh_saxs_explainer_2026-09-02/examples/minimal_projected_gradient_optimizer.py'
spec = importlib.util.spec_from_file_location('previous_mesh_reference', OLD)
old = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = old
spec.loader.exec_module(old)
MODES = [(k, kind) for k in (4,6,8,10) for kind in ('cos','sin')]
NAMES = ['log_axis_u','log_axis_v','log_exponent','roll_rad'] + [f'{kind}{k}' for k,kind in MODES] + ['log_signal_scale','log_background']

SPLAT = r'''
extern "C" __global__ void splat(const double* v,const int* f,const double* normal,
 const double* dv,const double* dn,const double* nodes,int nv,int nf,int ns,int np,
 int n,double du,double u0,double* grids){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nf*ns*(np+1))return;
 int p=t/(nf*ns),rem=t%(nf*ns),face=rem/ns,s=rem%ns;
 int i=f[3*face],j=f[3*face+1],k=f[3*face+2];
 double b0=nodes[4*s],b1=nodes[4*s+1],b2=nodes[4*s+2],w=nodes[4*s+3]/(du*du);
 double x=(b0*v[3*i]+b1*v[3*j]+b2*v[3*k]-u0)/du;
 double y=(b0*v[3*i+1]+b1*v[3*j+1]+b2*v[3*k+1]-u0)/du;
 int ix=(int)floor(x),iy=(int)floor(y);double tx=x-ix,ty=y-iy;
 double dx=0.,dy=0.;
 if(p){const double* d=dv+(p-1)*nv*3;
 dx=(b0*d[3*i]+b1*d[3*j]+b2*d[3*k])/du;
 dy=(b0*d[3*i+1]+b1*d[3*j+1]+b2*d[3*k+1])/du;}
 for(int oy=0;oy<2;oy++)for(int ox=0;ox<2;ox++){
  int xx=ix+ox,yy=iy+oy;if(xx<0||xx>=n||yy<0||yy>=n)continue;
  double wx=ox?tx:1.-tx,wy=oy?ty:1.-ty,cic=wx*wy;
  double dc=(ox?dx:-dx)*wy+wx*(oy?dy:-dy);
  for(int c=0;c<2;c++){
   double val=normal[3*face+c]*w*cic;
   if(p)val=w*(dn[((p-1)*nf+face)*3+c]*cic+normal[3*face+c]*dc);
   atomicAdd(grids+((p*2+c)*n+yy)*n+xx,val);
  }
 }
}
'''

class MeshFamily:
    def __init__(self, nlat=20, nlon=64, axis_w=23.):
        self.base, self.faces = old.unit_uv_sphere(nlat,nlon)
        self.axis_w = axis_w
        phi=np.arctan2(self.base[:,1],self.base[:,0])
        rho2=self.base[:,0]**2+self.base[:,1]**2
        self.basis=np.column_stack([rho2*(np.cos(k*phi) if kind=='cos' else np.sin(k*phi)) for k,kind in MODES])

    def mesh(self, theta):
        a,b,p=np.exp(theta[:3]);roll=theta[3]
        logs=np.log(np.maximum(np.abs(self.base),1e-300))
        powers=np.abs(self.base)**p
        total=powers.sum(axis=1)
        logr=-np.log(total)/p
        r=np.exp(logr)
        dlogr=np.log(total)/p-(powers*logs).sum(axis=1)/total
        deform=1+self.basis@theta[4:4+len(MODES)]
        axes=np.array([a,b,self.axis_w])
        v=self.base*(r*deform)[:,None]*axes
        fields=np.zeros((4+len(MODES),len(v),3))
        fields[0,:,0]=v[:,0];fields[1,:,1]=v[:,1]
        fields[2]=v*dlogr[:,None]
        for j in range(len(MODES)):
            fields[4+j]=self.base*(r*self.basis[:,j])[:,None]*axes
        rot=np.array([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]])
        v=v@rot.T;fields=fields@rot.T
        fields[3,:,0]=-v[:,1];fields[3,:,1]=v[:,0]
        tri=v[self.faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
        normal=np.cross(e1,e2)
        dt=fields[:,self.faces]
        dn=np.cross(dt[:,:,1]-dt[:,:,0],e2)+np.cross(e1,dt[:,:,2]-dt[:,:,0])
        volume=old.signed_volume(v,self.faces)
        dvolume=np.array([old.volume_derivative(v,self.faces,d) for d in fields])
        return v,fields,normal,dn,volume,dvolume

class DifferentiableForward:
    def __init__(self,family,n=384,quad=4,qsize=65,qmax=.70):
        self.family=family;self.n=n;self.extent=140.;self.du=self.extent/n
        self.qaxis=np.linspace(-qmax,qmax,qsize)
        self.qx,self.qy=np.meshgrid(self.qaxis,self.qaxis)
        self.q2=self.qx**2+self.qy**2
        dq=self.qaxis[1]-self.qaxis[0]
        self.plan=CZT(n=n,m=qsize,w=np.exp(-1j*dq*self.du),a=np.exp(1j*self.qaxis[0]*self.du))
        u0=-(n-1)*self.du/2
        self.phase=cp.asarray(self.du*np.exp(-1j*self.qaxis*u0))
        self.qx_gpu=cp.asarray(self.qx);self.qy_gpu=cp.asarray(self.qy)
        self.den=cp.asarray(np.where(self.q2>0,-1j*self.q2,1))
        # CIC is convolution with a triangular basis; remove its known transfer.
        self.cic=cp.asarray(np.sinc(self.qx*self.du/(2*np.pi))**2*np.sinc(self.qy*self.du/(2*np.pi))**2)
        s,t,w=old.duffy_nodes(quad)
        self.nodes=cp.asarray(np.column_stack([(1-s)*(1-t),s,(1-s)*t,w]).ravel())
        self.ns=len(w);self.faces=cp.asarray(family.faces.astype(np.int32))
        self.kernel=cp.RawKernel(SPLAT,'splat',options=('--std=c++11',))
        self.calls=0

    def amplitude(self,theta,derivatives=True):
        v,dv,normal,dn,vol,dvol=self.family.mesh(theta)
        if not derivatives:dv=dv[:0];dn=dn[:0];dvol=dvol[:0]
        npg=len(dv);nf=len(normal)
        grids=cp.zeros((npg+1,2,self.n,self.n),dtype=cp.float64)
        arrays=[cp.asarray(a,dtype=cp.float64) for a in (v,normal,dv,dn)]
        total=nf*self.ns*(npg+1)
        self.kernel(((total+255)//256,),(256,),(arrays[0],self.faces,arrays[1],arrays[2],arrays[3],self.nodes,
          np.int32(len(v)),np.int32(nf),np.int32(self.ns),np.int32(npg),np.int32(self.n),
          np.float64(self.du),np.float64(-(self.n-1)*self.du/2),grids))
        ft=self.plan(grids.astype(cp.complex128),axis=-1)*self.phase[None,None,None,:]
        ft=self.plan(ft,axis=-2)*self.phase[None,None,:,None]
        amp=(self.qx_gpu*ft[:,0]+self.qy_gpu*ft[:,1])/self.den/self.cic
        zero=self.q2==0
        amp[:,cp.asarray(zero)]=cp.asarray(np.r_[vol,dvol])[:,None]
        self.calls+=1
        return cp.asnumpy(amp)

    def model(self,theta):
        amp=self.amplitude(theta)
        scale,bg=np.exp(theta[-2:])
        signal=scale*np.abs(amp[0])**2
        mu=signal+bg
        derivatives=2*scale*np.real(np.conj(amp[0])[None]*amp[1:])
        jac=np.concatenate([derivatives,signal[None],np.full_like(signal[None],bg)],axis=0)
        return mu,jac

    def exact_intensity(self,theta):
        v,_,normal,_,_,_=self.family.mesh(theta)
        mesh=SimpleNamespace(vertices=v,faces=self.family.faces,face_normals=normal/np.linalg.norm(normal,axis=1)[:,None])
        q=np.column_stack([self.qx.ravel(),self.qy.ravel(),np.zeros(self.qx.size)])
        ref=MeshBoundaryReference([(mesh,1.)],q)
        return ref.intensity(np.eye(3)).reshape(self.qx.shape)

def poisson_residual(mu,y):
    mu=np.maximum(mu,1e-12);y=np.asarray(y)
    d=mu.copy()
    positive=y>0
    z=(y[positive]-mu[positive])/mu[positive]
    term=(1+z)*np.log1p(z)-z
    small=np.abs(z)<1e-3
    zs=z[small]
    term[small]=zs**2/2-zs**3/6+zs**4/12-zs**5/20+zs**6/30
    d[positive]=mu[positive]*term
    d=np.maximum(2*d,0)
    r=np.sign(mu-y)*np.sqrt(d)
    derivative=np.empty_like(mu)
    near=np.abs(mu-y)<1e-5*np.maximum(mu,1.)
    derivative[near]=1/np.sqrt(mu[near])
    derivative[~near]=np.abs(1-y[~near]/mu[~near])/np.maximum(np.sqrt(d[~near]),1e-20)
    return r,derivative

def masks(forward):
    r=np.sqrt(forward.q2)
    valid=(r>=.16)&(r<=.66)&(np.abs(forward.qx)>.018)&(np.abs(forward.qy)>.018)
    # Detector gap; no intensity/score-based masking.
    valid &= ~((forward.qx>.32)&(forward.qy<-.12)&(forward.qy>-.22))
    band=np.floor((r-.16)/.055).astype(int)%5
    train=valid&np.isin(band,[0,1,3]);validation=valid&(band==2);test=valid&(band==4)
    return {'valid':valid,'train':train,'validation':validation,'test':test}

def regularizer():
    # Tikhonov angular-curvature penalty, expressed in units of 5% deformation.
    # Bulk size, exponent, rotation, scale and background are unpenalized.
    diag=np.zeros(len(NAMES))
    for j,(k,_) in enumerate(MODES):diag[4+j]=(k/4)**2/.05
    return np.diag(diag)[4:4+len(MODES)]

def fit(forward,y,mask,starts,lam=0.,max_nfev=80):
    lower=np.r_[np.log([17.,17.,1.7]),-.5,np.full(len(MODES),-.10),-35.,np.log(.01)]
    upper=np.r_[np.log([34.,34.,16.]),.65,np.full(len(MODES),.10),5.,np.log(5.)]
    L=regularizer();records=[]
    for start_no,start in enumerate(starts):
        start=start.copy()
        signal=forward.model(start)[0]-np.exp(start[-1])
        normalizer=np.max(signal[mask])
        nuisance,_=nnls(np.column_stack([signal[mask]/normalizer,np.ones(mask.sum())]),y[mask])
        start[-2]=np.log(max(nuisance[0]/normalizer,1e-15))
        start[-1]=np.log(np.clip(nuisance[1],.011,4.9))
        cache={};history=[]
        def evaluate(theta):
            if 'theta' not in cache or not np.array_equal(theta,cache['theta']):
                mu,jac=forward.model(theta)
                r,dr=poisson_residual(mu[mask],y[mask])
                J=jac[:,mask].T*dr[:,None]
                rr=np.r_[r,np.sqrt(lam)*(L@theta)]
                JJ=np.vstack([J,np.sqrt(lam)*L])
                cache.update(theta=theta.copy(),r=rr,j=JJ)
                history.append(float(np.mean(r*r)))
            return cache
        t=time.perf_counter()
        # First fit the six bulk/nuisance variables with shape modes fixed to 0.
        bulk_indices=np.r_[np.arange(4),len(NAMES)-2,len(NAMES)-1]
        def expand(z):
            full=start.copy();full[bulk_indices]=z;return full
        bulk=least_squares(lambda z:evaluate(expand(z))['r'],start[bulk_indices],
          jac=lambda z:evaluate(expand(z))['j'][:,bulk_indices],
          bounds=(lower[bulk_indices],upper[bulk_indices]),method='trf',x_scale='jac',
          max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-6)
        bulk_theta=expand(bulk.x)
        result=least_squares(lambda x:evaluate(x)['r'],bulk_theta,jac=lambda x:evaluate(x)['j'],
          bounds=(lower,upper),method='trf',x_scale='jac',max_nfev=max_nfev,
          ftol=1e-7,xtol=1e-7,gtol=1e-6)
        records.append({'theta':result.x,'cost':float(result.cost),'success':bool(result.success),
          'status':int(result.status),'nfev':result.nfev,'njev':result.njev,'optimality':result.optimality,
          'seconds':time.perf_counter()-t,'start':start_no,'history':history,
          'initial_theta':start,'bulk_theta':bulk_theta,'bulk_nfev':bulk.nfev,
          'bound_hits':int(np.sum((result.x-lower<1e-4)|(upper-result.x<1e-4)))})
    best=min(records,key=lambda x:x['cost'])
    best['all_starts']=[{k:v for k,v in r.items() if k not in ('history','theta')} for r in records]
    return best

def shape_error(family,theta,target):
    v=family.mesh(theta)[0];truth=family.mesh(target)[0]
    return float(np.sqrt(np.mean(np.sum((v-truth)**2,axis=1)))/np.sqrt(np.mean(np.sum(truth**2,axis=1))))

def targets():
    rows=[]
    for name,a,b,p,roll,c4,s6 in [('ellipsoid',27.,21.,2.,.19,0.,0.),
        ('rounded_box',26.,24.,4.,.24,0.,0.),('deformed_rounded_box',28.,22.,5.,.16,.045,.025)]:
        theta=np.r_[np.log([a,b,p]),roll,np.zeros(len(MODES)),0.,np.log(.4)]
        theta[4]=c4;theta[7]=s6
        rows.append((name,theta))
    return rows

def prepare_coarse_library(forward):
    candidates=[];signals=[]
    import itertools
    for a,b,p,roll in itertools.product([19.,22.,25.,28.,31.],[19.,22.,25.,28.,31.],[2.,4.,8.,12.],[-.1,.1,.3,.5]):
        theta=np.r_[np.log([a,b,p]),roll,np.zeros(len(MODES)),0.,np.log(.4)]
        candidates.append(theta)
        signals.append(np.abs(forward.amplitude(theta,derivatives=False)[0])**2)
    forward.coarse_candidates=candidates;forward.coarse_signals=np.array(signals)
    print(f'Coarse library ready: {len(candidates)} candidates',flush=True)

def initializations(forward,observed,mask):
    # One training-selected initial candidate per exponent family. Every candidate
    # receives a nonnegative scale/background fit before ranking.
    winners={}
    for theta,signal in zip(forward.coarse_candidates,forward.coarse_signals):
        normalizer=np.max(signal[mask])
        nuisance,_=nnls(np.column_stack([signal[mask]/normalizer,np.ones(mask.sum())]),observed[mask])
        scale=max(nuisance[0]/normalizer,1e-15);background=np.clip(nuisance[1],.011,4.9)
        residual,_=poisson_residual(scale*signal[mask]+background,observed[mask])
        cost=float(np.sum(residual**2));p=round(float(np.exp(theta[2])))
        if p not in winners or cost<winners[p][0]:
            start=theta.copy();start[-2:]=np.log([scale,background]);winners[p]=(cost,start)
    return [winners[p][1] for p in (12,8,4,2)]

def qa(forward,mask):
    records=[]
    for name,theta in targets():
        analytic=forward.amplitude(theta)
        exact=forward.exact_intensity(theta)
        approximate=np.abs(analytic[0])**2
        error=np.linalg.norm((approximate-exact)[mask])/np.linalg.norm(exact[mask])
        fd_errors=[]
        for j in range(len(theta)-2):
            h=1e-6;plus=theta.copy();minus=theta.copy();plus[j]+=h;minus[j]-=h
            fd=(forward.amplitude(plus)[0]-forward.amplitude(minus)[0])/(2*h)
            fd_errors.append(np.linalg.norm((analytic[j+1]-fd)[mask])/max(np.linalg.norm(fd[mask]),1e-20))
        records.append({'model':name,'exact_intensity_relative_l2':float(error),
          'gradient_relative_l2':dict(zip(NAMES[:-2],map(float,fd_errors)))})
    # Independent numerical check of Poisson residual derivative (including zeros).
    mu=np.array([.1,2.,7.,3.]);y=np.array([0.,1.,9.,3.])
    _,d=poisson_residual(mu,y)
    finite=(poisson_residual(mu+1e-6,y)[0]-poisson_residual(mu-1e-6,y)[0])/2e-6
    residual_error=float(np.max(np.abs(d-finite)))
    result={'models':records,'poisson_residual_derivative_max_abs':residual_error,
      'grid_n':forward.n,'faces':len(forward.family.faces),'gradient_gate':.001,'forward_gate':.001}
    dump(OUT/'numerical_qa.json',result)
    return result

def run(forward,seeds):
    mm=masks(forward);np.savez_compressed(OUT/'detector_masks.npz',qaxis=forward.qaxis,**mm)
    config={'date':'2026-10-11 Asia/Seoul','parameter_names':NAMES,'modes':MODES,
      'grid_n':forward.n,'mesh_faces':len(forward.family.faces),'q_min':.16,'q_max':.66,
      'axis_w_nm_fixed':forward.family.axis_w,'signal_photons':20000.,'background_per_pixel':.4,
      'noise':'Poisson independent counts; a synthetic surrogate, not measured CXIDB ADU noise',
      'seed_ids':list(range(seeds)),'calibration_seed':91000,'lambda_grid':[0.,.01,.1,1.,10.,100.,1000.],
      'objective':'sum Poisson deviance + lambda ||L theta||^2',
      'penalty':'Tikhonov on cos/sin shape modes only; L=(k/4)^2 / 0.05',
      'selection':'positive lambda on a separate calibration noise draw, validation bands only; lambda=0 retained as a diagnostic baseline; no shape truth or test bands used',
      'coarse_grid':{'axes_nm':[19,22,25,28,31],'exponents':[2,4,8,12],'roll_rad':[-.1,.1,.3,.5],'candidates':400},
      'initial_exponents':[12.,8.,4.,2.],'initial_nuisance':'nonnegative signal/background fitted to training pixels only',
      'initialization':'one training-selected coarse candidate per exponent; six bulk/nuisance variables first; all shape modes next; best penalized training objective across four starts',
      'reference':'independent exact triangle boundary integral, same finite mesh family',
      'scope':'conditional shape fitting with fixed depth, topology, density contrast and planar geometry'}
    dump(OUT/'experiment_config.json',config)
    qa_result=qa(forward,mm['valid'])
    if max(r['exact_intensity_relative_l2'] for r in qa_result['models'])>.001 or max(v for r in qa_result['models'] for v in r['gradient_relative_l2'].values())>.001:
        raise RuntimeError('Numerical accuracy gate failed; inspect numerical_qa.json')
    prepare_coarse_library(forward)
    all_rows=[];calibration=[]
    for model_no,(name,truth) in enumerate(targets()):
        raw=forward.exact_intensity(truth)
        scale=20000./raw[mm['valid']].sum();truth[-2]=np.log(scale)
        expected=raw*scale+np.exp(truth[-1])
        noise_calibration=np.random.default_rng(91000+model_no).poisson(expected).astype(float)
        starts=initializations(forward,noise_calibration,mm['train'])
        candidates=[]
        for lam in config['lambda_grid']:
            fitted=fit(forward,noise_calibration,mm['train'],starts,lam)
            mu,_=forward.model(fitted['theta'])
            vr,_=poisson_residual(mu[mm['validation']],noise_calibration[mm['validation']])
            row={'model':name,'lambda':lam,'validation_deviance_mean':float(np.mean(vr**2)),
              'theta':fitted['theta'].tolist(),'success':fitted['success'],'bound_hits':fitted['bound_hits']}
            candidates.append(row);calibration.append(row)
            dump(OUT/'lambda_calibration.json',calibration)
            print(f'CAL {name} lambda={lam:g} validation={row["validation_deviance_mean"]:.5f}',flush=True)
        selected=min((r for r in candidates if r['lambda']>0),key=lambda r:r['validation_deviance_mean'])['lambda']
        for seed in [-1]+list(range(seeds)):
            observed=expected.copy() if seed==-1 else np.random.default_rng(11000+model_no*100+seed).poisson(expected).astype(float)
            starts=initializations(forward,observed,mm['train'])
            conditions=[('zero_noise',0.)] if seed==-1 else [('noise',0.),('noise_tikhonov',selected)]
            for condition,lam in conditions:
                fitted=fit(forward,observed,mm['train'],starts,lam)
                theta=fitted['theta'];mu,_=forward.model(theta)
                exact_fitted=forward.exact_intensity(theta)*np.exp(theta[-2])+np.exp(theta[-1])
                test_r,_=poisson_residual(mu[mm['test']],observed[mm['test']])
                clean_error=np.linalg.norm((exact_fitted-expected)[mm['test']])/np.linalg.norm(expected[mm['test']])
                row={'model':name,'seed':seed,'condition':condition,'lambda':lam,
                  'surface_relative_rmse':shape_error(forward.family,theta,truth),
                  'clean_test_intensity_relative_l2':float(clean_error),'test_deviance_mean':float(np.mean(test_r**2)),
                  'axis_u_nm':float(np.exp(theta[0])),'axis_v_nm':float(np.exp(theta[1])),
                  'exponent':float(np.exp(theta[2])),'roll_deg':float(np.rad2deg(theta[3])),
                  'mode_error_l2':float(np.linalg.norm(theta[4:-2]-truth[4:-2])),
                  'success':fitted['success'],'bound_hits':fitted['bound_hits'],'nfev':fitted['nfev'],
                  'seconds':sum(r['seconds'] for r in fitted['all_starts']),
                  'selected_start':fitted['start']}
                all_rows.append(row);save_csv(OUT/'fit_metrics.csv',all_rows)
                dump(OUT/f'{name}_{condition}_seed{seed}.json',{'metrics':row,'fit':fitted,'truth':truth})
                np.savez_compressed(OUT/f'{name}_{condition}_seed{seed}.npz',truth=truth,fitted=theta,
                  expected=expected,observed=observed,predicted=mu,exact_predicted=exact_fitted,
                  initial=fitted['initial_theta'],bulk=fitted['bulk_theta'])
                print(f'FIT {name} {condition} seed={seed} shape={row["surface_relative_rmse"]:.4%} test={clean_error:.4%}',flush=True)
    summary=[]
    for name,_ in targets():
        for condition in ('zero_noise','noise','noise_tikhonov'):
            rows=[r for r in all_rows if r['model']==name and r['condition']==condition]
            summary.append({'model':name,'condition':condition,'n':len(rows),'lambda':rows[0]['lambda'],
              **{f'{k}_median':float(np.median([r[k] for r in rows])) for k in ('surface_relative_rmse','clean_test_intensity_relative_l2','test_deviance_mean','mode_error_l2')},
              'successes':sum(r['success'] for r in rows),'bound_hits_total':sum(r['bound_hits'] for r in rows)})
    save_csv(OUT/'summary.csv',summary)
    dump(OUT/'summary.json',{'groups':summary,'total_fits':len(all_rows),'config':config,'qa':qa_result,
      'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(__file__),str(OLD.relative_to(ROOT)):sha(OLD)}})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--qa-only',action='store_true');ap.add_argument('--grid',type=int,default=384);ap.add_argument('--seeds',type=int,default=3)
    args=ap.parse_args();fwd=DifferentiableForward(MeshFamily(),n=args.grid)
    if args.qa_only:print(json.dumps(qa(fwd,masks(fwd)['valid']),indent=2))
    else:run(fwd,args.seeds)
