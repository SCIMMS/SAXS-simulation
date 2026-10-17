"""Cached 3D volume versus mesh projection, identical detector and accuracy reference.
Run each variant in a separate process after other GPU experiments finish.
"""
from revision_common import *
from revision_gpu_fourier import GPUCZT2
from revision_gpu_surface_reference import CoreShellReference
import cupy as cp
from cupyx.scipy.signal import CZT
from cupyx.scipy.ndimage import map_coordinates,spline_filter
import argparse,time,psutil,gc,platform
p=argparse.ArgumentParser();p.add_argument('--method',choices=['mesh_cpu_czt','mesh_gpu_czt','fft','czt'],required=True)
p.add_argument('--n',type=int,default=192);p.add_argument('--m',type=int,default=257);p.add_argument('--pad',type=int,default=4);p.add_argument('--accuracy-only',action='store_true');p.add_argument('--order',type=int,default=1);p.add_argument('--qa-count',type=int,default=3);p.add_argument('--slab',action='store_true');a=p.parse_args()
stage=OUT/'04_benchmark';stage.mkdir(exist_ok=True)
process=psutil.Process();rss0=process.memory_info().rss
cfg,data,exp=id57_data();data.pop('clean_patterns',None);del exp
qaxis=data['qaxis'];roi=data['roi'];qx,qy=np.meshgrid(qaxis,qaxis)
q=np.column_stack([qx[roi],qy[roi],np.zeros(roi.sum())]);extent=144.
meshes=id57_meshes();reference=CoreShellReference(meshes[1],q)
rotations=data['rotation_matrices'].astype(float)
# Warm context and kernel independently from measured algorithm setup.
qa_indices=[0,144,238] if a.qa_count==3 else list(range(a.qa_count))
ref=[reference.intensity(rotations[i]) for i in qa_indices]
cp.cuda.Device().synchronize();cp.get_default_memory_pool().free_all_blocks()
pool_observed=[]
setup_start=time.perf_counter()
def voxelize(n):
    du=extent/n;axis=cp.asarray((np.arange(n)-(n-1)/2)*du,dtype=cp.float32)
    lo=axis-du/2;hi=axis+du/2
    fraction=cp.clip(cp.minimum(hi,26)-cp.maximum(lo,-26),0,du)/du
    rho=fraction[:,None,None]*fraction[None,:,None]*fraction[None,None,:]
    core=cp.zeros_like(rho)
    for dx in [-1,1]:
      x=axis+dx*du/(2*np.sqrt(3))
      for dy in [-1,1]:
       y=axis+dy*du/(2*np.sqrt(3))
       for dz in [-1,1]:
        z=axis+dz*du/(2*np.sqrt(3))
        core+=((cp.abs(x)[:,None,None]+cp.abs(y)[None,:,None]+cp.abs(z)[None,None,:]<=26.2)&
               (cp.abs(x)[:,None,None]<=26)&(cp.abs(y)[None,:,None]<=26)&(cp.abs(z)[None,None,:]<=26)).astype(cp.float32)/8
    return rho+.5*core,du
cache=None;gpu_plan=None;axis3=None
if a.method.startswith('mesh'):
    if a.method=='mesh_gpu_czt':gpu_plan=GPUCZT2(a.n,extent,qaxis)
    # Include plan construction; lazy raster/FFT initialization warmed below and reported separately.
else:
    rho,du=voxelize(a.n)
    mass=float(cp.sum(rho,dtype=cp.float64).get())*du**3
    expected=52**3+.5*sm.signed_mesh_volume(meshes[1].vertices,meshes[1].faces)
    if a.method=='fft':
        size=a.n*a.pad;left=(size-a.n)//2
        padded=cp.pad(rho,((left,size-a.n-left),)*3)
        cache=cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(padded))).astype(cp.complex64)*du**3
        axis3=cp.asnumpy(cp.fft.fftshift(cp.fft.fftfreq(size,d=du)))*2*np.pi
        ph=cp.exp(-1j*cp.asarray(axis3)*du/2).astype(cp.complex64)
        cache*=ph[:,None,None]*ph[None,:,None]*ph[None,None,:]
        del padded,ph
    else:
        axis3=np.linspace(-1.35,1.35,a.m);dq=axis3[1]-axis3[0]
        plan=CZT(n=a.n,m=a.m,w=np.exp(-1j*dq*du),a=np.exp(1j*axis3[0]*du))
        phase=cp.asarray(du*np.exp(-1j*axis3*(-(a.n-1)*du/2)),dtype=cp.complex128)
        if a.slab:
            # Bounded batch workspace with complex64 intermediate storage.
            first=cp.empty((a.n,a.n,a.m),dtype=cp.complex64)
            for j in range(0,a.n,16):first[j:j+16]=plan(rho[j:j+16],axis=2)*phase[None,None,:]
            second=cp.empty((a.n,a.m,a.m),dtype=cp.complex64)
            for j in range(0,a.n,16):second[j:j+16]=plan(first[j:j+16],axis=1)*phase[None,:,None]
            del first
            cache=cp.empty((a.m,a.m,a.m),dtype=cp.complex64)
            for j in range(0,a.m,16):cache[:,j:j+16]=plan(second[:,j:j+16],axis=0)*phase[:,None,None]
            del second
        else:
            cache=plan(cp.asarray(rho,dtype=cp.complex128),axis=2)*phase[None,None,:]
            cache=plan(cache,axis=1)*phase[None,:,None]
            cache=plan(cache,axis=0)*phase[:,None,None]
            cache=cache.astype(cp.complex64)
        del plan,phase
    del rho
    if a.order==3:
        pool_observed.append(cp.get_default_memory_pool().total_bytes())
        cp.get_default_memory_pool().free_all_blocks()
        coeff_re=spline_filter(cache.real,order=3,output=cp.float32,mode='constant')
        coeff_im=spline_filter(cache.imag,order=3,output=cp.float32,mode='constant')
        cache=coeff_re.astype(cp.complex64)+1j*coeff_im
        del coeff_re,coeff_im
cp.cuda.Device().synchronize();setup_s=time.perf_counter()-setup_start
setup_pool=max([cp.get_default_memory_pool().total_bytes(),*pool_observed]);rss_peak=process.memory_info().rss
cp.get_default_memory_pool().free_all_blocks()
cache_bytes=0 if cache is None else cache.nbytes
qgpu=cp.asarray(q)
def render(r):
    if a.method.startswith('mesh'):
        projection,_=id57_projection(r,a.n,extent,meshes,'gpu')
        full=czt_intensity(projection,extent,qaxis) if gpu_plan is None else gpu_plan.intensity(projection)
        return full[roi]
    rotated=qgpu@cp.asarray(r)
    coordinates=((rotated-axis3[0])/(axis3[1]-axis3[0])).T
    re=map_coordinates(cache.real,coordinates,order=a.order,mode='constant',cval=0,prefilter=False)
    im=map_coordinates(cache.imag,coordinates,order=a.order,mode='constant',cval=0,prefilter=False)
    return cp.asnumpy(re*re+im*im)
errors=[];warm_start=time.perf_counter()
for i,v in zip(qa_indices,ref):
    result=render(rotations[i]);errors.append({'orientation':i,'intensity_relative_l2':rel_l2(result,v)})
cp.cuda.Device().synchronize();warm_s=time.perf_counter()-warm_start
warm_pool=cp.get_default_memory_pool().total_bytes()
counts=[1,10,100] if a.accuracy_only else [1,10,100,1000]
measurements=[]
if not a.accuracy_only:
  for count in counts:
    times=[]
    for rep in range(3):
      t=time.perf_counter()
      for j in range(count):render(rotations[j%len(rotations)])
      cp.cuda.Device().synchronize();times.append(time.perf_counter()-t)
      rss_peak=max(rss_peak,process.memory_info().rss)
    print('Benchmark progress',a.method,'orientations',count,'seconds',times,flush=True)
    measurements.append({'orientations':count,'median_render_s':float(np.median(times)),
      'min_render_s':min(times),'max_render_s':max(times),'total_including_setup_s':setup_s+float(np.median(times))})
print('Benchmark timing complete',a.method,measurements,flush=True)
record={'method':a.method,'n':a.n,'m':None if axis3 is None else len(axis3),'padding':a.pad if a.method=='fft' else None,
 'slabbed_transform':a.slab,'qa_count':a.qa_count,'interpolation_order':a.order,'setup_s':setup_s,'warmup_three_outputs_s':warm_s,'accuracy':errors,'measurements':measurements,
 'complex_cache_MiB':cache_bytes/2**20,'cupy_pool_setup_high_water_MiB':setup_pool/2**20,
 'cupy_pool_warm_high_water_MiB':warm_pool/2**20,'host_RSS_increase_MiB':(rss_peak-rss0)/2**20,
 'voxel_mass_relative_error':None if cache is None else mass/expected-1,
 'hardware':'RTX 2070 SUPER 8GB','python':platform.python_version(),'cupy':cp.__version__,'numpy':np.__version__,
 'output':'401 by 401 detector, common annular mask91567 pixels; streamed to host and no outputs cached',
 'precision':'3D cache and slab intermediates complex64; FFT complex64, CZT plans complex128; mesh CPU and research GPU CZT complex128',
 'caveats':['3D setup amortized over rotations of one fixed morphology; changed morphology requires rebuilding cache.',
 '2D mesh raster includes host projection return and transforms then host intensity; production CPU-CZT and optional research GPU-CZT distinguished.',
 'CuPy allocator high-water excludes external CUDA FFT workspace and driver allocations; no claim of exact total GPU peak.',
 'Host RSS sampled after completed batches, not an instrumented continuous peak.',
 'Planar samples compared to independent continuous polyhedral reference; interpolated complex amplitude then squared.',
 'CPU-CZT versus GPU-CZT timings should not be attributed solely to representation.']}
path=stage/f"{a.method}_n{a.n}_m{a.m}_pad{a.pad}_order{a.order}{'_slab' if a.slab else ''}{'_accuracy' if a.accuracy_only else ''}.json"
dump(path,record);print(json.dumps({'method':a.method,'n':a.n,'m':a.m,'order':a.order,'slabbed_transform':a.slab,'qa_count':a.qa_count,'worst_l2':max(e['intensity_relative_l2'] for e in errors),'setup_s':setup_s,'pool_MiB':setup_pool/2**20,'last_measurement':measurements[-1] if measurements else None}),flush=True)
