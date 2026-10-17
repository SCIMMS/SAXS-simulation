"""Reproducible accuracy and isolated-process cost audit of q_z projection stacks.

Commands: controls; accuracy; benchmark --radius {0,4,12} --method ...; report.
All pre-existing scientific outputs and revision documents are read-only inputs.
"""
from __future__ import annotations
import argparse,gc,json,time,platform,sys,threading
from pathlib import Path
from types import SimpleNamespace
from revision_common import ROOT,np,sm,sha,dump,save_csv,id57_meshes,id57_projection,rel_l2
from revision_complex_projection import (ComplexMeshProjector,ComplexProjectionRenderer,
    longitudinal_nodes,longitudinal_weights)
from revision_gpu_boundary_general import MeshBoundaryReference,_GENERIC
from revision_polyhedron_reference import polyhedron_amplitude
from revision_gpu_fourier import GPUCZT2
import cupy as cp
import psutil

OUT=ROOT/'output/complex_projection_2026-10-15'
RADII=[0.,4.,12.]
METHODS=['legacy_planar','resident_planar','qz3','qz5','qz9']
_AMPLITUDE_SRC=_GENERIC.replace('generic_reference','generic_amplitude')
_AMPLITUDE_SRC=_AMPLITUDE_SRC.replace('out[t]=a*a;return;','out[2*t]=core_volume;out[2*t+1]=0.;return;')
_AMPLITUDE_SRC=_AMPLITUDE_SRC.replace('out[t]=real*real+imag*imag;',
    'out[2*t]=real;out[2*t+1]=imag;')


class IndependentBoundaryAmplitude:
    """Continuous triangular surface integral; no raster or interpolation."""
    def __init__(self,meshes,q):
        self.reference=MeshBoundaryReference(meshes,q)
        self.kernel=cp.RawKernel(_AMPLITUDE_SRC,'generic_amplitude',options=('--std=c++11',))

    def amplitude(self,rotation):
        ref=self.reference;r=cp.asarray(np.asarray(rotation).ravel(),dtype=cp.float64)
        chunks=[]
        for start in range(0,ref.n,4096):
            count=min(4096,ref.n-start);out=cp.empty(count,dtype=cp.complex128)
            self.kernel(((count+127)//128,),(128,),
                (*(v[start:start+count] for v in ref.q),r,ref.table,np.int32(ref.nf),
                 np.int32(count),np.float64(ref.volume),out))
            chunks.append(cp.asnumpy(out))
        return np.concatenate(chunks)


def inputs():
    folder=ROOT/'public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240'
    cfg=json.loads((folder/'synthetic_id57_summary.json').read_text())
    with np.load(folder/'synthetic_id57_patterns.npz') as f:
        valid=f['valid_mask'].astype(bool);rotations=f['rotation_matrices'].astype(float)
    yy,xx=np.indices(valid.shape);r=np.hypot(xx-200,yy-200)
    roi=valid&(r>=45)&(r<=190)
    qstep=2*np.pi/cfg['wavelength_nm']*cfg['pixel_size_m']/cfg['detector_distance_m']
    axis=(np.arange(401)-200)*qstep;qx,qy=np.meshgrid(axis,axis)
    k=2*np.pi/cfg['wavelength_nm'];factor=np.sqrt(1+(qx/k)**2+(qy/k)**2)
    curved=np.column_stack([(qx/factor)[roi],(qy/factor)[roi],(k*(1/factor-1))[roi]])
    flat=np.column_stack([qx[roi],qy[roi],np.zeros(roi.sum())])
    sampled=((xx%4==0)&(yy%4==0))[roi]
    return cfg,axis,roi,rotations,curved,flat,sampled,r[roi]*qstep


def geometry(radius):
    outer,core=id57_meshes(rounding=radius,segments=16)
    return [(outer,1.),(core,.5)]


def protected_hashes():
    paths=[ROOT/'uploaded_SAXS_simulator/saxs_module.py',
        ROOT/'validation/revision_common.py',ROOT/'validation/revision_gpu_fourier.py',
        ROOT/'validation/revision_gpu_boundary_general.py',ROOT/'validation/revision_polyhedron_reference.py',
        ROOT/'public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240/synthetic_id57_summary.json',
        ROOT/'public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240/synthetic_id57_patterns.npz']
    paths+=list((ROOT/'output/revision_manuscript_R12_2026-10-15').glob('*.docx'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def summarize(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),
                max=float(a.max()),min=float(a.min()))


def controls():
    OUT.mkdir(exist_ok=True,parents=True);tic=time.perf_counter()
    cfg,axis,roi,rotations,qcur,qflat,sampled,radial=inputs()
    hashes=protected_hashes();checks=[]
    # Analytic longitudinal integral on a translated box checks phase sign,
    # both kappa signs, kappa=0, and the small-kappa limit independently.
    box=geometry(0)[0][0];shift=np.array([1.2,-2.1,7.3])
    moved=SimpleNamespace(vertices=box.vertices+shift,faces=box.faces,face_normals=box.face_normals)
    projector=ComplexMeshProjector([(moved,1.)],n=192,extent=144.)
    ks=np.array([0.,-1e-10,-.005,-.021,.021])
    stack,balance=projector.project(np.eye(3),ks,True);stack=cp.asnumpy(stack)
    xy=(np.arange(192)-(192-1)/2)*144/192;x,y=np.meshgrid(xy,xy)
    inside=(np.abs(x-shift[0])<26)&(np.abs(y-shift[1])<26)
    exact=52*np.sinc(ks*26/np.pi)*np.exp(-1j*ks*shift[2])
    err=rel_l2(stack,exact[:,None,None]*inside[None])
    assert err<2e-6,err
    assert float(cp.max(cp.abs(balance)).get())==0
    checks.append(dict(test='translated_box_analytic_longitudinal_integral',relative_L2=err,
                       kappas=ks.tolist(),closure_max=0))
    # Barycentric interpolation reproduces polynomial moments through degree K-1.
    for count in [3,5,9]:
        nodes=longitudinal_nodes(float(qcur[:,2].min()),0.,count)
        w=longitudinal_weights(nodes,qcur[::37,2]);scale=abs(nodes[0])
        error=max(np.max(np.abs((nodes/scale)**j@w-(qcur[::37,2]/scale)**j))
                  for j in range(count))
        assert error<1e-11,error
        checks.append(dict(test='barycentric_polynomial_reproduction',planes=count,max_abs=error))
    # Complex amplitude checks, including asymmetric/translated density, compare
    # two continuous boundary implementations and an analytic shifted-box FT.
    rng=np.random.default_rng(57)
    probe=np.vstack([np.zeros((1,3)),np.array([[.3,0,0],[.3,.3,0],[.3,.3,.3]]),
                     rng.uniform(-.8,.8,(33,3))])
    analytic=52**3*np.prod(np.sinc(probe*26/np.pi),axis=1)*np.exp(-1j*probe@shift)
    reference=IndependentBoundaryAmplitude([(moved,1.)],probe).amplitude(np.eye(3))
    e=rel_l2(reference,analytic);assert e<1e-7,e
    checks.append(dict(test='continuous_GPU_boundary_vs_analytic_shifted_box_amplitude',relative_L2=e))
    for radius in RADII:
        meshes=geometry(radius)
        q=qcur[::max(1,len(qcur)//43)][:43]
        for oi in [0,144,238]:
            gpu=IndependentBoundaryAmplitude(meshes,q).amplitude(rotations[oi])
            cpu=sum(c*polyhedron_amplitude(q@rotations[oi],mesh) for mesh,c in meshes)
            e=rel_l2(gpu,cpu);assert e<1e-7,(radius,oi,e)
            checks.append(dict(test='independent_GPU_CPU_continuous_boundary_amplitude',
                               radius_nm=radius,orientation=oi,relative_L2=e))
        # q_z=0 matches the existing raster; new complex accumulation is not clamped.
        new=ComplexMeshProjector(meshes)
        for oi in [0,144,238]:
            got,balance=new.project(rotations[oi],[0.],True)
            old,_=id57_projection(rotations[oi],768,144.,tuple(m for m,c in meshes),'gpu')
            got=cp.asnumpy(got[0]);e=rel_l2(got,old)
            bad=int(cp.count_nonzero(cp.abs(balance)>.5).get())
            # Production clamps isolated negative real pixels after float32
            # edge tests. The new closed complex raster uses float64 edges and
            # no clamp, so bitwise equality is neither expected nor desirable.
            assert e<1e-2,(radius,oi,e)
            assert bad==0,(radius,oi,bad)
            checks.append(dict(test='zero_plane_vs_existing_raster',radius_nm=radius,
                               orientation=oi,relative_L2=e,unbalanced_pixels=bad))
    # Existing CPU and GPU CZTs must handle the new genuinely complex projection.
    n=96;small_axis=axis[::10];extent=144.
    p=ComplexMeshProjector([(moved,1.)],n=n,extent=extent).project(np.eye(3),[-.021])[0]
    cpu_axis=(np.arange(n)-(n-1)/2)*extent/n
    cpu=sm.czt2_from_real_axes(cp.asnumpy(p),cpu_axis,cpu_axis,small_axis,small_axis)
    gpu=cp.asnumpy(GPUCZT2(n,extent,small_axis).amplitude(p))
    e=rel_l2(gpu,cpu);assert e<1e-10,e
    checks.append(dict(test='complex_projection_GPU_CPU_CZT_amplitude',relative_L2=e))
    assert hashes==protected_hashes()
    dump(OUT/'controls.json',dict(status='complete',checks=checks,source_hashes=hashes,
                                 seconds=time.perf_counter()-tic))
    print('CONTROLS complete',len(checks),'seconds',time.perf_counter()-tic,flush=True)


def errors(predicted,reference,regions):
    result={}
    for name,region in regions.items():
        result['intensity_L2_'+name]=rel_l2(abs(predicted[region])**2,abs(reference[region])**2)
        result['amplitude_L2_'+name]=rel_l2(predicted[region],reference[region])
    return result


def accuracy():
    OUT.mkdir(exist_ok=True,parents=True);tic=time.perf_counter()
    cfg,axis,roi,rotations,qcur,qflat,sampled,radial=inputs();hashes=protected_hashes()
    config=dict(n=768,extent_nm=144.,detector_shape=[401,401],roi_pixels=int(roi.sum()),
        sweep_sampled_pixels=int(sampled.sum()),sweep_orientation_count=len(rotations),
        rounding_radii_nm=RADII,planes=[3,5,9],qz_band_inv_nm=[float(qcur[:,2].min()),0.],
        nominal_detector_q_inv_nm=[float(radial.min()),float(radial.max())],
        longitudinal_nodes='Chebyshev-Lobatto including endpoints',
        primary_longitudinal_interpolation='barycentric polynomial on complex amplitude',
        secondary_longitudinal_interpolation='piecewise linear on the same planes',
        lateral_interpolation='tensor product four-point Lagrange on complex amplitude',
        precision='float64 oriented geometry/edge tests, float32 phasors and complex64 projection accumulation; complex128 CZT/interpolation/reference',
        reference='continuous oriented triangular boundary integrals; no raster, voxel grid, or interpolation',
        source_hashes=hashes)
    dump(OUT/'config.json',config)
    summary={};allrows=[];dense_rows=[]
    for radius in RADII:
        path=OUT/f'accuracy_radius{radius:g}.csv'
        completed=OUT/f'accuracy_radius{radius:g}.json'
        if completed.exists():
            saved=json.loads(completed.read_text());assert saved['source_hashes']==hashes
            import csv
            allrows+=list(csv.DictReader(path.open()))
            dense_rows+=saved['dense_rows'];summary[str(radius)]=saved['summary'];continue
        meshes=geometry(radius)
        refs={name:IndependentBoundaryAmplitude(meshes,q[sampled])
              for name,q in [('curved',qcur),('flat',qflat)]}
        renderers={K:ComplexProjectionRenderer(meshes,axis,qcur,roi,planes=K) for K in [3,5,9]}
        rr=radial[sampled]
        regions={'roi':np.ones(sampled.sum(),bool),'outer_q_gt_0p9':rr>.9,
                 'q_le_0p416':rr<=70*(axis[201]-axis[200])}
        rows=[]
        for oi,rotation in enumerate(rotations):
            truth=refs['curved'].amplitude(rotation)
            flat=refs['flat'].amplitude(rotation)
            baseline=cp.asnumpy(renderers[3].planar_amplitude(rotation))[sampled]
            legacy_projection,_=id57_projection(rotation,768,144.,tuple(m for m,c in meshes),'gpu')
            legacy=cp.asnumpy(renderers[3].plan.amplitude(legacy_projection))[roi][sampled]
            transverse=cp.asnumpy(renderers[3].planar_amplitude(rotation,True))[sampled]
            preds={'legacy_planar':legacy,'resident_planar':baseline,
                   'transverse_only':transverse,'exact_planar':flat}
            for K,renderer in renderers.items():
                planes=renderer.transform_planes(rotation)
                for mode in ['barycentric','linear']:
                    preds[f'qz{K}_{mode}']=cp.asnumpy(renderer.interpolate(planes,mode))[sampled]
            for method,pred in preds.items():
                row=dict(radius_nm=radius,orientation=oi,method=method,
                         sampled_pixels=int(sampled.sum()),**errors(pred,truth,regions))
                if method in ['legacy_planar','resident_planar']:
                    row['planar_raster_numerical_intensity_L2']=rel_l2(abs(pred)**2,abs(flat)**2)
                else:row['planar_raster_numerical_intensity_L2']=''
                rows.append(row)
            if oi%20==0:
                save_csv(path,rows)
                print('ACCURACY radius',radius,'orientation',oi,'seconds',round(time.perf_counter()-tic,1),flush=True)
        save_csv(path,rows);allrows+=rows
        model_summary={}
        for method in preds:
            selected=[r for r in rows if r['method']==method]
            model_summary[method]={key:summarize([r[key] for r in selected]) for key in
                ['intensity_L2_roi','amplitude_L2_roi','intensity_L2_outer_q_gt_0p9']}
        # Dense checks at three predeclared orientations plus the worst sampled
        # orientation of each primary method. No claim of all-pixel/240 coverage.
        dense_ids={0,144,238}
        for method in ['legacy_planar','qz3_barycentric','qz5_barycentric','qz9_barycentric']:
            dense_ids.add(max((r for r in rows if r['method']==method),key=lambda r:r['intensity_L2_roi'])['orientation'])
        full_ref=IndependentBoundaryAmplitude(meshes,qcur)
        full_regions={'roi':np.ones(len(qcur),bool),'outer_q_gt_0p9':radial>.9,
                      'q_le_0p416':radial<=70*(axis[201]-axis[200])}
        these_dense=[]
        for oi in sorted(dense_ids):
            rotation=rotations[oi];truth=full_ref.amplitude(rotation)
            legacy_projection,_=id57_projection(rotation,768,144.,tuple(m for m,c in meshes),'gpu')
            predictions={'legacy_planar':cp.asnumpy(renderers[3].plan.amplitude(legacy_projection))[roi],
                         'resident_planar':cp.asnumpy(renderers[3].planar_amplitude(rotation))}
            for K,renderer in renderers.items():
                predictions[f'qz{K}_barycentric']=cp.asnumpy(renderer.amplitude(rotation))
            for method,pred in predictions.items():
                these_dense.append(dict(radius_nm=radius,orientation=oi,method=method,
                    pixels=len(qcur),**errors(pred,truth,full_regions)))
            if oi in [0,144,238]:
                np.savez_compressed(OUT/f'example_radius{radius:g}_orientation{oi}.npz',
                    truth=truth,**predictions,roi=roi,qaxis=axis,qphysical=qcur)
            print('DENSE radius',radius,'orientation',oi,'seconds',round(time.perf_counter()-tic,1),flush=True)
        dense_rows+=these_dense;summary[str(radius)]=model_summary
        dump(completed,dict(status='complete',source_hashes=hashes,summary=model_summary,
                            dense_rows=these_dense,dense_orientation_ids=sorted(dense_ids)))
        del refs,renderers,full_ref;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    save_csv(OUT/'dense_accuracy.csv',dense_rows)
    assert hashes==protected_hashes()
    dump(OUT/'accuracy_summary.json',dict(status='complete',config=config,models=summary,
         dense_intensity_L2_max_by_method={m:max(float(r['intensity_L2_roi']) for r in dense_rows if r['method']==m)
          for m in ['legacy_planar','qz3_barycentric','qz5_barycentric','qz9_barycentric']},
         seconds=time.perf_counter()-tic,source_hashes_preserved=True))


class AllocationPeak(cp.cuda.MemoryHook):
    name='ComplexProjectionAllocationPeak'
    def __init__(self):self.current=0;self.peak=0;self.live={}
    def malloc_postprocess(self,**kwargs):
        if kwargs['pmem_id']:
            size=kwargs['mem_size'];self.live[kwargs['pmem_id']]=size
            self.current+=size;self.peak=max(self.peak,self.current)
    def free_postprocess(self,**kwargs):
        self.current-=self.live.pop(kwargs['pmem_id'],0)


def benchmark(radius,method,repeats=3,views=48):
    """Run ONE method/model in a fresh OS process, reference work excluded."""
    OUT.mkdir(exist_ok=True,parents=True)
    cfg,axis,roi,rotations,qcur,qflat,sampled,radial=inputs();meshes=geometry(radius)
    ids=np.linspace(0,len(rotations)-1,views).round().astype(int)
    pair=tuple(m for m,c in meshes)

    def setup():
        if method=='legacy_planar':
            plan=GPUCZT2(768,144.,axis)
            def render(r):
                p,_=id57_projection(r,768,144.,pair,'gpu')
                return plan.intensity(p)[roi]
            return render,plan
        if method=='resident_planar':
            projector=ComplexMeshProjector(meshes)
            plan=GPUCZT2(768,144.,axis);roig=cp.asarray(roi)
            def render(r):
                p=projector.project(r,[0.])[0]
                a=plan.amplitude(p)[roig]
                return cp.asnumpy(a.real*a.real+a.imag*a.imag)
            return render,(projector,plan,roig)
        renderer=ComplexProjectionRenderer(meshes,axis,qcur,roi,
            planes=int(method[2:]) if method.startswith('qz') else 3)
        render=renderer.intensity
        return render,renderer

    # Warm CUDA, JIT, FFT plans, and every measured shape before recording costs.
    render,obj=setup()
    for i in [0,144,238]:render(rotations[i])
    cp.cuda.Device().synchronize();del render,obj;gc.collect()
    cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
    process=psutil.Process();rss0=process.memory_info().rss;rss_peak=[rss0];stop=threading.Event()
    def sample_rss():
        while not stop.wait(.01):rss_peak[0]=max(rss_peak[0],process.memory_info().rss)
    monitor=threading.Thread(target=sample_rss,daemon=True);monitor.start()
    # Timing excludes hooks and the RSS monitor. A separate identical memory pass
    # records transient CuPy allocations; host RSS is sampled during that pass.
    start=time.perf_counter();render,obj=setup();cp.cuda.Device().synchronize()
    setup_s=time.perf_counter()-start
    start=time.perf_counter();render(rotations[0]);cp.cuda.Device().synchronize()
    first_s=time.perf_counter()-start
    stop.set();monitor.join()
    times=[]
    for repeat in range(repeats):
        start=time.perf_counter()
        for i in ids:output=render(rotations[i])
        cp.cuda.Device().synchronize();times.append(time.perf_counter()-start)
    assert output.shape==(int(roi.sum()),) and np.isfinite(output).all()
    del render,obj,output;gc.collect();cp.fft.config.get_plan_cache().clear()
    cp.get_default_memory_pool().free_all_blocks()
    rss_before_memory=process.memory_info().rss;rss_peak[0]=rss_before_memory
    stop.clear();monitor=threading.Thread(target=sample_rss,daemon=True);monitor.start()
    hook=AllocationPeak()
    with hook:
        render,obj=setup()
        for i in ids[:12]:output=render(rotations[i])
        cp.cuda.Device().synchronize()
        reserved=cp.get_default_memory_pool().total_bytes()
        fft_cache=cp.fft.config.get_plan_cache().get_curr_memsize()
        current=cp.get_default_memory_pool().used_bytes()
    stop.set();monitor.join()
    record=dict(status='complete',method=method,radius_nm=radius,n=768,extent_nm=144.,
        detector_shape=[401,401],returned_roi_pixels=int(roi.sum()),timed_orientation_ids=ids.tolist(),
        repeats=repeats,batch_seconds=times,median_ms_per_orientation=1000*float(np.median(times))/views,
        setup_s=setup_s,first_output_s=first_s,
        cupy_live_allocation_peak_MiB=hook.peak/2**20,cupy_reserved_high_water_MiB=reserved/2**20,
        cupy_live_after_output_MiB=current/2**20,cached_FFT_workspace_MiB=fft_cache/2**20,
        sampled_host_RSS_increment_MiB=(rss_peak[0]-rss_before_memory)/2**20,
        logical_projection_stack_MiB=(0 if method=='legacy_planar' else
            (int(method[2:]) if method.startswith('qz') else 1)*768**2*8/2**20),
        logical_Fourier_stack_MiB=(0 if not method.startswith('qz') else int(method[2:])*401**2*16/2**20),
        hardware=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),python=platform.python_version(),
        cupy=cp.__version__,numpy=np.__version__,
        caveats=['Steady timing uses synchronized wall clock, includes host rotation/transfer and returned intensities, excludes setup, JIT and FFT first-use.',
          'Allocation peak is a separate hook-instrumented pass, not inferred from before/after snapshots.',
          'CuPy live allocation excludes CUDA context/driver and external library allocations; FFT cache is reported separately and may overlap allocator counts.',
          'Reserved memory is allocator high-water, not total GPU process memory.',
          'Host RSS uses 10ms sampling and may reuse previously allocated pages.',
          'Resident planar uses the new combined-domain GPU kernel; legacy retains existing per-domain projection transfers and CPU diagnostics.',
          'Complex renderers retain interpolation weights for both modes; timed path uses barycentric interpolation only.',
          'Full complex projection and Fourier stacks are stored for this first implementation; no orientation cache or streaming.'])
    dump(OUT/f'benchmark_radius{radius:g}_{method}.json',record)
    print('BENCHMARK',radius,method,json.dumps({k:record[k] for k in
        ['median_ms_per_orientation','cupy_live_allocation_peak_MiB','cupy_reserved_high_water_MiB']}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['controls','accuracy','benchmark'])
    parser.add_argument('--radius',type=float,choices=RADII,default=0.)
    parser.add_argument('--method',choices=METHODS,default='qz5')
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--views',type=int,default=48)
    args=parser.parse_args()
    if args.command=='controls':controls()
    elif args.command=='accuracy':accuracy()
    else:benchmark(args.radius,args.method,args.repeats,args.views)


if __name__=='__main__':main()
