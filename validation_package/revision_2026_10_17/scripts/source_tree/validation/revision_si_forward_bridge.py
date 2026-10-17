"""Same-mesh planar forward bridge for the differentiable SI extension.

Existing simulator, validation modules and experimental results are read-only.
This comparison tests discrete closed meshes, not recovery of physical structure.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from types import SimpleNamespace
from revision_common import ROOT, dump, save_csv, sha, np, sm, project_mesh, rel_l2
from revision_differentiable_synthetic import DifferentiableForward, old
from revision_differentiable_real import CoreShellFamily
from revision_gpu_fourier import GPUCZT2
from revision_gpu_boundary_general import MeshBoundaryReference
from revision_polyhedron_reference import polyhedron_amplitude
from cupyx.scipy.signal import CZT
import cupy as cp

OUT=ROOT/'output/differentiable_si_validation_2026-10-12/forward_bridge'
EXTENT=144.
GRIDS=[384,768,1536]
ORDERS=[4,8]
QAXIS=np.linspace(-.70,.70,65)
ROTATIONS=[np.eye(3),sm.rotation_matrix_axis_angle([1.,2.,3.],.47),sm.rotation_matrix_axis_angle([2.,-1.,1.],.91)]
SOURCES=[Path(__file__),ROOT/'validation/revision_common.py',ROOT/'validation/revision_differentiable_synthetic.py',
 ROOT/'validation/revision_differentiable_real.py',ROOT/'validation/revision_gpu_fourier.py',
 ROOT/'validation/revision_gpu_boundary_general.py',ROOT/'validation/revision_gpu_surface_reference.py',
 ROOT/'validation/revision_polyhedron_reference.py',ROOT/'uploaded_SAXS_simulator/saxs_module.py',
 ROOT/'deliverables/projected_gradient_mesh_saxs_explainer_2026-09-02/examples/minimal_projected_gradient_optimizer.py']

def input_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}

def make_mesh(v,f):
    v=np.asarray(v,dtype=np.float64);f=np.asarray(f,dtype=np.int32)
    f,_,_=sm.orient_faces_outward_by_volume(v,f)
    f=np.asarray(f,dtype=np.int32)
    return SimpleNamespace(vertices=v,faces=f,face_normals=sm.compute_face_normals(v,f))

def models():
    v,f=old.unit_uv_sphere(24,80)
    ellipsoid=make_mesh(v*np.array([31.,23.,17.]),f)
    family=CoreShellFamily(flat=12,arc=12)
    theta=np.zeros(13);theta[3]=5.
    boundaries=family.boundaries(theta)
    # Restrict each boundary to only its own vertices; avoid unused interface vertices.
    clean=[]
    for mesh,contrast in boundaries:
        used=np.unique(mesh.faces);mapping=np.full(len(mesh.vertices),-1,dtype=int);mapping[used]=np.arange(len(used))
        clean.append((make_mesh(mesh.vertices[used],mapping[mesh.faces]),contrast))
    return {'ellipsoid':[(ellipsoid,1.)], 'rounded_box':[clean[0]], 'core_shell':clean}

class StaticFamily:
    def __init__(self,boundaries,rotation):
        vertices=[];faces=[];contrasts=[];offset=0
        for mesh,contrast in boundaries:
            vertices.append(mesh.vertices@rotation.T)
            faces.append(mesh.faces+offset);offset+=len(mesh.vertices)
            contrasts.extend([contrast]*len(mesh.faces))
        self.vertices=np.concatenate(vertices);self.faces=np.asarray(np.concatenate(faces),dtype=np.int32)
        tri=self.vertices[self.faces]
        self.normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])*np.asarray(contrasts)[:,None]
        self.volume=float(np.einsum('ij,ij->i',tri[:,0],self.normal).sum()/6)
    def mesh(self,theta):
        return self.vertices,np.zeros((0,len(self.vertices),3)),self.normal,np.zeros((0,len(self.faces),3)),self.volume,np.zeros(0)

def diff_forward(family,n,order):
    fwd=DifferentiableForward(family,n=n,quad=order,qsize=len(QAXIS),qmax=.70)
    fwd.extent=EXTENT;fwd.du=EXTENT/n
    dq=QAXIS[1]-QAXIS[0]
    fwd.plan=CZT(n=n,m=len(QAXIS),w=np.exp(-1j*dq*fwd.du),a=np.exp(1j*QAXIS[0]*fwd.du))
    fwd.phase=cp.asarray(fwd.du*np.exp(-1j*QAXIS*(-(n-1)*fwd.du/2)))
    fwd.cic=cp.asarray(np.sinc(fwd.qx*fwd.du/(2*np.pi))**2*np.sinc(fwd.qy*fwd.du/(2*np.pi))**2)
    return fwd

def mesh_audit(mesh):
    f=mesh.faces;v=mesh.vertices;t=v[f]
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    sorted_edges=np.sort(edges,axis=1);_,counts=np.unique(sorted_edges,axis=0,return_counts=True)
    # Directed edges must cancel across adjacent faces.
    canon=np.unique(sorted_edges,axis=0);lookup={tuple(e):i for i,e in enumerate(canon)}
    balances=np.zeros(len(canon),dtype=int)
    for a,b in edges:balances[lookup[tuple(sorted((int(a),int(b))))]]+=1 if a<b else -1
    volumes=np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2]))/6
    return {'vertices':len(v),'faces':len(f),'positive_volume_nm3':float(volumes.sum()),
      'all_edges_have_two_faces':bool(np.all(counts==2)),'all_edges_consistently_oriented':bool(np.all(balances==0)),
      'min_triangle_double_area_nm2':float(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).min()),
      'volume_centroid_nm':(np.sum(volumes[:,None]*t.sum(axis=1)/4,axis=0)/volumes.sum()).tolist(),
      'max_abs_coordinate_nm':float(np.abs(v).max())}

def metric(a,b,mask):return rel_l2(np.asarray(a)[mask],np.asarray(b)[mask])

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time();initial=input_hashes()
    qx,qy=np.meshgrid(QAXIS,QAXIS);qr=np.sqrt(qx*qx+qy*qy)
    q=np.column_stack([qx.ravel(),qy.ravel(),np.zeros(qx.size)])
    masks={'all':np.ones(qx.shape,bool),'roi':(qr>=.16)&(qr<=.66),'low_q':qr<=.30}
    config={'schema':1,'date':'2026-10-12','length_unit':'nm','density_unit':'relative contrast',
     'extent_nm':EXTENT,'projection_grids':GRIDS,'quadrature_orders':ORDERS,'qaxis_nm_inverse':QAXIS.tolist(),
     'q_z':0.,'rotations':ROTATIONS,'metric':'unnormalized intensity relative L2; no fitted signal/background/scale',
     'roi':'.16 <= |q| <= .66 nm^-1','all_q_includes_q0':True,
     'same_geometry':'one saved triangulation per model; identical rotated vertices and contrast in all methods',
     'original_method':'unchanged production signed_raster_gpu (float32 accumulation), followed by float64 GPU CZT matching production CPU convention',
     'differentiable_method':'unchanged double-precision surface-gradient Duffy quadrature/CIC deposition, CZT and CIC transfer correction',
     'independent_reference':'analytic triangle boundary integral on identical mesh, evaluated at same planar q grid',
     'cpu_crosschecks':'production float64 signed raster on n384; CPU analytic boundary amplitude at 97 fixed q indices',
     'scope':'proof-of-concept planar density-model equivalence, not benchmark of curved-Ewald GroEL implementation or physical truth'}
    dump(OUT/'config.json',config);dump(OUT/'source_hashes.json',initial)
    all_models=models();audits={};rows=[];checks=[];cpu_rows=[]
    for name,boundaries in all_models.items():
        audits[name]=[dict(mesh_audit(m),contrast=c) for m,c in boundaries]
        arrays={}
        for j,(m,c) in enumerate(boundaries):arrays.update({f'vertices_{j}':m.vertices,f'faces_{j}':m.faces,f'contrast_{j}':np.array(c)})
        np.savez_compressed(OUT/f'{name}_mesh.npz',**arrays)
        for ri,R in enumerate(ROTATIONS):
            label=f'{name}_orientation_{ri}'
            family=StaticFamily(boundaries,R);charge=family.volume
            ref=MeshBoundaryReference(boundaries,q).intensity(R).reshape(qx.shape)
            selected=np.unique(np.linspace(0,len(q)-1,97,dtype=int))
            cpu_amplitude=sum(c*polyhedron_amplitude(q[selected]@R,m) for m,c in boundaries)
            cpu_ref=np.abs(cpu_amplitude)**2
            cpu_rows.append({'model':name,'orientation':ri,'points':len(selected),
              'gpu_cpu_exact_intensity_rel_l2':rel_l2(ref.ravel()[selected],cpu_ref),
              'q0_intensity_charge_squared_relative_error':float(abs(ref[32,32]-charge**2)/charge**2)})
            stored={'qaxis':QAXIS,'rotation':R,'reference_intensity':ref,'charge':np.array(charge)}
            for n in GRIDS:
                projection=np.zeros((n,n));diagnostics=[]
                for mesh,c in boundaries:
                    p,d=project_mesh(mesh,R,n,EXTENT,'gpu');projection+=c*p;diagnostics.append(d)
                original=GPUCZT2(n,EXTENT,QAXIS).intensity(projection)
                rastercharge=float(projection.sum()*(EXTENT/n)**2)
                record={'model':name,'orientation':ri,'grid':n,'quad':0,'method':'original_raster_czt',
                  'charge_relative_error':abs(rastercharge-charge)/charge}
                record.update({f'intensity_rel_l2_{k}':metric(original,ref,m) for k,m in masks.items()})
                record.update({f'original_diff_rel_l2_{k}':'' for k in masks});rows.append(record)
                stored[f'original_n{n}']=original
                if n==GRIDS[0]:
                    cpu_projection=np.zeros_like(projection)
                    for mesh,c in boundaries:cpu_projection+=c*project_mesh(mesh,R,n,EXTENT,'cpu')[0]
                    axes=(np.arange(n)-(n-1)/2)*EXTENT/n
                    cpu_czt=sm.czt2_from_real_axes(projection,axes,axes,QAXIS,QAXIS)
                    checks.append({'model':name,'orientation':ri,'cpu_gpu_projection_rel_l2':rel_l2(projection,cpu_projection),
                      'cpu_gpu_czt_intensity_rel_l2':rel_l2(original,np.abs(cpu_czt)**2),
                      'diagnostics':diagnostics})
                for order in ORDERS:
                    amplitude=diff_forward(family,n,order).amplitude(np.zeros(0),derivatives=False)[0]
                    intensity=np.abs(amplitude)**2
                    record={'model':name,'orientation':ri,'grid':n,'quad':order,'method':'differentiable_surface_czt',
                      'charge_relative_error':float(abs(amplitude[32,32]-charge)/charge)}
                    record.update({f'intensity_rel_l2_{k}':metric(intensity,ref,m) for k,m in masks.items()})
                    record.update({f'original_diff_rel_l2_{k}':metric(original,intensity,m) for k,m in masks.items()})
                    rows.append(record);stored[f'differentiable_n{n}_quad{order}']=intensity
                print(f'{label} n={n}: original ROI {rows[-3]["intensity_rel_l2_roi"]:.6g}; differential quad8 ROI {rows[-1]["intensity_rel_l2_roi"]:.6g}',flush=True)
            np.savez_compressed(OUT/f'{label}_predictions.npz',**stored)
    save_csv(OUT/'forward_metrics.csv',rows);save_csv(OUT/'exact_cpu_crosschecks.csv',cpu_rows)
    dump(OUT/'mesh_audit.json',audits);dump(OUT/'production_cpu_gpu_checks.json',checks)
    assert initial==input_hashes(),'An existing source changed during execution'
    dump(OUT/'run_state.json',{'status':'computed','elapsed_seconds':time.time()-start,'rows':len(rows)})
    audit()

def audit():
    cfg=json.loads((OUT/'config.json').read_text());rows=list(csv.DictReader((OUT/'forward_metrics.csv').open(encoding='utf-8')))
    audits=json.loads((OUT/'mesh_audit.json').read_text());mesh_pass=all(a['all_edges_have_two_faces'] and a['all_edges_consistently_oriented'] and a['positive_volume_nm3']>0 and a['min_triangle_double_area_nm2']>0 for v in audits.values() for a in v)
    hashes=json.loads((OUT/'source_hashes.json').read_text());unchanged=hashes==input_hashes()
    numerical_recomputations=[]
    qx,qy=np.meshgrid(QAXIS,QAXIS);qr=np.sqrt(qx*qx+qy*qy)
    masks={'all':np.ones(qx.shape,bool),'roi':(qr>=.16)&(qr<=.66),'low_q':qr<=.30}
    for row in rows:
        name=row['model'];ri=int(row['orientation']);n=int(row['grid']);order=int(row['quad'])
        with np.load(OUT/f'{name}_orientation_{ri}_predictions.npz') as z:
            assert np.array_equal(z['qaxis'],QAXIS) and np.array_equal(z['rotation'],ROTATIONS[ri])
            key=f'original_n{n}' if order==0 else f'differentiable_n{n}_quad{order}'
            ref=z['reference_intensity'];pred=z[key]
            assert np.all(np.isfinite(pred)) and np.all(pred>=0)
            for band,mask in masks.items():
                val=np.linalg.norm(pred[mask]-ref[mask])/np.linalg.norm(ref[mask])
                numerical_recomputations.append(abs(val-float(row[f'intensity_rel_l2_{band}'])))
    high=[r for r in rows if int(r['grid'])==1536]
    summary={'grid_1536':{method:{'max_intensity_rel_l2_'+band:max(float(r['intensity_rel_l2_'+band]) for r in high if r['method']==method and (int(r['quad'])==0 or int(r['quad'])==8)) for band in masks} for method in ['original_raster_czt','differentiable_surface_czt']},
      'max_original_vs_differentiable_roi_quad8_grid1536':max(float(r['original_diff_rel_l2_roi']) for r in high if int(r['quad'])==8),
      'grid_sensitivity':'All three grids and both Duffy orders retained; convergence need not be monotonic because sampling and CIC phase change.',
      'interpretation':'Both implementations approximate the same closed piecewise-density mesh under planar q. Numerical agreement does not make projected-gradient deposition the same algorithm as signed rasterization.',
      'caveats':['No amplitude/intensity normalization fitted; relative L2 is dominated by stronger scattering, so a separate .16-.66 annular ROI is reported.',
        'All-q metric includes direct beam; low-q and ROI metrics are additionally reported.',
        'Exact reference applies to the discrete saved mesh, not an ideal ellipsoid or analytically rounded shape.',
        'Duffy quadrature, CIC deposition and raster boundary sampling have distinct finite-grid errors.',
        'GPU original projection uses production float32; differentiable surface and Fourier evaluation use float64.',
        'This bridge excludes curved Ewald/atomic form factors; it does not validate every extended implementation.']}
    dump(OUT/'summary.json',summary)
    exact=list(csv.DictReader((OUT/'exact_cpu_crosschecks.csv').open(encoding='utf-8')))
    production=json.loads((OUT/'production_cpu_gpu_checks.json').read_text())
    audit_result={'status':'complete','expected_rows':81,'actual_rows':len(rows),'same_q_axes_and_orientation_verified':True,
     'meshes_closed_consistently_oriented_and_nondegenerate':mesh_pass,'existing_source_hashes_unchanged':unchanged,
     'all_predictions_finite_nonnegative':True,'all_csv_metrics_independently_recomputed':True,
     'max_metric_recompute_absolute_error':max(numerical_recomputations),
     'max_gpu_cpu_exact_intensity_rel_l2':max(float(r['gpu_cpu_exact_intensity_rel_l2']) for r in exact),
     'max_cpu_gpu_projection_rel_l2_n384':max(r['cpu_gpu_projection_rel_l2'] for r in production),
     'max_cpu_gpu_czt_intensity_rel_l2':max(r['cpu_gpu_czt_intensity_rel_l2'] for r in production),
     'q0':'Differentiable q0 is explicit signed volume (not a deposition error test); raster q0 tests integrated projection charge.',
     'checks_cover':'3 fixed meshes x 3 fixed rotations x 3 projection grids; 2 quadrature orders; no selected-case exclusion'}
    assert len(rows)==81 and mesh_pass and unchanged and max(numerical_recomputations)<1e-14
    assert audit_result['max_gpu_cpu_exact_intensity_rel_l2']<1e-7
    assert audit_result['max_cpu_gpu_czt_intensity_rel_l2']<1e-9
    dump(OUT/'validation_audit.json',audit_result)
    outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    dump(OUT/'manifest.json',{'validation_status':'complete','source_hashes':hashes,'output_hashes':outputs})
    print(json.dumps(summary,indent=2),flush=True)

def supplement():
    """CPU-only sampling crosscheck and static scientific figure after main run."""
    hashes=json.loads((OUT/'source_hashes.json').read_text());now=input_hashes()
    script=str(Path(__file__).relative_to(ROOT))
    assert all(now[k]==v for k,v in hashes.items() if k!=script)
    previous=hashes[script];hashes[script]=now[script]
    dump(OUT/'source_hashes.json',hashes)
    dump(OUT/'postprocessing_provenance.json',{'main_execution_script_sha256':previous,
      'final_script_sha256':now[script],'change_scope':'Added CPU-only raster sensitivity and static figure; primary GPU prediction arrays retained unchanged.'})
    all_models=models();rows=[]
    qx,qy=np.meshgrid(QAXIS,QAXIS);qr=np.sqrt(qx*qx+qy*qy);roi=(qr>=.16)&(qr<=.66)
    for name,boundaries in all_models.items():
        for ri,R in enumerate(ROTATIONS):
            selected_grids=GRIDS if name=='ellipsoid' and ri==0 else GRIDS[:1]
            for n in selected_grids:
                p=np.zeros((n,n));diagnostics=[]
                for mesh,c in boundaries:
                    part,diag=project_mesh(mesh,R,n,EXTENT,'cpu');p+=c*part;diagnostics.append(diag)
                axes=(np.arange(n)-(n-1)/2)*EXTENT/n
                amplitude=sm.czt2_from_real_axes(p,axes,axes,QAXIS,QAXIS);I=np.abs(amplitude)**2
                with np.load(OUT/f'{name}_orientation_{ri}_predictions.npz') as z:
                    original=z[f'original_n{n}'];ref=z['reference_intensity'];charge=float(z['charge'])
                    rows.append({'model':name,'orientation':ri,'grid':n,'cpu_raster_intensity_rel_l2_roi':metric(I,ref,roi),
                      'gpu_raster_intensity_rel_l2_roi':metric(original,ref,roi),
                      'cpu_gpu_intensity_rel_l2_roi':metric(original,I,roi),
                      'cpu_raster_charge_relative_error':abs(float(p.sum()*(EXTENT/n)**2)-charge)/charge,
                      'cpu_bad_balance_pixels':sum(int(d['bad_balance_pixels']) for d in diagnostics)})
                np.savez_compressed(OUT/f'{name}_orientation_{ri}_cpu_n{n}.npz',intensity=I)
    save_csv(OUT/'cpu_raster_sensitivity.csv',rows)
    plot()
    audit()

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((OUT/'forward_metrics.csv').open(encoding='utf-8')))
    contract={'question':'How accurately do the two forward implementations evaluate the same planar discrete mesh as projection sampling increases?',
      'takeaway':'Both approximate the same density mesh, with different finite-grid errors; axis-aligned box boundaries dominate the worst raster case.',
      'family':'comparison','variant':'three faceted numerical grid-sensitivity line plots',
      'static_renderer':'Matplotlib PNG/SVG for SI export','metric':'maximum annular intensity relative L2 over three predeclared orientations, percent',
      'orientation_count_per_model':3,'projection_grid_count':3,'source':'forward_metrics.csv (81 rows)',
      'palette':'blue/orange/green, differentiated by marker and line style','qa':'inspect standalone PNG'}
    dump(OUT/'chart_contract.json',contract)
    fig,axes=plt.subplots(1,3,figsize=(11.2,4.1),sharey=True)
    specs=[(0,'Original raster + CZT','#2563A6','o','-'),(4,'Differentiable, order 4','#C17700','s','--'),(8,'Differentiable, order 8','#23856A','^',':')]
    for ax,name,title in zip(axes,['ellipsoid','rounded_box','core_shell'],['Ellipsoid','Rounded box','Core-shell']):
        for order,label,color,marker,ls in specs:
            values=[100*max(float(r['intensity_rel_l2_roi']) for r in rows if r['model']==name and int(r['grid'])==n and int(r['quad'])==order) for n in GRIDS]
            ax.plot(GRIDS,values,label=label,color=color,marker=marker,linestyle=ls,linewidth=1.7,markersize=5)
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(GRIDS,[str(n) for n in GRIDS])
        ax.set_title(title,fontsize=11);ax.set_xlabel('Projection grid per axis',fontsize=10)
        ax.grid(True,axis='y',which='major',alpha=.20);ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Worst annular intensity relative L2 (%)',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.90),ncol=3,frameon=False,fontsize=9.5)
    fig.suptitle('Same-mesh planar forward comparison',fontsize=14,y=.985)
    fig.text(.5,.018,'Maximum over 3 fixed orientations per model; 0.16–0.66 nm⁻¹ annulus; no fitted normalization.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.73,bottom=.18,left=.09,right=.985,wspace=.12)
    fig.savefig(OUT/'forward_bridge_grid_sensitivity.png',dpi=220)
    fig.savefig(OUT/'forward_bridge_grid_sensitivity.svg')
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','audit','supplement']);args=parser.parse_args()
    {'run':run,'audit':audit,'supplement':supplement}[args.action]()
