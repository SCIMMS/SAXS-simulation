"""Known-phase finite-q edge blur and ID57 rounded-cube sensitivity.
Run: .venv_cxidb/Scripts/python.exe validation/revision_resolution_validation.py
No phase retrieval, experimental fitting, Ewald or atomistic validation.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, itertools, json, os, platform, time
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime_cache/matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_functions(source, names):
    text = source.read_text(encoding="utf-8-sig")
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace, hashlib.sha256(text.encode()).hexdigest()

KERNEL, KERNEL_HASH = load_functions(ROOT / "uploaded_SAXS_simulator/saxs_module.py",
    ["compute_face_normals", "signed_mesh_volume", "_top_left_edge", "_inside_half_open_barycentric", "_signed_thickness_from_rasterization_cpu"])
SCORING, SCORING_HASH = load_functions(ROOT / "validation/section5_id57_retrieval.py", ["zscore", "fit_scale_background_and_score"])

def finish_mesh(points, triangles):
    vertices, mapping, faces = [], {}, []
    for triangle in triangles:
        indices = []
        for p in triangle:
            key = tuple(np.round(points[p], 10))
            if key not in mapping:
                mapping[key] = len(vertices)
                vertices.append(points[p])
            indices.append(mapping[key])
        if len(set(indices)) == 3:
            faces.append(indices)
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    tri = vertices[faces]
    inward = np.einsum("ij,ij->i", np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), tri.mean(axis=1)) < 0
    faces[inward] = faces[inward][:,[0,2,1]]
    edges = np.sort(np.concatenate([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), "Mesh is not watertight"
    assert len(vertices)-len(counts)+len(faces) == 2
    return SimpleNamespace(vertices=vertices, faces=faces, face_normals=KERNEL["compute_face_normals"](vertices, faces))

def halfspace_mesh(normals, bounds):
    normals, bounds = np.asarray(normals, float), np.asarray(bounds, float)
    points = []
    for ids in itertools.combinations(range(len(bounds)), 3):
        matrix = normals[list(ids)]
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        p = np.linalg.solve(matrix, bounds[list(ids)])
        if np.all(normals @ p <= bounds+1e-8) and not any(np.linalg.norm(p-other)<1e-8 for other in points):
            points.append(p)
    points, triangles = np.asarray(points), []
    for normal, bound in zip(normals, bounds):
        ids = np.flatnonzero(np.abs(points @ normal-bound)<1e-7)
        if len(ids)<3:
            continue
        center = points[ids].mean(axis=0)
        u = points[ids[0]]-center
        u /= np.linalg.norm(u)
        v = np.cross(normal/np.linalg.norm(normal), u)
        delta = points[ids]-center
        ids = ids[np.argsort(np.arctan2(delta @ v, delta @ u))]
        triangles += [[ids[0],ids[j],ids[j+1]] for j in range(1,len(ids)-1)]
    return finish_mesh(points, triangles)

def rounded_cube(side, radius, segments):
    axes = np.vstack([np.eye(3),-np.eye(3)])
    h, r = side/2, radius
    if r == 0:
        return halfspace_mesh(axes, np.full(6,h))
    assert 0<r<h
    b, points, triangles = h-r, [], []
    angles = np.linspace(0,np.pi/2,segments+1)
    def patch(array):
        offset = len(points)
        ny,nx,_ = array.shape
        points.extend(array.reshape(-1,3))
        for j in range(ny-1):
            for i in range(nx-1):
                a=offset+j*nx+i
                triangles.extend([[a,a+1,a+nx+1],[a,a+nx+1,a+nx]])
    for axis in range(3):
        rest=[j for j in range(3) if j!=axis]
        for sign in [-1,1]:
            array=np.zeros((2,2,3))
            array[...,axis]=sign*h
            array[...,rest[0]]=np.array([-b,b])[None,:]
            array[...,rest[1]]=np.array([-b,b])[:,None]
            patch(array)
        for signs in itertools.product([-1,1],repeat=2):
            array=np.zeros((2,segments+1,3))
            array[...,axis]=np.array([-b,b])[:,None]
            array[...,rest[0]]=signs[0]*(b+r*np.cos(angles))[None,:]
            array[...,rest[1]]=signs[1]*(b+r*np.sin(angles))[None,:]
            patch(array)
    theta,phi=np.meshgrid(angles,angles,indexing="ij")
    sphere=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=-1)
    for signs in itertools.product([-1,1],repeat=3):
        patch((b+r*sphere)*np.asarray(signs))
    return finish_mesh(np.asarray(points),triangles)

def exact_volume(side,r):
    s=side-2*r
    return s**3+6*s*s*r+3*np.pi*s*r*r+4*np.pi*r**3/3

def project(mesh,rotation,n,extent):
    rotated=SimpleNamespace(vertices=mesh.vertices @ rotation.T,faces=mesh.faces,face_normals=mesh.face_normals @ rotation.T)
    return KERNEL["_signed_thickness_from_rasterization_cpu"](rotated,
        u=np.array([1.,0.,0.]),v=np.array([0.,1.,0.]),n=np.array([0.,0.,1.]),
        uv_bounds=(-extent/2,extent/2,-extent/2,extent/2),image_shape=(n,n))

def rel_l2(value,reference):
    return float(np.linalg.norm(value-reference)/np.linalg.norm(reference))

def forward_detector(projection,coords,qaxis):
    exponent=np.exp(-1j*qaxis[:,None]*coords[None,:])
    return np.abs((exponent @ projection @ exponent.T)*(coords[1]-coords[0])**2)**2

def save_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def json_safe(value):
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,np.generic): return json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

def crossing(x,profile,threshold,edge=26.):
    ids=np.flatnonzero((profile[:-1]>=threshold)&(profile[1:]<threshold))
    if not len(ids): return float("nan")
    positions=x[ids]+(threshold-profile[ids])*(x[ids+1]-x[ids])/(profile[ids+1]-profile[ids])
    return float(positions[np.argmin(np.abs(positions-edge))])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"output/resolution_validation_2026-10-09")
    parser.add_argument("--grid",type=int,default=1024)
    parser.add_argument("--segments",type=int,default=32)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    source=ROOT/"public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240"
    config=json.loads((source/"synthetic_id57_summary.json").read_text())
    with np.load(source/"synthetic_id57_patterns.npz") as bank:
        valid=bank["valid_mask"].astype(bool)
        rotations=bank["rotation_matrices"][[0,144,238]].astype(float)
    yy,xx=np.indices(valid.shape)
    radial_px=np.hypot(xx-200,yy-200)
    roi=valid&(radial_px>=45)&(radial_px<=190)
    q_per_pixel=2*np.pi/config["wavelength_nm"]*config["pixel_size_m"]/config["detector_distance_m"]
    qaxis=(np.arange(401)-200)*q_per_pixel
    qmax=190*q_per_pixel
    n,extent=args.grid,256.
    coords=(np.arange(n)-(n-1)/2)*extent/n
    radii=[0.,0.5,1.,2.,4.,8.]
    contracts={"surface":"standalone scientific PNG/SVG for manuscript revision","renderer":"Matplotlib",
        "branding":"third-party research; no OpenAI branding","palette":{"blue":"#2878A8","orange":"#C77B24","reference":"#222222"},
        "question":"Does finite-q loss blur a sharp edge, and how much rounding remains visible in the ID57 ROI?",
        "figures":[
            {"file":"forward_inverse_edge_blur","family":"heatmap + line","data":"one sharp cube, 10 q cutoffs","units":"nm and nm^-1","qa":"inspect PNG; shared scales; known-phase caveat"},
            {"file":"rounded_cube_sensitivity","family":"line with markers","data":"6 radii x 3 orientations","units":"nm, percent, correlation","qa":"inspect PNG; markers and line styles encode orientation"}]}
    (args.out/"chart_contract.json").write_text(json.dumps(contracts,indent=2),encoding="utf-8")
    meshes,mesh_rows,projection_diagnostics={},[],[]
    for r in radii:
        mesh=rounded_cube(52.,r,args.segments)
        meshes[r]=mesh
        volume=KERNEL["signed_mesh_volume"](mesh.vertices,mesh.faces)
        mesh_rows.append({"radius_nm":r,"vertices":len(mesh.vertices),"faces":len(mesh.faces),
            "mesh_volume_nm3":volume,"exact_volume_nm3":exact_volume(52.,r),"volume_relative_error":volume/exact_volume(52.,r)-1})
    axes=np.vstack([np.eye(3),-np.eye(3)])
    octa=np.asarray(list(itertools.product([-1,1],repeat=3)))
    core=halfspace_mesh(np.vstack([axes,octa]),np.r_[np.full(6,26.),np.full(8,26.2)])
    sharp,diag=project(meshes[0.],np.eye(3),n,extent)
    analytic=52.*((np.abs(coords[:,None])<26)&(np.abs(coords[None,:])<26))
    assert rel_l2(sharp,analytic)<1e-12
    padded=np.pad(sharp,((n//2,n//2),(n//2,n//2)))
    fft=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    inverse=lambda amp:np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(amp)))
    full_error=rel_l2(inverse(fft),padded)
    assert full_error<1e-12
    fft_axis=2*np.pi*np.fft.fftshift(np.fft.fftfreq(2*n,d=extent/n))
    qr=np.hypot(fft_axis[:,None],fft_axis[None,:])
    padded_coords=(np.arange(2*n)-(2*n-1)/2)*extent/n
    roundtrip,selected=[],{}
    for cutoff in [0.15,0.2,0.3,0.4,0.6,0.8,float(qmax),1.68,2.4,4.]:
        mask=qr<=cutoff
        recovered=inverse(fft*mask)
        profile=recovered.real[n,:]/52.
        width=crossing(padded_coords,profile,0.1)-crossing(padded_coords,profile,0.9)
        roundtrip.append({"qmax_nm_inv":cutoff,"half_period_resolution_nm":np.pi/cutoff,
            "edge_10_90_width_nm":width,"density_relative_l2_error":rel_l2(recovered,padded),
            "retained_spectral_energy_fraction":float(np.sum(np.abs(fft[mask])**2)/np.sum(np.abs(fft)**2)),
            "mass_relative_error":float(recovered.real.sum()/padded.sum()-1),
            "relative_imaginary_norm":float(np.linalg.norm(recovered.imag)/np.linalg.norm(recovered.real))})
        if cutoff in [0.2,0.8,float(qmax)]: selected[cutoff]=recovered.real
    rows,intensity_cache,density_cache=[],{},{}
    for oi,rotation in enumerate(rotations):
        core_projection,_=project(core,rotation,n,extent)
        for r in radii:
            outer,diagnostic=project(meshes[r],rotation,n,extent)
            density=outer+0.5*core_projection
            intensity=forward_detector(density,coords,qaxis)
            if r==0:
                reference_raw=intensity
                conversion=(config["target_positive_total_from_experiment"]-8*int(valid.sum()))/intensity[valid].sum()
            reference,signal=reference_raw*conversion+8.,intensity*conversion+8.
            scores,scales,backgrounds,rms=SCORING["fit_scale_background_and_score"](signal,reference[None],roi)
            fit=scales[0]*reference+backgrounds[0]
            mass=float(density.sum()*(extent/n)**2)
            mesh_mass=KERNEL["signed_mesh_volume"](meshes[r].vertices,meshes[r].faces)+0.5*KERNEL["signed_mesh_volume"](core.vertices,core.faces)
            row={"orientation_index":[0,144,238][oi],"radius_nm":r,"qmax_radius_product":qmax*r,
                "intensity_relative_l2_raw":rel_l2(intensity[roi],reference_raw[roi]),
                "intensity_relative_l2_after_scale_background":rel_l2(fit[roi],signal[roi]),
                "log_correlation_after_scale_background":float(scores[0]),"standardized_log_residual_rms":float(rms[0]),
                "fit_scale":float(scales[0]),"fit_background_adu":float(backgrounds[0]),
                "projection_mass_relative_error_vs_mesh":mass/mesh_mass-1,"roi_pixels":int(roi.sum()),
                "qmin_nm_inv":45*q_per_pixel,"qmax_nm_inv":qmax}
            rows.append(row)
            projection_diagnostics.append({"orientation_index":row["orientation_index"],"radius_nm":r,**diagnostic})
            if oi==2 and r in [0.,2.]: density_cache[r],intensity_cache[r]=density,intensity
            print(f'orientation {row["orientation_index"]}, r={r:g}: fitted intensity difference {100*row["intensity_relative_l2_after_scale_background"]:.4f}%',flush=True)
    qa={"full_spectrum_roundtrip_relative_l2":full_error,
        "sharp_mesh_vs_exact_projection_relative_l2":rel_l2(sharp,analytic),
        "mesh_max_relative_volume_error":max(abs(v["volume_relative_error"]) for v in mesh_rows)}
    for r in [0.,2.]:
        fine_outer,_=project(meshes[r],rotations[2],2*n,extent)
        fine_core,_=project(core,rotations[2],2*n,extent)
        fine_coords=(np.arange(2*n)-(2*n-1)/2)*extent/(2*n)
        fine_intensity=forward_detector(fine_outer+0.5*fine_core,fine_coords,qaxis)
        qa[f"grid_doubling_intensity_relative_l2_r{r:g}"]=rel_l2(intensity_cache[r][roi],fine_intensity[roi])
    refined_mesh=rounded_cube(52.,2.,2*args.segments)
    refined_outer,_=project(refined_mesh,rotations[2],n,extent)
    refined_core,_=project(core,rotations[2],n,extent)
    refined_intensity=forward_detector(refined_outer+0.5*refined_core,coords,qaxis)
    qa["tessellation_doubling_intensity_relative_l2_r2"]=rel_l2(intensity_cache[2.][roi],refined_intensity[roi])

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"figure.facecolor":"white","savefig.facecolor":"white"})
    blue,orange,ink="#2878A8","#C77B24","#222222"
    fig,ax=plt.subplots(2,4,figsize=(14,7),layout="constrained")
    view=np.abs(padded_coords)<=38
    views=[padded,selected[0.2],selected[0.8],selected[qmax]]
    labels=["Original sharp projection","$q_{max}=0.20$ nm$^{-1}$","$q_{max}=0.80$ nm$^{-1}$",f"ID57 outer ROI: {qmax:.2f} nm$^{{-1}}$"]
    for i,(image,label) in enumerate(zip(views,labels)):
        ax[0,i].imshow(image[np.ix_(view,view)]/52,origin="lower",extent=[-38,38,-38,38],cmap="Blues",vmin=-0.15,vmax=1.15)
        ax[0,i].set(title=label,xlabel="u (nm)",ylabel="v (nm)")
        ax[1,i].plot(padded_coords,padded[n]/52,color=ink,lw=1.2,ls="--",label="Original")
        ax[1,i].plot(padded_coords,image[n]/52,color=blue,lw=1.8,label="Inverse transform")
        ax[1,i].set(xlim=(15,37),ylim=(-0.2,1.2),xlabel="u (nm)",ylabel="Thickness / 52 nm")
        ax[1,i].axvline(26,color="0.65",lw=0.8)
    ax[1,0].legend(fontsize=8)
    fig.suptitle("Sharp-edge recovery after a finite-q forward/inverse transform\n52 nm cube; circular cutoff; complex phase retained; Gibbs ringing is not clipped",fontsize=13)
    for suffix in ["png","svg"]: fig.savefig(args.out/f"forward_inverse_edge_blur.{suffix}",dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4.6),layout="constrained")
    for oi,(color,style,marker) in enumerate([(ink,"--","o"),(blue,"-","s"),(orange,"-.","^")]):
        subset=[row for row in rows if row["orientation_index"]==[0,144,238][oi]]
        label=f"Library orientation {[0,144,238][oi]}"
        for a,key,factor in [(ax[0],"intensity_relative_l2_after_scale_background",100),
            (ax[1],"log_correlation_after_scale_background",1),(ax[2],"standardized_log_residual_rms",1)]:
            a.plot(radii,[row[key]*factor for row in subset],color=color,ls=style,marker=marker,label=label)
            a.set_xlabel("Edge-rounding radius (nm)")
            a.grid(axis="y",color="0.9",lw=0.6)
    ax[0].set(title="Intensity difference after fitting",ylabel="Relative L2 difference (%)",ylim=(0,None))
    ax[1].set(title="Log-intensity correlation",ylabel="Correlation with sharp template")
    ax[2].set(title="Standardized log residual",ylabel="RMS",ylim=(0,None))
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Rounded outer cube sensitivity in the ID57 scoring ROI\n52 nm envelope, fixed clipped Au octahedral core; 45-190 px ({45*q_per_pixel:.3f}-{qmax:.3f} nm$^{{-1}}$); no noise",fontsize=12)
    for suffix in ["png","svg"]: fig.savefig(args.out/f"rounded_cube_sensitivity.{suffix}",dpi=220)
    plt.close(fig)
    save_csv(args.out/"roundtrip_metrics.csv",roundtrip)
    save_csv(args.out/"rounding_metrics.csv",rows)
    save_csv(args.out/"mesh_metrics.csv",mesh_rows)
    np.savez_compressed(args.out/"illustrated_arrays.npz",padded_coords_nm=padded_coords,
        sharp_density=padded,recovered_qmax=selected[qmax],projection_coords_nm=coords,
        sharp_core_shell=density_cache[0.],rounded_r2_core_shell=density_cache[2.],
        detector_q_axis_nm_inv=qaxis,sharp_intensity=intensity_cache[0.],
        rounded_r2_intensity=intensity_cache[2.],valid_mask=valid,roi_mask=roi,rotations=rotations)
    summary={"environment":{"python":platform.python_version(),"numpy":np.__version__,"matplotlib":matplotlib.__version__},
        "source_hashes":{"saxs_module.py":KERNEL_HASH,"section5_id57_retrieval.py":SCORING_HASH},
        "configuration":{"grid":n,"extent_nm":extent,"surface_arc_segments":args.segments,"radii_nm":radii,
            "orientation_indices":[0,144,238],"qmin_nm_inv":45*q_per_pixel,"qmax_nm_inv":qmax,"roi_pixels":int(roi.sum()),
            "kernel":"unchanged CPU signed triangle rasterization",
            "detector_transform":"separable direct discrete Fourier sum on the original detector q grid",
            "roundtrip_transform":"zero-padded FFT / inverse FFT with known phase and circular q cutoff"},
        "qa":qa,"roundtrip":roundtrip,"rounding":rows,"mesh":mesh_rows,
        "projection_diagnostics":projection_diagnostics,"elapsed_seconds":time.perf_counter()-start,
        "limitations":["Inverse-transform demonstration retains known phase and q=0; it is not reconstruction from measured intensity.",
            "Edge blur alone does not establish statistical indistinguishability of sharp and rounded shapes.",
            "Only three existing orientations are sampled; sensitivity is synthetic, noiseless, and conditional on this model and ROI.",
            "Finite-q loss and high-q density-representation limitations are distinct; no atomistic benchmark was run.",
            "Ewald curvature, radiation damage, detector PSF, unknown background and geometry calibration are excluded.",
            "Density boundaries remain sharp; Fourier sampling is limited, not the source mesh."]}
    (args.out/"summary.json").write_text(json.dumps(json_safe(summary),indent=2,allow_nan=False),encoding="utf-8")
    (args.out/"README.txt").write_text(
        "Finite-q resolution validation, 2026-10-09\n\n"
        "Run: .venv_cxidb/Scripts/python.exe validation/revision_resolution_validation.py\n"
        "Outputs: known-phase edge-blur figure; noiseless rounded-cube sensitivity; CSV metrics; JSON QA and source hashes; arrays.\n"
        "Angular q convention: exp(-i q.r), q in nm^-1. pi/qmax is a half-period scale, not the measured 10-90 edge width.\n"
        "Rounded cube = inner cube of side 52-2r plus a sphere of radius r; total outer dimensions stay 52 nm.\n"
        "Core: |x|+|y|+|z|<=26.2 nm and |coordinate|<=26 nm, rho outer=1, core increment=0.5.\n"
        "Scoring: unchanged original scale/background/log-correlation functions, original mask, annulus 45-190 px.\n"
        "ADU scale and 8 ADU background are synthetic scoring conventions, not calibrated photon counts or a noise model.\n"
        "Complex phases are retained. Inverse transforming intensity would yield Patterson/autocorrelation, not this density.\n"
        "An edge can blur yet leave detectable diffraction differences. Read rounding_metrics.csv separately.\n"
        "Check grid/tessellation convergence in summary.json before citing small differences. No experimental rounded-particle fit or high-q extension is claimed.\n",encoding="utf-8")
    print(json.dumps(qa,indent=2),flush=True)
    print(f"Saved {args.out}; elapsed {summary['elapsed_seconds']:.1f} s",flush=True)

if __name__=="__main__":
    main()

