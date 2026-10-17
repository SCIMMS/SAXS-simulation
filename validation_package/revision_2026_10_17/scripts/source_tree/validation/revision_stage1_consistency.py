"""Stage 1: source branches separated into projection and Fourier-sampling errors."""
from revision_common import *
import time
stage=OUT/"01_consistency";stage.mkdir(exist_ok=True)
cfg,data,exp=id57_data()
meshes=id57_meshes(); rows=[]
for ti in [0,144,238]:
    rot=data["rotation_matrices"][ti].astype(float)
    start=time.perf_counter()
    oldp,_=idforward.project_core_shell_density(rot,384,144,256,120,52,26.2,1,1.5,24)
    oldi=idforward.projected_density_to_detector_intensity(oldp,144/384,data["qx_nm_inv"],data["qy_nm_inv"])
    oldclean=clean_id57(oldi,cfg,data["valid_mask"])
    old_czt=czt_intensity(oldp,144,data["qaxis"])
    gpu_p,diag=id57_projection(rot,384,144,meshes,"gpu")
    cpu_p,_=id57_projection(rot,384,144,meshes,"cpu")
    newi=czt_intensity(gpu_p,144,data["qaxis"])
    finep,fdiag=id57_projection(rot,768,144,meshes,"gpu")
    finei=czt_intensity(finep,144,data["qaxis"])
    m=data["roi"]
    row={"template":ti,"elapsed_s":time.perf_counter()-start,
        "original_clean_regeneration_l2":rel_l2(oldclean,data["clean_patterns"][ti]),
        "gpu_cpu_projection_l2":rel_l2(gpu_p,cpu_p),
        "projection_raymarch_mesh_l2":rel_l2(gpu_p,oldp),
        "transform_fft_intensity_interpolation_vs_czt_l2_roi":rel_l2(oldi[m],old_czt[m]),
        "projection_only_intensity_l2_roi":rel_l2(newi[m],old_czt[m]),
        "combined_old_new_l2_roi":rel_l2(newi[m],oldi[m]),
        "mesh_grid_384_768_intensity_l2_roi":rel_l2(newi[m],finei[m]),
        "gpu_mass_relative_error":float(gpu_p.sum()*(144/384)**2/(sm.signed_mesh_volume(meshes[0].vertices,meshes[0].faces)+.5*sm.signed_mesh_volume(meshes[1].vertices,meshes[1].faces))-1),
        "bad_balance_pixels":diag["outer"]["bad_balance_pixels"]+diag["core"]["bad_balance_pixels"]}
    rows.append(row); print(json.dumps(safe(row)),flush=True)
    np.savez_compressed(stage/f"id57_pilot_{ti}.npz",raymarch_projection=oldp,mesh_projection=gpu_p,
        old_intensity=oldi,raymarch_czt_intensity=old_czt,mesh_czt_intensity=newi,mesh_fine_intensity=finei,roi=m)
save_csv(stage/"id57_pilot_metrics.csv",rows)
dump(stage/"id57_pilot_summary.json",{"rows":rows,"source_preserved":True,"backend":"unaltered production GPU raster and CPU complex128 CZT",
    "caution":"Legacy ID57 interpolates intensity on a coarse FFT grid; same q ROI and physical model used."})
