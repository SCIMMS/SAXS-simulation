"""Shared, source-preserving helpers for the gui5014 remaining revision experiments."""
from __future__ import annotations
import csv, hashlib, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"output/revision_remaining_2026-10-09"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".runtime_cache/matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR",str(ROOT/".runtime_cache/cupy"))
_DLL_HANDLES=[]
os.environ["CUDA_PATH"]=str(ROOT/".venv_cxidb/Lib/site-packages/nvidia/cuda_runtime")
for base in [ROOT/".venv_cxidb/Lib/site-packages/nvidia"]:
    for p in sorted(base.glob("*/bin")):
        if p.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(p)))
            os.environ["PATH"]=str(p)+os.pathsep+os.environ.get("PATH","")
p=ROOT/"dist/SAXS_Simulator/_internal/nvidia/cublas/bin"
if p.exists():
    _DLL_HANDLES.append(os.add_dll_directory(str(p)))
    os.environ["PATH"]=str(p)+os.pathsep+os.environ.get("PATH","")
sys.path.insert(0,str(ROOT/"uploaded_SAXS_simulator"))
sys.path.insert(0,str(ROOT/"validation"))
import numpy as np
import h5py
import saxs_module as sm
import revision_resolution_validation as resolution
import section5_id57_retrieval as idscore
import section5_generate_id57_core_shell_test_images as idforward
import section5_generate_groel_model_ab_preview as groel
import section5_id57_make_random_control_npz as control

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk:=f.read(4*1024*1024): h.update(chunk)
    return h.hexdigest()
def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,np.ndarray): return safe(v.tolist())
    if isinstance(v,np.generic): return safe(v.item())
    if isinstance(v,Path): return str(v)
    if isinstance(v,float) and not np.isfinite(v): return None
    return v
def dump(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2),encoding="utf-8")
def save_csv(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(safe(rows))
def rel_l2(value,ref):
    return float(np.linalg.norm(np.asarray(value)-ref)/max(np.linalg.norm(ref),1e-30))
def project_mesh(mesh,rotation,n,extent,backend="gpu"):
    rotated=SimpleNamespace(vertices=mesh.vertices@rotation.T,faces=mesh.faces,face_normals=mesh.face_normals@rotation.T)
    func=sm._signed_thickness_from_rasterization_gpu if backend=="gpu" else sm._signed_thickness_from_rasterization_cpu
    return func(rotated,u=np.array([1.,0.,0.]),v=np.array([0.,1.,0.]),n=np.array([0.,0.,1.]),
        uv_bounds=(-extent/2,extent/2,-extent/2,extent/2),image_shape=(n,n))
def id57_data():
    folder=ROOT/"public_data/CXIDB_ID_57/synthetic_id57_core_shell_v3_uniform_n0240"
    cfg=json.loads((folder/"synthetic_id57_summary.json").read_text())
    with np.load(folder/"synthetic_id57_patterns.npz") as f: data={k:f[k] for k in f.files if k!="noisy_patterns"}
    with h5py.File(ROOT/"public_data/CXIDB_ID_57/normal-incidence-pattern.cxi") as f:
        exp=f[idforward.DATA_LINK_PATH][...].astype(np.float32)
    data["valid_mask"]=data["valid_mask"].astype(bool)
    data["roi"]=data["valid_mask"]&idscore.build_annular_mask(data["valid_mask"].shape,45,190)
    qstep=2*np.pi/cfg["wavelength_nm"]*cfg["pixel_size_m"]/cfg["detector_distance_m"]
    data["qaxis"]=(np.arange(401)-200)*qstep
    data["parameter_rows"]=list(csv.DictReader((folder/"synthetic_id57_parameters.csv").open(encoding="utf-8")))
    return cfg,data,exp
def id57_meshes(side=52,core_radius=26.2,rounding=0,segments=16):
    axes=np.vstack([np.eye(3),-np.eye(3)])
    import itertools
    octa=np.asarray(list(itertools.product([-1,1],repeat=3)))
    outer=resolution.rounded_cube(side,rounding,segments)
    core=resolution.halfspace_mesh(np.vstack([axes,octa]),np.r_[np.full(6,side/2),np.full(8,core_radius)])
    return outer,core
def id57_projection(rotation,n=768,extent=144.,meshes=None,backend="gpu"):
    outer,core=meshes or id57_meshes()
    p,d=project_mesh(outer,rotation,n,extent,backend)
    c,dc=project_mesh(core,rotation,n,extent,backend)
    return p+.5*c, {"outer":d,"core":dc}
def czt_intensity(p,extent,qaxis):
    axis=(np.arange(p.shape[0])-(p.shape[0]-1)/2)*extent/p.shape[0]
    return np.abs(sm.czt2_from_real_axes(p,axis,axis,qaxis,qaxis))**2
def clean_id57(raw,cfg,valid):
    raw=np.clip(raw,0,None)
    signal_total=cfg["target_positive_total_from_experiment"]-cfg["background_adu"]*int(valid.sum())
    scaled=raw*(signal_total/max(float(raw[valid].sum()),1e-30))+cfg["background_adu"]
    scaled[~valid]=0
    return scaled.astype(np.float32)
