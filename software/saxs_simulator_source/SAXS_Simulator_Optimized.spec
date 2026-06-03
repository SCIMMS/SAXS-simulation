# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


project_dir = Path(SPECPATH)
env_root = Path(sys.prefix)
site_packages = env_root / "Lib" / "site-packages"

build_cache = project_dir / ".pyinstaller_runtime_cache"
(build_cache / "matplotlib").mkdir(parents=True, exist_ok=True)
(build_cache / "cupy").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(build_cache / "matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR", str(build_cache / "cupy"))

gui_modules = [
    "gui_ellipsoid1",
    "gui_ellipsoid1_oop",
    "gui_ellipsoid2",
    "gui_ellipsoid3",
    "gui_polyhedron",
    "gui_polyhedron_ellipsoid1",
    "gui_polyhedron_ellipsoid2",
    "gui_random",
    "gui_random_ellipsoid1",
    "gui_random_ellipsoid2",
    "gui_coned_droplet",
    "gui_coned_droplet_ellipsoid1",
    "gui_coned_droplet_ellipsoid2",
    "gui_coned_droplet_polyhedron_ellipsoid",
    "gui_double_coned_droplet",
    "gui_double_coned_droplet_ellipsoid1",
    "gui_double_coned_droplet_ellipsoid2",
    "gui_double_coned_droplet_polyhedron",
    "gui_double_coned_droplet_polyhedron_ellipsoid",
]

datas = [
    (str(project_dir / "SAXS_mask.npy"), "."),
    (str(project_dir / "post_beamtime_tmp_saxs.poni"), "."),
    (str(project_dir / "cimg001.npz"), "."),
    (str(project_dir / "ACCEPTANCE_GATE_CHECKBOXES_NOTES.txt"), "."),
    (str(project_dir / "ANISOTROPY_ACCEPTANCE_NOTES.txt"), "."),
]
binaries = []
hiddenimports = [
    "graphlib",
    "saxs_module",
    "matplotlib.backends.backend_tkagg",
    "PIL.ImageTk",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "scipy.signal",
    "scipy.ndimage",
    "scipy.spatial",
    "skimage.measure",
    "skimage.metrics",
    *gui_modules,
]

def _not_test_module(name: str) -> bool:
    parts = name.split(".")
    last = parts[-1]
    return "tests" not in parts and last != "conftest" and not last.startswith("test_")

for package in ["cupy", "cupy_backends", "cupyx"]:
    hiddenimports += collect_submodules(package, filter=_not_test_module)
    datas += collect_data_files(
        package,
        includes=["**/*.cu", "**/*.cuh", "**/*.h", "**/*.pxd", "**/*.dll", "**/*.manifest"],
        excludes=["**/tests/**", "**/testing/**", "**/__pycache__/**"],
    )
    binaries += collect_dynamic_libs(package)

for package in ["skimage", "scipy"]:
    datas += collect_data_files(package, excludes=["**/tests/**", "**/__pycache__/**"])
    binaries += collect_dynamic_libs(package)

# RawKernel needs the CUDA runtime and NVRTC compiler stack, but not BLAS/FFT/Solver.
for component in ["cuda_runtime", "cuda_nvrtc", "nvjitlink"]:
    component_dir = site_packages / "nvidia" / component
    if component_dir.exists():
        datas.append((str(component_dir), f"nvidia/{component}"))

library_bin = env_root / "Library" / "bin"
if library_bin.exists():
    binaries += [(str(path), ".") for path in library_bin.glob("*.dll")]

excludes = [
    "open3d",
    "pyvista",
    "vtk",
    "vtkmodules",
    "dash",
    "plotly",
    "IPython",
    "ipywidgets",
    "jedi",
    "notebook",
    "pytest",
    "torch",
]

a = Analysis(
    ["simulator.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["TkAgg"]}},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SAXS_Simulator_Optimized",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SAXS_Simulator_Optimized",
)
