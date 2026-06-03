# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
)


project_dir = Path(SPECPATH)
build_cache = project_dir / ".pyinstaller_runtime_cache"
build_cache.mkdir(parents=True, exist_ok=True)
(build_cache / "matplotlib").mkdir(parents=True, exist_ok=True)
(build_cache / "cupy").mkdir(parents=True, exist_ok=True)
(build_cache / "pyvista").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(build_cache / "matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR", str(build_cache / "cupy"))
os.environ.setdefault("PYVISTA_USERDATA_PATH", str(build_cache / "pyvista"))

env_root = Path(sys.prefix)

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
    *gui_modules,
]

for package in [
    "cupy",
    "cupy_backends",
    "cuda",
    "nvidia",
    "open3d",
    "pyvista",
    "vtk",
    "vtkmodules",
    "skimage",
    "scipy",
    "matplotlib",
    "PIL",
]:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception:
        try:
            datas += collect_data_files(package)
            binaries += collect_dynamic_libs(package)
        except Exception:
            pass

library_bin = env_root / "Library" / "bin"
if library_bin.exists():
    binaries += [(str(path), ".") for path in library_bin.glob("*.dll")]

library_lib = env_root / "Library" / "lib"
for tcl_dir_name in ["tcl8", "tcl8.6", "tk8.6", "dde1.4", "reg1.3", "itcl4.3.0"]:
    tcl_dir = library_lib / tcl_dir_name
    if tcl_dir.exists():
        datas.append((str(tcl_dir), f"tcl/{tcl_dir_name}"))


a = Analysis(
    ["simulator.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SAXS_Simulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name="SAXS_Simulator",
)
