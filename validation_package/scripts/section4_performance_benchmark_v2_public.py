# ============================================================
# Section 4 performance benchmark data acquisition
# ============================================================
# Assumption:
#   - The main notebook cell has already been executed.
#   - In particular, the following functions/classes from cell.txt are available:
#       StructureGeometry
#       project_density
#       projection_basis_from_ray
#       compute_face_normals
#       orient_faces_outward_by_volume
#       signed_mesh_volume
#       czt2_from_real_axes
#       build_structure_geometry
#
# Purpose:
#   Generate benchmark data for:
#       4.1 Runtime scaling with detector-plane image size and mesh complexity
#       4.2 Throughput for orientation-library generation
#       4.3 Mesh-generation benchmark for droplet morphology models
#       4.4 Memory/computational cost comparison with dense voxel/FFT approaches
#
# Public-release scope:
#   This script benchmarks analytical and synthetic public-safe morphologies:
#   sphere, ellipsoid, multiple droplets, polyhedra, and Gaussian blobs.
#   It intentionally excludes non-public deformation classes and experimental
#   detector data.
#
# Recommended Colab usage after executing the main simulator cell:
#   exec(open("section4_performance_benchmark_v2_public.py").read())
#   results = run_all_section4_benchmarks(
#       out_root="section4_performance_outputs_public_quick",
#       prefer_gpu=True,
#       quick=True,
#   )
#
# For a more complete benchmark:
#   results = run_all_section4_benchmarks(
#       out_root="section4_performance_outputs_public_full",
#       prefer_gpu=True,
#       quick=False,
#   )
# ============================================================

from __future__ import annotations

import gc
import json
import platform
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "section4_public_v2"


# ------------------------------------------------------------
# Environment / backend helpers
# ------------------------------------------------------------
def check_required_symbols() -> None:
    required = [
        "StructureGeometry",
        "project_density",
        "projection_basis_from_ray",
        "compute_face_normals",
        "orient_faces_outward_by_volume",
        "signed_mesh_volume",
        "czt2_from_real_axes",
        "build_structure_geometry",
    ]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            "Required symbols are missing from the current namespace. "
            "Execute the main cell.txt code first, then run this benchmark file. "
            f"Missing: {missing}"
        )


def has_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def get_cupy():
    try:
        import cupy as cp
        return cp
    except Exception:
        return None


def choose_projection_method(prefer_gpu: bool = True) -> str:
    if prefer_gpu and has_cupy():
        return "signed_raster_gpu"
    return "signed_raster"


def sync_if_gpu() -> None:
    cp = get_cupy()
    if cp is not None:
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass


def get_gpu_info() -> Dict:
    info = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory_total_MB": None,
        "gpu_memory_free_MB": None,
        "cupy_memory_pool_used_MB": None,
    }
    cp = get_cupy()
    if cp is None:
        return info
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        info.update(
            {
                "gpu_available": True,
                "gpu_name": name,
                "gpu_memory_total_MB": float(total_b / 1024**2),
                "gpu_memory_free_MB": float(free_b / 1024**2),
                "cupy_memory_pool_used_MB": float(cp.get_default_memory_pool().used_bytes() / 1024**2),
            }
        )
    except Exception as exc:
        info["gpu_error"] = repr(exc)
    return info


def package_version(package_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def collect_environment_metadata(prefer_gpu: bool, quick: bool) -> Dict:
    cp = get_cupy()
    cupy_runtime = {
        "available": cp is not None,
        "version": getattr(cp, "__version__", None) if cp is not None else None,
        "cuda_runtime_version": None,
    }
    if cp is not None:
        try:
            cupy_runtime["cuda_runtime_version"] = int(cp.cuda.runtime.runtimeGetVersion())
        except Exception:
            pass

    return {
        "script_version": SCRIPT_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "projection_method": choose_projection_method(prefer_gpu=prefer_gpu),
        "prefer_gpu": bool(prefer_gpu),
        "quick": bool(quick),
        "gpu_info": get_gpu_info(),
        "cupy_runtime": cupy_runtime,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cupy": cupy_runtime["version"] or package_version("cupy"),
            "cupy-cuda12x": package_version("cupy-cuda12x"),
            "cupy-cuda11x": package_version("cupy-cuda11x"),
            "open3d": package_version("open3d"),
            "scipy": package_version("scipy"),
            "scikit-image": package_version("scikit-image"),
            "vtk": package_version("vtk"),
        },
        "public_release_scope": (
            "Performance benchmarks use synthetic public-safe morphologies only: "
            "sphere, ellipsoid, multiple droplets, polyhedra, and Gaussian blobs. "
            "Private deformation classes and experimental detector data are excluded."
        ),
    }


def reset_gpu_memory_pool() -> None:
    cp = get_cupy()
    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def ensure_output_dirs(root: str | Path = "section4_performance_outputs") -> Dict[str, Path]:
    root = Path(root)
    out = {
        "root": root,
        "csv": root / "summaries",
        "json": root / "json",
        "npz": root / "npz",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, obj: Dict) -> None:
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, complex):
            return {"real": float(np.real(x)), "imag": float(np.imag(x))}
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv(obj), f, ensure_ascii=False, indent=2)


def uv_axes_from_projection_meta(meta: Dict, du: float, dv: float) -> Tuple[np.ndarray, np.ndarray]:
    H, W = meta["image_shape"]
    u0 = float(meta["x0_unit"])
    v0 = float(meta["y0_unit"])
    u_axis = u0 + np.arange(W, dtype=np.float64) * float(du)
    v_axis = v0 + np.arange(H, dtype=np.float64) * float(dv)
    return u_axis, v_axis


def random_unit_vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(int(n), 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def random_rolls(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=int(n))


# ------------------------------------------------------------
# Mesh helpers
# ------------------------------------------------------------
def mesh_to_geom(
    vertices: np.ndarray,
    faces: np.ndarray,
    structure_type: str = "custom-mesh",
    params: Optional[Dict] = None,
    mesh_meta_extra: Optional[Dict] = None,
) -> "StructureGeometry":
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    faces, vol, flipped = orient_faces_outward_by_volume(vertices, faces)
    normals = compute_face_normals(vertices, faces)

    mesh_meta = {
        "mesh_source": "section4_benchmark_helper",
        "mesh_vertex_count": int(len(vertices)),
        "mesh_face_count": int(len(faces)),
        "signed_volume": float(vol),
        "faces_flipped_for_positive_volume": bool(flipped),
    }
    if mesh_meta_extra:
        mesh_meta.update(mesh_meta_extra)

    return StructureGeometry(
        structure_type=structure_type,
        params={} if params is None else dict(params),
        vertices=vertices,
        faces=faces,
        face_normals=normals,
        mesh_meta=mesh_meta,
        scene=None,
        geometry_id=None,
    )


def rotation_matrix_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def make_sphere_mesh(
    radius: float = 1.0,
    resolution: int = 64,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> "StructureGeometry":
    import open3d as o3d

    t0 = time.perf_counter()
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=int(resolution))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    vertices += np.asarray(center, dtype=np.float64)[None, :]
    total = time.perf_counter() - t0

    return mesh_to_geom(
        vertices,
        faces,
        structure_type="sphere-direct",
        params={"radius": float(radius), "resolution": int(resolution)},
        mesh_meta_extra={
            "mesh_generation_route": "open3d_direct_sphere",
            "total_mesh_generation_time_s": float(total),
            "field_generation_time_s": 0.0,
            "surface_extraction_time_s": 0.0,
        },
    )


def make_ellipsoid_mesh(
    axes: Tuple[float, float, float] = (1.0, 0.65, 0.42),
    resolution: int = 64,
    rotation: Optional[np.ndarray] = None,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> "StructureGeometry":
    import open3d as o3d

    t0 = time.perf_counter()
    if rotation is None:
        rotation = np.eye(3, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=int(resolution))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    vertices = vertices * axes[None, :]
    vertices = vertices @ np.asarray(rotation, dtype=np.float64).T + center[None, :]
    total = time.perf_counter() - t0

    return mesh_to_geom(
        vertices,
        faces,
        structure_type="ellipsoid-direct",
        params={"axes": axes.tolist(), "resolution": int(resolution)},
        mesh_meta_extra={
            "mesh_generation_route": "open3d_sphere_scaled_to_ellipsoid",
            "total_mesh_generation_time_s": float(total),
            "field_generation_time_s": 0.0,
            "surface_extraction_time_s": 0.0,
        },
    )


def append_meshes(meshes: List["StructureGeometry"], structure_type: str = "multi-mesh") -> "StructureGeometry":
    t0 = time.perf_counter()
    vertices_all = []
    faces_all = []
    offset = 0
    for geom in meshes:
        v = np.asarray(geom.vertices, dtype=np.float64)
        f = np.asarray(geom.faces, dtype=np.int32) + offset
        vertices_all.append(v)
        faces_all.append(f)
        offset += len(v)
    vertices = np.vstack(vertices_all)
    faces = np.vstack(faces_all)
    total = time.perf_counter() - t0
    return mesh_to_geom(
        vertices,
        faces,
        structure_type=structure_type,
        mesh_meta_extra={
            "mesh_generation_route": "component_mesh_assembly",
            "component_count": int(len(meshes)),
            "total_mesh_generation_time_s": float(total + sum(float(m.mesh_meta.get("total_mesh_generation_time_s", 0.0)) for m in meshes)),
            "field_generation_time_s": 0.0,
            "surface_extraction_time_s": 0.0,
            "assembly_time_s": float(total),
        },
    )


def make_two_droplet_mesh(
    radius1: float = 1.0,
    radius2: float = 0.8,
    separation: float = 2.4,
    resolution: int = 48,
) -> "StructureGeometry":
    g1 = make_sphere_mesh(radius=radius1, resolution=resolution, center=(-0.5 * separation, 0.0, 0.0))
    g2 = make_sphere_mesh(radius=radius2, resolution=resolution, center=(+0.5 * separation, 0.0, 0.0))
    geom = append_meshes([g1, g2], structure_type="two-droplet-direct")
    geom.params = {
        "radius1": float(radius1),
        "radius2": float(radius2),
        "separation": float(separation),
        "resolution": int(resolution),
    }
    return geom


def make_gaussian_blob_mesh(
    field_shape: Tuple[int, int, int] = (120, 120, 120),
    threshold: float = 0.55,
) -> "StructureGeometry":
    """Build a deterministic smooth non-analytical morphology from Gaussian kernels."""
    t0 = time.perf_counter()
    kernel_centers = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.42, 0.12, 0.18],
            [-0.35, 0.28, -0.10],
            [0.08, -0.45, 0.26],
            [-0.18, -0.22, -0.36],
            [0.46, -0.18, -0.20],
        ],
        dtype=np.float64,
    )
    kernel_sigmas = np.array(
        [
            [0.42, 0.34, 0.30],
            [0.30, 0.24, 0.38],
            [0.26, 0.38, 0.28],
            [0.36, 0.22, 0.30],
            [0.24, 0.34, 0.40],
            [0.28, 0.28, 0.24],
        ],
        dtype=np.float64,
    )
    kernel_weights = np.array([1.00, 0.82, 0.74, 0.70, 0.62, 0.58], dtype=np.float64)

    geom = build_structure_geometry(
        "blob",
        build_scene=False,
        field_shape=tuple(field_shape),
        field_sampling="auto",
        iso_extractor="vtk_flying_edges",
        field_dtype="float32",
        kernel_centers=kernel_centers,
        kernel_sigmas=kernel_sigmas,
        kernel_weights=kernel_weights,
        threshold=float(threshold),
        rotations=[([0, 0, 1], 0.35), ([0, 1, 0], -0.22)],
    )
    total = time.perf_counter() - t0
    geom.mesh_meta["total_mesh_generation_time_s"] = float(total)
    geom.mesh_meta["mesh_generation_route"] = "gaussian_blob_local_grid_plus_vtk_flying_edges"
    return geom


def make_polyhedron_mesh(name: str = "icosa", radius: float = 1.0) -> "StructureGeometry":
    t0 = time.perf_counter()
    geom = build_structure_geometry(
        "polyhedron",
        build_scene=False,
        use_direct_polyhedron_mesh=True,
        name=str(name),
        radius=float(radius),
    )
    total = time.perf_counter() - t0
    geom.mesh_meta["total_mesh_generation_time_s"] = float(total)
    geom.mesh_meta["mesh_generation_route"] = "predefined_vertices_plus_oriented_convex_hull"
    return geom


# ------------------------------------------------------------
# Model factory collection
# ------------------------------------------------------------
def make_benchmark_models(
    quick: bool = True,
) -> Dict[str, Callable[[], "StructureGeometry"]]:
    blob_shape = (96, 96, 96) if quick else (140, 140, 140)
    models = {
        "sphere": lambda: make_sphere_mesh(radius=1.0, resolution=64 if quick else 96),
        "ellipsoid": lambda: make_ellipsoid_mesh(
            axes=(1.0, 0.65, 0.42),
            resolution=64 if quick else 96,
            rotation=rotation_matrix_from_euler_xyz(0.33, -0.51, 0.27),
        ),
        "two_droplet": lambda: make_two_droplet_mesh(
            radius1=1.0,
            radius2=0.75,
            separation=2.5,
            resolution=40 if quick else 64,
        ),
        "icosahedron": lambda: make_polyhedron_mesh(name="icosa", radius=1.0),
        "gaussian_blob": lambda: make_gaussian_blob_mesh(field_shape=blob_shape),
    }
    return models


# ------------------------------------------------------------
# Timing functions
# ------------------------------------------------------------
def timed_project_and_czt(
    geom: "StructureGeometry",
    ray_vector: Sequence[float],
    roll_rad: float,
    image_shape: Tuple[int, int] = (512, 512),
    q_shape: Tuple[int, int] = (257, 257),
    qmax: float = 25.0,
    prefer_gpu: bool = True,
    margin_ratio: float = 0.05,
    keep_arrays: bool = False,
) -> Dict:
    method = choose_projection_method(prefer_gpu=prefer_gpu)
    q_H, q_W = int(q_shape[0]), int(q_shape[1])
    if q_H % 2 == 0:
        q_H += 1
    if q_W % 2 == 0:
        q_W += 1

    q_u = np.linspace(-float(qmax), float(qmax), q_W, dtype=np.float64)
    q_v = np.linspace(-float(qmax), float(qmax), q_H, dtype=np.float64)

    reset_gpu_memory_pool()
    gc.collect()

    gpu_before = get_gpu_info()
    sync_if_gpu()
    t0 = time.perf_counter()
    proj, du, dv, meta = project_density(
        geom,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
        image_shape=tuple(image_shape),
        margin_ratio=float(margin_ratio),
        projection_method=method,
        out_dtype="float64",
        edge_rule="top_left",
        supersample=1,
    )
    sync_if_gpu()
    t1 = time.perf_counter()

    u_axis, v_axis = uv_axes_from_projection_meta(meta, du, dv)
    A = czt2_from_real_axes(proj, u_axis, v_axis, q_u, q_v)
    t2 = time.perf_counter()
    I = np.abs(A) ** 2

    gpu_after = get_gpu_info()
    diag = meta.get("projection_diagnostics", {})

    projection_time = float(t1 - t0)
    czt_time = float(t2 - t1)
    total_time = float(t2 - t0)

    result = {
        "meta": meta,
        "row": {
            "projection_method": method,
            "image_H": int(image_shape[0]),
            "image_W": int(image_shape[1]),
            "q_H": int(q_H),
            "q_W": int(q_W),
            "qmax": float(qmax),
            "mesh_vertices": int(len(geom.vertices)),
            "mesh_faces": int(len(geom.faces)),
            "mesh_volume": float(abs(signed_mesh_volume(geom.vertices, geom.faces))),
            "projection_time_s": projection_time,
            "czt_time_s": czt_time,
            "total_time_s": total_time,
            "patterns_per_second": float(1.0 / total_time) if total_time > 0 else float("nan"),
            "num_projected_fragments": int(diag.get("num_fragments", -1)),
            "num_nonzero_pixels": int(diag.get("num_nonzero_pixels", -1)),
            "rasterized_triangles": int(diag.get("rasterized_triangles", -1)),
            "bad_balance_pixels": int(diag.get("bad_balance_pixels", -1)),
            "gpu_available": bool(gpu_after.get("gpu_available", False)),
            "gpu_name": gpu_after.get("gpu_name", None),
            "gpu_memory_total_MB": gpu_after.get("gpu_memory_total_MB", None),
            "gpu_memory_free_before_MB": gpu_before.get("gpu_memory_free_MB", None),
            "gpu_memory_free_after_MB": gpu_after.get("gpu_memory_free_MB", None),
            "cupy_memory_pool_used_MB": gpu_after.get("cupy_memory_pool_used_MB", None),
        },
    }
    if keep_arrays:
        result.update(
            {
                "projected_density": proj,
                "amplitude": A,
                "intensity": I,
                "q_u": q_u,
                "q_v": q_v,
                "u_axis": u_axis,
                "v_axis": v_axis,
            }
        )
    return result


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def timed_project_and_czt_repeated(
    geom: "StructureGeometry",
    ray_vector: Sequence[float],
    roll_rad: float,
    image_shape: Tuple[int, int] = (512, 512),
    q_shape: Tuple[int, int] = (257, 257),
    qmax: float = 25.0,
    prefer_gpu: bool = True,
    margin_ratio: float = 0.05,
    warmup: int = 2,
    repeats: int = 5,
) -> Dict:
    warmup = max(0, int(warmup))
    repeats = max(1, int(repeats))

    for _ in range(warmup):
        timed_project_and_czt(
            geom,
            ray_vector=ray_vector,
            roll_rad=roll_rad,
            image_shape=image_shape,
            q_shape=q_shape,
            qmax=qmax,
            prefer_gpu=prefer_gpu,
            margin_ratio=margin_ratio,
        )

    repeat_rows = []
    for repeat_id in range(repeats):
        result = timed_project_and_czt(
            geom,
            ray_vector=ray_vector,
            roll_rad=roll_rad,
            image_shape=image_shape,
            q_shape=q_shape,
            qmax=qmax,
            prefer_gpu=prefer_gpu,
            margin_ratio=margin_ratio,
        )
        row = dict(result["row"])
        row["repeat_id"] = int(repeat_id)
        repeat_rows.append(row)

    summary = dict(repeat_rows[-1])
    summary.pop("repeat_id", None)
    summary["warmup_count"] = int(warmup)
    summary["repeat_count"] = int(repeats)

    for field in ("projection_time_s", "czt_time_s", "total_time_s", "patterns_per_second"):
        stats = summarize_values(float(row[field]) for row in repeat_rows)
        for stat_name, stat_value in stats.items():
            summary[f"{field}_{stat_name}"] = stat_value
        summary[field] = stats["median"]

    return {"row": summary, "repeat_rows": repeat_rows}


# ------------------------------------------------------------
# 4.1 Runtime scaling with image size and mesh complexity
# ------------------------------------------------------------
def run_projection_scaling_benchmark(
    out_root: str | Path = "section4_performance_outputs",
    prefer_gpu: bool = True,
    quick: bool = True,
    warmup: Optional[int] = None,
    repeats: Optional[int] = None,
) -> pd.DataFrame:
    out = ensure_output_dirs(out_root)
    rows = []
    repeat_rows = []

    image_sizes = [(256, 256), (512, 512)] if quick else [(256, 256), (512, 512), (1024, 1024)]
    q_shape = (129, 129) if quick else (257, 257)
    warmup = 1 if warmup is None and quick else 2 if warmup is None else int(warmup)
    repeats = 3 if repeats is None and quick else 5 if repeats is None else int(repeats)

    models = make_benchmark_models(quick=quick)
    for model_name, factory in models.items():
        try:
            geom = factory()
        except Exception as exc:
            rows.append({"benchmark": "image_size_scaling", "model_class": model_name, "status": "failed_model_generation", "error": repr(exc)})
            continue

        for image_shape in image_sizes:
            try:
                result = timed_project_and_czt_repeated(
                    geom,
                    ray_vector=(0.25, -0.2, 1.0),
                    roll_rad=0.17,
                    image_shape=image_shape,
                    q_shape=q_shape,
                    qmax=25.0,
                    prefer_gpu=prefer_gpu,
                    warmup=warmup,
                    repeats=repeats,
                )
                row = dict(result["row"])
                row.update(
                    {
                        "benchmark": "image_size_scaling",
                        "model_class": model_name,
                        "status": "ok",
                    }
                )
                rows.append(row)
                for repeat_row in result["repeat_rows"]:
                    rr = dict(repeat_row)
                    rr.update(
                        {
                            "benchmark": "image_size_scaling",
                            "model_class": model_name,
                            "status": "ok",
                        }
                    )
                    repeat_rows.append(rr)
            except Exception as exc:
                rows.append(
                    {
                        "benchmark": "image_size_scaling",
                        "model_class": model_name,
                        "image_H": image_shape[0],
                        "image_W": image_shape[1],
                        "status": "failed_projection",
                        "error": repr(exc),
                    }
                )

    sphere_resolutions = [8, 16, 32, 64] if quick else [8, 16, 32, 64, 96, 128]
    for res in sphere_resolutions:
        geom = make_sphere_mesh(radius=1.0, resolution=res)
        try:
            result = timed_project_and_czt_repeated(
                geom,
                ray_vector=(0.25, -0.2, 1.0),
                roll_rad=0.17,
                image_shape=(512, 512),
                q_shape=q_shape,
                qmax=25.0,
                prefer_gpu=prefer_gpu,
                warmup=warmup,
                repeats=repeats,
            )
            row = dict(result["row"])
            row.update(
                {
                    "benchmark": "mesh_complexity_scaling",
                    "model_class": "sphere",
                    "sphere_resolution": int(res),
                    "status": "ok",
                }
            )
            rows.append(row)
            for repeat_row in result["repeat_rows"]:
                rr = dict(repeat_row)
                rr.update(
                    {
                        "benchmark": "mesh_complexity_scaling",
                        "model_class": "sphere",
                        "sphere_resolution": int(res),
                        "status": "ok",
                    }
                )
                repeat_rows.append(rr)
        except Exception as exc:
            rows.append(
                {
                    "benchmark": "mesh_complexity_scaling",
                    "model_class": "sphere",
                    "sphere_resolution": int(res),
                    "status": "failed_projection",
                    "error": repr(exc),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out["csv"] / "performance_projection_scaling.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(out["csv"] / "performance_projection_scaling_repeats.csv", index=False)
    return df


def run_q_grid_scaling_benchmark(
    out_root: str | Path = "section4_performance_outputs",
    prefer_gpu: bool = True,
    quick: bool = True,
    warmup: Optional[int] = None,
    repeats: Optional[int] = None,
) -> pd.DataFrame:
    out = ensure_output_dirs(out_root)
    rows = []
    repeat_rows = []

    q_shapes = [(129, 129), (257, 257)] if quick else [(129, 129), (257, 257), (513, 513)]
    image_shape = (512, 512)
    warmup = 1 if warmup is None and quick else 2 if warmup is None else int(warmup)
    repeats = 3 if repeats is None and quick else 5 if repeats is None else int(repeats)

    model_factories = make_benchmark_models(quick=quick)
    representative_models = {
        "sphere": model_factories["sphere"],
        "gaussian_blob": model_factories["gaussian_blob"],
    }

    for model_name, factory in representative_models.items():
        try:
            geom = factory()
        except Exception as exc:
            rows.append({"benchmark": "q_grid_scaling", "model_class": model_name, "status": "failed_model_generation", "error": repr(exc)})
            continue

        for q_shape in q_shapes:
            try:
                result = timed_project_and_czt_repeated(
                    geom,
                    ray_vector=(0.25, -0.2, 1.0),
                    roll_rad=0.17,
                    image_shape=image_shape,
                    q_shape=q_shape,
                    qmax=25.0,
                    prefer_gpu=prefer_gpu,
                    warmup=warmup,
                    repeats=repeats,
                )
                row = dict(result["row"])
                row.update(
                    {
                        "benchmark": "q_grid_scaling",
                        "model_class": model_name,
                        "status": "ok",
                    }
                )
                rows.append(row)
                for repeat_row in result["repeat_rows"]:
                    rr = dict(repeat_row)
                    rr.update(
                        {
                            "benchmark": "q_grid_scaling",
                            "model_class": model_name,
                            "status": "ok",
                        }
                    )
                    repeat_rows.append(rr)
            except Exception as exc:
                rows.append(
                    {
                        "benchmark": "q_grid_scaling",
                        "model_class": model_name,
                        "image_H": image_shape[0],
                        "image_W": image_shape[1],
                        "q_H": q_shape[0],
                        "q_W": q_shape[1],
                        "status": "failed_projection",
                        "error": repr(exc),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out["csv"] / "performance_q_grid_scaling.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(out["csv"] / "performance_q_grid_scaling_repeats.csv", index=False)
    return df


# ------------------------------------------------------------
# 4.2 Throughput for orientation-library generation
# ------------------------------------------------------------
def run_orientation_library_benchmark(
    out_root: str | Path = "section4_performance_outputs",
    prefer_gpu: bool = True,
    quick: bool = True,
    seed: int = 11,
) -> pd.DataFrame:
    out = ensure_output_dirs(out_root)

    orientation_counts = [1, 10, 50] if quick else [1, 10, 100, 500, 1000]
    image_shape = (256, 256) if quick else (512, 512)
    q_shape = (129, 129) if quick else (257, 257)

    rows = []
    models = make_benchmark_models(quick=quick)

    for model_name, factory in models.items():
        try:
            t_mesh0 = time.perf_counter()
            geom = factory()
            mesh_generation_time = float(time.perf_counter() - t_mesh0)
        except Exception as exc:
            for n_ori in orientation_counts:
                rows.append(
                    {
                        "model_class": model_name,
                        "orientation_count": int(n_ori),
                        "status": "failed_model_generation",
                        "error": repr(exc),
                    }
                )
            continue

        max_ori = max(orientation_counts)
        rays = random_unit_vectors(max_ori, seed=seed)
        rolls = random_rolls(max_ori, seed=seed + 100)

        for n_ori in orientation_counts:
            total_projection = 0.0
            total_czt = 0.0
            total_all = 0.0
            failure_count = 0
            fragments_sum = 0

            t_lib0 = time.perf_counter()
            for i in range(int(n_ori)):
                try:
                    result = timed_project_and_czt(
                        geom,
                        ray_vector=rays[i],
                        roll_rad=float(rolls[i]),
                        image_shape=image_shape,
                        q_shape=q_shape,
                        qmax=25.0,
                        prefer_gpu=prefer_gpu,
                    )
                    row_i = result["row"]
                    total_projection += float(row_i["projection_time_s"])
                    total_czt += float(row_i["czt_time_s"])
                    total_all += float(row_i["total_time_s"])
                    fragments_sum += int(row_i.get("num_projected_fragments", 0))
                except Exception as exc:
                    failure_count += 1
                    if failure_count <= 3:
                        print(f"[orientation benchmark] {model_name} failed at orientation {i}: {exc!r}")

            wall_time = float(time.perf_counter() - t_lib0)
            ok_count = int(n_ori) - failure_count
            amortized_ms = 1000.0 * wall_time / max(ok_count, 1)
            patterns_per_s = ok_count / wall_time if wall_time > 0 else float("nan")

            rows.append(
                {
                    "model_class": model_name,
                    "orientation_count": int(n_ori),
                    "ok_count": int(ok_count),
                    "failure_count": int(failure_count),
                    "image_H": int(image_shape[0]),
                    "image_W": int(image_shape[1]),
                    "q_H": int(q_shape[0] if q_shape[0] % 2 == 1 else q_shape[0] + 1),
                    "q_W": int(q_shape[1] if q_shape[1] % 2 == 1 else q_shape[1] + 1),
                    "mesh_vertices": int(len(geom.vertices)),
                    "mesh_faces": int(len(geom.faces)),
                    "mesh_generation_time_s": float(mesh_generation_time),
                    "total_projection_time_s_sum": float(total_projection),
                    "total_czt_time_s_sum": float(total_czt),
                    "total_inner_time_s_sum": float(total_all),
                    "total_wall_time_s": float(wall_time),
                    "amortized_time_ms_per_pattern": float(amortized_ms),
                    "patterns_per_second": float(patterns_per_s),
                    "mean_projected_fragments": float(fragments_sum / max(ok_count, 1)),
                    "projection_method": choose_projection_method(prefer_gpu=prefer_gpu),
                    "status": "ok" if failure_count == 0 else "partial",
                    **get_gpu_info(),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out["csv"] / "performance_orientation_library.csv", index=False)
    return df


# ------------------------------------------------------------
# 4.3 Mesh-generation benchmark
# ------------------------------------------------------------
def run_mesh_generation_benchmark(
    out_root: str | Path = "section4_performance_outputs",
    quick: bool = True,
) -> pd.DataFrame:
    out = ensure_output_dirs(out_root)
    rows = []

    tasks = [
        ("sphere_direct", lambda: make_sphere_mesh(radius=1.0, resolution=64 if quick else 96)),
        ("ellipsoid_direct", lambda: make_ellipsoid_mesh(
            axes=(1.0, 0.65, 0.42),
            resolution=64 if quick else 96,
            rotation=rotation_matrix_from_euler_xyz(0.33, -0.51, 0.27),
        )),
        ("two_droplet_component_assembly", lambda: make_two_droplet_mesh(
            radius1=1.0,
            radius2=0.75,
            separation=2.5,
            resolution=40 if quick else 64,
        )),
        ("icosahedron_predefined_topology", lambda: make_polyhedron_mesh(name="icosa", radius=1.0)),
    ]

    blob_shapes = [(80, 80, 80), (96, 96, 96)] if quick else [(80, 80, 80), (120, 120, 120), (160, 160, 160)]
    for shape in blob_shapes:
        tasks.append((f"gaussian_blob_scalar_field_{shape[0]}x{shape[1]}x{shape[2]}", lambda shape=shape: make_gaussian_blob_mesh(field_shape=shape)))

    for model_name, factory in tasks:
        gc.collect()
        reset_gpu_memory_pool()
        t0 = time.perf_counter()
        try:
            geom = factory()
            total = time.perf_counter() - t0
            meta = dict(geom.mesh_meta)
            rows.append(
                {
                    "model_class": model_name,
                    "status": "ok",
                    "mesh_generation_route": meta.get("mesh_generation_route", meta.get("mesh_source", None)),
                    "field_shape": json.dumps(meta.get("field_shape", None)),
                    "field_generation_time_s": float(meta.get("field_sampling_seconds", meta.get("field_generation_time_s", 0.0))),
                    "surface_extraction_time_s": float(meta.get("iso_extraction_seconds", meta.get("surface_extraction_time_s", 0.0))),
                    "assembly_time_s": float(meta.get("assembly_time_s", 0.0)),
                    "reported_total_mesh_generation_time_s": float(meta.get("total_mesh_generation_time_s", total)),
                    "wall_total_mesh_generation_time_s": float(total),
                    "vertices": int(len(geom.vertices)),
                    "faces": int(len(geom.faces)),
                    "mesh_volume": float(abs(signed_mesh_volume(geom.vertices, geom.faces))),
                    "mesh_source": meta.get("mesh_source", None),
                    "iso_extractor_used": meta.get("iso_extractor_used", None),
                    "field_sampling_method": meta.get("field_sampling_method", None),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_class": model_name,
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out["csv"] / "performance_mesh_generation.csv", index=False)
    return df


# ------------------------------------------------------------
# 4.4 Memory/computational cost comparison
# ------------------------------------------------------------
def estimate_mesh_memory_MB(n_vertices: int, n_faces: int, dtype_vertices: int = 8, dtype_faces: int = 4) -> float:
    bytes_total = n_vertices * 3 * dtype_vertices
    bytes_total += n_vertices * 3 * dtype_vertices
    bytes_total += n_faces * 3 * dtype_faces
    bytes_total += n_faces * 3 * dtype_vertices
    return float(bytes_total / 1024**2)


def run_memory_cost_comparison(
    out_root: str | Path = "section4_performance_outputs",
    mesh_face_counts: Sequence[int] = (1_000, 10_000, 100_000, 300_000),
    grid_sizes: Sequence[int] = (256, 500, 512, 1024),
) -> pd.DataFrame:
    out = ensure_output_dirs(out_root)
    rows = []
    for N in grid_sizes:
        voxel_n = int(N) ** 3
        image_n = int(N) ** 2
        for faces in mesh_face_counts:
            vertices = max(4, int(faces // 2))
            rows.append(
                {
                    "grid_or_image_size": int(N),
                    "mesh_faces_assumed": int(faces),
                    "mesh_vertices_assumed": int(vertices),
                    "voxel_float32_MB": float(voxel_n * 4 / 1024**2),
                    "voxel_float64_MB": float(voxel_n * 8 / 1024**2),
                    "voxel_complex64_MB": float(voxel_n * 8 / 1024**2),
                    "voxel_complex128_MB": float(voxel_n * 16 / 1024**2),
                    "projected_float32_MB": float(image_n * 4 / 1024**2),
                    "projected_float64_MB": float(image_n * 8 / 1024**2),
                    "projected_complex64_MB": float(image_n * 8 / 1024**2),
                    "projected_complex128_MB": float(image_n * 16 / 1024**2),
                    "estimated_mesh_arrays_MB": estimate_mesh_memory_MB(vertices, faces),
                    "dense_voxel_float32_over_projected_float32": float((voxel_n * 4) / (image_n * 4)),
                    "dense_voxel_complex64_over_projected_complex64": float((voxel_n * 8) / (image_n * 8)),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out["csv"] / "memory_cost_comparison.csv", index=False)
    return df


# ------------------------------------------------------------
# Master runner
# ------------------------------------------------------------
def run_all_section4_benchmarks(
    out_root: str | Path = "section4_performance_outputs",
    prefer_gpu: bool = True,
    quick: bool = True,
    seed: int = 11,
    timing_warmup: Optional[int] = None,
    timing_repeats: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    check_required_symbols()
    out = ensure_output_dirs(out_root)

    env = collect_environment_metadata(prefer_gpu=prefer_gpu, quick=quick)
    env["timing_warmup"] = 1 if timing_warmup is None and quick else 2 if timing_warmup is None else int(timing_warmup)
    env["timing_repeats"] = 3 if timing_repeats is None and quick else 5 if timing_repeats is None else int(timing_repeats)
    save_json(out["json"] / "section4_environment.json", env)

    print(f"[section4] Output root: {out['root'].resolve()}")
    print(f"[section4] Projection backend: {env['projection_method']}")
    if env["gpu_info"].get("gpu_available"):
        print(f"[section4] GPU: {env['gpu_info'].get('gpu_name')}")

    print("[1/5] Projection scaling benchmark")
    df_projection = run_projection_scaling_benchmark(
        out_root=out_root,
        prefer_gpu=prefer_gpu,
        quick=quick,
        warmup=env["timing_warmup"],
        repeats=env["timing_repeats"],
    )

    print("[2/5] q-grid scaling benchmark")
    df_q_grid = run_q_grid_scaling_benchmark(
        out_root=out_root,
        prefer_gpu=prefer_gpu,
        quick=quick,
        warmup=env["timing_warmup"],
        repeats=env["timing_repeats"],
    )

    print("[3/5] Orientation-library throughput benchmark")
    df_library = run_orientation_library_benchmark(out_root=out_root, prefer_gpu=prefer_gpu, quick=quick, seed=seed)

    print("[4/5] Mesh-generation benchmark")
    df_mesh = run_mesh_generation_benchmark(out_root=out_root, quick=quick)

    print("[5/5] Dense voxel/FFT memory-cost comparison")
    df_memory = run_memory_cost_comparison(out_root=out_root)

    compact = {
        "script_version": SCRIPT_VERSION,
        "projection_scaling_rows": int(len(df_projection)),
        "q_grid_scaling_rows": int(len(df_q_grid)),
        "orientation_library_rows": int(len(df_library)),
        "mesh_generation_rows": int(len(df_mesh)),
        "memory_cost_rows": int(len(df_memory)),
        "projection_backend": choose_projection_method(prefer_gpu=prefer_gpu),
        "timing_warmup": int(env["timing_warmup"]),
        "timing_repeats": int(env["timing_repeats"]),
    }
    save_json(out["json"] / "section4_compact_summary.json", compact)

    print("[section4] Done.")
    print(f"[section4] CSV summaries: {out['csv']}")
    return {
        "projection_scaling": df_projection,
        "q_grid_scaling": df_q_grid,
        "orientation_library": df_library,
        "mesh_generation": df_mesh,
        "memory_cost": df_memory,
    }


# Uncomment this line if you want automatic execution after pasting this file.
# results = run_all_section4_benchmarks(out_root="section4_performance_outputs", prefer_gpu=True, quick=True)
