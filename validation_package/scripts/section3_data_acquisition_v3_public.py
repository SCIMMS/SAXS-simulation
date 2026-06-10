# ============================================================
# Section 3 data acquisition for numerical validation
# ============================================================
# Assumption:
#   - The main notebook cell has already been executed.
#   - The following objects/functions are already available:
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
#   Generate validation data for:
#       3.1 Analytical reference shapes
#       3.2 Projected-mass conservation and orientation consistency
#       3.3 Mesh- and detector-plane resolution convergence
#       3.4 CZT scaling, q=0, and q-space sampling validation
#
# Public-release scope:
#   This script uses analytical and synthetic reference geometries only.
#   It does not read experimental detector frames, masks, PONI files, or
#   non-public morphology-specific datasets.
#
# Outputs:
#   section3_validation_outputs/
#       summaries/*.csv
#       npz/*.npz
#       json/*.json
#
# Run in Colab after executing cell.txt:
#   exec(open("section3_data_acquisition.py").read())
#   run_all_section3_data_acquisition()
# ============================================================

from __future__ import annotations

import json
import math
import platform
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Tuple, Sequence, Optional, List

import numpy as np
import pandas as pd


SCRIPT_VERSION = "section3_public_v3"

REQUIRED_MAIN_SYMBOLS = (
    "StructureGeometry",
    "project_density",
    "projection_basis_from_ray",
    "compute_face_normals",
    "orient_faces_outward_by_volume",
    "signed_mesh_volume",
    "czt2_from_real_axes",
    "build_structure_geometry",
)

# ------------------------------------------------------------
# Optional backend selection
# ------------------------------------------------------------
def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def check_required_symbols() -> None:
    missing = [name for name in REQUIRED_MAIN_SYMBOLS if name not in globals()]
    if missing:
        raise RuntimeError(
            "Required symbols are missing from the current namespace. "
            "Execute the main simulator/notebook code first, then run this "
            "Section 3 validation script. Missing: "
            + ", ".join(missing)
        )


def choose_projection_method(prefer_gpu: bool = True) -> str:
    if prefer_gpu and _has_cupy():
        return "signed_raster_gpu"
    return "signed_raster"


# ------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------
def ensure_output_dirs(root: str | Path = "section3_validation_outputs") -> Dict[str, Path]:
    root = Path(root)
    out = {
        "root": root,
        "npz": root / "npz",
        "csv": root / "summaries",
        "json": root / "json",
        "fig": root / "figures",
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
        json.dump(conv(obj), f, indent=2, ensure_ascii=False)


def _package_version(package_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def _cupy_runtime_metadata() -> Dict:
    info = {
        "available": False,
        "version": None,
        "cuda_runtime_version": None,
        "device_name": None,
        "device_memory_total_MB": None,
    }
    try:
        import cupy as cp

        info["available"] = True
        info["version"] = getattr(cp, "__version__", None)
        try:
            info["cuda_runtime_version"] = int(cp.cuda.runtime.runtimeGetVersion())
        except Exception:
            pass
        try:
            dev = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            name = props.get("name", b"")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            _, total_b = cp.cuda.runtime.memGetInfo()
            info["device_name"] = name
            info["device_memory_total_MB"] = float(total_b / 1024**2)
        except Exception:
            pass
    except Exception:
        pass
    return info


def collect_environment_metadata(prefer_gpu: bool = True) -> Dict:
    cupy_runtime = _cupy_runtime_metadata()
    packages = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cupy": cupy_runtime["version"] or _package_version("cupy"),
        "cupy-cuda12x": _package_version("cupy-cuda12x"),
        "cupy-cuda11x": _package_version("cupy-cuda11x"),
        "open3d": _package_version("open3d"),
        "scipy": _package_version("scipy"),
        "scikit-image": _package_version("scikit-image"),
        "vtk": _package_version("vtk"),
    }
    return {
        "script_version": SCRIPT_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "prefer_gpu": bool(prefer_gpu),
        "projection_backend": choose_projection_method(prefer_gpu=prefer_gpu),
        "packages": packages,
        "cupy_runtime": cupy_runtime,
        "public_release_scope": (
            "Analytical and synthetic numerical validation only; no experimental "
            "detector images, masks, PONI geometry files, or non-public morphology data."
        ),
    }


def uv_axes_from_projection_meta(meta: Dict, du: float, dv: float) -> Tuple[np.ndarray, np.ndarray]:
    H, W = meta["image_shape"]
    u0 = float(meta["x0_unit"])
    v0 = float(meta["y0_unit"])
    u_axis = u0 + np.arange(W, dtype=np.float64) * float(du)
    v_axis = v0 + np.arange(H, dtype=np.float64) * float(dv)
    return u_axis, v_axis


def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + eps))


def normalized_correlation(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    else:
        a = a.ravel()
        b = b.ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 0:
        return float("nan")
    return float(np.dot(a, b) / den)


def zero_frequency_index(q: np.ndarray, tol: float = 1e-12) -> int:
    """Return the index of q=0 and fail if the grid does not contain zero."""
    q = np.asarray(q, dtype=np.float64)
    idx = int(np.argmin(np.abs(q)))
    if abs(float(q[idx])) > float(tol):
        raise ValueError(
            f"q-grid does not contain q=0. Closest value is {q[idx]:.6e}. "
            "Use an odd number of q samples or construct q-grid explicitly with zero."
        )
    return idx


# ------------------------------------------------------------
# Mesh construction helpers
# ------------------------------------------------------------
def mesh_to_geom(
    vertices: np.ndarray,
    faces: np.ndarray,
    structure_type: str = "custom-mesh",
    params: Optional[Dict] = None,
) -> "StructureGeometry":
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    faces, vol, flipped = orient_faces_outward_by_volume(vertices, faces)
    normals = compute_face_normals(vertices, faces)

    return StructureGeometry(
        structure_type=structure_type,
        params={} if params is None else dict(params),
        vertices=vertices,
        faces=faces,
        face_normals=normals,
        mesh_meta={
            "mesh_source": "section3_validation_helper",
            "mesh_vertex_count": int(len(vertices)),
            "mesh_face_count": int(len(faces)),
            "signed_volume": float(vol),
            "faces_flipped_for_positive_volume": bool(flipped),
        },
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
    # local -> world rotation, consistent with row-vector convention used in cell.txt
    return Rz @ Ry @ Rx


def make_ellipsoid_mesh(
    axes: Tuple[float, float, float] = (1.0, 0.8, 0.6),
    resolution: int = 80,
    rotation: Optional[np.ndarray] = None,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> "StructureGeometry":
    import open3d as o3d

    if rotation is None:
        rotation = np.eye(3, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=int(resolution))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    vertices = vertices * axes[None, :]
    # local -> world with row-vector convention: local @ R.T + center
    vertices = vertices @ rotation.T + center[None, :]

    return mesh_to_geom(
        vertices,
        faces,
        structure_type="ellipsoid-mesh",
        params={"axes": axes.tolist(), "resolution": int(resolution)},
    )


def make_sphere_mesh(
    radius: float = 1.0,
    resolution: int = 80,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> "StructureGeometry":
    return make_ellipsoid_mesh(
        axes=(radius, radius, radius),
        resolution=resolution,
        rotation=np.eye(3, dtype=np.float64),
        center=center,
    )


def make_box_mesh(
    half_lengths: Tuple[float, float, float] = (1.0, 0.7, 0.5),
    rotation: Optional[np.ndarray] = None,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> "StructureGeometry":
    if rotation is None:
        rotation = np.eye(3, dtype=np.float64)
    a, b, c = map(float, half_lengths)
    vertices = np.array(
        [
            [-a, -b, -c],
            [ a, -b, -c],
            [ a,  b, -c],
            [-a,  b, -c],
            [-a, -b,  c],
            [ a, -b,  c],
            [ a,  b,  c],
            [-a,  b,  c],
        ],
        dtype=np.float64,
    )
    # 12 triangles; orientation will be repaired by volume.
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom z=-c
            [4, 5, 6], [4, 6, 7],  # top z=+c
            [0, 1, 5], [0, 5, 4],  # y=-b
            [1, 2, 6], [1, 6, 5],  # x=+a
            [2, 3, 7], [2, 7, 6],  # y=+b
            [3, 0, 4], [3, 4, 7],  # x=-a
        ],
        dtype=np.int32,
    )
    center = np.asarray(center, dtype=np.float64)
    vertices = vertices @ rotation.T + center[None, :]
    return mesh_to_geom(
        vertices,
        faces,
        structure_type="box-mesh",
        params={"half_lengths": [a, b, c]},
    )


def append_meshes(meshes: List["StructureGeometry"], structure_type: str = "multi-mesh") -> "StructureGeometry":
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
    return mesh_to_geom(vertices, faces, structure_type=structure_type)


# ------------------------------------------------------------
# Analytical projected thickness: sphere/ellipsoid
# ------------------------------------------------------------
def analytic_ellipsoid_projected_thickness(
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    axes: Tuple[float, float, float],
    rotation: np.ndarray,
    ray_vector: Sequence[float],
    roll_rad: float = 0.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Analytical path length through a rotated ellipsoid.

    Ellipsoid equation:
        x_local^T diag(1/a^2,1/b^2,1/c^2) x_local <= 1

    The main notebook uses row-vector convention:
        x_local = (x_world - center) @ R
    where R is local -> world rotation in the helper functions.
    Thus:
        Q_world = R diag(1/a^2,1/b^2,1/c^2) R^T
    """
    axes = np.asarray(axes, dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    e_u, e_v, k = projection_basis_from_ray(ray_vector, roll_rad=roll_rad)

    D = np.diag(1.0 / np.maximum(axes, 1e-30) ** 2)
    Q = R @ D @ R.T

    U, V = np.meshgrid(u_axis, v_axis, indexing="xy")
    P = U[..., None] * e_u[None, None, :] + V[..., None] * e_v[None, None, :] - center[None, None, :]

    A = float(k @ Q @ k)
    B = np.einsum("...i,ij,j->...", P, Q, k)
    C = np.einsum("...i,ij,...j->...", P, Q, P)

    disc = B * B - A * (C - 1.0)
    out = np.zeros_like(U, dtype=np.float64)
    mask = disc > 0.0
    out[mask] = 2.0 * np.sqrt(disc[mask]) / A
    return out


def analytic_sphere_projected_thickness(
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    radius: float = 1.0,
) -> np.ndarray:
    U, V = np.meshgrid(u_axis, v_axis, indexing="xy")
    r2 = U * U + V * V
    out = np.zeros_like(U, dtype=np.float64)
    mask = r2 <= radius * radius
    out[mask] = 2.0 * np.sqrt(radius * radius - r2[mask])
    return out


# ------------------------------------------------------------
# Analytical amplitudes: sphere, ellipsoid, box
# ------------------------------------------------------------
def _sphere_factor(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    small = np.abs(x) < 1e-6
    xs = x[small]
    # 3(sin x - x cos x)/x^3 = 1 - x^2/10 + x^4/280 - ...
    out[small] = 1.0 - xs * xs / 10.0 + xs**4 / 280.0
    xl = x[~small]
    out[~small] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / (xl**3)
    return out


def analytic_sphere_amplitude(
    q_u: np.ndarray,
    q_v: np.ndarray,
    radius: float = 1.0,
    density: float = 1.0,
) -> np.ndarray:
    QU, QV = np.meshgrid(q_u, q_v, indexing="xy")
    q = np.sqrt(QU * QU + QV * QV)
    V = 4.0 * np.pi * radius**3 / 3.0
    return density * V * _sphere_factor(q * radius)


def analytic_ellipsoid_amplitude(
    q_u: np.ndarray,
    q_v: np.ndarray,
    axes: Tuple[float, float, float],
    rotation: np.ndarray,
    ray_vector: Sequence[float],
    roll_rad: float = 0.0,
    density: float = 1.0,
) -> np.ndarray:
    axes = np.asarray(axes, dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64)
    e_u, e_v, k = projection_basis_from_ray(ray_vector, roll_rad=roll_rad)

    QU, QV = np.meshgrid(q_u, q_v, indexing="xy")
    q_world = QU[..., None] * e_u[None, None, :] + QV[..., None] * e_v[None, None, :]
    q_local = q_world @ R  # row-vector local coordinates

    Qeff = np.sqrt(
        (axes[0] * q_local[..., 0]) ** 2
        + (axes[1] * q_local[..., 1]) ** 2
        + (axes[2] * q_local[..., 2]) ** 2
    )
    V = 4.0 * np.pi * axes[0] * axes[1] * axes[2] / 3.0
    return density * V * _sphere_factor(Qeff)


def _sinc_unnormalized(x: np.ndarray) -> np.ndarray:
    # sin(x)/x
    x = np.asarray(x, dtype=np.float64)
    return np.sinc(x / np.pi)


def analytic_box_amplitude(
    q_u: np.ndarray,
    q_v: np.ndarray,
    half_lengths: Tuple[float, float, float],
    rotation: np.ndarray,
    ray_vector: Sequence[float],
    roll_rad: float = 0.0,
    density: float = 1.0,
) -> np.ndarray:
    a, b, c = map(float, half_lengths)
    R = np.asarray(rotation, dtype=np.float64)
    e_u, e_v, k = projection_basis_from_ray(ray_vector, roll_rad=roll_rad)

    QU, QV = np.meshgrid(q_u, q_v, indexing="xy")
    q_world = QU[..., None] * e_u[None, None, :] + QV[..., None] * e_v[None, None, :]
    q_local = q_world @ R

    Ax = 2.0 * a * _sinc_unnormalized(q_local[..., 0] * a)
    Ay = 2.0 * b * _sinc_unnormalized(q_local[..., 1] * b)
    Az = 2.0 * c * _sinc_unnormalized(q_local[..., 2] * c)
    return density * Ax * Ay * Az


# ------------------------------------------------------------
# Polyhedron amplitude using divergence theorem
# ------------------------------------------------------------
_DUNAVANT_7_BARY = np.array(
    [
        [1/3, 1/3, 1/3],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
    ],
    dtype=np.float64,
)
# Weights for reference triangle with area 1/2.
_DUNAVANT_7_W = np.array(
    [
        0.112500000000000,
        0.066197076394253,
        0.066197076394253,
        0.066197076394253,
        0.062969590272414,
        0.062969590272414,
        0.062969590272414,
    ],
    dtype=np.float64,
)


def triangle_exp_surface_integral(q: np.ndarray, tri: np.ndarray) -> complex:
    """
    Integral over one triangle:
        int_triangle exp(-i q dot r) dS

    Uses a closed-form barycentric expression when phase values are distinct,
    and a 7-point surface quadrature fallback near degenerate phase values.
    """
    q = np.asarray(q, dtype=np.float64)
    tri = np.asarray(tri, dtype=np.float64)
    jac = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    z = -1j * (tri @ q)

    # If z values are too close, the exact formula suffers cancellation.
    if np.min(np.abs(z[:, None] - z[None, :] + np.eye(3))) < 1e-7:
        pts = _DUNAVANT_7_BARY @ tri
        return complex(jac * np.sum(_DUNAVANT_7_W * np.exp(-1j * (pts @ q))))

    val = 0.0 + 0.0j
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        val += np.exp(z[i]) / ((z[i] - z[j]) * (z[i] - z[k]))
    return complex(jac * val)


def polyhedron_amplitude_divergence_theorem(
    geom: "StructureGeometry",
    q_u: np.ndarray,
    q_v: np.ndarray,
    ray_vector: Sequence[float],
    roll_rad: float = 0.0,
    density: float = 1.0,
) -> np.ndarray:
    """
    Surface-based reference amplitude for a closed triangular polyhedron.

    Uses:
        int_V exp(-i q dot r) dV
        = int_surface (i q dot n / |q|^2) exp(-i q dot r) dS

    At q=0, returns the mesh volume.
    """
    e_u, e_v, k = projection_basis_from_ray(ray_vector, roll_rad=roll_rad)
    verts = np.asarray(geom.vertices, dtype=np.float64)
    faces = np.asarray(geom.faces, dtype=np.int32)
    normals = np.asarray(geom.face_normals, dtype=np.float64)
    tris = verts[faces]
    volume = abs(signed_mesh_volume(verts, faces))

    out = np.empty((len(q_v), len(q_u)), dtype=np.complex128)
    for iv, qv in enumerate(q_v):
        for iu, qu in enumerate(q_u):
            q = qu * e_u + qv * e_v
            q2 = float(q @ q)
            if q2 < 1e-20:
                out[iv, iu] = density * volume
                continue
            s = 0.0 + 0.0j
            for tri, nface in zip(tris, normals):
                surf_int = triangle_exp_surface_integral(q, tri)
                s += (1j * float(q @ nface) / q2) * surf_int
            out[iv, iu] = density * s
    return out


# ------------------------------------------------------------
# Direct DFT for CZT validation
# ------------------------------------------------------------
def direct_dft2_from_real_axes(
    projected_density: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    q_u: np.ndarray,
    q_v: np.ndarray,
) -> np.ndarray:
    rho = np.asarray(projected_density, dtype=np.complex128)
    u_axis = np.asarray(u_axis, dtype=np.float64)
    v_axis = np.asarray(v_axis, dtype=np.float64)
    q_u = np.asarray(q_u, dtype=np.float64)
    q_v = np.asarray(q_v, dtype=np.float64)

    du = float(u_axis[1] - u_axis[0])
    dv = float(v_axis[1] - v_axis[0])

    Eu = np.exp(-1j * np.outer(q_u, u_axis))  # (Nq_u, Nu)
    Ev = np.exp(-1j * np.outer(q_v, v_axis))  # (Nq_v, Nv)
    return (dv * du) * (Ev @ rho @ Eu.T)


# ------------------------------------------------------------
# Projection wrapper
# ------------------------------------------------------------
def project_geom(
    geom: "StructureGeometry",
    ray_vector: Sequence[float],
    roll_rad: float,
    image_shape: Tuple[int, int],
    margin_ratio: float = 0.05,
    prefer_gpu: bool = True,
) -> Tuple[np.ndarray, float, float, Dict]:
    method = choose_projection_method(prefer_gpu=prefer_gpu)
    proj, du, dv, meta = project_density(
        geom,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
        image_shape=image_shape,
        margin_ratio=margin_ratio,
        projection_method=method,
        out_dtype="float64",
        edge_rule="top_left",
        supersample=1,
    )
    return proj.astype(np.float64, copy=False), du, dv, meta


# ------------------------------------------------------------
# Individual data acquisition routines
# ------------------------------------------------------------
def run_sphere_analytical_validation(
    out_root: str | Path = "section3_validation_outputs",
    radius: float = 1.0,
    mesh_resolution: int = 96,
    image_shape: Tuple[int, int] = (512, 512),
    qmax: float = 25.0,
    nq: int = 257,
    prefer_gpu: bool = True,
) -> Dict:
    out = ensure_output_dirs(out_root)
    geom = make_sphere_mesh(radius=radius, resolution=mesh_resolution)

    ray_vector = (0.3, -0.2, 1.0)
    roll_rad = 0.17
    proj, du, dv, meta = project_geom(geom, ray_vector, roll_rad, image_shape, prefer_gpu=prefer_gpu)
    u_axis, v_axis = uv_axes_from_projection_meta(meta, du, dv)

    T_ref = analytic_sphere_projected_thickness(u_axis, v_axis, radius=radius)
    q = np.linspace(-qmax, qmax, nq, dtype=np.float64)

    A_czt = czt2_from_real_axes(proj, u_axis, v_axis, q, q)
    A_ref = analytic_sphere_amplitude(q, q, radius=radius)

    I_czt = np.abs(A_czt) ** 2
    I_ref = np.abs(A_ref) ** 2

    mass_proj = float(np.sum(proj) * du * dv)
    mass_ref = float(4.0 * np.pi * radius**3 / 3.0)
    mesh_volume = float(abs(signed_mesh_volume(geom.vertices, geom.faces)))
    zero_idx = zero_frequency_index(q)

    metrics = {
        "case": "sphere",
        "radius": float(radius),
        "mesh_resolution": int(mesh_resolution),
        "mesh_faces": int(len(geom.faces)),
        "image_shape": list(image_shape),
        "qmax": float(qmax),
        "nq": int(nq),
        "projected_density_rel_l2": relative_l2(proj, T_ref),
        "amplitude_rel_l2": relative_l2(A_czt, A_ref),
        "intensity_rel_l2": relative_l2(I_czt, I_ref),
        "pattern_corr": normalized_correlation(I_czt, I_ref),
        "mass_projected": mass_proj,
        "mass_reference_analytical": mass_ref,
        "mass_reference_mesh": mesh_volume,
        "mass_relative_error_vs_analytical": float((mass_proj - mass_ref) / mass_ref),
        "mass_relative_error_vs_mesh": float((mass_proj - mesh_volume) / mesh_volume),
        "A00_czt": complex(A_czt[zero_idx, zero_idx]),
        "A00_reference_analytical": float(mass_ref),
        "A00_reference_mesh": float(mesh_volume),
        "A00_mass_relative_error_vs_projected": float((np.real(A_czt[zero_idx, zero_idx]) - mass_proj) / (abs(mass_proj) + 1e-30)),
        "q_zero_value": float(q[zero_idx]),
        "projection_method": meta["projection_method"],
    }

    np.savez_compressed(
        out["npz"] / "sphere_analytical_validation.npz",
        projected_density=proj,
        projected_density_ref=T_ref,
        q=q,
        A_czt=A_czt,
        A_ref=A_ref,
        I_czt=I_czt,
        I_ref=I_ref,
        u_axis=u_axis,
        v_axis=v_axis,
    )
    save_json(out["json"] / "sphere_analytical_validation.json", metrics)
    pd.DataFrame([metrics]).to_csv(out["csv"] / "sphere_analytical_validation.csv", index=False)
    return metrics


def run_ellipsoid_analytical_validation(
    out_root: str | Path = "section3_validation_outputs",
    axes: Tuple[float, float, float] = (1.0, 0.65, 0.42),
    mesh_resolution: int = 96,
    image_shape: Tuple[int, int] = (512, 512),
    qmax: float = 25.0,
    nq: int = 257,
    prefer_gpu: bool = True,
) -> Dict:
    out = ensure_output_dirs(out_root)
    R = rotation_matrix_from_euler_xyz(0.33, -0.51, 0.27)
    geom = make_ellipsoid_mesh(axes=axes, resolution=mesh_resolution, rotation=R)

    ray_vector = (0.25, 0.15, 1.0)
    roll_rad = -0.21
    proj, du, dv, meta = project_geom(geom, ray_vector, roll_rad, image_shape, prefer_gpu=prefer_gpu)
    u_axis, v_axis = uv_axes_from_projection_meta(meta, du, dv)

    T_ref = analytic_ellipsoid_projected_thickness(
        u_axis,
        v_axis,
        axes=axes,
        rotation=R,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
    )

    q = np.linspace(-qmax, qmax, nq, dtype=np.float64)
    A_czt = czt2_from_real_axes(proj, u_axis, v_axis, q, q)
    A_ref = analytic_ellipsoid_amplitude(
        q,
        q,
        axes=axes,
        rotation=R,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
    )

    I_czt = np.abs(A_czt) ** 2
    I_ref = np.abs(A_ref) ** 2

    mass_proj = float(np.sum(proj) * du * dv)
    mass_ref = float(4.0 * np.pi * axes[0] * axes[1] * axes[2] / 3.0)
    mesh_volume = float(abs(signed_mesh_volume(geom.vertices, geom.faces)))
    zero_idx = zero_frequency_index(q)

    metrics = {
        "case": "ellipsoid",
        "axes": list(map(float, axes)),
        "mesh_resolution": int(mesh_resolution),
        "mesh_faces": int(len(geom.faces)),
        "image_shape": list(image_shape),
        "qmax": float(qmax),
        "nq": int(nq),
        "projected_density_rel_l2": relative_l2(proj, T_ref),
        "amplitude_rel_l2": relative_l2(A_czt, A_ref),
        "intensity_rel_l2": relative_l2(I_czt, I_ref),
        "pattern_corr": normalized_correlation(I_czt, I_ref),
        "mass_projected": mass_proj,
        "mass_reference_analytical": mass_ref,
        "mass_reference_mesh": mesh_volume,
        "mass_relative_error_vs_analytical": float((mass_proj - mass_ref) / mass_ref),
        "mass_relative_error_vs_mesh": float((mass_proj - mesh_volume) / mesh_volume),
        "A00_czt": complex(A_czt[zero_idx, zero_idx]),
        "A00_mass_relative_error_vs_projected": float((np.real(A_czt[zero_idx, zero_idx]) - mass_proj) / (abs(mass_proj) + 1e-30)),
        "q_zero_value": float(q[zero_idx]),
        "projection_method": meta["projection_method"],
    }

    np.savez_compressed(
        out["npz"] / "ellipsoid_analytical_validation.npz",
        projected_density=proj,
        projected_density_ref=T_ref,
        q=q,
        A_czt=A_czt,
        A_ref=A_ref,
        I_czt=I_czt,
        I_ref=I_ref,
        u_axis=u_axis,
        v_axis=v_axis,
        rotation=R,
    )
    save_json(out["json"] / "ellipsoid_analytical_validation.json", metrics)
    pd.DataFrame([metrics]).to_csv(out["csv"] / "ellipsoid_analytical_validation.csv", index=False)
    return metrics


def run_box_polyhedron_validation(
    out_root: str | Path = "section3_validation_outputs",
    half_lengths: Tuple[float, float, float] = (1.0, 0.72, 0.45),
    image_shape: Tuple[int, int] = (512, 512),
    qmax: float = 25.0,
    nq: int = 257,
    prefer_gpu: bool = True,
) -> Dict:
    out = ensure_output_dirs(out_root)
    R = rotation_matrix_from_euler_xyz(0.25, 0.35, -0.18)
    geom = make_box_mesh(half_lengths=half_lengths, rotation=R)

    ray_vector = (0.2, -0.3, 1.0)
    roll_rad = 0.13
    proj, du, dv, meta = project_geom(geom, ray_vector, roll_rad, image_shape, prefer_gpu=prefer_gpu)
    u_axis, v_axis = uv_axes_from_projection_meta(meta, du, dv)

    q = np.linspace(-qmax, qmax, nq, dtype=np.float64)
    A_czt = czt2_from_real_axes(proj, u_axis, v_axis, q, q)
    A_ref = analytic_box_amplitude(
        q,
        q,
        half_lengths=half_lengths,
        rotation=R,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
    )

    I_czt = np.abs(A_czt) ** 2
    I_ref = np.abs(A_ref) ** 2

    volume_ref = float(8.0 * half_lengths[0] * half_lengths[1] * half_lengths[2])
    mesh_volume = float(abs(signed_mesh_volume(geom.vertices, geom.faces)))
    mass_proj = float(np.sum(proj) * du * dv)
    zero_idx = zero_frequency_index(q)

    # Additional divergence-theorem check on a smaller q grid.
    nq_poly = 65
    q_poly = np.linspace(-qmax, qmax, nq_poly, dtype=np.float64)
    A_poly = polyhedron_amplitude_divergence_theorem(
        geom,
        q_poly,
        q_poly,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
    )
    A_box_small = analytic_box_amplitude(
        q_poly,
        q_poly,
        half_lengths=half_lengths,
        rotation=R,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
    )

    metrics = {
        "case": "box_polyhedron",
        "half_lengths": list(map(float, half_lengths)),
        "mesh_faces": int(len(geom.faces)),
        "image_shape": list(image_shape),
        "qmax": float(qmax),
        "nq": int(nq),
        "amplitude_rel_l2_czt_vs_box": relative_l2(A_czt, A_ref),
        "intensity_rel_l2_czt_vs_box": relative_l2(I_czt, I_ref),
        "pattern_corr_czt_vs_box": normalized_correlation(I_czt, I_ref),
        "amplitude_rel_l2_divergence_vs_box_small_grid": relative_l2(A_poly, A_box_small),
        "mass_projected": mass_proj,
        "mass_reference_analytical": volume_ref,
        "mass_reference_mesh": mesh_volume,
        "mass_relative_error_vs_analytical": float((mass_proj - volume_ref) / volume_ref),
        "mass_relative_error_vs_mesh": float((mass_proj - mesh_volume) / mesh_volume),
        "A00_czt": complex(A_czt[zero_idx, zero_idx]),
        "A00_mass_relative_error_vs_projected": float((np.real(A_czt[zero_idx, zero_idx]) - mass_proj) / (abs(mass_proj) + 1e-30)),
        "q_zero_value": float(q[zero_idx]),
        "projection_method": meta["projection_method"],
    }

    np.savez_compressed(
        out["npz"] / "box_polyhedron_validation.npz",
        projected_density=proj,
        q=q,
        A_czt=A_czt,
        A_ref=A_ref,
        I_czt=I_czt,
        I_ref=I_ref,
        q_poly=q_poly,
        A_poly_divergence=A_poly,
        A_box_small=A_box_small,
        u_axis=u_axis,
        v_axis=v_axis,
        rotation=R,
    )
    save_json(out["json"] / "box_polyhedron_validation.json", metrics)
    pd.DataFrame([metrics]).to_csv(out["csv"] / "box_polyhedron_validation.csv", index=False)
    return metrics


def run_mass_conservation_orientation_sweep(
    out_root: str | Path = "section3_validation_outputs",
    n_orientations: int = 200,
    image_shape: Tuple[int, int] = (256, 256),
    prefer_gpu: bool = True,
    seed: int = 123,
) -> pd.DataFrame:
    out = ensure_output_dirs(out_root)
    rng = np.random.default_rng(seed)

    geoms = [
        ("sphere", make_sphere_mesh(radius=1.0, resolution=80), 4.0 * np.pi / 3.0),
        ("ellipsoid", make_ellipsoid_mesh(axes=(1.0, 0.65, 0.42), resolution=80), 4.0 * np.pi * 1.0 * 0.65 * 0.42 / 3.0),
        ("box", make_box_mesh(half_lengths=(1.0, 0.72, 0.45)), 8.0 * 1.0 * 0.72 * 0.45),
    ]
    geoms = [
        (name, geom, float(volume_analytical), float(abs(signed_mesh_volume(geom.vertices, geom.faces))))
        for name, geom, volume_analytical in geoms
    ]

    # Optional: add an icosahedron from the user's structure factory if available.
    try:
        ico = build_structure_geometry(
            "polyhedron",
            build_scene=False,
            name="icosa",
            radius=1.0,
            use_direct_polyhedron_mesh=True,
        )
        ico_vol = float(abs(signed_mesh_volume(ico.vertices, ico.faces)))
        geoms.append(("icosahedron", ico, ico_vol, ico_vol))
    except Exception as exc:
        print("[mass sweep] skipping icosahedron:", repr(exc))

    rows = []
    for name, geom, volume_ref_analytical, volume_ref_mesh in geoms:
        for oid in range(int(n_orientations)):
            # random unit ray
            ray = rng.normal(size=3)
            ray = ray / np.linalg.norm(ray)
            roll = float(rng.uniform(0.0, 2.0 * np.pi))

            proj, du, dv, meta = project_geom(
                geom,
                ray_vector=ray,
                roll_rad=roll,
                image_shape=image_shape,
                prefer_gpu=prefer_gpu,
            )
            mass = float(np.sum(proj) * du * dv)
            rel_analytical = (mass - volume_ref_analytical) / volume_ref_analytical
            rel_mesh = (mass - volume_ref_mesh) / volume_ref_mesh
            diag = meta.get("projection_diagnostics", {})
            rows.append(
                {
                    "shape": name,
                    "orientation_id": oid,
                    "mesh_faces": int(len(geom.faces)),
                    "image_H": int(image_shape[0]),
                    "image_W": int(image_shape[1]),
                    "ray_x": float(ray[0]),
                    "ray_y": float(ray[1]),
                    "ray_z": float(ray[2]),
                    "roll_rad": roll,
                    "projected_mass": mass,
                    "reference_volume_analytical": float(volume_ref_analytical),
                    "reference_volume_mesh": float(volume_ref_mesh),
                    "relative_mass_error_vs_analytical": float(rel_analytical),
                    "relative_mass_error_vs_mesh": float(rel_mesh),
                    "min_projected_density": float(np.min(proj)),
                    "max_projected_density": float(np.max(proj)),
                    "negative_pixel_count": int(np.count_nonzero(proj < -1e-12)),
                    "nonzero_pixel_count": int(np.count_nonzero(proj)),
                    "bad_balance_pixels": int(diag.get("bad_balance_pixels", -1)),
                    "projection_method": meta["projection_method"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out["csv"] / "mass_conservation_orientation_sweep.csv", index=False)
    summary = (
        df.groupby("shape")["relative_mass_error_vs_analytical"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary = summary.rename(columns={
        "mean": "mean_vs_analytical",
        "std": "std_vs_analytical",
        "min": "min_vs_analytical",
        "max": "max_vs_analytical",
    })
    summary["max_abs_vs_analytical"] = df.groupby("shape")["relative_mass_error_vs_analytical"].apply(lambda x: np.max(np.abs(x))).values
    mesh_summary = (
        df.groupby("shape")["relative_mass_error_vs_mesh"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "mean": "mean_vs_mesh",
            "std": "std_vs_mesh",
            "min": "min_vs_mesh",
            "max": "max_vs_mesh",
        })
    )
    mesh_summary["max_abs_vs_mesh"] = df.groupby("shape")["relative_mass_error_vs_mesh"].apply(lambda x: np.max(np.abs(x))).values
    summary = summary.merge(mesh_summary, on="shape", how="left")
    summary.to_csv(out["csv"] / "mass_conservation_orientation_sweep_summary.csv", index=False)
    return df


def run_resolution_convergence(
    out_root: str | Path = "section3_validation_outputs",
    prefer_gpu: bool = True,
) -> pd.DataFrame:
    out = ensure_output_dirs(out_root)

    rows = []

    # Sphere detector-plane convergence against analytical projected density and form factor.
    radius = 1.0
    geom = make_sphere_mesh(radius=radius, resolution=128)
    ray = (0.3, -0.2, 1.0)
    roll = 0.17
    q = np.linspace(-20.0, 20.0, 193, dtype=np.float64)
    A_ref = analytic_sphere_amplitude(q, q, radius=radius)
    I_ref = np.abs(A_ref) ** 2
    volume_ref = 4.0 * np.pi * radius**3 / 3.0
    mesh_volume = float(abs(signed_mesh_volume(geom.vertices, geom.faces)))

    for H in [128, 256, 512, 1024]:
        t0 = time.perf_counter()
        proj, du, dv, meta = project_geom(geom, ray, roll, (H, H), prefer_gpu=prefer_gpu)
        t1 = time.perf_counter()
        u_axis, v_axis = uv_axes_from_projection_meta(meta, du, dv)
        T_ref = analytic_sphere_projected_thickness(u_axis, v_axis, radius=radius)
        A = czt2_from_real_axes(proj, u_axis, v_axis, q, q)
        I = np.abs(A) ** 2
        t2 = time.perf_counter()
        mass = float(np.sum(proj) * du * dv)
        rows.append(
            {
                "case": "sphere_detector_resolution",
                "image_size": H,
                "mesh_resolution": 128,
                "mesh_faces": int(len(geom.faces)),
                "projection_time_s": float(t1 - t0),
                "czt_time_s": float(t2 - t1),
                "total_projection_czt_time_s": float(t2 - t0),
                "projected_density_rel_l2": relative_l2(proj, T_ref),
                "intensity_rel_l2": relative_l2(I, I_ref),
                "pattern_corr": normalized_correlation(I, I_ref),
                "mass_relative_error_vs_analytical": float((mass - volume_ref) / volume_ref),
                "mass_relative_error_vs_mesh": float((mass - mesh_volume) / mesh_volume),
                "projection_method": meta["projection_method"],
            }
        )

    # Mesh complexity convergence for ellipsoid using analytic reference.
    axes = (1.0, 0.65, 0.42)
    R = rotation_matrix_from_euler_xyz(0.33, -0.51, 0.27)
    ray = (0.25, 0.15, 1.0)
    roll = -0.21
    volume_ref = 4.0 * np.pi * axes[0] * axes[1] * axes[2] / 3.0
    q = np.linspace(-20.0, 20.0, 193, dtype=np.float64)
    A_ref = analytic_ellipsoid_amplitude(q, q, axes=axes, rotation=R, ray_vector=ray, roll_rad=roll)
    I_ref = np.abs(A_ref) ** 2

    for mesh_res in [24, 40, 64, 96, 128]:
        geom = make_ellipsoid_mesh(axes=axes, resolution=mesh_res, rotation=R)
        mesh_volume = float(abs(signed_mesh_volume(geom.vertices, geom.faces)))
        t0 = time.perf_counter()
        proj, du, dv, meta = project_geom(geom, ray, roll, (512, 512), prefer_gpu=prefer_gpu)
        t1 = time.perf_counter()
        u_axis, v_axis = uv_axes_from_projection_meta(meta, du, dv)
        T_ref = analytic_ellipsoid_projected_thickness(u_axis, v_axis, axes=axes, rotation=R, ray_vector=ray, roll_rad=roll)
        A = czt2_from_real_axes(proj, u_axis, v_axis, q, q)
        I = np.abs(A) ** 2
        t2 = time.perf_counter()
        mass = float(np.sum(proj) * du * dv)
        rows.append(
            {
                "case": "ellipsoid_mesh_resolution",
                "image_size": 512,
                "mesh_resolution": mesh_res,
                "mesh_faces": int(len(geom.faces)),
                "projection_time_s": float(t1 - t0),
                "czt_time_s": float(t2 - t1),
                "total_projection_czt_time_s": float(t2 - t0),
                "projected_density_rel_l2": relative_l2(proj, T_ref),
                "intensity_rel_l2": relative_l2(I, I_ref),
                "pattern_corr": normalized_correlation(I, I_ref),
                "mass_relative_error_vs_analytical": float((mass - volume_ref) / volume_ref),
                "mass_relative_error_vs_mesh": float((mass - mesh_volume) / mesh_volume),
                "projection_method": meta["projection_method"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out["csv"] / "resolution_convergence.csv", index=False)
    return df


def run_czt_scaling_validation(
    out_root: str | Path = "section3_validation_outputs",
    prefer_gpu: bool = True,
) -> Dict:
    out = ensure_output_dirs(out_root)

    geom = make_sphere_mesh(radius=1.0, resolution=96)
    ray = (0.3, -0.2, 1.0)
    roll = 0.17
    proj, du, dv, meta = project_geom(geom, ray, roll, (192, 192), prefer_gpu=prefer_gpu)
    u_axis, v_axis = uv_axes_from_projection_meta(meta, du, dv)

    q = np.linspace(-18.0, 18.0, 49, dtype=np.float64)

    A_czt = czt2_from_real_axes(proj, u_axis, v_axis, q, q)
    A_dft = direct_dft2_from_real_axes(proj, u_axis, v_axis, q, q)

    # Translation test: shifting real-space coordinate axes should multiply amplitude by phase only.
    t_u, t_v = 0.137, -0.091
    A_shift = czt2_from_real_axes(proj, u_axis + t_u, v_axis + t_v, q, q)
    QU, QV = np.meshgrid(q, q, indexing="xy")
    phase = np.exp(-1j * (QU * t_u + QV * t_v))
    A_shift_expected = A_czt * phase

    I = np.abs(A_czt) ** 2
    I_shift = np.abs(A_shift) ** 2

    zero_idx = zero_frequency_index(q)
    mass = float(np.sum(proj) * du * dv)
    A00 = complex(A_czt[zero_idx, zero_idx])

    metrics = {
        "case": "czt_scaling_validation",
        "qmax": float(np.max(np.abs(q))),
        "nq": int(len(q)),
        "czt_vs_direct_dft_amplitude_rel_l2": relative_l2(A_czt, A_dft),
        "czt_vs_direct_dft_intensity_rel_l2": relative_l2(np.abs(A_czt) ** 2, np.abs(A_dft) ** 2),
        "translation_amplitude_phase_rel_l2": relative_l2(A_shift, A_shift_expected),
        "translation_intensity_rel_l2": relative_l2(I_shift, I),
        "A00_real": float(np.real(A00)),
        "A00_imag": float(np.imag(A00)),
        "projected_mass": mass,
        "A00_mass_relative_error": float((np.real(A00) - mass) / (abs(mass) + 1e-30)),
        "q_zero_value": float(q[zero_idx]),
    }

    np.savez_compressed(
        out["npz"] / "czt_scaling_validation.npz",
        projected_density=proj,
        u_axis=u_axis,
        v_axis=v_axis,
        q=q,
        A_czt=A_czt,
        A_dft=A_dft,
        A_shift=A_shift,
        A_shift_expected=A_shift_expected,
        I=I,
        I_shift=I_shift,
    )
    save_json(out["json"] / "czt_scaling_validation.json", metrics)
    pd.DataFrame([metrics]).to_csv(out["csv"] / "czt_scaling_validation.csv", index=False)
    return metrics


def run_all_section3_data_acquisition(
    out_root: str | Path = "section3_validation_outputs",
    prefer_gpu: bool = True,
    n_orientations: int = 200,
) -> Dict:
    check_required_symbols()
    out = ensure_output_dirs(out_root)
    env = collect_environment_metadata(prefer_gpu=prefer_gpu)
    save_json(out["json"] / "section3_environment.json", env)

    print(f"[section3] Output root: {out['root'].resolve()}")
    print(f"[section3] Projection backend: {env['projection_backend']}")

    all_metrics = {
        "script_version": SCRIPT_VERSION,
        "environment_json": str(out["json"] / "section3_environment.json"),
    }

    print("[1/6] Sphere analytical validation")
    all_metrics["sphere"] = run_sphere_analytical_validation(out_root=out_root, prefer_gpu=prefer_gpu)

    print("[2/6] Ellipsoid analytical validation")
    all_metrics["ellipsoid"] = run_ellipsoid_analytical_validation(out_root=out_root, prefer_gpu=prefer_gpu)

    print("[3/6] Box / polyhedron validation")
    all_metrics["box_polyhedron"] = run_box_polyhedron_validation(out_root=out_root, prefer_gpu=prefer_gpu)

    print("[4/6] Projected-mass conservation orientation sweep")
    mass_df = run_mass_conservation_orientation_sweep(
        out_root=out_root,
        n_orientations=n_orientations,
        prefer_gpu=prefer_gpu,
    )
    all_metrics["mass_sweep_rows"] = int(len(mass_df))

    print("[5/6] Mesh and detector-plane resolution convergence")
    conv_df = run_resolution_convergence(out_root=out_root, prefer_gpu=prefer_gpu)
    all_metrics["resolution_convergence_rows"] = int(len(conv_df))

    print("[6/6] CZT scaling validation")
    all_metrics["czt"] = run_czt_scaling_validation(out_root=out_root, prefer_gpu=prefer_gpu)

    save_json(out["json"] / "all_section3_metrics.json", all_metrics)

    # Flatten selected metrics into a compact summary CSV.
    compact_rows = []
    for key in ["sphere", "ellipsoid", "box_polyhedron", "czt"]:
        d = all_metrics.get(key, {})
        if isinstance(d, dict):
            row = {"validation_block": key}
            for k, v in d.items():
                if isinstance(v, (str, int, float, bool)):
                    row[k] = v
            compact_rows.append(row)
    pd.DataFrame(compact_rows).to_csv(out["csv"] / "all_section3_compact_summary.csv", index=False)

    print("[section3] Done.")
    print(f"[section3] CSV summaries: {out['csv']}")
    print(f"[section3] NPZ arrays:     {out['npz']}")
    print(f"[section3] JSON metrics:   {out['json']}")
    return all_metrics


# Uncomment this line if you want automatic execution after pasting this cell.
# run_all_section3_data_acquisition(prefer_gpu=True, n_orientations=200)
