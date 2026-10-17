from __future__ import annotations

"""
Structure -> mesh / optional BVH(scene) -> projected density engine.

This module provides two production-oriented entry points:

1) build_structure_geometry(structure_type, **params, build_scene=False)
   - Builds a structure mesh from parameters.
   - Computes outward face normals.
   - Optionally builds an Open3D RaycastingScene only when pairwise reference
     projection is needed.
   - Returns a StructureGeometry object.

2) project_density(geom, ray_vector, roll, image_shape=(500,500), ...)
   - For signed_raster / signed_raster_gpu, uses only mesh vertices, faces,
     and face_normals. No BVH is required.
   - For pairwise / signed list_intersections reference methods, requires
     geom.scene, i.e. build_scene=True.
   - Returns projected density and (du, dv).

`build_bvh(...)` is retained as a backward-compatible wrapper with
build_scene=True by default.

Supported structure types:
- "coned-droplet"
- "blob"
- "polyhedron"

Notes
-----
- The acceleration structure is managed internally by Open3D's RaycastingScene.
  In Python, this scene/BVH is reusable in-memory, but not directly serialized.
  For cross-session reuse, save the mesh and metadata, then recreate the scene.
- Rotations are axis-angle only (Rodrigues formula). Euler rotations are not used.
- Projection uses an orthographic ray bundle defined by `ray_vector` and `roll`.
- The projected density is the total thickness along the ray direction.
- Output density can be float16/float32/float64.

Requires:
    pip install open3d pyvista vtk scikit-image scipy

CZT-based complex amplitude from a projected density, using q-axes defined from a
PONI file with BEAM-CENTER estimation.

Key point
---------
PONI is not generally equal to the beam center when detector tilts are present.
This module estimates the beam center from the full pyFAI geometry and uses that
beam center to build the separable Cartesian q-grid needed for the CZT.

Fourier convention
------------------
    A(q_u, q_v) = ∬ rho(u,v) exp[-i (q_u u + q_v v)] du dv

Array convention
----------------
    axis 0 -> v (rows)
    axis 1 -> u (cols)

Scaling law for a unit projected density
----------------------------------------
If rho_1(u,v) is the projected density of a unit object and the isotropic real-space
scale is s, then
    # rho_s(u,v) = s * rho_1(u/s, v/s)
    A_s(q_u,q_v) = s^3 * A_1(s q_u, s q_v)

Translation law
---------------
For a real-space translation (t_u, t_v),
    A_s^trans(q_u,q_v) = A_s(q_u,q_v) * exp[-i (q_u t_u + q_v t_v)]

PONI/beam center
----------------
The beam center is estimated from the pyFAI geometry by solving for detector pixel
coordinates whose mapped laboratory coordinates satisfy t1 = 0 and t2 = 0.
"""


import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

_MODULE_DIR = Path(__file__).resolve().parent
_RUNTIME_CACHE_DIR = _MODULE_DIR / ".runtime_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_CACHE_DIR / "matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR", str(_RUNTIME_CACHE_DIR / "cupy"))
for _cache_dir in (Path(os.environ["MPLCONFIGDIR"]), Path(os.environ["CUPY_CACHE_DIR"])):
    try:
        _cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

_DLL_DIR_HANDLES = []
if getattr(sys, "frozen", False):
    _BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", _MODULE_DIR))
    _nvidia_root = _BUNDLE_DIR / "nvidia"
    if _nvidia_root.exists():
        for _bin_dir in sorted(_nvidia_root.glob("*/bin")):
            if _bin_dir.is_dir():
                try:
                    _DLL_DIR_HANDLES.append(os.add_dll_directory(str(_bin_dir)))
                except (AttributeError, OSError):
                    pass
                os.environ["PATH"] = str(_bin_dir) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import czt
from skimage.measure import marching_cubes

import ast

_OPEN3D_MODULE = None


def _get_open3d_module():
    """Import Open3D only for reference raycasting, decimation, or mesh IO."""
    global _OPEN3D_MODULE
    if _OPEN3D_MODULE is None:
        try:
            import open3d as o3d
        except Exception as exc:
            raise ImportError(
                "Open3D is required only for build_scene=True, mesh decimation, "
                "or mesh save/load operations."
            ) from exc
        _OPEN3D_MODULE = o3d
    return _OPEN3D_MODULE

# ============================================================
# 0. Linear algebra helpers
# ============================================================
def normalize(v: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


def rotation_matrix_axis_angle(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Rodrigues rotation formula; local -> world rotation."""
    a = normalize(axis)
    x, y, z = a
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def compose_axis_rotations(rotations: Sequence[Tuple[Sequence[float], float]]) -> np.ndarray:
    """Compose axis-angle rotations in listed order; returns local -> world rotation."""
    R = np.eye(3, dtype=np.float64)
    for axis, angle in rotations:
        R = rotation_matrix_axis_angle(axis, angle) @ R
    return R


def world_to_local(points: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """points: (N,3), rotation: local -> world"""
    return (points - center[None, :]) @ rotation


def local_to_world(points: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """points: (N,3), rotation: local -> world"""
    return points @ rotation.T + center[None, :]


def smooth_max(values: np.ndarray, k: float = 12.0, axis: int = 0) -> np.ndarray:
    """Smooth approximation of max(values, axis=axis)."""
    v = np.asarray(values, dtype=np.float64)
    vmax = np.max(v, axis=axis, keepdims=True)
    out = vmax + np.log(np.sum(np.exp(k * (v - vmax)), axis=axis, keepdims=True)) / k
    return np.squeeze(out, axis=axis)


def corners_from_bounds(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    return np.array(
        [
            [x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
            [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1],
        ],
        dtype=np.float64,
    )


# ============================================================
# 1. Structure definitions (implicit field)
# ============================================================
@dataclass
class Structure:
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    def field(self, points_world: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def local_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def world_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        bmin, bmax = self.local_bounds()
        corners_local = corners_from_bounds(bmin, bmax)
        corners_world = local_to_world(corners_local, self.center, self.rotation)
        return corners_world.min(axis=0), corners_world.max(axis=0)


@dataclass
class ConcaveConeDroplet(Structure):
    """
    Ellipsoidal body + concave cone/tail along local +z.

    Cone radius profile:
        r(t) = r_tip + (r_neck - r_tip) * (1 - t^alpha)^beta,  0 <= t <= 1

    Requested physical regime:
        alpha = 1, beta in [2,3]  -> concave sidewall.
    """
    body_axes: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float64))
    cone_length: float = 1.5
    neck_radius: float = 0.35
    tip_radius: float = 0.05
    z_attach: float = 0.70
    alpha: float = 1.0
    beta: float = 2.5
    blend_k: float = 12.0

    @classmethod
    def spherical_body(
        cls,
        radius: float = 1.0,
        cone_length: float = 1.5,
        neck_radius: float = 0.35,
        tip_radius: float = 0.05,
        z_attach: float = 0.70,
        alpha: float = 1.0,
        beta: float = 2.5,
        center: Sequence[float] = (0.0, 0.0, 0.0),
        rotation: Optional[np.ndarray] = None,
        blend_k: float = 12.0,
    ) -> "ConcaveConeDroplet":
        if rotation is None:
            rotation = np.eye(3, dtype=np.float64)
        return cls(
            center=np.asarray(center, dtype=np.float64),
            rotation=np.asarray(rotation, dtype=np.float64),
            body_axes=np.array([radius, radius, radius], dtype=np.float64),
            cone_length=float(cone_length),
            neck_radius=float(neck_radius),
            tip_radius=float(tip_radius),
            z_attach=float(z_attach),
            alpha=float(alpha),
            beta=float(beta),
            blend_k=float(blend_k),
        )

    @classmethod
    def ellipsoidal_body(
        cls,
        body_axes: Sequence[float],
        cone_length: float = 1.5,
        neck_radius: float = 0.35,
        tip_radius: float = 0.05,
        z_attach: float = 0.70,
        alpha: float = 1.0,
        beta: float = 2.5,
        center: Sequence[float] = (0.0, 0.0, 0.0),
        rotation: Optional[np.ndarray] = None,
        blend_k: float = 12.0,
    ) -> "ConcaveConeDroplet":
        if rotation is None:
            rotation = np.eye(3, dtype=np.float64)
        return cls(
            center=np.asarray(center, dtype=np.float64),
            rotation=np.asarray(rotation, dtype=np.float64),
            body_axes=np.asarray(body_axes, dtype=np.float64),
            cone_length=float(cone_length),
            neck_radius=float(neck_radius),
            tip_radius=float(tip_radius),
            z_attach=float(z_attach),
            alpha=float(alpha),
            beta=float(beta),
            blend_k=float(blend_k),
        )

    def _cone_radius(self, t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        return self.tip_radius + (self.neck_radius - self.tip_radius) * ((1.0 - t ** self.alpha) ** self.beta)

    def field(self, points_world: np.ndarray) -> np.ndarray:
        p = world_to_local(points_world, self.center, self.rotation)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]

        a, b, c = self.body_axes
        body = 1.0 - (x / a) ** 2 - (y / b) ** 2 - (z / c) ** 2

        z0 = self.z_attach * c
        z1 = z0 + self.cone_length
        t = (z - z0) / max(self.cone_length, 1e-12)
        rz = self._cone_radius(t)
        rxy = np.sqrt(x * x + y * y)

        side = (rz - rxy) / max(self.neck_radius, self.tip_radius, 1e-12)
        zcap = np.minimum(z - z0, z1 - z) / max(self.cone_length, 1e-12)
        cone = np.minimum(side, zcap)

        return smooth_max(np.stack([body, cone], axis=0), k=self.blend_k, axis=0)

    def local_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        a, b, c = self.body_axes
        z0 = self.z_attach * c
        z1 = z0 + self.cone_length
        rmax = max(a, b, self.neck_radius, self.tip_radius)
        bmin = np.array([-rmax, -rmax, -c], dtype=np.float64)
        bmax = np.array([ rmax,  rmax,  z1], dtype=np.float64)
        return bmin, bmax


@dataclass
class AmorphousBlob(Structure):
    kernel_centers: np.ndarray = field(default_factory=lambda: np.zeros((1, 3), dtype=np.float64))
    kernel_sigmas: np.ndarray = field(default_factory=lambda: np.ones((1, 3), dtype=np.float64) * 0.3)
    kernel_weights: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=np.float64))
    threshold: float = 0.55

    def field(self, points_world: np.ndarray) -> np.ndarray:
        p = world_to_local(points_world, self.center, self.rotation)
        diff = p[:, None, :] - self.kernel_centers[None, :, :]
        u = diff / np.maximum(self.kernel_sigmas[None, :, :], 1e-12)
        g = np.exp(-0.5 * np.sum(u * u, axis=2)) * self.kernel_weights[None, :]
        return np.sum(g, axis=1) - self.threshold

    def local_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        r = 4.0 * self.kernel_sigmas
        bmin = np.min(self.kernel_centers - r, axis=0)
        bmax = np.max(self.kernel_centers + r, axis=0)
        return bmin, bmax


def polyhedron_vertices(name: str) -> np.ndarray:
    name = name.lower()

    if name == "tetra":
        v = np.array([
            [ 1,  1,  1],
            [ 1, -1, -1],
            [-1,  1, -1],
            [-1, -1,  1],
        ], dtype=np.float64)
    elif name == "cube":
        v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    elif name == "octa":
        v = np.array([
            [ 1, 0, 0], [-1, 0, 0],
            [ 0, 1, 0], [ 0,-1, 0],
            [ 0, 0, 1], [ 0, 0,-1],
        ], dtype=np.float64)
    elif name == "icosa":
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        v = np.array([
            [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
            [1,  phi, 0], [-1,  phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0,  1], [phi, 0, -1], [-phi, 0,  1], [-phi, 0, -1],
        ], dtype=np.float64)
    elif name == "dodeca":
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        iphi = 1.0 / phi
        v = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    v.append([sx, sy, sz])
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                v.append([0, s1 * iphi, s2 * phi])
                v.append([s1 * iphi, s2 * phi, 0])
                v.append([s1 * phi, 0, s2 * iphi])
        v = np.asarray(v, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported polyhedron name: {name}")

    r = np.linalg.norm(v, axis=1).max()
    return v / r


@dataclass
class RegularPolyhedron(Structure):
    name: str = "cube"
    radius: float = 1.0

    def __post_init__(self):
        self._vertices = polyhedron_vertices(self.name) * self.radius
        hull = ConvexHull(self._vertices)
        # equations: n.x + c <= 0 inside
        self._eq = hull.equations.copy()

    def field(self, points_world: np.ndarray) -> np.ndarray:
        p = world_to_local(points_world, self.center, self.rotation)
        vals = - (p @ self._eq[:, :3].T + self._eq[:, 3][None, :])
        return np.min(vals, axis=1)

    def local_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._vertices.min(axis=0), self._vertices.max(axis=0)


# ============================================================
# 2. Structure factory
# ============================================================
def _parse_rotation(rotation: Optional[np.ndarray] = None,
                    rotations: Optional[Sequence[Tuple[Sequence[float], float]]] = None) -> np.ndarray:
    if rotation is not None and rotations is not None:
        raise ValueError("Provide either `rotation` or `rotations`, not both.")
    if rotation is not None:
        R = np.asarray(rotation, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError("rotation must be (3,3).")
        return R
    if rotations is not None:
        return compose_axis_rotations(rotations)
    return np.eye(3, dtype=np.float64)


def make_structure(structure_type: str, **params) -> Structure:
    """
    Create a Structure instance from parameters.

    Supported types:
    - coned-droplet
    - blob
    - polyhedron

    Rotation:
    - Either provide `rotation=(3,3)`
    - Or `rotations=[([ax,ay,az], angle_rad), ...]`
    """
    st = structure_type.lower().strip()
    center = np.asarray(params.pop("center", [0.0, 0.0, 0.0]), dtype=np.float64)
    R = _parse_rotation(params.pop("rotation", None), params.pop("rotations", None))

    if st == "coned-droplet":
        body_axes = params.pop("body_axes", None)
        radius = params.pop("radius", None)

        common = dict(
            cone_length=float(params.pop("cone_length", 1.5)),
            neck_radius=float(params.pop("neck_radius", 0.35)),
            tip_radius=float(params.pop("tip_radius", 0.05)),
            z_attach=float(params.pop("z_attach", 0.70)),
            alpha=float(params.pop("alpha", 1.0)),
            beta=float(params.pop("beta", 2.5)),
            center=center,
            rotation=R,
            blend_k=float(params.pop("blend_k", 12.0)),
        )
        if params:
            raise ValueError(f"Unused parameters for coned-droplet: {sorted(params.keys())}")

        if body_axes is not None:
            return ConcaveConeDroplet.ellipsoidal_body(body_axes=body_axes, **common)
        return ConcaveConeDroplet.spherical_body(radius=float(radius if radius is not None else 1.0), **common)

    if st == "blob":
        required = ["kernel_centers", "kernel_sigmas", "kernel_weights"]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"blob requires parameters: {missing}")
        structure = AmorphousBlob(
            center=center,
            rotation=R,
            kernel_centers=np.asarray(params.pop("kernel_centers"), dtype=np.float64),
            kernel_sigmas=np.asarray(params.pop("kernel_sigmas"), dtype=np.float64),
            kernel_weights=np.asarray(params.pop("kernel_weights"), dtype=np.float64),
            threshold=float(params.pop("threshold", 0.55)),
        )
        if params:
            raise ValueError(f"Unused parameters for blob: {sorted(params.keys())}")
        return structure

    if st == "polyhedron":
        structure = RegularPolyhedron(
            center=center,
            rotation=R,
            name=str(params.pop("name", "cube")),
            radius=float(params.pop("radius", 1.0)),
        )
        if params:
            raise ValueError(f"Unused parameters for polyhedron: {sorted(params.keys())}")
        return structure

    raise ValueError(f"Unsupported structure_type: {structure_type}")



# ============================================================
# 2.5. Mesh orientation / face-normal helpers
# ============================================================
def compute_face_normals(vertices: np.ndarray, faces: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return unit face normals using the current triangle winding."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, eps)


def signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Signed volume of a closed triangular mesh.

    Positive/negative sign is determined by triangle winding. For a consistently
    oriented closed mesh, flipping faces changes only the sign.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    v0 = tri[:, 0]
    v1 = tri[:, 1]
    v2 = tri[:, 2]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def orient_faces_outward_by_volume(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """
    Force a closed mesh to have positive signed volume by flipping triangle winding if needed.

    Returns
    -------
    faces_oriented : np.ndarray
    signed_volume_after : float
    flipped : bool
    """
    faces = np.asarray(faces, dtype=np.int32)
    vol = signed_mesh_volume(vertices, faces)
    flipped = False
    if vol < 0:
        faces = faces[:, [0, 2, 1]].copy()
        vol = -vol
        flipped = True
    return faces.astype(np.int32, copy=False), float(vol), bool(flipped)


def orient_convex_hull_simplices_outward(
    vertices_local: np.ndarray,
    hull: ConvexHull,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, int]:
    """
    Return ConvexHull simplices with *per-triangle* outward winding.

    Why this is needed
    ------------------
    scipy.spatial.ConvexHull provides triangular simplices and outward plane
    equations, but relying only on a global signed-volume flip is not enough:
    individual simplices can still have mixed winding. Signed rasterization is
    sensitive to the sign of each face normal, so every triangle must be
    oriented consistently.

    ConvexHull.equations[i, :3] is the outward normal of simplices[i]. We flip
    simplices[i] when its cross-product normal points against that equation
    normal.
    """
    vertices_local = np.asarray(vertices_local, dtype=np.float64)
    faces = np.asarray(hull.simplices, dtype=np.int32).copy()
    target_normals = np.asarray(hull.equations[:, :3], dtype=np.float64)

    tri = vertices_local[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n_unit = n / np.maximum(n_norm, eps)

    dots = np.einsum("ij,ij->i", n_unit, target_normals)
    flip = dots < 0.0
    if np.any(flip):
        faces[flip] = faces[flip][:, [0, 2, 1]]

    return faces.astype(np.int32, copy=False), int(np.count_nonzero(flip))


def orient_convex_faces_outward_by_centroid(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Fallback per-face orientation for convex meshes.

    A triangle normal should point approximately from the mesh centroid toward
    the triangle centroid. This is less exact than using ConvexHull.equations,
    but useful for externally supplied convex triangular meshes.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32).copy()

    mesh_center = vertices.mean(axis=0)
    tri = vertices[faces]
    face_center = tri.mean(axis=1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward_hint = face_center - mesh_center[None, :]
    dots = np.einsum("ij,ij->i", n, outward_hint)

    flip = dots < 0.0
    if np.any(flip):
        faces[flip] = faces[flip][:, [0, 2, 1]]

    return faces.astype(np.int32, copy=False), int(np.count_nonzero(flip))


def regular_polyhedron_to_mesh(structure: RegularPolyhedron) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build a direct triangular surface mesh for RegularPolyhedron without
    sampling a 3D scalar field or running marching_cubes.

    Critical implementation detail
    ------------------------------
    Signed rasterization requires every face normal to point outward. A global
    signed-volume flip can only reverse the whole mesh; it cannot fix mixed
    per-face winding. Therefore, ConvexHull simplices are oriented individually
    using ConvexHull.equations before the mesh is returned.
    """
    verts_local = np.asarray(structure._vertices, dtype=np.float64)
    hull = ConvexHull(verts_local)

    # Per-triangle orientation using ConvexHull outward plane equations.
    faces, per_face_flipped = orient_convex_hull_simplices_outward(verts_local, hull)

    # Apply user-specified rigid rotation/translation after local orientation.
    verts = local_to_world(verts_local, structure.center, structure.rotation)

    # Defensive global volume check. This should normally not flip anything
    # after per-face orientation, but keeps the output convention consistent.
    faces, signed_volume, global_flipped = orient_faces_outward_by_volume(verts, faces)

    # Extra sanity check: for a convex mesh, each face normal should point away
    # from the mesh centroid. If something unusual happened, repair once more.
    face_normals_check = compute_face_normals(verts, faces)
    tri = verts[faces]
    mesh_center = verts.mean(axis=0)
    face_center = tri.mean(axis=1)
    outward_hint = face_center - mesh_center[None, :]
    bad = np.einsum("ij,ij->i", face_normals_check, outward_hint) < -1e-10
    centroid_repaired = 0
    if np.any(bad):
        faces[bad] = faces[bad][:, [0, 2, 1]]
        centroid_repaired = int(np.count_nonzero(bad))
        signed_volume = abs(signed_mesh_volume(verts, faces))

    meta = {
        "mesh_source": "direct_convex_hull",
        "polyhedron_name": str(structure.name),
        "mesh_vertex_count": int(len(verts)),
        "mesh_face_count": int(len(faces)),
        "signed_volume": float(signed_volume),
        "per_face_flipped_using_hull_equations": int(per_face_flipped),
        "global_faces_flipped_for_positive_volume": bool(global_flipped),
        "centroid_orientation_repairs": int(centroid_repaired),
    }
    return verts.astype(np.float64, copy=False), faces.astype(np.int32, copy=False), meta


def decimate_mesh_open3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_num_faces: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Optional mesh simplification using Open3D quadric decimation.

    Use this only when the marching-cubes mesh is too dense for projection.
    For low-q SAXS screening, a reduced mesh can be sufficient, but it should
    always be validated by projected mass and SAXS intensity comparisons.
    """
    if target_num_faces is None:
        return vertices, faces, {"decimated": False}

    target_num_faces = int(target_num_faces)
    if target_num_faces <= 0 or len(faces) <= target_num_faces:
        return vertices, faces, {
            "decimated": False,
            "target_num_faces": target_num_faces,
            "input_face_count": int(len(faces)),
            "output_face_count": int(len(faces)),
        }

    o3d = _get_open3d_module()
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))

    legacy.remove_duplicated_vertices()
    legacy.remove_duplicated_triangles()
    legacy.remove_degenerate_triangles()
    legacy.remove_non_manifold_edges()
    legacy.compute_vertex_normals()

    dec = legacy.simplify_quadric_decimation(target_number_of_triangles=target_num_faces)
    dec.remove_duplicated_vertices()
    dec.remove_duplicated_triangles()
    dec.remove_degenerate_triangles()
    dec.compute_triangle_normals()

    v = np.asarray(dec.vertices, dtype=np.float64)
    f = np.asarray(dec.triangles, dtype=np.int32)
    f, signed_volume, faces_flipped = orient_faces_outward_by_volume(v, f)

    return v, f, {
        "decimated": True,
        "target_num_faces": int(target_num_faces),
        "input_face_count": int(len(faces)),
        "output_face_count": int(len(f)),
        "signed_volume_after_decimation": float(signed_volume),
        "faces_flipped_after_decimation": bool(faces_flipped),
    }


# ============================================================
# 3. Structure -> mesh
# ============================================================
def _normalize_kernel_sigmas(sigmas: np.ndarray, n_kernels: int) -> np.ndarray:
    """
    Normalize Gaussian kernel sigmas to shape (K, 3).

    Accepted forms:
    - scalar: one isotropic sigma for all kernels
    - (K,): isotropic sigma for each kernel
    - (3,): one anisotropic sigma vector shared by all kernels
    - (K, 3): anisotropic sigma for each kernel
    """
    sigmas = np.asarray(sigmas, dtype=np.float64)

    if sigmas.ndim == 0:
        return np.full((n_kernels, 3), float(sigmas), dtype=np.float64)

    if sigmas.ndim == 1:
        if sigmas.shape[0] == n_kernels:
            return np.repeat(sigmas[:, None], 3, axis=1).astype(np.float64)
        if sigmas.shape[0] == 3:
            return np.repeat(sigmas[None, :], n_kernels, axis=0).astype(np.float64)
        raise ValueError(
            "kernel_sigmas with ndim=1 must have length K or 3. "
            f"Got shape {sigmas.shape} for K={n_kernels}."
        )

    if sigmas.ndim == 2 and sigmas.shape == (n_kernels, 3):
        return sigmas.astype(np.float64)

    raise ValueError(
        "kernel_sigmas should be scalar, shape (K,), shape (3,), or shape (K,3). "
        f"Got shape {sigmas.shape}."
    )


def sample_amorphous_blob_field_local_accumulation(
    structure: AmorphousBlob,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    local_cutoff: float = 4.5,
    dtype: Union[str, np.dtype] = "float32",
) -> np.ndarray:
    """
    Fast scalar-field evaluation for AmorphousBlob on a local-coordinate grid.

    The conventional pointwise evaluation computes every Gaussian kernel at
    every grid point, with cost O(N_grid * N_kernel).  This function instead
    loops over kernels and updates only the subgrid within cutoff * sigma of
    each kernel center. The grid spacing/resolution is unchanged; only
    negligible far-tail evaluations are skipped.

    The returned field is:
        sum_j w_j exp(-0.5 * ||(r-c_j)/sigma_j||^2) - threshold
    """
    dtype = np.dtype(dtype)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)

    centers = np.asarray(structure.kernel_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"kernel_centers should have shape (K,3). Got {centers.shape}.")

    K = centers.shape[0]
    sigmas = _normalize_kernel_sigmas(structure.kernel_sigmas, K)
    weights = np.asarray(structure.kernel_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != K:
        raise ValueError(f"kernel_weights should have length K={K}. Got {weights.shape}.")

    F = np.zeros((len(xs), len(ys), len(zs)), dtype=dtype)
    cutoff = float(local_cutoff)

    for c, s, w in zip(centers, sigmas, weights):
        sx, sy, sz = np.maximum(np.asarray(s, dtype=np.float64), 1e-12)

        ix0 = int(np.searchsorted(xs, c[0] - cutoff * sx, side="left"))
        ix1 = int(np.searchsorted(xs, c[0] + cutoff * sx, side="right"))
        iy0 = int(np.searchsorted(ys, c[1] - cutoff * sy, side="left"))
        iy1 = int(np.searchsorted(ys, c[1] + cutoff * sy, side="right"))
        iz0 = int(np.searchsorted(zs, c[2] - cutoff * sz, side="left"))
        iz1 = int(np.searchsorted(zs, c[2] + cutoff * sz, side="right"))

        ix0, iy0, iz0 = max(ix0, 0), max(iy0, 0), max(iz0, 0)
        ix1, iy1, iz1 = min(ix1, len(xs)), min(iy1, len(ys)), min(iz1, len(zs))

        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue

        gx = np.exp(-0.5 * ((xs[ix0:ix1] - c[0]) / sx) ** 2).astype(dtype, copy=False)
        gy = np.exp(-0.5 * ((ys[iy0:iy1] - c[1]) / sy) ** 2).astype(dtype, copy=False)
        gz = np.exp(-0.5 * ((zs[iz0:iz1] - c[2]) / sz) ** 2).astype(dtype, copy=False)

        # Separable local update.  This avoids constructing a full X,Y,Z grid.
        F[ix0:ix1, iy0:iy1, iz0:iz1] += (
            dtype.type(w)
            * gx[:, None, None]
            * gy[None, :, None]
            * gz[None, None, :]
        )


    F -= dtype.type(structure.threshold)
    return F


def sample_coned_droplet_field_local_direct(
    structure: ConcaveConeDroplet,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    dtype: Union[str, np.dtype] = "float32",
    chunk_points: int = 2_000_000,
) -> np.ndarray:
    """
    Fast scalar-field evaluation for ConcaveConeDroplet on a local-coordinate grid.

    The generic pointwise fallback forms local grid points, transforms them to
    world coordinates, and then calls structure.field(), which transforms them
    back to local coordinates.  That is robust but unnecessarily expensive for
    coned droplets because the implicit function is analytic in the structure's
    local coordinates.

    This function evaluates the same implicit field directly on the local grid:

        body = 1 - (x/a)^2 - (y/b)^2 - (z/c)^2
        cone = min((r(z)-sqrt(x^2+y^2))/r_ref,
                   min(z-z0, z1-z)/cone_length)
        F = smooth_max(body, cone; k=blend_k)

    It uses separable/broadcasted arrays and chunks along z, so the nominal grid
    resolution is unchanged while avoiding the large point-cloud construction and
    redundant coordinate transforms used by the general fallback.
    """
    dtype = np.dtype(dtype)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)

    nx, ny, nz = len(xs), len(ys), len(zs)
    F = np.empty((nx, ny, nz), dtype=dtype)

    a, b, c_axis = np.asarray(structure.body_axes, dtype=np.float64)
    a = max(float(a), 1e-12)
    b = max(float(b), 1e-12)
    c_axis = max(float(c_axis), 1e-12)

    cone_length = max(float(structure.cone_length), 1e-12)
    neck_radius = float(structure.neck_radius)
    tip_radius = float(structure.tip_radius)
    r_ref = max(neck_radius, tip_radius, 1e-12)
    z0 = float(structure.z_attach) * c_axis
    z1 = z0 + cone_length
    alpha = float(structure.alpha)
    beta = float(structure.beta)
    k = float(structure.blend_k)

    # 2D terms reused for all z-chunks.
    x = xs.astype(dtype, copy=False)
    y = ys.astype(dtype, copy=False)
    xy_body = (x[:, None] / dtype.type(a)) ** 2 + (y[None, :] / dtype.type(b)) ** 2
    rxy = np.sqrt(x[:, None] * x[:, None] + y[None, :] * y[None, :]).astype(dtype, copy=False)

    # Choose a z chunk size so body/cone temporary arrays remain modest.
    denom = max(nx * ny, 1)
    z_chunk = max(1, int(chunk_points // denom))
    z_chunk = min(z_chunk, nz)

    for k0 in range(0, nz, z_chunk):
        k1 = min(k0 + z_chunk, nz)
        z = zs[k0:k1].astype(dtype, copy=False)

        # Body term.
        body = (
            dtype.type(1.0)
            - xy_body[:, :, None]
            - (z[None, None, :] / dtype.type(c_axis)) ** 2
        )

        # Cone radius profile r(t).
        t = (zs[k0:k1] - z0) / cone_length
        t = np.clip(t, 0.0, 1.0)
        rz = tip_radius + (neck_radius - tip_radius) * ((1.0 - t ** alpha) ** beta)
        rz = rz.astype(dtype, copy=False)

        side = (rz[None, None, :] - rxy[:, :, None]) / dtype.type(r_ref)
        zcap_1d = np.minimum(zs[k0:k1] - z0, z1 - zs[k0:k1]) / cone_length
        zcap = zcap_1d.astype(dtype, copy=False)[None, None, :]
        cone = np.minimum(side, zcap)

        # Stable two-input smooth_max equivalent to smooth_max(stack([body, cone])).
        m = np.maximum(body, cone)
        if abs(k) < 1e-12:
            # Degenerate fallback; should not normally be used.
            out = np.maximum(body, cone)
        else:
            out = m + np.log(
                np.exp(dtype.type(k) * (body - m))
                + np.exp(dtype.type(k) * (cone - m))
            ) / dtype.type(k)

        F[:, :, k0:k1] = out.astype(dtype, copy=False)

    return F


def sample_structure_field_pointwise_local_grid(
    structure: Structure,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    dtype: Union[str, np.dtype] = "float32",
    chunk_points: int = 2_000_000,
) -> np.ndarray:
    """
    General fallback: evaluate structure.field on a local-coordinate grid.

    The vertices are extracted in the local coordinate frame and transformed to
    world coordinates after isosurface extraction.  This avoids the expanded
    world-axis bounding box for rotated objects while preserving the requested
    local-grid resolution.
    """
    dtype = np.dtype(dtype)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    nx, ny, nz = len(xs), len(ys), len(zs)
    F = np.empty((nx, ny, nz), dtype=dtype)

    # Chunk along x to avoid materializing a very large point cloud at once.
    denom = max(ny * nz, 1)
    x_chunk = max(1, int(chunk_points // denom))

    for i0 in range(0, nx, x_chunk):
        i1 = min(i0 + x_chunk, nx)
        X, Y, Z = np.meshgrid(xs[i0:i1], ys, zs, indexing="ij")
        pts_local = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        pts_world = local_to_world(pts_local, structure.center, structure.rotation)
        vals = structure.field(pts_world).reshape((i1 - i0, ny, nz))
        F[i0:i1, :, :] = vals.astype(dtype, copy=False)

    return F


def sample_structure_field_on_local_grid(
    structure: Structure,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    field_sampling: str = "auto",
    local_cutoff: float = 4.5,
    dtype: Union[str, np.dtype] = "float32",
    chunk_points: int = 2_000_000,
) -> Tuple[np.ndarray, Dict]:
    """Sample an implicit structure on a local-coordinate grid."""
    mode = str(field_sampling).lower().strip()
    t0 = time.perf_counter()

    local_modes = ("auto", "local", "local_kernel", "local-kernel", "direct", "direct_local")

    if mode in local_modes and isinstance(structure, AmorphousBlob):
        F = sample_amorphous_blob_field_local_accumulation(
            structure,
            xs,
            ys,
            zs,
            local_cutoff=local_cutoff,
            dtype=dtype,
        )
        method = "local_kernel_accumulation"
    elif mode in local_modes and isinstance(structure, ConcaveConeDroplet):
        F = sample_coned_droplet_field_local_direct(
            structure,
            xs,
            ys,
            zs,
            dtype=dtype,
            chunk_points=chunk_points,
        )
        method = "coned_droplet_direct_local_grid"
    else:
        if mode in ("local", "local_kernel", "local-kernel", "direct", "direct_local"):
            warnings.warn(
                "field_sampling requested a specialized local evaluator, but no specialized "
                "path is implemented for this structure. Falling back to pointwise local-grid sampling.",
                RuntimeWarning,
            )
        F = sample_structure_field_pointwise_local_grid(
            structure,
            xs,
            ys,
            zs,
            dtype=dtype,
            chunk_points=chunk_points,
        )
        method = "pointwise_local_grid"

    meta = {
        "field_sampling_method": method,
        "field_sampling_seconds": float(time.perf_counter() - t0),
        "field_dtype": str(np.dtype(dtype)),
        "local_cutoff": float(local_cutoff),
    }
    return F, meta


def skimage_marching_cubes_isosurface(
    F: np.ndarray,
    *,
    level: float = 0.0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Extract a triangular isosurface using scikit-image marching_cubes."""
    t0 = time.perf_counter()
    spacing = tuple(float(x) for x in spacing)
    origin = np.asarray(origin, dtype=np.float64)

    verts, faces, normals, values = marching_cubes(
        np.asarray(F, dtype=np.float32),
        level=float(level),
        spacing=spacing,
        allow_degenerate=False,
    )
    verts = verts.astype(np.float64, copy=False)
    verts += origin[None, :]
    faces = faces.astype(np.int32, copy=False)

    meta = {
        "iso_extractor_used": "skimage_marching_cubes",
        "iso_extraction_seconds": float(time.perf_counter() - t0),
    }
    return verts, faces, meta


def vtk_flying_edges_isosurface(
    F: np.ndarray,
    *,
    level: float = 0.0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract a triangular isosurface using VTK FlyingEdges3D via PyVista.

    Flying Edges works on regular image data.  This function uses one global
    uniform grid, not the bricked/octree route.  It is usually faster than
    scikit-image marching_cubes after the scalar field has been sampled.
    """
    t0 = time.perf_counter()
    try:
        import pyvista as pv
    except Exception as exc:
        raise ImportError(
            "vtk_flying_edges requires pyvista and vtk. In Colab, run: "
            "!pip -q install pyvista vtk"
        ) from exc

    F = np.asarray(F, dtype=np.float32)
    nx, ny, nz = F.shape

    grid = pv.ImageData()
    grid.dimensions = (int(nx), int(ny), int(nz))
    grid.origin = tuple(float(x) for x in origin)
    grid.spacing = tuple(float(x) for x in spacing)
    grid.point_data["values"] = F.ravel(order="F")

    surf = grid.contour(
        isosurfaces=[float(level)],
        scalars="values",
        method="flying_edges",
    ).triangulate()

    vertices = np.asarray(surf.points, dtype=np.float64)
    raw_faces = np.asarray(surf.faces, dtype=np.int64)
    if raw_faces.size == 0:
        raise ValueError("No isosurface was extracted. Check bounds, threshold, and field sign.")
    faces = raw_faces.reshape(-1, 4)[:, 1:4].astype(np.int32)

    meta = {
        "iso_extractor_used": "vtk_flying_edges",
        "iso_extraction_seconds": float(time.perf_counter() - t0),
    }
    return vertices, faces, meta


def extract_isosurface_from_field(
    F: np.ndarray,
    *,
    level: float = 0.0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    iso_extractor: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Dispatch to the selected isosurface extractor."""
    method = str(iso_extractor).lower().strip()

    if method in ("vtk", "vtk_flying_edges", "flying_edges", "flyingedges"):
        return vtk_flying_edges_isosurface(F, level=level, origin=origin, spacing=spacing)

    if method in ("skimage", "marching_cubes", "skimage_marching_cubes"):
        return skimage_marching_cubes_isosurface(F, level=level, origin=origin, spacing=spacing)

    if method == "auto":
        try:
            return vtk_flying_edges_isosurface(F, level=level, origin=origin, spacing=spacing)
        except Exception as exc:
            warnings.warn(
                f"VTK FlyingEdges failed ({exc!r}); falling back to skimage marching_cubes.",
                RuntimeWarning,
            )
            return skimage_marching_cubes_isosurface(F, level=level, origin=origin, spacing=spacing)

    raise ValueError(
        "iso_extractor should be 'vtk_flying_edges', 'skimage_marching_cubes', or 'auto'. "
        f"Got {iso_extractor!r}."
    )


def structure_to_mesh(
    structure: Structure,
    field_shape: Tuple[int, int, int] = (150, 150, 150),
    safety_cells: int = 1,
    use_direct_polyhedron_mesh: bool = True,
    target_num_faces: Optional[int] = None,
    field_sampling: str = "auto",
    local_cutoff: float = 4.5,
    iso_extractor: str = "auto",
    field_dtype: Union[str, np.dtype] = "float32",
    pointwise_chunk_points: int = 2_000_000,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convert a Structure object into a closed triangular mesh.

    Fast paths
    ----------
    RegularPolyhedron:
        If use_direct_polyhedron_mesh=True, build the polyhedron mesh directly.
        This avoids scalar-field sampling and isosurface extraction.

    AmorphousBlob:
        With field_sampling='auto' or 'local', evaluate the Gaussian field by
        local-kernel accumulation. The nominal grid resolution is unchanged;
        only negligible far-tail kernel evaluations are skipped.

    ConcaveConeDroplet:
        With field_sampling='auto', 'local', or 'direct', evaluate the analytic
        droplet field directly on the local grid. This avoids point-cloud
        construction and redundant world/local coordinate transforms.

    General implicit structures:
        Sample the implicit field on a local-coordinate uniform grid and extract
        one global isosurface, preferably by VTK FlyingEdges3D.

    Optional decimation
    -------------------
    target_num_faces:
        If provided, Open3D quadric decimation is applied after mesh generation.
        This can reduce signed-raster projection time, but should be validated
        against projected mass and SAXS intensity.
    """
    if use_direct_polyhedron_mesh and isinstance(structure, RegularPolyhedron):
        verts, faces, meta = regular_polyhedron_to_mesh(structure)
        if target_num_faces is not None:
            verts, faces, dec_meta = decimate_mesh_open3d(verts, faces, target_num_faces)
            meta.update({f"decimation_{k}": v for k, v in dec_meta.items()})
            meta["mesh_vertex_count"] = int(len(verts))
            meta["mesh_face_count"] = int(len(faces))
        return verts, faces, meta

    field_shape = np.asarray(field_shape, dtype=int)
    if np.any(field_shape < 8):
        raise ValueError("field_shape should be at least around 8 in each axis.")

    # Use the structure-local bounding box and transform extracted vertices to world
    # coordinates afterwards. This avoids an unnecessarily enlarged world-axis box
    # when the object is rotated.
    bmin, bmax = structure.local_bounds()
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    step = (bmax - bmin) / np.maximum(field_shape - 1, 1)
    bmin_s = bmin - safety_cells * step
    bmax_s = bmax + safety_cells * step

    xs = np.linspace(bmin_s[0], bmax_s[0], field_shape[0])
    ys = np.linspace(bmin_s[1], bmax_s[1], field_shape[1])
    zs = np.linspace(bmin_s[2], bmax_s[2], field_shape[2])
    spacing = (float(xs[1] - xs[0]), float(ys[1] - ys[0]), float(zs[1] - zs[0]))

    F, field_meta = sample_structure_field_on_local_grid(
        structure,
        xs,
        ys,
        zs,
        field_sampling=field_sampling,
        local_cutoff=local_cutoff,
        dtype=field_dtype,
        chunk_points=pointwise_chunk_points,
    )

    verts_local, faces, iso_meta = extract_isosurface_from_field(
        F,
        level=0.0,
        origin=bmin_s,
        spacing=spacing,
        iso_extractor=iso_extractor,
    )

    verts = local_to_world(verts_local, structure.center, structure.rotation)
    verts = verts.astype(np.float64, copy=False)
    faces = faces.astype(np.int32, copy=False)
    faces, signed_volume, faces_flipped = orient_faces_outward_by_volume(verts, faces)

    meta = {
        "mesh_source": "local_field_one_shot_isosurface",
        "field_shape": field_shape.tolist(),
        "sampled_local_bounds": [bmin_s.tolist(), bmax_s.tolist()],
        "sampled_world_bounds": [
            local_to_world(corners_from_bounds(bmin_s, bmax_s), structure.center, structure.rotation).min(axis=0).tolist(),
            local_to_world(corners_from_bounds(bmin_s, bmax_s), structure.center, structure.rotation).max(axis=0).tolist(),
        ],
        "local_grid_spacing": list(spacing),
        "iso_level": 0.0,
        "field_sampling_requested": str(field_sampling),
        "iso_extractor_requested": str(iso_extractor),
        "mesh_vertex_count": int(len(verts)),
        "mesh_face_count": int(len(faces)),
        "signed_volume": float(signed_volume),
        "faces_flipped_for_positive_volume": bool(faces_flipped),
    }
    meta.update(field_meta)
    meta.update(iso_meta)

    if target_num_faces is not None:
        verts, faces, dec_meta = decimate_mesh_open3d(verts, faces, target_num_faces)
        meta.update({f"decimation_{k}": v for k, v in dec_meta.items()})
        meta["mesh_vertex_count"] = int(len(verts))
        meta["mesh_face_count"] = int(len(faces))
        meta["signed_volume"] = float(signed_mesh_volume(verts, faces))

    return verts, faces, meta


# ============================================================
# 4. Projection basis from ray-vector and roll
# ============================================================
def projection_basis_from_ray(ray_vector: Sequence[float], roll_rad: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns orthonormal basis (u, v, n), where:
    - n: normalized ray direction (ray travel direction)
    - u, v: projection-plane basis vectors
    - roll_rad: in-plane rotation around n
    """
    n = normalize(ray_vector)

    # pick a reference vector not parallel to n
    if abs(n[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    u0 = normalize(np.cross(a, n))
    v0 = normalize(np.cross(n, u0))

    if abs(roll_rad) < 1e-15:
        return u0, v0, n

    Rroll = rotation_matrix_axis_angle(n, roll_rad)
    u = Rroll @ u0
    v = Rroll @ v0
    return normalize(u), normalize(v), n


# ============================================================
# 5. Reusable BVH-like projector object
# ============================================================
@dataclass
class StructureGeometry:
    """
    Mesh container used by all projection backends.

    For signed_raster and signed_raster_gpu:
        only vertices, faces, and face_normals are required.

    For pairwise and signed list_intersections reference methods:
        scene must be present, which requires build_scene=True.
    """
    structure_type: str
    params: Dict
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    mesh_meta: Dict
    scene: Optional[o3d.t.geometry.RaycastingScene] = None
    geometry_id: Optional[int] = None
    device_str: str = "CPU:0"
    nthreads: int = 0

    @property
    def has_scene(self) -> bool:
        return self.scene is not None

    def save_mesh(self, mesh_path: str, meta_path: Optional[str] = None) -> None:
        """Save mesh and metadata. Open3D scene/BVH itself is not serialized."""
        o3d = _get_open3d_module()
        mesh_path = str(mesh_path)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(self.faces.astype(np.int32))
        mesh.compute_vertex_normals()
        ok = o3d.io.write_triangle_mesh(mesh_path, mesh)
        if not ok:
            raise RuntimeError(f"Failed to save mesh to {mesh_path}")

        if meta_path is None:
            meta_path = str(Path(mesh_path).with_suffix(".json"))
        meta = {
            "structure_type": self.structure_type,
            "params": _jsonable(self.params),
            "mesh_meta": _jsonable(self.mesh_meta),
            "device_str": self.device_str,
            "nthreads": self.nthreads,
            "has_scene": bool(self.has_scene),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_saved_mesh(
        cls,
        mesh_path: str,
        meta_path: Optional[str] = None,
        nthreads: int = 0,
        device_str: str = "CPU:0",
        build_scene: bool = False,
    ) -> "StructureGeometry":
        o3d = _get_open3d_module()
        mesh_path = str(mesh_path)
        legacy = o3d.io.read_triangle_mesh(mesh_path)
        if legacy.is_empty():
            raise RuntimeError(f"Failed to load mesh from {mesh_path}")
        if meta_path is None:
            meta_path = str(Path(mesh_path).with_suffix(".json"))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        vertices = np.asarray(legacy.vertices, dtype=np.float64)
        faces = np.asarray(legacy.triangles, dtype=np.int32)
        faces, signed_volume, faces_flipped = orient_faces_outward_by_volume(vertices, faces)
        face_normals = compute_face_normals(vertices, faces)
        legacy.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        legacy.compute_vertex_normals()

        mesh_meta = dict(meta.get("mesh_meta", {}))
        mesh_meta.setdefault("signed_volume", float(signed_volume))
        mesh_meta.setdefault("faces_flipped_for_positive_volume_on_load", bool(faces_flipped))

        scene = None
        geom_id = None
        if build_scene:
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
            scene = o3d.t.geometry.RaycastingScene(nthreads=nthreads, device=o3d.core.Device(device_str))
            geom_id = int(scene.add_triangles(tmesh))

        return cls(
            structure_type=str(meta.get("structure_type", "loaded-mesh")),
            params=dict(meta.get("params", {})),
            vertices=vertices,
            faces=faces,
            face_normals=face_normals,
            mesh_meta=mesh_meta,
            scene=scene,
            geometry_id=geom_id,
            device_str=device_str,
            nthreads=nthreads,
        )


# Backward-compatible alias for old code/comments.
StructureBVH = StructureGeometry


def _create_open3d_scene(
    vertices: np.ndarray,
    faces: np.ndarray,
    nthreads: int = 0,
    device_str: str = "CPU:0",
) -> Tuple[o3d.t.geometry.RaycastingScene, int]:
    """
    Build Open3D RaycastingScene only when pairwise/list_intersections
    reference projection is requested.
    """
    o3d = _get_open3d_module()
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    legacy.compute_vertex_normals()

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene(nthreads=nthreads, device=o3d.core.Device(device_str))
    geometry_id = int(scene.add_triangles(tmesh))
    return scene, geometry_id


# ============================================================
# 6. Main API 1: build_bvh
# ============================================================
def build_structure_geometry(
    structure_type: str,
    *,
    field_shape: Tuple[int, int, int] = (150, 150, 150),
    nthreads: int = 0,
    device_str: str = "CPU:0",
    build_scene: bool = False,
    use_direct_polyhedron_mesh: bool = True,
    target_num_faces: Optional[int] = None,
    field_sampling: str = "auto",
    local_cutoff: float = 4.5,
    iso_extractor: str = "auto",
    field_dtype: Union[str, np.dtype] = "float32",
    pointwise_chunk_points: int = 2_000_000,
    **params,
) -> StructureGeometry:
    """
    Build structure mesh and optional Open3D scene.

    Parameters
    ----------
    build_scene:
        False by default. Use False for production signed_raster_gpu.
        Use True only when projection_method='pairwise' or 'signed' is needed.

    use_direct_polyhedron_mesh:
        If True, RegularPolyhedron objects are meshed by ConvexHull directly
        instead of by 3D marching cubes. This is much faster and produces far
        fewer triangles.

    target_num_faces:
        Optional Open3D quadric decimation target. Useful for dense
        marching-cubes meshes, but should be validated against SAXS intensity.

    field_sampling / local_cutoff:
        For AmorphousBlob, field_sampling='auto' uses local Gaussian-kernel
        accumulation with the specified cutoff. This preserves grid resolution
        while avoiding far-tail evaluations.

    iso_extractor:
        'vtk_flying_edges' uses one global VTK FlyingEdges3D contouring pass.
        Use 'skimage_marching_cubes' for the older extractor.
    """
    structure = make_structure(structure_type, **params)

    vertices, faces, mesh_meta = structure_to_mesh(
        structure,
        field_shape=field_shape,
        use_direct_polyhedron_mesh=use_direct_polyhedron_mesh,
        target_num_faces=target_num_faces,
        field_sampling=field_sampling,
        local_cutoff=local_cutoff,
        iso_extractor=iso_extractor,
        field_dtype=field_dtype,
        pointwise_chunk_points=pointwise_chunk_points,
    )

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    face_normals = compute_face_normals(vertices, faces)

    scene = None
    geometry_id = None
    if build_scene:
        scene, geometry_id = _create_open3d_scene(
            vertices,
            faces,
            nthreads=nthreads,
            device_str=device_str,
        )

    params_copy = dict(params)
    params_copy["field_shape"] = list(field_shape)
    params_copy["build_scene"] = bool(build_scene)
    params_copy["use_direct_polyhedron_mesh"] = bool(use_direct_polyhedron_mesh)
    params_copy["field_sampling"] = str(field_sampling)
    params_copy["local_cutoff"] = float(local_cutoff)
    params_copy["iso_extractor"] = str(iso_extractor)
    params_copy["field_dtype"] = str(np.dtype(field_dtype))
    params_copy["pointwise_chunk_points"] = int(pointwise_chunk_points)
    if target_num_faces is not None:
        params_copy["target_num_faces"] = int(target_num_faces)

    return StructureGeometry(
        structure_type=structure_type,
        params=params_copy,
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        mesh_meta=mesh_meta,
        scene=scene,
        geometry_id=geometry_id,
        device_str=device_str,
        nthreads=nthreads,
    )


def build_bvh(
    structure_type: str,
    *,
    field_shape: Tuple[int, int, int] = (150, 150, 150),
    nthreads: int = 0,
    device_str: str = "CPU:0",
    build_scene: bool = False,
    use_direct_polyhedron_mesh: bool = True,
    target_num_faces: Optional[int] = None,
    field_sampling: str = "auto",
    local_cutoff: float = 4.5,
    iso_extractor: str = "auto",
    field_dtype: Union[str, np.dtype] = "float32",
    pointwise_chunk_points: int = 2_000_000,
    **params,
) -> StructureGeometry:
    """
    Backward-compatible wrapper.

    By default, build_scene=False so GUI/EXE runs avoid Open3D BVH
    construction and use signed rasterization. For pairwise validation, call

        build_bvh(..., build_scene=True)
    """
    return build_structure_geometry(
        structure_type,
        field_shape=field_shape,
        nthreads=nthreads,
        device_str=device_str,
        build_scene=build_scene,
        use_direct_polyhedron_mesh=use_direct_polyhedron_mesh,
        target_num_faces=target_num_faces,
        field_sampling=field_sampling,
        local_cutoff=local_cutoff,
        iso_extractor=iso_extractor,
        field_dtype=field_dtype,
        pointwise_chunk_points=pointwise_chunk_points,
        **params,
    )


# ============================================================
# 7. Main API 2: project_density
# ============================================================
# ============================================================
# 7. Main API 2: project_density
# ============================================================
def _pairwise_thickness_from_list_intersections(ans: Dict, nrays: int) -> np.ndarray:
    """
    Ray thickness from list_intersections().
    For each ray, sort all hit distances and sum alternating entry/exit intervals.
    """
    ray_ids = ans["ray_ids"].numpy().astype(np.int64)
    t_hit = ans["t_hit"].numpy().astype(np.float64)

    thickness = np.zeros(nrays, dtype=np.float64)
    if len(ray_ids) == 0:
        return thickness

    order = np.lexsort((t_hit, ray_ids))
    ray_ids = ray_ids[order]
    t_hit = t_hit[order]

    start = 0
    n = len(ray_ids)
    while start < n:
        rid = ray_ids[start]
        end = start + 1
        while end < n and ray_ids[end] == rid:
            end += 1
        arr = t_hit[start:end]
        if len(arr) >= 2:
            if len(arr) % 2 == 1:
                arr = arr[:-1]
            thickness[rid] = np.sum(arr[1::2] - arr[0::2])
        start = end
    return thickness


def _signed_thickness_from_list_intersections(
    ans: Dict,
    nrays: int,
    face_normals: np.ndarray,
    ray_direction: Sequence[float],
    eps: float = 1e-8,
    fallback_pairwise: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    CPU signed accumulation of ray thickness from Open3D list_intersections().

    For a watertight, consistently outward-oriented triangular mesh and a ray
    direction k, each intersection contributes

        sign(k · n_face) * t_hit

    to the corresponding ray. This gives

        L = -t_entry + t_exit - t_entry2 + t_exit2 + ...

    without sorting intersections by ray/depth. The optional pairwise fallback
    replaces rays whose signed entry/exit balance is not closed.
    """
    ray_ids = ans["ray_ids"].numpy().astype(np.int64)
    t_hit = ans["t_hit"].numpy().astype(np.float64)
    primitive_ids = ans["primitive_ids"].numpy().astype(np.int64)

    thickness = np.zeros(nrays, dtype=np.float64)
    diagnostics = {
        "num_hits": int(len(ray_ids)),
        "num_valid_hits": 0,
        "num_bad_balance_rays": 0,
        "signed_median_before_clamp": 0.0,
        "used_global_sign_flip": False,
        "used_pairwise_fallback": False,
        "method": "signed",
    }

    if len(ray_ids) == 0:
        return thickness, diagnostics

    k = normalize(ray_direction)
    dots = np.asarray(face_normals, dtype=np.float64)[primitive_ids] @ k
    valid = np.abs(dots) > float(eps)
    diagnostics["num_valid_hits"] = int(np.count_nonzero(valid))

    if not np.any(valid):
        return thickness, diagnostics

    ray_ids_v = ray_ids[valid]
    t_hit_v = t_hit[valid]
    signs_v = np.sign(dots[valid])

    thickness = np.bincount(
        ray_ids_v,
        weights=signs_v * t_hit_v,
        minlength=nrays,
    ).astype(np.float64, copy=False)

    # For a closed mesh, signed crossings should balance to zero for each ray.
    balance = np.bincount(
        ray_ids_v,
        weights=signs_v,
        minlength=nrays,
    ).astype(np.float64, copy=False)
    bad = np.abs(balance) > 0.5
    diagnostics["num_bad_balance_rays"] = int(np.count_nonzero(bad))

    # If face orientation is globally reversed, most nonzero thickness values are negative.
    nonzero = np.abs(thickness) > float(eps)
    if np.any(nonzero):
        med = float(np.median(thickness[nonzero]))
        diagnostics["signed_median_before_clamp"] = med
        if med < 0:
            thickness *= -1.0
            diagnostics["used_global_sign_flip"] = True

    if fallback_pairwise and np.any(bad):
        pairwise = _pairwise_thickness_from_list_intersections(ans, nrays)
        thickness[bad] = pairwise[bad]
        diagnostics["used_pairwise_fallback"] = True

    # Numerical cleanup. Tiny negatives can appear near silhouettes/grazing faces.
    thickness[np.abs(thickness) < float(eps)] = 0.0
    thickness[thickness < 0.0] = 0.0
    return thickness, diagnostics


def _block_average_2d(a: np.ndarray, factor: int) -> np.ndarray:
    """Average a 2D array over non-overlapping factor × factor blocks."""
    factor = int(factor)
    if factor <= 1:
        return a
    Hs, Ws = a.shape
    if Hs % factor != 0 or Ws % factor != 0:
        raise ValueError("Array shape must be divisible by factor for block averaging.")
    H, W = Hs // factor, Ws // factor
    return a.reshape(H, factor, W, factor).mean(axis=(1, 3))


def _top_left_edge(xa: float, ya: float, xb: float, yb: float) -> bool:
    """
    Half-open edge ownership rule for rasterization.

    This avoids double-counting fragments exactly on shared triangle edges, which
    otherwise appears as thin diagonal seams in signed-depth accumulation.
    The convention is arbitrary but must be used consistently for all triangles.
    """
    dy = yb - ya
    dx = xb - xa
    return (dy < 0.0) or (dy == 0.0 and dx > 0.0)


def _inside_half_open_barycentric(
    l0: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    x0: float, y0: float,
    x1: float, y1: float,
    x2: float, y2: float,
    edge_eps: float = 1e-12,
) -> np.ndarray:
    """Barycentric inside test with a half-open top-left edge rule."""
    # l0 == 0 is the edge from vertex 1 to vertex 2, etc.
    e0 = _top_left_edge(x1, y1, x2, y2)
    e1 = _top_left_edge(x2, y2, x0, y0)
    e2 = _top_left_edge(x0, y0, x1, y1)

    def accept(lam: np.ndarray, top_left: bool) -> np.ndarray:
        if top_left:
            return lam >= -edge_eps
        return lam > edge_eps

    return accept(l0, e0) & accept(l1, e1) & accept(l2, e2)


def _signed_thickness_from_rasterization_cpu(
    bvh: StructureBVH,
    u: np.ndarray,
    v: np.ndarray,
    n: np.ndarray,
    uv_bounds: Sequence[float],
    image_shape: Tuple[int, int],
    eps: float = 1e-12,
    edge_eps: float = 1e-12,
    edge_rule: str = "top_left",
) -> Tuple[np.ndarray, Dict]:
    """
    Pure CPU signed surface rasterization.

    Each oriented triangle is projected onto the detector (u, v) plane. For every
    covered pixel center, the triangle plane gives the intersection coordinate w.
    The fragment contributes

        sign(n_ray · n_face) * w

    to the accumulation buffer. For a watertight, outward-oriented closed mesh,
    this is equivalent to summing exit depths minus entry depths along each ray,
    and therefore handles non-convex objects without explicit depth sorting.

    Refinements relative to the first CPU version
    ---------------------------------------------
    1. Cell-centered sampling is used: pixel i samples u_min + (i + 0.5) du.
    2. A half-open top-left edge rule is available to avoid double-counting
       fragments exactly on shared triangle edges.
    3. A signed crossing-balance buffer is returned in diagnostics. Ideally,
       this is near zero wherever the mesh is closed and intersections are paired.
    """
    H, W = map(int, image_shape)
    umin, umax, vmin, vmax = map(float, uv_bounds)
    du = float((umax - umin) / max(W, 1))
    dv = float((vmax - vmin) / max(H, 1))

    verts = np.asarray(bvh.vertices, dtype=np.float64)
    faces = np.asarray(bvh.faces, dtype=np.int64)
    face_normals = np.asarray(bvh.face_normals, dtype=np.float64)

    # Coordinates of all vertices in the projection frame.
    U = verts @ u
    V = verts @ v
    Wcoord = verts @ n

    # Pixel-space coordinates. Pixel centers are at integer indices.
    # index i corresponds to coordinate umin + (i + 0.5) * du.
    Xp = (U - (umin + 0.5 * du)) / du if du != 0 else np.zeros_like(U)
    Yp = (V - (vmin + 0.5 * dv)) / dv if dv != 0 else np.zeros_like(V)

    thickness = np.zeros((H, W), dtype=np.float64)
    balance = np.zeros((H, W), dtype=np.float64)

    skipped_parallel = 0
    skipped_degenerate = 0
    rasterized_triangles = 0
    fragments = 0

    use_top_left = str(edge_rule).lower().strip().replace("-", "_") in {"top_left", "half_open", "watertight"}

    for fi, face in enumerate(faces):
        sign_dot = float(face_normals[fi] @ n)
        if abs(sign_dot) <= eps:
            skipped_parallel += 1
            continue
        sgn = 1.0 if sign_dot > 0.0 else -1.0

        x0, x1, x2 = Xp[face]
        y0, y1, y2 = Yp[face]
        z0, z1, z2 = Wcoord[face]

        xmin = max(0, int(np.floor(min(x0, x1, x2) - 1.0)))
        xmax = min(W - 1, int(np.ceil(max(x0, x1, x2) + 1.0)))
        ymin = max(0, int(np.floor(min(y0, y1, y2) - 1.0)))
        ymax = min(H - 1, int(np.ceil(max(y0, y1, y2) + 1.0)))
        if xmax < xmin or ymax < ymin:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) <= eps:
            skipped_degenerate += 1
            continue

        xs = np.arange(xmin, xmax + 1, dtype=np.float64)
        ys = np.arange(ymin, ymax + 1, dtype=np.float64)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")

        l0 = ((y1 - y2) * (XX - x2) + (x2 - x1) * (YY - y2)) / denom
        l1 = ((y2 - y0) * (XX - x2) + (x0 - x2) * (YY - y2)) / denom
        l2 = 1.0 - l0 - l1

        if use_top_left:
            inside = _inside_half_open_barycentric(
                l0, l1, l2,
                float(x0), float(y0), float(x1), float(y1), float(x2), float(y2),
                edge_eps=float(edge_eps),
            )
        else:
            # Inclusive test retained as a diagnostic/reference option.
            inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        if not np.any(inside):
            continue

        z = l0 * z0 + l1 * z1 + l2 * z2
        patch = thickness[ymin:ymax + 1, xmin:xmax + 1]
        bal_patch = balance[ymin:ymax + 1, xmin:xmax + 1]
        patch[inside] += sgn * z[inside]
        bal_patch[inside] += sgn
        rasterized_triangles += 1
        fragments += int(np.count_nonzero(inside))

    nonzero = np.abs(thickness) > 1e-10
    used_global_sign_flip = False
    signed_median_before_clamp = 0.0
    if np.any(nonzero):
        signed_median_before_clamp = float(np.median(thickness[nonzero]))
        if signed_median_before_clamp < 0:
            thickness *= -1.0
            balance *= -1.0
            used_global_sign_flip = True

    # Small negative values can arise at silhouettes and shared edges.
    negative_before_clamp = int(np.count_nonzero(thickness < -1e-10))
    thickness[np.abs(thickness) < 1e-10] = 0.0
    thickness[thickness < 0.0] = 0.0

    bad_balance = np.abs(balance) > 0.5
    diagnostics = {
        "method": "signed_raster_cpu",
        "edge_rule": "top_left" if use_top_left else "inclusive",
        "num_faces": int(len(faces)),
        "rasterized_triangles": int(rasterized_triangles),
        "skipped_parallel_triangles": int(skipped_parallel),
        "skipped_degenerate_triangles": int(skipped_degenerate),
        "num_fragments": int(fragments),
        "num_nonzero_pixels": int(np.count_nonzero(thickness)),
        "signed_median_before_clamp": float(signed_median_before_clamp),
        "used_global_sign_flip": bool(used_global_sign_flip),
        "negative_pixels_before_clamp": int(negative_before_clamp),
        "bad_balance_pixels": int(np.count_nonzero(bad_balance)),
        "max_abs_balance": float(np.max(np.abs(balance))) if balance.size else 0.0,
    }
    return thickness, diagnostics



# ============================================================
# 7b. Optional CuPy/CUDA signed rasterization backend
# ============================================================
_SIGNED_RASTER_CUDA_KERNEL = None

_SIGNED_RASTER_CUDA_SRC = r"""
extern "C" {

__device__ __forceinline__ bool _top_left_edge_cuda(float xa, float ya, float xb, float yb) {
    float dy = yb - ya;
    float dx = xb - xa;
    return (dy < 0.0f) || (dy == 0.0f && dx > 0.0f);
}

__device__ __forceinline__ bool _accept_edge_cuda(float lam, bool top_left, float edge_eps) {
    return top_left ? (lam >= -edge_eps) : (lam > edge_eps);
}

__global__ void signed_raster_kernel(
    const float* __restrict__ U,
    const float* __restrict__ V,
    const float* __restrict__ Wcoord,
    const int* __restrict__ faces,
    const float* __restrict__ sign_dot,
    const int nfaces,
    const float umin,
    const float vmin,
    const float du,
    const float dv,
    const int H,
    const int Wimg,
    const float eps,
    const float edge_eps,
    const int use_top_left,
    float* __restrict__ thickness,
    float* __restrict__ balance,
    unsigned long long* __restrict__ counters
) {
    const int fi = blockIdx.x;
    if (fi >= nfaces) return;

    __shared__ int block_has_frag;
    if (threadIdx.x == 0) block_has_frag = 0;
    __syncthreads();

    const float sd = sign_dot[fi];
    if (fabsf(sd) <= eps) {
        if (threadIdx.x == 0) atomicAdd(&counters[0], 1ULL);  // skipped_parallel
        return;
    }
    const float sgn = (sd > 0.0f) ? 1.0f : -1.0f;

    const int i0 = faces[3 * fi + 0];
    const int i1 = faces[3 * fi + 1];
    const int i2 = faces[3 * fi + 2];

    const float x0 = (U[i0] - (umin + 0.5f * du)) / du;
    const float x1 = (U[i1] - (umin + 0.5f * du)) / du;
    const float x2 = (U[i2] - (umin + 0.5f * du)) / du;

    const float y0 = (V[i0] - (vmin + 0.5f * dv)) / dv;
    const float y1 = (V[i1] - (vmin + 0.5f * dv)) / dv;
    const float y2 = (V[i2] - (vmin + 0.5f * dv)) / dv;

    const float z0 = Wcoord[i0];
    const float z1 = Wcoord[i1];
    const float z2 = Wcoord[i2];

    const float xmin_f = fminf(x0, fminf(x1, x2));
    const float xmax_f = fmaxf(x0, fmaxf(x1, x2));
    const float ymin_f = fminf(y0, fminf(y1, y2));
    const float ymax_f = fmaxf(y0, fmaxf(y1, y2));

    int xmin = (int)floorf(xmin_f - 1.0f);
    int xmax = (int)ceilf (xmax_f + 1.0f);
    int ymin = (int)floorf(ymin_f - 1.0f);
    int ymax = (int)ceilf (ymax_f + 1.0f);

    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax > Wimg - 1) xmax = Wimg - 1;
    if (ymax > H - 1) ymax = H - 1;
    if (xmax < xmin || ymax < ymin) return;

    const float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (fabsf(denom) <= eps) {
        if (threadIdx.x == 0) atomicAdd(&counters[1], 1ULL);  // skipped_degenerate
        return;
    }

    const bool e0 = _top_left_edge_cuda(x1, y1, x2, y2);  // l0 == 0 edge
    const bool e1 = _top_left_edge_cuda(x2, y2, x0, y0);  // l1 == 0 edge
    const bool e2 = _top_left_edge_cuda(x0, y0, x1, y1);  // l2 == 0 edge

    const int span_x = xmax - xmin + 1;
    const int span_y = ymax - ymin + 1;
    const int total = span_x * span_y;

    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int yy = ymin + idx / span_x;
        const int xx = xmin + idx - (idx / span_x) * span_x;

        const float X = (float)xx;
        const float Y = (float)yy;

        const float l0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / denom;
        const float l1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / denom;
        const float l2 = 1.0f - l0 - l1;

        bool inside;
        if (use_top_left) {
            inside = _accept_edge_cuda(l0, e0, edge_eps) &&
                     _accept_edge_cuda(l1, e1, edge_eps) &&
                     _accept_edge_cuda(l2, e2, edge_eps);
        } else {
            inside = (l0 >= -1.0e-9f) && (l1 >= -1.0e-9f) && (l2 >= -1.0e-9f);
        }

        if (inside) {
            const float z = l0 * z0 + l1 * z1 + l2 * z2;
            const int out_idx = yy * Wimg + xx;
            atomicAdd(&thickness[out_idx], sgn * z);
            atomicAdd(&balance[out_idx], sgn);
            atomicAdd(&counters[2], 1ULL);  // fragments
            atomicExch(&block_has_frag, 1);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && block_has_frag) {
        atomicAdd(&counters[3], 1ULL);  // rasterized_triangles
    }
}

}
"""


def _get_cupy_module():
    """Import CuPy lazily so the notebook still runs on CPU-only environments."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"CUDA path could not be detected.*",
                category=UserWarning,
            )
            import cupy as cp
    except Exception as exc:
        raise ImportError(
            "CuPy is required for projection_method='signed_raster_gpu'. "
            "On Colab, install a CUDA-matched CuPy wheel if import cupy fails, "
            "for example: !pip -q install cupy-cuda12x. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc
    return cp


def _get_signed_raster_cuda_kernel():
    global _SIGNED_RASTER_CUDA_KERNEL
    cp = _get_cupy_module()
    if _SIGNED_RASTER_CUDA_KERNEL is None:
        _SIGNED_RASTER_CUDA_KERNEL = cp.RawKernel(
            _SIGNED_RASTER_CUDA_SRC,
            "signed_raster_kernel",
            options=("--std=c++11",),
        )
    return _SIGNED_RASTER_CUDA_KERNEL


def _signed_thickness_from_rasterization_gpu(
    bvh: StructureBVH,
    u: np.ndarray,
    v: np.ndarray,
    n: np.ndarray,
    uv_bounds: Sequence[float],
    image_shape: Tuple[int, int],
    eps: float = 1e-8,
    edge_eps: float = 1e-6,
    edge_rule: str = "top_left",
    block_size: int = 128,
) -> Tuple[np.ndarray, Dict]:
    """
    CuPy/CUDA signed surface rasterization.

    This is the GPU counterpart of _signed_thickness_from_rasterization_cpu().
    It launches one CUDA block per triangle. Each block rasterizes the projected
    triangle into the output image and atomically accumulates

        sign(k · n_face) * depth

    into a float32 thickness buffer. It is intended for Colab/NVIDIA CUDA GPUs.
    """
    cp = _get_cupy_module()
    kernel = _get_signed_raster_cuda_kernel()

    H, Wimg = map(int, image_shape)
    umin, umax, vmin, vmax = map(float, uv_bounds)
    du = float((umax - umin) / max(Wimg, 1))
    dv = float((vmax - vmin) / max(H, 1))

    verts = np.asarray(bvh.vertices, dtype=np.float32)
    faces = np.asarray(bvh.faces, dtype=np.int32)
    face_normals = np.asarray(bvh.face_normals, dtype=np.float32)

    u32 = np.asarray(u, dtype=np.float32)
    v32 = np.asarray(v, dtype=np.float32)
    n32 = np.asarray(n, dtype=np.float32)

    U = (verts @ u32).astype(np.float32, copy=False)
    V = (verts @ v32).astype(np.float32, copy=False)
    Wcoord = (verts @ n32).astype(np.float32, copy=False)
    sign_dot = (face_normals @ n32).astype(np.float32, copy=False)

    U_g = cp.asarray(U)
    V_g = cp.asarray(V)
    W_g = cp.asarray(Wcoord)
    faces_g = cp.asarray(faces.reshape(-1))
    sign_dot_g = cp.asarray(sign_dot)

    thickness_g = cp.zeros((H, Wimg), dtype=cp.float32)
    balance_g = cp.zeros((H, Wimg), dtype=cp.float32)
    counters_g = cp.zeros((4,), dtype=cp.uint64)

    use_top_left = int(str(edge_rule).lower().strip().replace("-", "_") in {"top_left", "half_open", "watertight"})
    nfaces = int(faces.shape[0])
    block_size = int(block_size)

    kernel(
        (nfaces,),
        (block_size,),
        (
            U_g, V_g, W_g,
            faces_g,
            sign_dot_g,
            np.int32(nfaces),
            np.float32(umin), np.float32(vmin),
            np.float32(du), np.float32(dv),
            np.int32(H), np.int32(Wimg),
            np.float32(eps), np.float32(edge_eps),
            np.int32(use_top_left),
            thickness_g,
            balance_g,
            counters_g,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    thickness = cp.asnumpy(thickness_g).astype(np.float64, copy=False)
    balance = cp.asnumpy(balance_g).astype(np.float64, copy=False)
    counters = cp.asnumpy(counters_g)

    nonzero = np.abs(thickness) > 1e-10
    used_global_sign_flip = False
    signed_median_before_clamp = 0.0
    if np.any(nonzero):
        signed_median_before_clamp = float(np.median(thickness[nonzero]))
        if signed_median_before_clamp < 0:
            thickness *= -1.0
            balance *= -1.0
            used_global_sign_flip = True

    negative_before_clamp = int(np.count_nonzero(thickness < -1e-7))
    thickness[np.abs(thickness) < 1e-10] = 0.0
    thickness[thickness < 0.0] = 0.0

    bad_balance = np.abs(balance) > 0.5
    diagnostics = {
        "method": "signed_raster_gpu",
        "edge_rule": "top_left" if use_top_left else "inclusive",
        "num_faces": int(nfaces),
        "rasterized_triangles": int(counters[3]),
        "skipped_parallel_triangles": int(counters[0]),
        "skipped_degenerate_triangles": int(counters[1]),
        "num_fragments": int(counters[2]),
        "num_nonzero_pixels": int(np.count_nonzero(thickness)),
        "signed_median_before_clamp": float(signed_median_before_clamp),
        "used_global_sign_flip": bool(used_global_sign_flip),
        "negative_pixels_before_clamp": int(negative_before_clamp),
        "bad_balance_pixels": int(np.count_nonzero(bad_balance)),
        "max_abs_balance": float(np.max(np.abs(balance))) if balance.size else 0.0,
        "cuda_block_size": int(block_size),
        "accumulation_dtype": "float32",
    }
    return thickness, diagnostics


def project_density(
    bvh: StructureBVH,
    ray_vector: Sequence[float],
    roll_rad: float = 0.0,
    image_shape: Tuple[int, int] = (500, 500),
    margin_ratio: float = 0.02,
    out_dtype: str = "float32",
    ray_pad_ratio: float = 0.05,
    projection_method: str = "auto",
    signed_eps: float = 1e-8,
    fallback_pairwise: bool = True,
    supersample: int = 1,
    edge_rule: str = "top_left",
    gpu_block_size: int = 128,
    uv_bounds_override: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    Project thickness/density from a reusable mesh/BVH object.

    Projection backends
    -------------------
    projection_method='auto'
        Use CuPy/CUDA signed rasterization when available, otherwise CPU signed rasterization.
    projection_method='signed_raster'
        CPU signed triangle rasterization. Accurate reference for raster mode.
    projection_method='signed_raster_gpu'
        CuPy/CUDA signed triangle rasterization. Best for Colab/NVIDIA GPUs.
    projection_method='signed'
        Open3D list_intersections + signed accumulation over primitive normals.
    projection_method='pairwise'
        Open3D list_intersections + sorted entry/exit pair summation reference.
    """
    H, W = map(int, image_shape)
    if H <= 0 or W <= 0:
        raise ValueError("image_shape must be positive.")

    u, v, n = projection_basis_from_ray(ray_vector, roll_rad=roll_rad)

    verts = bvh.vertices
    uu = verts @ u
    vv = verts @ v
    ww = verts @ n

    umin, umax = float(np.min(uu)), float(np.max(uu))
    vmin, vmax = float(np.min(vv)), float(np.max(vv))
    wmin, wmax = float(np.min(ww)), float(np.max(ww))

    du0 = max(umax - umin, 1e-12)
    dv0 = max(vmax - vmin, 1e-12)
    dw0 = max(wmax - wmin, 1e-12)

    if uv_bounds_override is not None:
        if len(uv_bounds_override) != 4:
            raise ValueError("uv_bounds_override should be (umin, umax, vmin, vmax).")
        umin, umax, vmin, vmax = map(float, uv_bounds_override)
        if not (umax > umin and vmax > vmin):
            raise ValueError("uv_bounds_override must have positive u and v spans.")
    else:
        umin -= margin_ratio * du0
        umax += margin_ratio * du0
        vmin -= margin_ratio * dv0
        vmax += margin_ratio * dv0

    pad_w = ray_pad_ratio * dw0 + 1e-6

    du = float((umax - umin) / max(W, 1))
    dv = float((vmax - vmin) / max(H, 1))
    u_axis = umin + (np.arange(W, dtype=np.float64) + 0.5) * du
    v_axis = vmin + (np.arange(H, dtype=np.float64) + 0.5) * dv
    UU, VV = np.meshgrid(u_axis, v_axis, indexing="xy")

    method_requested = str(projection_method).lower().strip().replace("-", "_")
    supersample = int(max(1, supersample))

    auto_raster_methods = {"auto", "signed_raster_auto", "raster_auto"}
    cpu_raster_methods = {"signed_raster", "raster", "signed_raster_cpu"}
    gpu_raster_methods = {"signed_raster_gpu", "raster_gpu", "gpu", "cuda"}

    if method_requested in auto_raster_methods or method_requested in cpu_raster_methods or method_requested in gpu_raster_methods:
        if method_requested in auto_raster_methods:
            try:
                cp = _get_cupy_module()
                use_gpu = cp.cuda.runtime.getDeviceCount() > 0
            except Exception as exc:
                warnings.warn(
                    f"GPU signed rasterization is unavailable ({exc!r}); falling back to CPU signed rasterization.",
                    RuntimeWarning,
                )
                use_gpu = False
        else:
            use_gpu = method_requested in gpu_raster_methods
        raster_func = _signed_thickness_from_rasterization_gpu if use_gpu else _signed_thickness_from_rasterization_cpu
        canonical_method = "signed_raster_gpu" if use_gpu else "signed_raster"

        if supersample == 1:
            kwargs = dict(
                bvh=bvh,
                u=u,
                v=v,
                n=n,
                uv_bounds=(umin, umax, vmin, vmax),
                image_shape=(H, W),
                eps=signed_eps,
                edge_rule=edge_rule,
            )
            if use_gpu:
                kwargs["block_size"] = gpu_block_size
            thickness, projection_diagnostics = raster_func(**kwargs)
        else:
            kwargs = dict(
                bvh=bvh,
                u=u,
                v=v,
                n=n,
                uv_bounds=(umin, umax, vmin, vmax),
                image_shape=(H * supersample, W * supersample),
                eps=signed_eps,
                edge_rule=edge_rule,
            )
            if use_gpu:
                kwargs["block_size"] = gpu_block_size
            thickness_hi, projection_diagnostics = raster_func(**kwargs)
            thickness = _block_average_2d(thickness_hi, supersample)
            projection_diagnostics = dict(projection_diagnostics)
            projection_diagnostics["supersample"] = int(supersample)
            projection_diagnostics["highres_shape"] = [int(H * supersample), int(W * supersample)]

        method = canonical_method
    else:
        method = method_requested
        if getattr(bvh, "scene", None) is None:
            raise ValueError(
                f"projection_method='{method}' requires an Open3D RaycastingScene. "
                "Rebuild the geometry with build_scene=True, or use "
                "projection_method='signed_raster'/'signed_raster_gpu'."
            )
        start_w = wmin - pad_w
        origins = (UU[..., None] * u[None, None, :] +
                   VV[..., None] * v[None, None, :] +
                   start_w * n[None, None, :])
        directions = np.broadcast_to(n[None, None, :], origins.shape)

        o3d = _get_open3d_module()
        rays = np.concatenate([origins, directions], axis=2).astype(np.float32)
        rays_o3 = o3d.core.Tensor(rays.reshape(-1, 6), dtype=o3d.core.Dtype.Float32)

        ans = bvh.scene.list_intersections(rays_o3)
        nrays = H * W
        if method == "pairwise":
            thickness_1d = _pairwise_thickness_from_list_intersections(ans, nrays=nrays)
            projection_diagnostics = {"method": "pairwise", "num_hits": int(len(ans["ray_ids"].numpy()))}
        elif method == "signed":
            thickness_1d, projection_diagnostics = _signed_thickness_from_list_intersections(
                ans,
                nrays=nrays,
                face_normals=bvh.face_normals,
                ray_direction=n,
                eps=signed_eps,
                fallback_pairwise=fallback_pairwise,
            )
        else:
            raise ValueError(
                "projection_method must be one of 'signed_raster', 'signed_raster_gpu', 'signed', or 'pairwise'."
            )
        thickness = thickness_1d.reshape(H, W)

    dtype_map = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
    if out_dtype not in dtype_map:
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")
    projected_density = thickness.astype(dtype_map[out_dtype], copy=False)

    meta = {
        "projection_u": u.tolist(),
        "projection_v": v.tolist(),
        "ray_vector": n.tolist(),
        "roll_rad": float(roll_rad),
        "uv_bounds": [float(umin), float(umax), float(vmin), float(vmax)],
        # Explicit aliases for plotting and downstream code.
        "u_min_unit": float(umin),
        "u_max_unit": float(umax),
        "v_min_unit": float(vmin),
        "v_max_unit": float(vmax),
        "w_bounds": [float(wmin), float(wmax)],
        "du": du,
        "dv": dv,
        "x0_unit": float(u_axis[0]),
        "y0_unit": float(v_axis[0]),
        "sampling": "cell_center",
        "supersample": int(supersample) if method in {"signed_raster", "signed_raster_gpu"} else 1,
        "edge_rule": str(edge_rule),
        "image_shape": [H, W],
        "structure_type": bvh.structure_type,
        "projection_method": method,
        "projection_diagnostics": projection_diagnostics,
        "projected_mass": float(np.sum(thickness) * du * dv),
    }
    return projected_density, du, dv, meta


def project_density_scaled(
    bvh: StructureBVH,
    ray_vector: Sequence[float],
    roll_rad: float = 0.0,
    image_shape: Tuple[int, int] = (500, 500),
    object_scale: float = 1.0,
    base_margin_ratio: float = 0.02,
    out_dtype: str = "float32",
    ray_pad_ratio: float = 0.05,
    projection_method: str = "auto",
    signed_eps: float = 1e-8,
    fallback_pairwise: bool = True,
    supersample: int = 1,
    edge_rule: str = "top_left",
    gpu_block_size: int = 128,
) -> Tuple[np.ndarray, float, float, Dict]:
    """Project density while controlling how much of the image the object occupies.

    object_scale=1.0 reproduces project_density with base_margin_ratio. Values
    below 1.0 expand the uv window around the projected object, so the object
    appears smaller in the fixed detector image without changing the mesh.
    """
    object_scale = float(object_scale)
    if not np.isfinite(object_scale) or object_scale <= 0:
        object_scale = 1.0

    u, v, n = projection_basis_from_ray(ray_vector, roll_rad=roll_rad)
    verts = np.asarray(bvh.vertices, dtype=np.float64)
    uu = verts @ u
    vv = verts @ v

    umin0, umax0 = float(np.min(uu)), float(np.max(uu))
    vmin0, vmax0 = float(np.min(vv)), float(np.max(vv))
    du0 = max(umax0 - umin0, 1e-12)
    dv0 = max(vmax0 - vmin0, 1e-12)

    target_du = du0 * (1.0 + 2.0 * float(base_margin_ratio)) / object_scale
    target_dv = dv0 * (1.0 + 2.0 * float(base_margin_ratio)) / object_scale
    ucen = 0.5 * (umin0 + umax0)
    vcen = 0.5 * (vmin0 + vmax0)
    uv_bounds = (
        ucen - 0.5 * target_du,
        ucen + 0.5 * target_du,
        vcen - 0.5 * target_dv,
        vcen + 0.5 * target_dv,
    )

    projected_density, du, dv, meta = project_density(
        bvh,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
        image_shape=image_shape,
        margin_ratio=base_margin_ratio,
        out_dtype=out_dtype,
        ray_pad_ratio=ray_pad_ratio,
        projection_method=projection_method,
        signed_eps=signed_eps,
        fallback_pairwise=fallback_pairwise,
        supersample=supersample,
        edge_rule=edge_rule,
        gpu_block_size=gpu_block_size,
        uv_bounds_override=uv_bounds,
    )
    meta["object_scale"] = float(object_scale)
    meta["base_margin_ratio"] = float(base_margin_ratio)
    meta["scaled_projection"] = True
    return projected_density, du, dv, meta


def compare_projection_methods(
    bvh: StructureBVH,
    ray_vector: Sequence[float] = (0.0, 0.0, 1.0),
    roll_rad: float = 0.0,
    image_shape: Tuple[int, int] = (500, 500),
    margin_ratio: float = 0.02,
    ray_pad_ratio: float = 0.05,
    signed_eps: float = 1e-8,
    supersample: int = 1,
    edge_rule: str = "top_left",
) -> Dict:
    """
    Compare the new signed accumulation projection against the pairwise reference.
    Useful for validating watertightness and normal orientation.
    """
    proj_signed, du_s, dv_s, meta_s = project_density(
        bvh,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
        image_shape=image_shape,
        margin_ratio=margin_ratio,
        ray_pad_ratio=ray_pad_ratio,
        out_dtype="float64",
        projection_method="signed_raster",
        signed_eps=signed_eps,
        fallback_pairwise=False,
        supersample=supersample,
        edge_rule=edge_rule,
    )
    proj_pairwise, du_p, dv_p, meta_p = project_density(
        bvh,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
        image_shape=image_shape,
        margin_ratio=margin_ratio,
        ray_pad_ratio=ray_pad_ratio,
        out_dtype="float64",
        projection_method="pairwise",
    )
    diff = proj_signed - proj_pairwise
    denom = np.linalg.norm(proj_pairwise.ravel()) + 1e-30
    rel_l2 = float(np.linalg.norm(diff.ravel()) / denom)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mass_signed = float(proj_signed.sum() * du_s * dv_s)
    mass_pairwise = float(proj_pairwise.sum() * du_p * dv_p)
    return {
        "relative_l2_error": rel_l2,
        "max_abs_error": max_abs,
        "mass_signed": mass_signed,
        "mass_pairwise": mass_pairwise,
        "mass_relative_error": float((mass_signed - mass_pairwise) / (abs(mass_pairwise) + 1e-30)),
        "signed_raster_meta": meta_s,
        "pairwise_meta": meta_p,
    }



def compare_gpu_projection_methods(
    bvh: StructureBVH,
    ray_vector: Sequence[float] = (0.0, 0.0, 1.0),
    roll_rad: float = 0.0,
    image_shape: Tuple[int, int] = (500, 500),
    margin_ratio: float = 0.02,
    ray_pad_ratio: float = 0.05,
    signed_eps: float = 1e-8,
    supersample: int = 1,
    edge_rule: str = "top_left",
    reference_method: str = "signed_raster",
    gpu_block_size: int = 128,
) -> Dict:
    """
    Compare CuPy/CUDA signed rasterization against a CPU reference.

    reference_method='signed_raster' compares GPU rasterization against the CPU
    rasterizer. reference_method='pairwise' compares against the Open3D pairwise
    ray-intersection reference.
    """
    proj_gpu, du_g, dv_g, meta_g = project_density(
        bvh,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
        image_shape=image_shape,
        margin_ratio=margin_ratio,
        ray_pad_ratio=ray_pad_ratio,
        out_dtype="float64",
        projection_method="signed_raster_gpu",
        signed_eps=signed_eps,
        fallback_pairwise=False,
        supersample=supersample,
        edge_rule=edge_rule,
        gpu_block_size=gpu_block_size,
    )
    proj_ref, du_r, dv_r, meta_r = project_density(
        bvh,
        ray_vector=ray_vector,
        roll_rad=roll_rad,
        image_shape=image_shape,
        margin_ratio=margin_ratio,
        ray_pad_ratio=ray_pad_ratio,
        out_dtype="float64",
        projection_method=reference_method,
        signed_eps=signed_eps,
        fallback_pairwise=False,
        supersample=supersample if reference_method in {"signed_raster", "signed_raster_gpu"} else 1,
        edge_rule=edge_rule,
    )

    diff = proj_gpu - proj_ref
    denom = np.linalg.norm(proj_ref.ravel()) + 1e-30
    rel_l2 = float(np.linalg.norm(diff.ravel()) / denom)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mass_gpu = float(proj_gpu.sum() * du_g * dv_g)
    mass_ref = float(proj_ref.sum() * du_r * dv_r)
    return {
        "relative_l2_error": rel_l2,
        "max_abs_error": max_abs,
        "mass_gpu": mass_gpu,
        "mass_reference": mass_ref,
        "mass_relative_error": float((mass_gpu - mass_ref) / (abs(mass_ref) + 1e-30)),
        "gpu_meta": meta_g,
        "reference_meta": meta_r,
    }



# ============================================================
# 8. Small helpers
# ============================================================
def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_projected_density(projected_density: np.ndarray, meta: Dict, npy_path: str, json_path: Optional[str] = None) -> None:
    np.save(npy_path, projected_density)
    if json_path is None:
        json_path = str(Path(npy_path).with_suffix(".json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(meta), f, ensure_ascii=False, indent=2)




# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def _as_float64_1d(x: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a 1D array with length >= 2")
    return arr


def _uniform_spacing(axis: Sequence[float], name: str) -> Tuple[float, float]:
    arr = _as_float64_1d(axis, name)
    d = np.diff(arr)
    d0 = float(d[0])
    if not np.allclose(d, d0, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced")
    return float(arr[0]), d0


# -----------------------------------------------------------------------------
# PONI parsing
# -----------------------------------------------------------------------------
@dataclass
class PoniGeometry:
    pixel1_m: float
    pixel2_m: float
    max_shape: Tuple[int, int]
    distance_m: float
    poni1_m: float
    poni2_m: float
    rot1_rad: float
    rot2_rad: float
    rot3_rad: float
    wavelength_m: float



def _parse_poni_value(s: str):
    s = s.strip()
    if s.startswith("{") or s.startswith("["):
        return ast.literal_eval(s)
    try:
        if any(ch in s for ch in ".eE"):
            return float(s)
        return int(s)
    except Exception:
        return s



def parse_poni(poni_path: Union[str, Path]) -> PoniGeometry:
    data = {}
    with open(poni_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = _parse_poni_value(v)

    detcfg = data.get("Detector_config", {})
    pixel1 = float(detcfg.get("pixel1", 7.5e-5))
    pixel2 = float(detcfg.get("pixel2", 7.5e-5))
    max_shape = tuple(int(x) for x in detcfg.get("max_shape", [500, 500]))

    return PoniGeometry(
        pixel1_m=pixel1,
        pixel2_m=pixel2,
        max_shape=max_shape,
        distance_m=float(data["Distance"]),
        poni1_m=float(data["Poni1"]),
        poni2_m=float(data["Poni2"]),
        rot1_rad=float(data.get("Rot1", 0.0)),
        rot2_rad=float(data.get("Rot2", 0.0)),
        rot3_rad=float(data.get("Rot3", 0.0)),
        wavelength_m=float(data["Wavelength"]),
    )


# -----------------------------------------------------------------------------
# Beam-center estimation from pyFAI geometry
# -----------------------------------------------------------------------------
def _R1(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c],
    ], dtype=np.float64)



def _R2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)



def _R3(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)



def pyfai_rotation(rot1: float, rot2: float, rot3: float) -> np.ndarray:
    """
    pyFAI convention:
        R = R3(rot3) @ R2(-rot2) @ R1(-rot1)
    """
    return _R3(rot3) @ _R2(-rot2) @ _R1(-rot1)



def estimate_beam_center_from_poni(
    poni_path: Union[str, Path],
) -> Tuple[float, float, dict]:
    """
    Estimate beam center pixel coordinates from full pyFAI geometry.

    Returns
    -------
    center_v_px : float
        Vertical/row beam-center coordinate in detector-pixel units.
    center_u_px : float
        Horizontal/column beam-center coordinate in detector-pixel units.
    meta : dict
        Additional diagnostic information.

    Notes
    -----
    We solve for detector pixel coordinates (dH, dV) such that the corresponding
    laboratory-frame point lies on the direct-beam axis, i.e. t1 = 0 and t2 = 0.

    pyFAI detector-coordinate mapping can be written as:
        t = R @ (D @ d + o)
    where
        d = [dH, dV]^T
        D = [[0, pixel1], [pixel2, 0], [0, 0]]
        o = [-poni1, -poni2, distance]^T
    and R is the pyFAI rotation matrix.
    """
    geom = parse_poni(poni_path)

    D = np.array([
        [0.0,           geom.pixel1_m],
        [geom.pixel2_m, 0.0          ],
        [0.0,           0.0          ],
    ], dtype=np.float64)

    o = np.array([
        -geom.poni1_m,
        -geom.poni2_m,
         geom.distance_m,
    ], dtype=np.float64)

    R = pyfai_rotation(geom.rot1_rad, geom.rot2_rad, geom.rot3_rad)
    A = R @ D
    c = R @ o

    M = A[:2, :]
    rhs = -c[:2]
    dH, dV = np.linalg.solve(M, rhs)

    center_u_px = float(dH)
    center_v_px = float(dV)

    meta = {
        "beam_center_u_px": center_u_px,
        "beam_center_v_px": center_v_px,
        "poni_center_u_px": float(geom.poni2_m / geom.pixel2_m),
        "poni_center_v_px": float(geom.poni1_m / geom.pixel1_m),
        "rot1_rad": geom.rot1_rad,
        "rot2_rad": geom.rot2_rad,
        "rot3_rad": geom.rot3_rad,
    }
    return center_v_px, center_u_px, meta


# -----------------------------------------------------------------------------
# q-axis generation using BEAM CENTER
# -----------------------------------------------------------------------------
def q_axes_from_poni_small_angle(
    poni_path: Union[str, Path],
    image_shape: Optional[Tuple[int, int]] = None,
    q_unit: str = "A^-1",
    center_mode: str = "beam",
    row_origin: str = "upper",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build separable Cartesian q_u, q_v axes from a PONI file.

    Parameters
    ----------
    poni_path : str or Path
        Path to .poni file.
    image_shape : (H, W), optional
        Detector image shape. Defaults to Detector_config['max_shape'].
    q_unit : {'A^-1', 'm^-1'}
        Reciprocal-space unit of returned axes.
    center_mode : {'beam', 'poni', 'image'}
        - 'beam'  : use beam center estimated from full pyFAI geometry (recommended)
        - 'poni'  : use raw PONI pixel location
        - 'image' : force center to image midpoint
    row_origin : {'upper', 'lower'}
        - 'upper' : NumPy/matplotlib image convention (row 0 at top)
        - 'lower' : geometric lower-origin convention

    Returns
    -------
    q_u : (W,) ndarray
    q_v : (H,) ndarray
    meta : dict
    """
    geom = parse_poni(poni_path)
    if image_shape is None:
        H, W = geom.max_shape
    else:
        H, W = int(image_shape[0]), int(image_shape[1])

    if center_mode not in {"beam", "poni", "image"}:
        raise ValueError("center_mode must be 'beam', 'poni', or 'image'")
    if row_origin not in {"upper", "lower"}:
        raise ValueError("row_origin must be 'upper' or 'lower'")

    beam_meta = {}
    if center_mode == "beam":
        center_v_px, center_u_px, beam_meta = estimate_beam_center_from_poni(poni_path)
    elif center_mode == "poni":
        center_v_px = geom.poni1_m / geom.pixel1_m
        center_u_px = geom.poni2_m / geom.pixel2_m
        beam_meta = {
            "beam_center_u_px": float(center_u_px),
            "beam_center_v_px": float(center_v_px),
            "poni_center_u_px": float(center_u_px),
            "poni_center_v_px": float(center_v_px),
        }
    else:
        center_v_px = 0.5 * (H - 1)
        center_u_px = 0.5 * (W - 1)
        beam_meta = {
            "beam_center_u_px": float(center_u_px),
            "beam_center_v_px": float(center_v_px),
            "poni_center_u_px": float(geom.poni2_m / geom.pixel2_m),
            "poni_center_v_px": float(geom.poni1_m / geom.pixel1_m),
        }

    u_px = np.arange(W, dtype=np.float64) + 0.5
    v_px = np.arange(H, dtype=np.float64) + 0.5

    u_det_m = (u_px - center_u_px) * geom.pixel2_m
    if row_origin == "upper":
        v_det_m = (center_v_px - v_px) * geom.pixel1_m
    else:
        v_det_m = (v_px - center_v_px) * geom.pixel1_m

    k = 2.0 * np.pi / geom.wavelength_m
    q_u_minv = k * (u_det_m / geom.distance_m)
    q_v_minv = k * (v_det_m / geom.distance_m)

    if q_unit == "m^-1":
        q_u = q_u_minv
        q_v = q_v_minv
        q_conversion = 1.0
    elif q_unit == "A^-1":
        q_conversion = 1e-10
        q_u = q_u_minv * q_conversion
        q_v = q_v_minv * q_conversion
    else:
        raise ValueError("q_unit must be 'A^-1' or 'm^-1'")

    meta = {
        "pixel1_m": geom.pixel1_m,
        "pixel2_m": geom.pixel2_m,
        "distance_m": geom.distance_m,
        "poni1_m": geom.poni1_m,
        "poni2_m": geom.poni2_m,
        "wavelength_m": geom.wavelength_m,
        "center_v_px": float(center_v_px),
        "center_u_px": float(center_u_px),
        "q_unit": q_unit,
        "center_mode": center_mode,
        "row_origin": row_origin,
        "small_angle": True,
        "q_conversion_from_m^-1": q_conversion,
        **beam_meta,
    }
    return q_u, q_v, meta


# -----------------------------------------------------------------------------
# Real-space axes for projected density
# -----------------------------------------------------------------------------
def centered_uv_axes(shape: Tuple[int, int], du: float, dv: float) -> Tuple[np.ndarray, np.ndarray]:
    H, W = int(shape[0]), int(shape[1])
    u = (np.arange(W, dtype=np.float64) - 0.5 * (W - 1)) * float(du)
    v = (np.arange(H, dtype=np.float64) - 0.5 * (H - 1)) * float(dv)
    return u, v



def uv_axes_from_origin(shape: Tuple[int, int], du: float, dv: float, u0: float, v0: float) -> Tuple[np.ndarray, np.ndarray]:
    H, W = int(shape[0]), int(shape[1])
    u = float(u0) + np.arange(W, dtype=np.float64) * float(du)
    v = float(v0) + np.arange(H, dtype=np.float64) * float(dv)
    return u, v


# -----------------------------------------------------------------------------
# 2D CZT with explicit axes
# -----------------------------------------------------------------------------
def czt2_from_real_axes(
    projected_density: np.ndarray,
    u_axis: Sequence[float],
    v_axis: Sequence[float],
    q_u: Sequence[float],
    q_v: Sequence[float],
) -> np.ndarray:
    """
    2D CZT Fourier transform with explicit real-space axes.

    projected_density: shape (H, W), axis0=v, axis1=u
    Fourier convention:
        A(q_u,q_v) = integral integral rho(u,v) exp[-i (q_u u + q_v v)] du dv
    """
    rho = np.asarray(projected_density, dtype=np.complex128)
    if rho.ndim != 2:
        raise ValueError("projected_density must be 2D")
    H, W = rho.shape

    u_axis = _as_float64_1d(u_axis, "u_axis")
    v_axis = _as_float64_1d(v_axis, "v_axis")
    q_u = _as_float64_1d(q_u, "q_u")
    q_v = _as_float64_1d(q_v, "q_v")

    if u_axis.size != W or v_axis.size != H:
        raise ValueError("u_axis/v_axis lengths must match projected_density shape")

    u0, du = _uniform_spacing(u_axis, "u_axis")
    v0, dv = _uniform_spacing(v_axis, "v_axis")
    q_u0, dq_u = _uniform_spacing(q_u, "q_u")
    q_v0, dq_v = _uniform_spacing(q_v, "q_v")

    w_u = np.exp(-1j * dq_u * du)
    a_u = np.exp(+1j * q_u0 * du)
    A_u = czt(rho, m=q_u.size, w=w_u, a=a_u, axis=1)
    A_u *= du * np.exp(-1j * q_u[None, :] * u0)

    w_v = np.exp(-1j * dq_v * dv)
    a_v = np.exp(+1j * q_v0 * dv)
    A_uv = czt(A_u, m=q_v.size, w=w_v, a=a_v, axis=0)
    A_uv *= dv * np.exp(-1j * q_v[:, None] * v0)

    return A_uv



def czt2_from_projected_density(
    projected_density: np.ndarray,
    du: float,
    dv: float,
    q_u: Sequence[float],
    q_v: Sequence[float],
    u0: Optional[float] = None,
    v0: Optional[float] = None,
) -> np.ndarray:
    H, W = np.asarray(projected_density).shape
    if u0 is None or v0 is None:
        u_axis, v_axis = centered_uv_axes((H, W), du=du, dv=dv)
    else:
        u_axis, v_axis = uv_axes_from_origin((H, W), du=du, dv=dv, u0=u0, v0=v0)
    return czt2_from_real_axes(projected_density, u_axis=u_axis, v_axis=v_axis, q_u=q_u, q_v=q_v)


# -----------------------------------------------------------------------------
# Scaling + translation
# -----------------------------------------------------------------------------
def translation_phase(q_u: Sequence[float], q_v: Sequence[float], t_u: float = 0.0, t_v: float = 0.0) -> np.ndarray:
    q_u = _as_float64_1d(q_u, "q_u")
    q_v = _as_float64_1d(q_v, "q_v")
    return np.exp(-1j * (q_v[:, None] * float(t_v) + q_u[None, :] * float(t_u)))



def complex_amplitude_from_projected_density(
    projected_density: np.ndarray,
    du: float,
    dv: float,
    scale_factor: float,
    translation_u: float,
    translation_v: float,
    poni_path: Union[str, Path],
    detector_image_shape: Optional[Tuple[int, int]] = None,
    offset: Optional[Tuple[int, int]] = (0, 0),
    q_unit: str = "A^-1",
    center_mode: str = "beam",
    row_origin: str = "upper",
    u0: Optional[float] = None,
    v0: Optional[float] = None,
    weight: float = 1.0,
    contrast: float = 1.0,
    return_q_axes: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    """
    Compute a size-scaled and translated complex amplitude on the detector q-grid
    defined from a PONI file, using beam-center estimation by default.
    """
    s = float(scale_factor)
    q_u, q_v, q_meta = q_axes_from_poni_small_angle(
        poni_path=poni_path,
        image_shape=detector_image_shape,
        q_unit=q_unit,
        center_mode=center_mode,
        row_origin=row_origin,
    )

    dq_u, dq_v = q_u[1] - q_u[0], q_v[1] - q_v[0]

    A_unit_scaled_q = czt2_from_projected_density(
        projected_density=projected_density,
        du=du,
        dv=dv,
        q_u=(q_u - dq_u * offset[0]) * s,
        q_v=(q_v - dq_v * offset[1]) * s,
        u0=u0,
        v0=v0,
    )

    A = (float(weight) * float(contrast) * (s ** 3)) * A_unit_scaled_q
    A *= translation_phase(q_u=q_u, q_v=q_v, t_u=translation_u, t_v=translation_v)

    meta = {
        "scale_factor": s,
        "translation_u": float(translation_u),
        "translation_v": float(translation_v),
        "du": float(du),
        "dv": float(dv),
        "u0": None if u0 is None else float(u0),
        "v0": None if v0 is None else float(v0),
        **q_meta,
    }

    if return_q_axes:
        return A, q_u, q_v, meta
    return A


__all__ = [
    "StructureBVH",
    "build_bvh",
    "project_density",
    "project_density_scaled",
    "compare_projection_methods",
    "compute_face_normals",
    "signed_mesh_volume",
    "orient_faces_outward_by_volume",
    "PoniGeometry",
    "parse_poni",
    "pyfai_rotation",
    "estimate_beam_center_from_poni",
    "q_axes_from_poni_small_angle",
    "centered_uv_axes",
    "uv_axes_from_origin",
    "czt2_from_real_axes",
    "czt2_from_projected_density",
    "translation_phase",
    "complex_amplitude_from_projected_density",
]

def proj_ellipsoid(
    b: float,
    angle: float,
    img_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:

    if b <= 0 or b > 1:
        raise Exception(f"the value of b should be 0 < b <= 1, but it was {b}")

    img_shape = (500, 500) if img_shape is None else img_shape

    x = np.linspace(-1, 1, img_shape[0])
    y = np.linspace(-1, 1, img_shape[1])

    c, s = np.cos(angle), np.sin(angle)

    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)

    x, y = (c * x + s * y) / b, (c * y - s * x)

    r = np.sqrt(x ** 2 + y ** 2)

    return np.sqrt(np.where(r < 1, 1 - r ** 2, 0)), 2 / 500, 2 / 500
