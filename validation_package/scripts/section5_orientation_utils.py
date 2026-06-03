"""Orientation sampling helpers for Section 5 public validation.

The ID 57 core-shell model has the proper octahedral symmetry of a cube.
For the normal-incidence CXIDB subset we use one symmetry representative:
the +z cube face points near the beam direction and the in-plane roll is
restricted to [0, 90 deg).
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np


def halton(index: int, base: int) -> float:
    """Return a deterministic low-discrepancy value in [0, 1)."""
    result = 0.0
    factor = 1.0 / base
    i = int(index)
    while i > 0:
        result += factor * (i % base)
        i //= base
        factor /= base
    return result


def axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotation_from_z_to_vector(vector: np.ndarray) -> np.ndarray:
    target = np.asarray(vector, dtype=np.float64)
    target /= np.linalg.norm(target)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(z_axis, target)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    if cross_norm < 1.0e-12:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        return axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.pi)
    return axis_angle_matrix(cross / cross_norm, math.atan2(cross_norm, dot))


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q *= -1.0
    return q


def rotation_angle_deg(matrix: np.ndarray) -> float:
    value = (float(np.trace(matrix)) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(value, -1.0, 1.0))))


def octahedral_group_matrices() -> np.ndarray:
    """Return the 24 proper signed-permutation rotations of cube symmetry."""
    group = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, col in enumerate(permutation):
                matrix[row, col] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                group.append(matrix)
    return np.stack(group, axis=0)


def symmetry_distance_deg(rotation_a: np.ndarray, rotation_b: np.ndarray, group: np.ndarray | None = None) -> float:
    """Minimum rotation angle under right-multiplied object symmetry."""
    if group is None:
        group = octahedral_group_matrices()
    best = 180.0
    inv_a = np.asarray(rotation_a, dtype=np.float64).T
    b = np.asarray(rotation_b, dtype=np.float64)
    for symmetry in group:
        angle = rotation_angle_deg(inv_a @ b @ symmetry)
        if angle < best:
            best = angle
    return best


def normal_incidence_orientation_table(
    count: int,
    tilt_max_deg: float,
    *,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Uniformly sample the normal-incidence fundamental region.

    The beam-facing cube normal is sampled uniformly over a spherical cap.
    The in-plane roll is sampled uniformly over [0, 90 deg), which removes
    the C4 duplicate orientations of the cube/octahedron model.
    """
    rows: list[dict[str, Any]] = []
    cos_max = math.cos(math.radians(tilt_max_deg))
    for local_index in range(count):
        halton_index = start_index + local_index
        u_tilt = halton(halton_index, 2)
        u_azimuth = halton(halton_index, 3)
        u_roll = halton(halton_index, 5)

        cos_theta = 1.0 - u_tilt * (1.0 - cos_max)
        theta = math.acos(float(np.clip(cos_theta, -1.0, 1.0)))
        azimuth = 2.0 * math.pi * u_azimuth
        roll = 0.5 * math.pi * u_roll

        normal_lab = np.array(
            [
                math.sin(theta) * math.cos(azimuth),
                math.sin(theta) * math.sin(azimuth),
                math.cos(theta),
            ],
            dtype=np.float64,
        )
        base = rotation_from_z_to_vector(normal_lab)
        rotation = axis_angle_matrix(normal_lab, roll) @ base
        quaternion = matrix_to_quaternion_wxyz(rotation)

        rows.append(
            {
                "index": local_index,
                "halton_index": halton_index,
                "tilt_deg": math.degrees(theta),
                "azimuth_deg": math.degrees(azimuth),
                "roll_deg": math.degrees(roll),
                "rotation_matrix": rotation.astype(np.float32),
                "quaternion_wxyz": quaternion.astype(np.float32),
            }
        )
    return rows


def deduplicate_by_octahedral_symmetry(
    rows: list[dict[str, Any]],
    *,
    min_separation_deg: float,
) -> list[dict[str, Any]]:
    if min_separation_deg <= 0.0:
        return rows
    group = octahedral_group_matrices()
    kept: list[dict[str, Any]] = []
    kept_rotations: list[np.ndarray] = []
    for row in rows:
        rotation = np.asarray(row["rotation_matrix"], dtype=np.float64)
        if all(symmetry_distance_deg(prev, rotation, group) >= min_separation_deg for prev in kept_rotations):
            row = dict(row)
            row["index"] = len(kept)
            kept.append(row)
            kept_rotations.append(rotation)
    return kept
