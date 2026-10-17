#!/usr/bin/env python3
"""Standalone CPU reference for differentiable projected-gradient mesh SAXS."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Config:
    n_lat: int = 10
    n_lon: int = 20
    image_size: int = 96
    quad_order: int = 2
    uv_half_extent: float = 1.55
    axis_w: float = 0.72
    true_axis_u: float = 1.15
    true_axis_v: float = 0.82
    initial_axis_u: float = 1.14
    initial_axis_v: float = 0.83
    true_scale: float = 1.0
    initial_scale: float = 0.98
    q_min: float = 0.35
    q_max: float = 8.0
    fit_stride: int = 2
    log_floor_fraction: float = 1.0e-10
    fd_step: float = 1.0e-5
    max_nfev: int = 35


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    return float(np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0)


def volume_derivative(vertices: np.ndarray, faces: np.ndarray, dvertices: np.ndarray) -> float:
    tri, dtri = vertices[faces], dvertices[faces]
    value = (
        np.einsum("ij,ij->i", dtri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
        + np.einsum("ij,ij->i", tri[:, 0], np.cross(dtri[:, 1], tri[:, 2]))
        + np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], dtri[:, 2]))
    )
    return float(np.sum(value) / 6.0)


def unit_uv_sphere(n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    if n_lat < 3 or n_lon < 6:
        raise ValueError("n_lat >= 3 and n_lon >= 6 are required")
    vertices: list[list[float]] = [[0.0, 0.0, 1.0]]
    rings: list[list[int]] = []
    for ilat in range(1, n_lat):
        theta = np.pi * ilat / n_lat
        ring: list[int] = []
        for ilon in range(n_lon):
            phi = 2.0 * np.pi * ilon / n_lon
            ring.append(len(vertices))
            vertices.append([
                float(np.sin(theta) * np.cos(phi)),
                float(np.sin(theta) * np.sin(phi)),
                float(np.cos(theta)),
            ])
        rings.append(ring)
    south = len(vertices)
    vertices.append([0.0, 0.0, -1.0])
    faces: list[list[int]] = []
    for j in range(n_lon):
        faces.append([0, rings[0][j], rings[0][(j + 1) % n_lon]])
    for upper, lower in zip(rings[:-1], rings[1:]):
        for j in range(n_lon):
            jp = (j + 1) % n_lon
            faces.extend(([upper[j], lower[j], lower[jp]], [upper[j], lower[jp], upper[jp]]))
    for j in range(n_lon):
        faces.append([rings[-1][j], south, rings[-1][(j + 1) % n_lon]])
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    if signed_volume(v, f) < 0.0:
        f = f[:, [0, 2, 1]]
    return v, f


def ellipsoid_with_derivatives(
    theta: np.ndarray,
    axis_w: float,
    base: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    axes = np.asarray([np.exp(theta[0]), np.exp(theta[1]), axis_w])
    vertices = base * axes[None, :]
    du = np.zeros_like(vertices)
    dv = np.zeros_like(vertices)
    du[:, 0] = vertices[:, 0]
    dv[:, 1] = vertices[:, 1]
    return vertices, faces, {"log_axis_u": du, "log_axis_v": dv}


def duffy_nodes(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = 0.5 * (nodes + 1.0), 0.5 * weights
    samples = [
        (float(s), float(t), float(ws * wt * (1.0 - s)))
        for s, ws in zip(nodes, weights)
        for t, wt in zip(nodes, weights)
    ]
    return tuple(np.asarray(x) for x in zip(*samples))  # type: ignore[return-value]


def projected_gradient_grids(
    vertices: np.ndarray,
    faces: np.ndarray,
    derivative_fields: dict[str, np.ndarray],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], float, float, float, float]:
    n = cfg.image_size
    umin = vmin = -cfg.uv_half_extent
    du = dv = 2.0 * cfg.uv_half_extent / n
    u0, v0 = umin + 0.5 * du, vmin + 0.5 * dv
    density_scale = 1.0 / (du * dv)
    tri = vertices[faces]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    normal = np.cross(e1, e2)
    face_derivatives: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, dvertices in derivative_fields.items():
        dtri = dvertices[faces]
        de1, de2 = dtri[:, 1] - dtri[:, 0], dtri[:, 2] - dtri[:, 0]
        face_derivatives[name] = (dvertices, np.cross(de1, e2) + np.cross(e1, de2))

    gu, gv = np.zeros((n, n)), np.zeros((n, n))
    dgrids = {name: (np.zeros_like(gu), np.zeros_like(gv)) for name in derivative_fields}
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    sample_s, sample_t, sample_w = duffy_nodes(cfg.quad_order)
    for s, t, ref_w in zip(sample_s, sample_t, sample_w):
        b0, b1, b2 = (1.0 - s) * (1.0 - t), s, (1.0 - s) * t
        sample = b0 * vertices[i0] + b1 * vertices[i1] + b2 * vertices[i2]
        x, y = (sample[:, 0] - u0) / du, (sample[:, 1] - v0) / dv
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        tx, ty = x - ix, y - iy
        base_weight = float(ref_w) * density_scale
        source_u, source_v = normal[:, 0] * base_weight, normal[:, 1] * base_weight
        sample_derivatives: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, (dvertices, dnormal) in face_derivatives.items():
            dsample = b0 * dvertices[i0] + b1 * dvertices[i1] + b2 * dvertices[i2]
            sample_derivatives[name] = (
                dsample[:, 0] / du,
                dsample[:, 1] / dv,
                dnormal[:, 0] * base_weight,
                dnormal[:, 1] * base_weight,
            )
        for oy in (0, 1):
            yy = iy + oy
            valid_y = (yy >= 0) & (yy < n)
            wy = (1.0 - ty) if oy == 0 else ty
            for ox in (0, 1):
                xx = ix + ox
                valid = valid_y & (xx >= 0) & (xx < n)
                if not np.any(valid):
                    continue
                wx = (1.0 - tx) if ox == 0 else tx
                cic = wx * wy
                np.add.at(gu, (yy[valid], xx[valid]), (source_u * cic)[valid])
                np.add.at(gv, (yy[valid], xx[valid]), (source_v * cic)[valid])
                for name, (dtx, dty, dsource_u, dsource_v) in sample_derivatives.items():
                    dwx = -dtx if ox == 0 else dtx
                    dwy = -dty if oy == 0 else dty
                    dcic = dwx * wy + wx * dwy
                    dgu, dgv = dgrids[name]
                    np.add.at(dgu, (yy[valid], xx[valid]), (dsource_u * cic + source_u * dcic)[valid])
                    np.add.at(dgv, (yy[valid], xx[valid]), (dsource_v * cic + source_v * dcic)[valid])
    return gu, gv, dgrids, du, dv, u0, v0


def continuous_fft2(
    grid: np.ndarray, du: float, dv: float, u0: float, v0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.shape
    q_u = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(w, d=du))
    q_v = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(h, d=dv))
    phase = np.exp(-1j * (q_v[:, None] * v0 + q_u[None, :] * u0))
    spectrum = du * dv * np.fft.fftshift(np.fft.fft2(grid)) * phase
    return spectrum, q_u, q_v


def amplitude_and_derivatives(
    theta: np.ndarray, cfg: Config, base: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    vertices, faces, derivs = ellipsoid_with_derivatives(theta, cfg.axis_w, base, faces)
    gu, gv, dgrids, du, dv, u0, v0 = projected_gradient_grids(vertices, faces, derivs, cfg)
    fu, q_u, q_v = continuous_fft2(gu, du, dv, u0, v0)
    fv, _, _ = continuous_fft2(gv, du, dv, u0, v0)
    q2 = q_u[None, :] ** 2 + q_v[:, None] ** 2
    active = q2 > 1.0e-24
    numerator = q_u[None, :] * fu + q_v[:, None] * fv
    amplitude = np.zeros_like(fu, dtype=np.complex128)
    amplitude[active] = numerator[active] / (-1j * q2[active])
    volume = signed_volume(vertices, faces)
    amplitude[~active] = volume

    damp: dict[str, np.ndarray] = {}
    for name, (dgu, dgv) in dgrids.items():
        dfu, _, _ = continuous_fft2(dgu, du, dv, u0, v0)
        dfv, _, _ = continuous_fft2(dgv, du, dv, u0, v0)
        da = np.zeros_like(amplitude)
        dnumerator = q_u[None, :] * dfu + q_v[:, None] * dfv
        da[active] = dnumerator[active] / (-1j * q2[active])
        da[~active] = volume_derivative(vertices, faces, derivs[name])
        damp[name] = da
    diagnostics = {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "signed_volume": volume,
        "amplitude_q0_real": float(np.real(amplitude[~active][0])),
        "grid_integral_gu": float(np.sum(gu) * du * dv),
        "grid_integral_gv": float(np.sum(gv) * du * dv),
    }
    return amplitude, damp, q_u, q_v, diagnostics


def make_fit_mask(q_u: np.ndarray, q_v: np.ndarray, cfg: Config) -> np.ndarray:
    q_abs = np.sqrt(q_u[None, :] ** 2 + q_v[:, None] ** 2)
    mask = (q_abs >= cfg.q_min) & (q_abs <= cfg.q_max)
    stride_mask = np.zeros_like(mask)
    stride_mask[:: max(cfg.fit_stride, 1), :: max(cfg.fit_stride, 1)] = True
    return mask & stride_mask


def model(
    theta: np.ndarray,
    cfg: Config,
    base: np.ndarray,
    faces: np.ndarray,
    target_log: np.ndarray,
    mask: np.ndarray,
    log_floor: float,
) -> dict[str, Any]:
    amplitude, damp, q_u, q_v, diagnostics = amplitude_and_derivatives(theta, cfg, base, faces)
    scale = float(np.exp(theta[2]))
    signal = np.abs(amplitude) ** 2
    intensity = scale * signal
    residual_full = np.log(intensity + log_floor) - target_log
    denom = intensity + log_floor
    jac_columns = []
    for name in ("log_axis_u", "log_axis_v"):
        dintensity = 2.0 * scale * np.real(np.conj(amplitude) * damp[name])
        jac_columns.append(dintensity / denom)
    jac_columns.append(intensity / denom)
    return {
        "intensity": intensity,
        "residual": residual_full[mask],
        "residual_full": residual_full,
        "jacobian": np.column_stack([column[mask] for column in jac_columns]),
        "q_u": q_u,
        "q_v": q_v,
        "diagnostics": diagnostics,
    }


def verify_derivatives(
    theta: np.ndarray, cfg: Config, base: np.ndarray, faces: np.ndarray
) -> dict[str, float]:
    _, damp, q_u, q_v, _ = amplitude_and_derivatives(theta, cfg, base, faces)
    mask = make_fit_mask(q_u, q_v, cfg)
    errors: dict[str, float] = {}
    for index, name in enumerate(("log_axis_u", "log_axis_v")):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += cfg.fd_step
        minus[index] -= cfg.fd_step
        a_plus = amplitude_and_derivatives(plus, cfg, base, faces)[0]
        a_minus = amplitude_and_derivatives(minus, cfg, base, faces)[0]
        finite = (a_plus - a_minus) / (2.0 * cfg.fd_step)
        num = np.linalg.norm((damp[name] - finite)[mask])
        den = max(np.linalg.norm(finite[mask]), np.finfo(float).tiny)
        errors[name + "_relative_l2"] = float(num / den)
    return errors


def save_diagnostic(
    path: Path,
    target: np.ndarray,
    initial: np.ndarray,
    final: np.ndarray,
    q_u: np.ndarray,
    q_v: np.ndarray,
    history: list[float],
) -> None:
    floor = max(float(np.max(target)) * 1.0e-10, np.finfo(float).tiny)
    extent = [q_u[0], q_u[-1], q_v[0], q_v[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.3), constrained_layout=True)
    for ax, image, title in zip(
        axes[0],
        (target, initial, final),
        ("Target log10 intensity", "Initial log10 intensity", "Recovered log10 intensity"),
    ):
        artist = ax.imshow(np.log10(image + floor), origin="lower", extent=extent, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("q_u")
        ax.set_ylabel("q_v")
        fig.colorbar(artist, ax=ax, shrink=0.8)
    for ax, residual, title in zip(
        axes[1, :2],
        (np.log(initial + floor) - np.log(target + floor), np.log(final + floor) - np.log(target + floor)),
        ("Initial log residual", "Final log residual"),
    ):
        limit = max(float(np.percentile(np.abs(residual), 99)), 1.0e-12)
        artist = ax.imshow(residual, origin="lower", extent=extent, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.set_xlabel("q_u")
        ax.set_ylabel("q_v")
        fig.colorbar(artist, ax=ax, shrink=0.8)
    axes[1, 2].semilogy(np.maximum(history, 1.0e-16), "o-")
    axes[1, 2].set(title="Residual evaluations", xlabel="Evaluation", ylabel="RMS log residual")
    axes[1, 2].grid(True, which="both", alpha=0.3)
    fig.suptitle("Differentiable projected-gradient mesh SAXS", fontsize=15, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_experiment(cfg: Config, output_dir: Path, make_plot: bool = True) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base, faces = unit_uv_sphere(cfg.n_lat, cfg.n_lon)
    true_theta = np.log([cfg.true_axis_u, cfg.true_axis_v, cfg.true_scale])
    initial_theta = np.log([cfg.initial_axis_u, cfg.initial_axis_v, cfg.initial_scale])
    true_amp, _, q_u, q_v, mesh_diag = amplitude_and_derivatives(true_theta, cfg, base, faces)
    target = cfg.true_scale * np.abs(true_amp) ** 2
    log_floor = max(float(np.max(target)) * cfg.log_floor_fraction, np.finfo(float).tiny)
    target_log = np.log(target + log_floor)
    mask = make_fit_mask(q_u, q_v, cfg)
    derivative_check = verify_derivatives(initial_theta, cfg, base, faces)
    history: list[float] = []
    cache: dict[str, Any] = {}

    def components(theta: np.ndarray) -> dict[str, Any]:
        if "theta" not in cache or not np.array_equal(cache["theta"], theta):
            cache["theta"] = theta.copy()
            cache["value"] = model(theta, cfg, base, faces, target_log, mask, log_floor)
        return cache["value"]

    def residual(theta: np.ndarray) -> np.ndarray:
        value = components(theta)["residual"]
        history.append(float(np.sqrt(np.mean(value * value))))
        return value

    initial = model(initial_theta, cfg, base, faces, target_log, mask, log_floor)
    result = least_squares(
        residual,
        initial_theta,
        jac=lambda theta: components(theta)["jacobian"],
        bounds=(np.log([0.45, 0.45, 0.1]), np.log([1.6, 1.6, 3.0])),
        x_scale="jac",
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=cfg.max_nfev,
    )
    final = model(result.x, cfg, base, faces, target_log, mask, log_floor)
    recovered, target_parameters, initial_parameters = np.exp(result.x), np.exp(true_theta), np.exp(initial_theta)
    relative_error = (recovered - target_parameters) / target_parameters
    summary: dict[str, Any] = {
        "method": "CPU Duffy quadrature + CIC projected-gradient + FFT + analytic mesh Jacobian",
        "config": asdict(cfg),
        "mesh": mesh_diag,
        "fit_pixels": int(np.count_nonzero(mask)),
        "derivative_check": derivative_check,
        "optimization": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "initial": dict(zip(("axis_u", "axis_v", "intensity_scale"), map(float, initial_parameters))),
            "target": dict(zip(("axis_u", "axis_v", "intensity_scale"), map(float, target_parameters))),
            "recovered": dict(zip(("axis_u", "axis_v", "intensity_scale"), map(float, recovered))),
            "relative_error": dict(zip(("axis_u", "axis_v", "intensity_scale"), map(float, relative_error))),
            "initial_rms_log_residual": float(np.sqrt(np.mean(initial["residual"] ** 2))),
            "final_rms_log_residual": float(np.sqrt(np.mean(final["residual"] ** 2))),
            "history": history,
        },
    }
    (output_dir / "minimal_example_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if make_plot:
        save_diagnostic(
            output_dir / "minimal_example_diagnostic.png",
            target,
            initial["intensity"],
            final["intensity"],
            q_u,
            q_v,
            history,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--max-nfev", type=int, default=Config.max_nfev)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    cfg = Config(image_size=args.image_size, max_nfev=args.max_nfev)
    summary = run_experiment(cfg, args.output_dir, make_plot=not args.no_plot)
    opt = summary["optimization"]
    print(json.dumps({
        "derivative_check": summary["derivative_check"],
        "initial": opt["initial"],
        "target": opt["target"],
        "recovered": opt["recovered"],
        "initial_rms": opt["initial_rms_log_residual"],
        "final_rms": opt["final_rms_log_residual"],
        "nfev": opt["nfev"],
    }, indent=2))


if __name__ == "__main__":
    main()
