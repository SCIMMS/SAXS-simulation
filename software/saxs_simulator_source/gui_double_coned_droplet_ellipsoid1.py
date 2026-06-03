from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse

import saxs_module as sx
from saxs_module import (
    build_bvh,
    project_density,
    proj_ellipsoid,
    complex_amplitude_from_projected_density,
)

PONI_PATH = "post_beamtime_tmp_saxs.poni"
MASK_PATH = "SAXS_mask.npy"
IMG_SHAPE = (500, 500)
EPS = 1e-12


class ToolTip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self.after_id = None
        widget.bind("<Enter>", self.schedule)
        widget.bind("<Leave>", self.hide)
        widget.bind("<ButtonPress>", self.hide)

    def schedule(self, event=None):
        self.unschedule()
        self.after_id = self.widget.after(self.delay, self.show)

    def unschedule(self):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def show(self):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            font=("맑은 고딕", 9),
            wraplength=380,
        )
        label.pack()

    def hide(self, event=None):
        self.unschedule()
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, width=500):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        # When the pointer is on an entry, that entry handles the wheel for step adjustment.
        if isinstance(event.widget, ttk.Entry) or event.widget.winfo_class() in ("TEntry", "Entry"):
            return
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except tk.TclError:
            pass


class SaveFolderDialog:
    def __init__(self, parent, default_folder_name):
        self.parent = parent
        self.result = None
        self.top = tk.Toplevel(parent)
        self.top.title("Save result")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.resizable(False, False)

        self.parent_dir = tk.StringVar(value="")
        self.folder_name = tk.StringVar(value=default_folder_name)
        self.full_path = tk.StringVar(value="")

        outer = ttk.Frame(self.top, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)

        ttk.Label(outer, text="Parent directory").grid(row=0, column=0, sticky="w")
        parent_row = ttk.Frame(outer)
        parent_row.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        parent_row.columnconfigure(0, weight=1)
        ttk.Entry(parent_row, textvariable=self.parent_dir, width=58).grid(row=0, column=0, sticky="ew")
        ttk.Button(parent_row, text="Browse", command=self.browse).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(outer, text="Result folder name").grid(row=2, column=0, sticky="w")
        folder_entry = ttk.Entry(outer, textvariable=self.folder_name, width=58)
        folder_entry.grid(row=3, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(outer, text="Full save path").grid(row=4, column=0, sticky="w")
        ttk.Entry(outer, textvariable=self.full_path, width=58, state="readonly").grid(
            row=5, column=0, sticky="ew", pady=(2, 10)
        )

        button_row = ttk.Frame(outer)
        button_row.grid(row=6, column=0, sticky="e")
        ttk.Button(button_row, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(button_row, text="Save", command=self.ok).pack(side=tk.RIGHT)

        self.parent_dir.trace_add("write", lambda *_: self.update_full_path())
        self.folder_name.trace_add("write", lambda *_: self.update_full_path())
        self.update_full_path()

        folder_entry.focus_set()
        folder_entry.selection_range(0, tk.END)
        self.top.bind("<Return>", lambda _event: self.ok())
        self.top.bind("<Escape>", lambda _event: self.cancel())
        self.center()
        parent.wait_window(self.top)

    def center(self):
        self.top.update_idletasks()
        x = self.parent.winfo_rootx() + max(0, (self.parent.winfo_width() - self.top.winfo_width()) // 2)
        y = self.parent.winfo_rooty() + max(0, (self.parent.winfo_height() - self.top.winfo_height()) // 2)
        self.top.geometry(f"+{x}+{y}")

    def browse(self):
        selected = filedialog.askdirectory(parent=self.top, title="Choose parent directory")
        if selected:
            self.parent_dir.set(selected)
            self.top.lift()
            self.top.focus_force()

    def sanitized_folder_name(self):
        name = self.folder_name.get().strip()
        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "_")
        return name.strip(" .")

    def update_full_path(self):
        parent = self.parent_dir.get().strip()
        name = self.sanitized_folder_name()
        if parent and name:
            self.full_path.set(str(Path(parent) / name))
        elif parent:
            self.full_path.set(str(Path(parent)))
        else:
            self.full_path.set("")

    def ok(self):
        parent = self.parent_dir.get().strip()
        name = self.sanitized_folder_name()
        if not parent:
            messagebox.showerror("Save error", "Parent directory를 선택해주세요.", parent=self.top)
            return
        parent_path = Path(parent)
        if not parent_path.exists() or not parent_path.is_dir():
            messagebox.showerror("Save error", "선택한 parent directory가 존재하지 않습니다.", parent=self.top)
            return
        if not name:
            messagebox.showerror("Save error", "Result folder name을 입력해주세요.", parent=self.top)
            return
        target = parent_path / name
        if target.exists():
            ok = messagebox.askyesno(
                "Folder exists",
                f"이미 같은 이름의 폴더가 있습니다.\n계속 저장하시겠습니까?\n\n{target}",
                parent=self.top,
            )
            if not ok:
                return
        self.result = target
        self.top.destroy()

    def cancel(self):
        self.result = None
        self.top.destroy()


@dataclass
class IndependentDoubleConedDroplet(sx.Structure):
    """Ellipsoidal body with two independently parameterized cones.

    Cone 1 extends along local +z. Cone 2 extends along local -z.
    Each cone has its own length, neck radius, tip radius, attachment position,
    and tapering parameters.
    """
    body_axes: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float64))

    cone1_length: float = 1.5
    cone1_neck_radius: float = 0.35
    cone1_tip_radius: float = 0.05
    cone1_z_attach: float = 0.70
    cone1_alpha: float = 1.0
    cone1_beta: float = 2.5

    cone2_length: float = 1.5
    cone2_neck_radius: float = 0.35
    cone2_tip_radius: float = 0.05
    cone2_z_attach: float = 0.70
    cone2_alpha: float = 1.0
    cone2_beta: float = 2.5

    blend_k: float = 12.0

    @staticmethod
    def _cone_radius(t: np.ndarray, neck_radius: float, tip_radius: float, alpha: float, beta: float) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        return tip_radius + (neck_radius - tip_radius) * ((1.0 - t ** alpha) ** beta)

    def field(self, points_world: np.ndarray) -> np.ndarray:
        p = sx.world_to_local(points_world, self.center, self.rotation)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]

        a, b, c = self.body_axes
        body = 1.0 - (x / a) ** 2 - (y / b) ** 2 - (z / c) ** 2

        rxy = np.sqrt(x * x + y * y)

        # Cone 1: local +z direction.
        l1 = max(self.cone1_length, 1e-12)
        nr1 = max(self.cone1_neck_radius, self.cone1_tip_radius, 1e-12)
        z0_pos = self.cone1_z_attach * c
        z1_pos = z0_pos + self.cone1_length
        t_pos = (z - z0_pos) / l1
        rz_pos = self._cone_radius(
            t_pos,
            self.cone1_neck_radius,
            self.cone1_tip_radius,
            self.cone1_alpha,
            self.cone1_beta,
        )
        side_pos = (rz_pos - rxy) / nr1
        zcap_pos = np.minimum(z - z0_pos, z1_pos - z) / l1
        cone_pos = np.minimum(side_pos, zcap_pos)

        # Cone 2: local -z direction, 180 degrees opposite to cone 1.
        l2 = max(self.cone2_length, 1e-12)
        nr2 = max(self.cone2_neck_radius, self.cone2_tip_radius, 1e-12)
        z0_neg = -self.cone2_z_attach * c
        z1_neg = z0_neg - self.cone2_length
        t_neg = (z0_neg - z) / l2
        rz_neg = self._cone_radius(
            t_neg,
            self.cone2_neck_radius,
            self.cone2_tip_radius,
            self.cone2_alpha,
            self.cone2_beta,
        )
        side_neg = (rz_neg - rxy) / nr2
        zcap_neg = np.minimum(z0_neg - z, z - z1_neg) / l2
        cone_neg = np.minimum(side_neg, zcap_neg)

        return sx.smooth_max(np.stack([body, cone_pos, cone_neg], axis=0), k=self.blend_k, axis=0)

    def local_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        a, b, c = self.body_axes
        z_pos = self.cone1_z_attach * c + self.cone1_length
        z_neg = -self.cone2_z_attach * c - self.cone2_length
        rmax = max(
            a,
            b,
            self.cone1_neck_radius,
            self.cone1_tip_radius,
            self.cone2_neck_radius,
            self.cone2_tip_radius,
        )
        bmin = np.array([-rmax, -rmax, min(-c, z_neg)], dtype=np.float64)
        bmax = np.array([ rmax,  rmax, max( c, z_pos)], dtype=np.float64)
        return bmin, bmax


_ORIGINAL_MAKE_STRUCTURE = sx.make_structure


def _make_structure_with_independent_double_coned(structure_type: str, **params):
    st = structure_type.lower().strip()
    if st != "independent-double-coned-droplet":
        return _ORIGINAL_MAKE_STRUCTURE(structure_type, **params)

    center = np.asarray(params.pop("center", [0.0, 0.0, 0.0]), dtype=np.float64)
    R = sx._parse_rotation(params.pop("rotation", None), params.pop("rotations", None))
    body_axes = params.pop("body_axes", None)
    if body_axes is None:
        radius = float(params.pop("radius", 1.0))
        body_axes = [radius, radius, radius]

    structure = IndependentDoubleConedDroplet(
        center=center,
        rotation=R,
        body_axes=np.asarray(body_axes, dtype=np.float64),
        cone1_length=float(params.pop("cone1_length", 1.5)),
        cone1_neck_radius=float(params.pop("cone1_neck_radius", 0.35)),
        cone1_tip_radius=float(params.pop("cone1_tip_radius", 0.05)),
        cone1_z_attach=float(params.pop("cone1_z_attach", 0.70)),
        cone1_alpha=float(params.pop("cone1_alpha", 1.0)),
        cone1_beta=float(params.pop("cone1_beta", 2.5)),
        cone2_length=float(params.pop("cone2_length", 1.5)),
        cone2_neck_radius=float(params.pop("cone2_neck_radius", 0.35)),
        cone2_tip_radius=float(params.pop("cone2_tip_radius", 0.05)),
        cone2_z_attach=float(params.pop("cone2_z_attach", 0.70)),
        cone2_alpha=float(params.pop("cone2_alpha", 1.0)),
        cone2_beta=float(params.pop("cone2_beta", 2.5)),
        blend_k=float(params.pop("blend_k", 12.0)),
    )
    if params:
        raise ValueError(f"Unused parameters for independent-double-coned-droplet: {sorted(params.keys())}")
    return structure


sx.make_structure = _make_structure_with_independent_double_coned



def safe_log(x):
    return np.log(np.asarray(x, dtype=np.float64) + EPS)


def get_amplitude(proj, du, dv, scale_factor, translation_u, translation_v):
    A, q_u, q_v, meta = complex_amplitude_from_projected_density(
        projected_density=proj,
        du=du,
        dv=dv,
        scale_factor=scale_factor,
        translation_u=translation_u,
        translation_v=translation_v,
        poni_path=PONI_PATH,
        detector_image_shape=IMG_SHAPE,
        q_unit="A^-1",
        center_mode="beam",
        row_origin="upper",
    )
    return A, q_u, q_v, meta


def projection_extent(proj, du, dv, scale_factor, translation_u, translation_v):
    n_v, n_u = proj.shape
    width_u = float(n_u) * float(du) * float(scale_factor)
    width_v = float(n_v) * float(dv) * float(scale_factor)
    left = float(translation_u) - 0.5 * width_u
    right = float(translation_u) + 0.5 * width_u
    bottom = float(translation_v) - 0.5 * width_v
    top = float(translation_v) + 0.5 * width_v
    return (left, right, bottom, top)


def combined_real_space_limits(extents, margin_fraction=0.08):
    left = min(e[0] for e in extents)
    right = max(e[1] for e in extents)
    bottom = min(e[2] for e in extents)
    top = max(e[3] for e in extents)
    span = max(right - left, top - bottom, 1.0)
    margin = span * margin_fraction
    cx = 0.5 * (left + right)
    cy = 0.5 * (bottom + top)
    half = 0.5 * span + margin
    return (cx - half, cx + half, cy - half, cy + half)


def projected_ellipsoid_preview(body_axes, scale_factor, rot_x_deg, rot_y_deg, rot_z_deg):
    """Return width, height, angle_deg for the detector-plane projection of the body ellipsoid only.

    This excludes cone/tail dimensions, so changing cone length/radius does not
    change the schematic body size in the real-space placement preview.
    """
    a, b, c = [float(x) for x in body_axes]
    rx, ry, rz = np.deg2rad([float(rot_x_deg), float(rot_y_deg), float(rot_z_deg)])

    def R_axis(axis, th):
        ct, st = np.cos(th), np.sin(th)
        if axis == "x":
            return np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]], dtype=float)
        if axis == "y":
            return np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]], dtype=float)
        return np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    # Match sx.compose_axis_rotations order used in run_simulation: Rx then Ry then Rz.
    R = R_axis("z", rz) @ R_axis("y", ry) @ R_axis("x", rx)
    Q = R @ np.diag([a * a, b * b, c * c]) @ R.T
    Q2 = Q[:2, :2]
    vals, vecs = np.linalg.eigh(Q2)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    width = 2.0 * float(scale_factor) * float(np.sqrt(max(vals[0], 0.0)))
    height = 2.0 * float(scale_factor) * float(np.sqrt(max(vals[1], 0.0)))
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return width, height, angle


def make_double_coned_droplet(params):
    bvh = build_bvh(
        "independent-double-coned-droplet",
        body_axes=[params["body_x"], params["body_y"], params["body_z"]],
        cone1_length=params["cone1_length"],
        cone1_neck_radius=params["cone1_neck_radius"],
        cone1_tip_radius=params["cone1_tip_radius"],
        cone1_z_attach=params["cone1_z_attach"],
        cone1_alpha=params["cone1_alpha"],
        cone1_beta=params["cone1_beta"],
        cone2_length=params["cone2_length"],
        cone2_neck_radius=params["cone2_neck_radius"],
        cone2_tip_radius=params["cone2_tip_radius"],
        cone2_z_attach=params["cone2_z_attach"],
        cone2_alpha=params["cone2_alpha"],
        cone2_beta=params["cone2_beta"],
        rotations=[
            ([1, 0, 0], np.deg2rad(params["rot_x_deg"])),
            ([0, 1, 0], np.deg2rad(params["rot_y_deg"])),
            ([0, 0, 1], np.deg2rad(params["rot_z_deg"])),
        ],
        field_shape=(
            int(params["field_shape"]),
            int(params["field_shape"]),
            int(params["field_shape"]),
        ),
    )

    proj, du, dv, meta_proj = project_density(
        bvh,
        ray_vector=(0.0, 0.0, 1.0),
        roll_rad=0.0,
        image_shape=IMG_SHAPE,
        out_dtype="float32",
    )
    A, q_u, q_v, meta = get_amplitude(
        proj=proj,
        du=du,
        dv=dv,
        scale_factor=params["drop_scale"],
        translation_u=params["drop_tu"],
        translation_v=params["drop_tv"],
    )
    return proj, du, dv, A, q_u, q_v, meta_proj, meta


def make_ellipsoid(params):
    proj, du, dv = proj_ellipsoid(
        b=params["ell_b"],
        angle=np.deg2rad(params["ell_angle_deg"]),
        img_shape=IMG_SHAPE,
    )
    A, q_u, q_v, meta = get_amplitude(
        proj=proj,
        du=du,
        dv=dv,
        scale_factor=params["ell_scale"],
        translation_u=params["ell_tu"],
        translation_v=params["ell_tv"],
    )
    return proj, du, dv, A, q_u, q_v, meta


def run_simulation(params):
    drop_proj, drop_du, drop_dv, A_drop, q_u, q_v, meta_proj_drop, meta_drop = make_double_coned_droplet(params)
    ell_proj, ell_du, ell_dv, A_ell, _, _, meta_ell = make_ellipsoid(params)

    A_total = params["drop_weight"] * A_drop + params["ell_weight"] * A_ell
    I_total = np.abs(A_total) ** 2

    return {
        "drop_proj": drop_proj,
        "ell_proj": ell_proj,
        "drop_du": drop_du,
        "drop_dv": drop_dv,
        "ell_du": ell_du,
        "ell_dv": ell_dv,
        "A_drop": A_drop,
        "A_ell": A_ell,
        "A_total": A_total,
        "I_total": I_total,
        "q_u": q_u,
        "q_v": q_v,
        "meta_proj_drop": meta_proj_drop,
        "meta_drop": meta_drop,
        "meta_ell": meta_ell,
    }


class DoubleConedDropletEllipsoidGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Independent Double Coned Droplet + Ellipsoid SAXS Simulator")
        self.root.geometry("1750x900")
        self.update_job = None
        self.last_result = None

        self.mask = np.load(MASK_PATH)
        if self.mask.shape != IMG_SHAPE:
            self.mask = self.mask.reshape(IMG_SHAPE)

        self.vars = {}
        self.entries = {}
        self.slider_meta = {}
        self.sliders = {}
        self._setting_value = set()

        self._build_ui()
        self.update_plot()

    def defaults(self):
        return {
            "body_x": 1.0,
            "body_y": 2.0,
            "body_z": 1.0,
            "cone1_length": 1.8,
            "cone1_neck_radius": 0.34,
            "cone1_tip_radius": 0.025,
            "cone1_z_attach": 0.72,
            "cone1_alpha": 1.0,
            "cone1_beta": 2.5,
            "cone2_length": 1.8,
            "cone2_neck_radius": 0.34,
            "cone2_tip_radius": 0.025,
            "cone2_z_attach": 0.72,
            "cone2_alpha": 1.0,
            "cone2_beta": 2.5,
            "rot_x_deg": -18.0,
            "rot_y_deg": 22.0,
            "rot_z_deg": 12.0,
            "field_shape": 120.0,
            "drop_scale": 5000.0,
            "drop_tu": -1500.0,
            "drop_tv": 0.0,
            "drop_weight": 1.0,
            "ell_b": 0.5,
            "ell_angle_deg": 35.0,
            "ell_scale": 1200.0,
            "ell_tu": 1500.0,
            "ell_tv": 500.0,
            "ell_weight": 3.0,
        }

    def tooltip(self, name):
        tips = {
            "body_x": "coned droplet body ellipsoid의 local x축 반축 길이입니다.",
            "body_y": "coned droplet body ellipsoid의 local y축 반축 길이입니다.",
            "body_z": "coned droplet body ellipsoid의 local z축 반축 길이입니다. cone/tail 연결 위치와도 관련됩니다.",
            "cone1_length": "Cone 1(local +z 방향)의 길이입니다.",
            "cone1_neck_radius": "Cone 1이 body와 만나는 neck 부분의 반지름입니다.",
            "cone1_tip_radius": "Cone 1 끝부분의 반지름입니다. 작을수록 sharp tip에 가깝습니다.",
            "cone1_z_attach": "Cone 1이 body의 +z 쪽 어느 위치에서 시작되는지 정합니다.",
            "cone1_alpha": "Cone 1 tapering의 초기 감소 형태를 조절합니다.",
            "cone1_beta": "Cone 1 tapering의 곡률을 조절합니다.",
            "cone2_length": "Cone 2(local -z 방향)의 길이입니다.",
            "cone2_neck_radius": "Cone 2가 body와 만나는 neck 부분의 반지름입니다.",
            "cone2_tip_radius": "Cone 2 끝부분의 반지름입니다. 작을수록 sharp tip에 가깝습니다.",
            "cone2_z_attach": "Cone 2가 body의 -z 쪽 어느 위치에서 시작되는지 정합니다.",
            "cone2_alpha": "Cone 2 tapering의 초기 감소 형태를 조절합니다.",
            "cone2_beta": "Cone 2 tapering의 곡률을 조절합니다.",
            "rot_x_deg": "coned droplet 전체를 x축 기준으로 회전시키는 각도입니다.",
            "rot_y_deg": "coned droplet 전체를 y축 기준으로 회전시키는 각도입니다.",
            "rot_z_deg": "coned droplet 전체를 z축 기준으로 회전시키는 각도입니다.",
            "field_shape": "coned droplet mesh 해상도입니다. 클수록 정교하지만 계산이 느려집니다.",
            "drop_scale": "coned droplet의 실공간 크기 scale입니다.",
            "drop_tu": "coned droplet의 real-space u 방향 translation입니다.",
            "drop_tv": "coned droplet의 real-space v 방향 translation입니다.",
            "drop_weight": "coned droplet complex amplitude의 가중치입니다.",
            "ell_b": "ellipsoid의 단축 비율입니다. b=1이면 원형 projection에 가깝습니다.",
            "ell_angle_deg": "ellipsoid projected density의 detector plane 내 회전 각도입니다.",
            "ell_scale": "ellipsoid의 실공간 크기 scale입니다.",
            "ell_tu": "ellipsoid의 real-space u 방향 translation입니다.",
            "ell_tv": "ellipsoid의 real-space v 방향 translation입니다.",
            "ell_weight": "ellipsoid complex amplitude의 가중치입니다.",
        }
        return tips.get(name, "")

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        scroll = ScrollableFrame(main, width=510)
        scroll.pack(side=tk.LEFT, fill=tk.Y)
        controls = scroll.inner

        plot_area = ttk.Frame(main, padding=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls,
            text="Independent Double Coned Droplet + Ellipsoid Parameters",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(4, 10), padx=8)

        for name, value in self.defaults().items():
            self.vars[name] = tk.DoubleVar(value=value)

        body_frame = ttk.LabelFrame(controls, text="Object 1: Coned droplet body", padding=6)
        body_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(body_frame, "body_x", 0.2, 3.0, 0.1)
        self._add(body_frame, "body_y", 0.2, 3.0, 0.1)
        self._add(body_frame, "body_z", 0.2, 3.0, 0.1)

        cone1_frame = ttk.LabelFrame(controls, text="Object 1: Cone 1 (+z)", padding=6)
        cone1_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(cone1_frame, "cone1_length", 0.2, 4.0, 0.1)
        self._add(cone1_frame, "cone1_neck_radius", 0.05, 1.0, 0.01)
        self._add(cone1_frame, "cone1_tip_radius", 0.005, 0.2, 0.005)
        self._add(cone1_frame, "cone1_z_attach", 0.0, 1.0, 0.01)
        self._add(cone1_frame, "cone1_alpha", 0.5, 3.0, 0.1)
        self._add(cone1_frame, "cone1_beta", 1.0, 5.0, 0.1)

        cone2_frame = ttk.LabelFrame(controls, text="Object 1: Cone 2 (-z)", padding=6)
        cone2_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(cone2_frame, "cone2_length", 0.2, 4.0, 0.1)
        self._add(cone2_frame, "cone2_neck_radius", 0.05, 1.0, 0.01)
        self._add(cone2_frame, "cone2_tip_radius", 0.005, 0.2, 0.005)
        self._add(cone2_frame, "cone2_z_attach", 0.0, 1.0, 0.01)
        self._add(cone2_frame, "cone2_alpha", 0.5, 3.0, 0.1)
        self._add(cone2_frame, "cone2_beta", 1.0, 5.0, 0.1)

        droplet_rot_frame = ttk.LabelFrame(controls, text="Object 1: 3D rotation", padding=6)
        droplet_rot_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(droplet_rot_frame, "rot_x_deg", -90.0, 90.0, 1.0)
        self._add(droplet_rot_frame, "rot_y_deg", -90.0, 90.0, 1.0)
        self._add(droplet_rot_frame, "rot_z_deg", -180.0, 180.0, 1.0)

        droplet_scat_frame = ttk.LabelFrame(controls, text="Object 1: scattering / placement", padding=6)
        droplet_scat_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(droplet_scat_frame, "field_shape", 50.0, 180.0, 10.0)
        self._add(droplet_scat_frame, "drop_scale", 100.0, 20000.0, 100.0)
        self._add(droplet_scat_frame, "drop_tu", -8000.0, 8000.0, 50.0)
        self._add(droplet_scat_frame, "drop_tv", -8000.0, 8000.0, 50.0)
        self._add(droplet_scat_frame, "drop_weight", 0.0, 50.0, 0.5)

        ell_frame = ttk.LabelFrame(controls, text="Object 2: Ellipsoid", padding=6)
        ell_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(ell_frame, "ell_b", 0.05, 1.0, 0.01)
        self._add(ell_frame, "ell_angle_deg", -180.0, 180.0, 1.0)
        self._add(ell_frame, "ell_scale", 50.0, 10000.0, 50.0)
        self._add(ell_frame, "ell_tu", -8000.0, 8000.0, 50.0)
        self._add(ell_frame, "ell_tv", -8000.0, 8000.0, 50.0)
        self._add(ell_frame, "ell_weight", 0.0, 50.0, 0.5)

        ttk.Button(controls, text="Save current result", command=self.save).pack(fill=tk.X, padx=8, pady=(10, 4))
        ttk.Button(controls, text="Reset", command=self.reset).pack(fill=tk.X, padx=8, pady=4)

        self.info = tk.Text(controls, width=54, height=10)
        self.info.pack(fill=tk.X, padx=8, pady=(10, 10))

        self.fig, self.ax = plt.subplots(1, 4, figsize=(15.8, 4.4))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add(self, parent, name, vmin, vmax, resolution):
        self.slider_meta[name] = {"vmin": float(vmin), "vmax": float(vmax), "step": float(resolution)}
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        f.columnconfigure(2, weight=1)

        label = ttk.Label(f, text=name, width=14)
        label.grid(row=0, column=0, sticky="w")
        ToolTip(label, self.tooltip(name))

        ttk.Button(f, text="◀", width=3, command=lambda key=name: self.step_value(key, -1)).grid(
            row=0, column=1, padx=(0, 2)
        )
        slider = ttk.Scale(
            f,
            from_=vmin,
            to=vmax,
            orient=tk.HORIZONTAL,
            variable=self.vars[name],
            command=lambda _value, key=name: self.on_slider_change(key),
        )
        slider.grid(row=0, column=2, sticky="ew")
        self.sliders[name] = slider
        ttk.Button(f, text="▶", width=3, command=lambda key=name: self.step_value(key, 1)).grid(
            row=0, column=3, padx=(2, 4)
        )

        entry = ttk.Entry(f, width=10)
        entry.insert(0, f"{self.vars[name].get():.6g}")
        entry.grid(row=0, column=4, sticky="e")
        self.entries[name] = entry

        entry.bind("<Return>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<FocusOut>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<Up>", lambda event, key=name: self.on_entry_step_key(event, key, 1))
        entry.bind("<Right>", lambda event, key=name: self.on_entry_step_key(event, key, 1))
        entry.bind("<Down>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<Left>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<MouseWheel>", lambda event, key=name: self.on_entry_mousewheel(event, key))
        entry.bind("<Button-4>", lambda event, key=name: self.on_entry_step_key(event, key, 1))
        entry.bind("<Button-5>", lambda event, key=name: self.on_entry_step_key(event, key, -1))

    def clamp_to_range(self, name, value):
        meta = self.slider_meta[name]
        return min(max(float(value), meta["vmin"]), meta["vmax"])

    def round_to_step(self, name, value):
        meta = self.slider_meta[name]
        step = meta["step"]
        vmin = meta["vmin"]
        value = self.clamp_to_range(name, value)
        if step > 0:
            value = vmin + round((value - vmin) / step) * step
            value = self.clamp_to_range(name, value)
            decimals = max(0, int(np.ceil(-np.log10(step))) + 2) if step < 1 else 6
            value = round(value, decimals)
        return value

    def set_value(self, name, value, schedule=True):
        value = self.round_to_step(name, value)
        self._setting_value.add(name)
        try:
            self.vars[name].set(value)
        finally:
            self._setting_value.discard(name)
        self.sync_entry(name)
        if schedule:
            self.schedule()

    def step_value(self, name, direction):
        meta = self.slider_meta[name]
        try:
            current = float(self.entries[name].get())
        except Exception:
            current = float(self.vars[name].get())
        self.set_value(name, current + float(direction) * meta["step"])
        self.entries[name].focus_set()
        self.entries[name].selection_range(0, tk.END)

    def on_entry_step_key(self, event, name, direction):
        self.step_value(name, direction)
        return "break"

    def on_entry_mousewheel(self, event, name):
        direction = 1 if event.delta > 0 else -1
        self.step_value(name, direction)
        return "break"

    def on_slider_change(self, name):
        if name in self._setting_value:
            return
        self.set_value(name, self.vars[name].get(), schedule=True)

    def on_entry_change(self, name):
        try:
            value = float(self.entries[name].get())
        except ValueError:
            messagebox.showerror("Input error", f"{name} must be numeric.")
            self.sync_entry(name)
            return
        self.set_value(name, value)

    def sync_entry(self, name):
        entry = self.entries[name]
        entry.delete(0, tk.END)
        entry.insert(0, f"{self.vars[name].get():.6g}")

    def schedule(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(600, self.update_plot)

    def params(self):
        p = {k: float(v.get()) for k, v in self.vars.items()}
        p["field_shape"] = int(round(p["field_shape"]))
        p["field_shape"] = max(30, p["field_shape"])
        return p

    def reset(self):
        for name, value in self.defaults().items():
            self.vars[name].set(value)
            self.sync_entry(name)
        self.update_plot()

    def draw_translated_density_sum(self, ax, result, p):
        ax.clear()
        ax.set_title("translated projected density sum")
        ax.set_aspect("equal", adjustable="box")

        extent_drop = projection_extent(
            result["drop_proj"], result["drop_du"], result["drop_dv"],
            p["drop_scale"], p["drop_tu"], p["drop_tv"]
        )
        extent_ell = projection_extent(
            result["ell_proj"], result["ell_du"], result["ell_dv"],
            p["ell_scale"], p["ell_tu"], p["ell_tv"]
        )
        ax.imshow(p["drop_weight"] * result["drop_proj"], extent=extent_drop, origin="lower", interpolation="nearest")
        ax.imshow(p["ell_weight"] * result["ell_proj"], extent=extent_ell, origin="lower", interpolation="nearest", alpha=0.65)
        xmin, xmax, ymin, ymax = combined_real_space_limits([extent_drop, extent_ell])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("real-space u")
        ax.set_ylabel("real-space v")
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_position_preview(self, ax, p):
        ax.clear()
        ax.set_title("real-space placement preview")
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, linewidth=0.6)
        ax.axvline(0, linewidth=0.6)

        result = getattr(self, "last_result", None)
        extents = []
        if result is not None:
            drop_extent = projection_extent(result["drop_proj"], result["drop_du"], result["drop_dv"], p["drop_scale"], p["drop_tu"], p["drop_tv"])
            ell_extent = projection_extent(result["ell_proj"], result["ell_du"], result["ell_dv"], p["ell_scale"], p["ell_tu"], p["ell_tv"])
            extents = [drop_extent, ell_extent]

        drop_width, drop_height, drop_angle = projected_ellipsoid_preview(
            [p["body_x"], p["body_y"], p["body_z"]],
            p["drop_scale"], p["rot_x_deg"], p["rot_y_deg"], p["rot_z_deg"]
        )
        if extents:
            ell_width = extents[1][1] - extents[1][0]
            ell_height = ell_width * p["ell_b"]
        else:
            ell_width = 2.0 * p["ell_scale"]
            ell_height = 2.0 * p["ell_scale"] * p["ell_b"]

        drop_patch = Ellipse((p["drop_tu"], p["drop_tv"]), width=drop_width, height=drop_height,
                             angle=drop_angle, fill=False, linewidth=1.5)
        ax.add_patch(drop_patch)
        ax.text(p["drop_tu"], p["drop_tv"], "D", ha="center", va="center")

        ell_patch = Ellipse((p["ell_tu"], p["ell_tv"]), width=ell_width, height=ell_height,
                            angle=p["ell_angle_deg"], fill=False, linewidth=1.5)
        ax.add_patch(ell_patch)
        ax.text(p["ell_tu"], p["ell_tv"], "E", ha="center", va="center")

        if extents:
            xmin, xmax, ymin, ymax = combined_real_space_limits(extents)
        else:
            centers = np.array([[p["drop_tu"], p["drop_tv"]], [p["ell_tu"], p["ell_tv"]]], dtype=float)
            max_abs = max(1.0, float(np.max(np.abs(centers))) * 1.25)
            xmin, xmax, ymin, ymax = -max_abs, max_abs, -max_abs, max_abs
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("u translation")
        ax.set_ylabel("v translation")

    def update_plot(self):
        self.update_job = None
        try:
            p = self.params()
            result = run_simulation(p)
        except Exception as e:
            self.show_error(e)
            return

        I_total = result["I_total"]
        I_masked = I_total * self.mask
        result["I_masked"] = I_masked
        result["parameters"] = p
        self.last_result = result

        self.draw_translated_density_sum(self.ax[0], result, p)

        images = [safe_log(I_total), I_masked]
        titles = ["log theoretical: |wD AD + wE AE|²", "masked theoretical"]
        for ax, img, title in zip(self.ax[1:3], images, titles):
            ax.clear()
            ax.matshow(img)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self.draw_position_preview(self.ax[3], p)
        self.fig.tight_layout()
        self.canvas.draw_idle()

        q_u = result["q_u"]
        q_v = result["q_v"]
        du_rel = p["ell_tu"] - p["drop_tu"]
        dv_rel = p["ell_tv"] - p["drop_tv"]
        dr = float(np.hypot(du_rel, dv_rel))

        self.info.delete("1.0", tk.END)
        self.info.insert(
            tk.END,
            f"A_total = {p['drop_weight']}*A_drop + {p['ell_weight']}*A_ell\n\n"
            f"double-drop position = ({p['drop_tu']:.6g}, {p['drop_tv']:.6g})\n"
            f"ell position = ({p['ell_tu']:.6g}, {p['ell_tv']:.6g})\n"
            f"relative Δu, Δv = ({du_rel:.6g}, {dv_rel:.6g})\n"
            f"relative distance = {dr:.6g}\n\n"
            f"field_shape = {p['field_shape']}\n"
            f"q_u = [{q_u.min():.6g}, {q_u.max():.6g}]\n"
            f"q_v = [{q_v.min():.6g}, {q_v.max():.6g}]\n"
        )

    def save(self):
        if self.last_result is None:
            messagebox.showerror("Save error", "No simulation result to save.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dialog = SaveFolderDialog(self.root, f"double_coned_droplet_ellipsoid_sim_{timestamp}")
        d = dialog.result
        if d is None:
            return
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Save error", f"Could not create save folder:\n{d}\n\n{e}")
            return
        try:
            self.save_translated_density_sum(d / "01_translated_projected_density_sum.png")
            self.save_image(d / "02_log_theoretical.png", safe_log(self.last_result["I_total"]), "log theoretical")
            self.save_image(d / "03_masked_theoretical.png", self.last_result["I_masked"], "masked theoretical")
            self.save_position_preview(d / "04_real_space_placement_preview.png")
            self.save_params(d / "parameters.txt")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save result:\n{d}\n\n{e}")
            return
        messagebox.showinfo("Saved", f"Saved result to:\n{d}")

    def save_translated_density_sum(self, path):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.draw_translated_density_sum(ax, self.last_result, self.last_result["parameters"])
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def save_position_preview(self, path):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.draw_position_preview(ax, self.last_result["parameters"])
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def save_image(self, path, img, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def save_params(self, path):
        p = self.last_result["parameters"]
        q_u = self.last_result["q_u"]
        q_v = self.last_result["q_v"]
        lines = [
            "Independent double-coned droplet + Ellipsoid SAXS simulation parameters",
            "=" * 58,
            "",
            "[Merge equation]",
            "A_total = drop_weight * A_double_coned_droplet + ell_weight * A_ellipsoid",
            "I_total = |A_total|^2",
            "",
            "[Object 1: Independent double-coned droplet]",
            f"body_axes = [{p['body_x']}, {p['body_y']}, {p['body_z']}]",
            f"cone1_length = {p['cone1_length']}",
            f"cone1_neck_radius = {p['cone1_neck_radius']}",
            f"cone1_tip_radius = {p['cone1_tip_radius']}",
            f"cone1_z_attach = {p['cone1_z_attach']}",
            f"cone1_alpha = {p['cone1_alpha']}",
            f"cone1_beta = {p['cone1_beta']}",
            f"cone2_length = {p['cone2_length']}",
            f"cone2_neck_radius = {p['cone2_neck_radius']}",
            f"cone2_tip_radius = {p['cone2_tip_radius']}",
            f"cone2_z_attach = {p['cone2_z_attach']}",
            f"cone2_alpha = {p['cone2_alpha']}",
            f"cone2_beta = {p['cone2_beta']}",
            f"rot_x_deg = {p['rot_x_deg']}",
            f"rot_y_deg = {p['rot_y_deg']}",
            f"rot_z_deg = {p['rot_z_deg']}",
            f"field_shape = {p['field_shape']}",
            f"drop_scale = {p['drop_scale']}",
            f"drop_translation_u = {p['drop_tu']}",
            f"drop_translation_v = {p['drop_tv']}",
            f"drop_weight = {p['drop_weight']}",
            "",
            "[Object 2: Ellipsoid]",
            f"ell_b = {p['ell_b']}",
            f"ell_angle_deg = {p['ell_angle_deg']}",
            f"ell_scale = {p['ell_scale']}",
            f"ell_translation_u = {p['ell_tu']}",
            f"ell_translation_v = {p['ell_tv']}",
            f"ell_weight = {p['ell_weight']}",
            "",
            "[Relative placement]",
            f"delta_u = {p['ell_tu'] - p['drop_tu']}",
            f"delta_v = {p['ell_tv'] - p['drop_tv']}",
            f"delta_r = {float(np.hypot(p['ell_tu'] - p['drop_tu'], p['ell_tv'] - p['drop_tv']))}",
            "",
            "[Files]",
            f"poni_path = {PONI_PATH}",
            f"mask_path = {MASK_PATH}",
            "",
            "[q axes]",
            f"q_u_min = {float(q_u.min())}",
            f"q_u_max = {float(q_u.max())}",
            f"q_v_min = {float(q_v.min())}",
            f"q_v_max = {float(q_v.max())}",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def show_error(self, e):
        for ax in self.ax:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax[1].text(
            0.5,
            0.5,
            f"Error:\n{e}",
            ha="center",
            va="center",
            transform=self.ax[1].transAxes,
            wrap=True,
        )
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = DoubleConedDropletEllipsoidGUI(root)
    root.mainloop()
