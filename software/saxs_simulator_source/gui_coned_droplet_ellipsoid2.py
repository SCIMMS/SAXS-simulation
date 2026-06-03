from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse

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
    def __init__(self, parent, width=520):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        # Do not steal wheel events from numeric entries.
        if isinstance(self.focus_get(), ttk.Entry):
            return
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except tk.TclError:
            pass

    def _on_mousewheel_linux(self, event):
        if isinstance(self.focus_get(), ttk.Entry):
            return
        direction = -1 if event.num == 4 else 1
        try:
            self.canvas.yview_scroll(direction, "units")
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

        ttk.Label(outer, text="Parent directory").grid(row=0, column=0, sticky="w")
        parent_row = ttk.Frame(outer)
        parent_row.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        parent_row.columnconfigure(0, weight=1)
        ttk.Entry(parent_row, textvariable=self.parent_dir, width=60).grid(row=0, column=0, sticky="ew")
        ttk.Button(parent_row, text="Browse", command=self.browse).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(outer, text="Result folder name").grid(row=2, column=0, sticky="w")
        folder_entry = ttk.Entry(outer, textvariable=self.folder_name, width=60)
        folder_entry.grid(row=3, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(outer, text="Full save path").grid(row=4, column=0, sticky="w")
        ttk.Entry(outer, textvariable=self.full_path, width=60, state="readonly").grid(
            row=5, column=0, sticky="ew", pady=(2, 10)
        )

        button_row = ttk.Frame(outer)
        button_row.grid(row=6, column=0, sticky="e")
        ttk.Button(button_row, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(button_row, text="Save", command=self.ok).pack(side=tk.RIGHT)

        outer.columnconfigure(0, weight=1)
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

    def sanitized_folder_name(self):
        name = self.folder_name.get().strip()
        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "_")
        return name.strip(" .")

    def update_full_path(self):
        parent = self.parent_dir.get().strip()
        name = self.sanitized_folder_name()
        self.full_path.set(str(Path(parent) / name) if parent and name else "")

    def ok(self):
        parent = self.parent_dir.get().strip()
        name = self.sanitized_folder_name()
        if not parent:
            messagebox.showerror("Save error", "Parent directory를 선택해주세요.", parent=self.top)
            return
        if not Path(parent).exists():
            messagebox.showerror("Save error", "선택한 parent directory가 존재하지 않습니다.", parent=self.top)
            return
        if not name:
            messagebox.showerror("Save error", "Result folder name을 입력해주세요.", parent=self.top)
            return
        target = Path(parent) / name
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
    return (
        float(translation_u) - 0.5 * width_u,
        float(translation_u) + 0.5 * width_u,
        float(translation_v) - 0.5 * width_v,
        float(translation_v) + 0.5 * width_v,
    )


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
    return cx - half, cx + half, cy - half, cy + half


def projected_ellipsoid_preview(body_axes, scale_factor, rot_x_deg, rot_y_deg, rot_z_deg):
    """Return width, height, angle_deg for the detector-plane projection of the body ellipsoid only.

    This intentionally excludes cone/tail dimensions so changing cone length/radius does not
    make the main droplet body look larger or squashed in the schematic placement preview.
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


def build_coned_droplet_projection(params):
    bvh = build_bvh(
        "coned-droplet",
        body_axes=[params["cd_body_x"], params["cd_body_y"], params["cd_body_z"]],
        cone_length=params["cd_cone_length"],
        neck_radius=params["cd_neck_radius"],
        tip_radius=params["cd_tip_radius"],
        z_attach=params["cd_z_attach"],
        alpha=params["cd_alpha"],
        beta=params["cd_beta"],
        rotations=[
            ([1, 0, 0], np.deg2rad(params["cd_rot_x_deg"])),
            ([0, 1, 0], np.deg2rad(params["cd_rot_y_deg"])),
            ([0, 0, 1], np.deg2rad(params["cd_rot_z_deg"])),
        ],
        field_shape=(
            int(params["cd_field_shape"]),
            int(params["cd_field_shape"]),
            int(params["cd_field_shape"]),
        ),
    )
    proj, du, dv, meta_proj = project_density(
        bvh,
        ray_vector=(0.0, 0.0, 1.0),
        roll_rad=0.0,
        image_shape=IMG_SHAPE,
        out_dtype="float32",
    )
    return proj, du, dv, meta_proj


def build_ellipsoid_projection(prefix, params):
    proj, du, dv = proj_ellipsoid(
        b=params[f"{prefix}_b"],
        angle=np.deg2rad(params[f"{prefix}_angle_deg"]),
        img_shape=IMG_SHAPE,
    )
    return proj, du, dv, None


def run_simulation(params):
    cd_proj, cd_du, cd_dv, cd_meta_proj = build_coned_droplet_projection(params)
    e1_proj, e1_du, e1_dv, _ = build_ellipsoid_projection("ell1", params)
    e2_proj, e2_du, e2_dv, _ = build_ellipsoid_projection("ell2", params)

    A_cd, q_u, q_v, cd_meta = get_amplitude(
        cd_proj,
        cd_du,
        cd_dv,
        params["cd_scale"],
        params["cd_tu"],
        params["cd_tv"],
    )
    A_e1, _, _, e1_meta = get_amplitude(
        e1_proj,
        e1_du,
        e1_dv,
        params["ell1_scale"],
        params["ell1_tu"],
        params["ell1_tv"],
    )
    A_e2, _, _, e2_meta = get_amplitude(
        e2_proj,
        e2_du,
        e2_dv,
        params["ell2_scale"],
        params["ell2_tu"],
        params["ell2_tv"],
    )

    A_total = params["cd_weight"] * A_cd + params["ell1_weight"] * A_e1 + params["ell2_weight"] * A_e2
    I_total = np.abs(A_total) ** 2

    return {
        "cd_proj": cd_proj,
        "ell1_proj": e1_proj,
        "ell2_proj": e2_proj,
        "cd_du": cd_du,
        "cd_dv": cd_dv,
        "ell1_du": e1_du,
        "ell1_dv": e1_dv,
        "ell2_du": e2_du,
        "ell2_dv": e2_dv,
        "A_cd": A_cd,
        "A_ell1": A_e1,
        "A_ell2": A_e2,
        "A_total": A_total,
        "I_total": I_total,
        "q_u": q_u,
        "q_v": q_v,
        "cd_meta_proj": cd_meta_proj,
        "cd_meta": cd_meta,
        "ell1_meta": e1_meta,
        "ell2_meta": e2_meta,
    }


class ConedDropletTwoEllipsoidsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Coned Droplet + Two Ellipsoids SAXS Simulator")
        self.root.geometry("1780x900")
        self.update_job = None
        self.last_result = None

        self.mask = np.load(MASK_PATH)
        if self.mask.shape != IMG_SHAPE:
            self.mask = self.mask.reshape(IMG_SHAPE)

        self.vars = {}
        self.entries = {}
        self.slider_meta = {}
        self._programmatic_update = set()

        self._build_ui()
        self.update_plot()

    def tooltip(self, name):
        tips = {
            "cd_body_x": "coned droplet body ellipsoid의 local x 반축 길이입니다.",
            "cd_body_y": "coned droplet body ellipsoid의 local y 반축 길이입니다.",
            "cd_body_z": "coned droplet body ellipsoid의 local z 반축 길이입니다. cone/tail 연결부에도 영향을 줍니다.",
            "cd_cone_length": "tail의 local +z 방향 길이입니다.",
            "cd_neck_radius": "body와 tail이 만나는 neck 반지름입니다.",
            "cd_tip_radius": "tail 끝부분의 반지름입니다.",
            "cd_z_attach": "tail이 body의 local z축 어느 위치에서 시작되는지 정합니다.",
            "cd_alpha": "tail 반지름 변화의 초기 형태를 조절합니다.",
            "cd_beta": "tail tapering 곡률을 조절합니다.",
            "cd_rot_x_deg": "coned droplet 전체를 x축 기준으로 회전시키는 각도입니다.",
            "cd_rot_y_deg": "coned droplet 전체를 y축 기준으로 회전시키는 각도입니다.",
            "cd_rot_z_deg": "coned droplet 전체를 z축 기준으로 회전시키는 각도입니다.",
            "cd_scale": "coned droplet의 실공간 크기 scale입니다.",
            "cd_tu": "coned droplet의 real-space u 방향 위치입니다.",
            "cd_tv": "coned droplet의 real-space v 방향 위치입니다.",
            "cd_weight": "coned droplet complex amplitude의 가중치입니다.",
            "cd_field_shape": "coned droplet 3D mesh field 해상도입니다. 클수록 정교하지만 느립니다.",
        }
        if name.startswith("ell"):
            if name.endswith("_b"):
                return "ellipsoid projected density의 단축 비율입니다. b=1이면 원형에 가깝습니다."
            if name.endswith("_angle_deg"):
                return "ellipsoid projected density의 detector plane 내 회전 각도입니다."
            if name.endswith("_scale"):
                return "ellipsoid의 실공간 크기 scale입니다. 커질수록 q-space feature가 좁아집니다."
            if name.endswith("_tu"):
                return "ellipsoid의 real-space u 방향 위치입니다. coherent interference phase를 바꿉니다."
            if name.endswith("_tv"):
                return "ellipsoid의 real-space v 방향 위치입니다. coherent interference phase를 바꿉니다."
            if name.endswith("_weight"):
                return "ellipsoid complex amplitude의 가중치입니다."
        return tips.get(name, "")

    def defaults(self):
        return {
            "cd_body_x": 1.0,
            "cd_body_y": 2.0,
            "cd_body_z": 1.0,
            "cd_cone_length": 1.8,
            "cd_neck_radius": 0.34,
            "cd_tip_radius": 0.025,
            "cd_z_attach": 0.72,
            "cd_alpha": 1.0,
            "cd_beta": 2.5,
            "cd_rot_x_deg": -18.0,
            "cd_rot_y_deg": 22.0,
            "cd_rot_z_deg": 12.0,
            "cd_field_shape": 120.0,
            "cd_scale": 5000.0,
            "cd_tu": -1800.0,
            "cd_tv": 0.0,
            "cd_weight": 1.0,
            "ell1_b": 0.50,
            "ell1_angle_deg": 60.0,
            "ell1_scale": 900.0,
            "ell1_tu": 1500.0,
            "ell1_tv": 700.0,
            "ell1_weight": 8.0,
            "ell2_b": 0.80,
            "ell2_angle_deg": -20.0,
            "ell2_scale": 650.0,
            "ell2_tu": 2000.0,
            "ell2_tv": -700.0,
            "ell2_weight": 5.0,
        }

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        scroll = ScrollableFrame(main, width=535)
        scroll.pack(side=tk.LEFT, fill=tk.Y)
        controls = scroll.inner

        plot_area = ttk.Frame(main, padding=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls,
            text="Coned Droplet + Two Ellipsoids Parameters",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(4, 10), padx=8)

        for name, value in self.defaults().items():
            self.vars[name] = tk.DoubleVar(value=value)

        cd_body = ttk.LabelFrame(controls, text="Object 1: Coned Droplet - Body", padding=6)
        cd_body.pack(fill=tk.X, padx=8, pady=4)
        self._add(cd_body, "cd_body_x", 0.2, 3.0, 0.1)
        self._add(cd_body, "cd_body_y", 0.2, 3.0, 0.1)
        self._add(cd_body, "cd_body_z", 0.2, 3.0, 0.1)

        cd_tail = ttk.LabelFrame(controls, text="Object 1: Coned Droplet - Cone / Tail", padding=6)
        cd_tail.pack(fill=tk.X, padx=8, pady=4)
        self._add(cd_tail, "cd_cone_length", 0.2, 4.0, 0.1)
        self._add(cd_tail, "cd_neck_radius", 0.05, 1.0, 0.01)
        self._add(cd_tail, "cd_tip_radius", 0.005, 0.2, 0.005)
        self._add(cd_tail, "cd_z_attach", 0.0, 1.0, 0.01)
        self._add(cd_tail, "cd_alpha", 0.5, 3.0, 0.1)
        self._add(cd_tail, "cd_beta", 1.0, 5.0, 0.1)

        cd_rot = ttk.LabelFrame(controls, text="Object 1: Coned Droplet - Rotation / Placement", padding=6)
        cd_rot.pack(fill=tk.X, padx=8, pady=4)
        self._add(cd_rot, "cd_rot_x_deg", -90.0, 90.0, 1.0)
        self._add(cd_rot, "cd_rot_y_deg", -90.0, 90.0, 1.0)
        self._add(cd_rot, "cd_rot_z_deg", -180.0, 180.0, 1.0)
        self._add(cd_rot, "cd_field_shape", 50.0, 180.0, 10.0)
        self._add(cd_rot, "cd_scale", 100.0, 20000.0, 100.0)
        self._add(cd_rot, "cd_tu", -8000.0, 8000.0, 50.0)
        self._add(cd_rot, "cd_tv", -8000.0, 8000.0, 50.0)
        self._add(cd_rot, "cd_weight", 0.0, 50.0, 0.5)

        ell1 = ttk.LabelFrame(controls, text="Object 2: Ellipsoid 1", padding=6)
        ell1.pack(fill=tk.X, padx=8, pady=4)
        self._add_ellipsoid_controls(ell1, "ell1")

        ell2 = ttk.LabelFrame(controls, text="Object 3: Ellipsoid 2", padding=6)
        ell2.pack(fill=tk.X, padx=8, pady=4)
        self._add_ellipsoid_controls(ell2, "ell2")

        ttk.Button(controls, text="Save current result", command=self.save).pack(fill=tk.X, padx=8, pady=(10, 4))
        ttk.Button(controls, text="Reset", command=self.reset).pack(fill=tk.X, padx=8, pady=4)

        self.info = tk.Text(controls, width=55, height=12)
        self.info.pack(fill=tk.X, padx=8, pady=(10, 10))

        self.fig, self.ax = plt.subplots(1, 4, figsize=(16.0, 4.4))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_ellipsoid_controls(self, parent, prefix):
        self._add(parent, f"{prefix}_b", 0.05, 1.0, 0.01)
        self._add(parent, f"{prefix}_angle_deg", -180.0, 180.0, 1.0)
        self._add(parent, f"{prefix}_scale", 50.0, 10000.0, 50.0)
        self._add(parent, f"{prefix}_tu", -8000.0, 8000.0, 50.0)
        self._add(parent, f"{prefix}_tv", -8000.0, 8000.0, 50.0)
        self._add(parent, f"{prefix}_weight", 0.0, 50.0, 0.5)

    def _add(self, parent, name, vmin, vmax, resolution):
        self.slider_meta[name] = {"vmin": float(vmin), "vmax": float(vmax), "step": float(resolution)}
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        frame.columnconfigure(2, weight=1)

        label = ttk.Label(frame, text=name, width=18)
        label.grid(row=0, column=0, sticky="w")
        ToolTip(label, self.tooltip(name))

        ttk.Button(frame, text="◀", width=3, command=lambda key=name: self.step_value(key, -1)).grid(
            row=0, column=1, padx=(2, 2)
        )
        slider = ttk.Scale(
            frame,
            from_=vmin,
            to=vmax,
            orient=tk.HORIZONTAL,
            variable=self.vars[name],
            command=lambda _value, key=name: self.on_slider_change(key),
        )
        slider.grid(row=0, column=2, sticky="ew")
        ttk.Button(frame, text="▶", width=3, command=lambda key=name: self.step_value(key, 1)).grid(
            row=0, column=3, padx=(2, 4)
        )

        entry = ttk.Entry(frame, width=10)
        entry.grid(row=0, column=4, sticky="e")
        entry.insert(0, f"{self.vars[name].get():.6g}")
        self.entries[name] = entry

        entry.bind("<Return>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<FocusOut>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<Up>", lambda event, key=name: self.on_entry_step_key(event, key, 1))
        entry.bind("<Right>", lambda event, key=name: self.on_entry_step_key(event, key, 1))
        entry.bind("<Down>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<Left>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<MouseWheel>", lambda event, key=name: self.on_entry_mousewheel(event, key))
        entry.bind("<Button-4>", lambda event, key=name: self.on_entry_mousewheel_linux(event, key, 1))
        entry.bind("<Button-5>", lambda event, key=name: self.on_entry_mousewheel_linux(event, key, -1))

    def normalize_value(self, name, value, snap=True):
        meta = self.slider_meta[name]
        value = min(max(float(value), meta["vmin"]), meta["vmax"])
        step = meta["step"]
        if snap and step > 0:
            value = meta["vmin"] + round((value - meta["vmin"]) / step) * step
            value = min(max(value, meta["vmin"]), meta["vmax"])
        if step > 0:
            decimals = max(0, int(np.ceil(-np.log10(step))) + 2) if step < 1 else 6
            value = round(value, decimals)
        return value

    def set_value(self, name, value, schedule=True):
        value = self.normalize_value(name, value, snap=True)
        self._programmatic_update.add(name)
        try:
            self.vars[name].set(value)
        finally:
            self._programmatic_update.discard(name)
        self.sync_entry(name)
        if schedule:
            self.schedule()

    def current_entry_or_var_value(self, name):
        try:
            return float(self.entries[name].get())
        except Exception:
            return float(self.vars[name].get())

    def step_value(self, name, direction):
        step = self.slider_meta[name]["step"]
        self.set_value(name, float(self.vars[name].get()) + direction * step)

    def on_entry_step_key(self, event, name, direction):
        step = self.slider_meta[name]["step"]
        self.set_value(name, self.current_entry_or_var_value(name) + direction * step)
        self.entries[name].focus_set()
        self.entries[name].selection_range(0, tk.END)
        return "break"

    def on_entry_mousewheel(self, event, name):
        direction = 1 if event.delta > 0 else -1
        return self.on_entry_step_key(event, name, direction)

    def on_entry_mousewheel_linux(self, event, name, direction):
        return self.on_entry_step_key(event, name, direction)

    def on_slider_change(self, name):
        if name in self._programmatic_update:
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
        self.entries[name].delete(0, tk.END)
        self.entries[name].insert(0, f"{self.vars[name].get():.6g}")

    def schedule(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(700, self.update_plot)

    def params(self):
        p = {k: float(v.get()) for k, v in self.vars.items()}
        p["cd_field_shape"] = int(round(p["cd_field_shape"]))
        p["cd_field_shape"] = max(30, p["cd_field_shape"])
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

        specs = [
            ("cd", result["cd_proj"], result["cd_du"], result["cd_dv"], p["cd_scale"], p["cd_tu"], p["cd_tv"], p["cd_weight"], 0.85),
            ("ell1", result["ell1_proj"], result["ell1_du"], result["ell1_dv"], p["ell1_scale"], p["ell1_tu"], p["ell1_tv"], p["ell1_weight"], 0.65),
            ("ell2", result["ell2_proj"], result["ell2_du"], result["ell2_dv"], p["ell2_scale"], p["ell2_tu"], p["ell2_tv"], p["ell2_weight"], 0.65),
        ]
        extents = []
        for label, proj, du, dv, scale, tu, tv, weight, alpha in specs:
            extent = projection_extent(proj, du, dv, scale, tu, tv)
            extents.append(extent)
            ax.imshow(weight * proj, extent=extent, origin="lower", interpolation="nearest", alpha=alpha)
            ax.text(tu, tv, label.upper(), ha="center", va="center", fontsize=8)

        xmin, xmax, ymin, ymax = combined_real_space_limits(extents)
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
            extents = [
                projection_extent(result["cd_proj"], result["cd_du"], result["cd_dv"], p["cd_scale"], p["cd_tu"], p["cd_tv"]),
                projection_extent(result["ell1_proj"], result["ell1_du"], result["ell1_dv"], p["ell1_scale"], p["ell1_tu"], p["ell1_tv"]),
                projection_extent(result["ell2_proj"], result["ell2_du"], result["ell2_dv"], p["ell2_scale"], p["ell2_tu"], p["ell2_tv"]),
            ]

        cd_width, cd_height, cd_angle = projected_ellipsoid_preview(
            [p["cd_body_x"], p["cd_body_y"], p["cd_body_z"]],
            p["cd_scale"], p["cd_rot_x_deg"], p["cd_rot_y_deg"], p["cd_rot_z_deg"]
        )

        cd_body = Ellipse((p["cd_tu"], p["cd_tv"]), width=cd_width, height=cd_height,
                          angle=cd_angle, fill=False, linewidth=1.5)
        ax.add_patch(cd_body)
        ax.text(p["cd_tu"], p["cd_tv"], "CD", ha="center", va="center")

        for idx, (prefix, label) in enumerate([("ell1", "E1"), ("ell2", "E2")], start=1):
            if extents:
                width = extents[idx][1] - extents[idx][0]
                height = width * p[f"{prefix}_b"]
            else:
                width = 2.0 * p[f"{prefix}_scale"]
                height = 2.0 * p[f"{prefix}_scale"] * p[f"{prefix}_b"]
            patch = Ellipse((p[f"{prefix}_tu"], p[f"{prefix}_tv"]), width=width, height=height,
                            angle=p[f"{prefix}_angle_deg"], fill=False, linewidth=1.5)
            ax.add_patch(patch)
            ax.text(p[f"{prefix}_tu"], p[f"{prefix}_tv"], label, ha="center", va="center")

        if extents:
            xmin, xmax, ymin, ymax = combined_real_space_limits(extents)
        else:
            centers = np.array([[p["cd_tu"], p["cd_tv"]], [p["ell1_tu"], p["ell1_tv"]], [p["ell2_tu"], p["ell2_tv"]]], dtype=float)
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

        for ax, im, title in zip(
            self.ax[1:3],
            [safe_log(I_total), I_masked],
            ["log theoretical: |wCD ACD + w1 A1 + w2 A2|²", "masked theoretical"],
        ):
            ax.clear()
            ax.matshow(im)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self.draw_position_preview(self.ax[3], p)
        self.fig.tight_layout()
        self.canvas.draw_idle()

        q_u = result["q_u"]
        q_v = result["q_v"]
        self.info.delete("1.0", tk.END)
        self.info.insert(
            tk.END,
            f"A_total = {p['cd_weight']}*A_cd + {p['ell1_weight']}*A_ell1 + {p['ell2_weight']}*A_ell2\n\n"
            f"CD position = ({p['cd_tu']:.6g}, {p['cd_tv']:.6g})\n"
            f"E1 position = ({p['ell1_tu']:.6g}, {p['ell1_tv']:.6g})\n"
            f"E2 position = ({p['ell2_tu']:.6g}, {p['ell2_tv']:.6g})\n\n"
            f"q_u = [{q_u.min():.6g}, {q_u.max():.6g}]\n"
            f"q_v = [{q_v.min():.6g}, {q_v.max():.6g}]\n"
        )

    def save(self):
        if self.last_result is None:
            messagebox.showerror("Save error", "No simulation result to save.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dialog = SaveFolderDialog(self.root, f"coned_droplet_two_ellipsoids_sim_{timestamp}")
        d = dialog.result
        if d is None:
            return
        try:
            d.mkdir(parents=True, exist_ok=True)
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
            "Coned Droplet + Two Ellipsoids SAXS simulation parameters",
            "=" * 64,
            "",
            "[Merge equation]",
            "A_total = cd_weight*A_coned_droplet + ell1_weight*A_ellipsoid1 + ell2_weight*A_ellipsoid2",
            "I_total = |A_total|^2",
            "",
            "[Object 1: Coned Droplet]",
            f"body_axes = [{p['cd_body_x']}, {p['cd_body_y']}, {p['cd_body_z']}]",
            f"cone_length = {p['cd_cone_length']}",
            f"neck_radius = {p['cd_neck_radius']}",
            f"tip_radius = {p['cd_tip_radius']}",
            f"z_attach = {p['cd_z_attach']}",
            f"alpha = {p['cd_alpha']}",
            f"beta = {p['cd_beta']}",
            f"rotation_deg = [{p['cd_rot_x_deg']}, {p['cd_rot_y_deg']}, {p['cd_rot_z_deg']}]",
            f"field_shape = {int(round(p['cd_field_shape']))}",
            f"scale = {p['cd_scale']}",
            f"translation_u = {p['cd_tu']}",
            f"translation_v = {p['cd_tv']}",
            f"weight = {p['cd_weight']}",
            "",
            "[Object 2: Ellipsoid 1]",
            f"b = {p['ell1_b']}",
            f"angle_deg = {p['ell1_angle_deg']}",
            f"scale = {p['ell1_scale']}",
            f"translation_u = {p['ell1_tu']}",
            f"translation_v = {p['ell1_tv']}",
            f"weight = {p['ell1_weight']}",
            "",
            "[Object 3: Ellipsoid 2]",
            f"b = {p['ell2_b']}",
            f"angle_deg = {p['ell2_angle_deg']}",
            f"scale = {p['ell2_scale']}",
            f"translation_u = {p['ell2_tu']}",
            f"translation_v = {p['ell2_tv']}",
            f"weight = {p['ell2_weight']}",
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
        self.ax[1].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", transform=self.ax[1].transAxes, wrap=True)
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = ConedDropletTwoEllipsoidsGUI(root)
    root.mainloop()
