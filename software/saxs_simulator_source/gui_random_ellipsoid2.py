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
    def __init__(self, parent, width=510):
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


def make_blob(params):
    centers = [[params[f"c{i}_x"], params[f"c{i}_y"], params[f"c{i}_z"]] for i in range(3)]
    sigmas = [[params[f"s{i}_x"], params[f"s{i}_y"], params[f"s{i}_z"]] for i in range(3)]
    weights = [params[f"w{i}"] for i in range(3)]
    axis = [params["axis_x"], params["axis_y"], params["axis_z"]]

    if np.linalg.norm(axis) < 1e-12:
        raise ValueError("blob rotation axis cannot be zero vector.")

    bvh = build_bvh(
        "blob",
        kernel_centers=centers,
        kernel_sigmas=sigmas,
        kernel_weights=weights,
        threshold=params["threshold"],
        rotations=[(axis, np.deg2rad(params["angle_deg"]))],
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
        scale_factor=params["blob_scale"],
        translation_u=params["blob_tu"],
        translation_v=params["blob_tv"],
    )
    return proj, du, dv, A, q_u, q_v, meta_proj, meta


def make_ellipsoid(prefix, params):
    proj, du, dv = proj_ellipsoid(
        b=params[f"{prefix}_b"],
        angle=np.deg2rad(params[f"{prefix}_angle_deg"]),
        img_shape=IMG_SHAPE,
    )
    A, q_u, q_v, meta = get_amplitude(
        proj=proj,
        du=du,
        dv=dv,
        scale_factor=params[f"{prefix}_scale"],
        translation_u=params[f"{prefix}_tu"],
        translation_v=params[f"{prefix}_tv"],
    )
    return proj, du, dv, A, q_u, q_v, meta


def run_simulation(params):
    blob_proj, blob_du, blob_dv, A_blob, q_u, q_v, meta_proj_blob, meta_blob = make_blob(params)
    ell1_proj, ell1_du, ell1_dv, A_ell1, _, _, meta_ell1 = make_ellipsoid("ell1", params)
    ell2_proj, ell2_du, ell2_dv, A_ell2, _, _, meta_ell2 = make_ellipsoid("ell2", params)

    A_total = (
        params["blob_weight"] * A_blob
        + params["ell1_weight"] * A_ell1
        + params["ell2_weight"] * A_ell2
    )
    I_total = np.abs(A_total) ** 2

    return {
        "blob_proj": blob_proj,
        "ell1_proj": ell1_proj,
        "ell2_proj": ell2_proj,
        "blob_du": blob_du,
        "blob_dv": blob_dv,
        "ell1_du": ell1_du,
        "ell1_dv": ell1_dv,
        "ell2_du": ell2_du,
        "ell2_dv": ell2_dv,
        "A_blob": A_blob,
        "A_ell1": A_ell1,
        "A_ell2": A_ell2,
        "A_total": A_total,
        "I_total": I_total,
        "q_u": q_u,
        "q_v": q_v,
        "meta_proj_blob": meta_proj_blob,
        "meta_blob": meta_blob,
        "meta_ell1": meta_ell1,
        "meta_ell2": meta_ell2,
    }

class RandomBlobTwoEllipsoidsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Random Blob + Two Ellipsoids SAXS Simulator")
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
        values = {}
        centers = [[0.0, 0.0, 0.0], [0.3, 0.1, 0.2], [-0.2, 0.2, -0.1]]
        sigmas = [[0.35, 0.30, 0.25], [0.20, 0.18, 0.70], [0.22, 0.50, 0.20]]
        weights = [1.0, 0.8, 2.7]
        for i in range(3):
            values[f"c{i}_x"] = centers[i][0]
            values[f"c{i}_y"] = centers[i][1]
            values[f"c{i}_z"] = centers[i][2]
            values[f"s{i}_x"] = sigmas[i][0]
            values[f"s{i}_y"] = sigmas[i][1]
            values[f"s{i}_z"] = sigmas[i][2]
            values[f"w{i}"] = weights[i]
        values.update(
            {
                "threshold": 0.55,
                "axis_x": 0.0,
                "axis_y": 0.0,
                "axis_z": 1.0,
                "angle_deg": np.rad2deg(0.5),
                "field_shape": 100.0,
                "blob_scale": 5000.0,
                "blob_tu": -1500.0,
                "blob_tv": 0.0,
                "blob_weight": 1.0,
                "ell1_b": 0.5,
                "ell1_angle_deg": 35.0,
                "ell1_scale": 1200.0,
                "ell1_tu": 1500.0,
                "ell1_tv": 500.0,
                "ell1_weight": 3.0,
                "ell2_b": 0.75,
                "ell2_angle_deg": -35.0,
                "ell2_scale": 900.0,
                "ell2_tu": 2500.0,
                "ell2_tv": -700.0,
                "ell2_weight": 2.0,
            }
        )
        return values

    def tooltip(self, name):
        if name.startswith("c") and len(name) == 4:
            return "Gaussian kernel 중심 좌표입니다. blob의 비대칭성과 elongation을 조절합니다."
        if name.startswith("s") and len(name) == 4:
            return "Gaussian kernel의 sigma입니다. 값이 클수록 해당 방향으로 더 퍼집니다."
        if name.startswith("w") and len(name) == 2:
            return "Gaussian kernel의 weight입니다. 전체 random blob shape에 대한 기여도를 조절합니다."
        tips = {
            "threshold": "Gaussian mixture field에서 object surface를 자르는 cutoff입니다.",
            "axis_x": "random blob 3D 회전축의 x 성분입니다.",
            "axis_y": "random blob 3D 회전축의 y 성분입니다.",
            "axis_z": "random blob 3D 회전축의 z 성분입니다.",
            "angle_deg": "random blob을 지정한 회전축 기준으로 회전시키는 각도입니다.",
            "field_shape": "random blob mesh 해상도입니다. 클수록 정교하지만 계산이 느려집니다.",
            "blob_scale": "random blob의 실공간 크기 scale입니다.",
            "blob_tu": "random blob의 real-space u 방향 translation입니다.",
            "blob_tv": "random blob의 real-space v 방향 translation입니다.",
            "blob_weight": "random blob complex amplitude의 가중치입니다.",
            "ell1_b": "ellipsoid 1의 단축 비율입니다. b=1이면 원형 projection에 가깝습니다.",
            "ell1_angle_deg": "ellipsoid 1 projected density의 detector plane 내 회전 각도입니다.",
            "ell1_scale": "ellipsoid 1의 실공간 크기 scale입니다.",
            "ell1_tu": "ellipsoid 1의 real-space u 방향 translation입니다.",
            "ell1_tv": "ellipsoid 1의 real-space v 방향 translation입니다.",
            "ell1_weight": "ellipsoid 1 complex amplitude의 가중치입니다.",
            "ell2_b": "ellipsoid 2의 단축 비율입니다. b=1이면 원형 projection에 가깝습니다.",
            "ell2_angle_deg": "ellipsoid 2 projected density의 detector plane 내 회전 각도입니다.",
            "ell2_scale": "ellipsoid 2의 실공간 크기 scale입니다.",
            "ell2_tu": "ellipsoid 2의 real-space u 방향 translation입니다.",
            "ell2_tv": "ellipsoid 2의 real-space v 방향 translation입니다.",
            "ell2_weight": "ellipsoid 2 complex amplitude의 가중치입니다.",
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
            text="Random Blob + Two Ellipsoids Parameters",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(4, 10), padx=8)

        for name, value in self.defaults().items():
            self.vars[name] = tk.DoubleVar(value=value)

        for i in range(3):
            k_frame = ttk.LabelFrame(controls, text=f"Object 1: Random blob kernel {i}", padding=6)
            k_frame.pack(fill=tk.X, padx=8, pady=4)
            self._add(k_frame, f"c{i}_x", -1.5, 1.5, 0.01)
            self._add(k_frame, f"c{i}_y", -1.5, 1.5, 0.01)
            self._add(k_frame, f"c{i}_z", -1.5, 1.5, 0.01)
            self._add(k_frame, f"s{i}_x", 0.05, 1.2, 0.01)
            self._add(k_frame, f"s{i}_y", 0.05, 1.2, 0.01)
            self._add(k_frame, f"s{i}_z", 0.05, 1.2, 0.01)
            self._add(k_frame, f"w{i}", 0.1, 4.0, 0.05)

        shape_frame = ttk.LabelFrame(controls, text="Object 1: Shape threshold", padding=6)
        shape_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(shape_frame, "threshold", 0.1, 1.5, 0.01)

        blob_rot_frame = ttk.LabelFrame(controls, text="Object 1: 3D rotation", padding=6)
        blob_rot_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(blob_rot_frame, "axis_x", -1.0, 1.0, 0.1)
        self._add(blob_rot_frame, "axis_y", -1.0, 1.0, 0.1)
        self._add(blob_rot_frame, "axis_z", -1.0, 1.0, 0.1)
        self._add(blob_rot_frame, "angle_deg", -180.0, 180.0, 1.0)

        blob_scat_frame = ttk.LabelFrame(controls, text="Object 1: scattering / placement", padding=6)
        blob_scat_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(blob_scat_frame, "field_shape", 50.0, 180.0, 10.0)
        self._add(blob_scat_frame, "blob_scale", 100.0, 20000.0, 100.0)
        self._add(blob_scat_frame, "blob_tu", -8000.0, 8000.0, 50.0)
        self._add(blob_scat_frame, "blob_tv", -8000.0, 8000.0, 50.0)
        self._add(blob_scat_frame, "blob_weight", 0.0, 50.0, 0.5)

        ell1_frame = ttk.LabelFrame(controls, text="Object 2: Ellipsoid 1", padding=6)
        ell1_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(ell1_frame, "ell1_b", 0.05, 1.0, 0.01)
        self._add(ell1_frame, "ell1_angle_deg", -180.0, 180.0, 1.0)
        self._add(ell1_frame, "ell1_scale", 50.0, 10000.0, 50.0)
        self._add(ell1_frame, "ell1_tu", -8000.0, 8000.0, 50.0)
        self._add(ell1_frame, "ell1_tv", -8000.0, 8000.0, 50.0)
        self._add(ell1_frame, "ell1_weight", 0.0, 50.0, 0.5)

        ell2_frame = ttk.LabelFrame(controls, text="Object 3: Ellipsoid 2", padding=6)
        ell2_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(ell2_frame, "ell2_b", 0.05, 1.0, 0.01)
        self._add(ell2_frame, "ell2_angle_deg", -180.0, 180.0, 1.0)
        self._add(ell2_frame, "ell2_scale", 50.0, 10000.0, 50.0)
        self._add(ell2_frame, "ell2_tu", -8000.0, 8000.0, 50.0)
        self._add(ell2_frame, "ell2_tv", -8000.0, 8000.0, 50.0)
        self._add(ell2_frame, "ell2_weight", 0.0, 50.0, 0.5)

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
        self.update_job = self.root.after(700, self.update_plot)

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

        extent_blob = projection_extent(
            result["blob_proj"], result["blob_du"], result["blob_dv"],
            p["blob_scale"], p["blob_tu"], p["blob_tv"]
        )
        extent_ell1 = projection_extent(
            result["ell1_proj"], result["ell1_du"], result["ell1_dv"],
            p["ell1_scale"], p["ell1_tu"], p["ell1_tv"]
        )
        extent_ell2 = projection_extent(
            result["ell2_proj"], result["ell2_du"], result["ell2_dv"],
            p["ell2_scale"], p["ell2_tu"], p["ell2_tv"]
        )

        ax.imshow(p["blob_weight"] * result["blob_proj"], extent=extent_blob, origin="lower", interpolation="nearest")
        ax.imshow(p["ell1_weight"] * result["ell1_proj"], extent=extent_ell1, origin="lower", interpolation="nearest", alpha=0.65)
        ax.imshow(p["ell2_weight"] * result["ell2_proj"], extent=extent_ell2, origin="lower", interpolation="nearest", alpha=0.65)

        xmin, xmax, ymin, ymax = combined_real_space_limits([extent_blob, extent_ell1, extent_ell2])
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
                projection_extent(result["blob_proj"], result["blob_du"], result["blob_dv"], p["blob_scale"], p["blob_tu"], p["blob_tv"]),
                projection_extent(result["ell1_proj"], result["ell1_du"], result["ell1_dv"], p["ell1_scale"], p["ell1_tu"], p["ell1_tv"]),
                projection_extent(result["ell2_proj"], result["ell2_du"], result["ell2_dv"], p["ell2_scale"], p["ell2_tu"], p["ell2_tv"]),
            ]

        if extents:
            blob_size = max(extents[0][1] - extents[0][0], extents[0][3] - extents[0][2])
        else:
            blob_size = 2.0 * p["blob_scale"]

        blob_patch = Ellipse((p["blob_tu"], p["blob_tv"]), width=blob_size, height=blob_size,
                             fill=False, linewidth=1.5)
        ax.add_patch(blob_patch)
        ax.text(p["blob_tu"], p["blob_tv"], "Blob", ha="center", va="center")

        for idx, (prefix, label) in enumerate([("ell1", "Ell1"), ("ell2", "Ell2")], start=1):
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
            centers = np.array([[p["blob_tu"], p["blob_tv"]], [p["ell1_tu"], p["ell1_tv"]], [p["ell2_tu"], p["ell2_tv"]]], dtype=float)
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
        titles = ["log theoretical: |w_blob A_blob + w_ell1 A_ell1 + w_ell2 A_ell2|²", "masked theoretical"]
        for ax, im, title in zip(self.ax[1:3], images, titles):
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
        d1u = p["ell1_tu"] - p["blob_tu"]
        d1v = p["ell1_tv"] - p["blob_tv"]
        d2u = p["ell2_tu"] - p["blob_tu"]
        d2v = p["ell2_tv"] - p["blob_tv"]
        self.info.delete("1.0", tk.END)
        self.info.insert(
            tk.END,
            f"A_total = {p['blob_weight']}*A_blob + {p['ell1_weight']}*A_ell1 + {p['ell2_weight']}*A_ell2\n\n"
            f"blob position = ({p['blob_tu']:.6g}, {p['blob_tv']:.6g})\n"
            f"ell1 position = ({p['ell1_tu']:.6g}, {p['ell1_tv']:.6g})\n"
            f"ell2 position = ({p['ell2_tu']:.6g}, {p['ell2_tv']:.6g})\n"
            f"blob→ell1 Δu, Δv = ({d1u:.6g}, {d1v:.6g})\n"
            f"blob→ell2 Δu, Δv = ({d2u:.6g}, {d2v:.6g})\n\n"
            f"field_shape = {p['field_shape']}\n"
            f"q_u = [{q_u.min():.6g}, {q_u.max():.6g}]\n"
            f"q_v = [{q_v.min():.6g}, {q_v.max():.6g}]\n"
        )

    def save(self):
        if self.last_result is None:
            messagebox.showerror("Save error", "No simulation result to save.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dialog = SaveFolderDialog(self.root, f"random_blob_plus_two_ellipsoids_sim_{timestamp}")
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
            "Random Blob + Two Ellipsoids SAXS simulation parameters",
            "=" * 64,
            "",
            "[Merge equation]",
            "A_total = blob_weight * A_blob + ell1_weight * A_ellipsoid1 + ell2_weight * A_ellipsoid2",
            "I_total = |A_total|^2",
            "",
            "[Object 1: Random blob]",
            f"threshold = {p['threshold']}",
            f"rotation_axis = [{p['axis_x']}, {p['axis_y']}, {p['axis_z']}]",
            f"angle_deg = {p['angle_deg']}",
            f"field_shape = {p['field_shape']}",
            f"blob_scale = {p['blob_scale']}",
            f"blob_translation_u = {p['blob_tu']}",
            f"blob_translation_v = {p['blob_tv']}",
            f"blob_weight = {p['blob_weight']}",
            "",
        ]
        for i in range(3):
            lines.extend([
                f"[Kernel {i}]",
                f"center = [{p[f'c{i}_x']}, {p[f'c{i}_y']}, {p[f'c{i}_z']}]",
                f"sigma = [{p[f's{i}_x']}, {p[f's{i}_y']}, {p[f's{i}_z']}]",
                f"weight = {p[f'w{i}']}",
                "",
            ])
        for prefix, title in [("ell1", "Object 2: Ellipsoid 1"), ("ell2", "Object 3: Ellipsoid 2")]:
            lines.extend([
                f"[{title}]",
                f"{prefix}_b = {p[f'{prefix}_b']}",
                f"{prefix}_angle_deg = {p[f'{prefix}_angle_deg']}",
                f"{prefix}_scale = {p[f'{prefix}_scale']}",
                f"{prefix}_translation_u = {p[f'{prefix}_tu']}",
                f"{prefix}_translation_v = {p[f'{prefix}_tv']}",
                f"{prefix}_weight = {p[f'{prefix}_weight']}",
                "",
            ])
        lines.extend([
            "[Files]",
            f"poni_path = {PONI_PATH}",
            f"mask_path = {MASK_PATH}",
            "",
            "[q axes]",
            f"q_u_min = {float(q_u.min())}",
            f"q_u_max = {float(q_u.max())}",
            f"q_v_min = {float(q_v.min())}",
            f"q_v_max = {float(q_v.max())}",
        ])
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
    app = RandomBlobTwoEllipsoidsGUI(root)
    root.mainloop()
