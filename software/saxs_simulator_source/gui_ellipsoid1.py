from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from saxs_module import (
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
            wraplength=360,
        )
        label.pack()

    def hide(self, event=None):
        self.unschedule()
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


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

        self.parent_entry = ttk.Entry(parent_row, textvariable=self.parent_dir, width=58)
        self.parent_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(parent_row, text="Browse", command=self.browse).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(outer, text="Result folder name").grid(row=2, column=0, sticky="w")
        folder_entry = ttk.Entry(outer, textvariable=self.folder_name, width=58)
        folder_entry.grid(row=3, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(outer, text="Full save path").grid(row=4, column=0, sticky="w")
        full_entry = ttk.Entry(outer, textvariable=self.full_path, width=58, state="readonly")
        full_entry.grid(row=5, column=0, sticky="ew", pady=(2, 10))

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
        if not Path(parent).exists() or not Path(parent).is_dir():
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



def proj_ellipsoid_single_physical(b: float, angle: float, img_shape=IMG_SHAPE, object_scale: float = 1.0):
    """Projected ellipsoid density for the single-ellipsoid GUI.

    Array/axis convention is explicit:
      - axis 1 / columns = real-space u, displayed horizontally
      - axis 0 / rows    = real-space v, displayed vertically
      - angle is the in-plane angle of the real-space major axis from +u toward +v

    For 0 < b <= 1, the unit ellipsoid has major semi-axis object_scale and
    minor semi-axis b * object_scale in the detector plane.  Thus b controls
    the aspect ratio, while object_scale controls the actual object size within
    the fixed calculation box.  The returned density is proportional to the
    projected thickness sqrt(R^2 - u_major^2 - (v_minor/b)^2).
    """
    if b <= 0 or b > 1:
        raise ValueError(f"the value of b should be 0 < b <= 1, but it was {b}")
    object_scale = float(object_scale)
    if not np.isfinite(object_scale) or object_scale <= 0:
        raise ValueError(f"object_scale must be positive and finite, but it was {object_scale}")

    H, W = img_shape
    u = np.linspace(-1.0, 1.0, W, dtype=np.float64).reshape(1, W)
    v = np.linspace(-1.0, 1.0, H, dtype=np.float64).reshape(H, 1)

    c, s = np.cos(angle), np.sin(angle)

    # Rotate the coordinate frame into the ellipsoid's principal-axis frame.
    # u_major is along the real-space major axis; v_minor is perpendicular to it.
    u_major = c * u + s * v
    v_minor = -s * u + c * v

    r2 = (u_major / object_scale) ** 2 + (v_minor / (float(b) * object_scale)) ** 2
    proj = object_scale * np.sqrt(np.clip(1.0 - r2, 0.0, None))

    du = 2.0 / float(W)
    dv = 2.0 / float(H)
    return proj, du, dv


def image_extent_from_axes(x_axis, y_axis):
    """Return a robust imshow extent from monotonic 1D axes."""
    x = np.asarray(x_axis, dtype=np.float64)
    y = np.asarray(y_axis, dtype=np.float64)
    return (float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y)))


def safe_log(x):
    return np.log(np.asarray(x, dtype=np.float64) + EPS)


def run_simulation(b, angle_deg, scale_factor, object_scale=1.0):
    angle_rad = np.deg2rad(angle_deg)

    proj, du, dv = proj_ellipsoid_single_physical(
        b=b,
        angle=angle_rad,
        img_shape=IMG_SHAPE,
        object_scale=object_scale,
    )

    A, q_u, q_v, meta = complex_amplitude_from_projected_density(
        projected_density=proj,
        du=du,
        dv=dv,
        scale_factor=scale_factor,
        translation_u=0.0,
        translation_v=0.0,
        poni_path=PONI_PATH,
        detector_image_shape=IMG_SHAPE,
        q_unit="A^-1",
        center_mode="beam",
        row_origin="upper",
    )

    I_total = np.abs(A) ** 2
    return proj, du, dv, I_total, q_u, q_v, meta


class EllipsoidSAXSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sphere / Ellipsoid SAXS Simulator")
        self.root.geometry("1450x720")

        self.update_job = None
        self.last_result = None

        self.mask = np.load(MASK_PATH)
        if self.mask.shape != IMG_SHAPE:
            self.mask = self.mask.reshape(IMG_SHAPE)

        self.vars = {}
        self.entries = {}
        self.sliders = {}
        self.control_specs = {}
        self._slider_updating = set()

        self._build_ui()
        self.update_plot()

    def get_tooltip_text(self, name):
        texts = {
            "b": (
                "ellipsoid의 단축 비율입니다.\n"
                "b = 1이면 원형/sphere projection에 가깝고,\n"
                "b가 작아질수록 더 납작한 ellipsoid projected density가 됩니다."
            ),
            "angle_deg": (
                "detector plane 안에서 ellipsoid가 회전한 각도입니다.\n"
                "3D out-of-plane tilt가 아니라 2D projected image의 in-plane rotation입니다.\n"
                "단위는 degree입니다."
            ),
            "scale_factor": (
                "실공간 object의 크기 스케일입니다.\n"
                "값이 커지면 reciprocal-space scattering pattern이 중심부로 더 좁게 모이고,\n"
                "값이 작아지면 패턴이 더 넓게 퍼집니다."
            ),
        }
        return texts.get(name, "")

    def defaults(self):
        return {
            "b": 0.2,
            "angle_deg": float(np.rad2deg(2 / 5)),
            "object_scale": 0.7,
            "scale_factor": 5000.0,
        }

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        plot_area = ttk.Frame(main, padding=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls,
            text="Ellipsoid parameters",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        for name, value in self.defaults().items():
            self.vars[name] = tk.DoubleVar(value=value)

        shape_frame = ttk.LabelFrame(controls, text="Shape", padding=6)
        shape_frame.pack(fill=tk.X, pady=5)
        self._add_control(shape_frame, "b", 0.05, 1.0, 0.01)
        self._add_control(shape_frame, "angle_deg", 0.0, 180.0, 1.0)
        self._add_control(shape_frame, "object_scale", 0.1, 1.0, 0.01)

        scat_frame = ttk.LabelFrame(controls, text="Scattering", padding=6)
        scat_frame.pack(fill=tk.X, pady=5)
        self._add_control(scat_frame, "scale_factor", 100.0, 20000.0, 100.0)

        ttk.Button(
            controls,
            text="Save current result",
            command=self.save_current_result,
        ).pack(fill=tk.X, pady=(10, 4))

        ttk.Button(
            controls,
            text="Reset",
            command=self.reset,
        ).pack(fill=tk.X, pady=4)

        self.info = tk.Text(controls, width=42, height=13)
        self.info.pack(fill=tk.X, pady=(12, 0))

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_control(self, parent, name, vmin, vmax, step):
        self.control_specs[name] = {
            "min": float(vmin),
            "max": float(vmax),
            "step": float(step),
        }

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        frame.columnconfigure(2, weight=1)

        label_widget = ttk.Label(frame, text=name, width=13)
        label_widget.grid(row=0, column=0, sticky="w")
        ToolTip(label_widget, self.get_tooltip_text(name))

        left_btn = ttk.Button(
            frame,
            text="◀",
            width=3,
            command=lambda key=name: self.step_value(key, -1),
        )
        left_btn.grid(row=0, column=1, padx=(2, 2))

        slider = ttk.Scale(
            frame,
            variable=self.vars[name],
            from_=vmin,
            to=vmax,
            orient=tk.HORIZONTAL,
            command=lambda _value, key=name: self.on_slider_change(key),
        )
        slider.grid(row=0, column=2, sticky="ew")
        self.sliders[name] = slider

        right_btn = ttk.Button(
            frame,
            text="▶",
            width=3,
            command=lambda key=name: self.step_value(key, +1),
        )
        right_btn.grid(row=0, column=3, padx=(2, 4))

        entry = ttk.Entry(frame, width=10)
        entry.insert(0, f"{self.vars[name].get():.6g}")
        entry.grid(row=0, column=4, sticky="e")
        self.entries[name] = entry

        entry.bind("<Return>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<FocusOut>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<Up>", lambda event, key=name: self.on_entry_step_key(event, key, +1))
        entry.bind("<Right>", lambda event, key=name: self.on_entry_step_key(event, key, +1))
        entry.bind("<Down>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<Left>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<MouseWheel>", lambda event, key=name: self.on_entry_mousewheel(event, key))
        entry.bind("<Button-4>", lambda event, key=name: self.on_entry_step_key(event, key, +1))
        entry.bind("<Button-5>", lambda event, key=name: self.on_entry_step_key(event, key, -1))

    def normalize_value(self, name, value, snap_to_step=True):
        spec = self.control_specs[name]
        vmin = spec["min"]
        vmax = spec["max"]
        step = spec["step"]

        value = min(max(float(value), vmin), vmax)

        if snap_to_step and step > 0:
            n_step = round((value - vmin) / step)
            value = vmin + n_step * step
            value = min(max(value, vmin), vmax)

        if step > 0:
            decimals = max(0, int(np.ceil(-np.log10(step))) + 2) if step < 1 else 6
            value = round(value, decimals)

        return value

    def set_parameter_value(self, name, value, snap_to_step=True, schedule_update=True, select_entry=False):
        value = self.normalize_value(name, value, snap_to_step=snap_to_step)

        self._slider_updating.add(name)
        try:
            self.vars[name].set(value)
        finally:
            self._slider_updating.discard(name)

        self._sync_entry(name)
        if select_entry and name in self.entries:
            entry = self.entries[name]
            entry.focus_set()
            entry.selection_range(0, tk.END)
        if schedule_update:
            self.schedule_update()

    def get_entry_or_var_value(self, name):
        try:
            return float(self.entries[name].get())
        except Exception:
            return float(self.vars[name].get())

    def step_value(self, name, direction, select_entry=False):
        spec = self.control_specs[name]
        base = self.get_entry_or_var_value(name)
        self.set_parameter_value(
            name,
            base + float(direction) * spec["step"],
            snap_to_step=True,
            schedule_update=True,
            select_entry=select_entry,
        )

    def on_entry_step_key(self, event, name, direction):
        self.step_value(name, direction, select_entry=True)
        return "break"

    def on_entry_mousewheel(self, event, name):
        direction = +1 if event.delta > 0 else -1
        self.step_value(name, direction, select_entry=True)
        return "break"

    def on_slider_change(self, name):
        if name in self._slider_updating:
            return
        value = self.normalize_value(name, self.vars[name].get(), snap_to_step=True)
        self._slider_updating.add(name)
        try:
            self.vars[name].set(value)
        finally:
            self._slider_updating.discard(name)
        self._sync_entry(name)
        self.schedule_update()

    def on_entry_change(self, name):
        try:
            value = float(self.entries[name].get())
        except ValueError:
            messagebox.showerror("Input error", f"{name} must be numeric.")
            self._sync_entry(name)
            return

        self.set_parameter_value(name, value, snap_to_step=True, schedule_update=True)

    def schedule_update(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(250, self.update_plot)

    def _sync_entry(self, name):
        entry = self.entries[name]
        entry.delete(0, tk.END)
        entry.insert(0, f"{self.vars[name].get():.6g}")

    def reset(self):
        for name, value in self.defaults().items():
            self.set_parameter_value(name, value, snap_to_step=True, schedule_update=False)
        self.update_plot()

    def update_plot(self):
        self.update_job = None

        try:
            b = float(self.vars["b"].get())
            angle_deg = float(self.vars["angle_deg"].get())
            object_scale = float(self.vars["object_scale"].get())
            scale_factor = float(self.vars["scale_factor"].get())

            proj, du, dv, I_total, q_u, q_v, meta = run_simulation(
                b=b,
                angle_deg=angle_deg,
                object_scale=object_scale,
                scale_factor=scale_factor,
            )
        except Exception as e:
            self._show_error(e)
            return

        I_masked = I_total * self.mask

        self.last_result = {
            "parameters": {
                "shape": "ellipsoid",
                "b": b,
                "angle_deg": angle_deg,
                "angle_rad": float(np.deg2rad(angle_deg)),
                "object_scale": object_scale,
                "scale_factor": scale_factor,
                "translation_u": 0.0,
                "translation_v": 0.0,
                "poni_path": PONI_PATH,
                "mask_path": MASK_PATH,
                "image_shape": IMG_SHAPE,
            },
            "proj": proj,
            "du": du,
            "dv": dv,
            "I_total": I_total,
            "I_masked": I_masked,
            "q_u": q_u,
            "q_v": q_v,
            "meta": meta,
        }

        # q_v returned from the PONI-based detector convention is ordered from
        # high to low when row_origin="upper" is used.  For a physical q_u/q_v
        # plot with q_v increasing upward (origin="lower"), the q-space images
        # must be vertically flipped.  Otherwise the reciprocal-space pattern
        # appears to rotate in the same sense as the real-space ellipsoid.
        I_log_display = np.flipud(safe_log(I_total))
        I_masked_display = np.flipud(I_masked)

        images = [proj, I_log_display, I_masked_display]
        titles = ["projected density", "log theoretical", "masked theoretical"]
        extents = [
            (-1.0 * scale_factor, 1.0 * scale_factor, -1.0 * scale_factor, 1.0 * scale_factor),
            image_extent_from_axes(q_u, q_v),
            image_extent_from_axes(q_u, q_v),
        ]
        xlabels = ["real-space u", "q_u", "q_u"]
        ylabels = ["real-space v", "q_v", "q_v"]

        for ax, img, title, extent, xlabel, ylabel in zip(self.axes, images, titles, extents, xlabels, ylabels):
            ax.clear()
            ax.imshow(img, extent=extent, origin="lower", interpolation="nearest", aspect="equal")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks([])
            ax.set_yticks([])

        self.fig.tight_layout()
        self.canvas.draw_idle()

        self._update_info(du, dv, q_u, q_v)

    def _update_info(self, du, dv, q_u, q_v):
        self.info.delete("1.0", tk.END)
        self.info.insert(
            tk.END,
            f"b = {self.vars['b'].get():.6g}\n"
            f"angle = {self.vars['angle_deg'].get():.6g} deg\n"
            f"scale_factor = {self.vars['scale_factor'].get():.6g}\n\n"
            f"du = {du:.6g}\n"
            f"dv = {dv:.6g}\n"
            f"q_u = [{q_u.min():.6g}, {q_u.max():.6g}]\n"
            f"q_v = [{q_v.min():.6g}, {q_v.max():.6g}]\n"
        )

    def save_current_result(self):
        if self.last_result is None:
            messagebox.showerror("Save error", "No simulation result to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dialog = SaveFolderDialog(self.root, f"ellipsoid_sim_{timestamp}")
        save_dir = dialog.result
        if save_dir is None:
            return

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Save error", f"Could not create save folder:\n{save_dir}\n\n{e}")
            return

        try:
            proj = self.last_result["proj"]
            I_total = self.last_result["I_total"]
            I_masked = self.last_result["I_masked"]

            q_u = self.last_result["q_u"]
            q_v = self.last_result["q_v"]
            scale_factor = self.last_result["parameters"]["scale_factor"]
            self._save_image(
                save_dir / "01_projected_density.png",
                proj,
                title="projected density",
                extent=(-1.0 * scale_factor, 1.0 * scale_factor, -1.0 * scale_factor, 1.0 * scale_factor),
                xlabel="real-space u",
                ylabel="real-space v",
            )
            self._save_image(
                save_dir / "02_log_theoretical.png",
                safe_log(I_total),
                title="log theoretical",
                extent=image_extent_from_axes(q_u, q_v),
                xlabel="q_u",
                ylabel="q_v",
            )
            self._save_image(
                save_dir / "03_masked_theoretical.png",
                I_masked,
                title="masked theoretical",
                extent=image_extent_from_axes(q_u, q_v),
                xlabel="q_u",
                ylabel="q_v",
            )
            self._save_parameters_txt(save_dir / "parameters.txt")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save result:\n{save_dir}\n\n{e}")
            return

        messagebox.showinfo("Saved", f"Saved result to:\n{save_dir}")

    def _save_image(self, path, img, title="", extent=None, xlabel="", ylabel=""):
        fig, ax = plt.subplots(figsize=(6, 6))
        if extent is None:
            ax.imshow(img, origin="lower", interpolation="nearest", aspect="equal")
        else:
            ax.imshow(img, extent=extent, origin="lower", interpolation="nearest", aspect="equal")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _save_parameters_txt(self, path):
        result = self.last_result
        p = result["parameters"]

        lines = [
            "Ellipsoid SAXS simulation parameters",
            "=" * 40,
            "",
            "[Shape]",
            f"shape = {p['shape']}",
            f"b = {p['b']}",
            f"angle_deg = {p['angle_deg']}",
            f"angle_rad = {p['angle_rad']}",
            f"object_scale = {p.get('object_scale', 1.0)}",
            "",
            "[Scattering]",
            f"scale_factor = {p['scale_factor']}",
            f"translation_u = {p['translation_u']}",
            f"translation_v = {p['translation_v']}",
            "",
            "[Files]",
            f"poni_path = {p['poni_path']}",
            f"mask_path = {p['mask_path']}",
            "",
            "[Grid]",
            f"image_shape = {p['image_shape']}",
            f"du = {result['du']}",
            f"dv = {result['dv']}",
            "",
            "[q axes]",
            f"q_u_min = {float(result['q_u'].min())}",
            f"q_u_max = {float(result['q_u'].max())}",
            f"q_v_min = {float(result['q_v'].min())}",
            f"q_v_max = {float(result['q_v'].max())}",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _show_error(self, e):
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        self.axes[1].text(
            0.5,
            0.5,
            f"Error:\n{e}",
            ha="center",
            va="center",
            transform=self.axes[1].transAxes,
            wrap=True,
        )
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = EllipsoidSAXSGUI(root)
    root.mainloop()
