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
    build_bvh,
    project_density,
    project_density_scaled,
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
            wraplength=350,
        )
        label.pack()

    def hide(self, event=None):
        self.unschedule()
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


def safe_log(x):
    return np.log(np.asarray(x, dtype=np.float64) + EPS)


def run_simulation(params):
    axis = np.array(
        [
            params["axis_x"],
            params["axis_y"],
            params["axis_z"],
        ],
        dtype=np.float64,
    )

    if np.linalg.norm(axis) < 1e-12:
        raise ValueError("rotation axis cannot be zero vector.")

    bvh = build_bvh(
        "polyhedron",
        name=params["polyhedron_name"],
        radius=params["radius"],
        rotations=[
            (axis.tolist(), np.deg2rad(params["angle_deg"])),
        ],
    )

    proj, du, dv, meta_proj = project_density_scaled(
        bvh,
        ray_vector=(0.0, 0.0, 1.0),
        roll_rad=0.0,
        image_shape=IMG_SHAPE,
        object_scale=params["object_scale"],
        out_dtype="float32",
    )

    A, q_u, q_v, meta = complex_amplitude_from_projected_density(
        projected_density=proj,
        du=du,
        dv=dv,
        scale_factor=params["scale_factor"],
        translation_u=0.0,
        translation_v=0.0,
        poni_path=PONI_PATH,
        detector_image_shape=IMG_SHAPE,
        q_unit="A^-1",
        center_mode="beam",
        row_origin="upper",
    )

    I_total = np.abs(A) ** 2

    return proj, du, dv, I_total, q_u, q_v, meta_proj, meta


class PolyhedronGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Polyhedron SAXS Simulator")
        self.root.geometry("1450x720")

        self.update_job = None
        self.last_result = None

        self.mask = np.load(MASK_PATH)
        if self.mask.shape != IMG_SHAPE:
            self.mask = self.mask.reshape(IMG_SHAPE)

        self.sliders = {}
        self.control_specs = {}
        self._slider_updating = set()

        self._build_ui()
        self.update_plot()

    def get_tooltip_text(self, name):
        texts = {
            "polyhedron_name": (
                "생성할 정다면체 종류입니다.\n"
                "tetra: 정사면체, cube: 정육면체, octa: 정팔면체,\n"
                "dodeca: 정십이면체, icosa: 정이십면체입니다."
            ),
            "radius": (
                "정다면체 vertex들이 정규화된 구 안에 들어가도록 정한 전체 크기 스케일입니다.\n"
                "값이 커지면 real-space object가 커지고, scattering pattern은 q-space에서 더 좁아집니다."
            ),
            "axis_x": (
                "3D 회전축의 x 성분입니다.\n"
                "axis_x, axis_y, axis_z가 함께 하나의 회전축 벡터를 이룹니다."
            ),
            "axis_y": (
                "3D 회전축의 y 성분입니다.\n"
                "예를 들어 axis=(1,1,1)은 body diagonal 방향을 기준으로 회전시키는 의미입니다."
            ),
            "axis_z": (
                "3D 회전축의 z 성분입니다.\n"
                "axis=(0,0,1)에 가까우면 detector plane 안에서의 in-plane rotation에 가까워집니다."
            ),
            "angle_deg": (
                "정다면체를 지정한 회전축(axis) 기준으로 회전시키는 각도입니다.\n"
                "단위는 degree입니다. projected density와 scattering pattern의 방향성이 바뀝니다."
            ),
            "scale_factor": (
                "실공간 object의 크기 스케일입니다.\n"
                "값이 커지면 reciprocal-space 패턴이 중심부로 더 좁게 모이고,\n"
                "값이 작아지면 패턴이 더 넓게 퍼집니다."
            ),
        }
        return texts.get(name, "")

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        plot_area = ttk.Frame(main, padding=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls,
            text="Polyhedron parameters",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        self.name_var = tk.StringVar(value="dodeca")

        self.vars = {
            "radius": tk.DoubleVar(value=1.0),
            "axis_x": tk.DoubleVar(value=1.0),
            "axis_y": tk.DoubleVar(value=1.0),
            "axis_z": tk.DoubleVar(value=1.0),
            "angle_deg": tk.DoubleVar(value=np.rad2deg(0.6)),
            "object_scale": tk.DoubleVar(value=0.7),
            "scale_factor": tk.DoubleVar(value=5000.0),
        }

        self.entries = {}

        shape_frame = ttk.LabelFrame(controls, text="Shape", padding=6)
        shape_frame.pack(fill=tk.X, pady=5)

        label_widget = ttk.Label(shape_frame, text="polyhedron_name")
        label_widget.pack(anchor="w")
        ToolTip(label_widget, self.get_tooltip_text("polyhedron_name"))

        combo = ttk.Combobox(
            shape_frame,
            textvariable=self.name_var,
            values=["tetra", "cube", "octa", "dodeca", "icosa"],
            state="readonly",
            width=18,
        )
        combo.pack(fill=tk.X, pady=(2, 4))
        combo.bind("<<ComboboxSelected>>", lambda _event: self.schedule_update())

        self._add_slider_entry(shape_frame, "radius", 0.2, 3.0, 0.1)
        self._add_slider_entry(shape_frame, "object_scale", 0.2, 1.0, 0.01)

        rot_frame = ttk.LabelFrame(controls, text="Rotation", padding=6)
        rot_frame.pack(fill=tk.X, pady=5)

        self._add_slider_entry(rot_frame, "axis_x", -2.0, 2.0, 0.1)
        self._add_slider_entry(rot_frame, "axis_y", -2.0, 2.0, 0.1)
        self._add_slider_entry(rot_frame, "axis_z", -2.0, 2.0, 0.1)
        self._add_slider_entry(rot_frame, "angle_deg", -180.0, 180.0, 1.0)

        scat_frame = ttk.LabelFrame(controls, text="Scattering", padding=6)
        scat_frame.pack(fill=tk.X, pady=5)

        self._add_slider_entry(scat_frame, "scale_factor", 100.0, 20000.0, 100.0)

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

        self.info = tk.Text(controls, width=40, height=11)
        self.info.pack(fill=tk.X, pady=(12, 0))

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_slider_entry(self, parent, name, vmin, vmax, resolution):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)

        label_widget = ttk.Label(frame, text=name, width=13)
        label_widget.pack(side=tk.LEFT)
        ToolTip(label_widget, self.get_tooltip_text(name))

        left_btn = ttk.Button(
            frame,
            text="◀",
            width=3,
            command=lambda key=name: self.step_value(key, -1),
        )
        left_btn.pack(side=tk.LEFT, padx=(2, 2))

        slider = ttk.Scale(
            frame,
            variable=self.vars[name],
            from_=vmin,
            to=vmax,
            orient=tk.HORIZONTAL,
            length=180,
            command=lambda _value, key=name: self.on_slider_change(key),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sliders[name] = slider

        right_btn = ttk.Button(
            frame,
            text="▶",
            width=3,
            command=lambda key=name: self.step_value(key, 1),
        )
        right_btn.pack(side=tk.LEFT, padx=(2, 4))

        entry = ttk.Entry(frame, width=10)
        entry.pack(side=tk.RIGHT, padx=(4, 0))
        entry.insert(0, f"{self.vars[name].get():.6g}")
        self.entries[name] = entry

        if not hasattr(self, "control_specs"):
            self.control_specs = {}
        self.control_specs[name] = {
            "min": float(vmin),
            "max": float(vmax),
            "step": float(resolution),
        }

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
            n = round((value - vmin) / step)
            value = vmin + n * step
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
            self.entries[name].selection_range(0, tk.END)
            self.entries[name].icursor(tk.END)
        if schedule_update:
            self.schedule_update()

    def current_entry_or_var_value(self, name):
        entry = self.entries.get(name)
        if entry is not None:
            text = entry.get().strip()
            if text:
                try:
                    return float(text)
                except ValueError:
                    pass
        return float(self.vars[name].get())

    def on_entry_step_key(self, event, name, direction):
        spec = self.control_specs[name]
        base = self.current_entry_or_var_value(name)
        self.set_parameter_value(
            name,
            base + float(direction) * spec["step"],
            snap_to_step=True,
            schedule_update=True,
            select_entry=True,
        )
        return "break"

    def on_entry_mousewheel(self, event, name):
        direction = +1 if event.delta > 0 else -1
        return self.on_entry_step_key(event, name, direction)

    def step_value(self, name, direction):
        spec = self.control_specs[name]
        value = float(self.vars[name].get()) + float(direction) * spec["step"]
        self.set_parameter_value(name, value, snap_to_step=True, schedule_update=True)

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
        entry = self.entries[name]
        try:
            value = float(entry.get())
        except ValueError:
            messagebox.showerror("Input error", f"{name} must be numeric.")
            self._sync_entry(name)
            return

        self.set_parameter_value(name, value, snap_to_step=True, schedule_update=True)

    def _sync_entry(self, name):
        entry = self.entries[name]
        entry.delete(0, tk.END)
        entry.insert(0, f"{self.vars[name].get():.6g}")

    def schedule_update(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(400, self.update_plot)

    def get_params(self):
        params = {name: float(var.get()) for name, var in self.vars.items()}
        params["polyhedron_name"] = self.name_var.get()
        return params

    def reset(self):
        self.name_var.set("dodeca")

        defaults = {
            "radius": 1.0,
            "axis_x": 1.0,
            "axis_y": 1.0,
            "axis_z": 1.0,
            "angle_deg": np.rad2deg(0.6),
            "scale_factor": 5000.0,
        }

        for name, value in defaults.items():
            self.vars[name].set(value)
            self._sync_entry(name)

        self.update_plot()

    def update_plot(self):
        self.update_job = None

        try:
            params = self.get_params()
            proj, du, dv, I_total, q_u, q_v, meta_proj, meta = run_simulation(params)

        except Exception as e:
            self._show_error(e)
            return

        I_masked = I_total * self.mask

        self.last_result = {
            "parameters": params,
            "proj": proj,
            "du": du,
            "dv": dv,
            "I_total": I_total,
            "I_masked": I_masked,
            "q_u": q_u,
            "q_v": q_v,
            "meta_proj": meta_proj,
            "meta": meta,
        }

        images = [
            proj,
            safe_log(I_total),
            I_masked,
        ]

        titles = [
            "projected density",
            "log theoretical",
            "masked theoretical",
        ]

        for ax, img, title in zip(self.axes, images, titles):
            ax.clear()
            ax.matshow(img)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self.fig.tight_layout()
        self.canvas.draw_idle()

        self._update_info(du, dv, q_u, q_v)

    def _update_info(self, du, dv, q_u, q_v):
        self.info.delete("1.0", tk.END)
        self.info.insert(
            tk.END,
            f"shape = {self.name_var.get()}\n"
            f"radius = {self.vars['radius'].get():.6g}\n"
            f"axis = [{self.vars['axis_x'].get():.3g}, "
            f"{self.vars['axis_y'].get():.3g}, "
            f"{self.vars['axis_z'].get():.3g}]\n"
            f"angle = {self.vars['angle_deg'].get():.6g} deg\n"
            f"object_scale = {self.vars['object_scale'].get():.6g}\n"
            f"scale_factor = {self.vars['scale_factor'].get():.6g}\n\n"
            f"du = {du:.6g}\n"
            f"dv = {dv:.6g}\n"
            f"q_u = [{q_u.min():.6g}, {q_u.max():.6g}]\n"
            f"q_v = [{q_v.min():.6g}, {q_v.max():.6g}]\n"
        )

    def ask_save_directory(self):
        if self.last_result is None:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shape_name = self.last_result["parameters"]["polyhedron_name"]
        default_name = f"polyhedron_{shape_name}_sim_{timestamp}"

        dialog = tk.Toplevel(self.root)
        dialog.title("Save result")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        parent_var = tk.StringVar(value="")
        name_var = tk.StringVar(value=default_name)
        full_var = tk.StringVar(value="")
        result = {"path": None}

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="Parent directory").grid(row=0, column=0, sticky="w")
        parent_entry = ttk.Entry(body, textvariable=parent_var, width=54)
        parent_entry.grid(row=1, column=0, sticky="ew", pady=(2, 6))

        def browse_parent():
            chosen = filedialog.askdirectory(title="Choose parent directory")
            if chosen:
                parent_var.set(chosen)
                update_full_path()

        ttk.Button(body, text="Browse...", command=browse_parent).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=(2, 6)
        )

        ttk.Label(body, text="Result folder name").grid(row=2, column=0, sticky="w")
        name_entry = ttk.Entry(body, textvariable=name_var, width=54)
        name_entry.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 6))

        ttk.Label(body, text="Full save path").grid(row=4, column=0, sticky="w")
        full_entry = ttk.Entry(body, textvariable=full_var, width=64, state="readonly")
        full_entry.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        button_frame = ttk.Frame(body)
        button_frame.grid(row=6, column=0, columnspan=2, sticky="e")

        def sanitize_folder_name(name):
            name = name.strip()
            invalid = '<>:"/\\|?*'
            for ch in invalid:
                name = name.replace(ch, "_")
            return name

        def update_full_path(*_args):
            parent = parent_var.get().strip()
            folder = sanitize_folder_name(name_var.get())
            if parent and folder:
                full_var.set(str(Path(parent) / folder))
            elif parent:
                full_var.set(str(Path(parent)))
            else:
                full_var.set("")

        def cancel():
            dialog.destroy()

        def confirm():
            parent = parent_var.get().strip()
            folder = sanitize_folder_name(name_var.get())

            if not parent:
                messagebox.showerror("Save error", "Please choose a parent directory.", parent=dialog)
                return
            if not Path(parent).exists():
                messagebox.showerror("Save error", "The parent directory does not exist.", parent=dialog)
                return
            if not folder:
                messagebox.showerror("Save error", "Please enter a result folder name.", parent=dialog)
                return

            save_dir = Path(parent) / folder
            if save_dir.exists():
                ok = messagebox.askyesno(
                    "Folder exists",
                    f"This folder already exists:\n{save_dir}\n\nSave into this folder?",
                    parent=dialog,
                )
                if not ok:
                    return

            result["path"] = save_dir
            dialog.destroy()

        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(button_frame, text="Save", command=confirm).pack(side=tk.RIGHT)

        parent_var.trace_add("write", update_full_path)
        name_var.trace_add("write", update_full_path)
        update_full_path()

        name_entry.focus_set()
        dialog.wait_window()
        return result["path"]

    def save_current_result(self):
        if self.last_result is None:
            messagebox.showerror("Save error", "No simulation result to save.")
            return

        save_dir = self.ask_save_directory()
        if save_dir is None:
            return

        try:
            save_dir.mkdir(parents=True, exist_ok=True)

            proj = self.last_result["proj"]
            I_total = self.last_result["I_total"]
            I_masked = self.last_result["I_masked"]

            self._save_image(save_dir / "01_projected_density.png", proj, title="projected density")
            self._save_image(save_dir / "02_log_theoretical.png", safe_log(I_total), title="log theoretical")
            self._save_image(save_dir / "03_masked_theoretical.png", I_masked, title="masked theoretical")
            self._save_parameters_txt(save_dir / "parameters.txt")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save result:\n{e}")
            return

        messagebox.showinfo("Saved", f"Saved result to:\n{save_dir}")

    def _save_image(self, path, img, title=""):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _save_parameters_txt(self, path):
        result = self.last_result
        p = result["parameters"]

        lines = [
            "Polyhedron SAXS simulation parameters",
            "=" * 42,
            "",
            "[Shape]",
            f"shape = polyhedron",
            f"polyhedron_name = {p['polyhedron_name']}",
            f"radius = {p['radius']}",
            "",
            "[Rotation]",
            f"axis_x = {p['axis_x']}",
            f"axis_y = {p['axis_y']}",
            f"axis_z = {p['axis_z']}",
            f"angle_deg = {p['angle_deg']}",
            f"angle_rad = {np.deg2rad(p['angle_deg'])}",
            "",
            "[Scattering]",
            f"object_scale = {p.get('object_scale', 1.0)}",
            f"scale_factor = {p['scale_factor']}",
            f"translation_u = 0.0",
            f"translation_v = 0.0",
            "",
            "[Files]",
            f"poni_path = {PONI_PATH}",
            f"mask_path = {MASK_PATH}",
            "",
            "[Grid]",
            f"image_shape = {IMG_SHAPE}",
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
    app = PolyhedronGUI(root)
    root.mainloop()