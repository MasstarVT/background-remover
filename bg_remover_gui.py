"""
Background Remover
===================
A small cross-platform (Windows / Linux / macOS) desktop GUI that removes
the background from a photo, with a live-updating strength slider so you
can dial in exactly how much gets cut before saving.

How it works
------------
The AI model (rembg / U^2-Net) runs once per image and produces a
continuous "confidence" alpha mask (0-255) of how likely each pixel is to
be foreground. Moving the slider does NOT re-run the model - it just
re-thresholds that cached mask, so the preview updates instantly:

    - Low strength  -> only very-confident background pixels are removed
                       (safe, but some background may remain).
    - High strength -> only very-confident foreground pixels are kept
                       (aggressive, may eat into the subject's edges).

Run:
    pip install -r requirements.txt
    python bg_remover_gui.py
"""
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

try:
    from rembg import new_session
    from rembg import remove as rembg_remove
except ImportError:
    rembg_remove = None
    new_session = None

PREVIEW_MAX = 480       # max width/height of each preview panel, in pixels
CHECKER_SIZE = 12       # checkerboard square size, in pixels (shows transparency)
DEFAULT_STRENGTH = 50   # slider default, 0-100
SOFT_BAND = 40          # width of the alpha feather band (0-255 scale), keeps edges smooth


def make_checkerboard(size, square=CHECKER_SIZE):
    """Return an RGBA checkerboard image used to visualize transparent areas."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    light, dark = 235, 200
    for y in range(0, h, square):
        for x in range(0, w, square):
            color = light if ((x // square) + (y // square)) % 2 == 0 else dark
            arr[y:y + square, x:x + square] = color
    return Image.fromarray(arr, mode="RGB").convert("RGBA")


def fit_size(w, h, max_dim):
    """Scale (w, h) down to fit within max_dim while keeping aspect ratio."""
    scale = min(max_dim / w, max_dim / h, 1.0)
    return max(1, int(w * scale)), max(1, int(h * scale))


class BackgroundRemoverApp:
    def __init__(self, root):
        self.root = root
        root.title("Background Remover")
        root.geometry("1040x680")
        root.minsize(760, 520)

        self.session = None
        self.original_image = None   # PIL RGBA, full resolution
        self.raw_alpha = None        # numpy float32 array, full resolution, values 0-255
        self._original_photo = None  # keep a reference so Tk doesn't garbage-collect it
        self._result_photo = None
        self.work_queue = queue.Queue()

        self._build_ui()
        self._poll_queue()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="Open Image...", command=self.open_image).pack(side="left")
        self.save_btn = ttk.Button(toolbar, text="Save Result...", command=self.save_image, state="disabled")
        self.save_btn.pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="Open an image to begin.")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="left", padx=16)

        panels = ttk.Frame(self.root, padding=(8, 0))
        panels.pack(side="top", fill="both", expand=True)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(1, weight=1)

        ttk.Label(panels, text="Original", anchor="center").grid(row=0, column=0, sticky="ew")
        ttk.Label(panels, text="Preview (background removed)", anchor="center").grid(row=0, column=1, sticky="ew")

        self.original_canvas = tk.Canvas(panels, bg="#2b2b2b", highlightthickness=1, highlightbackground="#555")
        self.original_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=4)
        self.result_canvas = tk.Canvas(panels, bg="#2b2b2b", highlightthickness=1, highlightbackground="#555")
        self.result_canvas.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=4)

        self.original_canvas.bind("<Configure>", lambda e: self.original_image and self._render_original())
        self.result_canvas.bind("<Configure>", lambda e: self.raw_alpha is not None and self._render_result())

        slider_frame = ttk.Frame(self.root, padding=8)
        slider_frame.pack(side="bottom", fill="x")

        ttk.Label(slider_frame, text="Removal Strength").pack(side="left")
        self.strength_var = tk.IntVar(value=DEFAULT_STRENGTH)
        self.strength_label = ttk.Label(slider_frame, text=f"{DEFAULT_STRENGTH}%", width=5)
        self.strength_label.pack(side="right")

        self.slider = ttk.Scale(
            slider_frame, from_=0, to=100, orient="horizontal",
            variable=self.strength_var, command=self._on_slider_move,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.slider.state(["disabled"])

        ttk.Label(
            self.root,
            text="Low = keep more of the image (safer, may leave background). "
                 "High = cut more aggressively (may eat into the subject).",
            foreground="#888",
        ).pack(side="bottom", pady=(0, 6))

    # ------------------------------------------------------------ Actions --
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            return

        self.original_image = img
        self.raw_alpha = None
        self.save_btn.state(["disabled"])
        self.slider.state(["disabled"])
        self.result_canvas.delete("all")
        self._render_original()
        self._start_background_removal()

    def _start_background_removal(self):
        if rembg_remove is None:
            messagebox.showerror(
                "rembg not installed",
                "The 'rembg' package is required for background removal.\n\n"
                "Install it with:\n    pip install -r requirements.txt",
            )
            self.status_var.set("rembg is not installed.")
            return

        self.status_var.set("Removing background... (first run downloads the AI model, ~180 MB)")
        threading.Thread(target=self._remove_background_worker, daemon=True).start()

    def _remove_background_worker(self):
        try:
            if self.session is None:
                self.session = new_session("u2net")
            result = rembg_remove(self.original_image, session=self.session)
            alpha = np.array(result.split()[-1], dtype=np.float32)
            self.work_queue.put(("done", alpha))
        except Exception as exc:
            self.work_queue.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.work_queue.get_nowait()
                if kind == "done":
                    self.raw_alpha = payload
                    self.slider.state(["!disabled"])
                    self.save_btn.state(["!disabled"])
                    self.status_var.set("Done. Drag the slider to adjust removal strength.")
                    self._render_result()
                elif kind == "error":
                    self.status_var.set("Background removal failed.")
                    messagebox.showerror("Background removal failed", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_slider_move(self, _value):
        self.strength_label.config(text=f"{self.strength_var.get()}%")
        if self.raw_alpha is not None:
            self._render_result()

    def _thresholded_alpha(self):
        """Re-threshold the cached raw alpha mask using the current slider value."""
        strength = self.strength_var.get()  # 0-100
        threshold = strength / 100.0 * 255.0
        low = threshold - SOFT_BAND / 2
        high = threshold + SOFT_BAND / 2
        if high <= low:
            high = low + 1
        alpha = (self.raw_alpha - low) / (high - low) * 255.0
        return np.clip(alpha, 0, 255).astype(np.uint8)

    def _composited_result(self):
        alpha = self._thresholded_alpha()
        rgb = np.array(self.original_image.convert("RGB"))
        rgba = np.dstack([rgb, alpha])
        return Image.fromarray(rgba, mode="RGBA")

    # ---------------------------------------------------------- Rendering --
    def _render_original(self):
        canvas = self.original_canvas
        cw = canvas.winfo_width() or PREVIEW_MAX
        ch = canvas.winfo_height() or PREVIEW_MAX
        max_dim = max(200, min(cw, ch, PREVIEW_MAX))
        w, h = self.original_image.size
        disp_w, disp_h = fit_size(w, h, max_dim)
        disp = self.original_image.convert("RGB").resize((disp_w, disp_h), Image.LANCZOS)
        self._original_photo = ImageTk.PhotoImage(disp)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=self._original_photo, anchor="center")

    def _render_result(self):
        result = self._composited_result()
        canvas = self.result_canvas
        cw = canvas.winfo_width() or PREVIEW_MAX
        ch = canvas.winfo_height() or PREVIEW_MAX
        max_dim = max(200, min(cw, ch, PREVIEW_MAX))
        w, h = result.size
        disp_w, disp_h = fit_size(w, h, max_dim)
        disp = result.resize((disp_w, disp_h), Image.LANCZOS)
        checker = make_checkerboard((disp_w, disp_h))
        composed = Image.alpha_composite(checker, disp)
        self._result_photo = ImageTk.PhotoImage(composed)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=self._result_photo, anchor="center")

    def save_image(self):
        if self.raw_alpha is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save result",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return
        result = self._composited_result()
        try:
            result.save(path)
            self.status_var.set(f"Saved to {path}")
        except Exception as exc:
            messagebox.showerror("Could not save image", str(exc))


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    BackgroundRemoverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
