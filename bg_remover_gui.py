"""
Background Remover
===================
A small cross-platform (Windows / Linux / macOS) desktop GUI that removes
the background from a photo, with a live-updating strength slider so you
can dial in exactly how much gets cut before saving.

How it works
------------
The AI model (rembg; pick from a few U^2-Net/IS-Net variants) runs once
per image and produces a
confidence mask, which is lightly smoothed and binarized into a cutout mask.
That mask is converted to a signed distance field (how many pixels each
point is from the cutout edge) and cached. Moving the slider does NOT
re-run the model - it just grows or shrinks the cutout boundary by a number
of pixels read off that cached field, so the preview updates instantly and
the direction is guaranteed:

    - Low strength  -> the kept area is grown outward, so more of the
                       image survives (safer, may keep a thin edge of
                       background).
    - High strength -> the kept area is shrunk inward (aggressive, may
                       eat into the subject's edges).

On top of that, a manual touch-up brush lets you paint directly over the
preview to force an area to always be kept or always removed (e.g. a stray
background blob the AI insists is foreground) - painted areas override the
slider wherever you've painted, with a soft edge.

Run:
    pip install -r requirements.txt
    python bg_remover_gui.py
"""
import math
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import distance_transform_edt, gaussian_filter

try:
    from rembg import new_session
    from rembg import remove as rembg_remove
except ImportError:
    rembg_remove = None
    new_session = None

PREVIEW_MAX = 480       # max width/height of each preview panel, in pixels
CHECKER_SIZE = 12       # checkerboard square size, in pixels (shows transparency)
DEFAULT_STRENGTH = 50   # slider default, 0-100
MASK_THRESHOLD = 127    # binarize the AI's (alpha-matted) output at this level
MAX_RADIUS = 40         # max pixels the cutout boundary can grow/shrink by
FEATHER = 3             # half-width, in pixels, of the smoothed edge transition
BRUSH_MIN = 5           # smallest touch-up brush radius, in original-image pixels
BRUSH_MAX = 150         # largest touch-up brush radius, in original-image pixels
DEFAULT_BRUSH = 40

# Model to use for the initial AI pass. Pick based on your subject matter.
# The three "High Quality" options are all large (~930-980MB one-time
# download) and slow (~5-10s/image on CPU) - they trade speed for accuracy.
#
#   High Quality (birefnet-massive) - best result in testing: the ONLY model
#                                   of six tried that correctly separated a
#                                   background object (a moon) visually
#                                   merged with the subject, at every
#                                   strength setting. Try this first.
#   High Quality (birefnet-general) - very good edges, but did not separate
#                                   the moon case above; may still win on
#                                   other images
#   High Quality (bria-rmbg)     - well-regarded commercial-grade model;
#                                   also did not separate the moon case
#   General (u2net)              - fast (~1s), small (~180MB), solid default
#                                   when you don't need the slow models
#   General v2 (isnet-general)   - newer, often sharper edges than u2net,
#                                   similar speed
#   Human / Portrait             - tuned for people, best on portrait photos
#
# NOTE: rembg also ships an "isnet-anime" model, but it was tested here and
# found to output near-zero confidence across the entire frame on real
# anime/illustration art (not just at the edges - everywhere), making its
# mask unusable. It's deliberately left out of this list until that's
# understood; u2net performs well on illustrated/anime content in practice.
#
# A "confident but wrong" mask (a background object visually merged into the
# subject's silhouette) is a composition ambiguity most models can't resolve
# on their own - birefnet-massive was the exception in testing, but for
# anything it still gets wrong, that's what the touch-up brush is for.
MODELS = [
    ("High Quality (birefnet-massive, slower)", "birefnet-massive"),
    ("High Quality (birefnet-general, slower)", "birefnet-general"),
    ("High Quality (bria-rmbg, slower)", "bria-rmbg"),
    ("General (u2net)", "u2net"),
    ("General v2 (isnet-general-use)", "isnet-general-use"),
    ("Human / Portrait (u2net_human_seg)", "u2net_human_seg"),
]
SLOW_MODELS = {"birefnet-massive", "birefnet-general", "bria-rmbg"}


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

        self.sessions = {}           # model_key -> rembg session (cached, lazy-created)
        self.original_image = None   # PIL RGBA, full resolution
        self.signed_dist = None      # numpy float32 array, full resolution, px to cutout edge
                                      # (positive = inside the kept area, negative = outside)
        self.override = None         # numpy float32 array, full resolution, manual touch-ups:
                                      # +1 = always keep, -1 = always remove, 0 = no override
        self._original_photo = None  # keep a reference so Tk doesn't garbage-collect it
        self._result_photo = None
        self._result_geom = None     # (offset_x, offset_y, scale) mapping result canvas -> image
        self._last_paint_xy = None
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

        ttk.Label(toolbar, text="Model:").pack(side="left", padx=(16, 4))
        self.model_var = tk.StringVar(value=MODELS[0][0])
        model_combo = ttk.Combobox(
            toolbar, textvariable=self.model_var, values=[label for label, _ in MODELS],
            state="readonly", width=30,
        )
        model_combo.pack(side="left")
        model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        self.status_var = tk.StringVar(value="Open an image to begin.")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="left", padx=16)

        touchup_bar = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        touchup_bar.pack(side="top", fill="x")

        ttk.Label(touchup_bar, text="Touch-up brush:").pack(side="left")
        self.brush_mode_var = tk.StringVar(value="off")
        self.brush_radios = []
        for value, text in (("off", "Off"), ("keep", "Keep"), ("remove", "Remove")):
            rb = ttk.Radiobutton(touchup_bar, text=text, value=value, variable=self.brush_mode_var)
            rb.pack(side="left", padx=(4, 0))
            self.brush_radios.append(rb)

        ttk.Label(touchup_bar, text="   Brush size:").pack(side="left")
        self.brush_size_var = tk.IntVar(value=DEFAULT_BRUSH)
        self.brush_scale = ttk.Scale(
            touchup_bar, from_=BRUSH_MIN, to=BRUSH_MAX, orient="horizontal",
            variable=self.brush_size_var, length=120,
        )
        self.brush_scale.pack(side="left", padx=(4, 0))

        self.clear_touchup_btn = ttk.Button(touchup_bar, text="Clear Touch-ups", command=self.clear_touchups)
        self.clear_touchup_btn.pack(side="left", padx=(12, 0))
        self._set_brush_enabled(False)

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
        self.result_canvas.bind("<Configure>", lambda e: self.signed_dist is not None and self._render_result())
        self.result_canvas.bind("<ButtonPress-1>", self._on_paint_start)
        self.result_canvas.bind("<B1-Motion>", self._on_paint_drag)
        self.result_canvas.bind("<ButtonRelease-1>", self._on_paint_end)

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
            text="Low = grow the cutout outward, keeping more of the image (safer, may keep a "
                 "thin edge of background). High = shrink it inward (may eat into the subject). "
                 "Stray areas the slider can't fix? Pick Keep/Remove above and paint over them.",
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
        self.signed_dist = None
        self.override = None
        self.save_btn.state(["disabled"])
        self.slider.state(["disabled"])
        self._set_brush_enabled(False)
        self.result_canvas.delete("all")
        self._render_original()
        self._start_background_removal()

    def _on_model_change(self, _event=None):
        if self.original_image is not None:
            self.signed_dist = None
            self.override = None
            self.save_btn.state(["disabled"])
            self.slider.state(["disabled"])
            self._set_brush_enabled(False)
            self._start_background_removal()

    def _model_key(self):
        label = self.model_var.get()
        return next((key for lbl, key in MODELS if lbl == label), MODELS[0][1])

    def _start_background_removal(self):
        if rembg_remove is None:
            messagebox.showerror(
                "rembg not installed",
                "The 'rembg' package is required for background removal.\n\n"
                "Install it with:\n    pip install -r requirements.txt",
            )
            self.status_var.set("rembg is not installed.")
            return

        model_key = self._model_key()
        if model_key in self.sessions:
            note = ""
        elif model_key in SLOW_MODELS:
            note = " (first use downloads ~930-980 MB, then this model itself takes ~5-10s/image)"
        else:
            note = " (first use of this model downloads it, ~40-180 MB)"
        self.status_var.set(f"Removing background with '{model_key}'...{note}")
        threading.Thread(target=self._remove_background_worker, args=(model_key,), daemon=True).start()

    def _remove_background_worker(self, model_key):
        try:
            session = self.sessions.get(model_key)
            if session is None:
                session = new_session(model_key)
                self.sessions[model_key] = session
            # NOTE: alpha_matting is intentionally NOT used here. Its trimap
            # step hard-erodes the model's initial mask (by
            # alpha_matting_erode_size px) to decide what counts as "certain"
            # foreground, and for thin/detailed subjects (anime line art,
            # thin limbs, weapons) that erosion can wipe out most of the
            # subject before matting even begins - it was observed to erase
            # nearly an entire character down to a couple of small fragments.
            result = rembg_remove(self.original_image, session=session)
            raw_alpha = np.array(result.split()[-1], dtype=np.float32)

            # Light smoothing to reduce speckle noise before binarizing,
            # without eroding thin real detail the way matting's trimap did.
            raw_alpha = gaussian_filter(raw_alpha, sigma=1.5)

            # Binarize once, then precompute a signed distance field (pixels
            # to the nearest edge, positive inside the kept area) so the
            # slider can grow/shrink the cutout with just a compare per move.
            mask = raw_alpha >= MASK_THRESHOLD
            dist_in = distance_transform_edt(mask)
            dist_out = distance_transform_edt(~mask)
            signed_dist = (dist_in - dist_out).astype(np.float32)
            self.work_queue.put(("done", signed_dist))
        except Exception as exc:
            self.work_queue.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.work_queue.get_nowait()
                if kind == "done":
                    self.signed_dist = payload
                    self.override = np.zeros_like(payload)
                    self.slider.state(["!disabled"])
                    self.save_btn.state(["!disabled"])
                    self._set_brush_enabled(True)
                    self.status_var.set("Done. Drag the slider, or paint touch-ups on the preview.")
                    self._render_result()
                elif kind == "error":
                    self.status_var.set("Background removal failed.")
                    messagebox.showerror("Background removal failed", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_slider_move(self, _value):
        self.strength_label.config(text=f"{self.strength_var.get()}%")
        if self.signed_dist is not None:
            self._render_result()

    def _thresholded_alpha(self):
        """Grow/shrink the cached cutout boundary using the current slider value.

        strength=0   -> boundary pushed outward by MAX_RADIUS px (keep more)
        strength=50  -> boundary unchanged (the AI's own cutout)
        strength=100 -> boundary pulled inward by MAX_RADIUS px (cut more)

        This reads directly off the precomputed signed distance field, so it
        stays correct regardless of how noisy or oddly-scaled a given model's
        raw confidence output is - the direction can't invert and the result
        can't linger at a ghostly half-opacity the way raw-value thresholding
        could.
        """
        strength = self.strength_var.get()  # 0-100
        offset = (50 - strength) / 50.0 * MAX_RADIUS
        alpha = (self.signed_dist + offset) / FEATHER * 127.5 + 127.5
        alpha = np.clip(alpha, 0, 255)

        if self.override is not None and np.any(self.override):
            # Manual touch-ups win over the slider wherever painted, with a
            # soft edge so the brushed area doesn't have a hard seam.
            influence = np.clip(gaussian_filter(self.override, sigma=FEATHER), -1, 1)
            forced = np.where(influence >= 0, 255.0, 0.0)
            weight = np.abs(influence)
            alpha = alpha * (1 - weight) + forced * weight

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
        scale = disp_w / w
        offset_x = cw // 2 - disp_w // 2
        offset_y = ch // 2 - disp_h // 2
        self._result_geom = (offset_x, offset_y, scale)
        disp = result.resize((disp_w, disp_h), Image.LANCZOS)
        checker = make_checkerboard((disp_w, disp_h))
        composed = Image.alpha_composite(checker, disp)
        self._result_photo = ImageTk.PhotoImage(composed)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=self._result_photo, anchor="center")

    # -------------------------------------------------------- Touch-up brush --
    def _set_brush_enabled(self, enabled):
        state = ["!disabled"] if enabled else ["disabled"]
        for rb in self.brush_radios:
            rb.state(state)
        self.brush_scale.state(state)
        self.clear_touchup_btn.state(state)
        if not enabled:
            self.brush_mode_var.set("off")

    def clear_touchups(self):
        if self.override is not None:
            self.override[:] = 0
            self._render_result()

    def _canvas_to_image_xy(self, event):
        if self._result_geom is None or self.original_image is None:
            return None
        offset_x, offset_y, scale = self._result_geom
        img_x = (event.x - offset_x) / scale
        img_y = (event.y - offset_y) / scale
        w, h = self.original_image.size
        if 0 <= img_x < w and 0 <= img_y < h:
            return img_x, img_y
        return None

    def _stamp_brush(self, cx, cy):
        r = self.brush_size_var.get()
        value = 1.0 if self.brush_mode_var.get() == "keep" else -1.0
        h, w = self.override.shape
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
        y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        self.override[y0:y1, x0:x1][circle] = value

    def _stamp_line(self, p0, p1):
        (x0, y0), (x1, y1) = p0, p1
        dist = math.hypot(x1 - x0, y1 - y0)
        step = max(1.0, self.brush_size_var.get() / 2.0)
        steps = max(1, int(dist / step))
        for i in range(steps + 1):
            t = i / steps
            self._stamp_brush(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    def _on_paint_start(self, event):
        if self.brush_mode_var.get() == "off" or self.override is None:
            return
        pt = self._canvas_to_image_xy(event)
        if pt is None:
            return
        self._last_paint_xy = pt
        self._stamp_brush(*pt)
        self._render_result()

    def _on_paint_drag(self, event):
        if self.brush_mode_var.get() == "off" or self.override is None:
            return
        pt = self._canvas_to_image_xy(event)
        if pt is None:
            return
        if self._last_paint_xy is not None:
            self._stamp_line(self._last_paint_xy, pt)
        else:
            self._stamp_brush(*pt)
        self._last_paint_xy = pt
        self._render_result()

    def _on_paint_end(self, _event):
        self._last_paint_xy = None

    def save_image(self):
        if self.signed_dist is None:
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
