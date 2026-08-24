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
import json
import math
import queue
import threading
import tkinter as tk
from pathlib import Path
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
FEATHER_MIN = 1         # smallest allowed edge-softness slider value, in pixels
FEATHER_MAX = 15        # largest allowed edge-softness slider value, in pixels
DEFAULT_FEATHER = 3     # default half-width, in pixels, of the smoothed edge transition
BRUSH_MIN = 5           # smallest touch-up brush radius, in original-image pixels
BRUSH_MAX = 150         # largest touch-up brush radius, in original-image pixels
DEFAULT_BRUSH = 40
RESULT_ZOOM_MAX = 16.0  # max result-canvas zoom (1600%) - past this is pointless magnification
RESULT_ZOOM_STEP = 1.15 # multiplicative scale change per mouse-wheel notch
UNDO_STACK_MAX = 50     # cap on touch-up undo/redo history depth, oldest entries dropped first

CONFIG_PATH = Path.home() / ".background_remover_config.json"

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


def load_config(path=CONFIG_PATH):
    """Load saved preferences (last model, last folders) as a plain dict.

    A missing file (first run) or a corrupt/unreadable one (hand-edited,
    truncated by a crash, etc.) is not fatal - either way we just fall back
    to an empty dict and the app uses its normal defaults.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_config(config, path=CONFIG_PATH):
    """Best-effort save of preferences. Never raises - a read-only or
    missing home directory shouldn't crash the app over a convenience
    feature.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)
    except OSError:
        pass


class BackgroundRemoverApp:
    def __init__(self, root):
        self.root = root
        root.title("Background Remover")
        root.geometry("1040x680")
        root.minsize(760, 520)

        self.config = load_config()  # persisted preferences: model, last open/save dirs
        self.sessions = {}           # model_key -> rembg session (cached, lazy-created)
        self.sessions_lock = threading.Lock()  # guards session creation, see _get_or_create_session
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

        # --- Result-canvas view state (zoom/pan) -----------------------
        # canvas_x = image_x * scale + offset_x (same for y). "_result_fit"
        # means "chase whatever scale/offset fits the whole image to the
        # canvas" - it's re-derived every render while True. Any manual
        # zoom or pan sets it False and freezes scale/offset until the
        # user hits "Fit" again. See _fit_result_geom / _zoom_result.
        self._result_scale = 1.0
        self._result_offset_x = 0.0
        self._result_offset_y = 0.0
        self._result_fit = True
        self._result_last_canvas_size = None  # (cw, ch) as of the last render, for Configure handling
        self._last_mouse_xy = None            # last known cursor pos on the result canvas
        self._brush_cursor_ids = (None, None)  # canvas item ids for the brush-size preview ring
        self._panning = False
        self._pan_last_xy = None

        # --- Touch-up undo/redo -----------------------------------------
        self._stroke_bbox = None        # running (y0, y1, x0, x1) touched by the in-progress stroke
        self._stroke_prior_full = None  # full self.override snapshot at the start of the stroke
        self.undo_stack = []            # list of (y0, y1, x0, x1, prior_region_array)
        self.redo_stack = []

        self._build_ui()
        self._poll_queue()
        self._preload_default_model()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="Open Image...", command=self.open_image).pack(side="left")
        self.save_btn = ttk.Button(toolbar, text="Save Result...", command=self.save_image, state="disabled")
        self.save_btn.pack(side="left", padx=(8, 0))

        ttk.Label(toolbar, text="Model:").pack(side="left", padx=(16, 4))
        self.model_var = tk.StringVar(value=self._model_label_for_key(self.config.get("model")))
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

        # Keep the brush-size preview ring (see _update_brush_cursor) in sync
        # whenever the mode or size changes, not just when the mouse moves -
        # e.g. dragging the size slider should visibly resize the ring even
        # if the cursor itself is sitting still over the canvas.
        self.brush_mode_var.trace_add("write", self._on_brush_setting_change)
        self.brush_size_var.trace_add("write", self._on_brush_setting_change)

        self.clear_touchup_btn = ttk.Button(touchup_bar, text="Clear Touch-ups", command=self.clear_touchups)
        self.clear_touchup_btn.pack(side="left", padx=(12, 0))

        self.undo_btn = ttk.Button(touchup_bar, text="Undo", command=self.undo_touchup, state="disabled")
        self.undo_btn.pack(side="left", padx=(12, 0))
        self.redo_btn = ttk.Button(touchup_bar, text="Redo", command=self.redo_touchup, state="disabled")
        self.redo_btn.pack(side="left", padx=(4, 0))

        self._set_brush_enabled(False)

        zoom_frame = ttk.Frame(touchup_bar)
        zoom_frame.pack(side="right")
        self.fit_zoom_btn = ttk.Button(zoom_frame, text="Fit", command=self.reset_zoom, width=5)
        self.fit_zoom_btn.pack(side="right")
        self.zoom_label = ttk.Label(zoom_frame, text="Zoom: --", width=10, anchor="e")
        self.zoom_label.pack(side="right", padx=(0, 6))

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
        self.result_canvas.bind("<Configure>", self._on_result_configure)

        # Left-button drag: paints when a brush mode is active, otherwise
        # pans (see _on_paint_start). Middle-button drag always pans
        # regardless of brush mode, as a platform-independent alternative
        # for anyone whose pointing device makes chording awkward, or who
        # just prefers not to have to flip the brush to "Off" to pan around.
        self.result_canvas.bind("<ButtonPress-1>", self._on_paint_start)
        self.result_canvas.bind("<B1-Motion>", self._on_paint_drag)
        self.result_canvas.bind("<ButtonRelease-1>", self._on_paint_end)
        self.result_canvas.bind("<ButtonPress-2>", self._on_pan_button_press)
        self.result_canvas.bind("<B2-Motion>", self._drag_pan)
        self.result_canvas.bind("<ButtonRelease-2>", self._end_pan)

        # Mouse-wheel zoom: Windows/macOS deliver <MouseWheel> with a signed
        # event.delta; X11 has no delta at all and instead sends discrete
        # <Button-4> (up) / <Button-5> (down) clicks - bind all three so
        # zooming works the same way on every platform.
        self.result_canvas.bind("<MouseWheel>", self._on_result_wheel)
        self.result_canvas.bind("<Button-4>", self._on_result_wheel)
        self.result_canvas.bind("<Button-5>", self._on_result_wheel)

        self.result_canvas.bind("<Motion>", self._on_result_motion)
        self.result_canvas.bind("<Leave>", self._on_result_leave)

        self.root.bind_all("<Control-z>", lambda e: self.undo_touchup())
        self.root.bind_all("<Control-y>", lambda e: self.redo_touchup())
        self.root.bind_all("<Control-Shift-Z>", lambda e: self.redo_touchup())
        self.root.bind_all("<Control-Shift-z>", lambda e: self.redo_touchup())

        slider_frame = ttk.Frame(self.root, padding=8)
        slider_frame.pack(side="bottom", fill="x")

        strength_row = ttk.Frame(slider_frame)
        strength_row.pack(side="top", fill="x")

        ttk.Label(strength_row, text="Removal Strength").pack(side="left")
        self.strength_var = tk.IntVar(value=DEFAULT_STRENGTH)
        self.strength_label = ttk.Label(strength_row, text=f"{DEFAULT_STRENGTH}%", width=5)
        self.strength_label.pack(side="right")

        self.slider = ttk.Scale(
            strength_row, from_=0, to=100, orient="horizontal",
            variable=self.strength_var, command=self._on_slider_move,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.slider.state(["disabled"])

        feather_row = ttk.Frame(slider_frame)
        feather_row.pack(side="top", fill="x", pady=(6, 0))

        ttk.Label(feather_row, text="Edge Softness").pack(side="left")
        self.feather_var = tk.IntVar(value=DEFAULT_FEATHER)
        self.feather_label = ttk.Label(feather_row, text=f"{DEFAULT_FEATHER}px", width=5)
        self.feather_label.pack(side="right")

        self.feather_slider = ttk.Scale(
            feather_row, from_=FEATHER_MIN, to=FEATHER_MAX, orient="horizontal",
            variable=self.feather_var, command=self._on_feather_move,
        )
        self.feather_slider.pack(side="left", fill="x", expand=True, padx=8)
        self.feather_slider.state(["disabled"])

        ttk.Label(
            self.root,
            text="Low = grow the cutout outward, keeping more of the image (safer, may keep a "
                 "thin edge of background). High = shrink it inward (may eat into the subject). "
                 "Stray areas the slider can't fix? Pick Keep/Remove above and paint over them. "
                 "Scroll to zoom the preview, middle-drag (or left-drag with the brush Off) to "
                 "pan, Ctrl+Z/Ctrl+Y to undo/redo a brush stroke.",
            foreground="#888",
        ).pack(side="bottom", pady=(0, 6))

    # ------------------------------------------------------------ Actions --
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
            initialdir=self.config.get("last_open_dir", ""),
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            return

        self.config["last_open_dir"] = str(Path(path).resolve().parent)
        save_config(self.config)

        self.original_image = img
        self.signed_dist = None
        self.override = None
        self.save_btn.state(["disabled"])
        self.slider.state(["disabled"])
        self.feather_slider.state(["disabled"])
        self._set_brush_enabled(False)
        # A new image is a different array (different shape, unrelated
        # content) - any zoom/pan or undo/redo history from the previous
        # image no longer makes sense, so reset both.
        self._result_fit = True
        self._result_last_canvas_size = None
        self._reset_undo_redo()
        self.result_canvas.delete("all")
        self._render_original()
        self._start_background_removal()

    def _on_model_change(self, _event=None):
        self.config["model"] = self._model_key()
        save_config(self.config)
        if self.original_image is not None:
            self.signed_dist = None
            self.override = None
            self.save_btn.state(["disabled"])
            self.slider.state(["disabled"])
            self.feather_slider.state(["disabled"])
            self._set_brush_enabled(False)
            # The override array is about to be replaced (same shape, but a
            # different model's touch-ups don't apply) - the undo/redo
            # history would otherwise point at stale data.
            self._reset_undo_redo()
            self._start_background_removal()

    def _model_key(self):
        label = self.model_var.get()
        return next((key for lbl, key in MODELS if lbl == label), MODELS[0][1])

    def _model_label_for_key(self, model_key):
        """Look up the combobox label for a saved model key, falling back to
        the default model if the key is missing or no longer recognized
        (e.g. an older config file, or MODELS changed between versions)."""
        return next((label for label, key in MODELS if key == model_key), MODELS[0][0])

    def _get_or_create_session(self, model_key):
        """Return the cached rembg session for model_key, creating it if needed.

        Both the startup preload and a user-triggered removal call this, and
        it's guarded by a lock so they can't race to build two sessions for
        the same (possibly large, slow-to-load) model at once: whichever
        call arrives first does the loading while the other blocks, then
        finds the session already cached and returns it immediately.
        """
        with self.sessions_lock:
            session = self.sessions.get(model_key)
            if session is None:
                session = new_session(model_key)
                self.sessions[model_key] = session
            return session

    def _preload_default_model(self):
        """Kick off loading the last-used (or default) model as soon as the
        window appears, in a background thread, so its load time - which can
        be 15-30s for the large "High Quality" models - is hidden behind the
        time the user spends picking a file instead of happening afterward,
        when they're just staring at "Removing background...".
        """
        if new_session is None:
            return
        model_key = self._model_key()
        self.status_var.set(f"Warming up '{model_key}'... this may take a while the first time.")
        threading.Thread(target=self._preload_worker, args=(model_key,), daemon=True).start()

    def _preload_worker(self, model_key):
        try:
            self._get_or_create_session(model_key)
        except Exception:
            # Not fatal here - any real problem (bad model key, no internet
            # for the download, etc.) will surface with a proper error
            # dialog when the user actually tries to remove a background.
            pass
        self.work_queue.put(("preload_done", model_key))

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
            session = self._get_or_create_session(model_key)
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
                    self.feather_slider.state(["!disabled"])
                    self.save_btn.state(["!disabled"])
                    self._set_brush_enabled(True)
                    self.status_var.set("Done. Drag the slider, or paint touch-ups on the preview.")
                    self._render_result()
                elif kind == "error":
                    self.status_var.set("Background removal failed.")
                    messagebox.showerror("Background removal failed", payload)
                elif kind == "preload_done":
                    # Only touch the status bar if nothing else has already
                    # taken it over (the user opened an image while the
                    # preload was still running, a removal is in progress
                    # or done, etc.) - the preload message is only relevant
                    # while it's still the most recent thing that happened.
                    if self.status_var.get().startswith("Warming up"):
                        self.status_var.set("Ready.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_slider_move(self, _value):
        self.strength_label.config(text=f"{self.strength_var.get()}%")
        if self.signed_dist is not None:
            self._render_result()

    def _on_feather_move(self, _value):
        self.feather_label.config(text=f"{self.feather_var.get()}px")
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
        feather = self.feather_var.get()    # edge-softness slider, in pixels
        offset = (50 - strength) / 50.0 * MAX_RADIUS
        alpha = (self.signed_dist + offset) / feather * 127.5 + 127.5
        alpha = np.clip(alpha, 0, 255)

        if self.override is not None and np.any(self.override):
            # Manual touch-ups win over the slider wherever painted, with a
            # soft edge so the brushed area doesn't have a hard seam.
            influence = np.clip(gaussian_filter(self.override, sigma=feather), -1, 1)
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

    def _result_canvas_size(self):
        canvas = self.result_canvas
        cw = canvas.winfo_width() or PREVIEW_MAX
        ch = canvas.winfo_height() or PREVIEW_MAX
        return cw, ch

    def _fit_result_geom(self, w, h):
        """Compute (scale, offset_x, offset_y) that fits a w x h image
        centered in the result canvas, capped at PREVIEW_MAX like the
        preview always was pre-zoom. Used both to render "Fit" mode and as
        the floor _zoom_result won't let you zoom out past (there's no
        useful reason to zoom "out" further than the whole image already
        fitting on screen)."""
        cw, ch = self._result_canvas_size()
        max_dim = max(200, min(cw, ch, PREVIEW_MAX))
        scale = min(max_dim / w, max_dim / h, 1.0)
        offset_x = cw / 2 - (w * scale) / 2
        offset_y = ch / 2 - (h * scale) / 2
        return scale, offset_x, offset_y

    def _clamp_pan(self, w, h):
        """Keep the image from being panned/zoomed entirely out of view -
        clamp so at least a margin's worth of it always overlaps the canvas
        (and center it outright if it's smaller than the canvas, same as
        Fit mode would)."""
        cw, ch = self._result_canvas_size()
        scale = self._result_scale
        disp_w, disp_h = w * scale, h * scale
        margin = 40
        if disp_w <= cw:
            self._result_offset_x = (cw - disp_w) / 2
        else:
            self._result_offset_x = min(max(self._result_offset_x, margin - disp_w), cw - margin)
        if disp_h <= ch:
            self._result_offset_y = (ch - disp_h) / 2
        else:
            self._result_offset_y = min(max(self._result_offset_y, margin - disp_h), ch - margin)

    def _update_zoom_label(self):
        self.zoom_label.config(text=f"Zoom: {round(self._result_scale * 100)}%")

    def reset_zoom(self):
        """"Fit" button: snap back to the whole image fitting the canvas."""
        if self.original_image is None:
            return
        self._result_fit = True
        self._render_result()

    def _zoom_result(self, direction, canvas_x, canvas_y):
        """Zoom the result view in/out by one step, keeping the image point
        currently under (canvas_x, canvas_y) fixed on screen ("zoom to
        cursor") rather than zooming around the canvas center, which would
        make the subject drift out from under the mouse on every scroll.
        """
        w, h = self.original_image.size
        fit_scale, fit_offset_x, fit_offset_y = self._fit_result_geom(w, h)
        if self._result_fit:
            # Leaving "Fit" mode - adopt its current geometry as the
            # explicit starting point for the zoom step below.
            self._result_scale, self._result_offset_x, self._result_offset_y = (
                fit_scale, fit_offset_x, fit_offset_y,
            )

        old_scale = self._result_scale
        new_scale = old_scale * (RESULT_ZOOM_STEP if direction > 0 else 1 / RESULT_ZOOM_STEP)
        new_scale = max(fit_scale, min(RESULT_ZOOM_MAX, new_scale))
        if new_scale == old_scale:
            return

        img_x = (canvas_x - self._result_offset_x) / old_scale
        img_y = (canvas_y - self._result_offset_y) / old_scale
        self._result_scale = new_scale
        self._result_offset_x = canvas_x - img_x * new_scale
        self._result_offset_y = canvas_y - img_y * new_scale
        # Snapping back to exactly "Fit" (rather than just very close to it)
        # keeps a later window resize chasing the canvas size again, instead
        # of freezing at whatever scale zooming out happened to land on.
        self._result_fit = new_scale <= fit_scale + 1e-9
        self._clamp_pan(w, h)
        self._render_result()

    def _on_result_wheel(self, event):
        if self.original_image is None:
            return
        if getattr(event, "num", None) == 4:      # X11 scroll-up
            direction = 1
        elif getattr(event, "num", None) == 5:    # X11 scroll-down
            direction = -1
        else:                                     # Windows/macOS <MouseWheel>
            direction = 1 if event.delta > 0 else -1
        self._zoom_result(direction, event.x, event.y)

    def _on_result_configure(self, event):
        if self.signed_dist is None:
            return
        if not self._result_fit and self._result_last_canvas_size is not None:
            # Manual zoom: re-center the view on whatever image point was in
            # the middle of the canvas before the resize, so a zoomed-in view
            # doesn't jump toward a corner when the window is resized. "Fit"
            # mode needs no such correction - it recomputes its centered
            # geometry from scratch on every render anyway.
            old_cw, old_ch = self._result_last_canvas_size
            old_scale = self._result_scale
            center_img_x = (old_cw / 2 - self._result_offset_x) / old_scale
            center_img_y = (old_ch / 2 - self._result_offset_y) / old_scale
            self._result_offset_x = event.width / 2 - center_img_x * old_scale
            self._result_offset_y = event.height / 2 - center_img_y * old_scale
        self._render_result()

    def _render_result(self):
        result = self._composited_result()
        canvas = self.result_canvas
        cw, ch = self._result_canvas_size()
        w, h = result.size

        if self._result_fit:
            scale, offset_x, offset_y = self._fit_result_geom(w, h)
            self._result_scale, self._result_offset_x, self._result_offset_y = scale, offset_x, offset_y
        self._clamp_pan(w, h)
        scale, offset_x, offset_y = self._result_scale, self._result_offset_x, self._result_offset_y

        self._result_geom = (offset_x, offset_y, scale)  # read by _canvas_to_image_xy for painting
        self._result_last_canvas_size = (cw, ch)
        self._update_zoom_label()

        # Crop to just the region of the image visible in the canvas before
        # resampling - once zoomed in on a large image, resizing the whole
        # image to its (potentially huge) on-screen size on every render
        # would be wasteful; only the visible slice needs to go through
        # LANCZOS.
        img_x0 = max(0, min(w, int((0 - offset_x) / scale)))
        img_y0 = max(0, min(h, int((0 - offset_y) / scale)))
        img_x1 = max(0, min(w, int(math.ceil((cw - offset_x) / scale))))
        img_y1 = max(0, min(h, int(math.ceil((ch - offset_y) / scale))))

        canvas.delete("all")
        self._brush_cursor_ids = (None, None)  # delete("all") just removed these items too

        if img_x1 > img_x0 and img_y1 > img_y0:
            crop = result.crop((img_x0, img_y0, img_x1, img_y1))
            disp_w = max(1, round((img_x1 - img_x0) * scale))
            disp_h = max(1, round((img_y1 - img_y0) * scale))
            disp = crop.resize((disp_w, disp_h), Image.LANCZOS)
            checker = make_checkerboard((disp_w, disp_h))
            composed = Image.alpha_composite(checker, disp)
            self._result_photo = ImageTk.PhotoImage(composed)
            paste_x = img_x0 * scale + offset_x
            paste_y = img_y0 * scale + offset_y
            canvas.create_image(paste_x, paste_y, image=self._result_photo, anchor="nw")
        else:
            self._result_photo = None  # panned/zoomed entirely out of view - nothing to draw

        if self._last_mouse_xy is not None:
            self._update_brush_cursor(*self._last_mouse_xy)

    # ------------------------------------------------------------- Panning --
    def _on_pan_button_press(self, event):
        if self.original_image is None:
            return
        self._start_pan(event)

    def _start_pan(self, event):
        self._panning = True
        self._pan_last_xy = (event.x, event.y)

    def _drag_pan(self, event):
        self._last_mouse_xy = (event.x, event.y)
        if not self._panning or self._pan_last_xy is None or self.original_image is None:
            return
        if self._result_fit:
            # Any pan breaks out of "Fit" mode - sync explicit scale/offset
            # to what's currently on screen first so the image doesn't jump.
            w, h = self.original_image.size
            self._result_scale, self._result_offset_x, self._result_offset_y = self._fit_result_geom(w, h)
            self._result_fit = False
        dx = event.x - self._pan_last_xy[0]
        dy = event.y - self._pan_last_xy[1]
        self._result_offset_x += dx
        self._result_offset_y += dy
        self._pan_last_xy = (event.x, event.y)
        self._render_result()

    def _end_pan(self, _event=None):
        self._panning = False
        self._pan_last_xy = None

    # ------------------------------------------------------ Brush cursor --
    def _on_result_motion(self, event):
        self._last_mouse_xy = (event.x, event.y)
        self._update_brush_cursor(event.x, event.y)

    def _hide_brush_cursor(self):
        # Guard against firing before result_canvas exists: brush_mode_var's
        # write-trace (registered in _build_ui) can fire as a side effect of
        # _set_brush_enabled(False) setting it to "off" during that same
        # _build_ui call, before the canvas further down in the method has
        # been created yet.
        canvas = getattr(self, "result_canvas", None)
        if canvas is None:
            return
        for item_id in self._brush_cursor_ids:
            if item_id is not None:
                canvas.delete(item_id)
        self._brush_cursor_ids = (None, None)

    def _on_result_leave(self, _event):
        self._last_mouse_xy = None
        self._hide_brush_cursor()

    def _on_brush_setting_change(self, *_args):
        if self._last_mouse_xy is not None:
            self._update_brush_cursor(*self._last_mouse_xy)
        else:
            self._hide_brush_cursor()

    def _update_brush_cursor(self, cx, cy):
        """Draw/move a ring at (cx, cy) showing the brush's actual on-screen
        footprint at the current zoom. Drawn as two concentric ovals (black
        outside a white ring) rather than one colored outline so it stays
        visible over both light and dark image content - a single flat
        color would disappear against a same-colored background."""
        canvas = self.result_canvas
        show = (
            self.brush_mode_var.get() != "off"
            and self.override is not None
            and self._result_geom is not None
        )
        if not show:
            self._hide_brush_cursor()
            return
        _, _, scale = self._result_geom
        r = max(2.0, self.brush_size_var.get() * scale)
        outer = (cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1)
        inner = (cx - r, cy - r, cx + r, cy + r)
        outer_id, inner_id = self._brush_cursor_ids
        if outer_id is None:
            outer_id = canvas.create_oval(*outer, outline="black", width=1)
            inner_id = canvas.create_oval(*inner, outline="white", width=1)
        else:
            canvas.coords(outer_id, *outer)
            canvas.coords(inner_id, *inner)
        self._brush_cursor_ids = (outer_id, inner_id)

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
        if self.override is None or not np.any(self.override):
            return
        # Clearing is just another undoable edit - snapshot the whole array
        # (its bbox is, by definition, the whole array) before zeroing it,
        # rather than special-casing it out of the undo system.
        h, w = self.override.shape
        self._push_undo(0, h, 0, w, self.override.copy())
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
        self._extend_stroke_bbox(y0, y1, x0, x1)

    def _stamp_line(self, p0, p1):
        (x0, y0), (x1, y1) = p0, p1
        dist = math.hypot(x1 - x0, y1 - y0)
        step = max(1.0, self.brush_size_var.get() / 2.0)
        steps = max(1, int(dist / step))
        for i in range(steps + 1):
            t = i / steps
            self._stamp_brush(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    def _on_paint_start(self, event):
        if self.original_image is None:
            return
        if self.brush_mode_var.get() == "off":
            # Left-drag pans instead of painting whenever the brush is off,
            # since nothing else uses left-drag in that state - this avoids
            # needing a modifier key or a dedicated mode switch just to pan.
            self._start_pan(event)
            return
        if self.override is None:
            return
        pt = self._canvas_to_image_xy(event)
        if pt is None:
            return
        self._begin_stroke()
        self._last_paint_xy = pt
        self._stamp_brush(*pt)
        self._render_result()

    def _on_paint_drag(self, event):
        self._last_mouse_xy = (event.x, event.y)
        if self.original_image is None:
            return
        if self.brush_mode_var.get() == "off":
            self._drag_pan(event)
            return
        if self.override is None:
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

    def _on_paint_end(self, event):
        if self.brush_mode_var.get() == "off":
            self._end_pan(event)
            return
        self._last_paint_xy = None
        self._end_stroke()

    # ------------------------------------------------------- Undo / redo --
    def _begin_stroke(self):
        self._stroke_bbox = None
        # This full-array copy is transient - freed the moment the stroke
        # ends - so its memory cost is fine even on a large image. What
        # actually lands on the undo stack (_end_stroke, below) is just the
        # small bounding box the stroke touched, not this whole copy.
        self._stroke_prior_full = self.override.copy()

    def _extend_stroke_bbox(self, y0, y1, x0, x1):
        if self._stroke_bbox is None:
            self._stroke_bbox = [y0, y1, x0, x1]
        else:
            b = self._stroke_bbox
            b[0], b[1] = min(b[0], y0), max(b[1], y1)
            b[2], b[3] = min(b[2], x0), max(b[3], x1)

    def _end_stroke(self):
        prior_full = self._stroke_prior_full
        bbox = self._stroke_bbox
        self._stroke_prior_full = None
        self._stroke_bbox = None
        if prior_full is None or bbox is None or self.override is None:
            return  # stroke never actually stamped anything
        y0, y1, x0, x1 = bbox
        before = prior_full[y0:y1, x0:x1]
        after = self.override[y0:y1, x0:x1]
        if np.array_equal(before, after):
            return  # no-op stroke (e.g. a click that landed on an already-painted area)
        self._push_undo(y0, y1, x0, x1, before.copy())

    def _push_undo(self, y0, y1, x0, x1, before):
        self.undo_stack.append((y0, y1, x0, x1, before))
        if len(self.undo_stack) > UNDO_STACK_MAX:
            self.undo_stack.pop(0)
        self.redo_stack.clear()  # a fresh edit invalidates whatever was available to redo
        self._update_undo_redo_buttons()

    def undo_touchup(self):
        if not self.undo_stack or self.override is None:
            return
        y0, y1, x0, x1, before = self.undo_stack.pop()
        after = self.override[y0:y1, x0:x1].copy()
        self.redo_stack.append((y0, y1, x0, x1, after))
        if len(self.redo_stack) > UNDO_STACK_MAX:
            self.redo_stack.pop(0)
        self.override[y0:y1, x0:x1] = before
        self._update_undo_redo_buttons()
        self._render_result()

    def redo_touchup(self):
        if not self.redo_stack or self.override is None:
            return
        y0, y1, x0, x1, after = self.redo_stack.pop()
        before = self.override[y0:y1, x0:x1].copy()
        self.undo_stack.append((y0, y1, x0, x1, before))
        if len(self.undo_stack) > UNDO_STACK_MAX:
            self.undo_stack.pop(0)
        self.override[y0:y1, x0:x1] = after
        self._update_undo_redo_buttons()
        self._render_result()

    def _reset_undo_redo(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self):
        self.undo_btn.state(["!disabled"] if self.undo_stack else ["disabled"])
        self.redo_btn.state(["!disabled"] if self.redo_stack else ["disabled"])

    def save_image(self):
        if self.signed_dist is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save result",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialdir=self.config.get("last_save_dir", ""),
        )
        if not path:
            return
        result = self._composited_result()
        try:
            result.save(path)
            self.status_var.set(f"Saved to {path}")
            self.config["last_save_dir"] = str(Path(path).resolve().parent)
            save_config(self.config)
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
