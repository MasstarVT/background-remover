# Background Remover

A small cross-platform desktop app (Windows / Linux / macOS) that removes
the background from a photo. It shows a side-by-side before/after preview,
a **Removal Strength** slider, an **Edge Softness** slider, a **Model**
picker, and a manual **touch-up brush** for anything the AI gets wrong. You
can drag and drop an image onto the window to load it, replace the
transparent background with a solid color or another image at save time,
and export the cutout mask on its own.

## How it works

The AI model ([rembg](https://github.com/danielgatis/rembg)) runs once per
image and produces a cutout mask, which is converted to a signed distance
field (how many pixels each point is from the cutout edge) and cached.
Moving the slider does NOT re-run the model - it just grows or shrinks the
cutout boundary by pixels read off that cached field, so the preview
updates instantly:

- Low strength → the kept area is grown outward, so more of the image
  survives (safer, may keep a thin edge of background).
- High strength → the kept area is shrunk inward (aggressive, may eat into
  the subject's edges).

**Edge Softness** controls how gradual that boundary transition is, from a
crisp 1px edge up to a soft 15px blend (default 3px) - useful for matching
the cutout to how sharp or soft the original photo's focus is.

**Touch-up brush**: for anything the slider can't fix - a background blob
the AI is genuinely confident is foreground (common when something bright
sits right behind/against the subject) - pick **Keep** or **Remove** and
paint directly over the preview. Painted areas override the slider with a
soft edge, and **Save Result** bakes in both. A ring at the cursor shows
the brush's actual size (it scales with zoom). Made a mistake? **Ctrl+Z**
undoes the last stroke, **Ctrl+Y** (or Ctrl+Shift+Z) redoes it - or use the
**Undo**/**Redo** buttons. **Clear Touch-ups** wipes everything painted so
far (also undoable).

**Zoom & pan** on the result preview: scroll the mouse wheel to zoom in/out
around the cursor (so whatever you're pointing at stays put), up to 1600%,
down to the point where the whole image fits the panel. Click **Fit** to
snap back to that fitted view. To pan around while zoomed in, drag with the
**middle mouse button** (works no matter what the brush is set to), or
left-click-drag while the touch-up brush is **Off** (left-drag paints
instead whenever a brush mode is active).

**Background replacement**: by default **Save Result...** exports a
transparent PNG, same as before. Use the **Background** dropdown next to it
to instead flatten the cutout onto a **Solid Color** (opens a color picker)
or another **Image** (opens a file picker) before saving - **Choose...**
re-opens that picker if you want to change the color/image without
switching the dropdown away and back. A background image is scaled up just
enough to cover the subject's full frame and then center-cropped to match
its exact dimensions (like CSS `background-size: cover`) rather than being
letterboxed or stretched. Once a background is flattened in, the alpha
channel is no longer meaningful, so the save dialog also offers **JPEG** as
an option alongside PNG (PNG stays the default, and is the only option for
"Transparent").

**Export Mask...** saves just the current cutout mask - after the slider
and any touch-ups, same as what Save Result uses - as a standalone
grayscale PNG (0 = fully removed, 255 = fully kept), for anyone who wants
to bring the mask into another editing tool.

**Model picker**: six models are available, trading speed for accuracy.
Default is `birefnet-massive` - in testing against a real illustration
where a background object (a moon) was drawn visually merged with the
subject, it was the *only* one of six models that correctly separated
them, at every strength setting.

| Model | Speed | Notes |
|---|---|---|
| High Quality (birefnet-massive, slower) | slow (~5-10s/image on CPU), ~930MB download | **default** - best result in testing, see above |
| High Quality (birefnet-general, slower) | slow, ~930MB download | very good edges; didn't separate the moon case above |
| High Quality (bria-rmbg, slower) | slow, ~980MB download | well-regarded commercial-grade model; also didn't separate the moon case |
| General (u2net) | fast (~1s), ~180MB download | good default when you don't need the slow models |
| General v2 (isnet-general-use) | fast | newer, often sharper edges than u2net |
| Human / Portrait (u2net_human_seg) | fast | tuned for people |

An AI-anime-specific model (`isnet-anime`) was also tested and dropped -
it output near-zero confidence across the entire frame on real anime art.

Even the best model here can't always resolve a genuine *composition*
ambiguity - something visually merged with the subject reads as part of
the same object to any generic segmentation model. That's what the
touch-up brush is for.

## Download a pre-built executable

Don't want to install Python? Grab a standalone build for Windows or Linux
from the [Releases page](https://github.com/MasstarVT/background-remover/releases)
(built automatically by [GitHub Actions](.github/workflows/build.yml) from a
version tag) - or from that workflow's
[Actions run artifacts](https://github.com/MasstarVT/background-remover/actions/workflows/build.yml)
for the latest unreleased build. Unzip/untar it anywhere and run
`BackgroundRemover.exe` (Windows) or `./BackgroundRemover` (Linux) - it's a
folder, not a single file, so keep the whole folder together.

**First launch still needs internet.** The app doesn't bundle the AI model -
`rembg` downloads it the first time it's used and caches it locally (see
"On first use of a given model" below). The default model is ~930MB, so the
very first run will pause on "Removing background..." while that downloads;
every run after that is offline and instant to start.

### Building it yourself

```bash
pip install -r requirements.txt
pip install -r requirements-build.txt
pyinstaller app.spec
```

The build lands in `dist/BackgroundRemover/` - run the executable inside
that folder. `app.spec` is set up to pull in onnxruntime/numpy/scipy's and
tkinterdnd2's non-obvious runtime pieces (PyInstaller's static import
analysis misses some of these on its own); it works as-is on both Windows
and Linux. It's a folder build rather than `--onefile` on purpose - a
onefile build has to unpack this app's fairly large dependencies (onnxruntime,
scipy) to a fresh temp directory on every single launch, which was tested
here and made every startup noticeably slower than just running the exe
directly out of a folder.

## Requirements

- Python 3.9+
- On Linux, Tk isn't always bundled with Python — install it via your
  package manager if `import tkinter` fails:

  ```bash
  # Debian/Ubuntu
  sudo apt install python3-tk

  # Fedora
  sudo dnf install python3-tkinter

  # Arch
  sudo pacman -S tk
  ```

- `tkinterdnd2` (in requirements.txt) is optional - it enables dragging an
  image file onto the window to load it. If it's missing or fails to
  install, the app still runs fine; you just lose the drag-and-drop
  shortcut and use **Open Image...** instead (a note is printed to the
  console, nothing pops up).

## Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python bg_remover_gui.py
```

On first use of a given model, `rembg` downloads it to a local cache
(`~/.u2net` on Linux/macOS, `%USERPROFILE%\.u2net` on Windows). This only
happens once per model - see the size/speed table above before picking
"High Quality" on a slow connection.

The app starts warming up the selected model in the background as soon as
the window opens (status bar shows "Warming up..." then "Ready."), so that
load time is hidden behind the time you spend picking a file instead of
happening after you've already opened one.

## Usage

1. Click **Open Image...** and pick a photo, or drag an image file onto
   the window.
2. Wait for the AI pass to finish (status bar shows progress). Try a
   different **Model** if the default cutout isn't a good starting point.
3. Drag **Removal Strength** while watching the preview (checkerboard =
   transparent) to dial in the cutout boundary. Use **Edge Softness** to
   make that boundary crisper or blurrier.
4. For anything left over the slider can't fix (or anything it took away
   that it shouldn't have), pick **Keep** or **Remove** under Touch-up
   brush and paint directly on the preview. Scroll to zoom in first for
   precise work near edges; middle-drag to pan around while zoomed in.
5. Leave **Background** set to **Transparent** to export a PNG with
   transparency (the default), or switch it to **Solid Color** / **Image**
   to flatten the cutout onto something else first. Click **Save
   Result...** to export.
6. Click **Export Mask...** any time after the AI pass finishes to save
   just the cutout mask on its own, separate from the color image.

Your last-used model and the folders you last opened/saved from are
remembered between runs, in a small config file at
`~/.background_remover_config.json` (`%USERPROFILE%` on Windows). Delete
it any time to reset to the defaults.
