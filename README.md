# Background Remover

A small cross-platform desktop app (Windows / Linux / macOS) that removes
the background from a photo. It shows a side-by-side before/after preview,
a **Removal Strength** slider, an **Edge Softness** slider, a **Model**
picker, and a manual **touch-up brush** for anything the AI gets wrong.

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
soft edge, and **Save Result** bakes in both.

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

1. Click **Open Image...** and pick a photo.
2. Wait for the AI pass to finish (status bar shows progress). Try a
   different **Model** if the default cutout isn't a good starting point.
3. Drag **Removal Strength** while watching the preview (checkerboard =
   transparent) to dial in the cutout boundary. Use **Edge Softness** to
   make that boundary crisper or blurrier.
4. For anything left over the slider can't fix (or anything it took away
   that it shouldn't have), pick **Keep** or **Remove** under Touch-up
   brush and paint directly on the preview.
5. Click **Save Result...** to export as a PNG with transparency.

Your last-used model and the folders you last opened/saved from are
remembered between runs, in a small config file at
`~/.background_remover_config.json` (`%USERPROFILE%` on Windows). Delete
it any time to reset to the defaults.
