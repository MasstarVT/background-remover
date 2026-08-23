# Background Remover

A small cross-platform desktop app (Windows / Linux / macOS) that removes
the background from a photo. It shows a side-by-side before/after preview,
a **Removal Strength** slider, a **Model** picker, and a manual **touch-up
brush** for anything the AI gets wrong.

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

**Touch-up brush**: for anything the slider can't fix - a background blob
the AI is genuinely confident is foreground (common when something bright
sits right behind/against the subject) - pick **Keep** or **Remove** and
paint directly over the preview. Painted areas override the slider with a
soft edge, and **Save Result** bakes in both.

**Model picker**: four models are available, trading speed for accuracy:

| Model | Speed | Notes |
|---|---|---|
| General (u2net) | fast (~1s), ~180MB download | good all-purpose default |
| General v2 (isnet-general-use) | fast | newer, often sharper edges than u2net |
| Human / Portrait (u2net_human_seg) | fast | tuned for people |
| High Quality (birefnet, slower) | slow (~5-10s/image on CPU), ~970MB download | 2024 state-of-the-art model, noticeably sharper edges in testing |

None of these (including the AI-anime-specific `isnet-anime`, which was
tested and dropped - it output near-zero confidence across the entire
frame on real anime art) can resolve a genuine *composition* ambiguity,
like a moon drawn directly behind a character so it reads as part of the
same visual object. That's what the touch-up brush is for, not a model
quality issue.

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

## Usage

1. Click **Open Image...** and pick a photo.
2. Wait for the AI pass to finish (status bar shows progress). Try a
   different **Model** if the default cutout isn't a good starting point.
3. Drag **Removal Strength** while watching the preview (checkerboard =
   transparent) to dial in the cutout boundary.
4. For anything left over the slider can't fix (or anything it took away
   that it shouldn't have), pick **Keep** or **Remove** under Touch-up
   brush and paint directly on the preview.
5. Click **Save Result...** to export as a PNG with transparency.
