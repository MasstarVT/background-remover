# Background Remover

A small cross-platform desktop app (Windows / Linux / macOS) that removes
the background from a photo. It shows a side-by-side before/after preview
and a **Removal Strength** slider you can drag to control exactly how much
gets cut, before saving.

- Low strength → keeps more of the image (safer, some background may remain).
- High strength → cuts more aggressively (may eat into the subject's edges).

The AI model ([rembg](https://github.com/danielgatis/rembg), a U²-Net based
background remover) runs once per image. Moving the slider just
re-thresholds the cached result, so the preview updates instantly with no
lag.

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

On first use, `rembg` downloads its AI model (~180 MB) to a local cache
(`~/.u2net` on Linux/macOS, `%USERPROFILE%\.u2net` on Windows). This only
happens once.

## Usage

1. Click **Open Image...** and pick a photo.
2. Wait a moment for the AI pass to finish (status bar shows progress).
3. Drag the **Removal Strength** slider while watching the preview
   (checkerboard = transparent) until only the subject remains.
4. Click **Save Result...** to export as a PNG with transparency.
