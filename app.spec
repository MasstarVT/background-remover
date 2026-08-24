# PyInstaller spec for Background Remover.
#
# Build (after `pip install -r requirements-build.txt`):
#   pyinstaller app.spec
#
# Output goes to dist/BackgroundRemover/ (a folder you can zip and hand to
# someone - see below for why this is a folder build, not --onefile).
#
# This spec is platform-agnostic (no hardcoded path separators, no OS-
# specific assumptions) and is used as-is for both the Windows and Linux
# builds in .github/workflows/build.yml.
#
# Notes on the --collect-all calls below: onnxruntime, numpy and scipy all
# ship compiled extension modules and/or dynamically-discovered submodules
# that PyInstaller's static import analysis can miss on its own (this was
# verified by trial build - a naive `pyinstaller bg_remover_gui.py` fails at
# runtime with "DLL load failed" / ModuleNotFoundError for onnxruntime's
# capi and scipy's ndimage backends). tkinterdnd2 ships a prebuilt native Tcl
# extension (the `tkdnd` folder, one subfolder per OS/arch) that is loaded
# from a data path at runtime, not imported as Python - collect_data is
# needed for it specifically, not collect_all's import-hook piece.
import sys

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

for pkg in ("onnxruntime", "rembg", "numpy", "scipy", "tkinterdnd2", "PIL"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    ["bg_remover_gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BackgroundRemover",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app - no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# --onedir build: onnxruntime/numpy/scipy are large (the birefnet-massive
# model download alone is ~930MB, and the onnxruntime/scipy binaries
# themselves add a few hundred MB more) - a --onefile build has to unpack
# all of that to a fresh temp directory on *every* launch, which was tested
# here and adds a very noticeable multi-second delay before the window even
# appears, versus a folder build that just runs its exe directly. A folder
# a user can unzip and double-click is a perfectly normal way to ship a
# desktop app on both Windows and Linux, so onedir is what this spec
# produces.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BackgroundRemover",
)
