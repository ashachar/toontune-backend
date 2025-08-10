# Potrace + ImageMagick (Outline Tracing) — Install & End-to-End Examples

These instructions convert a **raster PNG** doodle into a crisp **SVG** using:
- **ImageMagick** for preprocessing (grayscale, normalize, threshold, denoise)
- **Potrace** for **outline tracing** (filled vector shapes around strokes)

Works on macOS, Linux, and Windows.

---

## 0) Prerequisites & Install

### macOS (Homebrew)
```bash
# If you don't have Homebrew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update
brew install imagemagick potrace
# Verify
magick -version   # or: convert -version (older ImageMagick)
potrace -v
```

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y imagemagick potrace
# Verify
convert -version  # or: magick -version on newer builds
potrace -v
```

### Windows (winget or Chocolatey)

```powershell
# Option A: winget (Windows 10/11)
winget install ImageMagick.ImageMagick
winget install Potrace.Potrace

# Option B: Chocolatey (Admin PowerShell)
choco install imagemagick -y
choco install potrace -y

# Verify (PowerShell or CMD)
magick -version         # ImageMagick 7+ uses "magick" prefix
potrace -v
```

> **Note:** On ImageMagick 7+, the command is `magick`. On older versions it's `convert`. In the examples below, we show **both** forms—use whichever works on your machine.

---

## 1) Minimal Example — One PNG → One SVG

> Input: `./uploads/hand.png`
> Output: `./uploads/hand.svg`

### A) Using `magick` (ImageMagick 7+)

```bash
# 1) Preprocess to clean, high-contrast 1-bit PGM (Potrace's favorite)
magick ./input/doodle.png \
  -colorspace Gray \
  -auto-level \
  -blur 0x0.4 \
  -threshold 60% \
  -despeckle -despeckle \
  ./out/doodle.pgm

# 2) Trace with Potrace → SVG (outline/fills)
potrace ./out/doodle.pgm \
  -s \                       # output SVG
  -t 2 \                     # ignore tiny specks (turdsize)
  -a 1.0 \                   # curve optimization aggressiveness (higher = smoother)
  -O 0.2 \                   # path optimization tolerance (lower = more detail)
  -o ./out/doodle.svg
```

### B) Using `convert` (older ImageMagick)

```bash
# 1) Preprocess
convert ./input/doodle.png \
  -colorspace Gray \
  -auto-level \
  -blur 0x0.4 \
  -threshold 60% \
  -despeckle -despeckle \
  ./out/doodle.pgm

# 2) Trace
potrace ./out/doodle.pgm -s -t 2 -a 1.0 -O 0.2 -o ./out/doodle.svg
```

**Parameter tips for the "one image" flow**

* If lines are too **thin** or broken → **lower** `-threshold` (e.g., `55%`).
* If tiny dust remains → **increase** `-t` (e.g., `-t 4`).
* If curves seem too stiff → **raise** `-a` (e.g., `1.2`) and **lower** `-O` (e.g., `0.1`).
* If you see JPEG noise → keep the small `-blur 0x0.4`; you can go to `0x0.6` if needed.

---

## 2) One-Shot Bash Script (Single Image)

> **Save** as `png_to_svg.sh`, then run:
> `bash png_to_svg.sh ./input/doodle.png ./out/doodle.svg`

```bash
#!/usr/bin/env bash
set -euo pipefail

INPNG="${1:-}"
OUTSVG="${2:-}"

if [[ -z "$INPNG" || -z "$OUTSVG" ]]; then
  echo "Usage: bash png_to_svg.sh <input.png> <output.svg>"
  exit 1
fi

mkdir -p "$(dirname "$OUTSVG")"
TMPPGM="${OUTSVG%.svg}.pgm"

# Use ImageMagick 7's 'magick' if available, else fallback to 'convert'
if command -v magick >/dev/null 2>&1; then
  magick "$INPNG" -colorspace Gray -auto-level -blur 0x0.4 -threshold 60% -despeckle -despeckle "$TMPPGM"
elif command -v convert >/dev/null 2>&1; then
  convert "$INPNG" -colorspace Gray -auto-level -blur 0x0.4 -threshold 60% -despeckle -despeckle "$TMPPGM"
else
  echo "ImageMagick not found. Install it first."
  exit 1
fi

# Potrace to SVG
potrace "$TMPPGM" -s -t 2 -a 1.0 -O 0.2 -o "$OUTSVG" -q
echo "Wrote: $OUTSVG"
```

---

## 3) Batch Convert a Folder (100+ PNG/JPG → SVG)

> **Input folder:** `./input` (png/jpg/jpeg/tif)
> **Output folder:** `./out_svg`

### A) Bash (macOS/Linux, or Git Bash on Windows)

```bash
#!/usr/bin/env bash
set -euo pipefail

IN="./input"
OUT="./out_svg"
TMP="./_tmp_pgm"
mkdir -p "$OUT" "$TMP"

# Choose IM command
IMCMD=""
if command -v magick >/dev/null 2>&1; then IMCMD="magick"; fi
if [[ -z "$IMCMD" && "$(command -v convert || true)" != "" ]]; then IMCMD="convert"; fi
if [[ -z "$IMCMD" ]]; then echo "ImageMagick not found."; exit 1; fi
if ! command -v potrace >/dev/null 2>&1; then echo "Potrace not found."; exit 1; fi

shopt -s nullglob
for f in "$IN"/*.{png,PNG,jpg,JPG,jpeg,JPEG,tif,TIF,tiff,TIFF}; do
  name="$(basename "$f")"; base="${name%.*}"
  echo ">> $name"
  "$IMCMD" "$f" -colorspace Gray -auto-level -blur 0x0.4 -threshold 60% -despeckle -despeckle "$TMP/$base.pgm"
  potrace "$TMP/$base.pgm" -s -t 2 -a 1.0 -O 0.2 -o "$OUT/$base.svg" -q
done

echo "All done → $OUT"
```

### B) Python (Cross-platform, single file)

```python
#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys
from pathlib import Path

def which(cmds):
    for c in cmds:
        p = shutil.which(c)
        if p: return p
    return None

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print("ERROR:", " ".join(cmd))
        print(p.stderr)
        sys.exit(1)
    return p.stdout

def main():
    ap = argparse.ArgumentParser(description="Batch PNG/JPG → SVG using ImageMagick + Potrace")
    ap.add_argument("--in", dest="indir", default="./input")
    ap.add_argument("--out", dest="outdir", default="./out_svg")
    ap.add_argument("--threshold", type=float, default=0.60, help="0..1 (e.g., 0.60 = 60%)")
    ap.add_argument("--despeckle", type=int, default=2, help="number of -despeckle passes")
    ap.add_argument("--turdsize", type=int, default=2, help="Potrace: ignore tiny specks (px)")
    ap.add_argument("--alphamax", type=float, default=1.0, help="Potrace curve fit aggressiveness")
    ap.add_argument("--opttolerance", type=float, default=0.2, help="Potrace path optimization tol.")
    args = ap.parse_args()

    magick = which(["magick", "convert"])
    potrace = which(["potrace"])
    if not magick:  sys.exit("ImageMagick not found.")
    if not potrace: sys.exit("Potrace not found.")

    os.makedirs(args.outdir, exist_ok=True)
    tmpdir = Path(args.outdir) / "_tmp_pgm"
    tmpdir.mkdir(exist_ok=True)

    exts = (".png",".jpg",".jpeg",".tif",".tiff",".bmp")
    files = [p for p in Path(args.indir).glob("*") if p.suffix.lower() in exts]
    if not files:
        sys.exit(f"No raster files found in {args.indir}")

    for i, src in enumerate(files, 1):
        base = src.stem
        tmp_pgm = tmpdir / f"{base}.pgm"
        out_svg = Path(args.outdir) / f"{base}.svg"
        print(f"[{i}/{len(files)}] {src.name} → {out_svg.name}")

        # Build IM command
        cmd = [magick, str(src),
               "-colorspace","Gray","-auto-level",
               "-blur","0x0.4",
               "-threshold", f"{int(args.threshold*100)}%"]
        for _ in range(args.despeckle):
            cmd += ["-despeckle"]
        cmd += [str(tmp_pgm)]
        run(cmd)

        # Potrace
        cmd = [potrace, str(tmp_pgm),
               "-s",
               "-t", str(args.turdsize),
               "-a", str(args.alphamax),
               "-O", str(args.opttolerance),
               "-o", str(out_svg),
               "-q"]
        run(cmd)

    print("Done.")

if __name__ == "__main__":
    main()
```

Run:

```bash
python3 batch_trace.py --in ./input --out ./out_svg --threshold 0.60 --despeckle 2 --turdsize 2 --alphamax 1.0 --opttolerance 0.2
```

---

## 4) Quality & Accuracy Checklist

* **Threshold tuning:**

  * Dark/thick lines → `-threshold 55%` (lower value includes more ink).
  * Faint/pencil lines → `-threshold 65–75%` (higher value excludes paper texture).
* **Speckle removal:** increase `-despeckle` passes or set Potrace `-t 4–8` for dusty scans.
* **Smoothness vs. fidelity:**

  * Smoother curves → raise `-a` (e.g., `1.2`) and lower `-O` (e.g., `0.1`).
  * Maximum detail → lower `-a` (e.g., `0.8`) and raise `-O` (e.g., `0.3`).
* **Node count (file size/perf):** nudge `-O` up slightly (e.g., `0.25`) to reduce points.
* **Output viewbox:** Potrace preserves pixel dimensions; you can rescale in an editor if needed.

---

## 5) Troubleshooting

* **"magick/convert not found"** → reinstall ImageMagick; ensure PATH is updated. On Windows, open a new terminal after install.
* **"no PGM output / permission denied"** → ensure `./out`/`./_tmp_pgm` exist and you have write permission.
* **Jagged edges** → keep the small `-blur`; try `-blur 0x0.6` for heavy JPEG artifacts.
* **Broken lines after tracing** → reduce threshold (e.g., `55%`) or fewer `-despeckle` passes.

---

## 6) Quick Recap

1. **Install** ImageMagick + Potrace.
2. **Preprocess** PNG → clean 1-bit PGM with grayscale, auto-level, small blur, threshold, despeckle.
3. **Trace** with Potrace to SVG using `-s -t 2 -a 1.0 -O 0.2`.
4. **Tune** threshold/turdsize/alphamax/opttolerance for your doodles.
5. **Batch** the whole folder with the bash or Python script.

Happy vectorizing!