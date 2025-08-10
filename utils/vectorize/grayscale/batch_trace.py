#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys
from pathlib import Path

def which(cmds):
    for c in cmds:
        p = shutil.which(c)
        if p:
            return p
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
    if not magick:
        sys.exit("ImageMagick not found.")
    if not potrace:
        sys.exit("Potrace not found.")

    os.makedirs(args.outdir, exist_ok=True)
    tmpdir = Path(args.outdir) / "_tmp_pgm"
    tmpdir.mkdir(exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = [p for p in Path(args.indir).glob("*") if p.suffix.lower() in exts]
    if not files:
        sys.exit(f"No raster files found in {args.indir}")

    for i, src in enumerate(files, 1):
        base = src.stem
        tmp_pgm = tmpdir / f"{base}.pgm"
        out_svg = Path(args.outdir) / f"{base}.svg"
        print(f"[{i}/{len(files)}] {src.name} → {out_svg.name}")

        # ImageMagick preprocess
        cmd = [magick, str(src),
               "-colorspace", "Gray", "-auto-level",
               "-blur", "0x0.4",
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
