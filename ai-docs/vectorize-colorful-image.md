# vtracer — High-Quality CLI Vectorization for Colored Images

`vtracer` is a modern, open-source command-line tool (written in Rust) that converts raster images (PNG/JPG/etc.) into vector graphics (SVG) while preserving **color regions** and producing smooth, clean paths.

It's excellent for:
- Colored doodles
- Flat illustrations
- Logos and icons
- Scanned artwork with clear regions

---

## 1) Install vtracer

### macOS (Homebrew)
```bash
brew install vtracer
```

### Linux (Cargo / Rust)

```bash
# Install Rust & Cargo if you don't have them:
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install vtracer
```

### Windows

```powershell
# If you have Rust & Cargo installed:
cargo install vtracer

# Or download prebuilt binaries:
# https://github.com/visioncortex/vtracer/releases
```

---

## 2) Basic Usage

```bash
vtracer --input ./input/doodle.png --output ./out/doodle.svg
```

* `--input` : path to your raster file (PNG, JPG, BMP, etc.)
* `--output`: path for the resulting SVG file

---

## 3) Recommended Settings for Colored Doodles

For **flat-color** doodles or cartoons:

```bash
vtracer \
  --input ./input/doodle.png \
  --output ./out/doodle.svg \
  --mode polygon \
  --hierarchical \
  --filter-speckle 4 \
  --color-precision 18 \
  --paths 512
```

### Explanation of Flags:

* `--mode polygon` → outputs polygon shapes (good for flat fills) instead of curves (good for smooth shading).
* `--hierarchical` → nests paths logically for cleaner grouping.
* `--filter-speckle 4` → removes small isolated color specks (adjust for noise level).
* `--color-precision 18` → controls color quantization; higher = more colors preserved.
* `--paths 512` → maximum number of paths; higher = more detail, larger file size.

---

## 4) Preprocessing for Best Results

If your image has **anti-aliasing or gradients** and you want cleaner regions:

```bash
# Reduce to a limited number of colors, no dithering:
magick ./input/doodle.png -dither None -colors 24 ./work/flat.png

# Then run vtracer:
vtracer --input ./work/flat.png --output ./out/doodle.svg --mode polygon --hierarchical --filter-speckle 4 --color-precision 18 --paths 512
```

---

## 5) Batch Processing a Folder

```bash
IN=./input
OUT=./out_vtracer
mkdir -p "$OUT"

for f in "$IN"/*.{png,PNG,jpg,JPG,jpeg,JPEG}; do
  [ -e "$f" ] || continue
  base="${f##*/}"; name="${base%.*}"
  vtracer --input "$f" \
          --output "$OUT/$name.svg" \
          --mode polygon \
          --hierarchical \
          --filter-speckle 4 \
          --color-precision 18 \
          --paths 512
done
```

---

## 6) Tips for Tuning

| If…                                | Try This                                        |
| ---------------------------------- | ----------------------------------------------- |
| Colors merging together            | Increase `--color-precision` (e.g., 20–24)      |
| Too many tiny blobs                | Increase `--filter-speckle` (e.g., 6–10)        |
| Output too heavy / too many points | Reduce `--paths` (e.g., 256)                    |
| Jagged edges                       | Pre-blur slightly before tracing: `-blur 0x0.4` |
| Want smoother shading regions      | Use `--mode curve` instead of `polygon`         |

---

## 7) Transparency Handling

* **vtracer** supports transparent PNG input.
* Fully transparent pixels are ignored.
* Semi-transparent regions will be approximated as solid paths unless preprocessed into flat transparency shapes.
* To preserve alpha exactly, you can output separate mask layers and apply `fill-opacity` in post-processing.

---

## 8) Example Before/After

**Before:** doodle.png — 2000×2000 px raster
**After:** doodle.svg — ~30 KB vector with flat colored polygons, infinitely scalable

---

## 9) Resources

* GitHub: [https://github.com/visioncortex/vtracer](https://github.com/visioncortex/vtracer)
* Issues & feature requests: [GitHub Issues](https://github.com/visioncortex/vtracer/issues)

---

## Summary

For **colored doodles**:

1. Pre-quantize if your art has AA/gradients you want to remove.
2. Run `vtracer` with `--mode polygon --hierarchical --color-precision 18 --filter-speckle 4`.
3. Batch process for speed.
4. Tune `--color-precision` and `--paths` for your art style.

`vtracer` gives you a **fast, free, and high-quality** color-preserving SVG — ideal for automation pipelines.

---

## Additional Notes

If you want, a workflow that keeps transparency in gradients can be added by post-processing vtracer's SVG into `<linearGradient>` or `<radialGradient>` stops so fades remain smooth instead of stepped.