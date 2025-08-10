# Color Vectorization Module

High-quality vectorization of colored raster images using **vtracer**, preserving color regions and producing smooth, clean SVG paths.

## Features

- **Color Preservation**: Maintains distinct color regions from the original image
- **Multiple Presets**: Optimized settings for different image types (doodles, logos, photos)
- **Batch Processing**: Process multiple images with parallel execution
- **Preprocessing**: Optional image preprocessing for better results
- **Customizable**: Fine-tune vectorization parameters for specific needs

## Installation

### Quick Install
```bash
./install_vtracer.sh
```

### Manual Installation
```bash
# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install vtracer
cargo install vtracer
```

### Optional: ImageMagick (for preprocessing)
```bash
# macOS
brew install imagemagick

# Ubuntu/Debian
sudo apt-get install imagemagick

# RHEL/CentOS
sudo yum install ImageMagick
```

## Usage

### Single Image Vectorization

```bash
# Basic usage with preset
python3 vectorize_color.py input.png output.svg --preset doodle

# Custom parameters
python3 vectorize_color.py input.png output.svg \
    --mode polygon \
    --color-precision 7 \
    --filter-speckle 5

# With preprocessing
python3 vectorize_color.py input.png output.svg \
    --preset doodle \
    --preprocess \
    --reduce-colors 24 \
    --blur 0.4
```

### Batch Processing

```bash
# Process entire directory
python3 batch_vectorize.py ./input_folder ./output_folder --preset doodle

# Parallel processing with 8 workers
python3 batch_vectorize.py ./input ./output --parallel 8

# Preserve directory structure
python3 batch_vectorize.py ./input ./output --preserve-structure

# Skip existing files
python3 batch_vectorize.py ./input ./output --skip-existing
```

## Presets

| Preset | Description | Best For |
|--------|-------------|----------|
| `doodle` | Flat colors, clean edges | Cartoons, drawings, doodles |
| `illustration` | More detail, balanced | Detailed illustrations |
| `logo` | Clean, minimal paths | Logos, icons, simple graphics |
| `detailed` | Maximum detail with splines | Complex artwork |
| `simplified` | Minimal output, heavy filtering | Simple shapes, reduced file size |
| `photo` | Built-in vtracer photo preset | Photographs |
| `poster` | Built-in poster style | Poster-style images |
| `bw` | Built-in black & white | Monochrome images |

## Parameters

### Vectorization Parameters

- `--mode`: Curve fitting mode (`pixel`, `polygon`, `spline`)
  - `polygon`: Best for flat colors and sharp edges
  - `spline`: Smoother curves, better for gradients
  - `pixel`: Pixel-perfect accuracy

- `--color-precision`: Significant bits in RGB channel (1-8)
  - Lower = fewer colors, smaller file
  - Higher = more colors, larger file

- `--filter-speckle`: Remove patches smaller than N pixels
  - Higher values = cleaner output, less detail

- `--path-precision`: Decimal places in SVG paths (1-5)
  - Lower = smaller file size
  - Higher = more accurate paths

- `--corner-threshold`: Minimum angle for corners (degrees)
  - Lower = more corners detected
  - Higher = smoother curves

- `--hierarchical`: Path organization (`stacked` or `cutout`)
  - `stacked`: Nested paths (default)
  - `cutout`: Non-overlapping paths

### Preprocessing Parameters

- `--preprocess`: Enable preprocessing
- `--reduce-colors N`: Reduce to N colors before vectorization
- `--blur N`: Apply blur with radius N (smooths edges)

## Examples

### Colored Doodle
```bash
python3 vectorize_color.py doodle.png doodle.svg --preset doodle
```

### Clean Logo
```bash
python3 vectorize_color.py logo.png logo.svg \
    --preset logo \
    --preprocess \
    --reduce-colors 8
```

### Batch Convert Folder
```bash
python3 batch_vectorize.py ./drawings ./vectors \
    --preset illustration \
    --parallel 4 \
    --skip-existing
```

### Custom High-Quality
```bash
python3 vectorize_color.py art.png art.svg \
    --mode spline \
    --color-precision 8 \
    --path-precision 4 \
    --filter-speckle 2
```

## Testing

Run the comprehensive test suite:
```bash
python3 test_vectorize.py
```

This will:
- Check dependencies
- Test all presets
- Test custom configurations
- Test batch processing
- Generate sample outputs in `uploads/vectorized/tests/`

## Tips for Best Results

1. **For flat-color artwork**: Use `polygon` mode with `doodle` or `illustration` preset
2. **For photographs**: Use the `photo` preset or `spline` mode
3. **If colors merge**: Increase `--color-precision`
4. **If output is noisy**: Increase `--filter-speckle`
5. **If file is too large**: Reduce `--path-precision` or use `--reduce-colors`
6. **For cleaner edges**: Use `--blur 0.3-0.5` with preprocessing

## File Size Optimization

To reduce SVG file size:
1. Use lower `--color-precision` (4-6)
2. Use lower `--path-precision` (1-2)
3. Increase `--filter-speckle` (10-20)
4. Preprocess with `--reduce-colors` (8-24)
5. Use `simplified` preset for minimal output

## Troubleshooting

### vtracer not found
- Run `./install_vtracer.sh`
- Or manually install: `cargo install vtracer`
- Ensure `~/.cargo/bin` is in your PATH

### Large output files
- Reduce color precision: `--color-precision 4`
- Filter more aggressively: `--filter-speckle 10`
- Preprocess to reduce colors: `--reduce-colors 16`

### Lost details
- Increase color precision: `--color-precision 8`
- Reduce filtering: `--filter-speckle 2`
- Use `spline` mode for smoother curves

### Jagged edges
- Apply preprocessing blur: `--blur 0.5`
- Use `spline` mode instead of `polygon`
- Increase `--segment-length` for smoother curves

## Performance

- Parallel processing scales well up to CPU core count
- Typical processing time: 0.5-2 seconds per image
- Memory usage scales with image size and complexity
- SSD recommended for batch processing large datasets

## Output Format

Generated SVG files are:
- Infinitely scalable without quality loss
- Compatible with all modern browsers
- Editable in vector graphics software (Illustrator, Inkscape, etc.)
- Optimized for web with proper structure and grouping

## Dependencies

- **Required**: Python 3.6+, vtracer
- **Optional**: ImageMagick (for preprocessing)
- **Python packages**: None (uses only standard library)

## License

This module is part of the ToonTune backend system.