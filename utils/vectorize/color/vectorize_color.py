#!/usr/bin/env python3
"""
High-quality color vectorization using vtracer
Preserves color regions and produces smooth, clean SVG paths
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import json
import tempfile
from typing import Optional, Dict, Any

class ColorVectorizer:
    """Vectorizes colored raster images to SVG using vtracer"""
    
    DEFAULT_SETTINGS = {
        'mode': 'polygon',           # polygon for flat colors, spline for smooth curves
        'hierarchical': 'stacked',   # stacked (default) or cutout
        'filter_speckle': 4,        # remove small isolated color specks
        'color_precision': 6,       # significant bits in RGB channel (default: 6)
        'path_precision': 3,        # decimal places in path strings
        'corner_threshold': 60,     # minimum angle for corners
        'segment_length': 5.0,      # segment length for subdivision
    }
    
    PRESET_CONFIGS = {
        'doodle': {
            'mode': 'polygon',
            'hierarchical': 'stacked',
            'filter_speckle': 4,
            'color_precision': 6,
            'path_precision': 2,
            'corner_threshold': 60,
        },
        'illustration': {
            'mode': 'polygon',
            'hierarchical': 'stacked',
            'filter_speckle': 6,
            'color_precision': 7,
            'path_precision': 3,
            'corner_threshold': 55,
        },
        'logo': {
            'mode': 'polygon',
            'hierarchical': 'stacked',
            'filter_speckle': 8,
            'color_precision': 8,
            'path_precision': 2,
            'corner_threshold': 50,
        },
        'detailed': {
            'mode': 'spline',
            'hierarchical': 'stacked',
            'filter_speckle': 2,
            'color_precision': 8,
            'path_precision': 4,
            'segment_length': 3.0,
        },
        'photorealistic': {
            'mode': 'spline',
            'hierarchical': 'stacked',
            'filter_speckle': 1,  # Minimal filtering to keep details
            'color_precision': 8,  # Maximum color precision
            'path_precision': 5,   # High path precision
            'corner_threshold': 45,  # Lower threshold for more detail
            'segment_length': 3.5,  # Minimum valid segment length for finer detail
            'splice_threshold': 45,  # Splice threshold for smoother curves
        },
        'ultrafine': {
            'mode': 'pixel',  # Pixel-perfect mode
            'hierarchical': 'stacked',
            'filter_speckle': 0,  # No filtering
            'color_precision': 8,  # Maximum colors
            'path_precision': 5,  # Maximum precision
            'corner_threshold': 40,
        },
        'simplified': {
            'mode': 'polygon',
            'hierarchical': 'cutout',
            'filter_speckle': 10,
            'color_precision': 5,
            'path_precision': 1,
            'corner_threshold': 70,
        },
        'clean': {
            'mode': 'polygon',
            'hierarchical': 'stacked',
            'filter_speckle': 16,  # Maximum filtering
            'color_precision': 5,   # Reduced colors
            'path_precision': 2,    # Simple paths
            'corner_threshold': 65,
        },
        'photo': {
            'preset': 'photo',  # Use built-in photo preset
        },
        'poster': {
            'preset': 'poster',  # Use built-in poster preset
        },
        'bw': {
            'preset': 'bw',  # Use built-in black and white preset
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize vectorizer with configuration"""
        self.config = config or self.DEFAULT_SETTINGS.copy()
        self._check_vtracer()
    
    def _check_vtracer(self):
        """Check if vtracer is installed"""
        try:
            subprocess.run(['vtracer', '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: vtracer is not installed!")
            print("\nInstallation options:")
            print("1. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
            print("2. Install vtracer: cargo install vtracer")
            print("\nOr download binaries from: https://github.com/visioncortex/vtracer/releases")
            sys.exit(1)
    
    def preprocess_image(self, input_path: Path, output_path: Path, 
                        reduce_colors: Optional[int] = None,
                        blur: Optional[float] = None) -> Path:
        """Preprocess image to improve vectorization results"""
        if not self._check_magick():
            return input_path
        
        cmd = ['magick', str(input_path)]
        
        # Apply blur to reduce jagged edges
        if blur:
            cmd.extend(['-blur', f'0x{blur}'])
        
        # Reduce colors for cleaner regions
        if reduce_colors:
            cmd.extend(['-dither', 'None', '-colors', str(reduce_colors)])
        
        cmd.append(str(output_path))
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Warning: Preprocessing failed: {e}")
            return input_path
    
    def _check_magick(self) -> bool:
        """Check if ImageMagick is installed"""
        try:
            subprocess.run(['magick', '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def vectorize(self, input_path: Path, output_path: Path,
                 preset: Optional[str] = None,
                 preprocess: bool = False,
                 reduce_colors: Optional[int] = None,
                 blur: Optional[float] = None) -> bool:
        """
        Vectorize a colored image to SVG
        
        Args:
            input_path: Path to input raster image
            output_path: Path for output SVG
            preset: Use a preset configuration ('doodle', 'illustration', 'logo', etc.)
            preprocess: Apply preprocessing before vectorization
            reduce_colors: Number of colors to reduce to (requires ImageMagick)
            blur: Blur amount for smoothing edges (requires ImageMagick)
        
        Returns:
            True if successful, False otherwise
        """
        # Apply preset if specified
        if preset and preset in self.PRESET_CONFIGS:
            self.config = self.PRESET_CONFIGS[preset].copy()
        
        # Ensure paths are Path objects
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return False
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Preprocess if requested
        work_input = input_path
        temp_file = None
        
        if preprocess or reduce_colors or blur:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            work_input = self.preprocess_image(
                input_path, temp_path, 
                reduce_colors=reduce_colors, 
                blur=blur
            )
        
        # Build vtracer command
        cmd = [
            'vtracer',
            '--input', str(work_input),
            '--output', str(output_path),
        ]
        
        # Check if using built-in preset
        if 'preset' in self.config:
            cmd.extend(['--preset', self.config['preset']])
        else:
            # Add mode if specified
            if 'mode' in self.config:
                cmd.extend(['--mode', self.config['mode']])
            
            # Add hierarchical if specified
            if 'hierarchical' in self.config:
                cmd.extend(['--hierarchical', self.config['hierarchical']])
            
            # Add other parameters
            if 'filter_speckle' in self.config:
                cmd.extend(['--filter_speckle', str(self.config['filter_speckle'])])
            
            if 'color_precision' in self.config:
                cmd.extend(['--color_precision', str(self.config['color_precision'])])
            
            if 'path_precision' in self.config:
                cmd.extend(['--path_precision', str(self.config['path_precision'])])
            
            if 'corner_threshold' in self.config:
                cmd.extend(['--corner_threshold', str(self.config['corner_threshold'])])
            
            if 'segment_length' in self.config:
                cmd.extend(['--segment_length', str(self.config['segment_length'])])
            
            if 'splice_threshold' in self.config:
                cmd.extend(['--splice_threshold', str(self.config['splice_threshold'])])
        
        # Run vtracer
        try:
            result = subprocess.run(cmd, capture_output=True, check=True, text=True)
            
            # Get file sizes for comparison
            input_size = input_path.stat().st_size / 1024  # KB
            output_size = output_path.stat().st_size / 1024  # KB
            
            print(f"✓ Vectorized: {input_path.name}")
            print(f"  Input:  {input_size:.1f} KB")
            print(f"  Output: {output_size:.1f} KB")
            print(f"  Preset: {preset or 'custom'}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error: Vectorization failed for {input_path}")
            print(f"  {e.stderr}")
            return False
        
        finally:
            # Clean up temp file
            if temp_file and Path(temp_file.name).exists():
                Path(temp_file.name).unlink()

def main():
    parser = argparse.ArgumentParser(
        description='Vectorize colored images to SVG using vtracer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  doodle       - Flat-color doodles and cartoons
  illustration - Detailed illustrations
  logo         - Clean logos and icons
  detailed     - Maximum detail preservation with splines
  simplified   - Minimal, clean output
  photo        - Built-in preset for photos
  poster       - Built-in preset for poster-style images
  bw           - Built-in preset for black and white images

Examples:
  %(prog)s input.png output.svg
  %(prog)s input.png output.svg --preset doodle
  %(prog)s input.png output.svg --preprocess --reduce-colors 24
  %(prog)s input.png output.svg --color-precision 20 --paths 768
        """
    )
    
    parser.add_argument('input', type=Path, help='Input raster image')
    parser.add_argument('output', type=Path, help='Output SVG file')
    
    parser.add_argument('--preset', choices=ColorVectorizer.PRESET_CONFIGS.keys(),
                       help='Use a preset configuration')
    
    # Preprocessing options
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply preprocessing')
    parser.add_argument('--reduce-colors', type=int,
                       help='Reduce to N colors before vectorization', dest='reduce_colors')
    parser.add_argument('--blur', type=float,
                       help='Apply blur to smooth edges (e.g., 0.4)')
    
    # Manual configuration
    parser.add_argument('--mode', choices=['pixel', 'polygon', 'spline'],
                       help='Curve fitting mode')
    parser.add_argument('--hierarchical', choices=['stacked', 'cutout'],
                       help='Hierarchical clustering mode')
    parser.add_argument('--filter-speckle', type=int,
                       help='Remove small specks (default: 4)', dest='filter_speckle')
    parser.add_argument('--color-precision', type=int,
                       help='Significant bits in RGB channel (default: 6)', dest='color_precision')
    parser.add_argument('--path-precision', type=int,
                       help='Decimal places in path string (default: 3)', dest='path_precision')
    parser.add_argument('--corner-threshold', type=float,
                       help='Minimum angle for corners (default: 60)', dest='corner_threshold')
    parser.add_argument('--segment-length', type=float,
                       help='Maximum segment length', dest='segment_length')
    
    args = parser.parse_args()
    
    # Build configuration
    config = ColorVectorizer.DEFAULT_SETTINGS.copy()
    
    if args.mode:
        config['mode'] = args.mode
    if args.hierarchical:
        config['hierarchical'] = args.hierarchical
    if args.filter_speckle is not None:
        config['filter_speckle'] = args.filter_speckle
    if args.color_precision is not None:
        config['color_precision'] = args.color_precision
    if args.path_precision is not None:
        config['path_precision'] = args.path_precision
    if args.corner_threshold is not None:
        config['corner_threshold'] = args.corner_threshold
    if args.segment_length is not None:
        config['segment_length'] = args.segment_length
    
    # Create vectorizer and process
    vectorizer = ColorVectorizer(config)
    
    success = vectorizer.vectorize(
        args.input,
        args.output,
        preset=args.preset,
        preprocess=args.preprocess,
        reduce_colors=args.reduce_colors,
        blur=args.blur
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()