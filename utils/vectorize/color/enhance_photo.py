#!/usr/bin/env python3
"""
Enhanced photo vectorization with preprocessing for better detail preservation
"""

import subprocess
import sys
from pathlib import Path
import tempfile
import argparse
from typing import Optional

def preprocess_for_detail(input_path: Path, output_path: Path, 
                         sharpness: float = 1.0,
                         contrast: float = 1.2,
                         denoise: bool = False) -> bool:
    """
    Preprocess image to enhance details before vectorization
    
    Args:
        input_path: Input image path
        output_path: Output processed image path
        sharpness: Sharpening amount (0.5-2.0, default 1.0)
        contrast: Contrast enhancement (0.5-2.0, default 1.2)
        denoise: Apply noise reduction
    """
    try:
        cmd = ['magick', str(input_path)]
        
        # Enhance contrast to make details more distinct
        if contrast != 1.0:
            cmd.extend(['-contrast-stretch', f'{int((1-contrast)*5)}%x{int((1-contrast)*5)}%'])
        
        # Sharpen to enhance edges
        if sharpness > 0:
            cmd.extend(['-sharpen', f'0x{sharpness}'])
        
        # Denoise if requested (helps reduce artifacts)
        if denoise:
            cmd.extend(['-despeckle'])
        
        # Increase color depth for better gradients
        cmd.extend(['-depth', '16'])
        
        # Output
        cmd.append(str(output_path))
        
        subprocess.run(cmd, capture_output=True, check=True)
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: ImageMagick not available for preprocessing")
        return False

def vectorize_photorealistic(input_path: Path, output_path: Path,
                           preset: str = 'photorealistic',
                           preprocess: bool = True,
                           use_gradients: bool = False) -> bool:
    """
    Vectorize with enhanced settings for photorealistic images
    """
    work_input = input_path
    temp_file = None
    
    # Preprocess if ImageMagick is available
    if preprocess:
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            if preprocess_for_detail(input_path, temp_path):
                work_input = temp_path
                print(f"✓ Preprocessed image for enhanced detail")
        except Exception as e:
            print(f"Preprocessing failed: {e}")
    
    # Build vtracer command with optimal settings
    cmd = [
        'vtracer',
        '--input', str(work_input),
        '--output', str(output_path),
    ]
    
    if preset == 'maximum':
        # Maximum detail settings
        cmd.extend([
            '--mode', 'pixel',  # Pixel-perfect accuracy
            '--hierarchical', 'stacked',
            '--filter_speckle', '0',  # No filtering
            '--color_precision', '8',  # Maximum color precision
            '--path_precision', '5',  # Maximum path precision
        ])
    elif preset == 'photorealistic':
        # Balanced photorealistic
        cmd.extend([
            '--mode', 'spline',  # Smooth curves
            '--hierarchical', 'stacked',
            '--filter_speckle', '1',  # Minimal filtering
            '--color_precision', '8',  # High color precision
            '--path_precision', '4',  # Good path precision
            '--segment_length', '3.5',  # Fine segments
        ])
    elif preset == 'enhanced_photo':
        # Use built-in photo preset with preprocessing
        cmd.extend(['--preset', 'photo'])
    else:
        # Default to built-in photo preset
        cmd.extend(['--preset', 'photo'])
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        
        # Get file sizes
        input_size = input_path.stat().st_size / 1024
        output_size = output_path.stat().st_size / 1024
        
        print(f"✓ Vectorized: {input_path.name}")
        print(f"  Input:  {input_size:.1f} KB")
        print(f"  Output: {output_size:.1f} KB")
        print(f"  Preset: {preset}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Vectorization failed")
        print(e.stderr)
        return False
        
    finally:
        # Clean up temp file
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink()

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced photo vectorization for detail preservation',
        epilog="""
Presets:
  photorealistic - Balanced detail and file size
  maximum        - Maximum possible detail (large files)
  enhanced_photo - Built-in photo preset with preprocessing
  
Examples:
  %(prog)s hand.png hand_detailed.svg
  %(prog)s hand.png hand_max.svg --preset maximum
  %(prog)s hand.png hand.svg --no-preprocess
        """
    )
    
    parser.add_argument('input', type=Path, help='Input image')
    parser.add_argument('output', type=Path, help='Output SVG')
    parser.add_argument('--preset', default='photorealistic',
                       choices=['photorealistic', 'maximum', 'enhanced_photo'],
                       help='Detail level preset')
    parser.add_argument('--no-preprocess', dest='preprocess',
                       action='store_false', default=True,
                       help='Skip preprocessing')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Process
    success = vectorize_photorealistic(
        args.input,
        args.output,
        preset=args.preset,
        preprocess=args.preprocess
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()