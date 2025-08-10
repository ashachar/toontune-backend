#!/usr/bin/env python3
"""
Clean vectorization with aggressive noise removal
Removes small clusters and noise from vectorized images
"""

import subprocess
import sys
from pathlib import Path
import tempfile
import argparse
import xml.etree.ElementTree as ET
import re
from typing import Optional, Tuple
import math

def preprocess_clean(input_path: Path, output_path: Path,
                    despeckle_iterations: int = 3,
                    median_radius: int = 2,
                    reduce_colors: int = 32) -> bool:
    """
    Aggressive preprocessing to remove noise before vectorization
    """
    try:
        cmd = ['magick', str(input_path)]
        
        # Multiple despeckle passes to remove noise
        for _ in range(despeckle_iterations):
            cmd.append('-despeckle')
        
        # Median filter to smooth out noise
        if median_radius > 0:
            cmd.extend(['-median', str(median_radius)])
        
        # Reduce colors to merge similar regions
        if reduce_colors:
            cmd.extend(['-dither', 'None', '-colors', str(reduce_colors)])
        
        # Morphological operations to clean up
        cmd.extend(['-morphology', 'Close', 'Diamond:1'])
        cmd.extend(['-morphology', 'Open', 'Square:1'])
        
        # Final cleanup
        cmd.extend(['-despeckle'])
        
        cmd.append(str(output_path))
        
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"✓ Preprocessed with aggressive noise removal")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: ImageMagick not available, skipping preprocessing")
        return False

def calculate_path_area(path_data: str) -> float:
    """
    Estimate the area of an SVG path (simplified calculation)
    """
    # Extract coordinates from path
    coords = re.findall(r'[-+]?\d*\.?\d+', path_data)
    if len(coords) < 4:
        return 0
    
    # Convert to float pairs
    points = []
    for i in range(0, len(coords)-1, 2):
        try:
            points.append((float(coords[i]), float(coords[i+1])))
        except (ValueError, IndexError):
            continue
    
    if len(points) < 3:
        return 0
    
    # Calculate bounding box area as approximation
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return width * height
    
    return 0

def clean_svg(svg_path: Path, output_path: Path, 
             min_area: float = 100.0,
             min_path_points: int = 4) -> int:
    """
    Post-process SVG to remove small noise clusters
    
    Args:
        svg_path: Input SVG file
        output_path: Cleaned output SVG
        min_area: Minimum area for paths to keep
        min_path_points: Minimum number of points in path
    
    Returns:
        Number of paths removed
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Find all path elements
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        paths_removed = 0
        
        # Process all path elements
        for path in root.findall('.//svg:path', namespaces) or root.findall('.//path'):
            d = path.get('d', '')
            
            # Count path commands
            commands = len(re.findall(r'[MLHVCSQTAZmlhvcsqtaz]', d))
            
            # Calculate approximate area
            area = calculate_path_area(d)
            
            # Remove small paths
            if commands < min_path_points or area < min_area:
                parent = path.getparent() if hasattr(path, 'getparent') else None
                if parent is not None:
                    parent.remove(path)
                else:
                    # For ElementTree without getparent
                    for parent in root.iter():
                        if path in parent:
                            parent.remove(path)
                            break
                paths_removed += 1
        
        # Write cleaned SVG
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        if paths_removed > 0:
            print(f"✓ Removed {paths_removed} small noise paths")
        
        return paths_removed
        
    except Exception as e:
        print(f"Warning: Could not clean SVG: {e}")
        return 0

def vectorize_clean(input_path: Path, output_path: Path,
                   filter_speckle: int = 20,
                   color_precision: int = 6,
                   preprocess: bool = True,
                   clean_svg_output: bool = True,
                   min_cluster_area: float = 150.0) -> bool:
    """
    Vectorize with aggressive noise filtering
    """
    work_input = input_path
    temp_files = []
    
    try:
        # Preprocessing step
        if preprocess:
            temp_preprocessed = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_files.append(temp_preprocessed.name)
            temp_preprocessed.close()
            
            if preprocess_clean(input_path, Path(temp_preprocessed.name)):
                work_input = Path(temp_preprocessed.name)
        
        # Temporary output for initial vectorization
        if clean_svg_output:
            temp_svg = tempfile.NamedTemporaryFile(suffix='.svg', delete=False)
            temp_files.append(temp_svg.name)
            temp_svg.close()
            vector_output = Path(temp_svg.name)
        else:
            vector_output = output_path
        
        # Vectorize with aggressive filtering
        cmd = [
            'vtracer',
            '--input', str(work_input),
            '--output', str(vector_output),
            '--mode', 'polygon',  # Clean polygon mode
            '--hierarchical', 'stacked',
            '--filter_speckle', str(filter_speckle),  # Aggressive filtering
            '--color_precision', str(color_precision),
            '--path_precision', '2',  # Lower precision for cleaner paths
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        
        # Post-process SVG if requested
        if clean_svg_output:
            clean_svg(vector_output, output_path, min_area=min_cluster_area)
        
        # Report file sizes
        input_size = input_path.stat().st_size / 1024
        output_size = output_path.stat().st_size / 1024
        
        print(f"✓ Clean vectorized: {input_path.name}")
        print(f"  Input:  {input_size:.1f} KB")
        print(f"  Output: {output_size:.1f} KB")
        print(f"  Filter: {filter_speckle}px speckle removal")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Vectorization failed")
        if e.stderr:
            print(e.stderr)
        return False
        
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

def main():
    parser = argparse.ArgumentParser(
        description='Clean vectorization with noise removal',
        epilog="""
Examples:
  %(prog)s man.png man_clean.svg
  %(prog)s man.png man_clean.svg --filter-speckle 30
  %(prog)s man.png man_clean.svg --colors 16 --min-area 200
  %(prog)s man.png man_clean.svg --no-preprocess --no-clean-svg
        """
    )
    
    parser.add_argument('input', type=Path, help='Input image')
    parser.add_argument('output', type=Path, help='Output SVG')
    
    # Noise filtering options
    parser.add_argument('--filter-speckle', type=int, default=20,
                       help='Speckle filter size in pixels (default: 20)')
    parser.add_argument('--colors', type=int, default=32,
                       help='Number of colors to reduce to (default: 32)')
    parser.add_argument('--color-precision', type=int, default=6,
                       help='Color precision bits (default: 6)')
    parser.add_argument('--min-area', type=float, default=150.0,
                       help='Minimum path area to keep (default: 150)')
    parser.add_argument('--despeckle', type=int, default=3,
                       help='Despeckle iterations (default: 3)')
    parser.add_argument('--median', type=int, default=2,
                       help='Median filter radius (default: 2)')
    
    # Processing options
    parser.add_argument('--no-preprocess', dest='preprocess',
                       action='store_false', default=True,
                       help='Skip preprocessing')
    parser.add_argument('--no-clean-svg', dest='clean_svg',
                       action='store_false', default=True,
                       help='Skip SVG cleaning')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Process
    success = vectorize_clean(
        args.input,
        args.output,
        filter_speckle=args.filter_speckle,
        color_precision=args.color_precision,
        preprocess=args.preprocess,
        clean_svg_output=args.clean_svg,
        min_cluster_area=args.min_area
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()