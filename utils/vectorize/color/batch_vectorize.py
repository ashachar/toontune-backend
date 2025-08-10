#!/usr/bin/env python3
"""
Batch vectorization of colored images using vtracer
Processes multiple images with configurable settings and parallel processing
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any
import time
from vectorize_color import ColorVectorizer

class BatchVectorizer:
    """Batch process multiple images for vectorization"""
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff'}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 preset: Optional[str] = None,
                 parallel: int = 4):
        """
        Initialize batch vectorizer
        
        Args:
            config: Custom configuration for vtracer
            preset: Use a preset configuration
            parallel: Number of parallel workers
        """
        self.vectorizer = ColorVectorizer(config)
        self.preset = preset
        self.parallel = max(1, parallel)
    
    def find_images(self, input_path: Path, recursive: bool = True) -> List[Path]:
        """Find all supported images in directory"""
        images = []
        
        if input_path.is_file():
            if input_path.suffix.lower() in self.SUPPORTED_FORMATS:
                images.append(input_path)
        elif input_path.is_dir():
            pattern = '**/*' if recursive else '*'
            for ext in self.SUPPORTED_FORMATS:
                images.extend(input_path.glob(f'{pattern}{ext}'))
                images.extend(input_path.glob(f'{pattern}{ext.upper()}'))
        
        return sorted(set(images))
    
    def process_single(self, input_path: Path, output_dir: Path,
                      preserve_structure: bool = False,
                      base_dir: Optional[Path] = None,
                      **kwargs) -> Tuple[Path, bool, float]:
        """
        Process a single image
        
        Returns:
            Tuple of (input_path, success, processing_time)
        """
        start_time = time.time()
        
        # Determine output path
        if preserve_structure and base_dir:
            # Preserve directory structure
            rel_path = input_path.relative_to(base_dir)
            output_path = output_dir / rel_path.with_suffix('.svg')
        else:
            # Flat output structure
            output_path = output_dir / f"{input_path.stem}.svg"
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process image
        success = self.vectorizer.vectorize(
            input_path, 
            output_path,
            preset=self.preset,
            **kwargs
        )
        
        elapsed = time.time() - start_time
        return (input_path, success, elapsed)
    
    def process_batch(self, input_path: Path, output_dir: Path,
                     recursive: bool = True,
                     preserve_structure: bool = False,
                     skip_existing: bool = False,
                     **kwargs) -> Dict[str, Any]:
        """
        Process multiple images in batch
        
        Args:
            input_path: Input file or directory
            output_dir: Output directory for SVGs
            recursive: Search subdirectories
            preserve_structure: Preserve directory structure in output
            skip_existing: Skip if output already exists
            **kwargs: Additional arguments for vectorization
        
        Returns:
            Statistics about the batch processing
        """
        # Find all images
        images = self.find_images(input_path, recursive)
        
        if not images:
            print(f"No supported images found in: {input_path}")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
        
        print(f"Found {len(images)} images to process")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter existing if requested
        to_process = []
        skipped = []
        
        for img in images:
            if preserve_structure and input_path.is_dir():
                rel_path = img.relative_to(input_path)
                out_path = output_dir / rel_path.with_suffix('.svg')
            else:
                out_path = output_dir / f"{img.stem}.svg"
            
            if skip_existing and out_path.exists():
                skipped.append(img)
            else:
                to_process.append(img)
        
        if skipped:
            print(f"Skipping {len(skipped)} existing files")
        
        # Process images
        results = []
        base_dir = input_path if input_path.is_dir() else input_path.parent
        
        if self.parallel > 1 and len(to_process) > 1:
            # Parallel processing
            print(f"Processing with {self.parallel} workers...")
            
            with ProcessPoolExecutor(max_workers=self.parallel) as executor:
                futures = {
                    executor.submit(
                        self.process_single, 
                        img, 
                        output_dir,
                        preserve_structure,
                        base_dir,
                        **kwargs
                    ): img 
                    for img in to_process
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Progress update
                        completed = len(results) + len(skipped)
                        print(f"[{completed}/{len(images)}] Processed: {result[0].name}")
                        
                    except Exception as e:
                        img = futures[future]
                        print(f"Error processing {img}: {e}")
                        results.append((img, False, 0))
        else:
            # Sequential processing
            for i, img in enumerate(to_process, 1):
                result = self.process_single(
                    img, output_dir, preserve_structure, base_dir, **kwargs
                )
                results.append(result)
                
                # Progress update
                completed = i + len(skipped)
                print(f"[{completed}/{len(images)}] Processed: {result[0].name}")
        
        # Calculate statistics
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        total_time = sum(t for _, _, t in results)
        
        stats = {
            'total': len(images),
            'processed': len(results),
            'success': successful,
            'failed': failed,
            'skipped': len(skipped),
            'total_time': total_time,
            'avg_time': total_time / len(results) if results else 0
        }
        
        # Print summary
        print("\n" + "="*50)
        print("Batch Processing Complete")
        print("="*50)
        print(f"Total images:    {stats['total']}")
        print(f"Processed:       {stats['processed']}")
        print(f"Successful:      {stats['success']}")
        print(f"Failed:          {stats['failed']}")
        print(f"Skipped:         {stats['skipped']}")
        print(f"Total time:      {stats['total_time']:.1f}s")
        if stats['processed'] > 0:
            print(f"Average time:    {stats['avg_time']:.2f}s per image")
        
        # List failed files if any
        if failed > 0:
            print("\nFailed files:")
            for path, success, _ in results:
                if not success:
                    print(f"  - {path}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(
        description='Batch vectorize colored images using vtracer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./input_folder ./output_folder
  %(prog)s ./input_folder ./output_folder --preset doodle
  %(prog)s ./image.png ./output --reduce-colors 24
  %(prog)s ./input ./output --parallel 8 --skip-existing
  %(prog)s ./input ./output --preserve-structure --recursive
        """
    )
    
    parser.add_argument('input', type=Path, 
                       help='Input file or directory')
    parser.add_argument('output', type=Path,
                       help='Output directory for SVGs')
    
    # Batch options
    parser.add_argument('--parallel', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Search subdirectories (default: True)')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false',
                       help='Don\'t search subdirectories')
    parser.add_argument('--preserve-structure', action='store_true',
                       help='Preserve directory structure in output')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files that already exist')
    
    # Vectorization options
    parser.add_argument('--preset', choices=ColorVectorizer.PRESET_CONFIGS.keys(),
                       help='Use a preset configuration')
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply preprocessing')
    parser.add_argument('--reduce-colors', type=int,
                       help='Reduce to N colors before vectorization', dest='reduce_colors')
    parser.add_argument('--blur', type=float,
                       help='Apply blur to smooth edges')
    
    # Manual configuration
    parser.add_argument('--mode', choices=['pixel', 'polygon', 'spline'],
                       help='Curve fitting mode')
    parser.add_argument('--hierarchical', choices=['stacked', 'cutout'],
                       help='Hierarchical clustering mode')
    parser.add_argument('--filter-speckle', type=int,
                       help='Remove small specks', dest='filter_speckle')
    parser.add_argument('--color-precision', type=int,
                       help='Significant bits in RGB channel', dest='color_precision')
    parser.add_argument('--path-precision', type=int,
                       help='Decimal places in path string', dest='path_precision')
    parser.add_argument('--corner-threshold', type=float,
                       help='Minimum angle for corners', dest='corner_threshold')
    parser.add_argument('--segment-length', type=float,
                       help='Maximum segment length', dest='segment_length')
    
    # Output format
    parser.add_argument('--stats-file', type=Path,
                       help='Save processing statistics to JSON file')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {}
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
    
    # Create batch processor
    batch = BatchVectorizer(
        config=config if config else None,
        preset=args.preset,
        parallel=args.parallel
    )
    
    # Process batch
    stats = batch.process_batch(
        args.input,
        args.output,
        recursive=args.recursive,
        preserve_structure=args.preserve_structure,
        skip_existing=args.skip_existing,
        preprocess=args.preprocess,
        reduce_colors=args.reduce_colors,
        blur=args.blur
    )
    
    # Save statistics if requested
    if args.stats_file:
        with open(args.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {args.stats_file}")
    
    # Exit with error if any failed
    sys.exit(0 if stats['failed'] == 0 else 1)

if __name__ == '__main__':
    main()