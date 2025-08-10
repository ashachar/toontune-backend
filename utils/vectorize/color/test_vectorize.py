#!/usr/bin/env python3
"""
Test script for color vectorization
Demonstrates various presets and options
"""

import subprocess
import sys
import os
from pathlib import Path
import time
import shutil

def check_dependencies():
    """Check if required tools are installed"""
    print("Checking dependencies...")
    
    # Check vtracer
    try:
        result = subprocess.run(['vtracer', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✓ vtracer installed: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ vtracer not installed!")
        print("  Run: ./install_vtracer.sh")
        return False
    
    # Check ImageMagick (optional)
    try:
        subprocess.run(['magick', '--version'], 
                      capture_output=True, check=True)
        print("✓ ImageMagick installed (preprocessing available)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ ImageMagick not installed (preprocessing disabled)")
    
    return True

def run_test(cmd, description):
    """Run a test command and report results"""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start
        print(f"✓ Success ({elapsed:.2f}s)")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def main():
    print("Color Vectorization Test Suite")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup test directories
    test_image = Path("uploads/assets/woman.png")
    output_dir = Path("uploads/vectorized/tests")
    
    if not test_image.exists():
        print(f"\nError: Test image not found: {test_image}")
        print("Please ensure uploads/assets/woman.png exists")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Get script path
    script_dir = Path(__file__).parent
    vectorize_script = script_dir / "vectorize_color.py"
    batch_script = script_dir / "batch_vectorize.py"
    
    # Test cases
    tests = [
        # Basic presets
        (
            ["python3", str(vectorize_script), 
             str(test_image), str(output_dir / "test_doodle.svg"),
             "--preset", "doodle"],
            "Doodle preset (flat colors, good for cartoons)"
        ),
        (
            ["python3", str(vectorize_script),
             str(test_image), str(output_dir / "test_illustration.svg"),
             "--preset", "illustration"],
            "Illustration preset (more detail)"
        ),
        (
            ["python3", str(vectorize_script),
             str(test_image), str(output_dir / "test_logo.svg"),
             "--preset", "logo"],
            "Logo preset (clean, minimal)"
        ),
        (
            ["python3", str(vectorize_script),
             str(test_image), str(output_dir / "test_photo.svg"),
             "--preset", "photo"],
            "Photo preset (built-in vtracer preset)"
        ),
        (
            ["python3", str(vectorize_script),
             str(test_image), str(output_dir / "test_poster.svg"),
             "--preset", "poster"],
            "Poster preset (built-in vtracer preset)"
        ),
        
        # Custom configurations
        (
            ["python3", str(vectorize_script),
             str(test_image), str(output_dir / "test_custom_spline.svg"),
             "--mode", "spline", "--color-precision", "7"],
            "Custom: Spline mode with higher color precision"
        ),
        (
            ["python3", str(vectorize_script),
             str(test_image), str(output_dir / "test_custom_minimal.svg"),
             "--mode", "polygon", "--filter-speckle", "20",
             "--color-precision", "4"],
            "Custom: Minimal output with heavy filtering"
        ),
        
        # With preprocessing (if ImageMagick available)
        (
            ["python3", str(vectorize_script),
             str(test_image), str(output_dir / "test_preprocessed.svg"),
             "--preset", "doodle", "--preprocess", "--reduce-colors", "16"],
            "With preprocessing: Reduce to 16 colors"
        ),
        (
            ["python3", str(vectorize_script),
             str(test_image), str(output_dir / "test_blurred.svg"),
             "--preset", "doodle", "--blur", "0.5"],
            "With preprocessing: Slight blur for smoother edges"
        ),
    ]
    
    # Run individual tests
    print("\n" + "="*60)
    print("INDIVIDUAL FILE TESTS")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for cmd, description in tests:
        if run_test(cmd, description):
            passed += 1
        else:
            failed += 1
    
    # Batch processing test
    print("\n" + "="*60)
    print("BATCH PROCESSING TEST")
    print("="*60)
    
    batch_output = output_dir / "batch"
    batch_cmd = [
        "python3", str(batch_script),
        "uploads/assets", str(batch_output),
        "--preset", "doodle",
        "--parallel", "2",
        "--skip-existing"
    ]
    
    if run_test(batch_cmd, "Batch process all images in assets folder"):
        passed += 1
    else:
        failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n⚠ {failed} test(s) failed")
    
    # List output files
    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    
    svg_files = list(output_dir.rglob("*.svg"))
    if svg_files:
        print(f"Generated {len(svg_files)} SVG files:")
        for svg in sorted(svg_files):
            size = svg.stat().st_size / 1024
            print(f"  - {svg.relative_to(output_dir)}: {size:.1f} KB")
    else:
        print("No SVG files generated")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())