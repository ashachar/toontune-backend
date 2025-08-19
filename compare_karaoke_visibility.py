#!/usr/bin/env python3
"""
Compare karaoke visibility between test and final videos
"""

import cv2
import numpy as np
from pathlib import Path

def compare_frames():
    print("="*70)
    print("🔍 COMPARING KARAOKE VISIBILITY")
    print("="*70)
    
    analysis_dir = Path("uploads/assets/videos/do_re_mi/karaoke_analysis")
    
    # Compare test vs final at same timestamp (10s)
    test_frame = analysis_dir / "test_10s.png"
    final_frame = analysis_dir / "final_10s.png"
    
    if not (test_frame.exists() and final_frame.exists()):
        print("❌ Frames not found")
        return
    
    test_img = cv2.imread(str(test_frame))
    final_img = cv2.imread(str(final_frame))
    
    height = test_img.shape[0]
    
    # Get bottom regions
    test_bottom = test_img[height-100:, :]
    final_bottom = final_img[height-100:, :]
    
    # Convert to grayscale
    test_gray = cv2.cvtColor(test_bottom, cv2.COLOR_BGR2GRAY)
    final_gray = cv2.cvtColor(final_bottom, cv2.COLOR_BGR2GRAY)
    
    print("\n📊 Brightness Comparison (bottom 100px):")
    print("-"*50)
    print(f"Test karaoke:  Mean={np.mean(test_gray):.1f}, Max={np.max(test_gray)}")
    print(f"Final video:   Mean={np.mean(final_gray):.1f}, Max={np.max(final_gray)}")
    
    # Create difference image
    diff = cv2.absdiff(test_gray, final_gray)
    
    print(f"\nDifference:    Mean={np.mean(diff):.1f}, Max={np.max(diff)}")
    
    # Save difference image
    diff_path = analysis_dir / "difference_10s.png"
    cv2.imwrite(str(diff_path), diff)
    print(f"\nDifference image saved to: {diff_path.name}")
    
    # Create side-by-side comparison
    comparison = np.hstack([test_bottom, final_bottom])
    comp_path = analysis_dir / "comparison_10s.png"
    cv2.imwrite(str(comp_path), comparison)
    print(f"Comparison saved to: {comp_path.name}")
    
    # Check if karaoke text color changed
    print("\n🎨 Color Analysis:")
    print("-"*50)
    
    # Look for white/bright pixels
    _, test_binary = cv2.threshold(test_gray, 200, 255, cv2.THRESH_BINARY)
    _, final_binary = cv2.threshold(final_gray, 200, 255, cv2.THRESH_BINARY)
    
    test_white = np.sum(test_binary > 0)
    final_white = np.sum(final_binary > 0)
    
    print(f"White pixels in test:  {test_white}")
    print(f"White pixels in final: {final_white}")
    print(f"Ratio: {final_white/test_white:.1%}" if test_white > 0 else "N/A")
    
    # Check if something is overlaying the karaoke
    if final_white < test_white * 0.5:
        print("\n⚠️  WARNING: Final has <50% of the white pixels of test!")
        print("    Karaoke might be:")
        print("    • Covered by another overlay")
        print("    • Rendered with wrong opacity")
        print("    • Using wrong blend mode")
    
    # Check the actual pipeline command
    print("\n🔧 Checking Pipeline Implementation:")
    print("-"*50)
    
    # Read the pipeline file to see the filter order
    pipeline_file = Path("pipeline_single_pass_full.py")
    if pipeline_file.exists():
        with open(pipeline_file) as f:
            content = f.read()
        
        # Find the filter building section
        if "filter_complex" in content:
            print("✓ Pipeline uses filter_complex")
            
            # Check order of operations
            if content.index("ass=") < content.index("drawtext") if "drawtext" in content else True:
                print("✓ Karaoke (ass) is applied BEFORE phrases")
            else:
                print("❌ Karaoke might be applied AFTER phrases")
            
            if content.index("ass=") < content.index("overlay") if "overlay" in content else True:
                print("✓ Karaoke is applied BEFORE cartoons")
            else:
                print("❌ Karaoke might be applied AFTER cartoons")
    
    print("\n" + "="*70)
    print("💡 DIAGNOSIS:")
    print("-"*70)
    
    if final_white < test_white * 0.3:
        print("❌ KARAOKE IS SEVERELY DEGRADED OR HIDDEN")
        print("   The final video has karaoke but it's barely visible!")
    elif final_white < test_white * 0.7:
        print("⚠️  KARAOKE IS PARTIALLY VISIBLE")
        print("   Some karaoke is lost in the final encoding")
    else:
        print("✅ KARAOKE SEEMS PRESENT")
        print("   But user reports it's not visible - check the actual video!")
    
    print("\n🎯 Next steps:")
    print("1. Open the comparison images to see the difference")
    print("2. Check if phrases/cartoons are covering the karaoke")
    print("3. Verify the filter_complex order in the pipeline")
    print("="*70)

if __name__ == "__main__":
    compare_frames()