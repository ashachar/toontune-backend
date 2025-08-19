#!/usr/bin/env python3
"""
Comprehensive debug of why 'beginning' still appears on Maria's face.
Check everything: positions, masks, coordinate systems.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
from rembg import remove
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def check_all_position_files():
    """Check all position files to see what coordinates are used for 'beginning'."""
    print("="*70)
    print("CHECKING ALL POSITION FILES")
    print("="*70)
    
    position_files = [
        "uploads/assets/videos/do_re_mi/scenes/backgrounds_validated/word_positions_v2.json",
        "uploads/assets/videos/do_re_mi/scenes/backgrounds_fixed/word_positions_v2.json",
        "uploads/assets/videos/do_re_mi/inferences/scene_001_validated.json",
        "uploads/assets/videos/do_re_mi/inferences/scene_001_fixed.json",
        "uploads/assets/videos/do_re_mi/inferences/scene_001_final.json",
    ]
    
    for file_path in position_files:
        path = Path(file_path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                
                # Handle different file formats
                if 'text_overlays' in data:
                    words = data['text_overlays']
                else:
                    words = data
                
                # Find 'beginning'
                for i, word in enumerate(words):
                    if word.get('word', '').lower() == 'beginning':
                        print(f"\n{path.name}:")
                        print(f"  Index: {i}")
                        print(f"  Position: ({word.get('x')}, {word.get('y')})")
                        print(f"  Timing: {word.get('start', 0):.2f}s - {word.get('end', 0):.2f}s")
                        print(f"  Font size: {word.get('fontsize', 48)}")
                        break
        else:
            print(f"\n{path.name}: NOT FOUND")


def debug_position_with_mask(x, y, fontsize, timestamp):
    """Debug a specific position with detailed mask analysis."""
    print("\n" + "="*70)
    print(f"DETAILED POSITION DEBUG AT ({x}, {y})")
    print("="*70)
    
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Go to timestamp
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"ERROR: Could not read frame at {timestamp}s")
        return
    
    # Get background mask using rembg
    print("Applying rembg...")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    nobg_img = remove(pil_image)
    nobg_array = np.array(nobg_img)
    
    # Get alpha channel
    if nobg_array.shape[2] == 4:
        alpha = nobg_array[:, :, 3]
    else:
        alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
    
    # Background mask
    background_mask = np.where(alpha < 128, 255, 0).astype(np.uint8)
    
    # Calculate text bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = fontsize / 48.0
    thickness = 2
    text = "beginning"
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    print(f"\nText metrics:")
    print(f"  Text: '{text}'")
    print(f"  Width: {text_width}px")
    print(f"  Height: {text_height}px")
    print(f"  Baseline: {baseline}px")
    
    # Test both coordinate systems
    print(f"\nCoordinate System Analysis:")
    print(f"  Given position: ({x}, {y})")
    
    # FFmpeg convention (TOP-LEFT)
    ffmpeg_y1 = y
    ffmpeg_y2 = y + text_height + baseline
    ffmpeg_x1 = x
    ffmpeg_x2 = x + text_width
    
    print(f"\n  FFmpeg convention (TOP-LEFT origin):")
    print(f"    Text box: ({ffmpeg_x1}, {ffmpeg_y1}) to ({ffmpeg_x2}, {ffmpeg_y2})")
    
    # Check if this region is in background
    if ffmpeg_y1 >= 0 and ffmpeg_y2 <= height and ffmpeg_x1 >= 0 and ffmpeg_x2 <= width:
        region = background_mask[ffmpeg_y1:ffmpeg_y2, ffmpeg_x1:ffmpeg_x2]
        if region.size > 0:
            bg_ratio = np.mean(region == 255)
            print(f"    Background ratio: {bg_ratio:.1%}")
            if bg_ratio < 0.5:
                print(f"    ⚠️ MOSTLY FOREGROUND!")
        else:
            print(f"    ⚠️ EMPTY REGION!")
    else:
        print(f"    ⚠️ OUT OF BOUNDS!")
    
    # OpenCV convention (BOTTOM-LEFT) - for comparison
    opencv_y1 = y - text_height
    opencv_y2 = y + baseline
    opencv_x1 = x
    opencv_x2 = x + text_width
    
    print(f"\n  OpenCV convention (BOTTOM-LEFT origin):")
    print(f"    Text box: ({opencv_x1}, {opencv_y1}) to ({opencv_x2}, {opencv_y2})")
    
    # Check if this region is in background
    if opencv_y1 >= 0 and opencv_y2 <= height and opencv_x1 >= 0 and opencv_x2 <= width:
        region = background_mask[opencv_y1:opencv_y2, opencv_x1:opencv_x2]
        if region.size > 0:
            bg_ratio = np.mean(region == 255)
            print(f"    Background ratio: {bg_ratio:.1%}")
            if bg_ratio < 0.5:
                print(f"    ⚠️ MOSTLY FOREGROUND!")
        else:
            print(f"    ⚠️ EMPTY REGION!")
    else:
        print(f"    ⚠️ OUT OF BOUNDS!")
    
    # Create detailed visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original frame
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Frame at {timestamp:.2f}s')
    axes[0, 0].axis('off')
    
    # Background mask
    axes[0, 1].imshow(background_mask, cmap='gray')
    axes[0, 1].set_title('Background Mask')
    axes[0, 1].axis('off')
    
    # FFmpeg convention visualization
    ffmpeg_viz = frame.copy()
    cv2.rectangle(ffmpeg_viz, (ffmpeg_x1, ffmpeg_y1), (ffmpeg_x2, ffmpeg_y2), (0, 255, 0), 2)
    cv2.putText(ffmpeg_viz, "FFmpeg", (ffmpeg_x1, ffmpeg_y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    axes[0, 2].imshow(cv2.cvtColor(ffmpeg_viz, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('FFmpeg Position (GREEN)')
    axes[0, 2].axis('off')
    
    # OpenCV convention visualization
    opencv_viz = frame.copy()
    cv2.rectangle(opencv_viz, (opencv_x1, opencv_y1), (opencv_x2, opencv_y2), (255, 0, 0), 2)
    cv2.putText(opencv_viz, "OpenCV", (opencv_x1, opencv_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    axes[1, 0].imshow(cv2.cvtColor(opencv_viz, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('OpenCV Position (BLUE)')
    axes[1, 0].axis('off')
    
    # Both overlaid
    both_viz = frame.copy()
    cv2.rectangle(both_viz, (ffmpeg_x1, ffmpeg_y1), (ffmpeg_x2, ffmpeg_y2), (0, 255, 0), 2)
    cv2.rectangle(both_viz, (opencv_x1, opencv_y1), (opencv_x2, opencv_y2), (255, 0, 0), 2)
    axes[1, 1].imshow(cv2.cvtColor(both_viz, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Both (Green=FFmpeg, Blue=OpenCV)')
    axes[1, 1].axis('off')
    
    # Actual rendered text simulation
    rendered = frame.copy()
    # FFmpeg places text with top-left at (x, y)
    cv2.putText(rendered, text, (x, y + text_height), font, font_scale, (255, 255, 255), thickness + 2)
    cv2.putText(rendered, text, (x, y + text_height), font, font_scale, (0, 0, 0), thickness)
    axes[1, 2].imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Simulated FFmpeg Render')
    axes[1, 2].axis('off')
    
    plt.suptitle(f"Debug: 'beginning' position analysis")
    plt.tight_layout()
    
    output_dir = Path("tests/debug_beginning_comprehensive")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "position_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_dir}/position_analysis.png")
    
    cap.release()
    
    # Additional mask analysis
    print("\n" + "="*70)
    print("MASK QUALITY ANALYSIS")
    print("="*70)
    
    # Check Maria's expected position
    h, w = frame.shape[:2]
    center_region = background_mask[h//4:3*h//4, w//4:3*w//4]
    center_fg_ratio = 1 - (np.mean(center_region == 255))
    
    print(f"Center region foreground ratio: {center_fg_ratio:.1%}")
    if center_fg_ratio < 0.2:
        print("⚠️ WARNING: Very little foreground in center - rembg might be failing!")
    
    # Check overall mask
    total_fg_ratio = 1 - (np.mean(background_mask == 255))
    print(f"Total foreground ratio: {total_fg_ratio:.1%}")
    
    return background_mask


def test_fixed_algorithm():
    """Test the fixed algorithm directly."""
    print("\n" + "="*70)
    print("TESTING FIXED ALGORITHM")
    print("="*70)
    
    from utils.text_placement.proximity_text_placer_v2 import ProximityTextPlacerV2
    
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    placer = ProximityTextPlacerV2(video_path, "tests/debug_algorithm")
    
    # Test with just the word "beginning" at its timestamp
    test_word = {
        'word': 'beginning',
        'start': 10.1,
        'end': 11.479
    }
    
    print(f"\nTesting word: '{test_word['word']}' at {test_word['start']}s")
    
    # Get position
    result = placer.place_word_with_full_background_check(
        test_word['word'], 
        test_word['start']
    )
    
    print(f"\nAlgorithm result:")
    print(f"  Position: ({result['x']}, {result['y']})")
    print(f"  Font scale: {result['font_scale']}")
    print(f"  Font size: {result['fontsize']}")
    print(f"  Valid positions found: {result.get('valid_positions_found', 0)}")
    
    if result.get('valid_positions_found', 0) == 0:
        print("\n⚠️ NO VALID POSITIONS FOUND!")
        print("This means the algorithm couldn't find any position where")
        print("the entire word fits in the background!")


if __name__ == "__main__":
    # Check all position files
    check_all_position_files()
    
    # Use the most recent fixed position
    x, y = 330, 170  # From the fixed file
    fontsize = 48
    timestamp = 10.1
    
    # Debug this position
    mask = debug_position_with_mask(x, y, fontsize, timestamp)
    
    # Test the algorithm directly
    test_fixed_algorithm()