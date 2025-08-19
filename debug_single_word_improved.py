#!/usr/bin/env python3
"""
Improved debug for single word placement with better region detection.
Finds multiple background regions and selects the most suitable one.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import subprocess
from PIL import Image
from rembg import remove
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_video_info(video_path):
    """Get detailed video information."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    stream = data['streams'][0]
    format_info = data['format']
    
    # Parse frame rate
    fps_str = stream.get('r_frame_rate', '60/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)
    
    return {
        'width': int(stream['width']),
        'height': int(stream['height']),
        'fps': fps,
        'duration': float(format_info['duration']),
        'nb_frames': int(stream.get('nb_frames', 0))
    }


def extract_background_regions(video_path, timestamp):
    """
    Extract background using rembg and segment into multiple regions.
    Returns frame and segmented background regions.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Seek to frame
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame at {timestamp}s")
        return None, None
    
    # Apply rembg
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    nobg_img = remove(pil_image)
    nobg_array = np.array(nobg_img)
    
    # Get alpha channel
    if nobg_array.shape[2] == 4:
        alpha = nobg_array[:, :, 3]
    else:
        alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
    
    # Background is where alpha is low (transparent)
    background_mask = np.where(alpha < 128, 255, 0).astype(np.uint8)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
    
    # Now segment the background into regions based on spatial location
    # Divide into a grid and find background areas in each cell
    h, w = frame.shape[:2]
    regions = []
    
    # Try different grid sizes
    for grid_size in [(3, 3), (4, 4), (5, 5)]:
        rows, cols = grid_size
        cell_h = h // rows
        cell_w = w // cols
        
        for r in range(rows):
            for c in range(cols):
                # Extract cell
                y1 = r * cell_h
                y2 = min((r + 1) * cell_h, h)
                x1 = c * cell_w
                x2 = min((c + 1) * cell_w, w)
                
                cell_mask = background_mask[y1:y2, x1:x2]
                
                # Check if this cell has significant background
                bg_pixels = np.sum(cell_mask == 255)
                total_pixels = cell_mask.size
                bg_ratio = bg_pixels / total_pixels if total_pixels > 0 else 0
                
                if bg_ratio > 0.3:  # At least 30% background
                    regions.append({
                        'x': x1,
                        'y': y1,
                        'w': x2 - x1,
                        'h': y2 - y1,
                        'bg_ratio': bg_ratio,
                        'area': (x2 - x1) * (y2 - y1),
                        'grid': grid_size
                    })
    
    # Also find connected components for more organic regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        background_mask, connectivity=8
    )
    
    for i in range(1, min(num_labels, 10)):  # Limit to 10 components
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 5000:  # Minimum area threshold
            regions.append({
                'x': stats[i, cv2.CC_STAT_LEFT],
                'y': stats[i, cv2.CC_STAT_TOP],
                'w': stats[i, cv2.CC_STAT_WIDTH],
                'h': stats[i, cv2.CC_STAT_HEIGHT],
                'bg_ratio': 1.0,  # Fully background
                'area': area,
                'grid': 'connected'
            })
    
    cap.release()
    return frame, background_mask, regions


def find_best_region_for_text(regions, text, frame_width, frame_height):
    """
    Find the best region for text placement and calculate optimal font size.
    Prioritizes regions that are:
    1. Not too large (avoid full-frame regions)
    2. Have good background ratio
    3. Are in reasonable positions (avoid extreme edges)
    """
    if not regions:
        return None, None, None
    
    # Score each region
    scored_regions = []
    for region in regions:
        score = 0
        
        # Prefer medium-sized regions (not too big, not too small)
        area_ratio = region['area'] / (frame_width * frame_height)
        if 0.05 < area_ratio < 0.3:  # 5% to 30% of frame
            score += 50
        elif area_ratio <= 0.05:
            score += 20
        else:  # Too large
            score -= 30
        
        # Prefer high background ratio
        score += region['bg_ratio'] * 30
        
        # Prefer regions not at extreme edges
        center_x = region['x'] + region['w'] / 2
        center_y = region['y'] + region['h'] / 2
        
        # Distance from center (normalized)
        dist_from_center = np.sqrt(
            ((center_x - frame_width/2) / (frame_width/2))**2 +
            ((center_y - frame_height/2) / (frame_height/2))**2
        )
        
        # Prefer regions not too far from center
        if dist_from_center < 0.7:
            score += 20
        
        # Prefer upper regions for text
        if region['y'] < frame_height * 0.4:
            score += 10
        
        scored_regions.append((region, score))
    
    # Sort by score
    scored_regions.sort(key=lambda x: x[1], reverse=True)
    
    # Get best region
    best_region = scored_regions[0][0]
    
    # Calculate optimal font size for the text to fit in the region
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Start with a reasonable font scale
    font_scale = 1.0
    max_width = best_region['w'] * 0.8  # Use 80% of region width
    max_height = best_region['h'] * 0.5  # Use 50% of region height for better visibility
    
    # Adjust font scale to fit
    while font_scale > 0.3:
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        if text_size[0] <= max_width and text_size[1] <= max_height:
            break
        font_scale *= 0.9
    
    # Calculate position (centered in upper portion of region)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = best_region['x'] + (best_region['w'] - text_size[0]) // 2
    y = best_region['y'] + min(best_region['h'] // 3, 50)  # Upper third but not too high
    
    # Ensure text is fully visible
    x = max(10, min(x, frame_width - text_size[0] - 10))
    y = max(text_size[1] + 10, min(y, frame_height - 10))
    
    return (x, y), font_scale, best_region


def debug_first_word_improved():
    """Debug the placement and timing of the first word 'Let's' with improved region detection."""
    
    # Paths
    video_path = Path("uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4")
    transcript_path = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_words.json")
    output_dir = Path("tests/debug_word_placement")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    if not transcript_path.exists():
        print(f"Error: Transcript not found at {transcript_path}")
        return
    
    # Load transcript
    with open(transcript_path) as f:
        transcript_data = json.load(f)
        words = transcript_data.get('words', [])
    
    # Get video info
    video_info = get_video_info(video_path)
    print("="*70)
    print("VIDEO INFORMATION")
    print("="*70)
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    print(f"FPS: {video_info['fps']}")
    print(f"Duration: {video_info['duration']:.2f}s")
    
    # Get first word from original transcript
    first_word = words[0]
    print("\n" + "="*70)
    print("FIRST WORD FROM TRANSCRIPT")
    print("="*70)
    print(f"Word: '{first_word['word']}'")
    print(f"Original timing: {first_word['start']:.3f}s - {first_word['end']:.3f}s")
    
    # Scene 1 starts at 7.92s in the original video
    scene_offset = 7.92
    
    # Adjust timing for scene video
    word_start_in_scene = first_word['start'] - scene_offset
    word_end_in_scene = first_word['end'] - scene_offset
    
    print(f"\nAdjusted for scene video:")
    print(f"Start in scene: {word_start_in_scene:.3f}s")
    print(f"End in scene: {word_end_in_scene:.3f}s")
    
    # Extract background at word start time
    print("\n" + "="*70)
    print("EXTRACTING BACKGROUND REGIONS")
    print("="*70)
    
    frame, background_mask, regions = extract_background_regions(video_path, max(0, word_start_in_scene))
    
    if frame is None:
        print("Error: Could not extract frame")
        return
    
    print(f"Found {len(regions)} potential regions")
    
    # Find best position and size
    position, font_scale, best_region = find_best_region_for_text(
        regions, first_word['word'], 
        video_info['width'], video_info['height']
    )
    
    if position is None:
        print("Error: Could not find suitable background region")
        return
    
    print(f"\nBest region selected:")
    print(f"  Position: ({position[0]}, {position[1]})")
    print(f"  Region: {best_region['w']}x{best_region['h']} at ({best_region['x']}, {best_region['y']})")
    print(f"  Background ratio: {best_region['bg_ratio']:.2%}")
    print(f"  Auto-calculated font scale: {font_scale:.2f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original frame
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original Frame at t={word_start_in_scene:.3f}s')
    axes[0, 0].axis('off')
    
    # Background mask
    axes[0, 1].imshow(background_mask, cmap='gray')
    axes[0, 1].set_title('Background Mask')
    axes[0, 1].axis('off')
    
    # All regions
    regions_viz = frame.copy()
    for i, region in enumerate(regions[:10]):  # Show first 10 regions
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(regions_viz, 
                      (region['x'], region['y']), 
                      (region['x'] + region['w'], region['y'] + region['h']),
                      color, 2)
    axes[0, 2].imshow(cv2.cvtColor(regions_viz, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'All Regions ({len(regions)} total)')
    axes[0, 2].axis('off')
    
    # Best region highlighted
    best_region_viz = frame.copy()
    cv2.rectangle(best_region_viz, 
                  (best_region['x'], best_region['y']), 
                  (best_region['x'] + best_region['w'], best_region['y'] + best_region['h']),
                  (0, 255, 0), 3)
    axes[1, 0].imshow(cv2.cvtColor(best_region_viz, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Selected Best Region')
    axes[1, 0].axis('off')
    
    # Text placement
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    frame_with_text = frame.copy()
    cv2.putText(frame_with_text, first_word['word'], position,
                font, font_scale, (255, 255, 255), thickness + 2)
    cv2.putText(frame_with_text, first_word['word'], position,
                font, font_scale, (0, 0, 0), thickness)
    axes[1, 1].imshow(cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Final Placement (scale={font_scale:.2f})')
    axes[1, 1].axis('off')
    
    # Timing visualization
    axes[1, 2].set_xlim(-0.5, 3)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].barh(0.5, word_end_in_scene - word_start_in_scene, 
                    left=word_start_in_scene, height=0.3, color='green')
    axes[1, 2].set_xlabel('Time (seconds)')
    axes[1, 2].set_title(f"Timing: {word_start_in_scene:.3f}s - {word_end_in_scene:.3f}s")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f"Improved Debug: '{first_word['word']}'")
    plt.tight_layout()
    plt.savefig(output_dir / "first_word_debug_improved.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_dir}/first_word_debug_improved.png")
    
    # Create a test video with just this word
    print("\n" + "="*70)
    print("CREATING TEST VIDEO")
    print("="*70)
    
    output_video = output_dir / "test_first_word_improved.mp4"
    
    # Create FFmpeg command
    word_text = first_word['word'].replace("'", "\\'")
    start_time = max(0, word_start_in_scene)
    end_time = word_end_in_scene
    fontsize = int(font_scale * 48)
    
    filter_complex = (
        f"drawtext=text='{word_text}'"
        f":x={position[0]}:y={position[1]}"
        f":fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize={fontsize}"
        f":fontcolor=white:borderw=2:bordercolor=black"
        f":enable='between(t\\,{start_time:.3f}\\,{end_time:.3f})'"
    )
    
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", filter_complex,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "copy",
        "-t", "3",  # Just first 3 seconds
        str(output_video)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if output_video.exists():
        size_mb = output_video.stat().st_size / (1024 * 1024)
        print(f"\n✓ Test video created: {output_video} ({size_mb:.2f} MB)")
        print(f"  Word timing: {start_time:.3f}s - {end_time:.3f}s")
        print(f"  Font size: {fontsize}")
        print(f"  Position: ({position[0]}, {position[1]})")
    else:
        print(f"\n✗ Failed to create test video")
    
    # Save corrected data
    corrected_data = {
        "word": first_word['word'],
        "original_timing": {
            "start": first_word['start'],
            "end": first_word['end']
        },
        "scene_timing": {
            "start": float(start_time),
            "end": float(end_time)
        },
        "placement": {
            "x": int(position[0]),
            "y": int(position[1]),
            "font_scale": float(font_scale),
            "fontsize": fontsize
        },
        "region": {
            "x": int(best_region['x']),
            "y": int(best_region['y']),
            "w": int(best_region['w']),
            "h": int(best_region['h']),
            "bg_ratio": float(best_region['bg_ratio'])
        }
    }
    
    with open(output_dir / "first_word_improved.json", 'w') as f:
        json.dump(corrected_data, f, indent=2)
    
    print(f"\n✓ Data saved to: {output_dir}/first_word_improved.json")


if __name__ == "__main__":
    debug_first_word_improved()