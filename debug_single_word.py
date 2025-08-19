#!/usr/bin/env python3
"""
Debug single word placement - focusing on the first word "Let's"
Ensures correct timing and auto-sizes text to fit background region.
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


def extract_background_at_time(video_path, timestamp):
    """Extract background using rembg at specific timestamp."""
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
    
    cap.release()
    return frame, background_mask


def find_best_region_for_text(background_mask, text, font_scale_start=1.0):
    """
    Find the best region for text and calculate optimal font size.
    Returns position and font size that fits well in the background.
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        background_mask, connectivity=8
    )
    
    if num_labels <= 1:
        return None, None, None
    
    # Find suitable regions
    best_region = None
    best_area = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_region = {
                'x': stats[i, cv2.CC_STAT_LEFT],
                'y': stats[i, cv2.CC_STAT_TOP],
                'w': stats[i, cv2.CC_STAT_WIDTH],
                'h': stats[i, cv2.CC_STAT_HEIGHT],
                'area': area
            }
    
    if not best_region:
        return None, None, None
    
    # Calculate optimal font size for the text to fit in the region
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Start with a reasonable font scale
    font_scale = font_scale_start
    max_width = best_region['w'] * 0.8  # Use 80% of region width
    max_height = best_region['h'] * 0.3  # Use 30% of region height
    
    # Adjust font scale to fit
    while font_scale > 0.3:
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        if text_size[0] <= max_width and text_size[1] <= max_height:
            break
        font_scale *= 0.9
    
    # Calculate position (centered in upper portion of region)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = best_region['x'] + (best_region['w'] - text_size[0]) // 2
    y = best_region['y'] + best_region['h'] // 3  # Upper third
    
    return (x, y), font_scale, best_region


def debug_first_word():
    """Debug the placement and timing of the first word 'Let's'."""
    
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
    print(f"Total frames: {video_info['nb_frames']}")
    
    # Get first word from original transcript
    first_word = words[0]
    print("\n" + "="*70)
    print("FIRST WORD FROM TRANSCRIPT")
    print("="*70)
    print(f"Word: '{first_word['word']}'")
    print(f"Original timing: {first_word['start']:.3f}s - {first_word['end']:.3f}s")
    print(f"Duration: {first_word['end'] - first_word['start']:.3f}s")
    
    # Scene 1 starts at 7.92s in the original video
    # The scene video starts at 0s
    scene_offset = 7.92
    
    # Adjust timing for scene video
    word_start_in_scene = first_word['start'] - scene_offset
    word_end_in_scene = first_word['end'] - scene_offset
    
    print(f"\nAdjusted for scene video:")
    print(f"Start in scene: {word_start_in_scene:.3f}s")
    print(f"End in scene: {word_end_in_scene:.3f}s")
    
    # Extract background at word start time
    print("\n" + "="*70)
    print("EXTRACTING BACKGROUND")
    print("="*70)
    print(f"Extracting frame at {word_start_in_scene:.3f}s...")
    
    frame, background_mask = extract_background_at_time(video_path, max(0, word_start_in_scene))
    
    if frame is None:
        print("Error: Could not extract frame")
        return
    
    # Find best position and size
    position, font_scale, region = find_best_region_for_text(background_mask, first_word['word'])
    
    if position is None:
        print("Error: Could not find suitable background region")
        return
    
    print(f"\nBest region found:")
    print(f"  Position: ({position[0]}, {position[1]})")
    print(f"  Region: {region['w']}x{region['h']} at ({region['x']}, {region['y']})")
    print(f"  Auto-calculated font scale: {font_scale:.2f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original frame
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original Frame at t={word_start_in_scene:.3f}s')
    axes[0, 0].axis('off')
    
    # Background mask
    axes[0, 1].imshow(background_mask, cmap='gray')
    axes[0, 1].set_title('Background Mask (white = background)')
    axes[0, 1].axis('off')
    
    # Background region highlighted
    region_viz = frame.copy()
    cv2.rectangle(region_viz, 
                  (region['x'], region['y']), 
                  (region['x'] + region['w'], region['y'] + region['h']),
                  (0, 255, 0), 3)
    axes[0, 2].imshow(cv2.cvtColor(region_viz, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Best Background Region')
    axes[0, 2].axis('off')
    
    # Text with different sizes for comparison
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Too large (original size)
    large_scale = 2.0
    frame_large = frame.copy()
    cv2.putText(frame_large, first_word['word'], position,
                font, large_scale, (255, 255, 255), thickness + 2)
    cv2.putText(frame_large, first_word['word'], position,
                font, large_scale, (0, 0, 0), thickness)
    axes[1, 0].imshow(cv2.cvtColor(frame_large, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Too Large (scale={large_scale:.1f})')
    axes[1, 0].axis('off')
    
    # Auto-sized
    frame_auto = frame.copy()
    cv2.putText(frame_auto, first_word['word'], position,
                font, font_scale, (255, 255, 255), thickness + 2)
    cv2.putText(frame_auto, first_word['word'], position,
                font, font_scale, (0, 0, 0), thickness)
    axes[1, 1].imshow(cv2.cvtColor(frame_auto, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Auto-sized (scale={font_scale:.2f})')
    axes[1, 1].axis('off')
    
    # Timing visualization
    axes[1, 2].set_xlim(0, 5)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].barh(0.5, word_end_in_scene - word_start_in_scene, 
                    left=word_start_in_scene, height=0.3, color='green')
    axes[1, 2].set_xlabel('Time (seconds)')
    axes[1, 2].set_title(f"Word timing: {word_start_in_scene:.3f}s - {word_end_in_scene:.3f}s")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f"Debug: First word '{first_word['word']}'")
    plt.tight_layout()
    plt.savefig(output_dir / "first_word_debug.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_dir}/first_word_debug.png")
    
    # Create a test video with just this word
    print("\n" + "="*70)
    print("CREATING TEST VIDEO")
    print("="*70)
    
    output_video = output_dir / "test_first_word.mp4"
    
    # Create FFmpeg command with proper timing
    # Escape special characters and format timing properly
    word_text = first_word['word'].replace("'", "\\'")
    start_time = max(0, word_start_in_scene)
    end_time = word_end_in_scene
    
    filter_complex = (
        f"drawtext=text='{word_text}'"
        f":x={position[0]}:y={position[1]}"
        f":fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize={int(font_scale * 48)}"
        f":fontcolor=white:borderw=2:bordercolor=black"
        f":enable='between(t\\,{start_time:.3f}\\,{end_time:.3f})'"
    )
    
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", filter_complex,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "copy",
        "-t", "5",  # Just first 5 seconds
        str(output_video)
    ]
    
    print(f"FFmpeg command:\n{' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if output_video.exists():
        size_mb = output_video.stat().st_size / (1024 * 1024)
        print(f"\n✓ Test video created: {output_video} ({size_mb:.2f} MB)")
        print(f"  Word appears at {word_start_in_scene:.3f}s")
        print(f"  Word disappears at {word_end_in_scene:.3f}s")
        print(f"  Font scale: {font_scale:.2f}")
        print(f"  Position: ({position[0]}, {position[1]})")
    else:
        print(f"\n✗ Failed to create test video")
        print(f"Error: {result.stderr}")
    
    # Save corrected data
    corrected_word = {
        "word": first_word['word'],
        "start_in_scene": float(word_start_in_scene),
        "end_in_scene": float(word_end_in_scene),
        "x": int(position[0]),
        "y": int(position[1]),
        "font_scale": float(font_scale),
        "fontsize": int(font_scale * 48),
        "region": {
            "x": int(region['x']),
            "y": int(region['y']),
            "w": int(region['w']),
            "h": int(region['h']),
            "area": int(region['area'])
        }
    }
    
    with open(output_dir / "corrected_first_word.json", 'w') as f:
        json.dump(corrected_word, f, indent=2)
    
    print(f"\n✓ Corrected data saved to: {output_dir}/corrected_first_word.json")


if __name__ == "__main__":
    debug_first_word()