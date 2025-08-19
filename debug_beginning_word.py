#!/usr/bin/env python3
"""
Debug why the word "beginning" appears on Maria's face.
Extract the background mask at that exact timestamp to see what went wrong.
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


def debug_beginning_word():
    """Debug the placement of 'beginning' word."""
    
    # Load the word positions to find when "beginning" appears
    positions_file = Path("uploads/assets/videos/do_re_mi/scenes/backgrounds_validated/word_positions_v2.json")
    if positions_file.exists():
        with open(positions_file) as f:
            words = json.load(f)
    else:
        # Try other location
        positions_file = Path("uploads/assets/videos/do_re_mi/inferences/scene_001_validated.json")
        with open(positions_file) as f:
            data = json.load(f)
            words = data['text_overlays']
    
    # Find "beginning" in the word list
    beginning_word = None
    word_index = None
    for i, word in enumerate(words):
        if word['word'].lower() == 'beginning':
            beginning_word = word
            word_index = i
            break
    
    if not beginning_word:
        print("Word 'beginning' not found in transcript!")
        return
    
    print("="*70)
    print("DEBUGGING 'BEGINNING' PLACEMENT")
    print("="*70)
    print(f"Word index: {word_index}")
    print(f"Word: '{beginning_word['word']}'")
    print(f"Timing: {beginning_word['start']:.3f}s - {beginning_word['end']:.3f}s")
    print(f"Position: ({beginning_word['x']}, {beginning_word['y']})")
    print(f"Font size: {beginning_word.get('fontsize', 48)}")
    
    # Also check surrounding words
    if word_index > 0:
        prev_word = words[word_index - 1]
        print(f"\nPrevious word: '{prev_word['word']}' at ({prev_word['x']}, {prev_word['y']})")
    if word_index < len(words) - 1:
        next_word = words[word_index + 1]
        print(f"Next word: '{next_word['word']}' at ({next_word['x']}, {next_word['y']})")
    
    # Extract frame and background mask at this timestamp
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Go to the timestamp
    timestamp = beginning_word['start']
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame at {timestamp}s")
        return
    
    print(f"\n" + "="*70)
    print("EXTRACTING BACKGROUND MASK")
    print("="*70)
    print(f"Frame at {timestamp:.3f}s (frame {frame_number})")
    
    # Apply rembg to get background/foreground
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    print("Applying rembg...")
    nobg_img = remove(pil_image)
    nobg_array = np.array(nobg_img)
    
    # Get alpha channel
    if nobg_array.shape[2] == 4:
        alpha = nobg_array[:, :, 3]
    else:
        alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
    
    # Background is where alpha is low (transparent)
    # Foreground is where alpha is high (opaque)
    background_mask = np.where(alpha < 128, 255, 0).astype(np.uint8)
    foreground_mask = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    
    # Check what's at the word position
    x, y = beginning_word['x'], beginning_word['y']
    
    # Get text bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = beginning_word.get('fontsize', 48) / 48.0
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        beginning_word['word'], font, font_scale, thickness
    )
    
    # Calculate actual text region
    # OpenCV uses bottom-left as reference for drawText
    x1 = x
    y1 = y - text_height
    x2 = x + text_width
    y2 = y + baseline
    
    print(f"\nText bounding box: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"Text dimensions: {text_width}x{text_height} pixels")
    
    # Check if text region is in background
    text_region = background_mask[max(0, y1):min(background_mask.shape[0], y2), 
                                  max(0, x1):min(background_mask.shape[1], x2)]
    
    if text_region.size > 0:
        bg_ratio = np.mean(text_region == 255)
        print(f"Background ratio in text region: {bg_ratio:.1%}")
        
        if bg_ratio < 0.5:
            print("⚠️ WARNING: Text is mostly in FOREGROUND!")
        else:
            print("✓ Text is mostly in background")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original frame
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original Frame at t={timestamp:.3f}s')
    axes[0, 0].axis('off')
    
    # Background mask
    axes[0, 1].imshow(background_mask, cmap='gray')
    axes[0, 1].set_title('Background Mask (white = background)')
    axes[0, 1].axis('off')
    
    # Foreground mask
    axes[0, 2].imshow(foreground_mask, cmap='gray')
    axes[0, 2].set_title('Foreground Mask (white = foreground/Maria)')
    axes[0, 2].axis('off')
    
    # Show text position on original
    frame_with_text = frame.copy()
    cv2.rectangle(frame_with_text, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_with_text, beginning_word['word'], (x, y),
                font, font_scale, (255, 255, 255), thickness + 2)
    cv2.putText(frame_with_text, beginning_word['word'], (x, y),
                font, font_scale, (0, 0, 0), thickness)
    axes[1, 0].imshow(cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Text Position')
    axes[1, 0].axis('off')
    
    # Overlay showing problem
    overlay = frame.copy()
    # Make foreground red-tinted
    overlay[foreground_mask == 255, 2] = 255  # Red channel
    overlay[foreground_mask == 255, 0] = overlay[foreground_mask == 255, 0] // 2
    overlay[foreground_mask == 255, 1] = overlay[foreground_mask == 255, 1] // 2
    # Draw text box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Foreground (red) vs Text Box (green)')
    axes[1, 1].axis('off')
    
    # Zoom in on problem area
    zoom_size = 200
    cx, cy = x + text_width // 2, y - text_height // 2
    x_start = max(0, cx - zoom_size)
    x_end = min(frame.shape[1], cx + zoom_size)
    y_start = max(0, cy - zoom_size)
    y_end = min(frame.shape[0], cy + zoom_size)
    
    zoom_region = overlay[y_start:y_end, x_start:x_end]
    axes[1, 2].imshow(cv2.cvtColor(zoom_region, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Zoomed Problem Area')
    axes[1, 2].axis('off')
    
    plt.suptitle(f"Debug: Why 'beginning' appears on Maria's face")
    plt.tight_layout()
    
    output_dir = Path("tests/debug_beginning")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "beginning_debug.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_dir}/beginning_debug.png")
    
    # Analyze the issue
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Check if rembg is detecting Maria correctly at this timestamp
    foreground_pixels = np.sum(foreground_mask == 255)
    total_pixels = foreground_mask.size
    foreground_ratio = foreground_pixels / total_pixels
    
    print(f"Foreground coverage: {foreground_ratio:.1%} of frame")
    
    if foreground_ratio < 0.05:
        print("⚠️ PROBLEM: Very little foreground detected - rembg might be failing!")
    elif foreground_ratio > 0.5:
        print("⚠️ PROBLEM: Too much foreground detected - rembg might be over-detecting!")
    
    # Check Maria's expected position
    # Maria is typically in the center-left of the frame
    height, width = frame.shape[:2]
    maria_region = foreground_mask[height//4:3*height//4, width//4:3*width//4]
    maria_detected = np.mean(maria_region == 255)
    
    print(f"Foreground in center region: {maria_detected:.1%}")
    
    if maria_detected < 0.1:
        print("⚠️ CRITICAL: Maria not detected in center - rembg failed at this frame!")
        print("This explains why 'beginning' was placed on her face!")
    
    cap.release()
    
    return beginning_word, background_mask, foreground_mask


if __name__ == "__main__":
    word_data, bg_mask, fg_mask = debug_beginning_word()