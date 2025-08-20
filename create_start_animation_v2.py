#!/usr/bin/env python3
"""
Create animated "START" text - Version 2 with fixes:
1. Accurate per-frame masking
2. Text stays perfectly centered
3. Letter-by-letter fade out effect
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from rembg import remove, new_session
import imageio

sys.path.insert(0, os.path.expanduser("~/sam2"))


def create_start_animation_v2():
    """Create improved animation with accurate masking and fade effects"""
    
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    
    # Load video
    video_path = os.path.join(backend_dir, "uploads/assets/videos/do_re_mi.mov")
    cap = cv2.VideoCapture(video_path)
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {fps} fps, {total_frames} total frames")
    
    # Animation parameters
    start_frame = 890
    num_frames = 120  # 4 seconds at 30fps (more time for fade out)
    
    # Animation phases (in frames)
    phase1_end = 30    # Frames 0-30: Text shrinking in foreground
    phase2_end = 50    # Frames 31-50: Text moving behind
    phase3_end = 80    # Frames 51-80: Text stable behind (1+ second)
    phase4_end = 120   # Frames 81-120: Letter-by-letter fade out
    
    # Scale animation - ONLY SIZE CHANGES, NO POSITION
    start_scale = 2.0   # 200% size at start
    end_scale = 1.0     # 100% size when behind
    
    frames = []
    
    print(f"Extracting {num_frames} frames starting from frame {start_frame}...")
    
    # Extract all frames first
    for i in range(num_frames):
        frame_idx = start_frame + i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    # Initialize rembg session
    print("Initializing background removal...")
    session = new_session('u2net')
    
    # Get dimensions
    h, w = frames[0].shape[:2]
    
    # Create font
    font_size_base = min(150, int(h * 0.28))  # Base font size
    
    try:
        font_path = "/System/Library/Fonts/Helvetica.ttc"
        if not os.path.exists(font_path):
            font_path = None
    except:
        font_path = None
    
    # Fixed center position for text
    CENTER_X = w // 2
    CENTER_Y = int(h * 0.45)  # Slightly above center
    
    # Process each frame
    processed_frames = []
    
    print("Creating animation frames with accurate masking...")
    
    for i, frame in enumerate(frames):
        # Calculate animation phase
        if i <= phase1_end:
            phase = "shrinking"
            phase_progress = i / phase1_end
            current_scale = start_scale - (start_scale - 1.3) * phase_progress
            occlusion = False
            opacity = 1.0
            fade_letter = -1  # No letter fading
            
        elif i <= phase2_end:
            phase = "moving_behind"
            phase_progress = (i - phase1_end) / (phase2_end - phase1_end)
            current_scale = 1.3 - (1.3 - end_scale) * phase_progress
            occlusion = phase_progress > 0.3  # Start occlusion early in transition
            opacity = 1.0
            fade_letter = -1
            
        elif i <= phase3_end:
            phase = "stable_behind"
            phase_progress = (i - phase2_end) / (phase3_end - phase2_end)
            current_scale = end_scale
            occlusion = True
            opacity = 1.0
            fade_letter = -1
            
        else:
            phase = "fading_out"
            phase_progress = (i - phase3_end) / (phase4_end - phase3_end)
            current_scale = end_scale
            occlusion = True
            opacity = 1.0
            # Determine which letters are fading
            # START has 5 letters, fade them one by one
            fade_letter = int(phase_progress * 6)  # 0-5, where 5 means all faded
        
        # CRITICAL: Get accurate foreground mask for THIS specific frame
        if occlusion or phase == "moving_behind":
            # Process this specific frame with rembg
            img_pil = Image.fromarray(frame)
            img_no_bg = remove(img_pil, session=session)
            img_no_bg_np = np.array(img_no_bg)
            
            if img_no_bg_np.shape[2] == 4:
                foreground_mask = img_no_bg_np[:, :, 3] > 128
            else:
                foreground_mask = np.zeros((h, w), dtype=bool)
        else:
            # No occlusion needed
            foreground_mask = np.zeros((h, w), dtype=bool)
        
        # Create text layer for this frame
        text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # Scale font size
        current_font_size = int(font_size_base * current_scale)
        
        if font_path:
            font = ImageFont.truetype(font_path, current_font_size)
        else:
            font = ImageFont.load_default()
        
        # Draw text - handle letter-by-letter fade
        if fade_letter < 0:
            # Normal text (no fading)
            text = "START"
            
            # Get text dimensions
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # CRITICAL: Calculate position to keep text CENTERED
            text_x = CENTER_X - text_width // 2
            text_y = CENTER_Y - text_height // 2
            
            # Draw shadow
            shadow_offset = max(2, int(4 * current_scale))
            draw.text((text_x + shadow_offset, text_y + shadow_offset), text, 
                     font=font, fill=(0, 0, 0, 100))
            
            # Draw outline
            outline_width = max(1, int(2 * current_scale))
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if abs(dx) == outline_width or abs(dy) == outline_width:
                        draw.text((text_x + dx, text_y + dy), text, 
                                 font=font, fill=(255, 255, 255, 150))
            
            # Main text
            draw.text((text_x, text_y), text, font=font, fill=(255, 220, 0, 255))
            
        else:
            # Letter-by-letter fade effect
            letters = list("START")
            current_x = 0
            
            # First calculate total width to center properly
            total_width = 0
            for letter in letters:
                bbox = draw.textbbox((0, 0), letter, font=font)
                total_width += bbox[2] - bbox[0]
            
            # Starting position to keep centered
            text_x = CENTER_X - total_width // 2
            text_y = CENTER_Y - current_font_size // 2
            
            # Draw each letter with individual opacity
            for idx, letter in enumerate(letters):
                bbox = draw.textbbox((0, 0), letter, font=font)
                letter_width = bbox[2] - bbox[0]
                
                # Calculate opacity for this letter
                if idx < fade_letter:
                    # This letter is fading/faded
                    if idx == fade_letter - 1:
                        # Currently fading letter
                        letter_opacity = int(255 * (1 - (phase_progress * 6 - idx)))
                    else:
                        # Already faded
                        letter_opacity = 0
                else:
                    # Not yet fading
                    letter_opacity = 255
                
                if letter_opacity > 0:
                    # Shadow
                    shadow_offset = 3
                    draw.text((text_x + current_x + shadow_offset, text_y + shadow_offset), 
                             letter, font=font, fill=(0, 0, 0, min(100, letter_opacity // 2)))
                    
                    # Outline
                    for dx in [-2, 2]:
                        for dy in [-2, 2]:
                            draw.text((text_x + current_x + dx, text_y + dy), letter,
                                     font=font, fill=(255, 255, 255, min(150, letter_opacity)))
                    
                    # Main letter with fade
                    draw.text((text_x + current_x, text_y), letter, 
                             font=font, fill=(255, 220, 0, letter_opacity))
                
                current_x += letter_width
        
        # Convert to numpy
        text_layer_np = np.array(text_layer)
        
        # Composite with frame
        result = frame.copy()
        
        if occlusion:
            # Apply text ONLY in background pixels
            bg_mask = ~foreground_mask
            for c in range(3):
                text_visible = (text_layer_np[:, :, 3] > 0) & bg_mask
                if np.any(text_visible):
                    alpha_blend = text_layer_np[text_visible, 3] / 255.0
                    result[text_visible, c] = (
                        result[text_visible, c] * (1 - alpha_blend) + 
                        text_layer_np[text_visible, c] * alpha_blend
                    ).astype(np.uint8)
        else:
            # Text in foreground
            for c in range(3):
                text_visible = text_layer_np[:, :, 3] > 0
                if np.any(text_visible):
                    alpha_blend = text_layer_np[text_visible, 3] / 255.0
                    result[text_visible, c] = (
                        result[text_visible, c] * (1 - alpha_blend) + 
                        text_layer_np[text_visible, c] * alpha_blend
                    ).astype(np.uint8)
        
        processed_frames.append(result)
        
        # Progress indicator
        if i % 10 == 0:
            status = f"  Frame {i}/{num_frames}: {phase} (scale={current_scale:.2f}"
            if fade_letter >= 0:
                status += f", fading letter {fade_letter}"
            status += ")"
            print(status)
    
    print(f"Processed {len(processed_frames)} frames")
    
    # Save as video
    output_video = os.path.join(backend_dir, "start_animation_v2.mp4")
    print(f"Saving video to {output_video}...")
    
    writer = imageio.get_writer(output_video, fps=30)  # Output at 30fps
    for frame in processed_frames[::2]:  # Sample to 30fps if needed
        writer.append_data(frame)
    writer.close()
    
    print(f"Video saved: {output_video}")
    
    # Save key frames
    key_frames = [0, 25, 40, 60, 80, 90, 100, 110, 119]
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(key_frames):
        if frame_idx < len(processed_frames):
            axes[idx].imshow(processed_frames[frame_idx])
            
            if frame_idx <= phase1_end:
                title = f"Frame {frame_idx}: SHRINKING"
            elif frame_idx <= phase2_end:
                title = f"Frame {frame_idx}: MOVING BEHIND"
            elif frame_idx <= phase3_end:
                title = f"Frame {frame_idx}: STABLE BEHIND"
            else:
                title = f"Frame {frame_idx}: FADING OUT"
            
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
            axes[idx].axis('off')
    
    # Hide last unused axis
    axes[-1].axis('off')
    
    plt.suptitle("START Animation V2: Perfect Centering + Letter Fade", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    keyframes_path = os.path.join(backend_dir, "start_animation_v2_keyframes.png")
    plt.savefig(keyframes_path, dpi=150, bbox_inches='tight')
    print(f"Key frames saved: {keyframes_path}")
    
    # Create GIF
    gif_path = os.path.join(backend_dir, "start_animation_v2.gif")
    print(f"Creating GIF: {gif_path}...")
    
    # Sample frames for GIF
    gif_frames = processed_frames[::3]  # Every 3rd frame
    imageio.mimsave(gif_path, gif_frames, fps=20, loop=0)
    print(f"GIF saved: {gif_path}")
    
    return processed_frames


if __name__ == "__main__":
    import torch
    
    frames = create_start_animation_v2()
    print(f"\n✓ Animation V2 complete! Created {len(frames)} frames")
    print("\nImprovements:")
    print("  ✓ Accurate per-frame masking (no pixel bleeding)")
    print("  ✓ Text stays perfectly centered while scaling")
    print("  ✓ Letter-by-letter fade out effect")
    print("\nOutputs:")
    print("  - start_animation_v2.mp4 (full video)")
    print("  - start_animation_v2.gif (preview)")
    print("  - start_animation_v2_keyframes.png (key frames)")