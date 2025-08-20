#!/usr/bin/env python3
"""
Create animated "START" text - Version 3 with fixes:
1. Fix position jump issue
2. Random letter dissolve order
3. Letters float upward while fading
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from rembg import remove, new_session
import imageio
import random

sys.path.insert(0, os.path.expanduser("~/sam2"))


def create_start_animation_v3():
    """Create animation with floating dissolve effect"""
    
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    
    # Load video
    video_path = os.path.join(backend_dir, "uploads/assets/videos/do_re_mi.mov")
    cap = cv2.VideoCapture(video_path)
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video: {fps} fps")
    
    # Animation parameters
    start_frame = 890
    num_frames = 150  # 5 seconds total
    
    # Animation phases (in frames)
    phase1_end = 30    # Frames 0-30: Text shrinking in foreground
    phase2_end = 50    # Frames 31-50: Text moving behind
    phase3_end = 90    # Frames 51-90: Text stable behind (1.3 seconds)
    phase4_end = 150   # Frames 91-150: Dissolve effect (2 seconds)
    
    # Scale animation
    start_scale = 2.0
    end_scale = 1.0
    
    # Random dissolve order for letters (S-T-A-R-T)
    letter_indices = [0, 1, 2, 3, 4]
    random.shuffle(letter_indices)
    print(f"Letter dissolve order: {['START'[i] for i in letter_indices]}")
    
    # Dissolve timing - each letter takes 20 frames to dissolve
    dissolve_duration = 20
    dissolve_stagger = 10  # Frames between each letter starting
    
    frames = []
    
    print(f"Extracting {num_frames} frames...")
    
    # Extract all frames
    for i in range(num_frames):
        frame_idx = start_frame + i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    # Initialize rembg
    print("Initializing background removal...")
    session = new_session('u2net')
    
    # Get dimensions
    h, w = frames[0].shape[:2]
    
    # Create font
    font_size_base = min(150, int(h * 0.28))
    
    try:
        font_path = "/System/Library/Fonts/Helvetica.ttc"
        if not os.path.exists(font_path):
            font_path = None
    except:
        font_path = None
    
    # FIXED center position - no vertical changes until dissolve
    CENTER_X = w // 2
    CENTER_Y = int(h * 0.45)
    
    # Process each frame
    processed_frames = []
    
    print("Creating animation frames...")
    
    for i, frame in enumerate(frames):
        # Calculate animation phase
        if i <= phase1_end:
            phase = "shrinking"
            phase_progress = i / phase1_end
            current_scale = start_scale - (start_scale - 1.3) * phase_progress
            occlusion = False
            
        elif i <= phase2_end:
            phase = "moving_behind"
            phase_progress = (i - phase1_end) / (phase2_end - phase1_end)
            current_scale = 1.3 - (1.3 - end_scale) * phase_progress
            occlusion = phase_progress > 0.3
            
        elif i <= phase3_end:
            phase = "stable_behind"
            current_scale = end_scale
            occlusion = True
            
        else:
            phase = "dissolving"
            phase_progress = (i - phase3_end) / (phase4_end - phase3_end)
            current_scale = end_scale
            occlusion = True
        
        # Get accurate foreground mask for THIS frame
        if occlusion:
            img_pil = Image.fromarray(frame)
            img_no_bg = remove(img_pil, session=session)
            img_no_bg_np = np.array(img_no_bg)
            
            if img_no_bg_np.shape[2] == 4:
                foreground_mask = img_no_bg_np[:, :, 3] > 128
            else:
                foreground_mask = np.zeros((h, w), dtype=bool)
        else:
            foreground_mask = np.zeros((h, w), dtype=bool)
        
        # Create text layer
        text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # Scale font size
        current_font_size = int(font_size_base * current_scale)
        
        if font_path:
            font = ImageFont.truetype(font_path, current_font_size)
        else:
            font = ImageFont.load_default()
        
        # Draw text
        if phase != "dissolving":
            # Normal text rendering
            text = "START"
            
            # Get text dimensions
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position - ALWAYS CENTERED
            text_x = CENTER_X - text_width // 2
            text_y = CENTER_Y - text_height // 2
            
            # Draw effects
            shadow_offset = max(2, int(4 * current_scale))
            draw.text((text_x + shadow_offset, text_y + shadow_offset), text, 
                     font=font, fill=(0, 0, 0, 100))
            
            outline_width = 2
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if abs(dx) == outline_width or abs(dy) == outline_width:
                        draw.text((text_x + dx, text_y + dy), text, 
                                 font=font, fill=(255, 255, 255, 150))
            
            draw.text((text_x, text_y), text, font=font, fill=(255, 220, 0, 255))
            
        else:
            # DISSOLVE EFFECT - letters float up and fade
            letters = list("START")
            full_word_text = "START"
            
            # FIX: Calculate baseline Y using full word height (same as stable phase)
            full_word_bbox = draw.textbbox((0, 0), full_word_text, font=font)
            full_word_height = full_word_bbox[3] - full_word_bbox[1]
            baseline_y = CENTER_Y - full_word_height // 2  # This matches the stable phase exactly
            
            # FIX FOR SPREADING: Calculate starting X using full word width (with kerning)
            full_word_width = full_word_bbox[2] - full_word_bbox[0]
            base_start_x = CENTER_X - full_word_width // 2
            
            # Draw each letter with dissolve effect
            for idx, letter in enumerate(letters):
                # FIX FOR SPREADING: Calculate letter X position using prefix width (maintains kerning)
                prefix = full_word_text[:idx]
                if prefix:
                    # Get the width of all letters before this one
                    prefix_bbox = draw.textbbox((0, 0), prefix, font=font)
                    prefix_width = prefix_bbox[2] - prefix_bbox[0]
                else:
                    prefix_width = 0
                
                # This is the correct, kerned base position for the current letter
                letter_base_x = base_start_x + prefix_width
                
                # Find this letter's position in dissolve order
                dissolve_idx = letter_indices.index(idx)
                
                # Calculate when this letter starts dissolving
                letter_start_frame = phase3_end + (dissolve_idx * dissolve_stagger)
                
                if i >= letter_start_frame:
                    # This letter is dissolving
                    letter_progress = min(1.0, (i - letter_start_frame) / dissolve_duration)
                    
                    # Calculate upward float (moves up as it fades)
                    float_distance = letter_progress * 30  # Float up 30 pixels
                    
                    # Calculate opacity (fade out)
                    opacity = int(255 * (1 - letter_progress))
                    
                    # Calculate slight expansion (letters grow slightly as they dissolve)
                    dissolve_scale = 1.0 + (letter_progress * 0.2)  # Up to 20% larger
                    
                    # Create slightly larger font for this letter
                    dissolve_font_size = int(current_font_size * dissolve_scale)
                    if font_path:
                        dissolve_font = ImageFont.truetype(font_path, dissolve_font_size)
                    else:
                        dissolve_font = font
                    
                    # Get dimensions for scaling adjustments
                    orig_bbox = draw.textbbox((0, 0), letter, font=font)
                    orig_letter_width = orig_bbox[2] - orig_bbox[0]
                    orig_letter_height = orig_bbox[3] - orig_bbox[1]
                    
                    scaled_bbox = draw.textbbox((0, 0), letter, font=dissolve_font)
                    scaled_letter_width = scaled_bbox[2] - scaled_bbox[0]
                    scaled_letter_height = scaled_bbox[3] - scaled_bbox[1]
                    
                    # FIX: Adjust X position to center the growing letter
                    width_increase = scaled_letter_width - orig_letter_width
                    letter_x = letter_base_x - (width_increase // 2)
                    
                    # FIX: Start from baseline and adjust for scaling to keep centered
                    height_increase = scaled_letter_height - orig_letter_height
                    letter_y = baseline_y - int(float_distance) - (height_increase // 2)
                    
                    if opacity > 0:
                        # Shadow (fades with letter)
                        shadow_alpha = min(50, opacity // 2)
                        draw.text((letter_x + 2, letter_y + 2), letter,
                                 font=dissolve_font, fill=(0, 0, 0, shadow_alpha))
                        
                        # Glow effect (gets stronger as letter dissolves)
                        glow_alpha = int(opacity * 0.3)
                        for radius in [6, 4, 2]:
                            for angle in range(0, 360, 45):
                                gx = int(radius * np.cos(np.radians(angle)))
                                gy = int(radius * np.sin(np.radians(angle)))
                                draw.text((letter_x + gx, letter_y + gy), letter,
                                         font=dissolve_font, fill=(255, 255, 200, glow_alpha))
                        
                        # Main letter
                        draw.text((letter_x, letter_y), letter,
                                 font=dissolve_font, fill=(255, 220, 0, opacity))
                    
                else:
                    # Letter not yet dissolving - draw normally
                    # FIX: Use the baseline_y for perfect continuity
                    letter_y = baseline_y  # Use exact same Y as stable phase
                    # FIX: Use kerning-aware position
                    letter_x = letter_base_x
                    
                    # Shadow
                    draw.text((letter_x + 3, letter_y + 3), letter,
                             font=font, fill=(0, 0, 0, 100))
                    
                    # Outline
                    for dx in [-2, 2]:
                        for dy in [-2, 2]:
                            draw.text((letter_x + dx, letter_y + dy), letter,
                                     font=font, fill=(255, 255, 255, 150))
                    
                    # Main letter
                    draw.text((letter_x, letter_y), letter,
                             font=font, fill=(255, 220, 0, 255))
        
        # Convert to numpy
        text_layer_np = np.array(text_layer)
        
        # Composite with frame
        result = frame.copy()
        
        if occlusion:
            # Apply text ONLY in background
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
        
        # Progress
        if i % 10 == 0:
            print(f"  Frame {i}/{num_frames}: {phase} (scale={current_scale:.2f})")
    
    print(f"Processed {len(processed_frames)} frames")
    
    # Save video
    output_video = os.path.join(backend_dir, "start_animation_v3.mp4")
    print(f"Saving video to {output_video}...")
    
    writer = imageio.get_writer(output_video, fps=30)
    for frame in processed_frames[::2]:  # 30fps output
        writer.append_data(frame)
    writer.close()
    
    # Save key frames
    key_frames = [0, 25, 40, 60, 90, 100, 110, 120, 130, 140]
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(key_frames):
        if frame_idx < len(processed_frames):
            axes[idx].imshow(processed_frames[frame_idx])
            
            if frame_idx <= phase1_end:
                title = f"Frame {frame_idx}: SHRINKING"
            elif frame_idx <= phase2_end:
                title = f"Frame {frame_idx}: BEHIND"
            elif frame_idx <= phase3_end:
                title = f"Frame {frame_idx}: STABLE"
            else:
                title = f"Frame {frame_idx}: DISSOLVING"
            
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
            axes[idx].axis('off')
    
    plt.suptitle("START Animation V3: Random Dissolve with Upward Float", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    keyframes_path = os.path.join(backend_dir, "start_animation_v3_keyframes.png")
    plt.savefig(keyframes_path, dpi=150, bbox_inches='tight')
    
    # Create GIF
    gif_path = os.path.join(backend_dir, "start_animation_v3.gif")
    gif_frames = processed_frames[::3]
    imageio.mimsave(gif_path, gif_frames, fps=20, loop=0)
    
    print(f"\n✓ Animation V3 complete!")
    print(f"  - Fixed position jump issue")
    print(f"  - Letters dissolve in random order: {[letters[i] for i in letter_indices]}")
    print(f"  - Letters float upward while fading")
    
    return processed_frames


if __name__ == "__main__":
    import torch
    
    frames = create_start_animation_v3()
    print("\nOutputs:")
    print("  - start_animation_v3.mp4")
    print("  - start_animation_v3.gif")
    print("  - start_animation_v3_keyframes.png")