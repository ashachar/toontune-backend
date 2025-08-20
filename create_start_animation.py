#!/usr/bin/env python3
"""
Create animated "START" text that moves from foreground to background
Famous dolly zoom / parallax effect
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from rembg import remove, new_session
import imageio

# Add SAM2 to path if needed
sys.path.insert(0, os.path.expanduser("~/sam2"))


def create_start_animation():
    """Create multi-frame animation of START text moving to background"""
    
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    
    # Load video and extract frames
    video_path = os.path.join(backend_dir, "uploads/assets/videos/do_re_mi.mov")
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {fps} fps, {total_frames} total frames")
    
    # Parameters for animation
    start_frame = 890  # Start a bit before frame 900
    num_frames = 60    # 2 seconds at 30fps
    
    # Animation parameters
    # Text starts large and in foreground, ends smaller and in background
    start_scale = 1.8   # 180% size at start
    end_scale = 1.0     # 100% size at end
    
    # Position animation - text moves toward center
    start_x_offset = -100  # Start more to the left
    end_x_offset = 0       # End at intended position
    
    start_y_offset = -50   # Start higher
    end_y_offset = 0       # End at intended position
    
    # Opacity for depth effect
    start_opacity = 1.0    # Fully opaque at start
    mid_opacity = 0.9      # Slightly transparent in middle
    
    frames = []
    foreground_masks = []
    
    print(f"Extracting {num_frames} frames starting from frame {start_frame}...")
    
    # Extract frames and create foreground masks
    session = new_session('u2net')
    
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
        
        # Get foreground mask using rembg
        if i % 10 == 0:  # Process every 10th frame for speed
            print(f"Processing frame {i}/{num_frames}...")
            img_pil = Image.fromarray(frame_rgb)
            img_no_bg = remove(img_pil, session=session)
            img_no_bg_np = np.array(img_no_bg)
            
            if img_no_bg_np.shape[2] == 4:
                foreground_mask = img_no_bg_np[:, :, 3] > 128
            else:
                foreground_mask = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=bool)
            
            # Store mask and reuse for nearby frames
            for j in range(10):
                if i + j < num_frames:
                    foreground_masks.append(foreground_mask)
    
    cap.release()
    
    # Ensure we have masks for all frames
    while len(foreground_masks) < len(frames):
        foreground_masks.append(foreground_masks[-1])
    
    print(f"Extracted {len(frames)} frames with foreground masks")
    
    # Create font
    h, w = frames[0].shape[:2]
    font_size_base = min(180, int(h * 0.35))
    
    try:
        font_path = "/System/Library/Fonts/Helvetica.ttc"
        if not os.path.exists(font_path):
            font_path = None
    except:
        font_path = None
    
    # Process each frame
    processed_frames = []
    
    print("Creating animation frames...")
    
    for i, (frame, fg_mask) in enumerate(zip(frames, foreground_masks)):
        # Calculate animation progress (0 to 1)
        progress = i / (num_frames - 1)
        
        # Determine animation phase
        if progress < 0.3:
            # Phase 1: Text in foreground, shrinking
            phase = "foreground"
            phase_progress = progress / 0.3
        elif progress < 0.6:
            # Phase 2: Text moving behind person
            phase = "transition"
            phase_progress = (progress - 0.3) / 0.3
        else:
            # Phase 3: Text in background
            phase = "background"
            phase_progress = (progress - 0.6) / 0.4
        
        # Calculate current scale and position
        if phase == "foreground":
            # Text is fully in front
            current_scale = start_scale - (start_scale - 1.3) * phase_progress
            x_offset = start_x_offset * (1 - phase_progress)
            y_offset = start_y_offset * (1 - phase_progress)
            occlusion = False
            opacity = start_opacity
        elif phase == "transition":
            # Text moving behind person
            current_scale = 1.3 - (1.3 - end_scale) * phase_progress
            x_offset = 0
            y_offset = 0
            occlusion = phase_progress > 0.5  # Start occlusion halfway through transition
            opacity = mid_opacity
        else:
            # Text fully behind person
            current_scale = end_scale
            x_offset = end_x_offset
            y_offset = end_y_offset
            occlusion = True
            opacity = mid_opacity + (1.0 - mid_opacity) * phase_progress
        
        # Create text layer for this frame
        text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # Scale font size
        current_font_size = int(font_size_base * current_scale)
        
        if font_path:
            font = ImageFont.truetype(font_path, current_font_size)
        else:
            font = ImageFont.load_default()
        
        # Calculate text position
        text = "START"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center position with offsets
        base_x = (w - text_width) // 2 + int(x_offset)
        base_y = (h - text_height) // 2 + int(y_offset)
        
        # Draw text with effects
        # Shadow (scales with text)
        shadow_offset = max(2, int(5 * current_scale))
        shadow_alpha = int(100 * opacity)
        draw.text((base_x + shadow_offset, base_y + shadow_offset), text, 
                 font=font, fill=(0, 0, 0, shadow_alpha))
        
        # Outline
        outline_width = max(1, int(3 * current_scale))
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if abs(dx) == outline_width or abs(dy) == outline_width:
                    draw.text((base_x + dx, base_y + dy), text, 
                             font=font, fill=(255, 255, 255, int(150 * opacity)))
        
        # Main text - color changes slightly through animation
        red = 255
        green = int(220 + 35 * progress)  # Gets more yellow
        blue = int(100 * (1 - progress))   # Less blue over time
        alpha = int(255 * opacity)
        draw.text((base_x, base_y), text, font=font, fill=(red, green, blue, alpha))
        
        # Convert to numpy
        text_layer_np = np.array(text_layer)
        
        # Composite with frame
        result = frame.copy()
        
        if occlusion:
            # Apply text only in background
            bg_mask = ~fg_mask
            for c in range(3):
                text_visible = (text_layer_np[:, :, 3] > 0) & bg_mask
                # Blend text with background
                alpha_blend = text_layer_np[text_visible, 3] / 255.0
                result[text_visible, c] = (
                    result[text_visible, c] * (1 - alpha_blend) + 
                    text_layer_np[text_visible, c] * alpha_blend
                ).astype(np.uint8)
        else:
            # Text in foreground - normal composite
            for c in range(3):
                text_visible = text_layer_np[:, :, 3] > 0
                alpha_blend = text_layer_np[text_visible, 3] / 255.0
                result[text_visible, c] = (
                    result[text_visible, c] * (1 - alpha_blend) + 
                    text_layer_np[text_visible, c] * alpha_blend
                ).astype(np.uint8)
        
        processed_frames.append(result)
        
        # Progress indicator
        if i % 10 == 0:
            print(f"  Frame {i}/{num_frames}: {phase} (scale={current_scale:.2f})")
    
    print(f"Processed {len(processed_frames)} frames")
    
    # Save as video
    output_video = os.path.join(backend_dir, "start_animation.mp4")
    print(f"Saving video to {output_video}...")
    
    # Use imageio to save video
    writer = imageio.get_writer(output_video, fps=fps)
    for frame in processed_frames:
        writer.append_data(frame)
    writer.close()
    
    print(f"Video saved: {output_video}")
    
    # Also save key frames as images
    key_frames = [0, 15, 30, 45, 59]  # Beginning, quarter, middle, three-quarter, end
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for idx, frame_idx in enumerate(key_frames):
        if frame_idx < len(processed_frames):
            axes[idx].imshow(processed_frames[frame_idx])
            
            progress = frame_idx / (num_frames - 1)
            if progress < 0.3:
                title = f"Frame {frame_idx}: FOREGROUND"
            elif progress < 0.6:
                title = f"Frame {frame_idx}: TRANSITION"
            else:
                title = f"Frame {frame_idx}: BACKGROUND"
            
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
            axes[idx].axis('off')
    
    plt.suptitle("START Animation: Foreground → Background (Dolly Zoom Effect)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    keyframes_path = os.path.join(backend_dir, "start_animation_keyframes.png")
    plt.savefig(keyframes_path, dpi=150, bbox_inches='tight')
    print(f"Key frames saved: {keyframes_path}")
    
    # Create GIF for easy viewing
    gif_path = os.path.join(backend_dir, "start_animation.gif")
    print(f"Creating GIF: {gif_path}...")
    
    # Sample every 2nd frame for smaller GIF
    gif_frames = processed_frames[::2]
    imageio.mimsave(gif_path, gif_frames, fps=fps//2, loop=0)
    print(f"GIF saved: {gif_path}")
    
    return processed_frames


if __name__ == "__main__":
    import torch
    
    # Check dependencies
    try:
        import rembg
        import imageio
    except ImportError:
        print("Installing required packages...")
        os.system("pip install rembg imageio imageio-ffmpeg")
    
    frames = create_start_animation()
    print(f"\n✓ Animation complete! Created {len(frames)} frames")
    print("\nOutputs:")
    print("  - start_animation.mp4 (full video)")
    print("  - start_animation.gif (preview)")
    print("  - start_animation_keyframes.png (key frames)")
    print("\nWe're cooking! 🔥")