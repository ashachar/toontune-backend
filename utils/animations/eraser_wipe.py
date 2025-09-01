#!/usr/bin/env python3
"""
Unified eraser wipe animation with multiple modes
Consolidates all eraser animations into a single configurable function
"""

import subprocess
import os
import math


def create_eraser_wipe(character_video: str, original_video: str, 
                       eraser_image: str, output_video: str,
                       wipe_start: float = 0, wipe_duration: float = 0.6,
                       mode: str = "true_erase", erase_radius: int = 120, 
                       sample_points: int = 25):
    """
    Create eraser wipe animation with configurable modes.
    
    Args:
        character_video: Video with character overlay to be erased
        original_video: Original video to reveal underneath
        eraser_image: PNG image of eraser
        output_video: Output video path
        wipe_start: When to start the wipe animation
        wipe_duration: Duration of the wipe animation
        mode: Type of erase effect:
            - "true_erase": Permanent erase where touched (default)
            - "fade": Simple time-based fade
            - "dissolve": Gradual dissolve effect
        erase_radius: Radius of the erase effect (for true_erase mode)
        sample_points: Number of sample points for smoother trail (for true_erase mode)
    """
    
    print(f"Creating {mode} eraser wipe...")
    
    # Get video properties
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        character_video
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if not result.stdout.strip():
        print(f"Error: Could not read video properties from {character_video}")
        return False
    parts = result.stdout.strip().split('x')
    width = int(parts[0])
    height = int(parts[1])
    
    # Eraser path parameters - adjusted for character position
    center_x = width // 2 - 20
    center_y = height // 2 + 10
    radius_x = 200
    radius_y = 150
    
    # Motion and positioning parameters
    scale_factor = 0.70
    pivot_x_ratio = 0.1953
    pivot_y_ratio = 0.62
    tip_ratio = 0.12
    top_margin = 20
    amplitude = radius_y * 0.8
    bottom_safety = 2
    contact_x_ratio = 0.08
    
    # Calculate offsets
    scaled_width = int(768 * scale_factor)
    scaled_height = int(1344 * scale_factor)
    contact_dx = scaled_width * (contact_x_ratio - pivot_x_ratio)
    
    # Build filter based on mode
    if mode == "true_erase":
        # Create accumulating erase mask
        num_samples = sample_points
        geq_parts = []
        
        for i in range(num_samples):
            progress = i / (num_samples - 1) if num_samples > 1 else 0
            t = wipe_start + progress * wipe_duration
            angle = 2 * math.pi * progress
            
            # Calculate eraser tip position
            x_contact = int(center_x + radius_x * math.cos(angle) + contact_dx)
            sin_val = math.sin(angle)
            y_top_raw = center_y + amplitude * sin_val - scaled_height * pivot_y_ratio + \
                       (top_margin - (center_y - amplitude) + scaled_height * (pivot_y_ratio - tip_ratio))
            y_top = max(y_top_raw, height - scaled_height + bottom_safety)
            y_tip = int(y_top + scaled_height * tip_ratio)
            
            # Create erase condition
            radius_sq = erase_radius * erase_radius
            condition = f"(lte((X-{x_contact})*(X-{x_contact})+(Y-{y_tip})*(Y-{y_tip}),{radius_sq})*gte(T,{t:.3f}))"
            geq_parts.append(condition)
        
        # Combine conditions with nested max
        if len(geq_parts) == 1:
            combined = geq_parts[0]
        else:
            combined = geq_parts[0]
            for part in geq_parts[1:]:
                combined = f"max({combined},{part})"
        
        # Inverted mask: white where character remains, black where erased
        geq_expr = f"lum='255*(1-{combined})'"
        
        print(f"  Mode: True erase with {erase_radius}px radius, {num_samples} samples")
        
        # Build filter for true erase
        filter_complex = (
            "[0:v]format=rgba[char];"
            "[1:v]format=rgba[orig];"
            f"[2:v]format=rgba,scale=iw*{scale_factor}:ih*{scale_factor}[eraser];"
            f"[char]format=gray,geq={geq_expr}[erasemask];"
            f"[erasemask]boxblur={min(3, 1 + num_samples//10)}:{min(2, 1 + num_samples//20)}[erasemask_smooth];"
            "[orig][char][erasemask_smooth]maskedmerge[reveal];"
        )
        
    elif mode == "fade":
        # Simple time-based fade
        print(f"  Mode: Simple fade over {wipe_duration}s")
        
        filter_complex = (
            "[0:v]format=rgba[char];"
            "[1:v]format=rgba[orig];"
            f"[2:v]format=rgba,scale=iw*{scale_factor}:ih*{scale_factor}[eraser];"
            f"[orig][char]blend=all_expr='A*min(1,max(0,(T-{wipe_start})/{wipe_duration}))+B*(1-min(1,max(0,(T-{wipe_start})/{wipe_duration})))'[reveal];"
        )
        
    elif mode == "dissolve":
        # Cross-dissolve effect
        print(f"  Mode: Cross-dissolve over {wipe_duration}s")
        
        filter_complex = (
            "[0:v]format=rgba[char];"
            "[1:v]format=rgba[orig];"
            f"[2:v]format=rgba,scale=iw*{scale_factor}:ih*{scale_factor}[eraser];"
            f"[char][orig]blend=all_expr='A*(1-min(1,max(0,(T-{wipe_start})/{wipe_duration})))+B*min(1,max(0,(T-{wipe_start})/{wipe_duration}))':"
            f"enable='between(t,{wipe_start},{wipe_start + wipe_duration})'[reveal];"
        )
    else:
        print(f"Error: Unknown mode '{mode}'")
        return False
    
    # Add eraser overlay (common for all modes)
    filter_complex += (
        f"[reveal][eraser]overlay="
        f"x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-overlay_w*{pivot_x_ratio}':"
        f"y='max({center_y}+{amplitude}*sin(2*PI*(t-{wipe_start})/{wipe_duration})-overlay_h*{pivot_y_ratio}+"
        f"({top_margin} - ({center_y} - {amplitude}) + overlay_h*({pivot_y_ratio} - {tip_ratio})),"
        f"main_h - overlay_h + {bottom_safety})':"
        f"shortest=0:eof_action=pass:"
        f"enable='between(t,{wipe_start},{wipe_start + wipe_duration + 0.02})'[outv]"
    )
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-i', character_video,
        '-i', original_video,
        '-stream_loop', '-1',  # Loop eraser to prevent freeze
        '-i', eraser_image,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '1:a?',  # Map audio from original if exists
        '-c:a', 'copy',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_video
    ]
    
    print(f"Applying eraser wipe...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr[:500]}")
        return False
    
    print(f"Eraser wipe created successfully: {output_video}")
    return True


if __name__ == "__main__":
    # Test with default files
    import sys
    
    if len(sys.argv) > 1:
        character_video = sys.argv[1]
        original_video = sys.argv[2] if len(sys.argv) > 2 else "uploads/assets/runway_experiment/runway_demo_input.mp4"
        eraser_image = sys.argv[3] if len(sys.argv) > 3 else "uploads/assets/images/eraser.png"
        output_video = sys.argv[4] if len(sys.argv) > 4 else "outputs/test_unified_eraser.mp4"
        mode = sys.argv[5] if len(sys.argv) > 5 else "true_erase"
    else:
        # Default test files
        character_video = "outputs/runway_scaled_cropped.mp4"
        original_video = "uploads/assets/runway_experiment/runway_demo_input.mp4"
        eraser_image = "uploads/assets/images/eraser.png"
        output_video = "outputs/test_unified_eraser.mp4"
        mode = "true_erase"
    
    success = create_eraser_wipe(
        character_video, original_video, eraser_image, output_video,
        mode=mode, erase_radius=120, sample_points=25
    )
    
    if success:
        print(f"\n✅ Success! Video saved to: {output_video}")
        print(f"   Mode: {mode}")
    else:
        print("\n❌ Failed to create eraser wipe")