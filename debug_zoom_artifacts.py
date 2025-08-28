"""
Debug white pixel artifacts in Zoom3D animation
"""

import cv2
import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/scale_3d')

from base_3d_text_animation import Animation3DConfig
from scale_3d import Zoom3D


def debug_scaling_artifacts():
    """Debug the scaling artifacts issue"""
    
    print("Debugging Zoom3D scaling artifacts...")
    print("=" * 60)
    
    # Create a simple test letter sprite
    img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
    except:
        font = ImageFont.load_default()
    
    # Draw letter with shadow
    draw.text((110, 102), 'A', fill=(60, 60, 60, 255), font=font, anchor='mm')
    draw.text((100, 100), 'A', fill=(255, 255, 255, 255), font=font, anchor='mm', 
              stroke_width=2, stroke_fill=(255, 255, 255, 255))
    
    sprite = np.array(img)
    
    print("\nOriginal sprite:")
    print(f"  Shape: {sprite.shape}")
    print(f"  Non-zero alpha pixels: {np.sum(sprite[:,:,3] > 0)}")
    print(f"  Max alpha: {np.max(sprite[:,:,3])}")
    
    # Test different scale factors
    scale_factors = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    for scale in scale_factors:
        h, w = sprite.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        # Test different interpolation methods
        methods = [
            ('INTER_LINEAR', cv2.INTER_LINEAR),
            ('INTER_AREA', cv2.INTER_AREA),
            ('INTER_CUBIC', cv2.INTER_CUBIC),
            ('INTER_NEAREST', cv2.INTER_NEAREST)
        ]
        
        print(f"\nScale: {scale} (size: {new_w}x{new_h})")
        
        for method_name, method in methods:
            scaled = cv2.resize(sprite, (new_w, new_h), interpolation=method)
            
            # Check for artifacts
            if scaled.shape[2] == 4:
                # Find pixels that have color but low/no alpha (artifacts)
                rgb_sum = np.sum(scaled[:,:,:3], axis=2)
                alpha = scaled[:,:,3]
                
                # Artifacts are pixels with high RGB but low alpha
                artifact_mask = (rgb_sum > 300) & (alpha < 100)
                artifact_count = np.sum(artifact_mask)
                
                # Also check for edge artifacts
                if artifact_count > 0 or method == cv2.INTER_LINEAR:
                    # Save debug image
                    debug_img = scaled.copy()
                    if artifact_count > 0:
                        debug_img[artifact_mask] = [255, 0, 0, 255]  # Mark artifacts in red
                    
                    filename = f"outputs/scale_debug_{scale}_{method_name}.png"
                    cv2.imwrite(filename, debug_img)
                
                print(f"  {method_name:15s}: artifacts={artifact_count}")


def test_zoom_with_fix():
    """Test Zoom3D with potential fix"""
    
    print("\n" + "=" * 60)
    print("Testing Zoom3D with different interpolation...")
    
    width, height = 1280, 720
    fps = 30
    duration_seconds = 3
    total_frames = fps * duration_seconds
    
    # Create output video
    output_path = "outputs/zoom_debug.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create Zoom animation
    config = Animation3DConfig(
        text="ZOOM TEST",
        duration_ms=2500,
        position=(640, 360, 0),
        font_size=100,
        font_color=(100, 255, 100),
        stagger_ms=20
    )
    
    zoom_anim = Zoom3D(config, start_scale=0.1, end_scale=1.0)
    
    # Render frames
    for frame_num in range(total_frames):
        # Black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply animation
        frame = zoom_anim.apply_frame(frame, frame_num, fps)
        
        # Add info
        progress = frame_num / (fps * 2.5)
        scale = 0.1 + (1.0 - 0.1) * progress
        cv2.putText(frame, f"Frame: {frame_num} Scale: {scale:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        out.write(frame)
        
        # Save specific frames for analysis
        if frame_num in [15, 30, 45, 60]:  # Various scale points
            cv2.imwrite(f"outputs/zoom_frame_{frame_num}.jpg", frame)
    
    out.release()
    
    # Convert to H.264
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"Debug video created: {h264_output}")
    

if __name__ == "__main__":
    print("ZOOM3D ARTIFACT ANALYSIS")
    print("=" * 60)
    
    debug_scaling_artifacts()
    test_zoom_with_fix()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("Check outputs/scale_debug_*.png for artifact visualization")
    print("=" * 60)