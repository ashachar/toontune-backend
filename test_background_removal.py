#!/usr/bin/env python3
"""
Test background removal on ai_math1 video.
Shows the actual background being removed and replaced.
"""

import cv2
import numpy as np
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import subprocess

def extract_frame(video_path, frame_number=50):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    return None

def remove_background_frame(frame):
    """Remove background from a single frame using rembg."""
    # Convert to PIL Image
    pil_image = Image.fromarray(frame)
    
    # Remove background
    session = new_session('u2net')
    output = remove(pil_image, session=session)
    
    # Convert back to numpy array (RGBA)
    return np.array(output)

def composite_on_background(foreground_rgba, background_rgb):
    """Composite foreground with alpha onto background."""
    # Ensure background is same size
    h, w = foreground_rgba.shape[:2]
    background_resized = cv2.resize(background_rgb, (w, h))
    
    # Extract alpha channel
    alpha = foreground_rgba[:, :, 3] / 255.0
    alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
    
    # Extract RGB from foreground
    foreground_rgb = foreground_rgba[:, :, :3]
    
    # Composite
    composite = (alpha_3d * foreground_rgb + (1 - alpha_3d) * background_resized).astype(np.uint8)
    
    return composite

def main():
    # Paths
    video_path = Path("uploads/assets/videos/ai_math1_test_5sec.mp4")
    background_path = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    
    print("Testing background removal...")
    
    # Extract test frames
    print("1. Extracting frame from original video...")
    original_frame = extract_frame(video_path, frame_number=50)
    
    print("2. Extracting frame from background video...")
    background_frame = extract_frame(background_path, frame_number=50)
    
    if original_frame is None or background_frame is None:
        print("Error: Could not extract frames")
        return
    
    print("3. Removing background from original frame...")
    foreground_rgba = remove_background_frame(original_frame)
    
    print("4. Compositing onto new background...")
    composite = composite_on_background(foreground_rgba, background_frame)
    
    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save original
    cv2.imwrite(str(output_dir / "test_bg_1_original.png"), 
                cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))
    
    # Save foreground with alpha (as PNG)
    cv2.imwrite(str(output_dir / "test_bg_2_foreground.png"),
                cv2.cvtColor(foreground_rgba, cv2.COLOR_RGBA2BGRA))
    
    # Save new background
    cv2.imwrite(str(output_dir / "test_bg_3_background.png"),
                cv2.cvtColor(background_frame, cv2.COLOR_RGB2BGR))
    
    # Save composite
    cv2.imwrite(str(output_dir / "test_bg_4_composite.png"),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    
    print("\nResults saved to outputs/:")
    print("  - test_bg_1_original.png: Original frame")
    print("  - test_bg_2_foreground.png: Foreground with transparency")
    print("  - test_bg_3_background.png: New background frame")
    print("  - test_bg_4_composite.png: Final composite")
    
    # Open the composite
    subprocess.run(["open", str(output_dir / "test_bg_4_composite.png")])

if __name__ == "__main__":
    main()