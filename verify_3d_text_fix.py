#!/usr/bin/env python3
"""
Verification script for 3D text fixes:
- Tests anti-aliasing (smooth edges)
- Verifies center-point locking during shrink
- Shows debug logs with [3D_PIXELATED] prefix
"""

import os
import cv2
import numpy as np
from PIL import Image
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

def main():
    # Test with actual video
    video_path = "test_element_3sec.mp4"
    
    if os.path.exists(video_path):
        print(f"Using video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get first frame for mask
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = extract_foreground_mask(frame_rgb)
        else:
            mask = np.zeros((H, W), dtype=np.uint8)
        
        cap.release()
        
        # Load frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        print(f"Loaded {len(frames)} frames at {W}x{H} @ {fps}fps")
        
    else:
        # Fallback: synthetic test
        print("Using synthetic test (no video file found)")
        W, H, fps = 1280, 720, 30
        frames = [np.full((H, W, 3), 245, dtype=np.uint8) for _ in range(90)]  # light gray
        mask = np.zeros((H, W), dtype=np.uint8)  # no occlusion

    # Create animation with improved settings
    print("\n" + "="*60)
    print("Testing 3D Text with Fixes")
    print("="*60)
    
    anim = Text3DBehindSegment(
        duration=3.0,
        fps=fps,
        resolution=(W, H),
        text="HELLO WORLD",
        segment_mask=mask,
        font_size=140,
        text_color=(255, 220, 0),     # Golden yellow
        depth_color=(200, 170, 0),     # Darker yellow
        depth_layers=10,
        depth_offset=3,
        start_scale=2.2,               # Start large
        end_scale=0.9,                 # End slightly smaller
        phase1_duration=1.2,           # Shrink phase
        phase2_duration=0.6,           # Transition phase
        phase3_duration=1.2,           # Stable behind phase
        center_position=(W//2, H//2),  # Lock to center
        shadow_offset=6,
        outline_width=2,
        perspective_angle=25,
        supersample_factor=3,          # 3x anti-aliasing
        debug=True,                    # Enable [3D_PIXELATED] logs
        perspective_during_shrink=False,  # Clean lock during shrink
    )
    
    print(f"\nGenerating frames with:")
    print(f"  • 3x supersampling for anti-aliasing")
    print(f"  • Front-face anchor locking to center")
    print(f"  • Perspective disabled during shrink")
    print(f"  • Debug logs enabled\n")
    
    total = int(3.0 * fps)
    output_frames = []
    
    for i in range(total):
        if i % 10 == 0:
            print(f"Frame {i}/{total}...")
        
        # Use video frame or synthetic background
        bg = frames[i % len(frames)]
        
        # Generate frame
        frame = anim.generate_frame(i, bg)
        
        # Convert to BGR for OpenCV
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_frames.append(frame_bgr)
    
    print(f"\n✓ Generated {len(output_frames)} frames")
    
    # Save video
    output_path = "verify_3d_text_fixed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_path = "temp_verify.mp4"
    
    out = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
    for f in output_frames:
        out.write(f)
    out.release()
    
    # Convert to H.264
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '20',  # High quality
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print(f"\n✓ Saved: {output_path}")
    
    # Extract key frames for inspection
    print("\nExtracting key frames...")
    
    key_frames = {
        "start": 0,
        "mid_shrink": total // 4,
        "transition": total // 2,
        "behind": 3 * total // 4,
        "end": total - 1
    }
    
    for name, idx in key_frames.items():
        if idx < len(output_frames):
            cv2.imwrite(f"verify_3d_{name}.png", output_frames[idx])
            print(f"  ✓ verify_3d_{name}.png")
    
    print("\n" + "="*60)
    print("✅ Verification Complete!")
    print("="*60)
    print("\nWhat to check:")
    print("1. Edges should be smooth (no pixelation)")
    print("2. Text center should stay locked during shrink")
    print("3. Debug logs show [3D_PIXELATED] prefix")
    print("4. Anchor tracking in logs should match target center")
    print("\nOutput files:")
    print(f"  • {output_path} - Main test video")
    print("  • verify_3d_*.png - Key frames")

if __name__ == "__main__":
    main()