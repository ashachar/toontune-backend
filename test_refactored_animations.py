#!/usr/bin/env python3
"""
Test the refactored animation classes with the full START animation.
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import imageio
from rembg import remove, new_session

# Add paths
sys.path.insert(0, os.path.expanduser("~/sam2"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve


def test_full_animation():
    """Test the complete animation using refactored classes."""
    
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    
    # Load video
    video_path = os.path.join(backend_dir, "uploads/assets/videos/do_re_mi.mov")
    cap = cv2.VideoCapture(video_path)
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {fps} fps, {width}x{height}")
    
    # Animation parameters
    start_frame = 890
    total_frames = 150  # 5 seconds at 30fps output
    
    # Phase timing
    phase1_frames = 30   # Text shrinking in foreground
    phase2_frames = 20   # Text moving behind
    phase3_frames = 40   # Text stable behind
    phase4_frames = 60   # Dissolve effect
    
    # Extract frames
    frames = []
    print(f"Extracting {total_frames} frames...")
    
    for i in range(total_frames):
        frame_idx = start_frame + i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    # Initialize background removal for segmentation
    print("Initializing segmentation...")
    session = new_session('u2net')
    
    # Process frames with animations
    processed_frames = []
    
    # CRITICAL: Use consistent center position for both animations
    center_position = (width // 2, int(height * 0.45))
    font_size = int(min(150, height * 0.28))
    
    # Create TextBehindSegment animation for first 3 phases
    text_animator = TextBehindSegment(
        element_path=video_path,  # Not used directly but required
        background_path=video_path,
        position=center_position,
        text="START",
        font_size=font_size,
        text_color=(255, 220, 0),
        start_scale=2.0,
        end_scale=1.0,
        phase1_duration=phase1_frames / 30,  # Convert to seconds
        phase2_duration=phase2_frames / 30,
        phase3_duration=phase3_frames / 30,
        center_position=center_position,  # Explicitly set center
        fps=30
    )
    
    # Placeholder for WordDissolve - will be created with handoff data
    word_dissolver = None
    
    print("Processing frames...")
    
    # Track when we need to create WordDissolve with handoff data
    handoff_frame = phase1_frames + phase2_frames + phase3_frames
    
    for i, frame in enumerate(frames):
        # Get foreground mask for this frame
        if i % 10 == 0 or i >= phase1_frames + phase2_frames // 2:
            # Update mask periodically and more frequently during occlusion
            img_pil = Image.fromarray(frame)
            img_no_bg = remove(img_pil, session=session)
            img_no_bg_np = np.array(img_no_bg)
            
            if img_no_bg_np.shape[2] == 4:
                current_mask = img_no_bg_np[:, :, 3] > 128
            else:
                current_mask = np.zeros((height, width), dtype=bool)
        
        if i < phase1_frames + phase2_frames + phase3_frames:
            # First 3 phases: Use TextBehindSegment
            processed_frame = text_animator.render_text_frame(
                frame,
                i,
                current_mask if i > phase1_frames else None
            )
        else:
            # Phase 4: Dissolve effect - Create WordDissolve with handoff data if needed
            if word_dissolver is None:
                # Get handoff data from TextBehindSegment
                handoff_data = text_animator.get_handoff_data()
                print(f"📋 Handoff data: {len(handoff_data.get('final_letter_positions', []))} letter positions")
                
                # Create WordDissolve with frozen positions
                word_dissolver = WordDissolve(
                    element_path=video_path,
                    background_path=video_path,
                    position=center_position,
                    word="START",
                    font_size=font_size,
                    text_color=(255, 220, 0),
                    stable_duration=0.17,
                    dissolve_duration=0.67,
                    dissolve_stagger=0.33,
                    float_distance=30,
                    randomize_order=True,
                    maintain_kerning=True,
                    center_position=center_position,
                    handoff_data=handoff_data,  # CRITICAL: Pass handoff data
                    fps=30
                )
                print(f"✓ WordDissolve created with frozen letter positions")
            
            # Pass the original clean frame directly to WordDissolve
            dissolve_frame_idx = i - handoff_frame
            processed_frame = word_dissolver.render_word_frame(
                frame,  # Pass original frame
                dissolve_frame_idx,
                current_mask  # Pass the mask for occlusion
            )
        
        processed_frames.append(processed_frame)
        
        # Progress indicator
        if i % 10 == 0:
            if i < phase1_frames:
                phase = "SHRINKING"
            elif i < phase1_frames + phase2_frames:
                phase = "MOVING BEHIND"
            elif i < phase1_frames + phase2_frames + phase3_frames:
                phase = "STABLE"
            else:
                phase = "DISSOLVING"
            print(f"  Frame {i}/{total_frames}: {phase}")
    
    print(f"Processed {len(processed_frames)} frames")
    
    # Save video
    output_video = os.path.join(backend_dir, "start_animation_refactored.mp4")
    print(f"Saving video to {output_video}...")
    
    writer = imageio.get_writer(output_video, fps=30)
    for frame in processed_frames[::2]:  # 30fps output from 60fps input
        writer.append_data(frame)
    writer.close()
    
    # Create GIF
    gif_path = os.path.join(backend_dir, "start_animation_refactored.gif")
    gif_frames = processed_frames[::3]
    imageio.mimsave(gif_path, gif_frames, fps=20, loop=0)
    
    print(f"\n✓ Refactored animation complete!")
    print(f"Outputs:")
    print(f"  - {output_video}")
    print(f"  - {gif_path}")
    
    return processed_frames


if __name__ == "__main__":
    frames = test_full_animation()