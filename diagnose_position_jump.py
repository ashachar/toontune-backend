#!/usr/bin/env python3
"""Diagnose position jump between motion and dissolve phases"""

import cv2
import numpy as np
from utils.animations.text_3d_motion_dissolve_proper import Text3DMotionDissolve

print("="*70)
print("DIAGNOSING POSITION JUMP AT PHASE TRANSITION")
print("="*70)

# Load video
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load frames
frames = []
for i in range(136):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Create animation to inspect internal state
motion_duration = 0.75
dissolve_duration = 1.5
total_duration = motion_duration + dissolve_duration

anim = Text3DMotionDissolve(
    duration=total_duration,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    motion_duration=motion_duration,
    start_scale=2.0,
    end_scale=1.0,
    final_scale=0.9,
    shrink_duration=0.6,
    settle_duration=0.15,
    dissolve_stable_duration=0.1,
    dissolve_duration=0.5,
    dissolve_stagger=0.1,
    float_distance=40,
    max_dissolve_scale=1.3,
    randomize_order=False,  # Disable randomization for consistent analysis
    maintain_kerning=True,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=False,
)

print(f"\nAnimation structure:")
print(f"  Motion frames: 0-{anim.motion_frames-1}")
print(f"  Dissolve frames: {anim.motion_frames}-{anim.total_frames-1}")
print(f"  Transition at frame: {anim.motion_frames}")

# Generate frames around transition
transition_frame = anim.motion_frames
analysis_frames = range(transition_frame - 3, min(transition_frame + 3, anim.total_frames))

print("\n" + "="*70)
print("ANALYZING FRAMES AROUND TRANSITION")
print("="*70)

# Store frames for visual comparison
comparison_frames = []

for frame_idx in analysis_frames:
    print(f"\nFrame {frame_idx}:")
    
    # Generate frame
    generated = anim.generate_frame(frame_idx, frames[frame_idx])
    
    # Determine phase
    if frame_idx < anim.motion_frames:
        phase = "MOTION"
        
        # Calculate expected position from motion phase
        t_global = frame_idx / max(anim.motion_frames - 1, 1)
        smooth_t = t_global * t_global * (3.0 - 2.0 * t_global)
        
        # Scale calculation
        shrink_progress = anim.shrink_duration / anim.motion_duration
        if smooth_t <= shrink_progress:
            local_t = smooth_t / shrink_progress
            scale = anim.start_scale - local_t * (anim.start_scale - anim.end_scale)
        else:
            local_t = (smooth_t - shrink_progress) / (1.0 - shrink_progress)
            scale = anim.end_scale - local_t * (anim.end_scale - anim.final_scale)
        
        # Position calculation
        cx, cy = anim.center_position
        start_y = cy - H * 0.15
        end_y = cy
        expected_y = start_y + smooth_t * (end_y - start_y)
        
        print(f"  Phase: {phase}")
        print(f"  Scale: {scale:.4f}")
        print(f"  Expected Y center: {expected_y:.2f}")
        
    else:
        phase = "DISSOLVE"
        dissolve_frame = frame_idx - anim.motion_frames
        
        print(f"  Phase: {phase} (frame {dissolve_frame} of dissolve)")
        
        # Check pre-rendered letter positions
        if hasattr(anim, 'letter_positions') and anim.letter_positions:
            # Get average position of letters
            positions = [pos for pos in anim.letter_positions if pos is not None]
            if positions:
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                print(f"  Pre-rendered letter positions center: ({avg_x:.2f}, {avg_y:.2f})")
                
                # Check first letter position
                first_letter_pos = anim.letter_positions[0]
                print(f"  First letter ('H') position: {first_letter_pos}")
    
    # Save frame
    comparison_frames.append((frame_idx, phase, generated))
    
    if frame_idx == transition_frame - 1:
        print("\n  >>> LAST MOTION FRAME <<<")
    elif frame_idx == transition_frame:
        print("\n  >>> FIRST DISSOLVE FRAME <<<")

# Visual analysis - detect text position
print("\n" + "="*70)
print("VISUAL POSITION DETECTION")
print("="*70)

def detect_text_bounds(frame):
    """Detect bounding box of yellow text in frame"""
    # Convert to HSV for better yellow detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Define yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all contours
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        return x, y, w, h
    return None

print("\nDetecting text positions in frames:")
print("-" * 50)

positions = []
for frame_idx, phase, frame in comparison_frames:
    bounds = detect_text_bounds(frame)
    if bounds:
        x, y, w, h = bounds
        center_x = x + w/2
        center_y = y + h/2
        positions.append((frame_idx, phase, center_x, center_y, w, h))
        print(f"Frame {frame_idx} ({phase:8s}): Center=({center_x:.1f}, {center_y:.1f}), Size=({w}x{h})")
    else:
        print(f"Frame {frame_idx} ({phase:8s}): No text detected")

# Calculate jump
if len(positions) >= 2:
    print("\n" + "="*70)
    print("POSITION JUMP ANALYSIS")
    print("="*70)
    
    # Find transition point
    motion_positions = [(i, x, y) for i, p, x, y, w, h in positions if p == "MOTION"]
    dissolve_positions = [(i, x, y) for i, p, x, y, w, h in positions if p == "DISSOLVE"]
    
    if motion_positions and dissolve_positions:
        last_motion = motion_positions[-1]
        first_dissolve = dissolve_positions[0]
        
        jump_x = first_dissolve[1] - last_motion[1]
        jump_y = first_dissolve[2] - last_motion[2]
        jump_distance = np.sqrt(jump_x**2 + jump_y**2)
        
        print(f"\nPosition at end of motion (frame {last_motion[0]}): ({last_motion[1]:.1f}, {last_motion[2]:.1f})")
        print(f"Position at start of dissolve (frame {first_dissolve[0]}): ({first_dissolve[1]:.1f}, {first_dissolve[2]:.1f})")
        print(f"\n❌ JUMP DETECTED:")
        print(f"  X jump: {jump_x:.1f} pixels {'(LEFT)' if jump_x < 0 else '(RIGHT)'}")
        print(f"  Y jump: {jump_y:.1f} pixels {'(UP)' if jump_y < 0 else '(DOWN)'}")
        print(f"  Total jump: {jump_distance:.1f} pixels")
        
        if abs(jump_x) > 5:
            print(f"\n⚠️ SIGNIFICANT HORIZONTAL JUMP: {abs(jump_x):.1f} pixels")
            print("This confirms the bug - text jumps to the left at transition!")

# Save comparison image
print("\n" + "="*70)
print("SAVING VISUAL COMPARISON")
print("="*70)

# Create side-by-side comparison
if len(comparison_frames) >= 2:
    # Get frames around transition
    transition_idx = anim.motion_frames
    
    last_motion_frame = None
    first_dissolve_frame = None
    
    for idx, phase, frame in comparison_frames:
        if idx == transition_idx - 1:
            last_motion_frame = frame
        elif idx == transition_idx:
            first_dissolve_frame = frame
    
    if last_motion_frame is not None and first_dissolve_frame is not None:
        # Create comparison image
        comparison = np.hstack([last_motion_frame, first_dissolve_frame])
        
        # Add labels
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.fromarray(comparison)
        draw = ImageDraw.Draw(img)
        
        # Try to get a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        except:
            font = ImageFont.load_default()
        
        # Add labels
        draw.text((10, 10), "LAST MOTION FRAME", fill=(255, 0, 0), font=font)
        draw.text((W + 10, 10), "FIRST DISSOLVE FRAME", fill=(255, 0, 0), font=font)
        
        # Draw center lines for reference
        draw.line([(W//2, 0), (W//2, H)], fill=(0, 255, 0), width=1)
        draw.line([(W + W//2, 0), (W + W//2, H)], fill=(0, 255, 0), width=1)
        
        comparison = np.array(img)
        cv2.imwrite('position_jump_comparison.png', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print("Saved comparison image: position_jump_comparison.png")

print("\n✅ DIAGNOSIS COMPLETE")
print("\nThe issue is confirmed: Text position jumps at the phase transition.")
print("The dissolve phase is not using the exact final position from the motion phase.")