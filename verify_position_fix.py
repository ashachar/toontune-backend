#!/usr/bin/env python3
"""Verify the position fix - ensure perfect continuity between phases"""

import cv2
import numpy as np
from utils.animations.text_3d_motion_dissolve_fixed import Text3DMotionDissolve

print("="*70)
print("VERIFYING POSITION FIX")
print("="*70)
print("\n✅ Fix implemented:")
print("  • Motion final position calculated exactly")
print("  • Dissolve starts from exact motion endpoint")
print("  • Letter positions use motion final coordinates")
print("")

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

# Create animation with debug enabled
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
    randomize_order=False,  # Keep order consistent for testing
    maintain_kerning=True,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=True,  # Enable debug output
)

print("\n" + "="*70)
print("GENERATING FRAMES AROUND TRANSITION")
print("="*70)

# Generate frames around transition
transition_frame = anim.motion_frames
test_frames = list(range(max(0, transition_frame - 5), min(transition_frame + 10, anim.total_frames)))

output_frames = []
for i in test_frames:
    if i == transition_frame - 1:
        print(f"\n>>> Frame {i}: LAST MOTION FRAME <<<")
    elif i == transition_frame:
        print(f"\n>>> Frame {i}: FIRST DISSOLVE FRAME <<<")
    elif i % 5 == 0:
        print(f"\nFrame {i}:")
    
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append((i, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))

print("\n" + "="*70)
print("VISUAL POSITION ANALYSIS")
print("="*70)

def detect_text_bounds(frame):
    """Detect bounding box of yellow text in frame"""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
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

print("\nText position detection:")
print("-" * 50)
print("Frame | Center X | Center Y | Width | Height")
print("-" * 50)

positions = []
for frame_idx, frame in output_frames:
    bounds = detect_text_bounds(frame)
    if bounds:
        x, y, w, h = bounds
        center_x = x + w/2
        center_y = y + h/2
        positions.append((frame_idx, center_x, center_y, w, h))
        
        phase = "M" if frame_idx < transition_frame else "D"
        print(f"{frame_idx:5d}{phase} | {center_x:8.1f} | {center_y:8.1f} | {w:5d} | {h:6d}")

# Check for jump at transition
print("\n" + "="*70)
print("POSITION CONTINUITY CHECK")
print("="*70)

if positions:
    # Find transition positions
    last_motion_pos = None
    first_dissolve_pos = None
    
    for idx, cx, cy, w, h in positions:
        if idx == transition_frame - 1:
            last_motion_pos = (idx, cx, cy, w, h)
        elif idx == transition_frame:
            first_dissolve_pos = (idx, cx, cy, w, h)
    
    if last_motion_pos and first_dissolve_pos:
        jump_x = first_dissolve_pos[1] - last_motion_pos[1]
        jump_y = first_dissolve_pos[2] - last_motion_pos[2]
        size_change_w = first_dissolve_pos[3] - last_motion_pos[3]
        size_change_h = first_dissolve_pos[4] - last_motion_pos[4]
        jump_distance = np.sqrt(jump_x**2 + jump_y**2)
        
        print(f"\nTransition analysis (Frame {last_motion_pos[0]} → {first_dissolve_pos[0]}):")
        print(f"  Last motion position:    ({last_motion_pos[1]:.1f}, {last_motion_pos[2]:.1f})")
        print(f"  First dissolve position: ({first_dissolve_pos[1]:.1f}, {first_dissolve_pos[2]:.1f})")
        print(f"\n  Position change:")
        print(f"    X: {jump_x:+.1f} pixels")
        print(f"    Y: {jump_y:+.1f} pixels")
        print(f"    Distance: {jump_distance:.1f} pixels")
        print(f"\n  Size change:")
        print(f"    Width:  {size_change_w:+d} pixels")
        print(f"    Height: {size_change_h:+d} pixels")
        
        if jump_distance < 2.0:
            print("\n✅ POSITION CONTINUITY: EXCELLENT (< 2 pixels)")
        elif jump_distance < 5.0:
            print("\n✅ POSITION CONTINUITY: GOOD (< 5 pixels)")
        else:
            print(f"\n⚠️ POSITION JUMP DETECTED: {jump_distance:.1f} pixels")

# Save comparison frames
print("\n" + "="*70)
print("SAVING VERIFICATION FRAMES")
print("="*70)

# Save key frames
for idx, frame in output_frames:
    if idx == transition_frame - 1:
        cv2.imwrite('verify_last_motion.png', frame)
        print(f"Saved: verify_last_motion.png (frame {idx})")
    elif idx == transition_frame:
        cv2.imwrite('verify_first_dissolve.png', frame)
        print(f"Saved: verify_first_dissolve.png (frame {idx})")
    elif idx == transition_frame + 5:
        cv2.imwrite('verify_mid_dissolve.png', frame)
        print(f"Saved: verify_mid_dissolve.png (frame {idx})")

# Create side-by-side comparison
last_motion = None
first_dissolve = None
for idx, frame in output_frames:
    if idx == transition_frame - 1:
        last_motion = frame
    elif idx == transition_frame:
        first_dissolve = frame

if last_motion is not None and first_dissolve is not None:
    # Stack horizontally
    comparison = np.hstack([last_motion, first_dissolve])
    
    # Add visual markers
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    # Labels
    draw.text((10, 10), "LAST MOTION", fill=(255, 0, 0), font=font)
    draw.text((W + 10, 10), "FIRST DISSOLVE", fill=(255, 0, 0), font=font)
    
    # Center reference lines
    draw.line([(W//2, 0), (W//2, H)], fill=(0, 255, 0), width=1)
    draw.line([(W + W//2, 0), (W + W//2, H)], fill=(0, 255, 0), width=1)
    
    comparison = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite('verify_transition_comparison.png', comparison)
    print("\nSaved: verify_transition_comparison.png")

print("\n✅ VERIFICATION COMPLETE")
print("\nThe position fix has been applied. Check the results above.")