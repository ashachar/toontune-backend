#!/usr/bin/env python3
"""Verify that the FIXED animation maintains exact letter positions."""

import cv2
import numpy as np
from utils.animations.text_3d_motion_dissolve_fixed import Text3DMotionDissolveFixed

print("="*80)
print("VERIFYING FIXED LETTER POSITIONS")
print("="*80)

# Load video
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load frames
frames = []
for i in range(50):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Create animation with debug
anim = Text3DMotionDissolveFixed(
    duration=2.25,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    motion_duration=0.75,
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
    randomize_order=False,
    maintain_kerning=True,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=True,
)

# Generate frames at transition
transition_frame = anim.motion_frames
print(f"\nTransition at frame {transition_frame}")

# Generate frames
last_motion = anim.generate_frame(transition_frame - 1, frames[transition_frame - 1])
first_dissolve = anim.generate_frame(transition_frame, frames[transition_frame])

if last_motion.shape[2] == 4:
    last_motion = last_motion[:, :, :3]
if first_dissolve.shape[2] == 4:
    first_dissolve = first_dissolve[:, :, :3]

def detect_individual_letters(frame_rgb, debug_save=None):
    """Detect individual letter positions using connected components."""
    # Convert to HSV
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    
    # Detect yellow text
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Clean up mask
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Filter components
    letter_data = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 500:  # Filter small noise
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            letter_data.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'center_x': cx,
                'center_y': cy,
                'area': area
            })
    
    # Sort by x position
    letter_data.sort(key=lambda l: l['x'])
    
    if debug_save:
        debug_img = frame_rgb.copy()
        for i, letter in enumerate(letter_data):
            x, y, w, h = int(letter['x']), int(letter['y']), int(letter['width']), int(letter['height'])
            cx, cy = int(letter['center_x']), int(letter['center_y'])
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(debug_img, (cx, cy), 3, (0, 255, 0), -1)
        cv2.imwrite(debug_save, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    return letter_data

print("\nDetecting letters in last motion frame...")
motion_letters = detect_individual_letters(last_motion, 'fixed_motion_letters.png')
print(f"Found {len(motion_letters)} components")

print("\nDetecting letters in first dissolve frame...")
dissolve_letters = detect_individual_letters(first_dissolve, 'fixed_dissolve_letters.png')
print(f"Found {len(dissolve_letters)} components")

# Compare positions
print("\n" + "="*80)
print("INDIVIDUAL LETTER POSITION ANALYSIS")
print("="*80)

if len(motion_letters) > 0 and len(dissolve_letters) > 0:
    print("\nLetter-by-letter comparison:")
    print("-" * 70)
    print("Letter | Motion Center | Dissolve Center | Shift X | Shift Y | Status")
    print("-" * 70)
    
    max_shifts_x = []
    max_shifts_y = []
    
    for i, m_letter in enumerate(motion_letters[:11]):
        # Find closest dissolve letter
        min_dist = float('inf')
        best_match = None
        
        for d_letter in dissolve_letters:
            dist = np.sqrt((m_letter['center_x'] - d_letter['center_x'])**2 + 
                          (m_letter['center_y'] - d_letter['center_y'])**2)
            if dist < min_dist and dist < 100:
                min_dist = dist
                best_match = d_letter
        
        if best_match:
            shift_x = best_match['center_x'] - m_letter['center_x']
            shift_y = best_match['center_y'] - m_letter['center_y']
            
            max_shifts_x.append(abs(shift_x))
            max_shifts_y.append(abs(shift_y))
            
            # Much stricter threshold for FIXED version
            status = "✅ PERFECT" if abs(shift_x) < 2 and abs(shift_y) < 2 else "⚠️ MOVED"
            
            print(f"  {i:5d} | ({m_letter['center_x']:6.1f}, {m_letter['center_y']:6.1f}) | "
                  f"({best_match['center_x']:6.1f}, {best_match['center_y']:6.1f}) | "
                  f"{shift_x:7.2f} | {shift_y:7.2f} | {status}")
    
    if max_shifts_x:
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        print(f"\nMaximum shifts detected:")
        print(f"  Max X shift: {max(max_shifts_x):.2f} pixels")
        print(f"  Max Y shift: {max(max_shifts_y):.2f} pixels")
        print(f"  Average X shift: {np.mean(max_shifts_x):.2f} pixels")
        print(f"  Average Y shift: {np.mean(max_shifts_y):.2f} pixels")
        
        if max(max_shifts_x) < 2 and max(max_shifts_y) < 2:
            print("\n🎉 PERFECT! Every single letter maintains exact position!")
            print("   The position jump issue is COMPLETELY FIXED!")
        elif max(max_shifts_x) < 5 and max(max_shifts_y) < 5:
            print("\n✅ EXCELLENT! Individual letters maintain position!")
            print("   The position jump issue is FIXED!")
        else:
            print(f"\n⚠️ Some letters still shifting")

# Check internal state
print("\n" + "="*80)
print("INTERNAL STATE VERIFICATION")
print("="*80)

# Access letter states after motion phase
print("\nLetter states captured from motion phase:")
if anim.letter_states:
    print(f"  Number of letter states: {len(anim.letter_states)}")
    print("\n  Letter positions:")
    for i, state in enumerate(anim.letter_states):
        if state.sprite_3d is not None:
            print(f"    '{state.char}': position=({state.position[0]}, {state.position[1]})")
    
    # Calculate center
    x_positions = [s.position[0] for s in anim.letter_states if s.sprite_3d is not None]
    if x_positions:
        avg_x = np.mean(x_positions)
        print(f"\n  Average X position: {avg_x:.1f}")
        print(f"  Expected center: {W//2}")
        print(f"  Difference from center: {abs(avg_x - W//2):.1f} pixels")

print("\n✅ FIXED VERIFICATION COMPLETE")