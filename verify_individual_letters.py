#!/usr/bin/env python3
"""Verify EACH individual letter maintains position at transition"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.animations.text_3d_motion_dissolve_correct import Text3DMotionDissolve

print("="*80)
print("VERIFYING INDIVIDUAL LETTER POSITIONS")
print("="*80)
print("Checking that EACH letter stays in place at transition...\n")

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

# Create animation
motion_duration = 0.75
anim = Text3DMotionDissolve(
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
    randomize_order=False,
    maintain_kerning=True,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=False,
)

# Generate frames at transition
transition_frame = anim.motion_frames
print(f"Transition at frame {transition_frame}\n")

# Generate last motion and first dissolve frames
last_motion_frame = anim.generate_frame(transition_frame - 1, frames[transition_frame - 1])
first_dissolve_frame = anim.generate_frame(transition_frame, frames[transition_frame])

if last_motion_frame.shape[2] == 4:
    last_motion_frame = last_motion_frame[:, :, :3]
if first_dissolve_frame.shape[2] == 4:
    first_dissolve_frame = first_dissolve_frame[:, :, :3]

def detect_individual_letters(frame_rgb, debug_save=None):
    """Detect individual letter positions using connected components"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    
    # Detect yellow text (wider range to catch all letters)
    lower_yellow = np.array([20, 100, 100])  # Wider range
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Clean up mask
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Filter out background and small noise
    letter_data = []
    for i in range(1, num_labels):  # Skip background (label 0)
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
    
    # Save debug image if requested
    if debug_save:
        debug_img = frame_rgb.copy()
        for i, letter in enumerate(letter_data):
            x, y, w, h = int(letter['x']), int(letter['y']), int(letter['width']), int(letter['height'])
            cx, cy = int(letter['center_x']), int(letter['center_y'])
            
            # Draw bounding box
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw center point
            cv2.circle(debug_img, (cx, cy), 3, (0, 255, 0), -1)
            
            # Label with number
            cv2.putText(debug_img, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cv2.imwrite(debug_save, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    return letter_data

print("Detecting letters in last motion frame...")
motion_letters = detect_individual_letters(last_motion_frame, 'debug_motion_letters.png')
print(f"Found {len(motion_letters)} letters/components in motion frame")

print("\nDetecting letters in first dissolve frame...")
dissolve_letters = detect_individual_letters(first_dissolve_frame, 'debug_dissolve_letters.png')
print(f"Found {len(dissolve_letters)} letters/components in dissolve frame")

# Match and compare letters
print("\n" + "="*80)
print("INDIVIDUAL LETTER POSITION COMPARISON")
print("="*80)

# Try to match letters by position
if len(motion_letters) > 0 and len(dissolve_letters) > 0:
    print("\nLetter-by-letter analysis:")
    print("-" * 70)
    print("Letter | Motion Center | Dissolve Center | Shift X | Shift Y | Status")
    print("-" * 70)
    
    max_shifts_x = []
    max_shifts_y = []
    
    # Match letters by finding closest dissolve letter to each motion letter
    for i, m_letter in enumerate(motion_letters[:11]):  # Max 11 letters in "HELLO WORLD"
        # Find closest dissolve letter
        min_dist = float('inf')
        best_match = None
        
        for d_letter in dissolve_letters:
            dist = np.sqrt((m_letter['center_x'] - d_letter['center_x'])**2 + 
                          (m_letter['center_y'] - d_letter['center_y'])**2)
            if dist < min_dist and dist < 100:  # Within 100 pixels
                min_dist = dist
                best_match = d_letter
        
        if best_match:
            shift_x = best_match['center_x'] - m_letter['center_x']
            shift_y = best_match['center_y'] - m_letter['center_y']
            
            max_shifts_x.append(abs(shift_x))
            max_shifts_y.append(abs(shift_y))
            
            status = "✅ OK" if abs(shift_x) < 5 and abs(shift_y) < 5 else "⚠️ MOVED"
            
            print(f"  {i:5d} | ({m_letter['center_x']:6.1f}, {m_letter['center_y']:6.1f}) | "
                  f"({best_match['center_x']:6.1f}, {best_match['center_y']:6.1f}) | "
                  f"{shift_x:7.2f} | {shift_y:7.2f} | {status}")
        else:
            print(f"  {i:5d} | ({m_letter['center_x']:6.1f}, {m_letter['center_y']:6.1f}) | "
                  f"No match found | - | - | ❌ MISSING")
    
    if max_shifts_x:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\nMaximum shifts detected:")
        print(f"  Max X shift: {max(max_shifts_x):.2f} pixels")
        print(f"  Max Y shift: {max(max_shifts_y):.2f} pixels")
        print(f"  Average X shift: {np.mean(max_shifts_x):.2f} pixels")
        print(f"  Average Y shift: {np.mean(max_shifts_y):.2f} pixels")
        
        if max(max_shifts_x) < 5 and max(max_shifts_y) < 5:
            print("\n✅ INDIVIDUAL LETTER CONTINUITY: EXCELLENT!")
            print("   All letters maintain their positions")
        else:
            print(f"\n⚠️ LETTER POSITION SHIFTS DETECTED!")
            print(f"   Some letters moved more than 5 pixels")

# Create visual comparison
print("\n" + "="*80)
print("CREATING VISUAL COMPARISON")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Motion frame
axes[0, 0].imshow(last_motion_frame)
axes[0, 0].set_title('Last Motion Frame')
axes[0, 0].axis('off')

# Motion with letter detection
motion_debug = cv2.imread('debug_motion_letters.png')
axes[0, 1].imshow(cv2.cvtColor(motion_debug, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Motion - Letter Detection')
axes[0, 1].axis('off')

# Dissolve frame
axes[1, 0].imshow(first_dissolve_frame)
axes[1, 0].set_title('First Dissolve Frame')
axes[1, 0].axis('off')

# Dissolve with letter detection
dissolve_debug = cv2.imread('debug_dissolve_letters.png')
axes[1, 1].imshow(cv2.cvtColor(dissolve_debug, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Dissolve - Letter Detection')
axes[1, 1].axis('off')

plt.suptitle('Individual Letter Position Verification', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('individual_letter_verification.png', dpi=150)
print("\nSaved: individual_letter_verification.png")

# Create difference image
print("\nCreating difference image...")
diff = cv2.absdiff(last_motion_frame, first_dissolve_frame)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

# Threshold to highlight differences
_, diff_thresh = cv2.threshold(diff_gray, 10, 255, cv2.THRESH_BINARY)

# Save difference image
cv2.imwrite('letter_position_difference.png', diff_thresh)
print("Saved: letter_position_difference.png (white = movement)")

# Count moving pixels
moving_pixels = np.sum(diff_thresh > 0)
total_pixels = diff_thresh.shape[0] * diff_thresh.shape[1]
movement_percentage = (moving_pixels / total_pixels) * 100

print(f"\nMovement analysis:")
print(f"  Moving pixels: {moving_pixels:,}")
print(f"  Total pixels: {total_pixels:,}")
print(f"  Movement: {movement_percentage:.2f}%")

if movement_percentage < 5:
    print("\n✅ Minimal movement detected - letters are stable!")
else:
    print(f"\n⚠️ Significant movement detected: {movement_percentage:.1f}% of pixels changed")

print("\n✅ INDIVIDUAL LETTER VERIFICATION COMPLETE")
print("\nCheck the generated images for visual confirmation.")