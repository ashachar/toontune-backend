#!/usr/bin/env python3
"""Comprehensive debug of text left shift issue"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from utils.animations.text_3d_motion_dissolve_fixed import Text3DMotionDissolve
import matplotlib.pyplot as plt

print("="*80)
print("COMPREHENSIVE DEBUG: TEXT LEFT SHIFT ISSUE")
print("="*80)

# Load video
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load frames
frames = []
for i in range(150):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Create animation with MAXIMUM debug
class DebugText3DMotionDissolve(Text3DMotionDissolve):
    """Extended class with detailed position tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_log = []
        self.anchor_log = []
        self.render_log = []
    
    def _calculate_motion_final_state(self):
        """Override to add detailed logging"""
        super()._calculate_motion_final_state()
        print(f"\n[FINAL STATE CALCULATION]")
        print(f"  Motion final scale: {self.motion_final_scale:.6f}")
        print(f"  Motion final position: ({self.motion_final_x:.3f}, {self.motion_final_y:.3f})")
        print(f"  Motion final anchor: ({self.motion_final_anchor[0]:.3f}, {self.motion_final_anchor[1]:.3f})")
    
    def _prepare_3d_letter_sprites_at_position(self):
        """Override to log letter positions"""
        super()._prepare_3d_letter_sprites_at_position()
        print(f"\n[LETTER SPRITE PREPARATION]")
        print(f"  Number of letters: {len(self.text)}")
        print(f"  Letter positions calculated:")
        for i, (letter, pos) in enumerate(zip(self.text, self.letter_positions)):
            if pos:
                print(f"    '{letter}': ({pos[0]}, {pos[1]})")
        
        # Calculate bounding box of all letters
        if self.letter_positions:
            valid_positions = [p for p in self.letter_positions if p is not None]
            if valid_positions:
                min_x = min(p[0] for p in valid_positions)
                max_x = max(p[0] for p in valid_positions)
                min_y = min(p[1] for p in valid_positions)
                max_y = max(p[1] for p in valid_positions)
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                print(f"  Letter bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
                print(f"  Letter center: ({center_x:.1f}, {center_y:.1f})")
    
    def _generate_motion_frame(self, frame_number, frame, background):
        """Override to log motion positions"""
        # Calculate parameters
        t_global = frame_number / max(self.motion_frames - 1, 1)
        smooth_t_global = self._smoothstep(t_global)
        
        # Determine phase
        if frame_number < self.shrink_frames:
            phase = "shrink"
        else:
            phase = "settle"
        
        # Calculate scale
        shrink_progress = self.shrink_duration / self.motion_duration
        if smooth_t_global <= shrink_progress:
            local_t = smooth_t_global / shrink_progress
            scale = self.start_scale - local_t * (self.start_scale - self.end_scale)
        else:
            local_t = (smooth_t_global - shrink_progress) / (1.0 - shrink_progress)
            scale = self.end_scale - local_t * (self.end_scale - self.final_scale)
        
        # Get anchor before rendering
        _, (anchor_x, anchor_y) = self._render_3d_text(
            self.text, scale, 0.2, False, self.transition_scale
        )
        
        # Calculate position
        cx, cy = self.center_position
        start_y = cy - self.resolution[1] * 0.15
        end_y = cy
        pos_x = cx - anchor_x
        pos_y = start_y + smooth_t_global * (end_y - start_y) - anchor_y
        
        # Log detailed info for critical frames
        if frame_number >= self.motion_frames - 5 or frame_number % 10 == 0:
            print(f"\n[MOTION FRAME {frame_number}] Phase: {phase}")
            print(f"  t_global: {t_global:.4f}, smooth_t: {smooth_t_global:.4f}")
            print(f"  Scale: {scale:.6f}")
            print(f"  Anchor: ({anchor_x:.3f}, {anchor_y:.3f})")
            print(f"  Center: ({cx}, {cy})")
            print(f"  Calculated pos: ({pos_x:.3f}, {pos_y:.3f})")
            print(f"  Text top-left: ({int(pos_x)}, {int(pos_y)})")
        
        self.position_log.append((frame_number, "motion", pos_x, pos_y, scale))
        self.anchor_log.append((frame_number, anchor_x, anchor_y))
        
        # Call parent implementation
        return super()._generate_motion_frame(frame_number, frame, background)
    
    def _generate_3d_dissolve_frame(self, dissolve_frame, frame, background):
        """Override to log dissolve positions"""
        frame_number = dissolve_frame + self.motion_frames
        
        if dissolve_frame <= 5:
            print(f"\n[DISSOLVE FRAME {dissolve_frame}] (absolute frame {frame_number})")
            print(f"  Using letter positions from motion final state")
            print(f"  First letter position: {self.letter_positions[0] if self.letter_positions else 'None'}")
            
            # Calculate effective text bounds
            if self.letter_positions:
                valid_pos = [p for p in self.letter_positions if p is not None]
                if valid_pos:
                    min_x = min(p[0] for p in valid_pos)
                    max_x = max(p[0] for p in valid_pos) + 100  # Approximate letter width
                    center_x = (min_x + max_x) / 2
                    print(f"  Dissolve text bounds: x={min_x} to {max_x}")
                    print(f"  Dissolve text center X: {center_x:.1f}")
        
        return super()._generate_3d_dissolve_frame(dissolve_frame, frame, background)

# Create debug animation
print("\nCreating animation with debug logging...")
motion_duration = 0.75
dissolve_duration = 1.5
total_duration = motion_duration + dissolve_duration

anim = DebugText3DMotionDissolve(
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
    randomize_order=False,  # Consistent order for debugging
    maintain_kerning=True,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=True,
)

print("\n" + "="*80)
print("GENERATING CRITICAL FRAMES")
print("="*80)

# Generate frames around transition
transition_frame = anim.motion_frames
test_range = range(max(0, transition_frame - 10), min(transition_frame + 10, 136))

generated_frames = []
for i in test_range:
    if i == transition_frame - 1:
        print(f"\n{'='*50}")
        print(f">>> LAST MOTION FRAME {i} <<<")
        print(f"{'='*50}")
    elif i == transition_frame:
        print(f"\n{'='*50}")
        print(f">>> FIRST DISSOLVE FRAME {i} <<<")
        print(f"{'='*50}")
    
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    generated_frames.append((i, frame))

print("\n" + "="*80)
print("ANALYZING ACTUAL TEXT POSITIONS IN FRAMES")
print("="*80)

def detect_text_position(frame_rgb):
    """Detect yellow text position with high precision"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    
    # Yellow detection
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find leftmost and rightmost yellow pixels
    yellow_pixels = np.where(mask > 0)
    if len(yellow_pixels[0]) > 0:
        min_x = np.min(yellow_pixels[1])
        max_x = np.max(yellow_pixels[1])
        min_y = np.min(yellow_pixels[0])
        max_y = np.max(yellow_pixels[0])
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        return {
            'left': min_x,
            'right': max_x,
            'top': min_y,
            'bottom': max_y,
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height
        }
    return None

print("\nActual text positions in generated frames:")
print("-" * 80)
print("Frame | Left X | Center X | Right X | Width | Phase")
print("-" * 80)

actual_positions = []
for frame_idx, frame_rgb in generated_frames:
    pos = detect_text_position(frame_rgb)
    if pos:
        phase = "MOTION" if frame_idx < transition_frame else "DISSOLVE"
        actual_positions.append((frame_idx, pos, phase))
        
        # Highlight critical frames
        marker = ""
        if frame_idx == transition_frame - 1:
            marker = " <-- LAST MOTION"
        elif frame_idx == transition_frame:
            marker = " <-- FIRST DISSOLVE"
        
        print(f"{frame_idx:5d} | {pos['left']:6.1f} | {pos['center_x']:8.1f} | {pos['right']:7.1f} | {pos['width']:5.1f} | {phase:8s}{marker}")

# Analyze position changes
print("\n" + "="*80)
print("POSITION CHANGE ANALYSIS")
print("="*80)

if len(actual_positions) > 1:
    print("\nFrame-to-frame position changes:")
    print("-" * 60)
    print("Frame | ΔLeft | ΔCenter | ΔRight | Analysis")
    print("-" * 60)
    
    for i in range(1, len(actual_positions)):
        prev_frame, prev_pos, prev_phase = actual_positions[i-1]
        curr_frame, curr_pos, curr_phase = actual_positions[i]
        
        delta_left = curr_pos['left'] - prev_pos['left']
        delta_center = curr_pos['center_x'] - prev_pos['center_x']
        delta_right = curr_pos['right'] - prev_pos['right']
        
        # Flag significant jumps
        analysis = ""
        if abs(delta_left) > 5:
            analysis = f"LEFT JUMP: {delta_left:+.1f}px!"
        elif abs(delta_center) > 3:
            analysis = f"Center shift: {delta_center:+.1f}px"
        
        # Highlight transition
        if prev_frame == transition_frame - 1:
            print("-" * 60)
            print(">>> PHASE TRANSITION <<<")
            print("-" * 60)
        
        print(f"{curr_frame:5d} | {delta_left:6.1f} | {delta_center:8.1f} | {delta_right:7.1f} | {analysis}")

# Find the exact jump
print("\n" + "="*80)
print("IDENTIFYING THE JUMP")
print("="*80)

# Look for large left position changes
for i in range(1, len(actual_positions)):
    prev_frame, prev_pos, _ = actual_positions[i-1]
    curr_frame, curr_pos, _ = actual_positions[i]
    
    left_jump = curr_pos['left'] - prev_pos['left']
    
    if left_jump < -5:  # Significant leftward jump
        print(f"\n❌ FOUND LEFT JUMP at frame {curr_frame}!")
        print(f"  Previous left edge: {prev_pos['left']:.1f}")
        print(f"  Current left edge: {curr_pos['left']:.1f}")
        print(f"  Jump amount: {left_jump:.1f} pixels")
        print(f"  This is {'BEFORE' if curr_frame < transition_frame else 'AT' if curr_frame == transition_frame else 'AFTER'} the phase transition")
        
        # Save comparison frames
        for idx, frame in generated_frames:
            if idx == prev_frame:
                cv2.imwrite(f'debug_frame_{prev_frame:03d}_before_jump.png', 
                           cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            elif idx == curr_frame:
                cv2.imwrite(f'debug_frame_{curr_frame:03d}_after_jump.png', 
                           cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Visualize position tracking
print("\n" + "="*80)
print("CREATING POSITION GRAPH")
print("="*80)

if actual_positions:
    frames_list = [f for f, _, _ in actual_positions]
    left_positions = [p['left'] for _, p, _ in actual_positions]
    center_positions = [p['center_x'] for _, p, _ in actual_positions]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(frames_list, left_positions, 'b-o', markersize=4)
    plt.axvline(x=transition_frame, color='r', linestyle='--', label='Phase transition')
    plt.xlabel('Frame')
    plt.ylabel('Left edge X position')
    plt.title('Text Left Edge Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(frames_list, center_positions, 'g-o', markersize=4)
    plt.axvline(x=transition_frame, color='r', linestyle='--', label='Phase transition')
    plt.xlabel('Frame')
    plt.ylabel('Center X position')
    plt.title('Text Center Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.suptitle('Text Position Tracking - Debug Left Shift', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('debug_position_tracking.png', dpi=150)
    print("Saved: debug_position_tracking.png")

print("\n✅ DEBUG ANALYSIS COMPLETE")
print("\nCheck the logs above and the generated images to identify the exact cause.")