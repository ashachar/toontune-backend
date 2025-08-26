#!/usr/bin/env python3
"""Debug mask extraction to find the stale mask bug."""

import cv2
import numpy as np
import sys
import os
sys.path.append('.')

# Modify the dissolve class to add debug output
debug_code = '''
    def generate_frame(self, frame_number: int, background: np.ndarray) -> np.ndarray:
        # At the beginning of generate_frame, add logging
        print(f"\\n[MASK_DEBUG] Frame {frame_number}: generate_frame called")
        print(f"[MASK_DEBUG] Background shape: {background.shape}")
        print(f"[MASK_DEBUG] is_behind: {self.is_behind}")
        
        # Ensure sprites are created if not already (for standalone usage)
        if not self.letter_sprites:
            self._prepare_letter_sprites()
            self._init_dissolve_order()
            self._build_frame_timeline()
        
        frame = background.copy()
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        # Optional dynamic mask when behind subject
        current_mask = None
        if self.is_behind:
            print(f"[MASK_DEBUG] Frame {frame_number}: Extracting fresh mask...")
            # ALWAYS extract fresh mask for EVERY frame - NO CACHING!
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from video.segmentation.segment_extractor import extract_foreground_mask
                current_rgb = background[:, :, :3] if background.shape[2] == 4 else background
                
                # CRITICAL FIX: Extract mask for EVERY frame
                current_mask = extract_foreground_mask(current_rgb)
                
                # DEBUG: Prove mask is changing
                mask_pixels = np.sum(current_mask > 128)
                mask_center = np.array(np.where(current_mask > 128)).mean(axis=1) if mask_pixels > 0 else [0, 0]
                print(f"[MASK_DEBUG] Frame {frame_number}: Mask extracted - {mask_pixels:,} pixels, center at ({mask_center[1]:.0f}, {mask_center[0]:.0f})")
                
                # DEBUG: Save mask for frame 30 to visualize
                if frame_number == 30:
                    cv2.imwrite(f'outputs/debug_mask_frame_{frame_number}.png', current_mask)
                    print(f"[MASK_DEBUG] Saved mask to outputs/debug_mask_frame_{frame_number}.png")
'''

print("Debug code to add to letter_3d_dissolve.py:")
print(debug_code)

# Now let's create a test that logs mask extraction
from utils.animations.apply_3d_text_animation import apply_animation_to_video

# Create test with debug enabled
os.environ['DEBUG_3D_TEXT'] = '1'

print("\n" + "="*60)
print("Running test with debug logging...")
print("="*60)

result = apply_animation_to_video(
    video_path="test_person_h264.mp4",
    text="Hello",
    font_size=80,
    position=(640, 360),
    motion_duration=0.5,
    dissolve_duration=1.5,
    output_path="outputs/debug_mask_test.mp4",
    final_opacity=0.7,
    supersample=2,
    debug=True
)

print(f"\n✅ Created: {result}")
print("\nCheck the console output above for [MASK_DEBUG] lines to see if mask is extracted every frame.")