#!/usr/bin/env python3
"""
Fix to make text start fading only AFTER it passes behind the subject.
Text remains fully opaque during shrink (phase 1), then fades during transition (phase 2).
"""

import os

# Read text_behind_segment.py
with open('utils/animations/text_behind_segment.py', 'r') as f:
    content = f.read()

# Find the section with alpha transitions
old_section = """        # Apply transparency during shrink phase to show text moving to back
        if phase == "foreground" and frame_idx > 0:
            # During shrink (phase 1), fade from 1.0 to 0.5 alpha with EXPONENTIAL curve
            # Exponential curve: starts slowly, accelerates as progress increases
            phase_progress = frame_idx / self.phase1_end if self.phase1_end > 0 else 1
            
            # Exponential function: y = (e^(k*x) - 1) / (e^k - 1)
            # where k controls the curve steepness (higher = more exponential)
            k = 3.0  # Curve factor - adjust for more/less exponential behavior
            
            # Calculate exponential progress (0 to 1)
            import math
            exp_progress = (math.exp(k * phase_progress) - 1) / (math.exp(k) - 1)
            
            # Apply to alpha: still goes from 1.0 to 0.5, but with exponential rate
            target_alpha = 1.0 - (0.5 * exp_progress)
            text_alpha = text_alpha * target_alpha
        elif phase == "transition":
            # During transition (phase 2), maintain 0.5 alpha
            text_alpha = text_alpha * 0.5
        elif phase == "background":
            # During stable behind (phase 3), keep at 0.5 alpha
            # The final fade out will happen during dissolve
            text_alpha = text_alpha * 0.5"""

new_section = """        # Apply transparency ONLY AFTER text passes behind subject
        if phase == "foreground":
            # During shrink (phase 1), text stays FULLY OPAQUE
            # No fading at all - text is still in front
            pass  # Keep text_alpha as is (fully opaque)
        elif phase == "transition":
            # During transition (phase 2), text passes behind - START FADING HERE
            # Use exponential curve for dramatic fade as it goes behind
            phase_progress = (frame_idx - self.phase1_end) / (self.phase2_end - self.phase1_end)
            
            # Exponential function for smooth fade from 1.0 to 0.5
            import math
            k = 3.0  # Curve factor - higher = more exponential
            exp_progress = (math.exp(k * phase_progress) - 1) / (math.exp(k) - 1)
            
            # Fade from fully opaque (1.0) to semi-transparent (0.5)
            target_alpha = 1.0 - (0.5 * exp_progress)
            text_alpha = text_alpha * target_alpha
        elif phase == "background":
            # During stable behind (phase 3), maintain 0.5 alpha
            # The final fade out will happen during dissolve
            text_alpha = text_alpha * 0.5"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/text_behind_segment.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix: Text fades ONLY after passing behind subject")
    print("  Phase 1 (shrink): Text remains FULLY OPAQUE")
    print("  Phase 2 (transition): Text fades from 1.0 to 0.5 alpha")
    print("  Phase 3 (stable behind): Text maintains 0.5 alpha")
else:
    print("Could not find exact pattern")
    print("Manual fix may be needed")