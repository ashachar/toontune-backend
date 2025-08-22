#!/usr/bin/env python3
"""
Fix to make alpha (transparency) change exponentially during shrink phase.
Starts very slowly and accelerates as text shrinks more.
Final value remains 0.5 but the rate of change is exponential.
"""

import os

# Read text_behind_segment.py
with open('utils/animations/text_behind_segment.py', 'r') as f:
    content = f.read()

# Find the linear alpha transition section
old_section = """        # Apply transparency during shrink phase to show text moving to back
        if phase == "foreground" and frame_idx > 0:
            # During shrink (phase 1), fade from 1.0 to 0.5 alpha
            phase_progress = frame_idx / self.phase1_end if self.phase1_end > 0 else 1
            target_alpha = 1.0 - (0.5 * phase_progress)  # Goes from 1.0 to 0.5
            text_alpha = text_alpha * target_alpha"""

new_section = """        # Apply transparency during shrink phase to show text moving to back
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
            text_alpha = text_alpha * target_alpha"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('utils/animations/text_behind_segment.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix: Exponential alpha transition")
    print("  Alpha now changes exponentially - slow at start, fast at end")
    print("  Still reaches 0.5 (50% transparency) but with curved rate")
    print("  k=3.0 provides nice exponential curve")
else:
    print("Could not find exact pattern")
    print("Trying alternative approach...")
    
    # Try simpler pattern match
    alt_old = "target_alpha = 1.0 - (0.5 * phase_progress)  # Goes from 1.0 to 0.5"
    
    if alt_old in content:
        # Find the line and replace it with exponential version
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            if alt_old in line:
                indent = len(line) - len(line.lstrip())
                new_lines.append(line.replace(alt_old, "# OLD LINEAR: target_alpha = 1.0 - (0.5 * phase_progress)"))
                new_lines.append(" " * indent + "# Exponential curve: starts slowly, accelerates as progress increases")
                new_lines.append(" " * indent + "k = 3.0  # Curve factor - higher = more exponential")
                new_lines.append(" " * indent + "import math")
                new_lines.append(" " * indent + "exp_progress = (math.exp(k * phase_progress) - 1) / (math.exp(k) - 1)")
                new_lines.append(" " * indent + "target_alpha = 1.0 - (0.5 * exp_progress)  # Exponential from 1.0 to 0.5")
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        with open('utils/animations/text_behind_segment.py', 'w') as f:
            f.write(content)
        print("✓ Applied alternative fix")
    else:
        print("Could not apply fix automatically")