#!/usr/bin/env python3
"""
Fix for text remaining visible after all letters have dissolved.

Simple solution: When ALL letters are completed, just return the frame without any text.
"""

import os

# Read the current word_dissolve.py
with open('utils/animations/word_dissolve.py', 'r') as f:
    content = f.read()

# Find the section after states are computed
old_code = """        # states
        not_started, dissolving, completed = [], [], []
        for i in range(len(self.word)):
            info = next(d for d in self.letter_dissolvers if d["index"] == i)
            s = info["start_frame"]
            e = s + self.dissolve_frames
            if frame_idx < s:
                not_started.append(i)
            elif frame_idx < e:
                dissolving.append(i)
            else:
                completed.append(i)

        # (1) Grow the persistent kill mask"""

new_code = """        # states
        not_started, dissolving, completed = [], [], []
        for i in range(len(self.word)):
            info = next(d for d in self.letter_dissolvers if d["index"] == i)
            s = info["start_frame"]
            e = s + self.dissolve_frames
            if frame_idx < s:
                not_started.append(i)
            elif frame_idx < e:
                dissolving.append(i)
            else:
                completed.append(i)

        # CRITICAL FIX: If ALL letters are completed, return frame without any text
        if len(completed) == len(self.word):
            self._dbg(f"frame={frame_idx}: All letters completed, returning clean frame")
            return result  # result is frame.copy() from line 511

        # (1) Grow the persistent kill mask"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('utils/animations/word_dissolve.py', 'w') as f:
        f.write(content)
    print("✓ Applied fix: Function now returns immediately when all letters are completed")
    print("  This ensures no text rendering happens after dissolve completion")
else:
    print("Could not find exact pattern. Trying alternative approach...")
    
    # Alternative: Find the section and insert the check
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'completed.append(i)' in line:
            # Found the line where completed state is set
            # Insert the check after the for loop
            j = i + 1
            while j < len(lines) and lines[j].strip() != '':
                j += 1
            
            insert_lines = [
                "",
                "        # CRITICAL FIX: If ALL letters are completed, return frame without any text",
                "        if len(completed) == len(self.word):",
                "            self._dbg(f\"frame={frame_idx}: All letters completed, returning clean frame\")",
                "            return result  # Clean frame without text",
                ""
            ]
            
            for idx, new_line in enumerate(insert_lines):
                lines.insert(j + idx, new_line)
            
            content = '\n'.join(lines)
            with open('utils/animations/word_dissolve.py', 'w') as f:
                f.write(content)
            print("✓ Applied alternative fix at line", j)
            break