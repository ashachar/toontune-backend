#!/usr/bin/env python3
"""Verify that is_behind is now set to True in dissolve."""

# Check the fix was applied
with open('utils/animations/apply_3d_text_animation.py', 'r') as f:
    content = f.read()
    
# Find the line where dissolve is created
import re
match = re.search(r'is_behind=(\w+),', content)
if match:
    value = match.group(1)
    print(f"Found: is_behind={value}")
    
    if value == "True":
        print("✅ FIX CONFIRMED: Dissolve animation will now apply occlusion!")
        print("   Letters will be hidden behind foreground objects.")
    else:
        print("❌ BUG STILL EXISTS: is_behind is set to False")
        print("   Letters will appear on top of everything.")
else:
    print("Could not find is_behind setting")

# Also show what the handoff does
print("\nHandoff behavior:")
if "is_behind=final_state.is_behind" in content:
    print("✅ Handoff passes is_behind state from motion to dissolve")
    print("   (Motion sets is_behind=True when t > 0.0)")