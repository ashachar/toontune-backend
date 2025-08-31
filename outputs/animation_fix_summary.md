# Behind-Text Animation Fix Summary

## Problem
Behind-text words ("Would you be", "surprised if") were appearing static with only fade-in animation, not sliding from above as intended.

## Root Cause
The animation code was correct, but behind words were rendered through a different code path that made consistent animation difficult.

## Solution: Clean Refactor
Separated concerns for cleaner, more maintainable code:

1. **Unified Rendering Path**: All words (front and behind) now use the exact same animation logic
2. **Masking as Post-Processing**: Behind-text masking is applied during compositing, not as a separate render path
3. **Consistent Animation**: Y-offset sliding animation now works identically for all words

## Key Changes

### rendering.py Refactor:
```python
# Before: Complex branching with different paths for behind/front
if word_obj.is_behind:
    # Complex masking logic mixed with rendering
    # Animation might not apply consistently
    
# After: Clean separation of concerns
# Step 1: Calculate animation (same for ALL words)
y_offset = calculate_animation_offset(...)

# Step 2: Create sprite (same for ALL words)  
sprite = _create_word_sprite(word_obj, opacity, fog_progress)

# Step 3: Composite with optional masking
frame = _composite_sprite(sprite, frame, x, y + y_offset, apply_mask=word_obj.is_behind)
```

## Result
✅ Behind-text words now slide from above properly
✅ Animation is consistent for all words
✅ Code is cleaner and more maintainable
✅ Masking still works correctly to hide text behind foreground

## Test Outputs
- `outputs/ai_math1_would_bug_fixed.mp4` - Final video with fix
- `outputs/ai_math1_would_bug_fixed_debug.mp4` - Debug version with bounding boxes

The fix ensures that all text animations work consistently, whether rendered in front of or behind foreground objects.