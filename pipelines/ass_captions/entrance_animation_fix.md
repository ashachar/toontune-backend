# Entrance Animation Timing Issue

## Problem
Entrance animations are not visible - text appears abruptly at their start timestamps instead of animating in smoothly.

## Root Cause Analysis

### Current Behavior
1. **Visibility Gate**: `if phrase["start_time"] <= current_time <= draw_end`
   - Phrases only become visible AFTER their start_time
   - No rendering happens before start_time

2. **Animation Timing**:
   - At `current_time = phrase["start_time"]`:
     - `time_since_start = 0`
     - `progress = 0 / 0.4 = 0`
     - `base_opacity = 0` (invisible)
   - Text starts from 0% opacity AFTER it should already be visible

3. **Result**: 
   - Frame before start_time: No text (not rendered)
   - Frame at start_time: Text at 0% opacity
   - Frames after: Animation plays but text is already "late"

## The Fix

### Concept
Similar to disappearance fix, we need:
- **Logical Start**: When text should be fully visible (`phrase["start_time"]`)
- **Draw Start**: When to start rendering (`logical_start - entrance_duration`)

### Timeline Example
For a phrase starting at t=1.0s with 0.4s entrance:
- **t=0.6s**: Start rendering at 0% opacity (draw_start)
- **t=0.8s**: 50% through animation, 50% opacity
- **t=1.0s**: Animation complete, 100% opacity (logical_start)
- **t=2.0s**: Steady state at 100% opacity

### Implementation Changes

1. **render_phrase method**:
```python
# Calculate timing windows
logical_start = phrase["start_time"]
logical_end = scene_end_time if scene_end_time else phrase["end_time"]
entrance_duration = 0.4  # 400ms for entrance
disappear_duration = 0.5  # 500ms for disappearance

# Drawing windows extend beyond logical times
draw_start = logical_start - entrance_duration
draw_end = logical_end + disappear_duration

# Single visibility gate using extended window
if not (draw_start <= current_time <= draw_end):
    return None

# Calculate progress relative to logical times
if current_time < logical_start:
    # ENTRANCE PHASE
    time_until_start = logical_start - current_time
    progress = 1.0 - (time_until_start / entrance_duration)
    base_opacity = progress
elif current_time < logical_end:
    # STEADY PHASE
    base_opacity = 1.0
else:
    # DISAPPEAR PHASE
    disappear_progress = (current_time - logical_end) / disappear_duration
    base_opacity = 1.0 - disappear_progress
```

2. **Main loop phrase collection**:
```python
entrance_duration = 0.4  # Must match render_phrase
disappear_duration = 0.5  # Must match render_phrase

for phrase in transcript_data.get("phrases", []):
    logical_start = phrase["start_time"]
    logical_end = scene_end_times.get(phrase_key, phrase["end_time"])
    
    # Extended visibility window for animations
    draw_start = logical_start - entrance_duration
    draw_end = logical_end + disappear_duration
    
    if draw_start <= current_time <= draw_end:
        # Collect phrase for rendering
```

## Expected Results
- Text will fade/slide in BEFORE their logical start time
- At the logical start time, text will be at 100% opacity
- Smooth entrance animations matching the assigned effects
- Word-by-word effects will cascade properly