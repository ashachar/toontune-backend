# New Video Effects Documentation

## Overview
Seven new professional video effects have been added to match the requirements from "The Sound of Music" video analysis. These effects are designed to be generic and reusable for various video editing scenarios.

## Effects Added

### 1. Dolly Zoom (Camera Push/Pull)
**Function:** `apply_dolly_zoom()`

Creates the cinematic effect of a camera physically moving forward or backward.

```python
from utils.editing_tricks import apply_dolly_zoom

# Slow push in (dolly forward)
output = apply_dolly_zoom(
    "scene.mp4",
    dolly_speed=0.015,      # Slow movement
    dolly_direction="in",    # Push forward
    smooth_acceleration=True # Ease in/out
)

# Pull back effect
output = apply_dolly_zoom(
    "scene.mp4",
    dolly_direction="out"
)
```

**Use Cases:**
- Creating intimacy by slowly pushing into a scene
- Revealing context by pulling back
- Building tension or focus

### 2. Rack Focus
**Function:** `apply_rack_focus()`

Shifts focus between different subjects to guide viewer attention.

```python
from utils.editing_tricks import apply_rack_focus

# Shift focus between two characters
output = apply_rack_focus(
    "dialogue.mp4",
    focus_points=[(200, 300), (600, 300)],  # Character positions
    focus_timings=[0, 3, 6],                 # When to shift
    blur_strength=15.0,                      # Background blur amount
    transition_duration=1.0                  # Smooth transition
)
```

**Use Cases:**
- Dialogue scenes switching between speakers
- Revealing background elements
- Creating depth and directing attention

### 3. Handheld Camera Shake
**Function:** `apply_handheld_shake()`

Adds realistic camera movement for documentary-style footage.

```python
from utils.editing_tricks import apply_handheld_shake

# Subtle documentary feel
output = apply_handheld_shake(
    "interview.mp4",
    shake_intensity=3.0,     # Subtle movement
    shake_frequency=2.0,      # Natural frequency
    rotation_amount=1.0,      # Slight rotation
    smooth_motion=True        # Organic movement
)
```

**Use Cases:**
- Documentary or found-footage style
- Adding energy to static shots
- Creating urgency or tension

### 4. Speed Ramp (Variable Slow Motion)
**Function:** `apply_speed_ramp()`

Dynamically changes playback speed for dramatic effect.

```python
from utils.editing_tricks import apply_speed_ramp

# Slow motion at key moment
output = apply_speed_ramp(
    "action.mp4",
    speed_points=[
        (0, 1.0),    # Normal speed at start
        (2, 0.2),    # Slow to 20% speed at 2 seconds
        (2.5, 0.2),  # Hold slow motion
        (3, 1.0)     # Return to normal
    ],
    interpolation="smooth"
)
```

**Use Cases:**
- Emphasizing key action moments
- Creating dramatic reveals
- Athletic or dance sequences

### 5. Bloom Effect
**Function:** `apply_bloom_effect()`

Adds soft, dreamy glow to bright areas.

```python
from utils.editing_tricks import apply_bloom_effect

# Warm, romantic bloom
output = apply_bloom_effect(
    "sunset.mp4",
    threshold=180,                    # Brightness threshold
    bloom_intensity=1.5,               # Glow strength
    blur_radius=21,                    # Softness
    color_shift=(1.3, 1.1, 0.7)      # Warm tint
)
```

**Use Cases:**
- Romantic or dreamy sequences
- Enhancing sunlight and highlights
- Creating ethereal atmosphere
- Magic or fantasy effects

### 6. Ken Burns Effect
**Function:** `apply_ken_burns()`

Creates dynamic video from still images with pan and zoom.

```python
from utils.editing_tricks import apply_ken_burns

# Pan across a landscape while zooming
output = apply_ken_burns(
    "landscape.jpg",
    duration=8.0,                      # 8-second video
    fps=30,                            # Frame rate
    start_scale=1.0,                   # Original size
    end_scale=1.5,                     # Zoom to 150%
    start_position=(100, 200),        # Start position
    end_position=(800, 200),          # End position
    easing="ease_in_out"               # Smooth motion
)
```

**Use Cases:**
- Documentary photo montages
- Title sequences
- Historical photo presentations
- Slideshow enhancements

### 7. Light Sweep/Shimmer
**Function:** `apply_light_sweep()`

Creates a moving light reflection across the frame.

```python
from utils.editing_tricks import apply_light_sweep

# Golden shimmer on title
output = apply_light_sweep(
    "title.mp4",
    sweep_duration=1.0,               # 1-second sweep
    sweep_width=150,                   # Width of light band
    sweep_angle=45.0,                  # Diagonal sweep
    sweep_color=(255, 215, 0),        # Golden color
    sweep_intensity=0.7,               # Brightness
    repeat_interval=3.0                # Repeat every 3 seconds
)
```

**Use Cases:**
- Title card enhancements
- Logo animations
- Magical or sparkle effects
- Product showcase highlights

## Integration Example

Here's how these effects can work together for a complete scene:

```python
from utils.editing_tricks import *

# Scene 1: Establish with Ken Burns on landscape photo
intro = apply_ken_burns(
    "mountain_photo.jpg",
    duration=5.0,
    end_scale=1.3
)

# Scene 2: Dolly in to reveal subjects
reveal = apply_dolly_zoom(
    "wide_shot.mp4",
    dolly_direction="in",
    dolly_speed=0.02
)

# Scene 3: Dialogue with rack focus
dialogue = apply_rack_focus(
    "conversation.mp4",
    focus_points=[(300, 400), (700, 400)],
    focus_timings=[0, 3, 6, 9]
)

# Scene 4: Action with speed ramp
action = apply_speed_ramp(
    "jumping.mp4",
    speed_points=[(0, 1.0), (1.5, 0.3), (2, 1.0)]
)

# Scene 5: Romantic moment with bloom
romantic = apply_bloom_effect(
    "sunset_scene.mp4",
    bloom_intensity=2.0,
    color_shift=(1.2, 1.0, 0.8)
)

# Title: Add shimmer effect
title = apply_light_sweep(
    "end_title.mp4",
    sweep_color=(255, 255, 200),
    sweep_intensity=0.5
)
```

## Performance Considerations

- **Processing Time:** These effects process frame-by-frame, so longer videos take more time
- **Memory Usage:** Large videos may require significant temporary disk space
- **Quality Settings:** Higher blur radii and more complex calculations increase processing time
- **Optimization:** Consider downsampling videos first using `utils/downsample/video_downsample.py`

## Testing

Run the test script to verify all effects are working:

```bash
python test_new_effects.py
```

This will create sample outputs in the `output/` directory demonstrating each effect.