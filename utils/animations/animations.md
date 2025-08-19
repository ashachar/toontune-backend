# Animation Library Documentation

This document describes all available animation types in the ToonTune animation system. Each animation can be applied to images, videos, or text elements to create dynamic visual effects.

## Base Animation Class

All animations inherit from the `Animation` base class (`animate.py`) which provides:
- Frame extraction from videos/images
- Background video handling
- Frame compositing
- Path-based movement
- Video rendering

## Available Animations

### 🌀 Entry/Exit Animations

#### 1. **Fade In** (`fade_in.py`)
Element gradually appears by increasing opacity from 0 to 1.

**Parameters:**
- `center_point`: (x, y) position where element fades in
- `fade_speed`: Opacity change per frame (0.05 = slow, 0.2 = fast)
- `start_frame`: When to begin fade-in

**Example:**
```python
from utils.animations.fade_in import FadeIn

animation = FadeIn(
    element_path="character.mp4",
    background_path="scene.mp4",
    center_point=(640, 360),
    fade_speed=0.1
)
```

#### 2. **Fade Out** (`fade_out.py`)
Element gradually disappears by decreasing opacity from 1 to 0.

**Parameters:**
- `center_point`: (x, y) position where element fades out
- `fade_speed`: Opacity change per frame
- `fade_start_frame`: Frame at which to start fading out

**Example:**
```python
from utils.animations.fade_out import FadeOut

animation = FadeOut(
    element_path="character.mp4",
    background_path="scene.mp4",
    center_point=(640, 360),
    fade_speed=0.03,  # Slower fade
    fade_start_frame=60  # Start fading at 2 seconds
)
```

#### 3. **Slide In** (`slide_in.py`)
Element slides into view from off-screen edge.

**Parameters:**
- `slide_direction`: 'left', 'right', 'top', 'bottom'
- `slide_duration`: Number of frames for slide animation
- `easing`: 'linear', 'ease_in', 'ease_out', 'ease_in_out'
- `overshoot`: Amount to overshoot past target (0.0-0.2)

**Example:**
```python
from utils.animations.slide_in import SlideIn

animation = SlideIn(
    element_path="title.mp4",
    background_path="scene.mp4",
    position=(640, 360),
    slide_direction='left',
    slide_duration=30,
    easing='ease_out'
)
```

#### 4. **Zoom In** (`zoom_in.py`)
Element scales up from a point to full size.

**Parameters:**
- `start_scale`: Initial scale (0.0 = invisible, 1.0 = full size)
- `end_scale`: Final scale (1.0 = normal size)
- `zoom_duration`: Frames for zoom animation
- `zoom_center`: Center point for zoom effect
- `easing`: Animation easing function
- `rotation_during_zoom`: Degrees to rotate during zoom

**Example:**
```python
from utils.animations.zoom_in import ZoomIn

animation = ZoomIn(
    element_path="logo.mp4",
    background_path="scene.mp4",
    position=(640, 360),
    start_scale=0.0,
    end_scale=1.0,
    zoom_duration=20,
    easing='bounce'
)
```

#### 5. **Bounce** (`bounce.py`)
Element bounces when entering or exiting, with physics simulation.

**Parameters:**
- `bounce_height`: Initial bounce height in pixels
- `num_bounces`: Number of bounces
- `bounce_duration`: Total frames for bounce
- `bounce_type`: 'in', 'out', or 'both'
- `gravity`: Gravity strength
- `damping`: Energy loss per bounce
- `squash_stretch`: Apply cartoon squash/stretch effect

**Example:**
```python
from utils.animations.bounce import Bounce

animation = Bounce(
    element_path="ball.mp4",
    background_path="scene.mp4",
    position=(640, 500),
    bounce_height=200,
    num_bounces=3,
    squash_stretch=True
)
```

#### 6. **Emergence from Static Point** (`emergence_from_static_point.py`)
Element emerges pixel-by-pixel from a fixed point.

**Parameters:**
- `emergence_speed`: Pixels per frame to reveal
- `direction`: Direction of emergence (0° = up, 90° = right, etc.)

**Example:**
```python
from utils.animations.emergence_from_static_point import EmergenceFromStaticPoint

animation = EmergenceFromStaticPoint(
    element_path="character.mp4",
    background_path="water.mp4",
    position=(640, 360),
    direction=0,  # Emerge upward
    emergence_speed=2.0
)
```

#### 7. **Submerge to Static Point** (`submerge_to_static_point.py`)
Element disappears pixel-by-pixel into a fixed point (opposite of emergence).

**Parameters:**
- `submerge_speed`: Pixels per frame to hide
- `submerge_start_frame`: When to start submerging
- `direction`: Direction to submerge into

**Example:**
```python
from utils.animations.submerge_to_static_point import SubmergeToStaticPoint

animation = SubmergeToStaticPoint(
    element_path="character.mp4",
    background_path="water.mp4",
    position=(640, 360),
    direction=180,  # Submerge downward
    submerge_speed=2.0
)
```

### 🎨 Distortion Effects

#### 8. **Skew** (`skew.py`)
Element skews/tilts diagonally, creating a parallelogram effect.

**Parameters:**
- `skew_x`: Horizontal skew angle in degrees (-45 to 45)
- `skew_y`: Vertical skew angle in degrees (-45 to 45)
- `skew_duration`: Frames for skew animation
- `oscillate`: Continuously oscillate skew
- `easing`: Animation easing function

#### 9. **Stretch/Squash** (`stretch_squash.py`)
Cartoon-style stretch and squash deformation.

**Parameters:**
- `stretch_x`: Horizontal stretch factor (0.5 = compress, 2.0 = stretch)
- `stretch_y`: Vertical stretch factor
- `duration_frames`: Animation duration
- `oscillate`: Continuously oscillate
- `preserve_volume`: Maintain area when stretching

#### 10. **Warp** (`warp.py`)
Flexible distortion that warps the element during movement.

**Parameters:**
- `warp_type`: 'sine', 'spiral', 'ripple', 'twist', 'bulge'
- `warp_strength`: Distortion intensity (0.1 to 2.0)
- `warp_frequency`: Wave frequency for periodic warps
- `warp_speed`: Animation speed
- `warp_center`: Center point for warp effect

#### 11. **Wave** (`wave.py`)
Wave effect that runs through the element.

**Parameters:**
- `wave_type`: 'horizontal', 'vertical', 'radial', 'flag'
- `amplitude`: Wave height in pixels
- `frequency`: Number of wave cycles
- `speed`: Wave propagation speed
- `damping`: Wave decay over distance

### 📝 Text Dynamics

#### 12. **Typewriter** (`typewriter.py`)
Text appears character by character with typewriter effect.

**Parameters:**
- `text`: Text to animate
- `chars_per_second`: Typing speed
- `show_cursor`: Display blinking cursor
- `cursor_style`: Cursor appearance ('|', '_', '█')
- `font_size`: Text size
- `font_color`: Text color in hex
- `random_timing`: Add realistic typing variations

#### 13. **Word Build-up** (`word_buildup.py`)
Words appear sequentially to build complete text.

**Parameters:**
- `text`: Text to animate
- `buildup_mode`: 'fade', 'slide', 'pop', 'typewriter'
- `word_delay`: Frames between words
- `entrance_direction`: Direction for slide mode
- `emphasis_effect`: Highlight new words
- `hold_duration`: Frames to hold complete text

#### 14. **Split Text** (`split_text.py`)
Text splits apart into pieces that move in different directions.

**Parameters:**
- `text`: Text to animate
- `split_mode`: 'character', 'word', 'line', 'half'
- `split_direction`: 'horizontal', 'vertical', 'explode', 'random'
- `split_timing`: 'simultaneous', 'sequential', 'cascade'
- `split_distance`: Distance pieces travel
- `rotation_on_split`: Rotate pieces as they split
- `fade_on_split`: Fade pieces as they move
- `rejoin`: Rejoin pieces after splitting

### ✨ Special Effects

#### 15. **Glitch** (`glitch.py`)
Digital glitch and interference effects.

**Parameters:**
- `glitch_intensity`: Effect strength (0.1 to 1.0)
- `glitch_frequency`: How often glitches occur
- `glitch_duration`: Length of each glitch
- `color_shift`: RGB channel separation
- `scan_lines`: Add CRT scan lines
- `noise_amount`: Digital noise level
- `block_size`: Size of glitch blocks

#### 16. **Shatter** (`shatter.py`)
Element breaks into pieces and disperses.

**Parameters:**
- `num_pieces`: Number of shards (10 to 100)
- `shatter_point`: Impact point for shatter
- `explosion_force`: Dispersion strength
- `gravity`: Fall acceleration
- `rotation_speed`: Piece rotation rate
- `fade_pieces`: Fade shards as they fall

#### 17. **Neon Glow** (`neon_glow.py`)
Fluorescent neon light effect with glow.

**Parameters:**
- `glow_color`: Primary neon color in hex
- `glow_intensity`: Brightness level
- `pulse_rate`: Pulsing speed
- `outer_glow_size`: Outer glow radius
- `inner_glow_size`: Inner glow radius
- `flicker`: Add realistic neon flicker

#### 18. **Lens Flare** (`lens_flare.py`)
Camera lens flare and light streak effects.

**Parameters:**
- `flare_type`: 'anamorphic', 'circular', 'starburst', 'streaks', 'bokeh'
- `flare_position`: Light source position
- `flare_intensity`: Brightness (0.1 to 2.0)
- `flare_color`: Primary flare color
- `num_artifacts`: Number of lens artifacts
- `movement`: Animate flare movement
- `rainbow_effect`: Add chromatic aberration
- `bloom`: Add bloom/glow effect

### 🎬 Motion Effects

#### 19. **Flip** (`flip.py`)
Card flip rotation effect.

**Parameters:**
- `flip_axis`: 'horizontal' or 'vertical'
- `flip_duration`: Frames for flip
- `flip_direction`: 'forward' or 'backward'
- `perspective`: 3D perspective amount
- `double_sided`: Show both sides

#### 20. **Spin** (`spin.py`)
Continuous rotation/spinning effect.

**Parameters:**
- `spin_speed`: Degrees per frame
- `spin_axis`: 'z' (2D), 'x', or 'y' (3D-like)
- `spin_duration`: Total rotation frames
- `ease_in`: Gradual speed up
- `ease_out`: Gradual slow down
- `wobble`: Add wobble effect

#### 21. **Roll** (`roll.py`)
Rolling motion like a wheel.

**Parameters:**
- `roll_direction`: 'left' or 'right'
- `roll_distance`: Distance to roll in pixels
- `roll_speed`: Pixels per frame
- `bounce_on_land`: Add bounce when stopping
- `squash_on_impact`: Deform on ground contact

#### 22. **Slide Out** (`slide_out.py`)
Element slides out of view to off-screen edge.

**Parameters:**
- `slide_direction`: 'left', 'right', 'top', 'bottom'
- `slide_duration`: Number of frames for slide
- `easing`: Animation easing function
- `fade_while_sliding`: Fade during slide

#### 23. **Zoom Out** (`zoom_out.py`)
Element scales down from full size to a point.

**Parameters:**
- `start_scale`: Initial scale (1.0 = full size)
- `end_scale`: Final scale (0.0 = invisible)
- `zoom_duration`: Frames for zoom
- `zoom_center`: Center point for zoom
- `rotation_during_zoom`: Rotate while zooming

### 🎭 3D Effects

#### 24. **Carousel** (`carousel.py`)
Elements rotate in a 3D carousel/merry-go-round pattern.

**Parameters:**
- `num_items`: Number of carousel items (2 to 12)
- `radius`: Carousel radius in pixels
- `rotation_speed`: Degrees per frame
- `tilt_angle`: Tilt of carousel plane
- `perspective_scale`: Scale items based on depth
- `fade_back`: Fade items when in back
- `vertical_carousel`: Rotate vertically instead
- `item_rotation`: Rotate individual items

#### 25. **Depth Zoom** (`depth_zoom.py`)
Element moves along Z-axis with perspective depth effects.

**Parameters:**
- `zoom_type`: 'approach', 'recede', 'fly_through', 'dolly'
- `start_depth`: Starting Z position (-10 to 10)
- `end_depth`: Ending Z position
- `focal_length`: Camera focal length simulation
- `depth_blur`: Apply depth of field blur
- `motion_blur`: Apply motion blur
- `parallax_layers`: Number of parallax layers
- `fog_effect`: Add atmospheric fog

### 🎯 Planned Animations (Not Yet Implemented)

#### Remaining Effects
- **Handwriting**: Simulates handwriting animation
- **Character Fly-in**: Each letter enters from different direction
- **Particles**: Dissolves into particle effects
- **Fire/Smoke**: Appears with fire or smoke effects
- **3D Rotate**: Full rotation in 3D space (X/Y/Z axes)
- **Cube Flip**: Appears on rotating cube face

## Usage Pattern

All animations follow the same basic usage pattern:

```python
# 1. Import the animation class
from utils.animations.<animation_name> import <AnimationClass>

# 2. Create animation instance
animation = AnimationClass(
    element_path="path/to/element.mp4",     # Element to animate
    background_path="path/to/background.mp4", # Background video
    position=(x, y),                         # Target position
    # ... animation-specific parameters
    fps=30,                                  # Frame rate
    duration=7.0,                           # Total duration in seconds
    start_frame=0,                          # When element appears
    animation_start_frame=0                 # When animation starts
)

# 3. Render the animation
output_path = "output/animated_video.mp4"
success = animation.render(output_path)
```

## Combining Animations

Animations can be combined sequentially or in parallel:

### Sequential Combination
Apply one animation after another:

```python
# First: Fade in
fade_in = FadeIn(...)
fade_in.render("temp1.mp4")

# Then: Use output as input for next animation
bounce = Bounce(
    element_path="temp1.mp4",
    ...
)
bounce.render("final.mp4")
```

### Parallel Combination
Apply multiple effects simultaneously by creating custom animation classes that combine multiple effects in the `process_frames` method.

## Common Parameters

Most animations support these common parameters:

- `element_path`: Path to element video/image
- `background_path`: Path to background video
- `position`: (x, y) target position
- `direction`: Angle in degrees (for directional animations)
- `start_frame`: Frame when element first appears
- `animation_start_frame`: Frame when animation begins
- `path`: List of (frame, x, y) for movement paths
- `fps`: Frames per second (default 30)
- `duration`: Total duration in seconds
- `remove_background`: Remove background color
- `background_color`: Color to remove (default black)
- `background_similarity`: Threshold for color removal

## Tips for Best Results

1. **Pre-process Elements**: Ensure elements have transparent backgrounds for best compositing
2. **Match Frame Rates**: Keep consistent FPS across all videos
3. **Test Parameters**: Start with default values and adjust gradually
4. **Use Easing**: Apply easing functions for more natural motion
5. **Combine Effects**: Layer multiple animations for complex effects
6. **Optimize Performance**: Process videos at appropriate resolution

## File Structure

```
utils/animations/
├── animate.py                    # Base animation class
│
├── Entry/Exit Animations
│   ├── fade_in.py               # Fade in animation
│   ├── fade_out.py              # Fade out animation
│   ├── slide_in.py              # Slide in from edge
│   ├── slide_out.py             # Slide out to edge
│   ├── zoom_in.py               # Scale up animation
│   ├── zoom_out.py              # Scale down animation
│   ├── bounce.py                # Bounce with physics
│   ├── emergence_from_static_point.py # Pixel emergence
│   └── submerge_to_static_point.py    # Pixel submersion
│
├── Distortion Effects
│   ├── skew.py                  # Diagonal skew/tilt
│   ├── stretch_squash.py        # Stretch and squash
│   ├── warp.py                  # Flexible warping
│   └── wave.py                  # Wave distortion
│
├── Text Dynamics
│   ├── typewriter.py            # Character-by-character
│   ├── word_buildup.py          # Word-by-word appearance
│   └── split_text.py            # Text splitting apart
│
├── Special Effects
│   ├── glitch.py                # Digital glitch effect
│   ├── shatter.py               # Break into pieces
│   ├── neon_glow.py             # Neon light glow
│   └── lens_flare.py            # Camera lens flare
│
├── Motion Effects
│   ├── flip.py                  # Card flip rotation
│   ├── spin.py                  # Continuous spinning
│   └── roll.py                  # Rolling motion
│
├── 3D Effects
│   ├── carousel.py              # 3D carousel rotation
│   └── depth_zoom.py            # Z-axis movement
│
└── animations.md                # This documentation
```

## Requirements

- Python 3.7+
- FFmpeg (for video processing)
- ffprobe (for video analysis)
- Sufficient disk space for temporary frame storage

## Contributing

To add a new animation:

1. Create a new Python file in `utils/animations/`
2. Inherit from the `Animation` base class
3. Override `process_frames()` method
4. Optionally override `extract_element_frames()` for custom preprocessing
5. Document parameters and usage
6. Add entry to this documentation

## Performance Considerations

- Frame extraction creates temporary files (cleaned up after rendering)
- Complex effects may require more processing time
- Higher resolution videos need more memory and disk space
- Use appropriate `temp_dir` with sufficient space

## Troubleshooting

**Animation not appearing:**
- Check `start_frame` and `animation_start_frame` values
- Verify element has transparent background if needed
- Ensure position is within video bounds

**Poor quality:**
- Increase video resolution
- Adjust `background_similarity` for better color removal
- Check FFmpeg codec settings

**Performance issues:**
- Reduce video resolution
- Decrease animation duration
- Use simpler effects
- Ensure sufficient disk space for temp files