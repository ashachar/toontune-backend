# Letter 3D Burn Animation

A dramatic text animation effect where letters appear to burn and turn into smoke/gas particles.

## Features

### 🔥 Burn Effects
- **Edge Burning**: Letters burn from edges inward
- **Fire Particles**: Orange/red fire particles at burning edges  
- **Smoke Rising**: Gray smoke particles that rise and disperse
- **Charring Effect**: Letters darken and char as they burn
- **Staggered Timing**: Each letter burns with configurable delay

### 🎨 Visual Components
1. **Ignition Phase**: Edges start glowing orange/red
2. **Burn Phase**: Fire consumes letter from edges, particles emit
3. **Smoke Phase**: Remaining fragments turn to rising smoke
4. **Particle Physics**: Realistic particle movement with turbulence

## Usage

```python
from utils.animations.letter_3d_burn import Letter3DBurn

# Create burn animation
burn = Letter3DBurn(
    duration=3.0,
    fps=30,
    resolution=(1280, 720),
    text="BURN EFFECT",
    font_size=120,
    text_color=(255, 220, 0),    # Yellow text
    burn_color=(255, 50, 0),      # Orange burn
    burn_duration=0.8,            # How long each letter burns
    burn_stagger=0.15,            # Delay between letters
    smoke_rise_distance=150,      # How high smoke rises
    reverse_order=False,          # Burn direction
    supersample_factor=4          # Quality
)

# Generate frame
frame = burn.generate_frame(frame_num, background)
```

## Parameters

### Core Parameters
- `text`: The text to animate
- `font_size`: Size of the text
- `text_color`: Base color of text (RGB)
- `burn_color`: Color of burning edges (RGB)
- `duration`: Total animation duration in seconds
- `fps`: Frames per second

### Timing Controls
- `stable_duration`: Time before burning starts (default: 0.1s)
- `ignite_duration`: Time for ignition phase (default: 0.2s)
- `burn_duration`: Main burning phase duration (default: 0.8s)
- `smoke_duration`: Smoke rising duration (default: 0.5s)
- `burn_stagger`: Delay between letters starting to burn (default: 0.1s)

### Animation Options
- `reverse_order`: Burn from last to first letter
- `random_order`: Random burn order
- `smoke_rise_distance`: How high smoke particles rise
- `supersample_factor`: Rendering quality (1-8)

### Particle System
- `max_particles`: Maximum particles per letter (default: 50)
- `rise_speed`: Speed of smoke rising (default: 2.0)
- `spread_rate`: Horizontal spread of particles (default: 0.5)
- `turbulence`: Random motion intensity (default: 0.3)

## Animation Phases

### 1. Waiting
Letter is displayed normally, waiting for its turn to burn.

### 2. Stable
Brief stable display before ignition starts.

### 3. Ignite
Edges begin to glow with fire colors, preparing to burn.

### 4. Burn
Main burning phase:
- Edges erode inward
- Fire particles emit from burning edges
- Letter gradually becomes transparent
- Charring effect applied

### 5. Smoke
Final phase:
- Remaining fragments turn to smoke
- Particles rise and disperse
- Letter fades completely

### 6. Gone
Letter is completely burned away, only lingering smoke remains.

## Particle Effects

The particle system creates realistic fire and smoke:

### Fire Particles
- Small, bright orange/red particles
- Rise quickly with acceleration
- Transition to smoke as they cool
- Additive blending for glow effect

### Smoke Particles  
- Larger, gray particles
- Rise slowly with air resistance
- Expand as they rise
- Alpha blending for transparency

## Comparison with Dissolve

| Feature | Burn | Dissolve |
|---------|------|----------|
| Direction | Edges inward | Random/ordered |
| Particles | Fire & smoke | Floating fragments |
| Color change | Charring/darkening | Maintains color |
| Motion | Rising smoke | Floating upward |
| Speed | Gradual erosion | Quick dispersal |
| Visual style | Dramatic/destructive | Magical/ethereal |

## Integration Example

```python
# With motion animation handoff
from utils.animations.text_3d_motion import Text3DMotion
from utils.animations.letter_3d_burn import Letter3DBurn

# Motion phase
motion = Text3DMotion(...)
frame = motion.generate_frame(frame_num, background)
final_state = motion.get_final_state()

# Handoff to burn
burn = Letter3DBurn(...)
burn.set_initial_state(
    scale=final_state.scale,
    position=final_state.center_position,
    alpha=final_state.alpha,
    letter_sprites=final_state.letter_sprites
)

# Continue with burn
frame = burn.generate_frame(frame_num, background)
```

## Files in this Module

- `burn.py` - Main burn animation class
- `timing.py` - Timing control for burn phases
- `particles.py` - Particle system for fire/smoke effects
- `__init__.py` - Module exports
- `README.md` - This file

## Future Enhancements

- [ ] Wind effects on smoke direction
- [ ] Ember particles that fall down
- [ ] Variable burn patterns (top-down, center-out)
- [ ] Heat distortion effects
- [ ] Ash residue left behind
- [ ] Color temperature variations
- [ ] Sparks and crackle effects