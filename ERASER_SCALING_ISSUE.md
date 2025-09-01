# Eraser Scaling Issue - Complete Context

## Problem Statement

We have an eraser animation effect where an eraser image (hand holding an eraser) moves in an elliptical path across the video frame. The requirement is that **the bottom of the eraser image should NEVER be visible** - it should always extend past the bottom edge of the video frame to maintain the illusion that someone is holding the eraser from outside the frame.

**Current Issue**: Despite multiple attempts to scale the eraser image to be taller, the hand appears "detached" at certain frames, with green background visible beneath the hand/arm. This breaks the illusion completely.

## Visual Evidence

- At frame 88 (timestamp 3520ms): Hand appears detached with green beneath
- At frame 89 (timestamp 3560ms): Hand still detached
- At frame 90 (timestamp 3600ms): Hand detached even after attempted fixes

## Technical Details

### Video Specifications
- Frame dimensions: 640x360 pixels
- Frame rate: ~25-30 fps
- Duration of eraser effect: 0.6 seconds

### Eraser Image Specifications
- Original eraser.png dimensions: **768x1344 pixels** (already very tall!)
- Location: `/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/images/eraser.png`
- Content: A hand holding a blue eraser, with arm extending down

### Motion Path
The eraser moves in an elliptical path:
- Center point: (320, 180) - center of the 640x360 frame
- Horizontal radius: 200 pixels
- Vertical radius: 150 pixels (multiplied by 0.8 = 120 pixels effective)
- Highest Y position: 180 - 120 = 60 pixels from top
- Lowest Y position: 180 + 120 = 300 pixels from top

## Current Implementation

### File: `/utils/animations/masked_eraser_wipe.py`

```python
def create_masked_eraser_wipe(character_video: str, original_video: str, 
                              eraser_image: str, output_video: str,
                              wipe_start: float = 0, wipe_duration: float = 0.6):
    """
    Create eraser wipe using geometric masks instead of chromakey.
    This ensures the character remains fully opaque.
    """
    
    print(f"Creating masked eraser wipe (no chromakey)...")
    
    # Get video properties
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'csv=s=x:p=0',
        character_video
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    parts = result.stdout.strip().split('x')
    width = int(parts[0])  # 640
    height = int(parts[1])  # 360
    
    # Define eraser path
    center_x = width // 2  # 320
    center_y = height // 2  # 180
    radius_x = 200
    radius_y = 150
    
    # Eraser positioning calculations
    # The eraser moves in an elliptical path
    # At highest point: y = center_y - radius_y * 0.8 = 180 - 120 = 60
    # At lowest point: y = center_y + radius_y * 0.8 = 180 + 120 = 300
    
    # ... (pre-composite code omitted for brevity)
    
    filter_parts = []
    filter_parts.append(f"[0:v]null[char];")  # Character video (opaque)
    filter_parts.append(f"[1:v]null[orig];")  # Original video
    
    # Scale eraser to ensure bottom is always off-screen
    # The original eraser.png is 768x1344 - already very tall!
    # We should keep it large or make it even larger
    # Don't shrink it - scale to at least 1000px tall to ensure coverage
    eraser_height = 1000  # Keep eraser very tall
    
    # Scale eraser while maintaining aspect ratio
    # If already taller than 1000px, keep original size
    filter_parts.append(f"[2:v]scale=-2:'max(1000,ih)'[eraser];")  # Minimum 1000px tall
    
    # ... (progressive reveal code omitted for brevity)
    
    # Add the moving eraser on top
    # Shift the eraser down so more of it is visible in frame
    # Add 500 pixels offset to push it down, so the hand/arm portion is fully visible
    y_offset = 500
    eraser_motion = (f"overlay=x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-150':"
                    f"y='{center_y}+{radius_y*0.8}*sin(2*PI*(t-{wipe_start})/{wipe_duration})+{y_offset}':"
                    f"enable='between(t,{wipe_start},{wipe_start + wipe_duration})'")
    
    filter_parts.append(f"[{current}][eraser]{eraser_motion}")
    
    # Join all filter parts
    filter_complex = "".join(filter_parts)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_composite,  # Character (fully opaque)
        '-i', original_video,  # Original to reveal
        '-i', eraser_image,    # Eraser
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
```

## What We've Tried

### Attempt 1: Initial Implementation
- Used `scale=300:-1` to scale eraser to 300px width
- **Result**: Hand appeared detached, green showing beneath

### Attempt 2: Calculate Required Height
- Calculated minimum height needed: `height - highest_y_position + margin`
- Used `scale={eraser_width}:-1,scale=w=iw:h='max(ih,{int(min_eraser_height)})'`
- **Result**: Double scale filter didn't work properly, hand still detached

### Attempt 3: Fixed Height Scaling
- Changed to `scale=-2:700` for a fixed 700px height
- **Result**: Actually SHRANK the image (original is 1344px), hand still detached

### Attempt 4: Preserve Original Size
- Changed to `scale=-2:'max(1000,ih)'` to keep minimum 1000px height
- Since original is 1344px, this should preserve the full height
- **Result**: Hand STILL appears detached!

### Attempt 5: Add Y Offset
- Added 500px offset to push eraser down: `y=... + 500`
- **Result**: Helped somewhat but hand still detached in some frames

## The Mystery

**The core mystery**: The eraser.png is 1344px tall. The video frame is only 360px tall. Even at the highest position (y=60), the eraser should extend to y=60+1344=1404, which is way past the frame bottom (360). Yet the hand appears cut off with green beneath it.

## Hypotheses

1. **FFmpeg is cropping the overlay**: The overlay filter might be cropping the eraser to fit within the video frame bounds (360px)

2. **Scale filter not working**: Despite the syntax appearing correct (`scale=-2:'max(1000,ih)'`), the eraser might not actually be scaled/preserved at full height

3. **Wrong input to scale**: Perhaps the eraser image isn't being read at full resolution initially

4. **Overlay limitations**: The overlay filter might have inherent limitations on overlaying images larger than the base video

5. **Pipeline issue**: Something earlier in the pipeline might be pre-scaling or cropping the eraser before it reaches this function

## What We Need Help With

1. **Verify the scale filter is actually working**: How can we confirm the eraser is truly 1344px (or 1000px minimum) after the scale filter?

2. **Understand overlay behavior**: Does FFmpeg's overlay filter automatically crop overlaid images to the base video dimensions?

3. **Alternative approaches**: 
   - Should we use a different filter chain?
   - Should we composite the eraser differently?
   - Should we pad the video canvas to accommodate the full eraser height?

4. **Debug the actual dimensions**: How can we inspect the actual dimensions of the eraser at each stage of the filter chain?

## Expected Behavior

At frame 89 (highest point in motion):
- Eraser Y position (top edge): 60px
- Eraser height: 1344px (or at least 1000px)
- Eraser bottom edge: 60 + 1344 = 1404px
- Video frame height: 360px
- **Expected**: Bottom 1044px of eraser extends past frame, no green visible
- **Actual**: Hand appears cut off around 300-350px with green beneath

## Environment

- Python 3.x
- FFmpeg version: (needs to be checked with `ffmpeg -version`)
- OS: macOS
- Project: ToonTune backend for person-to-character animation

## Critical Code Context

The eraser animation is part of a larger pipeline that:
1. Takes a video of a person
2. Replaces them with an animated character (meerkat)
3. Uses sketch animation for entrance
4. Uses eraser wipe for exit transition
5. Returns to showing the original person

The eraser wipe is specifically the exit transition where we want to create the illusion of someone erasing the character from outside the frame.

## Question for Next LLM

**How can we ensure that an overlay image in FFmpeg that is taller than the video frame (1344px image on 360px video) is NOT cropped at the bottom, so that the full height of the image is used even though most of it extends beyond the video frame boundaries?**

Specifically:
- Is there a way to force overlay to not crop to frame bounds?
- Do we need to use a different approach entirely?
- Should we test with a simple colored rectangle first to isolate the issue?

## Files to Reference

1. Main implementation: `/utils/animations/masked_eraser_wipe.py`
2. Eraser image: `/uploads/assets/images/eraser.png` (768x1344)
3. Pipeline using this: `/pipelines/person_animation/main.py` (calls the eraser function around line 970-1050)
4. Test video location: `/outputs/final_character_video.mp4`

## Reproducibility

To reproduce the issue:
1. Run: `python pipelines/person_animation/main.py`
2. Check output at: `outputs/final_character_video.mp4`
3. Look at frames around 3.5-3.6 seconds (frames 88-90)
4. Observe the hand/arm appears cut off with green beneath

## Success Criteria

The eraser animation will be considered fixed when:
1. At ALL frames during the eraser motion (0.6 second duration)
2. The bottom of the hand/arm is NEVER visible
3. No green background appears beneath the hand
4. The hand/arm appears to extend infinitely downward out of frame
5. This maintains the illusion that someone is holding the eraser from below the video frame