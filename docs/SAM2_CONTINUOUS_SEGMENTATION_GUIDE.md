# SAM2 Continuous Video Segmentation Guide

## Overview

This guide documents the **production-ready approach** for achieving continuous, frame-by-frame video segmentation using Meta's SAM2 model via the Replicate API.

## Key Finding

⚠️ **Important**: The Replicate API implementation of SAM2 **does not automatically interpolate masks between keyframes**. Masks only appear at frames where explicit click points are defined. This differs from the original SAM2 paper's described behavior.

## Solution: Every-Frame Tracking

To achieve continuous segmentation throughout a video, we must place tracking points on **every frame** where we want masks to appear.

## Implementation

### Dynamic Frame Detection

Our implementation **dynamically detects video properties** rather than hard-coding frame counts:

```python
from utils.sam2_continuous_segmentation import ContinuousSAM2Segmenter, ObjectToTrack

# Initialize segmenter
segmenter = ContinuousSAM2Segmenter()

# Define objects to track (positions are dynamically validated)
objects = [
    ObjectToTrack("person", x=320, y=240, track_every_n_frames=1),  # Every frame
    ObjectToTrack("car", x=150, y=300, track_every_n_frames=2),     # Every 2 frames
]

# Segment video - frame count is detected automatically
result = segmenter.segment_video_continuous(
    "input_video.mp4",
    objects,
    output_path="output_continuous.mp4"
)
```

### How It Works

1. **Video Analysis**: Uses `ffprobe` to extract:
   - Total frame count
   - Frame rate (fps)
   - Video dimensions
   - Duration

2. **Dynamic Point Generation**: Creates click points based on actual video properties:
   ```python
   for frame in range(0, video_info.frame_count, track_every_n_frames):
       # Generate point for this frame
   ```

3. **Continuous Output**: Ensures masks appear on every desired frame without gaps

## Usage Examples

### Simple Usage

```python
from utils.sam2_continuous_segmentation import segment_video_simple

# Simplest approach - track objects on every frame
result = segment_video_simple(
    "my_video.mp4",
    [
        ("palm_tree", 320, 50),
        ("ocean", 400, 180),
        ("rocks", 200, 230)
    ]
)
```

### Advanced Usage with Control

```python
from utils.sam2_continuous_segmentation import ContinuousSAM2Segmenter, ObjectToTrack

segmenter = ContinuousSAM2Segmenter()

# Get video info first
video_info = segmenter.get_video_info("video.mp4")
print(f"Video has {video_info.frame_count} frames at {video_info.fps} fps")

# Define objects with different tracking densities
objects = [
    # Critical object - track every frame
    ObjectToTrack("main_subject", x=320, y=240, track_every_n_frames=1),
    
    # Secondary object - track every 3 frames (still smooth at 60fps)
    ObjectToTrack("background_object", x=100, y=100, track_every_n_frames=3),
    
    # Static object - track every 10 frames
    ObjectToTrack("stationary_item", x=500, y=300, track_every_n_frames=10)
]

# Run segmentation
result = segmenter.segment_video_continuous(
    "video.mp4",
    objects,
    mask_type="highlighted",  # or "binary", "greenscreen"
    output_quality=85
)
```

## Performance Considerations

### Tracking Density vs. API Cost

| Strategy | Frames/sec @ 60fps | Smoothness | API Cost | Use Case |
|----------|-------------------|------------|----------|----------|
| Every frame | 60 points/sec | Perfect | Highest | Critical tracking |
| Every 2 frames | 30 points/sec | Excellent | High | Most videos |
| Every 5 frames | 12 points/sec | Good | Medium | Slow-moving objects |
| Every 10 frames | 6 points/sec | Choppy | Low | Static objects |

### Optimization Tips

1. **Mixed Density**: Use different `track_every_n_frames` values for different objects:
   ```python
   objects = [
       ObjectToTrack("fast_moving", x=320, y=240, track_every_n_frames=1),
       ObjectToTrack("slow_moving", x=100, y=100, track_every_n_frames=3),
       ObjectToTrack("static", x=500, y=400, track_every_n_frames=10)
   ]
   ```

2. **Frame Rate Consideration**: For high fps videos (60+), you might track every 2-3 frames without noticeable gaps

3. **Video Length**: For long videos, consider processing in segments to manage API costs

## API Limitations

### Current Replicate Limitations

1. **No Temporal Propagation**: Unlike the original SAM2, Replicate doesn't propagate masks between keyframes
2. **Frame-by-Frame Processing**: Each frame with a mask needs an explicit click point
3. **No Memory Bank Utilization**: The temporal consistency features described in the paper aren't accessible

### Workarounds

- **This implementation**: Places points on every frame for guaranteed continuous masks
- **Alternative**: Use the official SAM2 implementation for true temporal propagation
- **Hybrid approach**: Dense points for critical objects, sparse for background

## Best Practices

1. **Always detect video properties dynamically** - Don't hard-code frame counts
2. **Validate object positions** - Ensure x,y coordinates are within video bounds
3. **Start with fewer objects** - Test with 2-3 objects before adding more
4. **Monitor API usage** - Each click point counts toward API limits
5. **Use appropriate mask types**:
   - `highlighted`: Best for visualization
   - `binary`: For post-processing
   - `greenscreen`: For compositing

## Common Issues and Solutions

### Issue: Masks still appear intermittent
**Solution**: Ensure `track_every_n_frames=1` for continuous coverage

### Issue: API timeout or rate limits
**Solution**: Reduce number of objects or increase `track_every_n_frames`

### Issue: Wrong object positions
**Solution**: Video coordinates are (0,0) at top-left, validate with video dimensions

### Issue: Masks on wrong objects
**Solution**: Ensure unique `object_id` for each distinct object

## Example: Processing a 5-second video

```python
# For a 5-second video at 30fps = 150 frames

objects = [
    # Main subject - perfect tracking (150 points)
    ObjectToTrack("person", x=320, y=240, track_every_n_frames=1),
    
    # Secondary elements - good tracking (50 points each)
    ObjectToTrack("car", x=150, y=300, track_every_n_frames=3),
    ObjectToTrack("tree", x=500, y=200, track_every_n_frames=3),
]
# Total: 250 click points for smooth, continuous segmentation
```

## Conclusion

While the Replicate API implementation of SAM2 doesn't provide automatic temporal propagation, our solution ensures **continuous, gap-free segmentation** by dynamically placing tracking points based on actual video properties. This approach guarantees that masks appear smoothly throughout the entire video duration.

For production use, the `ContinuousSAM2Segmenter` class provides a robust, flexible solution that adapts to any video's specifications automatically.