# SAM2 Video Segmentation API

This module provides continuous video segmentation using Meta's SAM2 model via Replicate API.

## Files

- `video_segmentation.py` - Core SAM2 wrapper for the Replicate API
- `sam2_continuous_segmentation.py` - Production implementation with automatic frame detection and continuous tracking
- `__init__.py` - Module exports

## Usage

```python
from backend.utils.sam2_api import segment_video_simple

# Simple usage - automatically detects video properties
result = segment_video_simple(
    "path/to/video.mp4",
    [("object1", x1, y1), ("object2", x2, y2)]
)
```

## Advanced Usage

```python
from backend.utils.sam2_api import ContinuousSAM2Segmenter, ObjectToTrack

segmenter = ContinuousSAM2Segmenter()

objects = [
    ObjectToTrack("person", x=320, y=240, track_every_n_frames=1),
    ObjectToTrack("car", x=150, y=300, track_every_n_frames=2)
]

result = segmenter.segment_video_continuous(
    "video.mp4",
    objects,
    output_path="segmented.mp4"
)
```

## Key Features

- **Dynamic frame detection** - Automatically detects video properties (fps, frame count)
- **Continuous tracking** - Ensures masks appear on every frame without gaps
- **Configurable density** - Control tracking frequency per object
- **Production ready** - Robust error handling and logging

## Important Note

The Replicate API implementation requires explicit click points at every frame where masks should appear. This module handles that automatically by placing tracking points based on the video's actual frame count.