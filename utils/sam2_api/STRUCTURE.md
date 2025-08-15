# SAM2 API Module Structure

## ✅ Properly Organized Structure

The SAM2 video segmentation module is now properly organized as a subfolder within `backend/utils/`:

```
backend/
├── utils/
│   ├── contour_extraction/     # Original - Image contour extraction
│   ├── draw-euler/             # Original - Euler path drawing
│   ├── end_to_end_drawing/     # Original - Complete drawing pipeline
│   ├── image_generation/       # Original - Image generation utilities
│   ├── vectorize/              # Original - SVG vectorization
│   ├── sam2_api/               # NEW - SAM2 video segmentation
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── video_segmentation.py
│   │   └── sam2_continuous_segmentation.py
│   ├── image_creation.py       # Original
│   └── svg_to_png.py          # Original
├── docs/
│   └── SAM2_CONTINUOUS_SEGMENTATION_GUIDE.md
└── [test scripts]
    ├── test_sam2_video.py
    ├── segment_sea_*.py
    └── run_*.py
```

## Module Import Path

```python
# From within backend/ or test scripts:
from backend.utils.sam2_api import segment_video_simple

# Or specific imports:
from backend.utils.sam2_api import (
    ContinuousSAM2Segmenter,
    ObjectToTrack,
    SAM2VideoSegmenter
)
```

## Key Files

### Core Implementation
- `video_segmentation.py` - Base SAM2 wrapper for Replicate API
- `sam2_continuous_segmentation.py` - Production implementation with dynamic frame detection

### Module Files
- `__init__.py` - Exports main classes and functions
- `README.md` - Usage documentation
- `STRUCTURE.md` - This file, describing organization

## Design Principles

1. **Subfolder Organization** - Keeps utils/ clean and organized by functionality
2. **Module Structure** - Proper Python module with __init__.py
3. **Clear Separation** - SAM2 API code separate from other utilities
4. **Easy Imports** - Clean import paths via __init__.py exports
5. **Documentation** - README and guides included in appropriate locations

## Test Scripts Location

All test and demo scripts remain in `backend/` root for easy access:
- `test_sam2_video.py`
- `segment_sea_every_frame.py` (production solution)
- Various other test variants

This keeps the utils folder clean while maintaining easy access to test scripts.