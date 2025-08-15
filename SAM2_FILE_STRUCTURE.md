# SAM2 Video Segmentation - File Structure

All SAM2-related files have been organized under the `backend/` directory:

## Main Implementation
- `backend/utils/video_segmentation.py` - Core SAM2 wrapper for Replicate API
- `backend/utils/sam2_continuous_segmentation.py` - Production-ready continuous segmentation with dynamic frame detection

## Test Scripts
- `backend/test_sam2_video.py` - Initial test implementation
- `backend/run_sam2_segmentation.py` - Basic segmentation runner
- `backend/run_sea_segmentation.py` - Sea video segmentation with .env support
- `backend/segment_sea_video.py` - Interactive segmentation script
- `backend/segment_sea_fixed.py` - Fixed frame distribution version
- `backend/segment_sea_continuous.py` - Dense tracking implementation
- `backend/segment_sea_every_frame.py` - Every-frame tracking (final solution)
- `backend/segment_sea_demo.py` - Demo/documentation script
- `backend/check_sam2_model.py` - Model verification utility

## Documentation
- `backend/docs/SAM2_CONTINUOUS_SEGMENTATION_GUIDE.md` - Complete implementation guide
- `backend/docs/SAM2_ISSUE_DESCRIPTION.md` - Problem analysis document (if moved)

## Video Artifacts
Located in `backend/uploads/assets/videos/` (or `../swifit/backend/uploads/assets/videos/`):
- `sea_small.mov` - Original test video
- `sea_small_segmented.mp4` - First attempt (sparse points)
- `sea_small_segmented_fixed.mp4` - Fixed distribution
- `sea_small_continuous.mp4` - Dense tracking (71 points)
- `sea_small_every_frame.mp4` - Final continuous version (every frame)
- `frames_*/` - Extracted frames for analysis

## Usage

### For Production Use:
```python
from backend.utils.sam2_continuous_segmentation import segment_video_simple

result = segment_video_simple(
    "video.mp4",
    [("object1", x1, y1), ("object2", x2, y2)]
)
```

### For Testing:
```bash
cd backend
python segment_sea_every_frame.py
```

## Key Finding
The Replicate API implementation requires explicit click points at every frame for continuous segmentation. The production implementation in `sam2_continuous_segmentation.py` handles this automatically by detecting video properties and generating appropriate tracking points.