# Claude Agents

This directory contains specialized Claude agents for handling common tasks with guaranteed success.

## Video Encoder Agent

**Purpose**: Ensure all videos are properly encoded for QuickTime and universal compatibility.

### Features
- ✅ Automatic QuickTime compatibility
- ✅ Fallback encoding for problematic videos
- ✅ Batch processing support
- ✅ Quality presets (standard/high)
- ✅ Transparency preservation (when possible)
- ✅ Audio handling
- ✅ Output verification

### Usage

#### Single Video
```bash
# Basic encoding
python .claude/agents/video_encoder.py input.mp4

# Specify output
python .claude/agents/video_encoder.py input.mp4 -o output_quicktime.mp4

# High quality encoding
python .claude/agents/video_encoder.py input.mp4 --quality high

# Preserve transparency (if present)
python .claude/agents/video_encoder.py input.mp4 --preserve-transparency
```

#### Batch Processing
```bash
# Encode all MP4s in a directory
python .claude/agents/video_encoder.py --batch /path/to/videos/

# Specific pattern
python .claude/agents/video_encoder.py --batch /path/to/videos/ --pattern "*.avi"

# Output to different directory
python .claude/agents/video_encoder.py --batch input_dir/ -o output_dir/
```

#### Using the Shell Wrapper
```bash
# Simple usage
./.claude/agents/encode_video.sh input.mp4

# With output path
./.claude/agents/encode_video.sh input.mp4 output.mp4
```

### Encoding Settings

#### Standard Quality (Default)
- Codec: H.264 (libx264)
- Pixel Format: yuv420p (maximum compatibility)
- Profile: baseline (works everywhere)
- CRF: 23 (good quality/size balance)
- Preset: medium (balanced speed)

#### High Quality
- Codec: H.264 (libx264)
- Pixel Format: yuv420p
- Profile: high (better compression)
- CRF: 18 (higher quality)
- Preset: slow (better compression)

### Automatic Fixes
The agent automatically handles:
- Odd dimensions (makes them even)
- Missing audio streams
- Incompatible pixel formats
- Streaming optimization (faststart)
- Profile/level compatibility

### When to Use This Agent
- ❌ Video won't open in QuickTime
- ❌ "Could not be opened" errors
- ❌ Compatibility issues across devices
- ✅ Need guaranteed playback
- ✅ Batch converting videos
- ✅ Preparing videos for distribution

### Integration with Animation Pipeline

To integrate with the 3D text animation pipeline, modify the final encoding step:

```python
# Instead of direct FFmpeg call
# subprocess.run(["ffmpeg", ...])

# Use the video encoder agent
from claude.agents.video_encoder import VideoEncoder

encoder = VideoEncoder()
final_video = encoder.encode_for_quicktime(
    "temp_output.mp4",
    "final_output.mp4",
    quality="high"
)
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| FFmpeg not found | Install FFmpeg: `brew install ffmpeg` |
| Still won't open | Agent will auto-fallback to maximum compatibility |
| Transparency lost | Use `--preserve-transparency` flag |
| Audio issues | Agent auto-detects and handles audio |

## Future Agents

Planned agents for common tasks:
- `image_optimizer.py` - Optimize images for web/processing
- `transcript_generator.py` - Generate transcripts from videos
- `scene_splitter.py` - Intelligent scene detection and splitting
- `mask_extractor.py` - Robust foreground/background extraction