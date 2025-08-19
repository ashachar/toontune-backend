# Claude Assistant Instructions

## File Creation Notifications
When creating any file during inference, always mention it as the last output of your response.
Also mention which files were created in the current inference session.
Additionally, mention any output files (artifacts) that were generated as a result of running code or tests.

## Project Context
This is a ToonTune backend project for processing SVG animations with human-like drawing sequences.

## Key Requirements
- Extract actual black lines from SVG images (not skeletons)
- Create continuous drawing paths that minimize pen lifts
- Draw in human-like order (head first for human figures)
- Group nearby components for logical drawing sequences

## URL and Link Policy
- NEVER attempt to download files or access URLs directly
- When needing to download models, weights, or external resources, STOP and provide the user with a search prompt for Perplexity
- Ask the user to find the correct download link and provide it back

## Bulk Image Animation Pipeline

### Main Script: `bulk_image_to_transparent_bg_videos.py`
Located in: `utils/bulk_animate_image/bulk_image_to_transparent_bg_videos.py`

**THIS IS THE GO-TO TOOL FOR BULK IMAGE ANIMATION WITH TRANSPARENT BACKGROUNDS**

This pipeline:
1. Takes a folder of transparent character images
2. Creates a grid (e.g., 5x2) with green screen background
3. Sends to Kling AI for animation (89% cost savings)
4. Splits the result and applies FFmpeg chromakey + despill for clean transparency
5. Outputs individual WebM videos with transparent backgrounds

### Usage
```bash
python utils/bulk_animate_image/bulk_image_to_transparent_bg_videos.py <input_folder> --grid 5x2
```

### Key Features
- **FFmpeg Despill**: Removes green edges using `chromakey=green:0.10:0.08` + `despill=type=green:mix=0.6`
- **Smart Prompting**: Uses BLIP-2 captioning for dynamic prompts
- **Cost Tracking**: Tracks Replicate API costs
- **Grid Optimization**: 5x2 default for tall character images
- **Natural Animations**: Characters wave, show off objects, continuous movement

## Image Contour Drawing Method

### Overview
Our system converts cartoon images into natural drawing animations that simulate how a human would draw the image, prioritizing the head/face first, then body parts.

### Core Algorithm: Euler Path Traversal
Located in: `utils/draw-euler/stroke_traversal_closed.py`

The algorithm uses graph theory to create continuous drawing paths:

1. **Line Extraction**: Extract black lines from images using edge detection and thinning
2. **Component Detection**: Find connected components (separate strokes) in the image
3. **Head Detection**: Use SAM2 (Segment Anything 2) to detect and create solid masks for head regions
4. **Component Classification**: Split components into:
   - Face outline (head boundary)
   - Face interior (eyes, mouth, nose)
   - Body (everything else)
5. **Euler Path Generation**: For each component, create a graph and find:
   - Eulerian circuit (if all vertices have even degree) - draws without lifting pen
   - Eulerian path (if exactly 2 vertices have odd degree) - minimal pen lifts
   - Greedy DFS traversal (fallback for complex graphs)
6. **Drawing Order**: Head outline → Head interior → Body (top to bottom)

### Key Features
- **All Disconnected Components**: Draws ALL parts of the image, even if disconnected
- **Solid Head Masks**: Uses aggressive hole-filling to create solid masks (no gaps in face)
- **Human-like Order**: Prioritizes head/face first, mimicking natural drawing behavior
- **Continuous Strokes**: Minimizes pen lifts using Euler path algorithms
- **H.264 Video Output**: Generates compatible MP4 videos with proper encoding

### File Structure
```
utils/draw-euler/
├── stroke_traversal_closed.py    # Main drawing algorithm
├── sam2_head_detector.py         # SAM2-based head detection module
└── test_output/                  # Generated videos and debug images

cartoon-test/
├── man_head_mask_solid.png       # Pre-generated SAM2 head mask
└── woman_head_mask_solid.png      # Pre-generated SAM2 head mask
```

### Usage
```bash
python utils/draw-euler/stroke_traversal_closed.py <image.png>
```

Output: `test_output/<image_name>_debug/drawing_animation.mp4`

## Video Segmentation and Annotation Pipeline

### Main Script: `video_segmentation_and_annotation.py`
Located in: `utils/video_segmentation/video_segmentation_and_annotation.py`

**Automatic video segmentation with AI-generated labels using SAM2 + Gemini**

This pipeline:
1. Extracts first frame from video
2. Uses SAM2 to automatically segment the frame (no manual clicks)
3. Creates a two-asset image (original + segment mask) to avoid color confusion
4. Sends to Gemini to get concise 2-4 word descriptions for each segment
5. Tracks segments through entire video using SAM2 video model
6. Outputs labeled video with segment overlays and text annotations

### Usage
```bash
python utils/video_segmentation/video_segmentation_and_annotation.py <input_video> [output_dir]
```

### Key Features
- **Automatic Segmentation**: No manual clicks required - uses SAM2's automatic mode
- **Two-Asset Protocol**: Prevents AI from describing artificial segment colors
- **Concise Labels**: Generates 2-4 word descriptions (e.g., "ocean water", "palm trees")
- **Multi-Segment Tracking**: Tracks up to 10 largest segments throughout video
- **H.264 Output**: Generates web-compatible MP4 videos

### Requirements
- `REPLICATE_API_TOKEN` in .env for SAM2 models
- `GEMINI_API_KEY` in .env for Gemini descriptions
- FFmpeg installed for video conversion

### Output Files
- `concatenated_input.png`: Two-asset image sent to Gemini
- `labeled_frame.png`: First frame with all segments labeled
- `final_video_h264.mp4`: Final video with tracked segments and labels

## Media Downsampling Utilities

### **IMPORTANT: Always use these utilities for downsampling videos and images**

Located in: `utils/downsample/`

These utilities provide efficient downsampling to reduce file sizes and processing time while maintaining quality.

### Video Downsampling: `video_downsample.py`

**Default preset: `small` (256x256)**

```bash
# Standard usage with default "small" preset
python utils/downsample/video_downsample.py input_video.mp4

# Specific presets
python utils/downsample/video_downsample.py video.mp4 --preset small  # 256x256 (default)
python utils/downsample/video_downsample.py video.mp4 --preset tiny   # 64x64 (extreme)
python utils/downsample/video_downsample.py video.mp4 --preset mini   # 128x128

# Custom resolution with FPS reduction
python utils/downsample/video_downsample.py video.mp4 --size 320 240 --fps 15
```

**Video Presets:**
- `micro`: 32x32 pixels (absolute minimum)
- `tiny`: 64x64 pixels (99% size reduction)
- `mini`: 128x128 pixels (93% reduction)
- **`small`: 256x256 pixels (DEFAULT, 82% reduction)**
- `medium`: 512x512 pixels (70% reduction)

### Image Downsampling: `image_downsample.py`

**Default preset: `small` (128x128 for images)**

```bash
# Standard usage with default "small" preset
python utils/downsample/image_downsample.py input_image.png

# Specific presets
python utils/downsample/image_downsample.py image.png --preset small  # 128x128 (default)
python utils/downsample/image_downsample.py image.png --preset mini   # 64x64
python utils/downsample/image_downsample.py image.png --preset icon   # 8x8 (extreme)

# With effects
python utils/downsample/image_downsample.py image.png --pixel-art     # Pixel art effect
python utils/downsample/image_downsample.py image.png --colors 16     # Reduce to 16 colors
python utils/downsample/image_downsample.py image.png --retro         # Retro game style

# Batch processing
python utils/downsample/image_downsample.py *.jpg --preset small
```

**Image Presets:**
- `icon`: 8x8 pixels (absolute minimum)
- `micro`: 16x16 pixels
- `tiny`: 32x32 pixels
- `mini`: 64x64 pixels
- **`small`: 128x128 pixels (DEFAULT for images)**
- `medium`: 256x256 pixels
- `large`: 512x512 pixels

### Key Benefits:
- **Preserves full video/image duration and content**
- **Maintains aspect ratio by default**
- **Extreme size reduction**: Up to 99% for videos, 99.9% for images
- **Configurable quality settings**
- **Batch processing support**
- **Special effects**: pixel art, retro filters, color reduction

### Example Results:
- 70-second video: 62MB → 2.9MB (small preset, 95% reduction)
- 70-second video: 62MB → 0.56MB (tiny preset, 99% reduction)
- 1024x1536 image: 163KB → 4.5KB (small preset, 97% reduction)

## ⚠️ CRITICAL: OpenCV vs FFmpeg Coordinate System Discrepancy ⚠️

### The Problem - NEVER FORGET THIS
OpenCV and FFmpeg use **DIFFERENT** coordinate systems for text placement:
- **OpenCV `cv2.putText()`**: Uses **BOTTOM-LEFT** as the text origin point
- **FFmpeg `drawtext` filter**: Uses **TOP-LEFT** as the text origin point

### Why This Matters
This creates a **vertical offset equal to the text height** (typically 20-30 pixels).
If you check text placement using OpenCV conventions but render with FFmpeg, the text will appear **lower than expected by the text height amount**, potentially placing text on foreground objects when you thought it was safe!

### Example of the Bug
```python
# WRONG - Mixing coordinate systems
text_height = 22
y_position = 150
# OpenCV check: text spans from y=128 to y=150 (bottom-left at 150)
# FFmpeg render: text spans from y=150 to y=172 (top-left at 150)
# Result: 22 pixel error! Text appears on foreground instead of background!

# CORRECT - Using consistent FFmpeg convention
y_position = 150  # This is the TOP of the text in FFmpeg
# Text will span from y=150 to y=172
# Check the same region: background_mask[150:172, x:x+width]
```

### Rules for Text Placement Code
1. **ALWAYS specify** which coordinate system you're using in comments
2. **For FFmpeg rendering** (most common in this project):
   - `y` coordinate = TOP edge of text
   - Text bounding box: `(x, y)` to `(x + width, y + height)`
3. **For checking if text fits**: Use same convention as renderer
4. **NEVER mix** OpenCV checking with FFmpeg rendering without adjustment

### Quick Reference
```python
# FFmpeg drawtext (TOP-LEFT origin) - USE THIS FOR VIDEO OVERLAYS
x = 100  # Left edge  
y = 50   # TOP edge of text
# Text occupies: x to x+width, y to y+height

# OpenCV putText (BOTTOM-LEFT origin) - ONLY for OpenCV rendering
x = 100  # Left edge
y = 50   # BOTTOM edge of text  
# Text occupies: x to x+width, y-height to y
```

### Common Mistake Pattern
```python
# WRONG - This will place text too low!
def check_position_opencv_style(x, y, text_height):
    # Checking y-text_height to y (OpenCV style)
    return background_mask[y-text_height:y, x:x+width]

# Then using with FFmpeg will place text from y to y+text_height
# Result: Text appears text_height pixels lower than checked!

# CORRECT - Match FFmpeg's coordinate system
def check_position_ffmpeg_style(x, y, text_height):
    # Checking y to y+text_height (FFmpeg style)
    return background_mask[y:y+text_height, x:x+width]
```

## Unified Video Pipeline Usage

**IMPORTANT: Always use the unified_video_pipeline.py with appropriate flags to avoid redundant processing**

### Main Script: `unified_video_pipeline.py`
The central pipeline that orchestrates all video processing steps.

### Usage with Flags
```bash
# Full processing (all steps)
python unified_video_pipeline.py video.mp4

# Skip only LLM inference (most common for testing)
python unified_video_pipeline.py video.mp4 --dry-run

# Skip specific steps to save time
python unified_video_pipeline.py video.mp4 --no-transcript  # Use existing transcript
python unified_video_pipeline.py video.mp4 --no-downsample  # Skip downsampling
python unified_video_pipeline.py video.mp4 --no-scenes     # Keep existing scenes
python unified_video_pipeline.py video.mp4 --no-prompts    # Skip prompt generation
python unified_video_pipeline.py video.mp4 --no-inference  # Skip LLM (same as --dry-run)
python unified_video_pipeline.py video.mp4 --no-editing    # Skip video editing

# Combine flags for specific workflows
python unified_video_pipeline.py video.mp4 --no-transcript --no-downsample --dry-run

# Adjust scene duration (default: 60 seconds)
python unified_video_pipeline.py video.mp4 --target-scene-duration 45
```

### Pipeline Flags Reference
- `--dry-run`: Skip LLM inference only (still does transcript, scenes, etc.)
- `--no-downsample`: Skip video downsampling step
- `--no-transcript`: Skip transcript generation (uses existing or mock)
- `--no-scenes`: Skip scene splitting
- `--no-prompts`: Skip prompt generation
- `--no-inference`: Skip LLM inference (same as --dry-run)
- `--no-editing`: Skip video editing
- `--target-scene-duration N`: Set target scene duration in seconds (default: 60)

### Scene Splitting Logic
- Scenes are split at sentence boundaries
- Target duration: ~60 seconds per scene (configurable)
- Algorithm respects sentence endings for natural breaks
- Tolerance: 90-120% of target duration

### When to Use Flags
- **Testing changes**: Use `--dry-run` to skip expensive LLM calls
- **Re-running with existing transcript**: Use `--no-transcript --no-downsample`
- **Adjusting scenes only**: Use `--no-transcript --no-downsample --no-prompts --no-inference --no-editing`
- **Testing editing only**: Use `--no-transcript --no-downsample --no-scenes --no-prompts --dry-run`