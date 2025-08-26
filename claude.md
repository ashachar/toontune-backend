# Claude Assistant Instructions

## Python File Length Rule - CRITICAL
**ALL Python files MUST be 200 lines or less!**
- This is a hard requirement - no exceptions
- If a file exceeds 200 lines, refactor it into smaller modules
- Each module should have a single, focused responsibility
- Use imports to connect modules together
- This improves code maintainability, readability, and testing

## /summarize Command - CRITICAL RULE
**The Python script for /summarize must NEVER decide which files are relevant!**
- Claude MUST use Grep/Glob tools to find relevant files
- Claude MUST curate a focused list (5-20 files max) 
- The Python script (`create_issue_xml.py`) is 100% deterministic - it only formats what Claude provides
- NEVER modify the script to search for files automatically

## File Naming and Versioning Policy
**CRITICAL: NEVER add prefixes or suffixes like _fixed, _correct, _proper, _final, etc. to file names**
- Always work on the SAME file instead of creating multiple versions
- Use version control (git) for tracking changes, not file name variations
- Only create a new version of a file if the user EXPLICITLY requests it
- When fixing bugs or improving code, modify the existing file directly
- If you need to test different approaches, use temporary test files that get deleted after use
- Avoid cluttering the codebase with multiple versions of the same functionality

## Video Output Naming Convention
**CRITICAL: NEVER CREATE MULTIPLE VERSIONS OF THE SAME VIDEO**
- When creating an edited version of a video, use descriptive effect names
- Format: `input_video_<effect_name>.mp4` where effect_name describes what was done
- Examples:
  - `ai_video1.mp4` → `ai_video1_3dtext.mp4` (added 3D text)
  - `ai_video1.mp4` → `ai_video1_smooth.mp4` (applied smoothing)
  - `ai_video1.mp4` → `ai_video1_dissolve.mp4` (added dissolve effect)
- NEVER use generic suffixes like:
  - `_final`, `_fixed`, `_correct`, `_proper`, `_better`, `_updated`
  - `_v2`, `_v3`, `_new`, `_latest`
- This keeps the output directory organized and makes it clear what each video contains

## Test Video Output Location
**ALL test videos MUST be saved to the `outputs/` folder**
- When running tests or generating sample videos, always output to `outputs/`
- Create subdirectories for specific test runs if needed: `outputs/test_3dtext/`, `outputs/test_dissolve/`
- This keeps the root directory clean and organized
- Examples:
  - Testing 3D text: save to `outputs/test_3dtext.mp4`
  - Testing dissolve effect: save to `outputs/test_dissolve.mp4`
  - Multiple iterations: `outputs/iteration1/`, `outputs/iteration2/`
- NEVER save test videos directly in the backend root directory

## Animation Code Duplication Policy
**CRITICAL: NEVER duplicate animation code - always check for existing implementations first**
- Before implementing ANY animation, MUST search for existing animation classes using:
  - Grep for class names and keywords
  - Check utils/animations/ directory
  - Look for similar functionality in existing files
- If an animation seems needed but not found:
  - MUST verify with user: "I couldn't find an existing [animation type] animation. Should I create a new one?"
  - Wait for confirmation before implementing
- Animation classes should be:
  - **Modular**: One animation effect per class
  - **Composable**: Can be combined with other animations
  - **Reusable**: No hardcoded values specific to one use case
- When multiple animations are needed together:
  - Create separate classes for each effect
  - Compose them in the test/usage file
  - NEVER merge multiple effects into one monolithic class

## File Creation Notifications
When creating any file during inference, always mention it as the last output of your response.
Also mention which files were created in the current inference session.
Additionally, mention any output files (artifacts) that were generated as a result of running code or tests.

## Video Output Format Requirements
**ALL video outputs MUST be encoded in H.264 format for compatibility**
- Use `libx264` codec with FFmpeg
- Add `-pix_fmt yuv420p` for maximum compatibility
- Add `-movflags +faststart` for web streaming
- Standard conversion command:
  ```bash
  ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart output_h264.mp4
  ```

## Video Continuity Validation - CRITICAL
**IMPORTANT: ALWAYS validate video continuity after creating ANY video**
- After generating any video output, MUST run the continuity validation script
- This detects duplicate frames that indicate the video jumps back to previous content
- Command:
  ```bash
  python utils/video/validation/validate_continuity.py <output_video.mp4>
  ```
- The script will:
  - Detect exact duplicate frames (excluding uniform screens like black/white)
  - Report problematic duplicates (>1s apart) that indicate content repetition
  - Exit with code 0 if valid, 1 if issues found
- If validation fails, investigate and fix the issue before delivering the video
- Common causes of failures:
  - Incorrect segment ordering in video pipelines
  - Overlapping timestamp extractions (check FFmpeg -ss and -t usage)
  - Gap assignments not matching chronological order
  - FFmpeg seeking issues with keyframes (use -ss before -i for input seeking)
- **Special attention for comment pipelines**: The precise_gap_pipeline.py automatically runs segment mapping diagnostics

## Audio Comment Video Generation - CRITICAL
**MANDATORY: When creating videos with audio comments, ALWAYS generate BOTH regular and debug versions**
- The debug version shows a timestamp overlay with all comment times and text
- This helps verify comment placement and spacing visually
- Commands:
  ```bash
  # Generate regular version
  python utils/auto_comment/precise_gap_pipeline.py <input_video.mp4>
  
  # ALWAYS also generate debug version with overlay
  python utils/auto_comment/precise_gap_pipeline.py <input_video.mp4> --debug
  ```
- The debug overlay shows:
  - List of all comments with timestamps
  - Comment text (truncated to 30 characters)
  - Visual verification of comment spacing
- Output files:
  - Regular: `<video_name>_precise_comments.mp4`
  - Debug: `<video_name>_precise_comments_debug.mp4`

## Audio Editing Transcript Verification - CRITICAL
**MANDATORY: After ANY video with audio editing (comments, music, etc.), MUST verify transcript alignment**
- This ensures inserted audio makes sense and is properly timed
- Command:
  ```bash
  python utils/video/validation/local_extract_transcript.py <output_video.mp4>
  ```
- The script will:
  - Extract and transcribe audio using Whisper (runs locally, no API costs)
  - Detect inserted comments/audio additions
  - Verify segments are aligned with the transcript
  - Check comments appear at natural pauses (not mid-sentence)
  - **🚨 DETECT REPEATED PHRASES**: Finds if same 3+ word sequences repeat within 20 seconds
  - Report timing and alignment issues
  - Exit with code 0 if valid, 1 if issues found
- Verification checks:
  1. **Segment Alignment**: Calculated segments match final transcript timing
  2. **Comment Placement**: Comments make contextual sense with surrounding content
  3. **Natural Pauses**: Comments appear at sentence boundaries, not mid-word
  4. **No Overlaps**: Audio doesn't overlap or cut off speech
  5. **🔴 NO REPEATED PHRASES**: Same 3+ word sequences should NOT repeat within 20 seconds
- Output files:
  - `*_whisper_transcript.json`: Full transcript with timestamps
  - `*_whisper_transcript.txt`: Plain text transcript
  - `*_verification.json`: Detailed verification report

### ⚠️ CRITICAL ISSUE: Repeated Phrases Detection
**If the script detects repeated phrases, this is a SEVERE PROBLEM that MUST be fixed immediately!**
- **What it means**: The video has duplicate content - segments are being played multiple times
- **Why it's critical**: This creates a stuttering effect where viewers hear the same words repeatedly
- **Common causes**:
  - Incorrect segment extraction with overlapping timestamps
  - FFmpeg seeking issues causing content duplication
  - Pipeline concatenating the same segments multiple times
  - Gap processing extracting wrong video portions
  - **KEYFRAME SEEKING BUG**: Using `-c:v copy` causes imprecise cuts at keyframes
- **IMMEDIATE ACTION REQUIRED**:
  1. **STOP** - Do NOT deliver the video to the user
  2. **ALERT** - Inform the user: "🚨 CRITICAL: Video has repeated content detected! The same phrases appear multiple times within 20 seconds. This needs immediate fixing."
  3. **DEBUG** - Run `utils/auto_comment/debug_repeated_content.py` to find exact duplication points
  4. **FIX** - Ensure ALL segments use re-encoding (NEVER use `-c:v copy` for extraction)
  5. **REVALIDATE** - After fixing, run validation again until no repeated phrases are found

### 📌 Comment Audio Organization
**Audio files for comments are stored in `uploads/assets/sounds/comments_audio/`**
- This folder contains audio files for ALL comment types (thoughtful, analytical, sarcastic, etc.)
- Comments can have various tones - sarcasm is just one style among many
- Audio files are named based on the comment text (e.g., "noncommutative.mp3", "synergistic.mp3")
- The pipeline searches for audio in this centralized location

### 📌 Comment Spacing Requirements
**CRITICAL: Comments MUST be spaced at least 20 seconds apart**
- The comment generator enforces minimum 20-second gaps between consecutive comments
- This ensures natural pacing and avoids overwhelming the viewer
- If remarks.json has comments closer than 20 seconds:
  - Use `regenerate_remarks_with_spacing.py` to filter them
  - Use `fix_remarks_for_scribe_gaps.py` to align with actual video gaps

### 📌 KNOWN FIX: Always Re-encode Video Segments
**CRITICAL**: When extracting video segments, ALWAYS re-encode with libx264, NEVER use `-c:v copy`
- **Problem**: Using copy codec causes FFmpeg to seek to nearest keyframe BEFORE the requested timestamp
- **Result**: Segments start earlier than intended, causing content overlap and repetition
- **Solution**: Always use `-c:v libx264` for precise cuts at exact timestamps
- **Example**:
  ```bash
  # WRONG - causes imprecise cuts
  ffmpeg -ss 10.5 -i video.mp4 -t 5 -c:v copy segment.mp4
  
  # CORRECT - precise cuts
  ffmpeg -ss 10.5 -i video.mp4 -t 5 -c:v libx264 -preset fast -crf 18 segment.mp4
  ```

## Project Context
This is a ToonTune backend project for processing SVG animations with human-like drawing sequences.

## Key Requirements
- Extract actual black lines from SVG images (not skeletons)
- Create continuous drawing paths that minimize pen lifts
- Draw in human-like order (head first for human figures)
- Group nearby components for logical drawing sequences

## Video Pipeline Output Rules
**IMPORTANT: ALL final video outputs MUST go to the `scenes/edited/` folder**
- The `scenes/edited/` folder is the ONLY output location for processed videos
- Each processing step (phrases, cartoons, karaoke) modifies videos IN-PLACE in the edited folder
- NO separate folders like `karaoke/`, `final_with_cartoons/`, etc.
- Always use ORIGINAL quality scenes for processing (not downsampled)
- Downsampled scenes are created for reference only
- Pipeline flow: original → edited → +phrases → +cartoons → +karaoke (all in edited/)

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

Located in: `utils/video/downsample/`

These utilities provide efficient downsampling to reduce file sizes and processing time while maintaining quality.

### Video Downsampling: `video_downsample.py`

**Default preset: `small` (256x256)**

```bash
# Standard usage with default "small" preset
python utils/video/downsample/video_downsample.py input_video.mp4

# Specific presets
python utils/video/downsample/video_downsample.py video.mp4 --preset small  # 256x256 (default)
python utils/video/downsample/video_downsample.py video.mp4 --preset tiny   # 64x64 (extreme)
python utils/video/downsample/video_downsample.py video.mp4 --preset mini   # 128x128

# Custom resolution with FPS reduction
python utils/video/downsample/video_downsample.py video.mp4 --size 320 240 --fps 15
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
python utils/video/downsample/image_downsample.py input_image.png

# Specific presets
python utils/video/downsample/image_downsample.py image.png --preset small  # 128x128 (default)
python utils/video/downsample/image_downsample.py image.png --preset mini   # 64x64
python utils/video/downsample/image_downsample.py image.png --preset icon   # 8x8 (extreme)

# With effects
python utils/video/downsample/image_downsample.py image.png --pixel-art     # Pixel art effect
python utils/video/downsample/image_downsample.py image.png --colors 16     # Reduce to 16 colors
python utils/video/downsample/image_downsample.py image.png --retro         # Retro game style

# Batch processing
python utils/video/downsample/image_downsample.py *.jpg --preset small
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

## ⚠️ CRITICAL: Dynamic Masking - NO CACHING EVER! ⚠️

### The Problem - 100% IMPORTANT
**NEVER CACHE MASKS - ALWAYS CALCULATE FRESH FOR EVERY FRAME**
- Moving objects (hands, body parts) change position every frame
- Cached masks will cause text to appear in front of objects incorrectly
- This is especially critical for "text behind object" effects

### Rules for Masking
1. **OBLITERATE ALL CACHING** - No `_frame_mask_cache` or similar
2. **FRESH CALCULATION EVERY FRAME** - Extract foreground mask dynamically
3. **NO STATIC MASKS** - Even first-frame masks become outdated immediately
4. **ALWAYS USE CURRENT FRAME** - Pass the actual current video frame to mask extraction

### Example of WRONG Implementation
```python
# WRONG - NEVER DO THIS!
if frame_number not in self._frame_mask_cache:
    mask = extract_foreground_mask(frame)
    self._frame_mask_cache[frame_number] = mask  # NO CACHING!
else:
    mask = self._frame_mask_cache[frame_number]  # STALE MASK!
```

### Example of CORRECT Implementation
```python
# CORRECT - ALWAYS DO THIS!
# Extract fresh mask for EVERY frame
mask = extract_foreground_mask(current_frame)
# Apply mask immediately without storing
```

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

## 3D Text Animation Optimal Positioning

**IMPORTANT: Text positioning now defaults to finding the least occluded location**

The `apply_3d_text_animation.py` script now automatically analyzes the video to find the optimal text position with minimal occlusion by foreground objects. This ensures text remains maximally visible throughout the animation.

### Positioning Behavior:
- **Default**: Automatic optimal position finding (least occluded across animation duration)
- **Manual override**: Use `--no-auto-position --position <x,y>` to specify exact position
- **Algorithm**: Analyzes multiple frames, evaluates grid of candidate positions, scores by visibility
- **Fallback**: If optimal position finding fails, uses center position

### Usage Examples:
```bash
# Default: auto-finds optimal position
python utils/animations/apply_3d_text_animation.py video.mp4 --text "HELLO"

# Disable auto-positioning, use center
python utils/animations/apply_3d_text_animation.py video.mp4 --text "HELLO" --no-auto-position

# Force specific position
python utils/animations/apply_3d_text_animation.py video.mp4 --text "HELLO" --no-auto-position --position 400,300
```

### Technical Details:
- Uses `OptimalTextPositionFinder` from `utils/text_placement/optimal_position_finder.py`
- Samples frames at regular intervals during motion animation period
- Evaluates 6x6 grid of candidate positions
- Scores based on average visibility, minimum visibility, and occlusion frames
- Applies slight center preference (15% weight) to avoid extreme edge positions