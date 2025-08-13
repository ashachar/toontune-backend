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