Here is the detailed specification for the video editing tricks, formatted for an automation software's architecture.

```markdown
# Video Editing Tricks: Technical Specifications
_Location: `utils/editing_tricks/`_

This document outlines the detailed specifications for automated video editing functions. Each function is designed to be a self-contained module that receives specific inputs (video clips, parameters) and returns a modified clip or a new layer to be composited.

---

### `utils/editing_tricks/color_effects.py`

#### Function: `apply_color_splash`

Automates the creation of a "color splash" effect, desaturating all colors in a clip except for a specified target color range.

```python
# Data Structures and Enums
from typing import List, Tuple, Union

ColorTarget = Union[str, Tuple[int, int]] # e.g., 'yellow' or a hue range like (15, 45)

# Function Signature
def apply_color_splash(
    video_clip: 'VideoClip',
    target_color_range: List[ColorTarget],
    saturation_preserved: float = 1.0,
    saturation_other: float = 0.0,
    feathering: int = 10
) -> 'VideoClip':
    """
    Isolates one or more color ranges, desaturating the rest of the video.

    Args:
        video_clip (VideoClip): The input video clip object.
        target_color_range (List[ColorTarget]): A list of colors to preserve.
            - Can be color names ('red', 'green', 'yellow', 'orange', 'blue', etc.)
            - Or specific HSL hue ranges e.g., [(15, 45), (45, 75)] for oranges and yellows.
        saturation_preserved (float): Saturation level for target colors (0.0 to 2.0). Default is 1.0 (unchanged).
        saturation_other (float): Saturation level for all other colors (0.0 to 2.0). Default is 0.0 (grayscale).
        feathering (int): Pixel value to blur the edges of the color mask for a smoother transition.

    Returns:
        VideoClip: A new video clip with the color splash effect applied.
    """
    # --- Implementation Details ---
    # 1.  **Color Range Mapping**: Convert string color names in `target_color_range` to their corresponding HSL hue ranges.
    #     - e.g., 'yellow' -> (45, 75), 'orange' -> (15, 45). Store this in an internal map.
    # 2.  **Frame-by-Frame Processing**: Iterate through each frame of the `video_clip`.
    # 3.  **Color Space Conversion**: Convert the current frame from RGB to HSL color space.
    # 4.  **Mask Generation**:
    #     - Create a binary mask of the same dimensions as the frame.
    #     - For each pixel, check if its Hue value falls within any of the specified `target_color_range` intervals.
    #     - If it does, set the corresponding pixel in the mask to white (255); otherwise, set it to black (0).
    # 5.  **Mask Feathering**: Apply a Gaussian blur with a kernel size derived from the `feathering` parameter to the binary mask. This softens the transition edges.
    # 6.  **Layer Creation**:
    #     - Create a fully desaturated version of the frame by setting its Saturation channel to `saturation_other`.
    #     - Create a version of the frame with target colors adjusted by setting its Saturation channel to `saturation_preserved`.
    # 7.  **Compositing**: Blend the two versions using the feathered mask. The mask determines the alpha channel for the `saturation_preserved` layer.
    # 8.  **Final Conversion**: Convert the processed frame back to RGB color space.
    # 9.  **Output**: Return a new `VideoClip` object composed of the processed frames.
    pass
```

---

### `utils/editing_tricks/text_effects.py`

#### Function: `apply_text_behind_subject`

Places text in 3D space so it appears behind a moving subject in the video. Includes an optional animation where the text moves from in front to behind the subject.

```python
# Data Structures and Enums
from typing import Optional, Dict
from enum import Enum

class TextAnimation(Enum):
    NONE = "none"
    ZOOM_IN_OUT = "zoom_in_out"

# Function Signature
def apply_text_behind_subject(
    video_clip: 'VideoClip',
    text_content: str,
    text_properties: Dict, # font, size, color, etc.
    start_time: float,
    end_time: float,
    animation: TextAnimation = TextAnimation.NONE,
    animation_params: Optional[Dict] = None
) -> 'VideoComposition':
    """
    Creates the text-behind-subject effect by layering a subject cutout over text.

    Args:
        video_clip (VideoClip): The input video clip.
        text_content (str): The text to display.
        text_properties (Dict): Dictionary of text styling (font, size, color, etc.).
        start_time (float): Start timestamp of the effect.
        end_time (float): End timestamp of the effect.
        animation (TextAnimation): The animation style to apply to the text.
        animation_params (Optional[Dict]): Parameters for the animation.
            - For ZOOM_IN_OUT: {'mid_scale': 1.1, 'end_scale': 0.9, 'transition_time': 2.5, 'fade_duration': 0.1}

    Returns:
        VideoComposition: A composition object containing the three layers (background, text, foreground).
    """
    # --- Implementation Details ---
    # 1.  **Input Validation**: Check if `animation_params` are provided when `animation` is not `NONE`.
    # 2.  **Clip Duplication**: Create two copies of the `video_clip` segment from `start_time` to `end_time`.
    #     - `background_layer`: The original, untouched video segment.
    #     - `foreground_layer`: The copy on which subject segmentation will be performed.
    # 3.  **Subject Segmentation**:
    #     - Apply a high-quality background removal model (e.g., U2-Net, BGMv2) to `foreground_layer`.
    #     - The result is a clip with only the main subject on a transparent background.
    # 4.  **Text Layer Creation**:
    #     - Generate a `TextClip` based on `text_content` and `text_properties`.
    #     - **If `animation` is `ZOOM_IN_OUT`**:
    #         a. Define 3 scale keyframes: start (1.0), middle (`mid_scale`), end (`end_scale`).
    #         b. Apply an easing curve for a smooth animation.
    #         c. Find the `transition_time` within the clip's local timeline.
    #         d. Split the animated text layer at `transition_time`.
    #         e. The first part will be placed above `foreground_layer`, the second part below it.
    #         f. To smooth the transition, extend the first part by `fade_duration` and apply a fade-out animation.
    #     - **If `animation` is `NONE`**:
    #         a. The text layer is static and will be placed entirely between the background and foreground layers.
    # 5.  **Layer Compositing**:
    #     - Create a `VideoComposition` object.
    #     - The final layer order (bottom to top) will be:
    #         1. `background_layer`
    #         2. Text Layer (or second part of animated text)
    #         3. `foreground_layer`
    #         4. First part of animated text (if applicable)
    # 6.  **Output**: Return the `VideoComposition`.
    pass
```

#### Function: `apply_motion_tracking_text`

Attaches a text layer to a moving object within the video.

```python
# Data Structures and Enums
from typing import Dict, Union, Tuple, List

BoundingBox = Tuple[int, int, int, int] # (x, y, width, height)
TrackTarget = Union[str, BoundingBox] # Text description or initial bounding box

# Function Signature
def apply_motion_tracking_text(
    video_clip: 'VideoClip',
    text_content: str,
    text_properties: Dict,
    track_target: TrackTarget,
    start_time: float,
    end_time: float,
    track_scale: bool = False
) -> 'TextClip':
    """
    Generates a text layer with position animated to follow a tracked object.

    Args:
        video_clip (VideoClip): The source video clip for tracking.
        text_content (str): The text to display.
        text_properties (Dict): Styling for the text.
        track_target (TrackTarget): The object to track. Can be a text description
            (e.g., "the person on the left") for an AI object detector, or a
            bounding box for the first frame.
        start_time (float): Timestamp to start tracking.
        end_time (float): Timestamp to end tracking.
        track_scale (bool): If True, the text will scale with the tracked object.

    Returns:
        TextClip: A new TextClip object with animated position (and scale) properties.
    """
    # --- Implementation Details ---
    # 1.  **Tracker Initialization**:
    #     - If `track_target` is a string, use an object detection model (e.g., YOLO) on the frame at `start_time` to find the object and get its initial bounding box.
    #     - Initialize a robust object tracker (e.g., KCF, MOSSE, or a more modern Siamese network-based tracker) with the initial bounding box.
    # 2.  **Tracking Loop**:
    #     - For each frame from `start_time` to `end_time`, feed the frame to the tracker.
    #     - The tracker will output the new bounding box for the object.
    #     - Store the center coordinates `(x, y)` and optionally the size of the bounding box for each frame.
    # 3.  **Keyframe Generation**: The list of coordinates and sizes becomes the animation path. This is a list of keyframes, one for every frame in the duration.
    # 4.  **Text Layer Creation**:
    #     - Create a `TextClip` with the specified content and properties.
    #     - Apply the generated animation path to the `position` property of the `TextClip`.
    #     - If `track_scale` is True, apply the size animation to the `scale` property.
    # 5.  **Output**: Return the fully animated `TextClip`, ready to be composited over the `video_clip`.
    pass
```

---

### `utils/editing_tricks/motion_effects.py`

#### Function: `apply_floating_effect`

Applies a gentle, continuous swaying motion to a layer (clip, image, or text) to make it appear as if it's floating.

```python
# Function Signature
def apply_floating_effect(
    target_layer: 'MediaLayer', # Can be VideoClip, ImageClip, or TextClip
    speed: float = 1.0,
    strength: float = 5.0,
    variability: float = 0.5
) -> 'MediaLayer':
    """
    Animates a layer with a smooth, floating motion.

    Args:
        target_layer (MediaLayer): The layer to be animated.
        speed (float): Controls the frequency of the swaying motion.
        strength (float): Controls the amplitude (in pixels/degrees) of the motion.
        variability (float): Adds organic randomness. 0.0 is a perfect sine wave, 1.0 is highly variable.

    Returns:
        MediaLayer: The same layer with the animation applied.
    """
    # --- Implementation Details ---
    # 1.  **Compound Clip Handling**: The video mentions this is crucial. The function should first ensure the `target_layer` is treated as a single, renderable entity before applying transforms.
    # 2.  **Procedural Animation**: The effect is a combination of procedural animations on the position and rotation properties.
    # 3.  **Noise Generation**:
    #     - Generate three 1D Perlin noise sequences over time for the duration of the layer.
    #     - One for X position, one for Y position, one for rotation.
    #     - The `speed` parameter controls the frequency of the noise sampling.
    # 4.  **Base Motion**:
    #     - Create base sine waves for X, Y, and rotation. The `speed` parameter also affects their frequency.
    # 5.  **Combining Motions**:
    #     - Combine the base sine wave and the Perlin noise using a weighted average, where `variability` is the weight of the noise. `final_motion = (1 - variability) * sine_wave + variability * perlin_noise`.
    # 6.  **Applying Strength**: Multiply the final combined motion curves by the `strength` parameter to scale the animation's amplitude.
    # 7.  **Animation Application**: Apply the resulting animation curves to the `position` and `rotation` properties of the `target_layer`.
    # 8.  **Output**: Return the modified `target_layer`.
    pass
```

#### Function: `apply_smooth_zoom`

Creates a professional pan-and-zoom effect with customizable easing curves.

```python
# Data Structures and Enums
from typing import Tuple, Union

EasingCurve = Union[str, Tuple[float, float, float, float]] # 'linear', 'ease_in_out' or Bezier curve (x1,y1,x2,y2)

# Function Signature
def apply_smooth_zoom(
    video_clip: 'VideoClip',
    start_time: float,
    end_time: float,
    start_scale: float = 1.0,
    end_scale: float = 1.2,
    start_position: Tuple[int, int] = (0, 0), # Relative to center
    end_position: Tuple[int, int] = (0, 0),
    easing_curve: EasingCurve = 'ease_in_out'
) -> 'VideoClip':
    """
    Applies a smooth scale and position animation to a video clip.

    Args:
        video_clip (VideoClip): The clip to apply the zoom to.
        start_time (float): Start timestamp of the zoom.
        end_time (float): End timestamp of the zoom.
        start_scale (float): Initial scale factor.
        end_scale (float): Final scale factor.
        start_position (Tuple[int, int]): Initial (x, y) offset from center.
        end_position (Tuple[int, int]): Final (x, y) offset from center.
        easing_curve (EasingCurve): The easing curve for the animation timing.
            Accepts presets or a tuple defining a cubic-bezier curve.

    Returns:
        VideoClip: The video clip with the zoom animation applied.
    """
    # --- Implementation Details ---
    # 1.  **Keyframe Definition**:
    #     - Create a keyframe for the `scale` and `position` properties at `start_time`.
    #     - Set their values to `start_scale` and `start_position`.
    #     - Create a second keyframe for `scale` and `position` at `end_time`.
    #     - Set their values to `end_scale` and `end_position`.
    # 2.  **Easing Application**:
    #     - The interpolation between these two keyframes must follow the specified `easing_curve`.
    #     - If `easing_curve` is a string, map it to a predefined bezier curve tuple (e.g., 'ease_in_out' -> (0.42, 0, 0.58, 1.0)).
    #     - Use the bezier curve function to calculate the animation's progress at each frame between `start_time` and `end_time`.
    # 3.  **Output**: Return the modified `VideoClip` object, which now contains the animation data. The rendering engine will use this data during composition.
    pass
```

#### Function: `apply_3d_photo_effect`

Converts a still image into a short video with a 3D parallax effect.

```python
# Data Structures and Enums
from enum import Enum

class ParallaxMovement(Enum):
    ZOOM_IN = "zoom_in"
    DOLLY_LEFT = "dolly_left"
    SLOW_PAN_RIGHT = "slow_pan_right"

# Function Signature
def apply_3d_photo_effect(
    image_clip: 'ImageClip',
    duration: float,
    movement_style: ParallaxMovement = ParallaxMovement.ZOOM_IN,
    intensity: float = 1.0
) -> 'VideoClip':
    """
    Creates a 3D parallax effect from a 2D image.

    Args:
        image_clip (ImageClip): The input image.
        duration (float): The desired duration of the output video clip.
        movement_style (ParallaxMovement): The type of virtual camera movement.
        intensity (float): The strength of the 3D effect. Higher values create more separation.

    Returns:
        VideoClip: A video clip with the 3D effect.
    """
    # --- Implementation Details ---
    # 1.  **Depth Estimation**:
    #     - Feed the source image into a monocular depth estimation AI model (e.g., MiDaS).
    #     - This produces a grayscale depth map where pixel intensity corresponds to distance.
    # 2.  **Layered Scene Creation (2.5D)**:
    #     - Use the depth map to project the 2D image onto a 3D mesh (plane displacement) or simulate a multi-plane scene.
    # 3.  **Inpainting/Content-Aware Fill**:
    #     - As the virtual camera moves, parts of the background previously occluded by the foreground will become visible.
    #     - Use an inpainting model to intelligently fill these newly visible gaps to avoid black borders or stretching artifacts.
    # 4.  **Virtual Camera Animation**:
    #     - Create a virtual camera in the 2.5D scene.
    #     - Animate its position over the specified `duration` according to the `movement_style`.
    #         - `ZOOM_IN`: Move the camera forward along the Z-axis.
    #         - `DOLLY_LEFT`: Move the camera sideways along the X-axis.
    #     - The `intensity` parameter will scale the magnitude of this movement.
    # 5.  **Frame Rendering**: Render the view from the virtual camera for each frame of the `duration`, creating the final video sequence.
    # 6.  **Output**: Return the rendered frames as a new `VideoClip` object.
    pass
```

---

### `utils/editing_tricks/layout_effects.py`

#### Function: `apply_highlight_focus`

Draws viewer attention by darkening the video except for a specified highlighted region.

```python
# Data Structures and Enums
from typing import Dict, Union

class HighlightShape(Enum):
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"

HighlightArea = Dict[str, Union[HighlightShape, int, float]] # e.g., {'shape': HighlightShape.RECTANGLE, 'x': 100, 'y': 200, 'width': 300, 'height': 50, 'rotation': 15}

# Function Signature
def apply_highlight_focus(
    video_clip: 'VideoClip',
    highlight_area: HighlightArea,
    start_time: float,
    end_time: float,
    darken_level: float = -0.8,
    feather: int = 50,
    fade_duration: float = 0.2
) -> 'VideoComposition':
    """
    Highlights a specific area by darkening everything else.

    Args:
        video_clip (VideoClip): The clip to apply the effect to.
        highlight_area (HighlightArea): A dictionary defining the shape and position of the focus area.
        start_time (float): Timestamp to begin the effect.
        end_time (float): Timestamp to end the effect.
        darken_level (float): The brightness/exposure adjustment for the background (-1.0 to 0.0).
        feather (int): Pixel value for softening the mask edges.
        fade_duration (float): Duration of the fade-in/fade-out animation.

    Returns:
        VideoComposition: A composition of the original clip and the darkening overlay.
    """
    # --- Implementation Details ---
    # 1.  **Layer Duplication**: Create a copy of the video segment (`top_layer`).
    # 2.  **Brightness Adjustment**: Apply a brightness/exposure filter to `top_layer` with the value `darken_level`.
    # 3.  **Mask Creation**:
    #     - Create a mask layer of the same dimensions.
    #     - Draw the shape defined by `highlight_area` onto this mask (e.g., a white rectangle on a black background).
    # 4.  **Mask Modification**:
    #     - Apply a Gaussian blur to the mask using the `feather` value.
    #     - **Invert** the mask, so the highlight area is black (transparent) and the surrounding area is white (opaque).
    # 5.  **Apply Mask**: Use this inverted, feathered mask as the alpha channel for `top_layer`.
    # 6.  **Animation**:
    #     - Apply a fade-in animation to `top_layer`'s opacity at `start_time`, lasting `fade_duration`.
    #     - Apply a fade-out animation at `end_time - fade_duration`.
    # 7.  **Composition**: Return a `VideoComposition` with the original `video_clip` on the bottom and the animated `top_layer` on top.
    pass
```

#### Function: `add_progress_bar`

Adds a horizontal progress bar that animates over the duration of a clip.

```python
# Data Structures and Enums
from enum import Enum

class BarPosition(Enum):
    TOP = "top"
    BOTTOM = "bottom"

# Function Signature
def add_progress_bar(
    duration: float,
    position: BarPosition = BarPosition.TOP,
    height: int = 10,
    color: str = "#FF0000",
    margin: int = 0
) -> 'ShapeClip':
    """
    Creates a progress bar layer that animates from left to right.

    Args:
        duration (float): The total duration the progress bar should animate over.
        position (BarPosition): Vertical position on the screen.
        height (int): Thickness of the bar in pixels.
        color (str): Hex color code for the bar.
        margin (int): Pixel distance from the top/bottom edge.

    Returns:
        ShapeClip: A shape layer containing the animated progress bar.
    """
    # --- Implementation Details ---
    # 1.  **Shape Creation**: Generate a `ShapeClip` containing a rectangle.
    #     - Width: Full screen width.
    #     - Height: `height`.
    #     - Color: `color`.
    #     - Position: Set Y-position based on `position` and `margin`.
    # 2.  **Animation Method (Masking)**:
    #     - The most robust method is to apply a "wipe" animation.
    #     - This is achieved by animating a rectangular mask over the shape layer.
    #     - At time 0, the mask reveals 0% of the shape.
    #     - At time `duration`, the mask reveals 100% of the shape.
    #     - The interpolation must be **linear** to accurately reflect progress.
    # 3.  **Output**: Return the `ShapeClip` with the linear animation data attached. It will be composited on top of the main timeline.
    pass
```

#### Function: `apply_video_in_text`

Displays a video within the boundaries of large text characters.

```python
# Function Signature
def apply_video_in_text(
    video_clip: 'VideoClip',
    text_content: str,
    text_properties: Dict, # Must include a thick/bold font for good results
    background_layer: 'MediaLayer'
) -> 'VideoComposition':
    """
    Composites a video to be visible only through the shape of text.

    Args:
        video_clip (VideoClip): The video to play inside the text.
        text_content (str): The text to use as a mask.
        text_properties (Dict): Styling for the text. A thick font is highly recommended.
        background_layer (MediaLayer): The layer to be seen around the text.

    Returns:
        VideoComposition: The final composition with the effect.
    """
    # --- Implementation Details ---
    # 1.  **Text Matte Creation**:
    #     - Generate a `TextClip` based on `text_content` and `text_properties`.
    #     - This clip should be rendered as a high-contrast matte: white text on a pure black background.
    # 2.  **Track Matte Application**:
    #     - Use the generated text matte as an Alpha Matte or Luma Matte for the `video_clip`.
    #     - The `video_clip` will inherit the alpha channel (or luminance values) of the matte. This makes the video transparent everywhere except where the white text was.
    # 3.  **Layer Ordering**:
    #     - Create a `VideoComposition` with the following layers (bottom to top):
    #         1. `background_layer`
    #         2. The matted `video_clip`
    # 4.  **Output**: Return the final `VideoComposition`.
    pass