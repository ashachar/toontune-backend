"""
Word rendering and fog effects for word-level pipeline
"""

import cv2
import numpy as np
from typing import List
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from .models import WordObject
from .masking import ForegroundMaskExtractor


class WordRenderer:
    """Handles word rendering with masking and fog effects"""
    
    def __init__(self, video_path=None):
        """Initialize renderer with optional video path for cached masks
        
        Args:
            video_path: Path to video file for cached mask lookup
        """
        self.mask_extractor = ForegroundMaskExtractor(video_path)
        self.current_frame_number = 0
    
    def set_frame_number(self, frame_number: int):
        """Set the current frame number for mask synchronization
        
        Args:
            frame_number: Current frame number being processed
        """
        self.current_frame_number = frame_number
        self.mask_extractor.current_frame_idx = frame_number
    
    def _draw_text_with_outline(self, draw, position, text, font, opacity, is_behind=False, base_color=None):
        """Draw text with multi-layer outline effect for better visibility
        
        Creates a 3-layer effect:
        1. Outer white/light outline (thickest)
        2. Middle black outline (medium)
        3. Inner colored text (main text)
        """
        x, y = position
        alpha = int(255 * opacity)
        
        if is_behind:
            # Behind text: Use high contrast colors
            # Yellow text with black and white outlines (like the Hebrew example)
            main_color = (255, 215, 0, alpha)  # Gold/yellow
            mid_color = (0, 0, 0, alpha)  # Black
            outer_color = (255, 255, 255, int(alpha * 0.8))  # White (slightly transparent)
        else:
            # Front text: Use provided color or default to bright colors
            if base_color:
                # Use provided color for main text
                main_color = (*base_color, alpha)
                # Create contrast outlines
                mid_color = (0, 0, 0, alpha)  # Black middle
                outer_color = (255, 255, 255, int(alpha * 0.8))  # White outer
            else:
                # Default bright colors with good contrast
                main_color = (255, 255, 0, alpha)  # Yellow
                mid_color = (0, 0, 0, alpha)  # Black
                outer_color = (255, 255, 255, int(alpha * 0.8))  # White
        
        # Draw multiple layers of outlines (from outside to inside)
        # CRITICAL FIX: Use anchor="lt" (left-top) instead of default baseline anchor
        # This ensures descenders like 'p', 'g', 'y' are not cut off at the bottom
        # Outer white outline (thickest - 3 pixels)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if abs(dx) == 3 or abs(dy) == 3:
                    draw.text((x + dx, y + dy), text, font=font, fill=outer_color, anchor="lt")
        
        # Middle black outline (2 pixels)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) == 2 or abs(dy) == 2:
                    draw.text((x + dx, y + dy), text, font=font, fill=mid_color, anchor="lt")
        
        # Inner colored text - use left-top anchor to prevent descender cutoff
        draw.text((x, y), text, font=font, fill=main_color, anchor="lt")
    
    def render_phrase_behind(self, frame: np.ndarray, phrase_words: List[WordObject], time_seconds: float) -> np.ndarray:
        """Render a complete phrase as a single image for proper behind text masking."""
        if not phrase_words:
            return frame
        
        # Use the first word's properties as reference
        first_word = phrase_words[0]
        
        # CRITICAL FIX: Calculate actual phrase dimensions using the full text string
        phrase_text = " ".join(w.text for w in phrase_words)
        
        # Get the actual font to measure properly
        try:
            measure_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', first_word.font_size)
        except:
            measure_font = ImageFont.load_default()
        
        # Create temporary image for text measurement
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Get the actual bounding box of the entire phrase
        phrase_bbox = temp_draw.textbbox((0, 0), phrase_text, font=measure_font)
        phrase_width = phrase_bbox[2] - phrase_bbox[0]
        phrase_height_actual = phrase_bbox[3] - phrase_bbox[1]
        
        # Use the leftmost word's X position as the phrase X
        min_x = min(w.x for w in phrase_words)
        phrase_y = first_word.y
        # CRITICAL: Include extra height for BOTH ascenders and descenders
        # This prevents clipping of letters like 'f', 'l', 't' at top and 'p', 'g', 'y' at bottom
        # Use the actual measured height of the phrase
        base_phrase_height = phrase_height_actual
        ascender_space = 30  # Extra space above for tall letters
        descender_space = 50  # Extra space below for descenders
        phrase_height_with_padding = base_phrase_height + ascender_space + descender_space
        phrase_height = base_phrase_height  # Keep original for reference
        
        # Create a single image for the entire phrase
        # CRITICAL: Use same padding as individual word rendering to prevent clipping
        padding = 100
        descender_padding = 50  # Extra space for descenders like 'p', 'g', 'y'
        outline_padding = 10    # For the 3-pixel outlines in all directions
        canvas_width = phrase_width + padding * 2 + outline_padding * 2
        canvas_height = phrase_height + padding * 2 + descender_padding + outline_padding * 2
        phrase_img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(phrase_img)
        
        # Debug for phrases with problematic words
        if any(word.text in ["surprised", "operator", "if"] for word in phrase_words):
            print(f"   📝 PHRASE HEIGHT CHECK: '{phrase_text}'")
            print(f"      Base phrase height: {phrase_height}")
            print(f"      With padding: {phrase_height_with_padding} (ascender={ascender_space}, descender={descender_space})")
            print(f"      Canvas height: {canvas_height}")
            print(f"      phrase_y position: {phrase_y}")
        
        # CRITICAL FIX: Render the ENTIRE phrase as a SINGLE text string
        # This ensures proper baseline alignment and kerning
        # (phrase_text was already defined above when measuring)
        
        # Check if any word hasn't started animating yet
        first_animation_start = min(w.start_time - w.rise_duration for w in phrase_words)
        if time_seconds < first_animation_start:
            return frame  # Don't render anything if animation hasn't started
        
        # Calculate overall animation progress based on the first word
        # (all words in a phrase should animate together)
        first_word = phrase_words[0]
        animation_start = first_word.start_time - first_word.rise_duration
        rise_progress = 1.0
        if time_seconds < first_word.start_time:
            rise_progress = (time_seconds - animation_start) / first_word.rise_duration
        
        # Calculate opacity and offset for the entire phrase
        eased_progress = (1 - np.cos(rise_progress * np.pi)) / 2
        opacity = eased_progress
        
        # Rise from below or above
        y_offset = 0
        if rise_progress < 1.0:
            if first_word.from_below:
                y_offset = int((1 - eased_progress) * 50)
            else:
                y_offset = int((eased_progress - 1) * 50)
        
        # Use the font size from the first word (they should all be the same in a phrase)
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', first_word.font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw the ENTIRE phrase as ONE text string at the correct position
        # This ensures perfect baseline alignment and proper kerning
        phrase_x_in_canvas = padding + outline_padding
        phrase_y_in_canvas = padding + outline_padding + y_offset
        
        # Debug output
        if "AI created new math" in phrase_text or "surprised if" in phrase_text:
            print(f"   🎨 Rendering phrase as SINGLE string: '{phrase_text}'")
            print(f"      Position in canvas: ({phrase_x_in_canvas}, {phrase_y_in_canvas})")
            print(f"      Font size: {first_word.font_size}")
        
        # Draw the entire phrase with multi-layer outline effect for visibility
        self._draw_text_with_outline(draw, (phrase_x_in_canvas, phrase_y_in_canvas), 
                                    phrase_text, font, opacity, is_behind=True)
        
        # Convert to numpy array
        phrase_array = np.array(phrase_img)
        
        # Define alpha extraction coordinates (used later for compositing)
        alpha_y_start = padding + outline_padding
        alpha_x_start = padding + outline_padding
        
        # Apply masking for the entire phrase at once
        if self.mask_extractor:
            foreground_mask = self.mask_extractor.get_mask_for_frame(frame, self.current_frame_number)
            if foreground_mask is not None and len(foreground_mask.shape) == 3:
                # The mask is a full frame green screen - extract the region where text will be placed
                y_start = max(0, phrase_y)
                y_end = min(frame.shape[0], phrase_y + phrase_height)
                x_start = max(0, min_x)
                x_end = min(frame.shape[1], min_x + phrase_width)  # Use actual phrase width
                
                # Get the mask region from the same position in the green screen frame
                mask_region = foreground_mask[y_start:y_end, x_start:x_end]
                
                # Process green screen mask
                if len(mask_region.shape) == 3:
                    TARGET_GREEN_BGR = np.array([154, 254, 119], dtype=np.float32)
                    diff = mask_region.astype(np.float32) - TARGET_GREEN_BGR
                    distance = np.sqrt(np.sum(diff * diff, axis=2))
                    is_green = (distance < 50)
                    bg_mask = is_green.astype(np.float32)
                    
                    # Apply dilation to fix edge artifacts
                    fg_mask = 1.0 - bg_mask
                    kernel = np.ones((5, 5), np.uint8)
                    fg_mask_binary = (fg_mask * 255).astype(np.uint8)
                    fg_mask_binary = cv2.dilate(fg_mask_binary, kernel, iterations=1)
                    bg_mask = 1.0 - (fg_mask_binary.astype(np.float32) / 255.0)
                    
                    # Apply mask to phrase alpha
                    # Use the previously defined alpha extraction coordinates
                    # CRITICAL: Use only base_phrase_height for mask matching, not with descenders
                    phrase_alpha = phrase_array[alpha_y_start:alpha_y_start+phrase_height, 
                                               alpha_x_start:alpha_x_start+phrase_width, 3].astype(np.float32) / 255.0
                    
                    # Ensure mask and alpha have same shape
                    mask_h, mask_w = bg_mask.shape[:2]
                    alpha_h, alpha_w = phrase_alpha.shape[:2]
                    min_h = min(mask_h, alpha_h)
                    min_w = min(mask_w, alpha_w)
                    
                    phrase_alpha[:min_h, :min_w] *= bg_mask[:min_h, :min_w]
                    phrase_array[alpha_y_start:alpha_y_start+phrase_height, 
                               alpha_x_start:alpha_x_start+phrase_width, 3] = (phrase_alpha * 255).astype(np.uint8)
        
        # Composite the phrase onto the frame
        # Extract the phrase region including the full text with ascenders and descenders
        # The text is drawn at (padding + outline_padding), so extract from there
        # But include extra space above and below for ascenders/descenders
        extract_y_start = max(0, alpha_y_start - ascender_space)
        extract_y_end = min(phrase_array.shape[0], alpha_y_start + phrase_height + descender_space)
        phrase_to_composite = phrase_array[extract_y_start:extract_y_end, 
                                          alpha_x_start:alpha_x_start+phrase_width]
        
        # Calculate actual composite region on the frame
        # CRITICAL: Start ABOVE phrase_y to show ascenders, end BELOW for descenders
        comp_y_start = max(0, phrase_y - ascender_space)
        comp_y_end = min(frame.shape[0], phrase_y + phrase_height + descender_space)
        comp_x_start = max(0, min_x)
        comp_x_end = min(frame.shape[1], min_x + phrase_width)  # Use actual phrase width
        
        # Get the region sizes
        comp_height = comp_y_end - comp_y_start
        comp_width = comp_x_end - comp_x_start
        
        # Ensure we don't exceed the phrase_to_composite dimensions
        comp_height = min(comp_height, phrase_to_composite.shape[0])
        comp_width = min(comp_width, phrase_to_composite.shape[1])
        
        # Extract alpha channel for compositing
        alpha = phrase_to_composite[:comp_height, :comp_width, 3].astype(np.float32) / 255.0
        
        # Composite
        for c in range(3):
            frame[comp_y_start:comp_y_start+comp_height, comp_x_start:comp_x_start+comp_width, c] = (
                frame[comp_y_start:comp_y_start+comp_height, comp_x_start:comp_x_start+comp_width, c].astype(np.float32) * (1.0 - alpha) +
                phrase_to_composite[:comp_height, :comp_width, c].astype(np.float32) * alpha
            ).astype(np.uint8)
        
        return frame
    
    def render_word_with_masking(self, word_obj: WordObject, frame: np.ndarray, 
                                 time_seconds: float, fog_progress: float = 0.0, 
                                 is_dissolved: bool = False) -> np.ndarray:
        """Render word with foreground/background masking based on is_behind flag"""
        
        # Debug print for "be" word
        if word_obj.text == "be" and 3.3 <= time_seconds <= 3.5:
            print(f"\n🔍 RENDERING 'be' at {time_seconds:.2f}s:")
            print(f"   is_behind: {word_obj.is_behind}")
            print(f"   position: ({word_obj.x}, {word_obj.y})")
            print(f"   font_size: {word_obj.font_size if hasattr(word_obj, 'font_size') else 'N/A'}")
        
        # Don't render if dissolved or not started yet
        # Note: We allow rendering even if previous scene is dissolving to enable crossfade
        # CRITICAL DEBUG: Track "operator" and "surprised" words throughout their lifetime
        is_operator = word_obj.text == "operator"
        is_surprised = word_obj.text == "surprised"
        if is_operator or is_surprised:
            print(f"\n🔴 DEBUG '{word_obj.text}' at t={time_seconds:.3f}s, frame={self.current_frame_number}:")
            print(f"   Word position: x={word_obj.x}, y={word_obj.y}")
            print(f"   Word dimensions: width={word_obj.width}, height={word_obj.height}")
            print(f"   Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
        
        # Words should FINISH their animation at start_time, so begin at start_time - rise_duration
        animation_start = word_obj.start_time - word_obj.rise_duration
        if is_dissolved or time_seconds < animation_start:
            return frame
        
        # Calculate rise animation progress
        # Animation runs from (start_time - rise_duration) to start_time
        rise_progress = 1.0
        if time_seconds < word_obj.start_time:
            rise_progress = (time_seconds - animation_start) / word_obj.rise_duration
        
        if is_operator or is_surprised:
            print(f"   Rise progress: {rise_progress:.3f}")
            print(f"   Animation active: {rise_progress < 1.0}")
        
        # Create word image with padding for effects
        padding = 100
        # CRITICAL: Add extra padding for descenders and animation movement
        # This ensures letters like 'p', 'g', 'y' don't get clipped
        descender_padding = 50  # Extra space for descenders and animation
        # CRITICAL FIX: Add extra padding for the 3-pixel outlines we draw
        outline_padding = 10  # For the 3-pixel outlines in all directions
        canvas_width = word_obj.width + padding * 2 + outline_padding * 2
        canvas_height = word_obj.height + padding * 2 + descender_padding + outline_padding * 2
        
        if is_operator or is_surprised:
            print(f"   Canvas: {canvas_width}x{canvas_height} (pad={padding}, desc_pad={descender_padding}, outline_pad={outline_padding})")
            print(f"   Original word dimensions: {word_obj.width}x{word_obj.height}")
            print(f"   HEIGHT CHECK #1: Original word height = {word_obj.height}")
            print(f"   HEIGHT CHECK #2: Canvas height = {canvas_height}")
        
        word_img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(word_img)
        
        # Calculate rise offset (only during rise animation)
        y_offset = 0
        opacity = 1.0
        if rise_progress < 1.0:
            # Smooth easing - gentle sine curve
            eased_progress = (1 - np.cos(rise_progress * np.pi)) / 2
            opacity = eased_progress
            
            # Rise from below or above
            if word_obj.from_below:
                y_offset = int((1 - eased_progress) * 50)
            else:
                y_offset = int((eased_progress - 1) * 50)
        
        if is_operator or is_surprised:
            print(f"   Y offset: {y_offset} (from_below={word_obj.from_below})")
            print(f"   Opacity: {opacity:.3f}")
        
        # Draw word with opacity using the word's specific font size and color
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', word_obj.font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw text with multi-layer outline effect for visibility
        # CRITICAL: Draw at padding + outline_padding to account for the 3-pixel outlines
        draw_x = padding + outline_padding
        draw_y = padding + outline_padding
        if is_operator or is_surprised:
            print(f"   Drawing text at canvas position: ({draw_x}, {draw_y})")
            print(f"   Font size: {word_obj.font_size}")
        
        self._draw_text_with_outline(draw, (draw_x, draw_y), word_obj.text, font, opacity, 
                                    is_behind=False, base_color=word_obj.color if hasattr(word_obj, 'color') else None)
        
        # Convert to numpy
        word_array = np.array(word_img)
        
        if is_operator or is_surprised:
            print(f"   HEIGHT CHECK #3: After drawing, image array shape = {word_array.shape}")
            print(f"   HEIGHT CHECK #4: Image height after conversion = {word_array.shape[0]}")
        
        # Apply fog effect if needed (but NOT position change!)
        if fog_progress > 0:
            word_array = self._apply_fog_to_word(word_array, word_obj, fog_progress)
            if is_operator or is_surprised:
                print(f"   HEIGHT CHECK #5: After fog, array shape = {word_array.shape}")
        
        # Convert RGBA to BGRA for OpenCV
        word_bgr = np.zeros_like(word_array)
        word_bgr[:, :, 0] = word_array[:, :, 2]  # B = R
        word_bgr[:, :, 1] = word_array[:, :, 1]  # G = G
        word_bgr[:, :, 2] = word_array[:, :, 0]  # R = B
        word_bgr[:, :, 3] = word_array[:, :, 3]  # A = A
        
        if is_operator or is_surprised:
            print(f"   HEIGHT CHECK #6: word_bgr shape = {word_bgr.shape}")
        
        # Apply to frame at FIXED position (only y_offset during rise)
        # CRITICAL: Account for both padding and outline_padding
        actual_x = word_obj.x - padding - outline_padding
        actual_y = word_obj.y + y_offset - padding - outline_padding
        
        if is_operator or is_surprised:
            print(f"   Actual position: x={actual_x}, y={actual_y}")
            print(f"   After padding adjustment: word.y={word_obj.y}, y_offset={y_offset}, padding={padding}, outline={outline_padding}")
            print(f"   Word sprite shape: {word_bgr.shape}")
            print(f"   Text should appear at frame position: ({word_obj.x}, {word_obj.y + y_offset})")
            print(f"   ALIGNMENT CHECK: Word top={word_obj.y}, bottom={word_obj.y + word_obj.height}")
            print(f"   With the y-position fix, text should now be bottom-aligned within its stripe")
        
        # CRITICAL FIX: Calculate regions properly to avoid clipping
        # Frame region - where on the frame we'll place the text
        frame_y_start = max(0, actual_y)
        frame_y_end = min(frame.shape[0], actual_y + word_bgr.shape[0])
        frame_x_start = max(0, actual_x)
        frame_x_end = min(frame.shape[1], actual_x + word_bgr.shape[1])
        
        if is_operator or is_surprised:
            print(f"   HEIGHT CHECK #6.5: Calculating frame boundaries:")
            print(f"     actual_y={actual_y}, word_bgr height={word_bgr.shape[0]}")
            print(f"     frame_y_start={frame_y_start}, frame_y_end={frame_y_end}")
            print(f"     Frame visible height: {frame_y_end - frame_y_start}")
        
        # Sprite region - which part of the word sprite to use
        # CRITICAL FIX: ALWAYS use the entire sprite, starting from 0
        # The text is drawn at a specific position on the canvas with padding,
        # so we need the WHOLE sprite to show the complete text
        sprite_y_start = 0  # Always start from top of sprite
        sprite_x_start = 0  # Always start from left of sprite
        sprite_y_end = word_bgr.shape[0]  # Full height of sprite
        sprite_x_end = word_bgr.shape[1]  # Full width of sprite
        
        if is_operator or is_surprised:
            print(f"   Frame region: y=[{frame_y_start}:{frame_y_end}], x=[{frame_x_start}:{frame_x_end}]")
            print(f"   Sprite region: y=[{sprite_y_start}:{sprite_y_end}], x=[{sprite_x_start}:{sprite_x_end}]")
            print(f"   Using ENTIRE sprite: {word_bgr.shape}")
            print(f"   actual_y={actual_y}, actual_x={actual_x}")
            print(f"   HEIGHT CHECK #7: Before sprite extraction, word_bgr height = {word_bgr.shape[0]}")
        
        if frame_y_end > frame_y_start and frame_x_end > frame_x_start:
            # Extract the full sprite - no clipping!
            sprite_region = word_bgr
            
            if is_operator or is_surprised:
                print(f"   Actual sprite shape: {sprite_region.shape}")
                print(f"   HEIGHT CHECK #8: sprite_region height = {sprite_region.shape[0]}")
            
            if sprite_region.shape[2] == 4:
                alpha = sprite_region[:, :, 3].astype(np.float32) / 255.0
                
                # If rendering behind, extract mask and apply only to background
                if word_obj.is_behind:
                    foreground_mask = self.mask_extractor.get_mask_for_frame(frame, self.current_frame_number)
                    
                    # CRITICAL DEBUG: Check mask application for 'be' and other behind words
                    if word_obj.text in ["Would", "you", "be", "surprised", "if"] and time_seconds >= 2.9 and time_seconds <= 4.5:
                        print(f"\n🔴 CRITICAL DEBUG for '{word_obj.text}' at {time_seconds:.2f}s, frame {self.current_frame_number}:")
                        print(f"   is_behind: {word_obj.is_behind}")
                        print(f"   Position: ({word_obj.x}, {word_obj.y})")
                        print(f"   Mask loaded: {foreground_mask is not None}")
                        if foreground_mask is not None:
                            print(f"   Mask shape: {foreground_mask.shape}")
                    
                    # Debug: Check if mask is loaded and what values it contains
                    if foreground_mask is not None:
                        # Get the mask region for this text
                        mask_region = foreground_mask[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
                        
                        # Debug: Print mask statistics for "Would" word at specific times
                        if word_obj.text in ["Would", "you", "be"] and time_seconds >= 2.96 and time_seconds <= 3.6:
                            if not hasattr(self, '_debug_printed'):
                                self._debug_printed = set()
                            
                            debug_key = f"{word_obj.text}_{self.current_frame_number}"
                            if debug_key not in self._debug_printed:
                                self._debug_printed.add(debug_key)
                                print(f"\n🔍 DEBUG Mask for '{word_obj.text}' at frame {self.current_frame_number}, time {time_seconds:.2f}s:")
                                print(f"   Mask shape: {mask_region.shape}")
                                print(f"   Mask min: {mask_region.min()}, max: {mask_region.max()}, mean: {mask_region.mean():.1f}")
                                unique_vals = np.unique(mask_region)
                                if len(unique_vals) < 20:
                                    print(f"   Unique values: {unique_vals}")
                                
                                # Check how much is foreground vs background with different thresholds
                                for threshold in [150, 180, 200, 220]:
                                    fg_pixels = (mask_region < threshold).sum()
                                    bg_pixels = (mask_region >= threshold).sum()
                                    print(f"   Threshold {threshold}: FG={fg_pixels}, BG={bg_pixels} ({bg_pixels/(fg_pixels+bg_pixels)*100:.1f}% background)")
                        
                        # Green screen detection
                        # The mask is now a green screen video where green = background
                        if len(mask_region.shape) == 3:  # Color image (BGR format in OpenCV)
                            # HARD-CODED GREEN SCREEN COLOR from actual analysis:
                            # BGR: [154, 254, 119] - this is the exact green used by Replicate RVM
                            TARGET_GREEN_BGR = np.array([154, 254, 119], dtype=np.float32)
                            
                            # Calculate distance from target green for each pixel
                            # Using L2 distance in color space
                            diff = mask_region.astype(np.float32) - TARGET_GREEN_BGR
                            distance = np.sqrt(np.sum(diff * diff, axis=2))
                            
                            # Pixels within tolerance of target green are background
                            # Tolerance of 50 allows for compression artifacts
                            TOLERANCE = 50
                            is_green = (distance < TOLERANCE)
                            
                            bg_mask = is_green.astype(np.float32)
                            
                            if word_obj.text == "be" and 3.3 <= time_seconds <= 3.5:
                                green_pixels = is_green.sum()
                                total_pixels = mask_region.shape[0] * mask_region.shape[1]
                                non_green_pixels = total_pixels - green_pixels
                                print(f"   GREEN SCREEN MASK: Green(BG)={green_pixels}, Person(FG)={non_green_pixels}")
                                print(f"   {green_pixels/total_pixels*100:.1f}% is background (green)")
                                # Debug: show some pixel values
                                if mask_region.shape[0] > 0 and mask_region.shape[1] > 0:
                                    sample = mask_region[0, 0]
                                    print(f"   Sample pixel BGR: {sample}")
                        else:
                            # Fallback for grayscale (shouldn't happen with green screen)
                            if mask_region.size > 0:
                                MASK_THRESHOLD = 100
                                bg_mask = (mask_region >= MASK_THRESHOLD).astype(np.float32)
                            else:
                                bg_mask = np.zeros_like(mask_region, dtype=np.float32)
                        
                        # Fix edge artifacts: dilate the foreground (inverted mask)
                        # This expands the person area to cover green edge artifacts
                        if len(mask_region.shape) == 3:  # Only for green screen
                            # Invert to get foreground mask (1=person, 0=background)
                            fg_mask = 1.0 - bg_mask
                            # Dilate foreground to expand person area by 2-3 pixels
                            kernel = np.ones((5, 5), np.uint8)
                            fg_mask_binary = (fg_mask * 255).astype(np.uint8)
                            fg_mask_binary = cv2.dilate(fg_mask_binary, kernel, iterations=1)
                            # Convert back to background mask
                            bg_mask = 1.0 - (fg_mask_binary.astype(np.float32) / 255.0)
                        
                        # Apply the mask to the alpha channel
                        # CRITICAL FIX: Ensure mask and alpha have same shape
                        # They may differ if we extended sprite region for descenders
                        if bg_mask.shape != alpha.shape:
                            if is_operator or is_surprised:
                                print(f"   ⚠️ Shape mismatch: mask {bg_mask.shape} vs alpha {alpha.shape}")
                            # Crop or pad mask to match alpha shape
                            min_h = min(bg_mask.shape[0], alpha.shape[0])
                            min_w = min(bg_mask.shape[1], alpha.shape[1])
                            # Use only the overlapping region
                            bg_mask_cropped = bg_mask[:min_h, :min_w]
                            alpha_cropped = alpha[:min_h, :min_w]
                            # Apply mask to the cropped alpha
                            alpha[:min_h, :min_w] = alpha_cropped * bg_mask_cropped
                            # Leave the rest of alpha unchanged (for descenders beyond frame)
                        else:
                            # DEBUG: Check actual masking for problematic words
                            if word_obj.text == "be" and time_seconds >= 3.3 and time_seconds <= 3.5:
                                print(f"     🎯 Applying mask to alpha:")
                                print(f"        Alpha before: min={alpha.min():.2f}, max={alpha.max():.2f}, mean={alpha.mean():.2f}")
                                print(f"        bg_mask: min={bg_mask.min():.2f}, max={bg_mask.max():.2f}, mean={bg_mask.mean():.2f}")
                            
                            alpha = alpha * bg_mask
                        
                        if word_obj.text == "be" and time_seconds >= 3.3 and time_seconds <= 3.5:
                            print(f"        Alpha after: min={alpha.min():.2f}, max={alpha.max():.2f}, mean={alpha.mean():.2f}")
                            visible_pixels = (alpha > 0.1).sum()
                            total_pixels = alpha.size
                            print(f"        Visible pixels: {visible_pixels}/{total_pixels} ({visible_pixels/total_pixels*100:.1f}%)")
                            print(f"        Text should be {100 - visible_pixels/total_pixels*100:.1f}% hidden")
                    else:
                        print(f"⚠️ WARNING: No mask available for frame {self.current_frame_number}")
                
                # CRITICAL FIX: Don't skip ANY part of the sprite based on position
                # The text is drawn at a specific location on the canvas (padding + outline_padding)
                # We need to show that part regardless of where the sprite is positioned
                
                # Calculate the visible frame region dimensions
                frame_h = frame_y_end - frame_y_start
                frame_w = frame_x_end - frame_x_start
                
                # For negative positions, we start from within the sprite
                # but we DON'T skip the text content - we show it all
                if actual_y < 0:
                    # Sprite extends above frame - we need to show the bottom part
                    # But we show FROM THE TOP of the sprite to preserve text
                    sprite_y_offset = 0  # Start from top of sprite to show all text
                    available_h = min(sprite_region.shape[0], frame_h)
                else:
                    sprite_y_offset = 0
                    available_h = min(sprite_region.shape[0], frame_h)
                
                if actual_x < 0:
                    # Sprite extends left of frame
                    sprite_x_offset = 0  # Start from left of sprite to show all text
                    available_w = min(sprite_region.shape[1], frame_w) 
                else:
                    sprite_x_offset = 0
                    available_w = min(sprite_region.shape[1], frame_w)
                
                # Use the full sprite content that fits in the frame region
                sprite_to_use = sprite_region[:available_h, :available_w]
                alpha_to_use = alpha[:available_h, :available_w]
                
                if is_operator or is_surprised:
                    print(f"   Sprite shape: {sprite_region.shape}, Using: {sprite_to_use.shape}")
                    print(f"   Frame region: {frame_h}x{frame_w} at ({frame_y_start}, {frame_x_start})")
                    print(f"   Available size: {available_h}x{available_w}")
                    print(f"   HEIGHT CHECK #9: sprite_to_use height = {sprite_to_use.shape[0]}")
                    print(f"   HEIGHT CHECK #10: available_h = {available_h}, frame_h = {frame_h}")
                    print(f"   HEIGHT CHECK #11: FINAL compositing height = {min(sprite_to_use.shape[0], frame_h)}")
                
                # Composite the sprite onto the frame
                for c in range(3):
                    frame[frame_y_start:frame_y_start+available_h, frame_x_start:frame_x_start+available_w, c] = (
                        frame[frame_y_start:frame_y_start+available_h, frame_x_start:frame_x_start+available_w, c].astype(np.float32) * (1.0 - alpha_to_use) +
                        sprite_to_use[:, :, c].astype(np.float32) * alpha_to_use
                    ).astype(np.uint8)
        
        return frame
    
    def _apply_fog_to_word(self, word_img: np.ndarray, word_obj: WordObject, 
                          progress: float) -> np.ndarray:
        """Apply fog effect to word image without changing position"""
        if progress <= 0:
            return word_img
        
        result = word_img.copy()
        
        # Adjust progress with word's dissolve speed
        adjusted_progress = min(1.0, progress * word_obj.dissolve_speed)
        
        # Phase 1: Progressive blur
        if adjusted_progress > 0:
            blur_amount = adjusted_progress * 15
            blur_x = blur_amount * word_obj.blur_x
            blur_y = blur_amount * word_obj.blur_y
            
            if blur_x > 0 or blur_y > 0:
                result = cv2.GaussianBlur(result, (0, 0), 
                                         sigmaX=blur_x, sigmaY=blur_y)
        
        # Phase 2: Fog texture
        if adjusted_progress > 0.3:
            fog_progress = (adjusted_progress - 0.3) / 0.5
            h, w = result.shape[:2]
            fog = np.random.randn(h, w) * 20 * fog_progress
            fog = gaussian_filter(fog, sigma=3)
            
            if result.shape[2] == 4:
                alpha = result[:, :, 3].astype(np.float32)
                alpha = alpha * (1.0 - fog_progress * 0.5)
                alpha = np.clip(alpha + fog * word_obj.fog_density, 0, 255)
                result[:, :, 3] = alpha.astype(np.uint8)
        
        # Phase 3: Final fade
        if adjusted_progress > 0.6:
            fade_progress = (adjusted_progress - 0.6) / 0.4
            fade_amount = 1.0 - (fade_progress * 0.9)
            
            if result.shape[2] == 4:
                result[:, :, 3] = (result[:, :, 3] * fade_amount).astype(np.uint8)
        
        return result