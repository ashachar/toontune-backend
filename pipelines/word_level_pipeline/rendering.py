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
        """Render word with consistent animation, apply masking as post-processing"""
        
        # Early exit if word shouldn't be visible yet
        animation_start = word_obj.start_time - word_obj.rise_duration
        if is_dissolved or time_seconds < animation_start:
            return frame
        
        # Step 1: Calculate animation parameters (same for ALL words)
        rise_progress = 1.0
        if time_seconds < word_obj.start_time:
            rise_progress = (time_seconds - animation_start) / word_obj.rise_duration
        
        y_offset = 0
        opacity = 1.0
        if rise_progress < 1.0:
            # Smooth easing
            eased_progress = (1 - np.cos(rise_progress * np.pi)) / 2
            opacity = eased_progress
            # Rise animation direction
            if word_obj.from_below:
                y_offset = int((1 - eased_progress) * 50)
            else:
                y_offset = int((eased_progress - 1) * 50)
        
        # Step 2: Create word sprite (same for ALL words)
        sprite = self._create_word_sprite(word_obj, opacity, fog_progress)
        
        # Step 3: Composite sprite onto frame with masking if needed
        # Pass apply_mask=True for behind words to handle masking during compositing
        frame = self._composite_sprite(sprite, frame, word_obj.x, word_obj.y + y_offset, 
                                      apply_mask=word_obj.is_behind)
        
        return frame
    
    def _create_word_sprite(self, word_obj: WordObject, opacity: float, fog_progress: float) -> np.ndarray:
        """Create the word sprite with consistent rendering for all words"""
        # Padding for effects
        padding = 100
        descender_padding = 50
        outline_padding = 10
        canvas_width = word_obj.width + padding * 2 + outline_padding * 2
        canvas_height = word_obj.height + padding * 2 + descender_padding + outline_padding * 2
        
        # Create canvas
        word_img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(word_img)
        
        # Get font
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', word_obj.font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw text at consistent position
        draw_x = padding + outline_padding
        draw_y = padding + outline_padding
        self._draw_text_with_outline(draw, (draw_x, draw_y), word_obj.text, font, opacity,
                                    is_behind=word_obj.is_behind, 
                                    base_color=word_obj.color if hasattr(word_obj, 'color') else None)
        
        # Convert to numpy array
        word_array = np.array(word_img)
        
        # Apply fog if needed
        if fog_progress > 0:
            word_array = self._apply_fog_to_word(word_array, word_obj, fog_progress)
        
        # Convert RGBA to BGRA for OpenCV
        word_bgr = np.zeros_like(word_array)
        word_bgr[:, :, 0] = word_array[:, :, 2]  # B = R
        word_bgr[:, :, 1] = word_array[:, :, 1]  # G = G
        word_bgr[:, :, 2] = word_array[:, :, 0]  # R = B
        word_bgr[:, :, 3] = word_array[:, :, 3]  # A = A
        
        return word_bgr
    
    def _composite_sprite(self, sprite: np.ndarray, frame: np.ndarray, x: int, y: int, 
                         apply_mask: bool = False) -> np.ndarray:
        """Composite sprite onto frame at given position with optional masking"""
        # Account for padding in sprite
        padding = 100
        outline_padding = 10
        actual_x = x - padding - outline_padding
        actual_y = y - padding - outline_padding
        
        # Calculate frame boundaries
        frame_y_start = max(0, actual_y)
        frame_y_end = min(frame.shape[0], actual_y + sprite.shape[0])
        frame_x_start = max(0, actual_x)
        frame_x_end = min(frame.shape[1], actual_x + sprite.shape[1])
        
        if frame_y_end <= frame_y_start or frame_x_end <= frame_x_start:
            return frame  # Sprite is completely out of frame
        
        # Calculate sprite region to use
        sprite_y_start = max(0, -actual_y) if actual_y < 0 else 0
        sprite_x_start = max(0, -actual_x) if actual_x < 0 else 0
        sprite_y_end = sprite_y_start + (frame_y_end - frame_y_start)
        sprite_x_end = sprite_x_start + (frame_x_end - frame_x_start)
        
        # Extract sprite region and alpha
        sprite_region = sprite[sprite_y_start:sprite_y_end, sprite_x_start:sprite_x_end]
        if sprite_region.shape[2] != 4:
            return frame  # No alpha channel
            
        alpha = sprite_region[:, :, 3].astype(np.float32) / 255.0
        
        # If masking is requested, modify alpha based on foreground mask
        if apply_mask:
            foreground_mask = self.mask_extractor.get_mask_for_frame(frame, self.current_frame_number)
            if foreground_mask is not None:
                # Extract mask for this region
                mask_region = foreground_mask[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
                
                # Process green screen mask
                if len(mask_region.shape) == 3:  # Color mask (green screen)
                    # Green = background (show text), Non-green = foreground (hide text)
                    TARGET_GREEN_BGR = np.array([154, 254, 119], dtype=np.float32)
                    diff = mask_region.astype(np.float32) - TARGET_GREEN_BGR
                    distance = np.sqrt(np.sum(diff * diff, axis=2))
                    is_background = (distance < 50).astype(np.float32)
                    
                    # Apply dilation to foreground to handle edges
                    kernel = np.ones((3, 3), np.uint8)
                    is_foreground = (1.0 - is_background).astype(np.uint8)
                    is_foreground = cv2.dilate(is_foreground, kernel, iterations=1)
                    is_background = 1.0 - is_foreground.astype(np.float32)
                    
                    # Multiply alpha by background mask
                    alpha = alpha * is_background
        
        # Composite onto frame
        for c in range(3):
            frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end, c] = (
                frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end, c].astype(np.float32) * (1.0 - alpha) +
                sprite_region[:, :, c].astype(np.float32) * alpha
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
        if 0.2 <= adjusted_progress <= 0.8:
            blur_amount = int(adjusted_progress * 15)
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
