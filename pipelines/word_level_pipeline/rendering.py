"""
Word rendering and fog effects for word-level pipeline
"""

import cv2
import numpy as np
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
    
    def render_word_with_masking(self, word_obj: WordObject, frame: np.ndarray, 
                                 time_seconds: float, fog_progress: float = 0.0, 
                                 is_dissolved: bool = False) -> np.ndarray:
        """Render word with foreground/background masking based on is_behind flag"""
        
        # Don't render if dissolved or not started yet
        # Note: We allow rendering even if previous scene is dissolving to enable crossfade
        # Words should FINISH their animation at start_time, so begin at start_time - rise_duration
        animation_start = word_obj.start_time - word_obj.rise_duration
        if is_dissolved or time_seconds < animation_start:
            return frame
        
        # Calculate rise animation progress
        # Animation runs from (start_time - rise_duration) to start_time
        rise_progress = 1.0
        if time_seconds < word_obj.start_time:
            rise_progress = (time_seconds - animation_start) / word_obj.rise_duration
        
        # Create word image with padding for effects
        padding = 100
        canvas_width = word_obj.width + padding * 2
        canvas_height = word_obj.height + padding * 2
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
        
        # Draw word with opacity using the word's specific font size and color
        # Use word's color if available, otherwise default to white
        base_color = word_obj.color if hasattr(word_obj, 'color') else (255, 255, 255)
        text_color = (*base_color, int(255 * opacity))
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', word_obj.font_size)
        except:
            font = ImageFont.load_default()
        draw.text((padding, padding), word_obj.text, fill=text_color, font=font)
        
        # Convert to numpy
        word_array = np.array(word_img)
        
        # Apply fog effect if needed (but NOT position change!)
        if fog_progress > 0:
            word_array = self._apply_fog_to_word(word_array, word_obj, fog_progress)
        
        # Convert RGBA to BGRA for OpenCV
        word_bgr = np.zeros_like(word_array)
        word_bgr[:, :, 0] = word_array[:, :, 2]  # B = R
        word_bgr[:, :, 1] = word_array[:, :, 1]  # G = G
        word_bgr[:, :, 2] = word_array[:, :, 0]  # R = B
        word_bgr[:, :, 3] = word_array[:, :, 3]  # A = A
        
        # Apply to frame at FIXED position (only y_offset during rise)
        actual_x = word_obj.x - padding
        actual_y = word_obj.y + y_offset - padding
        
        # Ensure within bounds for animated position
        y_start = max(0, actual_y)
        y_end = min(frame.shape[0], actual_y + word_bgr.shape[0])
        x_start = max(0, actual_x)
        x_end = min(frame.shape[1], actual_x + word_bgr.shape[1])
        
        if y_end > y_start and x_end > x_start:
            # Calculate sprite region
            sprite_y_start = max(0, -actual_y)
            sprite_y_end = sprite_y_start + (y_end - y_start)
            sprite_x_start = max(0, -actual_x)
            sprite_x_end = sprite_x_start + (x_end - x_start)
            
            sprite_region = word_bgr[sprite_y_start:sprite_y_end, 
                                    sprite_x_start:sprite_x_end]
            
            if sprite_region.shape[2] == 4:
                alpha = sprite_region[:, :, 3].astype(np.float32) / 255.0
                
                # If rendering behind, extract mask and apply only to background
                if word_obj.is_behind:
                    foreground_mask = self.mask_extractor.get_mask_for_frame(frame, self.current_frame_number)
                    
                    # IMPORTANT: For animated text, we need to check the mask at the CURRENT animated position
                    # not the final position, so the text is properly masked during its entire animation
                    mask_region = foreground_mask[y_start:y_end, x_start:x_end]
                    
                    # RVM mask (without thresholding) uses:
                    # - Background: 204 (uniform gray value)
                    # - Person/foreground: < 204 (darker values, variable)
                    # Use threshold of 203 to separate foreground from background
                    bg_mask = (mask_region >= 204).astype(np.float32)
                    # This creates a binary mask: 
                    # - mask < 204: bg_mask = 0 (NO text - person/foreground)
                    # - mask >= 204: bg_mask = 1 (text appears - background)
                    
                    alpha = alpha * bg_mask
                
                for c in range(3):
                    frame[y_start:y_end, x_start:x_end, c] = (
                        frame[y_start:y_end, x_start:x_end, c].astype(np.float32) * (1.0 - alpha) +
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