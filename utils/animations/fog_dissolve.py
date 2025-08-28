"""
Clean Fog Dissolve Effect - No artifacts, letters stay in place
Only the fog/smudge effect is randomized per letter
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List
from scipy.ndimage import gaussian_filter
import random

class CleanFogDissolve:
    """
    Creates fog dissolve without artifacts or letter movement
    Letters stay exactly in place, only fog effect varies
    """
    
    def __init__(self, font_size=55, color=(255, 255, 255)):
        self.font_size = font_size
        self.color = color
        
        # Load font
        try:
            self.font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
        except:
            self.font = ImageFont.load_default()
        
        # Cache for letter positions and random parameters
        self.letter_cache = {}
    
    def get_letter_positions(self, text: str, center_position: Tuple[int, int]):
        """Calculate exact position for each letter - these NEVER change"""
        if text in self.letter_cache:
            return self.letter_cache[text]
        
        # Create temp image for measurements
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        # Get total text dimensions
        text_bbox = draw.textbbox((0, 0), text, font=self.font)
        total_width = text_bbox[2] - text_bbox[0]
        total_height = text_bbox[3] - text_bbox[1]
        
        # Calculate starting position (centered)
        start_x = center_position[0] - total_width // 2
        start_y = center_position[1] - total_height // 2
        
        letter_data = []
        current_x = start_x
        
        for i, char in enumerate(text):
            # Get character dimensions
            char_bbox = draw.textbbox((0, 0), char, font=self.font)
            char_width = char_bbox[2] - char_bbox[0]
            char_height = char_bbox[3] - char_bbox[1]
            
            # Store fixed position and random fog parameters
            letter_data.append({
                'char': char,
                'x': current_x,  # Fixed X position
                'y': start_y,    # Fixed Y position
                'width': char_width,
                'height': char_height,
                # Random fog parameters (but NOT position changes!)
                'blur_x': random.uniform(0.8, 1.2),  # Horizontal blur variation
                'blur_y': random.uniform(0.8, 1.2),  # Vertical blur variation
                'fog_density': random.uniform(0.9, 1.1),  # Fog opacity variation
                'dissolve_speed': random.uniform(0.95, 1.05),  # Slight timing variation
            })
            
            current_x += char_width
        
        self.letter_cache[text] = letter_data
        return letter_data
    
    def apply_fog_to_letter(self, letter_img: np.ndarray, params: dict, progress: float) -> np.ndarray:
        """Apply fog effect to a single letter without moving it"""
        if progress <= 0:
            return letter_img
        
        result = letter_img.copy()
        
        # Adjust progress with letter's dissolve speed
        adjusted_progress = min(1.0, progress * params['dissolve_speed'])
        
        # Phase 1: Progressive blur (0.0 - 0.6)
        if adjusted_progress > 0:
            blur_amount = adjusted_progress * 15
            
            # Apply directional blur with random variations
            blur_x = blur_amount * params['blur_x']
            blur_y = blur_amount * params['blur_y']
            
            if blur_x > 0 or blur_y > 0:
                # Use Gaussian blur with different x/y values
                result = cv2.GaussianBlur(result, (0, 0), 
                                         sigmaX=blur_x, sigmaY=blur_y)
        
        # Phase 2: Fog texture overlay (0.3 - 0.8)
        if adjusted_progress > 0.3:
            fog_progress = (adjusted_progress - 0.3) / 0.5
            
            # Create fog texture
            h, w = result.shape[:2]
            fog = np.random.randn(h, w) * 20 * fog_progress
            fog = gaussian_filter(fog, sigma=3)
            
            # Apply fog to alpha channel only
            if result.shape[2] == 4:
                alpha = result[:, :, 3].astype(np.float32)
                alpha = alpha * (1.0 - fog_progress * 0.5)  # Reduce opacity
                alpha = np.clip(alpha + fog * params['fog_density'], 0, 255)
                result[:, :, 3] = alpha.astype(np.uint8)
        
        # Phase 3: Final fade (0.6 - 1.0)
        if adjusted_progress > 0.6:
            fade_progress = (adjusted_progress - 0.6) / 0.4
            fade_amount = 1.0 - (fade_progress * 0.9)  # Keep slight visibility
            
            if result.shape[2] == 4:
                result[:, :, 3] = (result[:, :, 3] * fade_amount).astype(np.uint8)
        
        return result
    
    def apply_clean_fog(self, text: str, frame: np.ndarray, position: Tuple[int, int], 
                       progress: float) -> np.ndarray:
        """
        Apply fog dissolve with letters staying at exact positions
        """
        if progress <= 0:
            # Render normal text
            return self.render_normal_text(text, frame, position)
        
        if progress >= 1:
            return frame
        
        # Get fixed letter positions
        letter_data = self.get_letter_positions(text, position)
        
        # Process each letter at its FIXED position
        for letter_info in letter_data:
            if letter_info['char'] == ' ':
                continue  # Skip spaces
            
            # Create letter image (larger for blur overflow)
            padding = 50
            letter_canvas = Image.new('RGBA', 
                                     (letter_info['width'] + padding*2, 
                                      letter_info['height'] + padding*2), 
                                     (0, 0, 0, 0))
            draw = ImageDraw.Draw(letter_canvas)
            
            # Draw letter in center of canvas
            draw.text((padding, padding), letter_info['char'], 
                     fill=(*self.color, 255), font=self.font)
            
            # Convert to numpy
            letter_array = np.array(letter_canvas)
            
            # Apply fog effect (without position change!)
            fogged = self.apply_fog_to_letter(letter_array, letter_info, progress)
            
            # Convert to BGR for OpenCV (fixing potential green artifacts)
            letter_bgr = np.zeros_like(fogged)
            if fogged.shape[2] == 4:
                # Properly convert RGBA to BGRA
                letter_bgr[:, :, 0] = fogged[:, :, 2]  # B = R
                letter_bgr[:, :, 1] = fogged[:, :, 1]  # G = G
                letter_bgr[:, :, 2] = fogged[:, :, 0]  # R = B
                letter_bgr[:, :, 3] = fogged[:, :, 3]  # A = A
            else:
                # RGB without alpha
                letter_bgr[:, :, 0] = fogged[:, :, 2]
                letter_bgr[:, :, 1] = fogged[:, :, 1]
                letter_bgr[:, :, 2] = fogged[:, :, 0]
            
            # Place at EXACT fixed position (no movement!)
            x_pos = letter_info['x'] - padding
            y_pos = letter_info['y'] - padding
            
            # Ensure within frame bounds
            y_start = max(0, y_pos)
            y_end = min(frame.shape[0], y_pos + letter_bgr.shape[0])
            x_start = max(0, x_pos)
            x_end = min(frame.shape[1], x_pos + letter_bgr.shape[1])
            
            # Calculate sprite region
            sprite_y_start = max(0, -y_pos)
            sprite_y_end = sprite_y_start + (y_end - y_start)
            sprite_x_start = max(0, -x_pos)
            sprite_x_end = sprite_x_start + (x_end - x_start)
            
            # Composite with proper alpha blending
            if y_end > y_start and x_end > x_start:
                sprite_region = letter_bgr[sprite_y_start:sprite_y_end, 
                                          sprite_x_start:sprite_x_end]
                
                if sprite_region.shape[2] == 4:
                    alpha = sprite_region[:, :, 3].astype(np.float32) / 255.0
                    
                    # Proper alpha blending to avoid artifacts
                    for c in range(3):
                        frame[y_start:y_end, x_start:x_end, c] = (
                            frame[y_start:y_end, x_start:x_end, c].astype(np.float32) * (1.0 - alpha) +
                            sprite_region[:, :, c].astype(np.float32) * alpha
                        ).astype(np.uint8)
        
        return frame
    
    def render_normal_text(self, text: str, frame: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Render clear text without effects"""
        # Get letter positions
        letter_data = self.get_letter_positions(text, position)
        
        for letter_info in letter_data:
            if letter_info['char'] == ' ':
                continue
            
            # Create letter image
            letter_canvas = Image.new('RGBA', 
                                     (letter_info['width'], letter_info['height']), 
                                     (0, 0, 0, 0))
            draw = ImageDraw.Draw(letter_canvas)
            draw.text((0, 0), letter_info['char'], 
                     fill=(*self.color, 255), font=self.font)
            
            # Convert to numpy and BGR
            letter_array = np.array(letter_canvas)
            letter_bgr = np.zeros_like(letter_array)
            letter_bgr[:, :, 0] = letter_array[:, :, 2]
            letter_bgr[:, :, 1] = letter_array[:, :, 1]
            letter_bgr[:, :, 2] = letter_array[:, :, 0]
            letter_bgr[:, :, 3] = letter_array[:, :, 3]
            
            # Place at exact position
            x_pos = letter_info['x']
            y_pos = letter_info['y']
            
            y_start = max(0, y_pos)
            y_end = min(frame.shape[0], y_pos + letter_bgr.shape[0])
            x_start = max(0, x_pos)
            x_end = min(frame.shape[1], x_pos + letter_bgr.shape[1])
            
            if y_end > y_start and x_end > x_start:
                alpha = letter_bgr[:y_end-y_pos, :x_end-x_pos, 3] / 255.0
                
                for c in range(3):
                    frame[y_start:y_end, x_start:x_end, c] = (
                        frame[y_start:y_end, x_start:x_end, c] * (1.0 - alpha) +
                        letter_bgr[:y_end-y_pos, :x_end-x_pos, c] * alpha
                    ).astype(np.uint8)
        
        return frame


def test_clean_fog():
    """Test the clean fog dissolve without artifacts"""
    
    print("Testing Clean Fog Dissolve (No Artifacts)")
    print("=" * 60)
    print("Features:")
    print("  • NO green artifacts")
    print("  • Letters stay at EXACT positions")
    print("  • Only fog effect is randomized")
    print("  • Clean alpha blending")
    print()
    
    input_video = "outputs/ai_math1_5sec.mp4"
    output_path = "outputs/clean_fog_test.mp4"
    
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create clean fog effect
    fog_effect = CleanFogDissolve(font_size=55)
    
    # Test with fixed positions
    test_sentences = [
        ("DISCOVER THE POWER", (640, 360)),  # Fixed center position
        ("TRANSFORM YOUR WORLD", (640, 360))
    ]
    
    print("Rendering clean fog animation...")
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        time_seconds = frame_num / fps
        
        # First sentence: visible 0-2s, fog dissolve 2-3s
        if time_seconds < 3:
            if time_seconds < 2:
                progress = 0
            else:
                progress = (time_seconds - 2)
            
            frame = fog_effect.apply_clean_fog(
                test_sentences[0][0], frame, test_sentences[0][1], progress
            )
        
        # Second sentence: visible 3-4s, fog dissolve 4-5s
        if time_seconds >= 3:
            if time_seconds < 4:
                progress = 0
            else:
                progress = (time_seconds - 4)
            
            frame = fog_effect.apply_clean_fog(
                test_sentences[1][0], frame, test_sentences[1][1], progress
            )
        
        cv2.putText(frame, f"Clean Fog (No Artifacts) | {time_seconds:.1f}s", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        out.write(frame)
        
        if frame_num % 25 == 0:
            print(f"  Frame {frame_num}/{total_frames}")
    
    out.release()
    cap.release()
    
    # Convert to H.264
    import os
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    os.system(f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y 2>/dev/null")
    os.remove(output_path)
    
    print(f"\n✅ Clean fog video created: {h264_output}")
    print("\nFixed issues:")
    print("  ✓ NO green artifacts - proper BGR conversion")
    print("  ✓ Letters stay at EXACT fixed positions")
    print("  ✓ Random fog variations are subtle")
    print("  ✓ Clean, professional appearance")
    
    return h264_output


if __name__ == "__main__":
    print("CLEAN FOG DISSOLVE EFFECT")
    print("=" * 60)
    print("Fixed version with no artifacts or letter movement")
    print()
    
    output = test_clean_fog()