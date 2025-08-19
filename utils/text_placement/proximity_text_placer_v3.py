#!/usr/bin/env python3
"""
ProximityTextPlacerV3: Enhanced text placement that checks multiple frames.
Ensures text remains in background throughout its entire display duration.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
from rembg import remove
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProximityTextPlacerV3:
    def __init__(self, video_path: str, output_dir: str = "output"):
        """Initialize with video path."""
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.release()
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.thickness = 2
        
        # Previous word position for proximity calculation
        self.previous_position = None
        
        print(f"Initialized ProximityTextPlacerV3 for {Path(video_path).name}")
        print(f"  Resolution: {self.width}x{self.height}, FPS: {self.fps}")
    
    def get_background_masks_for_duration(self, start_time: float, end_time: float, 
                                         num_samples: int = 5) -> List[np.ndarray]:
        """Get background masks at multiple points during a time period."""
        masks = []
        timestamps = np.linspace(start_time, end_time, num_samples)
        
        cap = cv2.VideoCapture(self.video_path)
        
        for timestamp in timestamps:
            frame_number = int(timestamp * self.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Could not read frame at {timestamp}s")
                continue
            
            # Apply rembg
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            nobg_img = remove(pil_image)
            nobg_array = np.array(nobg_img)
            
            # Extract alpha channel
            if nobg_array.shape[2] == 4:
                alpha = nobg_array[:, :, 3]
            else:
                alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
            
            # Create background mask: alpha < 128 = background (255), else foreground (0)
            background_mask = np.where(alpha < 128, 255, 0).astype(np.uint8)
            masks.append(background_mask)
        
        cap.release()
        return masks
    
    def check_text_fits_in_all_masks(self, text: str, x: int, y: int, 
                                     font_scale: float, masks: List[np.ndarray]) -> bool:
        """Check if text fits in background across all provided masks."""
        # Get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, font_scale, self.thickness
        )
        
        # FFmpeg drawtext uses TOP-LEFT as reference point
        x1 = x
        y1 = y  # Top of text (FFmpeg convention)
        x2 = x + text_width
        y2 = y + text_height + baseline  # Bottom including baseline
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height:
            return False
        
        # Check all masks
        for mask in masks:
            region = mask[y1:y2, x1:x2]
            if region.size == 0:
                return False
            
            # Calculate background ratio
            bg_ratio = np.mean(region == 255)
            
            # Require at least 95% background across ALL frames
            if bg_ratio < 0.95:
                return False
        
        return True
    
    def find_best_position_multi_frame(self, text: str, start_time: float, 
                                       end_time: float, font_scale: float = 1.0) -> Dict:
        """Find best position checking multiple frames during word display."""
        
        # Get background masks for the entire duration
        masks = self.get_background_masks_for_duration(start_time, end_time, num_samples=5)
        
        if not masks:
            logger.error("Could not get background masks")
            return {"x": self.width // 2, "y": self.height // 4, 
                   "font_scale": font_scale, "fontsize": int(48 * font_scale)}
        
        # Combine all masks - a pixel is background only if it's background in ALL frames
        combined_mask = np.ones_like(masks[0]) * 255
        for mask in masks:
            combined_mask = np.minimum(combined_mask, mask)
        
        # Get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, font_scale, self.thickness
        )
        
        # Grid search parameters
        step_size = 20
        valid_positions = []
        
        # Search for valid positions
        for y in range(10, self.height - text_height - baseline - 10, step_size):
            for x in range(10, self.width - text_width - 10, step_size):
                if self.check_text_fits_in_all_masks(text, x, y, font_scale, masks):
                    valid_positions.append((x, y))
        
        if not valid_positions:
            logger.warning(f"No valid positions found for '{text}' across all frames")
            # Try with smaller font
            if font_scale > 0.5:
                return self.find_best_position_multi_frame(text, start_time, end_time, 
                                                          font_scale * 0.8)
            # Fallback
            return {"x": self.width // 2, "y": self.height // 4,
                   "font_scale": font_scale, "fontsize": int(48 * font_scale)}
        
        # Find position closest to previous word (or center if first word)
        if self.previous_position:
            target_x, target_y = self.previous_position
        else:
            target_x, target_y = self.width // 2, self.height // 2
        
        best_position = min(valid_positions, 
                          key=lambda p: (p[0] - target_x)**2 + (p[1] - target_y)**2)
        
        logger.info(f"Found {len(valid_positions)} valid positions for '{text}'")
        logger.info(f"  Selected: ({best_position[0]}, {best_position[1]})")
        
        return {
            "x": best_position[0],
            "y": best_position[1],
            "font_scale": font_scale,
            "fontsize": int(48 * font_scale),
            "valid_positions_found": len(valid_positions)
        }
    
    def place_word_safe(self, word: str, start_time: float, end_time: float) -> Dict:
        """Place a word safely, checking entire duration."""
        result = self.find_best_position_multi_frame(word, start_time, end_time)
        
        # Update previous position for next word
        self.previous_position = (result["x"], result["y"])
        
        return result
    
    def process_words(self, words: List[Dict]) -> List[Dict]:
        """Process all words with multi-frame checking."""
        results = []
        
        for i, word_data in enumerate(words):
            word = word_data.get("word", "")
            start = word_data.get("start", 0)
            end = word_data.get("end", start + 0.5)
            
            print(f"\nProcessing word {i+1}/{len(words)}: '{word}' ({start:.2f}s - {end:.2f}s)")
            
            position = self.place_word_safe(word, start, end)
            
            result = {
                "word": word,
                "start": start,
                "end": end,
                "x": position["x"],
                "y": position["y"],
                "fontsize": position["fontsize"]
            }
            results.append(result)
            
            print(f"  Placed at ({position['x']}, {position['y']}) with fontsize {position['fontsize']}")
        
        return results


def main():
    """Test the multi-frame text placer."""
    
    # Test with scene 1
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    placer = ProximityTextPlacerV3(video_path, "tests/multi_frame_placement")
    
    # Test with just the problematic words
    test_words = [
        {"word": "Let's", "start": 7.92, "end": 8.56},
        {"word": "start", "start": 8.56, "end": 9.1},
        {"word": "at", "start": 9.1, "end": 9.34},
        {"word": "the", "start": 9.34, "end": 9.579},
        {"word": "very", "start": 9.579, "end": 10.1},
        {"word": "beginning", "start": 10.1, "end": 11.479},  # The problematic word
    ]
    
    results = placer.process_words(test_words)
    
    # Save results
    output_file = Path("tests/multi_frame_placement/word_positions_v3.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Compare with old positions
    print("\nComparison with previous positions:")
    for result in results:
        if result["word"] == "beginning":
            print(f"  'beginning': NEW position ({result['x']}, {result['y']})")
            print(f"              OLD position was (330, 170)")
            if result['y'] != 170:
                print(f"              ✓ Position changed by {abs(result['y'] - 170)}px vertically")


if __name__ == "__main__":
    main()