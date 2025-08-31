"""
Debug mode renderer for word-level pipeline
Shows binary mask and text bounding boxes
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
from dotenv import load_dotenv

from .models import WordObject
from .masking import ForegroundMaskExtractor

# Load debug flag from .env
load_dotenv()
IS_DEBUG_MODE = os.getenv('IS_DEBUG_MODE', 'false').lower() == 'true'


class DebugRenderer:
    """Renders debug visualization with mask and bounding boxes"""
    
    def __init__(self, video_path=None):
        """Initialize debug renderer
        
        Args:
            video_path: Path to video file for mask extraction
        """
        self.mask_extractor = ForegroundMaskExtractor(video_path)
        self.is_enabled = IS_DEBUG_MODE
        
        if self.is_enabled:
            print("🐛 DEBUG MODE ENABLED - Will create debug video with mask overlay")
            if self.mask_extractor and self.mask_extractor.cached_mask_video:
                print(f"   ✅ Mask loaded from: {self.mask_extractor.cached_mask_video}")
            else:
                print(f"   ⚠️ No mask loaded for video: {video_path}")
    
    def render_debug_frame(self, frame: np.ndarray, time_seconds: float,
                           word_objects: List[WordObject], 
                           frame_number: int = None) -> np.ndarray:
        """Render debug visualization frame
        
        Shows:
        - Original frame (top half)
        - Binary foreground mask (bottom left)
        - Text bounding boxes with labels (overlay)
        """
        if not self.is_enabled:
            return None
            
        h, w = frame.shape[:2]
        
        # Create debug canvas (double height)
        debug_frame = np.zeros((h * 2, w, 3), dtype=np.uint8)
        
        # Top half: Original frame
        debug_frame[:h, :] = frame
        
        # Get foreground mask
        mask_found = False
        if self.mask_extractor and frame_number is not None:
            self.mask_extractor.current_frame_idx = frame_number
            mask_frame = self.mask_extractor.get_mask_for_frame(frame, frame_number)
            
            if mask_frame is not None and len(mask_frame.shape) == 3:
                # Extract binary mask from green screen
                TARGET_GREEN_BGR = np.array([154, 254, 119], dtype=np.float32)
                diff = mask_frame.astype(np.float32) - TARGET_GREEN_BGR
                distance = np.sqrt(np.sum(diff * diff, axis=2))
                is_green = (distance < 50)
                
                # Create foreground mask (inverted - person is white)
                fg_mask = (~is_green).astype(np.uint8) * 255
                
                # Apply dilation to show the expanded mask used in rendering
                kernel = np.ones((5, 5), np.uint8)
                fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
                
                # Convert to 3-channel for display
                fg_mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                
                # Bottom left: Binary mask
                debug_frame[h:, :w//2] = fg_mask_colored[:, :w//2]
                
                # Bottom right: Mask overlay on frame
                overlay = frame.copy()
                # Tint the foreground areas red
                red_overlay = overlay.copy()
                red_overlay[:, :, 2] = np.where(fg_mask > 128, 255, red_overlay[:, :, 2])
                overlay = cv2.addWeighted(overlay, 0.7, red_overlay, 0.3, 0)
                debug_frame[h:, w//2:] = overlay[:, w//2:]
                mask_found = True
        
        # If no mask found, show placeholder
        if not mask_found:
            # Add dim copy of frame to bottom
            dim_frame = (frame * 0.3).astype(np.uint8)
            debug_frame[h:, :] = dim_frame
            
            # Add "NO MASK" text
            cv2.putText(debug_frame, "NO MASK AVAILABLE", 
                       (w//4, h + h//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (100, 100, 255), 3)
        
        # Draw text bounding boxes and labels
        for word_obj in word_objects:
            # Skip if word hasn't started yet
            animation_start = word_obj.start_time - word_obj.rise_duration
            if time_seconds < animation_start:
                continue
            
            # Determine color based on is_behind flag
            if word_obj.is_behind:
                box_color = (255, 100, 100)  # Light blue for behind text
                label_bg = (200, 100, 50)    # Darker blue
            else:
                box_color = (100, 255, 100)  # Light green for front text
                label_bg = (50, 200, 50)     # Darker green
            
            # Draw bounding box on top frame
            x, y = word_obj.x, word_obj.y
            w, h = word_obj.width, word_obj.height
            
            # Draw rectangle
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw label with text and properties
            label = f"{word_obj.text}"
            if word_obj.is_behind:
                label += " [B]"  # Behind indicator
            
            # Calculate label size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(debug_frame, 
                         (x, y - label_h - 4),
                         (x + label_w + 4, y),
                         label_bg, -1)
            
            # Draw label text
            cv2.putText(debug_frame, label, 
                       (x + 2, y - 2),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Add timing info in bottom left
            timing_text = f"t={word_obj.start_time:.1f}s"
            cv2.putText(debug_frame, timing_text,
                       (x, y + h + 12),
                       font, 0.3, box_color, 1)
        
        # Add frame info overlay
        info_text = [
            f"Time: {time_seconds:.2f}s",
            f"Frame: {frame_number if frame_number else 'N/A'}",
            f"Active words: {len([w for w in word_objects if w.start_time - w.rise_duration <= time_seconds])}",
            "Legend: [B]=Behind, Green=Front, Blue=Behind"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(debug_frame, text,
                       (10, h + y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_frame, text,
                       (10, h + y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return debug_frame
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled"""
        return self.is_enabled