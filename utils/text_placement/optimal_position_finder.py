#!/usr/bin/env python3
"""
Find optimal text placement position by analyzing occlusion across frames.
The algorithm finds the position where text will be maximally visible throughout its animation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='[TEXT_PLACEMENT] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PlacementScore:
    """Score for a potential text placement position."""
    position: Tuple[int, int]  # Center position
    visibility_score: float    # Average visibility (0-1)
    min_visibility: float      # Minimum visibility across frames
    occlusion_frames: int       # Number of frames with >50% occlusion
    

class OptimalTextPositionFinder:
    """
    Find optimal text placement by analyzing occlusion patterns across video frames.
    """
    
    def __init__(
        self,
        text_width: int,
        text_height: int,
        motion_frames: int = 22,
        sample_rate: int = 3,  # Sample every N frames for efficiency
        grid_divisions: int = 8,  # Divide screen into NxN grid for candidate positions
        debug: bool = False
    ):
        """
        Initialize position finder.
        
        Args:
            text_width: Expected width of text bounding box
            text_height: Expected height of text bounding box
            motion_frames: Number of frames for motion animation
            sample_rate: Sample every N frames (higher = faster but less accurate)
            grid_divisions: Grid resolution for candidate positions
            debug: Enable debug logging
        """
        self.text_width = text_width
        self.text_height = text_height
        self.motion_frames = motion_frames
        self.sample_rate = sample_rate
        self.grid_divisions = grid_divisions
        self.debug = debug
        
    def extract_foreground_masks(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract foreground masks from video frames.
        
        Returns list of binary masks where 255 = foreground, 0 = background.
        """
        try:
            from utils.segmentation.segment_extractor import extract_foreground_mask
        except ImportError:
            logger.warning("Segment extractor not available, using simple edge detection")
            extract_foreground_mask = self._simple_foreground_extraction
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit frames to analyze
        frames_to_analyze = min(
            max_frames or self.motion_frames * 2,
            total_frames
        )
        
        masks = []
        frame_indices = range(0, frames_to_analyze, self.sample_rate)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract foreground mask
            try:
                mask = extract_foreground_mask(frame_rgb)
                # Ensure binary mask
                mask = (mask > 128).astype(np.uint8) * 255
            except Exception as e:
                if self.debug:
                    logger.warning(f"Failed to extract mask for frame {idx}: {e}")
                mask = self._simple_foreground_extraction(frame_rgb)
                
            masks.append(mask)
            
            if self.debug and idx % 10 == 0:
                logger.info(f"Processed frame {idx}/{frames_to_analyze}")
                
        cap.release()
        return masks
        
    def _simple_foreground_extraction(self, frame: np.ndarray) -> np.ndarray:
        """
        Simple fallback foreground extraction using edge detection and thresholding.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations to fill regions
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=2)
        
        # Find contours and create mask
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        
        # Fill large contours (likely foreground objects)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
        return mask
        
    def calculate_visibility_score(
        self,
        masks: List[np.ndarray],
        center_x: int,
        center_y: int,
        frame_height: int,
        frame_width: int
    ) -> PlacementScore:
        """
        Calculate visibility score for a text position across all masks.
        
        The score considers:
        1. Average visibility across frames
        2. Minimum visibility (worst case)
        3. Number of heavily occluded frames
        """
        # Calculate bounding box
        x1 = max(0, center_x - self.text_width // 2)
        y1 = max(0, center_y - self.text_height // 2)
        x2 = min(frame_width, x1 + self.text_width)
        y2 = min(frame_height, y1 + self.text_height)
        
        visibility_scores = []
        occlusion_count = 0
        
        for mask in masks:
            # Extract region where text would be
            text_region = mask[y1:y2, x1:x2]
            
            if text_region.size == 0:
                visibility_scores.append(0.0)
                occlusion_count += 1
                continue
                
            # Calculate visibility (percentage of non-occluded pixels)
            total_pixels = text_region.size
            background_pixels = np.sum(text_region == 0)
            visibility = background_pixels / total_pixels
            
            visibility_scores.append(visibility)
            
            # Count heavily occluded frames
            if visibility < 0.5:
                occlusion_count += 1
                
        if not visibility_scores:
            return PlacementScore(
                position=(center_x, center_y),
                visibility_score=0.0,
                min_visibility=0.0,
                occlusion_frames=len(masks)
            )
            
        return PlacementScore(
            position=(center_x, center_y),
            visibility_score=np.mean(visibility_scores),
            min_visibility=np.min(visibility_scores),
            occlusion_frames=occlusion_count
        )
        
    def find_optimal_position(
        self,
        video_path: str,
        prefer_center: bool = True,
        center_weight: float = 0.1
    ) -> Tuple[int, int]:
        """
        Find the optimal text position for maximum visibility.
        
        Args:
            video_path: Path to input video
            prefer_center: Whether to prefer positions closer to center
            center_weight: Weight for center preference (0-1)
            
        Returns:
            Tuple of (x, y) for optimal text center position
        """
        logger.info(f"Analyzing video for optimal text placement...")
        logger.info(f"Text size: {self.text_width}x{self.text_height}")
        
        # Extract foreground masks
        masks = self.extract_foreground_masks(video_path)
        if not masks:
            logger.warning("No masks extracted, using center position")
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (width // 2, height // 2)
            
        frame_height, frame_width = masks[0].shape
        logger.info(f"Frame size: {frame_width}x{frame_height}")
        logger.info(f"Analyzing {len(masks)} frames...")
        
        # Generate candidate positions (grid-based)
        candidates = []
        
        # Calculate grid positions
        margin_x = self.text_width // 2 + 20
        margin_y = self.text_height // 2 + 20
        
        x_positions = np.linspace(
            margin_x,
            frame_width - margin_x,
            self.grid_divisions
        ).astype(int)
        
        y_positions = np.linspace(
            margin_y,
            frame_height - margin_y,
            self.grid_divisions
        ).astype(int)
        
        # Evaluate each candidate position
        best_score = None
        best_position = (frame_width // 2, frame_height // 2)
        
        total_positions = len(x_positions) * len(y_positions)
        position_count = 0
        
        for x in x_positions:
            for y in y_positions:
                position_count += 1
                
                if self.debug and position_count % 10 == 0:
                    logger.info(f"Evaluating position {position_count}/{total_positions}")
                    
                score = self.calculate_visibility_score(
                    masks, x, y, frame_height, frame_width
                )
                
                # Apply center preference if enabled
                if prefer_center:
                    center_x, center_y = frame_width // 2, frame_height // 2
                    distance_from_center = np.sqrt(
                        (x - center_x) ** 2 + (y - center_y) ** 2
                    )
                    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
                    center_score = 1.0 - (distance_from_center / max_distance)
                    
                    # Combined score with center weighting
                    combined_score = (
                        score.visibility_score * (1 - center_weight) +
                        center_score * center_weight
                    )
                else:
                    combined_score = score.visibility_score
                    
                # Penalize positions with very low minimum visibility
                if score.min_visibility < 0.2:
                    combined_score *= 0.5
                    
                # Track best position
                if best_score is None or combined_score > best_score:
                    best_score = combined_score
                    best_position = (x, y)
                    best_placement_score = score
                    
        logger.info(f"Optimal position found: {best_position}")
        logger.info(f"Visibility score: {best_placement_score.visibility_score:.2%}")
        logger.info(f"Minimum visibility: {best_placement_score.min_visibility:.2%}")
        logger.info(f"Frames with >50% occlusion: {best_placement_score.occlusion_frames}/{len(masks)}")
        
        # Create visualization if debug enabled
        if self.debug:
            self._create_debug_visualization(
                masks[0], best_position, frame_width, frame_height
            )
            
        return best_position
        
    def _create_debug_visualization(
        self,
        mask: np.ndarray,
        position: Tuple[int, int],
        frame_width: int,
        frame_height: int
    ):
        """Create debug visualization showing optimal position."""
        # Create color visualization
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Draw text bounding box
        x1 = position[0] - self.text_width // 2
        y1 = position[1] - self.text_height // 2
        x2 = x1 + self.text_width
        y2 = y1 + self.text_height
        
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, position, 5, (0, 0, 255), -1)
        
        # Add text label
        cv2.putText(
            vis, f"Optimal: {position}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )
        
        # Save visualization
        cv2.imwrite("optimal_text_position_debug.png", vis)
        logger.info("Debug visualization saved to optimal_text_position_debug.png")


def estimate_text_size(
    text: str,
    font_size: int,
    scale: float = 1.0
) -> Tuple[int, int]:
    """
    Estimate text bounding box size.
    
    Returns: (width, height)
    """
    # Rough estimation based on font size and text length
    # These are approximations - adjust based on your font
    char_width = font_size * 0.7  # Average character width
    text_width = int(len(text) * char_width * scale)
    text_height = int(font_size * 1.2 * scale)  # Include some padding
    
    return text_width, text_height