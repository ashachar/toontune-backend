#!/usr/bin/env python3
"""
Face-aware text placement utilities
Detects faces in video frames and ensures text avoids them
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FaceRegion:
    """Represents a detected face region"""
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    
    @property
    def left(self) -> int:
        return self.x
    
    @property
    def right(self) -> int:
        return self.x + self.width
    
    @property
    def top(self) -> int:
        return self.y
    
    @property
    def bottom(self) -> int:
        return self.y + self.height


class FaceDetector:
    """Detects faces in frames using OpenCV's Haar Cascade"""
    
    def __init__(self):
        # Initialize face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Cache for face detections across frames
        self._face_cache: Dict[int, List[FaceRegion]] = {}
    
    def detect_faces(self, frame: np.ndarray, cache_key: Optional[int] = None) -> List[FaceRegion]:
        """
        Detect faces in a frame
        
        Args:
            frame: BGR video frame
            cache_key: Optional cache key for this frame
            
        Returns:
            List of detected face regions
        """
        # Check cache first
        if cache_key is not None and cache_key in self._face_cache:
            return self._face_cache[cache_key]
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces - adjust parameters for better detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )
        
        # Convert to FaceRegion objects
        face_regions = []
        for (x, y, w, h) in faces:
            # Expand face region slightly for safety margin
            margin = int(w * 0.2)  # 20% margin
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = w + (margin * 2)
            h = h + (margin * 2)
            
            face_regions.append(FaceRegion(
                x=x,
                y=y,
                width=w,
                height=h,
                center_x=x + w // 2,
                center_y=y + h // 2
            ))
        
        # Cache results if key provided
        if cache_key is not None:
            self._face_cache[cache_key] = face_regions
        
        return face_regions
    
    def get_safe_x_positions(self, faces: List[FaceRegion], frame_width: int, 
                            text_width: int) -> List[Tuple[int, float]]:
        """
        Get safe X positions for text that avoid faces
        
        Args:
            faces: List of detected face regions
            frame_width: Width of the video frame
            text_width: Width of the text to place
            
        Returns:
            List of (x_position, safety_score) tuples, sorted by safety
        """
        safe_positions = []
        
        if not faces:
            # No faces detected, center is safe
            center_x = (frame_width - text_width) // 2
            safe_positions.append((center_x, 1.0))
        else:
            # For each face, calculate safe positions
            for face in faces:
                # Position 1: Between left edge and face (STRONGLY PREFERRED)
                if face.left > text_width + 20:  # Ensure enough space
                    # Place text centered between left edge and face
                    # Position is where text STARTS (left edge of text)
                    left_safe_x = (face.left - text_width) // 2
                    # Safety score based on distance from face
                    distance = face.left - (left_safe_x + text_width)
                    # Strong preference for left side (score 2.0+ to always win)
                    safety = min(1.0, distance / 100.0) + 2.0  # Strong preference bonus
                    safe_positions.append((left_safe_x, safety))
                    print(f"       DEBUG: Left safe position for face at {face.left}: x={left_safe_x} (text ends at {left_safe_x + text_width})")
                
                # Position 2: Between face and right edge (ONLY IF LEFT NOT POSSIBLE)
                # Only add right position if there's NO left position available
                if (frame_width - face.right > text_width + 20) and (face.left <= text_width + 20):
                    # Place text centered between face and right edge
                    right_safe_x = face.right + (frame_width - face.right - text_width) // 2
                    # Safety score based on distance from face
                    distance = right_safe_x - face.right
                    safety = min(1.0, distance / 100.0)  # Base score only (max 1.0)
                    safe_positions.append((right_safe_x, safety))
                    print(f"       DEBUG: Right safe position (no left available) for face at {face.right}: x={right_safe_x}")
            
            # If multiple faces, check gaps between them
            if len(faces) > 1:
                sorted_faces = sorted(faces, key=lambda f: f.left)
                for i in range(len(sorted_faces) - 1):
                    gap_start = sorted_faces[i].right
                    gap_end = sorted_faces[i + 1].left
                    gap_width = gap_end - gap_start
                    
                    if gap_width > text_width + 40:  # Minimum gap with margins
                        gap_x = gap_start + (gap_width - text_width) // 2
                        # Higher safety score for positions between faces
                        safety = 0.8
                        safe_positions.append((gap_x, safety))
        
        # Sort by safety score (highest first)
        safe_positions.sort(key=lambda x: x[1], reverse=True)
        
        return safe_positions
    
    def is_position_safe(self, x: int, y: int, width: int, height: int,
                        faces: List[FaceRegion], threshold: float = 0.1) -> bool:
        """
        Check if a text position overlaps with any face
        
        Args:
            x, y: Top-left position of text
            width, height: Dimensions of text
            faces: List of detected faces
            threshold: Maximum allowed overlap ratio (0.1 = 10%)
            
        Returns:
            True if position is safe (minimal face overlap)
        """
        text_area = width * height
        
        for face in faces:
            # Calculate intersection
            intersect_left = max(x, face.left)
            intersect_top = max(y, face.top)
            intersect_right = min(x + width, face.right)
            intersect_bottom = min(y + height, face.bottom)
            
            if intersect_right > intersect_left and intersect_bottom > intersect_top:
                # There's an intersection
                intersect_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
                overlap_ratio = intersect_area / text_area
                
                if overlap_ratio > threshold:
                    return False
        
        return True


def get_face_aware_x_position(frame: np.ndarray, text_width: int, 
                              detector: Optional[FaceDetector] = None) -> int:
    """
    Get the best X position for text that avoids faces
    
    Args:
        frame: Video frame
        text_width: Width of text to place
        detector: Optional pre-initialized FaceDetector
        
    Returns:
        Best X position for the text
    """
    if detector is None:
        detector = FaceDetector()
    
    faces = detector.detect_faces(frame)
    frame_width = frame.shape[1]
    
    safe_positions = detector.get_safe_x_positions(faces, frame_width, text_width)
    
    if safe_positions:
        # Return the safest position
        return safe_positions[0][0]
    else:
        # Fallback to center if no safe position found
        return (frame_width - text_width) // 2