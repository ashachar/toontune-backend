"""
Foreground masking utilities for word-level pipeline
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
import tempfile
import hashlib


class ForegroundMaskExtractor:
    """Extracts foreground masks from video frames using cached RVM when available"""
    
    def __init__(self, video_path=None):
        """Initialize the mask extractor
        
        Args:
            video_path: Path to the video file for cached mask lookup
        """
        self.video_path = Path(video_path) if video_path else None
        self.cached_mask_video = None
        self.mask_cap = None
        self.current_frame_idx = 0
        
        # Try to find cached RVM mask if video path provided
        if self.video_path:
            self._load_cached_mask()
    
    def _load_cached_mask(self):
        """Load cached RVM mask if available"""
        if not self.video_path or not self.video_path.exists():
            return
        
        # Check for cached RVM mask in video's folder
        video_name = self.video_path.stem
        project_folder = self.video_path.parent / video_name
        
        if project_folder.exists():
            # Look for RVM mask files
            mask_files = list(project_folder.glob(f"{video_name}_rvm_mask_*.mp4"))
            if mask_files:
                # Use the most recent mask file
                self.cached_mask_video = sorted(mask_files, key=lambda x: x.stat().st_mtime)[-1]
                print(f"   🎭 Found cached RVM mask: {self.cached_mask_video.name}")
                
                # Open the mask video
                self.mask_cap = cv2.VideoCapture(str(self.cached_mask_video))
                return
        
        # Check for green screen version (convert to mask on the fly)
        green_files = list(project_folder.glob(f"{video_name}_rvm_green_*.mp4"))
        if green_files:
            self.cached_green_video = sorted(green_files, key=lambda x: x.stat().st_mtime)[-1]
            print(f"   🎭 Found cached RVM green screen: {self.cached_green_video.name}")
            # We'll extract mask from green screen on the fly
            return
    
    def _extract_mask_from_green_screen(self, frame: np.ndarray) -> np.ndarray:
        """Extract mask from a green screen frame
        
        Args:
            frame: Green screen frame
            
        Returns:
            Binary mask where 255 = foreground, 0 = background
        """
        # Convert to HSV for better green detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define green color range
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        # Create mask of green pixels
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Invert to get foreground mask (non-green areas)
        foreground_mask = cv2.bitwise_not(green_mask)
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return foreground_mask
    
    def get_mask_for_frame(self, frame: np.ndarray, frame_number: int = None) -> np.ndarray:
        """Get foreground mask for a specific frame
        
        Args:
            frame: Input frame as BGR numpy array
            frame_number: Frame number in the video (optional, for cached mask sync)
            
        Returns:
            Binary mask where 255 = foreground, 0 = background
        """
        # If we have a cached mask video, use it
        if self.mask_cap is not None and frame_number is not None:
            # Seek to the correct frame
            self.mask_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, mask_frame = self.mask_cap.read()
            if ret:
                # Convert to grayscale if needed
                if len(mask_frame.shape) == 3:
                    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                # Threshold to binary
                _, mask = cv2.threshold(mask_frame, 128, 255, cv2.THRESH_BINARY)
                return mask
        
        # If we have a cached green screen, extract mask from it
        if hasattr(self, 'cached_green_video'):
            # Create a temporary video capture for the green screen
            green_cap = cv2.VideoCapture(str(self.cached_green_video))
            if frame_number is not None:
                green_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, green_frame = green_cap.read()
            green_cap.release()
            
            if ret:
                return self._extract_mask_from_green_screen(green_frame)
        
        # Fallback to basic edge detection (not recommended but keeping for compatibility)
        return self._extract_basic_mask(frame)
    
    def extract_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Extract foreground mask from frame (legacy compatibility)
        
        Args:
            frame: Input frame as BGR numpy array
            
        Returns:
            Binary mask where 255 = foreground, 0 = background
        """
        # Try to use cached mask if available
        return self.get_mask_for_frame(frame, self.current_frame_idx)
    
    def _extract_basic_mask(self, frame: np.ndarray) -> np.ndarray:
        """Basic mask extraction using edge detection (fallback method)
        
        Args:
            frame: Input frame as BGR numpy array
            
        Returns:
            Binary mask where 255 = foreground, 0 = background
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find actual objects
        edges = cv2.Canny(gray, 50, 150)
        
        # Use adaptive thresholding for better foreground detection
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 
                                               11, 2)
        
        # Combine edges and adaptive threshold
        mask = cv2.bitwise_or(edges, adaptive_thresh)
        
        # Clean up noise
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Conservative dilation to ensure foreground coverage
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def generate_rvm_mask_if_needed(self, video_path: Path, duration=None):
        """Generate RVM mask using Replicate if not already cached
        
        Args:
            video_path: Path to video file
            duration: Duration in seconds to process (None for full video)
            
        Returns:
            Path to mask video if generated/found, None otherwise
        """
        from utils.video.background.cached_rvm import CachedRobustVideoMatting
        
        try:
            processor = CachedRobustVideoMatting()
            
            # This will check cache first, only process if needed
            green_screen_path = processor.get_rvm_output(video_path, duration)
            
            # Get the mask path from cache info
            cache_info = processor.get_cache_path(video_path, duration)
            
            if cache_info['mask'].exists():
                return cache_info['mask']
            elif cache_info['green_screen'].exists():
                # Generate mask from green screen if needed
                processor.generate_mask(cache_info['green_screen'], cache_info['mask'])
                return cache_info['mask']
                
        except Exception as e:
            print(f"   ⚠️ Could not generate RVM mask: {e}")
            print("   📌 Falling back to basic edge detection")
        
        return None
    
    def __del__(self):
        """Cleanup video capture on deletion"""
        if self.mask_cap is not None:
            self.mask_cap.release()