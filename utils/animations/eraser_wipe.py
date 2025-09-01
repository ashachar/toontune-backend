#!/usr/bin/env python3
"""
Eraser wipe animation for removing character from video
"""

import cv2
import numpy as np
from typing import List, Tuple
import os
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


class EraserWipeAnimation:
    def __init__(self, eraser_image_path: str, duration: float = 0.6, fade_duration: float = 0.2):
        """
        Initialize eraser wipe animation
        
        Args:
            eraser_image_path: Path to eraser PNG image
            duration: Duration of eraser wipe animation in seconds
            fade_duration: Duration of fade-out for remaining pixels
        """
        self.duration = duration
        self.fade_duration = fade_duration
        
        # Load eraser image with alpha channel
        self.eraser_img = cv2.imread(eraser_image_path, cv2.IMREAD_UNCHANGED)
        if self.eraser_img is None:
            raise ValueError(f"Could not load eraser image from {eraser_image_path}")
        
        # If no alpha channel, create one
        if self.eraser_img.shape[2] == 3:
            self.eraser_img = cv2.cvtColor(self.eraser_img, cv2.COLOR_BGR2BGRA)
        
        self.eraser_h, self.eraser_w = self.eraser_img.shape[:2]
        
        # Calculate upward shift (eraser tip is about 1/3 from bottom)
        self.eraser_tip_offset = int(self.eraser_h * 0.33)
    
    def extract_character_mask(self, frame: np.ndarray) -> np.ndarray:
        """Extract character from frame (non-background pixels)"""
        # Simple background removal - assumes lighter background
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def extract_skeleton_path(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract skeleton path from character mask
        Returns ordered list of points to visit
        """
        # Get skeleton
        skeleton = skeletonize(mask > 0)
        
        # Find skeleton points
        skeleton_points = np.column_stack(np.where(skeleton))
        
        if len(skeleton_points) == 0:
            return []
        
        # Order points to create a continuous path
        # Start from top (head area)
        ordered_points = []
        remaining = skeleton_points.tolist()
        
        # Start with topmost point
        current = min(remaining, key=lambda p: p[0])
        ordered_points.append((current[1], current[0]))  # Convert to (x, y)
        remaining.remove(current)
        
        # Greedy nearest neighbor traversal
        while remaining:
            current_array = np.array([current])
            remaining_array = np.array(remaining)
            distances = cdist(current_array, remaining_array)[0]
            nearest_idx = np.argmin(distances)
            current = remaining[nearest_idx]
            ordered_points.append((current[1], current[0]))  # Convert to (x, y)
            remaining.pop(nearest_idx)
        
        return ordered_points
    
    def create_wipe_animation(self, video_path: str, start_time: float, output_path: str) -> str:
        """
        Create eraser wipe animation on video
        
        Args:
            video_path: Path to input video
            start_time: Time in video where wipe starts
            output_path: Path for output video
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range for animation
        start_frame = int(start_time * fps)
        wipe_frames = int(self.duration * fps)
        fade_frames = int(self.fade_duration * fps)
        
        # Get frame at start of wipe for character extraction
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, character_frame = cap.read()
        if not ret:
            raise ValueError("Could not read frame at wipe start time")
        
        # Extract character mask and skeleton path
        character_mask = self.extract_character_mask(character_frame)
        skeleton_path = self.extract_skeleton_path(character_mask)
        
        if not skeleton_path:
            print("Warning: No skeleton path found")
            return video_path
        
        # Scale eraser so bottom is never visible
        # Eraser should be tall enough that bottom is always off-screen
        scale_factor = (height * 1.5) / self.eraser_h
        scaled_eraser_h = int(self.eraser_h * scale_factor)
        scaled_eraser_w = int(self.eraser_w * scale_factor)
        scaled_eraser = cv2.resize(self.eraser_img, (scaled_eraser_w, scaled_eraser_h))
        scaled_tip_offset = int(self.eraser_tip_offset * scale_factor)
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0
        
        # Track which pixels have been erased
        erased_mask = np.zeros((height, width), dtype=bool)
        
        # Create coordinate grids for distance calculations
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num >= start_frame and frame_num < start_frame + wipe_frames:
                # During wipe animation
                progress = (frame_num - start_frame) / wipe_frames
                
                # Calculate how many skeleton points to visit
                points_to_visit = int(len(skeleton_path) * progress)
                
                # Update erased mask based on visited points
                for i in range(min(points_to_visit, len(skeleton_path))):
                    point = skeleton_path[i]
                    
                    # Calculate eraser position (with upward shift for tip)
                    eraser_x = point[0] - scaled_eraser_w // 2
                    eraser_y = point[1] - scaled_tip_offset
                    
                    # Calculate distance from this point to all pixels
                    distances = np.sqrt((x_coords - point[0])**2 + (y_coords - point[1])**2)
                    
                    # Erase pixels within eraser radius
                    erase_radius = scaled_eraser_w // 3  # Adjust for eraser width
                    new_erased = (distances < erase_radius) & (character_mask > 0)
                    erased_mask = erased_mask | new_erased
                
                # Apply erasure to frame
                frame_with_erase = frame.copy()
                
                # Get background (next frame or original background)
                if frame_num + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    next_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + wipe_frames + fade_frames)
                    ret2, background_frame = cap.read()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
                    if ret2:
                        frame_with_erase[erased_mask] = background_frame[erased_mask]
                else:
                    # Make erased pixels transparent/white
                    frame_with_erase[erased_mask] = [255, 255, 255]
                
                # Draw eraser at current position
                if points_to_visit > 0 and points_to_visit <= len(skeleton_path):
                    current_point = skeleton_path[min(points_to_visit - 1, len(skeleton_path) - 1)]
                    eraser_x = current_point[0] - scaled_eraser_w // 2
                    eraser_y = current_point[1] - scaled_tip_offset
                    
                    # Overlay eraser with alpha blending
                    self.overlay_image_alpha(frame_with_erase, scaled_eraser, (eraser_x, eraser_y))
                
                out.write(frame_with_erase)
                
            elif frame_num >= start_frame + wipe_frames and frame_num < start_frame + wipe_frames + fade_frames:
                # Fade out remaining pixels
                fade_progress = (frame_num - start_frame - wipe_frames) / fade_frames
                
                # Get remaining character pixels
                remaining_mask = (character_mask > 0) & (~erased_mask)
                
                # Blend with background
                frame_with_fade = frame.copy()
                if frame_num + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    next_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + wipe_frames + fade_frames)
                    ret2, background_frame = cap.read()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
                    if ret2:
                        alpha = fade_progress
                        frame_with_fade[remaining_mask] = (
                            (1 - alpha) * frame[remaining_mask] + 
                            alpha * background_frame[remaining_mask]
                        ).astype(np.uint8)
                
                out.write(frame_with_fade)
            else:
                # Before or after animation
                out.write(frame)
            
            frame_num += 1
        
        cap.release()
        out.release()
        
        # Convert to H.264
        h264_output = output_path.replace('.mp4', '_h264.mp4')
        cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p {h264_output} -y"
        os.system(cmd)
        os.remove(output_path)
        os.rename(h264_output, output_path)
        
        return output_path
    
    def overlay_image_alpha(self, background: np.ndarray, overlay: np.ndarray, position: Tuple[int, int]):
        """Overlay image with alpha channel on background"""
        x, y = position
        h, w = overlay.shape[:2]
        
        # Calculate valid region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(background.shape[1], x + w)
        y2 = min(background.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        # Adjust overlay region
        overlay_x1 = x1 - x
        overlay_y1 = y1 - y
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)
        
        # Extract alpha channel
        overlay_region = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        if overlay_region.shape[2] == 4:
            alpha = overlay_region[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]
            
            # Blend
            background[y1:y2, x1:x2] = (
                (1.0 - alpha) * background[y1:y2, x1:x2] + 
                alpha * overlay_region[:, :, :3]
            ).astype(np.uint8)