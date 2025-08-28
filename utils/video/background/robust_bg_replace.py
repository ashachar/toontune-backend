#!/usr/bin/env python3
"""
Robust background replacement using temporal consistency across entire video.
Uses advanced masking techniques to eliminate white edges.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import tempfile
from tqdm import tqdm
import shutil
from scipy.ndimage import binary_erosion, gaussian_filter


class RobustBackgroundReplacer:
    """Robust background replacement with temporal consistency."""
    
    def __init__(self):
        # Initialize rembg with best model
        print("Initializing high-quality background removal model...")
        self.session = new_session('u2netp')
        self.background_cache = {}
        
    def analyze_background(self, video_path, sample_frames=10):
        """
        Analyze the original video to detect background characteristics.
        
        Args:
            video_path: Path to input video
            sample_frames: Number of frames to sample
            
        Returns:
            Dictionary with background properties
        """
        print("Analyzing original background...")
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
        
        bg_colors = []
        edge_colors = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Get mask for this frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            output = remove(pil_image, session=self.session, only_mask=True)
            mask = np.array(output)
            
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            # Analyze background regions
            bg_mask = mask < 128
            if np.any(bg_mask):
                bg_pixels = frame[bg_mask]
                # Get dominant background color
                bg_color = np.median(bg_pixels, axis=0)
                bg_colors.append(bg_color)
            
            # Analyze edge regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            edge_mask = (dilated > 128) & (eroded < 128)
            
            if np.any(edge_mask):
                edge_pixels = frame[edge_mask]
                edge_color = np.median(edge_pixels, axis=0)
                edge_colors.append(edge_color)
        
        cap.release()
        
        # Compute average colors
        avg_bg_color = np.median(bg_colors, axis=0) if bg_colors else np.array([255, 255, 255])
        avg_edge_color = np.median(edge_colors, axis=0) if edge_colors else avg_bg_color
        
        print(f"Detected background color (BGR): {avg_bg_color}")
        print(f"Detected edge contamination color (BGR): {avg_edge_color}")
        
        return {
            'bg_color': avg_bg_color.astype(np.uint8),
            'edge_color': avg_edge_color.astype(np.uint8),
            'color_threshold': 40  # Threshold for color similarity
        }
    
    def create_temporal_mask(self, video_path, bg_info, output_mask_path):
        """
        Create temporally consistent mask for entire video.
        
        Args:
            video_path: Path to input video
            bg_info: Background information from analyze_background
            output_mask_path: Path to save mask video
        """
        print("Creating temporally consistent masks...")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_mask_path), fourcc, fps, (width, height), isColor=False)
        
        # Process each frame
        prev_mask = None
        
        for frame_idx in tqdm(range(total_frames), desc="Processing masks"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get mask for current frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Use alpha matting for better edges
            output_rgba = remove(pil_image, session=self.session,
                               alpha_matting=True,
                               alpha_matting_foreground_threshold=270,
                               alpha_matting_background_threshold=10,
                               alpha_matting_erode_size=0)
            
            output_np = np.array(output_rgba)
            if output_np.shape[2] == 4:
                mask = output_np[:, :, 3]
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
            
            # Remove edge contamination
            mask = self.remove_color_contamination(frame, mask, bg_info)
            
            # Apply temporal smoothing
            if prev_mask is not None:
                # Blend with previous mask for temporal consistency
                alpha = 0.7  # Current frame weight
                mask = (alpha * mask + (1-alpha) * prev_mask).astype(np.uint8)
            
            prev_mask = mask.copy()
            
            # Final cleanup
            mask = self.cleanup_mask(mask)
            
            # Write mask frame
            out.write(mask)
        
        cap.release()
        out.release()
        
        return output_mask_path
    
    def remove_color_contamination(self, frame, mask, bg_info):
        """
        Remove pixels that match the original background color.
        
        Args:
            frame: Current frame (BGR)
            mask: Current mask
            bg_info: Background information
            
        Returns:
            Cleaned mask
        """
        # Find edge region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        edge_region = (dilated > 128) & (eroded < 200)
        
        if np.any(edge_region):
            # Convert to LAB for better color comparison
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            bg_lab = cv2.cvtColor(bg_info['bg_color'].reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
            edge_lab = cv2.cvtColor(bg_info['edge_color'].reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
            
            # Calculate color distance
            bg_dist = np.sqrt(np.sum((frame_lab.astype(np.float32) - bg_lab.astype(np.float32))**2, axis=2))
            edge_dist = np.sqrt(np.sum((frame_lab.astype(np.float32) - edge_lab.astype(np.float32))**2, axis=2))
            
            # Remove pixels similar to background or edge color
            threshold = bg_info['color_threshold']
            contaminated = edge_region & ((bg_dist < threshold) | (edge_dist < threshold))
            
            # Aggressively remove contaminated pixels
            mask[contaminated] = 0
            
            # Reduce alpha for semi-contaminated pixels
            semi_contaminated = edge_region & ((bg_dist < threshold * 1.5) | (edge_dist < threshold * 1.5))
            mask[semi_contaminated & ~contaminated] = mask[semi_contaminated & ~contaminated] * 0.5
        
        return mask
    
    def cleanup_mask(self, mask):
        """
        Apply final cleanup operations to mask.
        
        Args:
            mask: Input mask
            
        Returns:
            Cleaned mask
        """
        # Fill small holes
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Erode slightly to ensure no background pixels
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.erode(mask, kernel_tiny, iterations=1)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (3, 3), 1)
        
        # Sharpen mask boundaries
        mask = np.where(mask > 200, 255, np.where(mask < 50, 0, mask))
        
        return mask.astype(np.uint8)
    
    def composite_with_background(self, video_path, mask_video_path, background_video_path, output_path):
        """
        Composite foreground with new background using mask video.
        
        Args:
            video_path: Original video
            mask_video_path: Mask video
            background_video_path: New background video
            output_path: Output path
        """
        print("Compositing with new background...")
        
        # Use FFmpeg for final composite with high quality
        cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video_path),  # Background
            "-i", str(video_path),             # Foreground
            "-i", str(mask_video_path),        # Mask
            "-filter_complex",
            "[0:v]scale=1280:720,loop=loop=-1:size=1000[bg];"  # Loop background
            "[1:v]scale=1280:720[fg];"
            "[2:v]scale=1280:720,format=gray,"
            "curves=preset=increase_contrast,"  # Sharpen mask
            "erosion,gblur=sigma=0.5[mask];"    # Final mask processing
            "[bg][fg][mask]maskedmerge[out]",
            "-map", "[out]",
            "-map", "1:a?",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Output saved to: {output_path}")
    
    def process(self, input_video, background_video, output_path):
        """
        Main processing pipeline.
        
        Args:
            input_video: Path to input video
            background_video: Path to background video
            output_path: Path to output video
        """
        temp_dir = tempfile.mkdtemp()
        temp_dir_path = Path(temp_dir)
        
        try:
            # Analyze background
            bg_info = self.analyze_background(input_video)
            
            # Create temporal mask
            mask_video = temp_dir_path / "mask.mp4"
            self.create_temporal_mask(input_video, bg_info, mask_video)
            
            # Composite
            self.composite_with_background(input_video, mask_video, background_video, output_path)
            
        finally:
            # Cleanup
            if temp_dir_path.exists():
                shutil.rmtree(temp_dir)


def main():
    """Test robust background replacement."""
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1_test_5sec.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_robust_bg_replaced.mp4")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Process video
    replacer = RobustBackgroundReplacer()
    replacer.process(input_video, background_video, output_video)
    
    # Open result
    subprocess.run(["open", str(output_video)])


if __name__ == "__main__":
    main()