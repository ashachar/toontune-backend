#!/usr/bin/env python3
"""
Advanced background replacement using FFmpeg filters and rembg.
Processes video in chunks for better temporal consistency.
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


class FFmpegBackgroundReplacer:
    """Background replacement with FFmpeg-based temporal filtering."""
    
    def __init__(self):
        print("Initializing advanced background removal...")
        self.session = new_session('u2netp')  # Best quality model
        
    def extract_masks(self, input_video, output_mask_dir, max_duration=5.0):
        """
        Extract masks for all frames with temporal consistency.
        
        Args:
            input_video: Path to input video
            output_mask_dir: Directory to save mask frames
            max_duration: Maximum duration to process
            
        Returns:
            Number of frames processed
        """
        print("Extracting masks with temporal consistency...")
        
        cap = cv2.VideoCapture(str(input_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = min(int(fps * max_duration), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        # Process frames and collect masks
        masks = []
        frames_processed = 0
        
        for frame_idx in tqdm(range(total_frames), desc="Extracting masks"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Extract mask with alpha matting
            output = remove(pil_image, session=self.session,
                          alpha_matting=True,
                          alpha_matting_foreground_threshold=270,
                          alpha_matting_background_threshold=0,
                          alpha_matting_erode_size=0,
                          only_mask=True)
            
            mask = np.array(output)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            masks.append(mask)
            frames_processed += 1
        
        cap.release()
        
        # Apply temporal filtering to masks
        print("Applying temporal filtering to masks...")
        filtered_masks = self.temporal_filter_masks(masks)
        
        # Save masks
        print("Saving filtered masks...")
        for idx, mask in enumerate(filtered_masks):
            mask_path = output_mask_dir / f"mask_{idx:05d}.png"
            cv2.imwrite(str(mask_path), mask)
        
        return frames_processed
    
    def temporal_filter_masks(self, masks):
        """
        Apply temporal filtering to reduce flickering.
        
        Args:
            masks: List of mask arrays
            
        Returns:
            List of filtered masks
        """
        if len(masks) < 3:
            return masks
        
        filtered = []
        
        for i in range(len(masks)):
            if i == 0:
                # First frame: average with next
                filtered_mask = (masks[i].astype(float) * 0.7 + 
                               masks[i+1].astype(float) * 0.3)
            elif i == len(masks) - 1:
                # Last frame: average with previous
                filtered_mask = (masks[i].astype(float) * 0.7 + 
                               masks[i-1].astype(float) * 0.3)
            else:
                # Middle frames: weighted average with neighbors
                filtered_mask = (masks[i-1].astype(float) * 0.2 + 
                               masks[i].astype(float) * 0.6 + 
                               masks[i+1].astype(float) * 0.2)
            
            # Apply aggressive edge cleanup
            filtered_mask = self.cleanup_mask_edges(filtered_mask.astype(np.uint8))
            filtered.append(filtered_mask)
        
        return filtered
    
    def cleanup_mask_edges(self, mask):
        """
        Aggressively clean mask edges to remove white borders.
        
        Args:
            mask: Input mask
            
        Returns:
            Cleaned mask
        """
        # Strong erosion to remove edge contamination
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_eroded = cv2.erode(mask, kernel_erode, iterations=1)
        
        # Slight dilation back
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask_eroded, kernel_dilate, iterations=1)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 2)
        
        # Strong threshold to ensure clean separation
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def composite_with_ffmpeg(self, input_video, mask_dir, background_video, 
                            output_path, num_frames, fps=30):
        """
        Use FFmpeg to composite video with masks.
        
        Args:
            input_video: Original video
            mask_dir: Directory with mask frames
            background_video: Background video
            output_path: Output path
            num_frames: Number of frames
            fps: Frame rate
        """
        print("Compositing with FFmpeg...")
        
        # Create mask video from frames
        mask_video = mask_dir / "mask.mp4"
        
        # Convert mask frames to video
        mask_pattern = str(mask_dir / "mask_%05d.png")
        mask_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", mask_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "gray",
            "-crf", "0",  # Lossless
            str(mask_video)
        ]
        subprocess.run(mask_cmd, check=True, capture_output=True)
        
        # Composite using FFmpeg with advanced filters
        composite_cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video),  # Background
            "-i", str(input_video),       # Foreground
            "-i", str(mask_video),        # Mask
            "-filter_complex",
            "[0:v]scale=1280:720,loop=loop=-1:size=1000[bg];"
            "[1:v]scale=1280:720[fg];"
            "[2:v]scale=1280:720,format=gray,"
            "erosion,"                     # Edge cleanup
            "erosion,"                     # More aggressive
            "dilation,"                    # Slight expansion back
            "gblur=sigma=2,"               # Smooth transitions
            "curves=preset=increase_contrast[mask];"  # Sharp boundaries
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
        
        subprocess.run(composite_cmd, check=True)
        print(f"✓ Output saved to: {output_path}")
    
    def process(self, input_video, background_video, output_path, max_duration=5.0):
        """
        Main processing pipeline.
        
        Args:
            input_video: Path to input video
            background_video: Path to background video  
            output_path: Path to output video
            max_duration: Maximum duration to process
        """
        temp_dir = tempfile.mkdtemp()
        temp_dir_path = Path(temp_dir)
        
        try:
            # Get video info
            cap = cv2.VideoCapture(str(input_video))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Extract and filter masks
            mask_dir = temp_dir_path / "masks"
            mask_dir.mkdir(exist_ok=True)
            
            num_frames = self.extract_masks(input_video, mask_dir, max_duration)
            
            # Composite with FFmpeg
            self.composite_with_ffmpeg(
                input_video, mask_dir, background_video, 
                output_path, num_frames, fps
            )
            
        finally:
            # Cleanup
            if temp_dir_path.exists():
                shutil.rmtree(temp_dir)


def main():
    """Test FFmpeg-based background replacement."""
    
    print("=" * 60)
    print("Advanced Background Replacement with FFmpeg")
    print("=" * 60)
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1_test_5sec.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_ffmpeg_bg_replaced.mp4")
    
    # Create 5-second test if needed
    if not input_video.exists():
        print(f"Creating 5-second test video...")
        original = Path("uploads/assets/videos/ai_math1.mp4")
        if original.exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", str(original),
                "-t", "5",
                "-c:v", "libx264",  # Re-encode for clean cut
                "-preset", "fast",
                "-crf", "18",
                str(input_video)
            ]
            subprocess.run(cmd, check=True)
            print(f"✓ Created: {input_video}")
        else:
            print(f"Error: Original not found: {original}")
            return
    
    if not background_video.exists():
        print(f"Error: Background not found: {background_video}")
        return
    
    # Process video
    print("\nProcessing video with temporal filtering...")
    replacer = FFmpegBackgroundReplacer()
    replacer.process(input_video, background_video, output_video, max_duration=5.0)
    
    # Open result
    if output_video.exists():
        print("\nOpening result...")
        subprocess.run(["open", str(output_video)])
    else:
        print(f"Error: Output not created: {output_video}")


if __name__ == "__main__":
    main()