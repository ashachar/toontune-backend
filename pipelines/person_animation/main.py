"""
Person to Character Animation Pipeline
Orchestrates the transformation of a person video into an animated character video
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import from same directory since they're now in pipelines/person_animation/
from nano_banana import NanoBananaGenerator
from runway import RunwayActTwo

# Import the sketch animation module
sys.path.append(str(project_root / 'utils' / 'draw-euler'))
from stroke_traversal_closed import extract_lines, create_drawing_animation


class PersonAnimationPipeline:
    """
    Main pipeline for person to character animation
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the pipeline
        
        Args:
            output_dir: Directory to save all outputs (will be set dynamically based on input video)
        """
        self.output_dir = output_dir  # Will be set in process_video
        
        # Initialize the components
        self.nano_banana = NanoBananaGenerator()
        self.act_two = RunwayActTwo()
        
        print(f"Pipeline initialized. Output directory: {self.output_dir}")
    
    def create_sketch_animation(self, character_image: str, start_frame: str, end_frame: str = None, duration: float = 0.4) -> str:
        """
        Create a sketch animation of the character image that draws from head first
        overlaid on a dissolving background from start_frame to end_frame
        
        Args:
            character_image: Path to the character image
            start_frame: Path to the starting video frame (person)
            end_frame: Path to the ending frame (character first frame), if None uses start_frame
            duration: Duration of the sketch animation in seconds (default 0.4s)
            
        Returns:
            Path to the sketch animation video with dissolving background
        """
        print(f"Creating sketch animation for: {character_image} (duration: {duration}s)")
        
        # Output path for sketch video
        sketch_video = os.path.join(self.output_dir, "character_sketch_animation.mp4")
        
        # Check if sketch animation already exists
        if os.path.exists(sketch_video):
            print(f"Sketch animation already exists: {sketch_video}")
            return sketch_video
        
        try:
            # Load the start and end frames
            start_bg = cv2.imread(start_frame)
            if start_bg is None:
                print(f"Warning: Could not load start frame: {start_frame}")
                return None
            
            # Load end frame (character first frame) or use start frame
            if end_frame:
                end_bg = cv2.imread(end_frame)
                if end_bg is None:
                    end_bg = start_bg
                else:
                    # Ensure both images have same dimensions
                    if end_bg.shape != start_bg.shape:
                        end_bg = cv2.resize(end_bg, (start_bg.shape[1], start_bg.shape[0]))
            else:
                end_bg = start_bg
            
            # Extract lines from the character image
            lines, original = extract_lines(character_image, output_dir=self.output_dir)
            
            # Create drawing animation frames
            frames = create_drawing_animation(
                lines, 
                original, 
                character_image,
                output_prefix=os.path.join(self.output_dir, "sketch"),
                output_dir=self.output_dir
            )
            
            if not frames:
                print("Warning: No frames generated for sketch animation")
                return None
            
            # Calculate frame duration to match desired video duration
            num_frames = len(frames)
            fps = 30  # Standard fps for smooth animation
            
            # We want exactly duration*fps frames (e.g., 0.4s * 30fps = 12 frames)
            target_frames = int(duration * fps)
            if num_frames > target_frames:
                # Sample frames evenly to speed up animation
                indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
                frames = [frames[i] for i in indices]
                print(f"Sampled {target_frames} frames from {num_frames} (speedup: {num_frames/target_frames:.1f}x)")
            elif num_frames < target_frames:
                # If we have fewer frames, repeat the last frame
                last_frame = frames[-1]
                frames.extend([last_frame] * (target_frames - num_frames))
            
            # Overlay sketch frames on dissolving background
            h, w = start_bg.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(sketch_video, fourcc, fps, (w, h))
            
            for i, frame in enumerate(frames):
                # Calculate dissolve progress (0.0 to 1.0)
                progress = i / max(1, len(frames) - 1)
                
                # Create dissolving background (interpolate between start and end)
                composite = cv2.addWeighted(start_bg, 1 - progress, end_bg, progress, 0)
                
                # Convert sketch frame to BGR if needed
                if len(frame.shape) == 2:
                    sketch_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    sketch_bgr = frame
                
                # Resize sketch to match background if needed
                if sketch_bgr.shape[:2] != (h, w):
                    sketch_bgr = cv2.resize(sketch_bgr, (w, h))
                
                # Find black pixels in sketch (the drawn lines)
                # In the sketch, black pixels are the lines, white is background
                gray_sketch = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY) if len(sketch_bgr.shape) == 3 else sketch_bgr
                mask = gray_sketch < 128  # Black pixels
                
                # Replace those pixels in the composite with the sketch
                composite[mask] = sketch_bgr[mask] if len(sketch_bgr.shape) == 3 else [0, 0, 0]
                
                out.write(composite)
            
            out.release()
            
            # Convert to H.264 for compatibility
            temp_video = sketch_video + ".temp.mp4"
            os.rename(sketch_video, temp_video)
            
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                sketch_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                os.remove(temp_video)
                print(f"Sketch animation saved to: {sketch_video}")
            else:
                os.rename(temp_video, sketch_video)
                print(f"Warning: Could not convert to H.264, using original format")
            
            return sketch_video
                
        except Exception as e:
            print(f"Error creating sketch animation: {e}")
            return None
    
    def scale_and_crop_to_match(self, runway_video: str, original_video: str) -> str:
        """
        Scale Runway video to cover original dimensions and crop to match exactly
        
        Args:
            runway_video: Path to Runway video
            original_video: Path to original video to match dimensions
            
        Returns:
            Path to scaled and cropped video
        """
        print(f"Scaling and cropping Runway video to match original dimensions...")
        
        # Get original video dimensions
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=s=x:p=0',
            original_video
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        orig_width, orig_height = map(int, result.stdout.strip().split('x'))
        
        # Get Runway video dimensions
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=s=x:p=0',
            runway_video
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        run_width, run_height = map(int, result.stdout.strip().split('x'))
        
        print(f"Original: {orig_width}x{orig_height}, Runway: {run_width}x{run_height}")
        
        # Calculate scale factor to ensure both dimensions are at least as large as original
        scale_factor = max(orig_width / run_width, orig_height / run_height)
        
        # If no scaling needed, return original
        if scale_factor <= 1.0:
            print("No scaling needed")
            return runway_video
        
        # Calculate new dimensions after scaling
        new_width = int(run_width * scale_factor)
        new_height = int(run_height * scale_factor)
        
        print(f"Scaling to {new_width}x{new_height} (factor: {scale_factor:.3f})")
        print(f"Then cropping to {orig_width}x{orig_height}")
        
        # Output path
        scaled_video = os.path.join(self.output_dir, "runway_scaled_cropped.mp4")
        
        # Use FFmpeg to scale and crop
        # scale2ref ensures aspect ratio is maintained
        # crop centers the video
        cmd = [
            'ffmpeg', '-y',
            '-i', runway_video,
            '-vf', f'scale={new_width}:{new_height},crop={orig_width}:{orig_height}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            scaled_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error scaling video: {result.stderr}")
            return runway_video
        
        print(f"Scaled and cropped video saved to: {scaled_video}")
        return scaled_video
    
    def remove_green_screen_and_composite(self, runway_video: str, original_video: str, 
                                          start_time: float = 0.5, end_time: float = 3.55) -> str:
        """
        Remove green screen from Runway video and composite over original video
        
        Args:
            runway_video: Path to aligned Runway video with green screen
            original_video: Path to original video to use as background
            start_time: Start time of segment in original video
            end_time: End time of segment in original video
            
        Returns:
            Path to composited video
        """
        print(f"Removing green screen and compositing character over original video...")
        
        # Output path
        composited_video = os.path.join(self.output_dir, "runway_composited.mp4")
        
        # Hard-coded green screen color from Runway
        # Based on analysis: RGB [123, 250, 157] or in hex: 0x7BFA9D
        green_color = "0x7BFA9D"  # RGB format for FFmpeg
        
        # Extract the segment from original video as background
        background_segment = os.path.join(self.output_dir, "temp_background.mp4")
        cmd_extract = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', original_video,
            '-t', str(end_time - start_time),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            background_segment
        ]
        subprocess.run(cmd_extract, capture_output=True, check=True)
        
        # Use FFmpeg's chromakey filter to remove green and composite
        # Note: chromakey is better for green screen removal than colorkey
        cmd_composite = [
            'ffmpeg', '-y',
            '-i', background_segment,  # Background (original video)
            '-i', runway_video,  # Foreground (character with green screen)
            '-filter_complex',
            f'[1:v]chromakey=green:0.08:0.12[ckout];[0:v][ckout]overlay[out]',
            '-map', '[out]',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            composited_video
        ]
        
        result = subprocess.run(cmd_composite, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Chromakey failed, trying with specific color")
            # Try with specific green color
            cmd_composite = [
                'ffmpeg', '-y',
                '-i', background_segment,
                '-i', runway_video,
                '-filter_complex',
                f'[1:v]colorkey={green_color}:0.15:0.1[ckout];[0:v][ckout]overlay[out]',
                '-map', '[out]',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                composited_video
            ]
            result = subprocess.run(cmd_composite, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error in compositing: {result.stderr}")
                return runway_video
        
        # Clean up temp files
        if os.path.exists(background_segment):
            os.remove(background_segment)
        
        print(f"Composited video saved to: {composited_video}")
        return composited_video
    
    def align_runway_video(self, original_video: str, runway_video: str, timestamp: float = 0.5) -> str:
        """
        Align Runway video to original video by matching foreground objects
        
        Args:
            original_video: Path to original video
            runway_video: Path to Runway generated video
            timestamp: Timestamp in original video to match (default 0.5s)
            
        Returns:
            Path to aligned Runway video
        """
        print(f"Aligning Runway video to original at {timestamp}s...")
        
        # Extract frames
        original_frame = self.extract_frame_numpy(original_video, timestamp)
        runway_frame = self.extract_frame_numpy(runway_video, 0.0)
        
        if original_frame is None or runway_frame is None:
            print("Error: Could not extract frames")
            return runway_video
        
        # Extract foregrounds using simple background removal
        orig_fg = self.extract_foreground(original_frame)
        runway_fg = self.extract_foreground(runway_frame)
        
        # Save masks for debugging
        cv2.imwrite(os.path.join(self.output_dir, "original_mask.png"), orig_fg)
        cv2.imwrite(os.path.join(self.output_dir, "runway_mask.png"), runway_fg)
        print(f"Saved masks for debugging: original_mask.png, runway_mask.png")
        
        # Find best alignment
        best_scale, best_shift = self.find_best_alignment(orig_fg, runway_fg)
        print(f"Best alignment: scale={best_scale:.2f}, shift=({best_shift[0]}, {best_shift[1]})")
        
        # Apply transformation to Runway video
        aligned_video = os.path.join(self.output_dir, "runway_aligned.mp4")
        h_orig, w_orig = original_frame.shape[:2]
        
        # Build FFmpeg filter for scaling and shifting
        scale_w = int(runway_frame.shape[1] * best_scale)
        scale_h = int(runway_frame.shape[0] * best_scale)
        pad_x = max(0, best_shift[0])
        pad_y = max(0, best_shift[1])
        
        # Create filter complex for alignment
        filter_complex = f"scale={scale_w}:{scale_h}"
        
        # Add padding to center the scaled video
        if pad_x > 0 or pad_y > 0 or scale_w < w_orig or scale_h < h_orig:
            # Extract background from original for padding
            filter_complex += f",pad={w_orig}:{h_orig}:{pad_x}:{pad_y}:color=black"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', runway_video,
            '-vf', filter_complex,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            aligned_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Could not align video: {result.stderr}")
            return runway_video
        
        # Create visualization
        self.visualize_alignment(original_frame, runway_frame, best_scale, best_shift)
        
        return aligned_video
    
    def extract_frame_numpy(self, video_path: str, timestamp: float) -> np.ndarray:
        """Extract a frame as numpy array"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    
    def extract_foreground(self, image: np.ndarray) -> np.ndarray:
        """Extract foreground using rembg or GrabCut"""
        try:
            # Try using rembg first
            from rembg import remove, new_session
            from PIL import Image
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Remove background
            session = new_session('u2net')
            output = remove(pil_image, session=session)
            
            # Convert back to numpy and extract alpha channel as mask
            output_np = np.array(output)
            if output_np.shape[2] == 4:
                mask = output_np[:, :, 3]  # Alpha channel
            else:
                # If no alpha, create mask from non-transparent pixels
                mask = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            
            print("Using rembg for foreground extraction")
            return mask
            
        except ImportError:
            print("rembg not available, using GrabCut instead")
            
            # Fallback to GrabCut
            h, w = image.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Initialize rectangle around center (assuming subject is centered)
            rect = (int(w*0.1), int(h*0.05), int(w*0.8), int(h*0.9))
            
            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            
            # Convert mask to binary
            mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
            
            # Clean up mask
            kernel = np.ones((5,5), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            return mask2
    
    def find_best_alignment(self, orig_mask: np.ndarray, runway_mask: np.ndarray) -> tuple:
        """
        Find best scale and shift to align runway mask with original mask
        
        Returns:
            (best_scale, best_shift) where shift is (x, y) offset
        """
        best_score = 0
        best_scale = 1.0
        best_shift = (0, 0)
        
        h_orig, w_orig = orig_mask.shape
        h_run, w_run = runway_mask.shape
        
        print("Starting alignment search...")
        
        # First pass: coarse search
        coarse_scales = np.arange(0.7, 1.3, 0.1)
        for scale in coarse_scales:
            # Resize runway mask
            new_w = int(w_run * scale)
            new_h = int(h_run * scale)
            
            if new_w > w_orig * 1.5 or new_h > h_orig * 1.5:
                continue
                
            scaled_mask = cv2.resize(runway_mask, (new_w, new_h))
            
            # Try different positions with coarse steps
            y_range = range(max(-new_h//3, -100), min(new_h//3, 100), 20)
            x_range = range(max(-new_w//3, -100), min(new_w//3, 100), 20)
            
            for y in y_range:
                for x in x_range:
                    # Calculate overlap
                    score = self.calculate_overlap(orig_mask, scaled_mask, (x, y))
                    
                    if score > best_score:
                        best_score = score
                        best_scale = scale
                        best_shift = (x, y)
                        print(f"  Better alignment found: scale={scale:.2f}, shift=({x},{y}), score={score:.3f}")
        
        # Second pass: fine search around best result
        print(f"Refining around best: scale={best_scale:.2f}, shift={best_shift}")
        
        fine_scales = np.arange(best_scale - 0.05, best_scale + 0.06, 0.02)
        fine_x_start = best_shift[0] - 15
        fine_x_end = best_shift[0] + 16
        fine_y_start = best_shift[1] - 15
        fine_y_end = best_shift[1] + 16
        
        for scale in fine_scales:
            # Resize runway mask
            new_w = int(w_run * scale)
            new_h = int(h_run * scale)
            
            if new_w > w_orig * 1.5 or new_h > h_orig * 1.5:
                continue
                
            scaled_mask = cv2.resize(runway_mask, (new_w, new_h))
            
            # Fine position search
            for y in range(fine_y_start, fine_y_end, 3):
                for x in range(fine_x_start, fine_x_end, 3):
                    # Calculate overlap
                    score = self.calculate_overlap(orig_mask, scaled_mask, (x, y))
                    
                    if score > best_score:
                        best_score = score
                        best_scale = scale
                        best_shift = (x, y)
                        print(f"  Refined: scale={scale:.2f}, shift=({x},{y}), score={score:.3f}")
        
        print(f"Final best alignment: scale={best_scale:.2f}, shift={best_shift}, score={best_score:.3f}")
        return best_scale, best_shift
    
    def calculate_overlap(self, mask1: np.ndarray, mask2: np.ndarray, shift: tuple) -> float:
        """Calculate overlap score between two masks with given shift"""
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        x_shift, y_shift = shift
        
        # Calculate overlap region
        x1 = max(0, x_shift)
        y1 = max(0, y_shift)
        x2 = min(w1, w2 + x_shift)
        y2 = min(h1, h2 + y_shift)
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        # Extract overlapping regions
        region1 = mask1[y1:y2, x1:x2]
        region2 = mask2[max(0, -y_shift):min(h2, h1-y_shift), 
                       max(0, -x_shift):min(w2, w1-x_shift)]
        
        # Calculate intersection over union
        intersection = np.sum(np.logical_and(region1 > 0, region2 > 0))
        union = np.sum(np.logical_or(region1 > 0, region2 > 0))
        
        return intersection / (union + 1e-6)
    
    def visualize_alignment(self, original_frame: np.ndarray, runway_frame: np.ndarray, 
                           scale: float, shift: tuple) -> None:
        """Create visualization of aligned frames and masks"""
        h_orig, w_orig = original_frame.shape[:2]
        
        # Apply transformation to runway frame
        scaled_h = int(runway_frame.shape[0] * scale)
        scaled_w = int(runway_frame.shape[1] * scale)
        scaled_runway = cv2.resize(runway_frame, (scaled_w, scaled_h))
        
        # Extract foreground masks
        orig_mask = self.extract_foreground(original_frame)
        runway_mask = self.extract_foreground(runway_frame)
        scaled_runway_mask = cv2.resize(runway_mask, (scaled_w, scaled_h))
        
        # Create two visualizations side by side
        canvas = np.zeros((h_orig, w_orig * 2, 3), dtype=np.uint8)
        
        # Left side: Original frames overlay
        # Place original in red channel
        canvas[:, :w_orig, 2] = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        
        # Place aligned runway in green channel
        x_shift, y_shift = shift
        x1 = max(0, x_shift)
        y1 = max(0, y_shift)
        x2 = min(w_orig, scaled_w + x_shift)
        y2 = min(h_orig, scaled_h + y_shift)
        
        if x2 > x1 and y2 > y1:
            runway_gray = cv2.cvtColor(scaled_runway, cv2.COLOR_BGR2GRAY)
            src_x1 = max(0, -x_shift)
            src_y1 = max(0, -y_shift)
            src_x2 = src_x1 + (x2 - x1)
            src_y2 = src_y1 + (y2 - y1)
            canvas[y1:y2, x1:x2, 1] = runway_gray[src_y1:src_y2, src_x1:src_x2]
        
        # Right side: Foreground masks overlay
        # Place original mask in red channel
        canvas[:, w_orig:, 2] = orig_mask
        
        # Place aligned runway mask in green channel
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, w_orig+x1:w_orig+x2, 1] = scaled_runway_mask[src_y1:src_y2, src_x1:src_x2]
        
        # Save visualization
        viz_path = os.path.join(self.output_dir, "alignment_visualization.png")
        cv2.imwrite(viz_path, canvas)
        print(f"Alignment visualization saved to: {viz_path}")
        print("LEFT: Frames overlay - Red=Original, Green=Runway (aligned), Yellow=Overlap")
        print("RIGHT: Masks overlay - Red=Original mask, Green=Runway mask, Yellow=Overlap")
    
    def extract_frame(self, video_path: str, timestamp: float, output_path: str) -> str:
        """
        Extract a frame from video at specific timestamp
        
        Args:
            video_path: Path to input video
            timestamp: Time in seconds
            output_path: Path to save the frame
            
        Returns:
            Path to extracted frame
        """
        print(f"Extracting frame at {timestamp}s from {video_path}...")
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract frame: {result.stderr}")
        
        print(f"Frame extracted to: {output_path}")
        return output_path
    
    def extract_video_segment(self, 
                            video_path: str, 
                            start_time: float, 
                            end_time: float,
                            output_path: str) -> str:
        """
        Extract a video segment
        
        Args:
            video_path: Path to input video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save the segment
            
        Returns:
            Path to extracted segment
        """
        duration = end_time - start_time
        print(f"Extracting video segment from {start_time}s to {end_time}s...")
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract video segment: {result.stderr}")
        
        print(f"Video segment extracted to: {output_path}")
        return output_path
    
    def replace_video_segment(self,
                            original_video: str,
                            replacement_video: str,
                            start_time: float,
                            end_time: float,
                            output_path: str,
                            dissolve_duration: float = 0.25,
                            sketch_video: str = None) -> str:
        """
        Replace a segment in the original video with the generated video
        with sketch animation at entrance and cross-dissolve at exit
        
        Args:
            original_video: Path to original video
            replacement_video: Path to replacement video
            start_time: Start time of segment to replace
            end_time: End time of segment to replace
            output_path: Path to save final video
            dissolve_duration: Duration of cross-dissolve transition in seconds (default 0.25s for faster transition)
            sketch_video: Path to sketch animation video (optional, for entrance transition)
            
        Returns:
            Path to final video
        """
        if sketch_video:
            print(f"Replacing video segment from {start_time}s to {end_time}s with sketch entrance and {dissolve_duration}s dissolve exit...")
        else:
            print(f"Replacing video segment from {start_time}s to {end_time}s with {dissolve_duration}s dissolve...")
        
        # Create temp files for the segments
        before_segment = os.path.join(self.output_dir, "temp_before.mp4")
        after_segment = os.path.join(self.output_dir, "temp_after.mp4")
        concat_list = os.path.join(self.output_dir, "concat_list.txt")
        
        # Get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            original_video
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        total_duration = float(result.stdout.strip())
        
        # Adjust times to include overlap for dissolve
        transition_start = max(0, start_time - dissolve_duration)
        transition_end = min(total_duration, end_time + dissolve_duration)
        
        # Extract segments with overlap for dissolve
        # Before segment (includes dissolve overlap at the end)
        before_with_overlap = os.path.join(self.output_dir, "temp_before_overlap.mp4")
        if start_time > 0:
            cmd_before = [
                'ffmpeg', '-y',
                '-i', original_video,
                '-t', str(start_time + dissolve_duration),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'copy',
                before_with_overlap
            ]
            subprocess.run(cmd_before, capture_output=True, check=True)
        
        # After segment (includes dissolve overlap at the beginning)
        after_with_overlap = os.path.join(self.output_dir, "temp_after_overlap.mp4")
        if end_time < total_duration:
            cmd_after = [
                'ffmpeg', '-y',
                '-ss', str(end_time - dissolve_duration),
                '-i', original_video,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'copy',
                after_with_overlap
            ]
            subprocess.run(cmd_after, capture_output=True, check=True)
        
        # Now we'll use FFmpeg's xfade filter to create smooth cross-dissolve transitions
        # This requires building a complex filter graph
        
        # Check if original video has audio
        probe_audio = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            original_video
        ]
        result = subprocess.run(probe_audio, capture_output=True, text=True)
        has_audio = bool(result.stdout.strip())
        
        # Get original video dimensions and frame rate
        probe_resolution = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'csv=s=x:p=0',
            original_video
        ]
        result = subprocess.run(probe_resolution, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split('x')
        orig_width = int(parts[0])
        orig_height = int(parts[1])
        fps_fraction = parts[2]
        # Convert frame rate fraction to float
        if '/' in fps_fraction:
            num, den = map(float, fps_fraction.split('/'))
            fps = num / den
        else:
            fps = float(fps_fraction)
        
        # Prepare the replacement video (ensure same format, dimensions, and frame rate)
        replacement_processed = os.path.join(self.output_dir, "temp_replacement.mp4")
        cmd_process_replacement = [
            'ffmpeg', '-y',
            '-i', replacement_video,
            '-vf', f'scale={orig_width}:{orig_height}:force_original_aspect_ratio=decrease,pad={orig_width}:{orig_height}:(ow-iw)/2:(oh-ih)/2,fps={fps}',
            '-c:v', 'libx264',
            '-preset', 'fast', 
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-an',  # Remove audio from replacement
            replacement_processed
        ]
        subprocess.run(cmd_process_replacement, capture_output=True, check=True)
        
        # Build the video with transitions
        temp_video_no_audio = os.path.join(self.output_dir, "temp_video_no_audio.mp4")
        
        if start_time > 0 and end_time < total_duration:
            # We have before, replacement, and after segments
            
            if sketch_video:
                # Sketch happens BEFORE character video with cross-dissolve to first frame
                print("Using sketch with cross-dissolve to character, then character animation...")
                
                # Get sketch duration
                probe_sketch = [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    sketch_video
                ]
                result = subprocess.run(probe_sketch, capture_output=True, text=True, check=True)
                sketch_duration = float(result.stdout.strip())
                
                # Adjust timeline: sketch starts at (start_time - sketch_duration)
                sketch_start = start_time - sketch_duration
                
                # Step 1: Extract segment BEFORE sketch (if any)
                if sketch_start > 0:
                    before_sketch = os.path.join(self.output_dir, "temp_before_sketch.mp4")
                    cmd_before = [
                        'ffmpeg', '-y',
                        '-i', original_video,
                        '-t', str(sketch_start),
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-crf', '18',
                        '-c:a', 'copy',
                        before_sketch
                    ]
                    subprocess.run(cmd_before, capture_output=True, check=True)
                
                # Step 2: Process sketch video to match dimensions and frame rate
                # The sketch already contains the dissolve from person to character
                sketch_processed = os.path.join(self.output_dir, "temp_sketch_processed.mp4")
                cmd_process_sketch = [
                    'ffmpeg', '-y',
                    '-i', sketch_video,
                    '-vf', f'scale={orig_width}:{orig_height}:force_original_aspect_ratio=decrease,pad={orig_width}:{orig_height}:(ow-iw)/2:(oh-ih)/2,fps={fps}',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    sketch_processed
                ]
                subprocess.run(cmd_process_sketch, capture_output=True, check=True)
                
                # Step 3: Build final sequence
                intermediate1 = os.path.join(self.output_dir, "temp_intermediate1.mp4")
                concat_list1 = os.path.join(self.output_dir, "concat_list1.txt")
                
                with open(concat_list1, 'w') as f:
                    if sketch_start > 0:
                        f.write(f"file '{before_sketch}'\n")
                    f.write(f"file '{sketch_processed}'\n")  # Sketch with dissolve
                    f.write(f"file '{replacement_processed}'\n")  # Character animation
                
                cmd_concat1 = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_list1,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    intermediate1
                ]
                
                result = subprocess.run(cmd_concat1, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error concatenating with sketch: {result.stderr}")
                    raise RuntimeError(f"Failed to concatenate with sketch: {result.stderr}")
                
                # Get duration of intermediate video
                probe_cmd = [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    intermediate1
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                intermediate_duration = float(result.stdout.strip())
                
                # Step 4: Apply dissolve transition for exit (intermediate1 -> after)
                offset_time_exit = intermediate_duration - dissolve_duration
                
                cmd_exit_dissolve = [
                    'ffmpeg', '-y',
                    '-i', intermediate1,
                    '-i', after_with_overlap,
                    '-filter_complex',
                    f'[0:v][1:v]xfade=transition=dissolve:duration={dissolve_duration}:offset={offset_time_exit}[v]',
                    '-map', '[v]',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    temp_video_no_audio
                ]
                
                result = subprocess.run(cmd_exit_dissolve, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error in exit dissolve: {result.stderr}")
                    raise RuntimeError(f"Failed to apply exit dissolve: {result.stderr}")
                    
            else:
                # Original logic with dissolves at both entrance and exit
                print("Applying cross-dissolve transitions at both entrance and exit...")
                
                # Create intermediate video with first dissolve (before -> replacement)
                intermediate = os.path.join(self.output_dir, "temp_intermediate.mp4")
                offset_time = start_time  # When to start the dissolve
                
                cmd_first_dissolve = [
                    'ffmpeg', '-y',
                    '-i', before_with_overlap,
                    '-i', replacement_processed,
                    '-filter_complex',
                    f'[0:v][1:v]xfade=transition=dissolve:duration={dissolve_duration}:offset={offset_time}[v]',
                    '-map', '[v]',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    intermediate
                ]
                
                result = subprocess.run(cmd_first_dissolve, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error in first dissolve: {result.stderr}")
                    raise RuntimeError(f"Failed to apply first dissolve: {result.stderr}")
                
                # Get duration of intermediate video
                probe_cmd = [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    intermediate
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                intermediate_duration = float(result.stdout.strip())
                
                # Now apply second dissolve (replacement -> after)
                # The offset is when the replacement ends in the intermediate video
                offset_time_2 = intermediate_duration - dissolve_duration
                
                cmd_second_dissolve = [
                    'ffmpeg', '-y',
                    '-i', intermediate,
                    '-i', after_with_overlap,
                    '-filter_complex',
                    f'[0:v][1:v]xfade=transition=dissolve:duration={dissolve_duration}:offset={offset_time_2}[v]',
                    '-map', '[v]',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    temp_video_no_audio
                ]
                
                result = subprocess.run(cmd_second_dissolve, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error in second dissolve: {result.stderr}")
                    raise RuntimeError(f"Failed to apply second dissolve: {result.stderr}")
                
        elif start_time > 0:
            # Only have before and replacement segments - one dissolve
            print("Applying cross-dissolve transition at entrance...")
            offset_time = start_time
            
            cmd_dissolve = [
                'ffmpeg', '-y',
                '-i', before_with_overlap,
                '-i', replacement_processed,
                '-filter_complex',
                f'[0:v][1:v]xfade=transition=dissolve:duration={dissolve_duration}:offset={offset_time}[v]',
                '-map', '[v]',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                temp_video_no_audio
            ]
            
            result = subprocess.run(cmd_dissolve, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to apply dissolve: {result.stderr}")
                
        elif end_time < total_duration:
            # Only have replacement and after segments - one dissolve
            print("Applying cross-dissolve transition at exit...")
            
            # Get duration of replacement video
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                replacement_processed
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            replacement_duration = float(result.stdout.strip())
            
            offset_time = replacement_duration - dissolve_duration
            
            cmd_dissolve = [
                'ffmpeg', '-y',
                '-i', replacement_processed,
                '-i', after_with_overlap,
                '-filter_complex',
                f'[0:v][1:v]xfade=transition=dissolve:duration={dissolve_duration}:offset={offset_time}[v]',
                '-map', '[v]',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                temp_video_no_audio
            ]
            
            result = subprocess.run(cmd_dissolve, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to apply dissolve: {result.stderr}")
        else:
            # Just the replacement video
            import shutil
            shutil.copy2(replacement_processed, temp_video_no_audio)
        
        # Check for accompanying MP3 audio file
        video_dir = os.path.dirname(original_video)
        video_name = os.path.splitext(os.path.basename(original_video))[0]
        
        # Try different audio file naming patterns
        possible_audio_files = [
            os.path.join(video_dir, f"{video_name}_audio.mp3"),
            os.path.join(video_dir, f"{video_name.replace('_input', '_audio')}.mp3"),
            os.path.join(video_dir, f"{video_name.replace('input', 'audio')}.mp3"),
        ]
        
        audio_file = None
        for audio_path in possible_audio_files:
            if os.path.exists(audio_path):
                audio_file = audio_path
                print(f"Found audio file: {audio_file}")
                break
        
        # If no exact match, look for any MP3 in the same directory
        if not audio_file:
            mp3_files = [f for f in os.listdir(video_dir) if f.endswith('.mp3')]
            if mp3_files:
                audio_file = os.path.join(video_dir, mp3_files[0])
                print(f"Using audio file: {audio_file}")
        
        # If audio file exists, add it to the video
        if audio_file and os.path.exists(audio_file):
            print(f"Adding audio from: {audio_file}")
            
            # Get video duration
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                temp_video_no_audio
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            video_duration = float(result.stdout.strip())
            
            # Add audio to video, trimmed to video length
            cmd_add_audio = [
                'ffmpeg', '-y',
                '-i', temp_video_no_audio,
                '-i', audio_file,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-t', str(video_duration),  # Trim audio to video length
                '-map', '0:v:0',
                '-map', '1:a:0',
                output_path
            ]
            
            result = subprocess.run(cmd_add_audio, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Failed to add audio: {result.stderr}")
                # Fallback: just copy video without audio
                import shutil
                shutil.copy2(temp_video_no_audio, output_path)
            
            # Clean up temp video
            if os.path.exists(temp_video_no_audio):
                os.remove(temp_video_no_audio)
        else:
            # No audio file found, just rename the temp video
            import shutil
            shutil.move(temp_video_no_audio, output_path)
            if not has_audio:
                print("No audio track in original video and no MP3 file found")
        
        # Clean up temp files
        temp_files = [
            before_segment, after_segment, replacement_processed, concat_list,
            before_with_overlap if 'before_with_overlap' in locals() else None,
            after_with_overlap if 'after_with_overlap' in locals() else None,
            intermediate if 'intermediate' in locals() else None,
            before_clean if 'before_clean' in locals() else None,
            sketch_processed if 'sketch_processed' in locals() else None,
            intermediate1 if 'intermediate1' in locals() else None,
            concat_list1 if 'concat_list1' in locals() else None,
            before_sketch if 'before_sketch' in locals() else None
        ]
        temp_files = [f for f in temp_files if f]  # Remove None values
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"Final video saved to: {output_path}")
        return output_path
    
    def process_video(self, 
                     video_path: str,
                     character_description: str = "friendly meerkat",
                     dissolve_duration: float = 0.5) -> str:
        """
        Process the entire video pipeline
        
        Args:
            video_path: Path to input video
            character_description: Description of character to generate
            dissolve_duration: Duration of cross-dissolve transitions in seconds
            
        Returns:
            Path to final processed video
        """
        # Set output directory to same folder as input video
        if self.output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(video_path))
        
        print(f"\n{'='*60}")
        print(f"Starting Person Animation Pipeline")
        print(f"Input: {video_path}")
        print(f"Character: {character_description}")
        print(f"Output folder: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract frame at 0.5s
        frame_path = os.path.join(self.output_dir, "frame0.png")
        if os.path.exists(frame_path):
            print(f"Frame already exists, skipping extraction: {frame_path}")
        else:
            self.extract_frame(video_path, 0.5, frame_path)
        
        # Step 2: Extract video segment from 0.5s to 3.5s
        subvideo_path = os.path.join(self.output_dir, "runway_subvideo.mp4")
        if os.path.exists(subvideo_path):
            print(f"Subvideo already exists, skipping extraction: {subvideo_path}")
        else:
            self.extract_video_segment(video_path, 0.5, 3.55, subvideo_path)  # 3.05 seconds to ensure > 3s
        
        # Step 3: Generate character image using Nano Banana
        character_image = os.path.join(self.output_dir, "runway_character.png")
        if os.path.exists(character_image):
            print(f"\nCharacter image already exists, skipping generation: {character_image}")
        else:
            print(f"\nGenerating character sketch for: {character_description}")
            character_image = self.nano_banana.generate_character(
                frame_path,
                character_description,
                self.output_dir
            )
        
        # Step 4: Generate character performance using Act-Two
        generated_video = os.path.join(self.output_dir, "runway_act_two_output.mp4")
        if os.path.exists(generated_video):
            print(f"\nAct-Two video already exists, skipping generation: {generated_video}")
        else:
            print(f"\nGenerating character performance with Act-Two...")
            generated_video = self.act_two.generate_character_performance(
                subvideo_path,
                character_image,
                self.output_dir,
                duration=3  # 3 seconds
            )
        
        # Step 4.1: Scale Runway video to cover original dimensions
        scaled_video = self.scale_and_crop_to_match(generated_video, video_path)
        if scaled_video != generated_video:
            print(f"Using scaled and cropped video: {scaled_video}")
            generated_video = scaled_video
        
        # Step 4.5: Extract first frame of character animation for dissolve
        character_first_frame_path = os.path.join(self.output_dir, "character_first_frame.png")
        if not os.path.exists(character_first_frame_path):
            cmd_extract = [
                'ffmpeg', '-y',
                '-i', generated_video,
                '-frames:v', '1',
                '-q:v', '2',
                character_first_frame_path
            ]
            subprocess.run(cmd_extract, capture_output=True, check=True)
        
        # Step 4.6: Create sketch animation with dissolving background
        # Dissolve from person frame to character first frame during sketch
        sketch_animation = self.create_sketch_animation(
            character_image, 
            frame_path,  # Person frame at 0.5s
            character_first_frame_path,  # Character first frame
            duration=0.4
        )
        
        # Step 5: Replace segment in original video with sketch entrance
        final_video_path = os.path.join(self.output_dir, "final_character_video.mp4")
        if os.path.exists(final_video_path):
            print(f"\nFinal video already exists, skipping replacement: {final_video_path}")
        else:
            self.replace_video_segment(
                video_path,
                generated_video,
                0.5,
                3.55,
                final_video_path,
                dissolve_duration=dissolve_duration,
                sketch_video=sketch_animation
            )
        
        # Step 6: Open the final video
        print(f"\nOpening final video...")
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", final_video_path])
        elif sys.platform == "win32":  # Windows
            subprocess.run(["start", "", final_video_path], shell=True)
        else:  # Linux
            subprocess.run(["xdg-open", final_video_path])
        
        print(f"\n{'='*60}")
        print(f"Pipeline Complete!")
        print(f"Final video: {final_video_path}")
        print(f"All artifacts saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return final_video_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Person to Character Animation Pipeline")
    parser.add_argument(
        "video",
        nargs="?",
        default="nano_video_input.mp4",
        help="Path to input video (default: nano_video_input.mp4)"
    )
    parser.add_argument(
        "--character",
        default="friendly meerkat",
        help="Character description (default: friendly meerkat)"
    )
    parser.add_argument(
        "--dissolve",
        type=float,
        default=0.5,
        help="Duration of cross-dissolve transitions in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for all artifacts (default: same as input video)"
    )
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.video):
        print(f"Error: Input video not found: {args.video}")
        sys.exit(1)
    
    # Run the pipeline
    pipeline = PersonAnimationPipeline(output_dir=args.output_dir)
    
    try:
        final_video = pipeline.process_video(args.video, args.character, args.dissolve)
        print(f"Success! Final video: {final_video}")
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()