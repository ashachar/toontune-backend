#!/usr/bin/env python3
"""
Advanced background replacement with edge decontamination.
Removes the thin white/colored layer between foreground and background.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import tempfile
from tqdm import tqdm
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.cluster import KMeans


def detect_background_color(frame, mask):
    """
    Detect the dominant background color from non-masked regions.
    
    Args:
        frame: Original frame (BGR)
        mask: Binary mask (255 for foreground, 0 for background)
    
    Returns:
        Dominant background color (BGR)
    """
    # Get background pixels
    bg_pixels = frame[mask < 128]
    
    if len(bg_pixels) > 0:
        # Use k-means to find dominant color
        kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
        kmeans.fit(bg_pixels.reshape(-1, 3))
        bg_color = kmeans.cluster_centers_[0]
        return bg_color.astype(np.uint8)
    else:
        # Default to white if no background found
        return np.array([255, 255, 255], dtype=np.uint8)


def decontaminate_edges(frame, mask, bg_color_bgr, edge_thickness=15):
    """
    Remove background color contamination from edges.
    
    Args:
        frame: Original frame (BGR)
        mask: Binary mask
        bg_color_bgr: Background color to remove (BGR)
        edge_thickness: Thickness of edge region to process
    
    Returns:
        Decontaminated mask
    """
    h, w = mask.shape
    
    # Create edge region mask
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_thickness, edge_thickness))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Dilated version of mask
    mask_dilated = cv2.dilate(mask, kernel_large, iterations=1)
    # Eroded version of mask  
    mask_eroded = cv2.erode(mask, kernel_small, iterations=1)
    
    # Edge region is between dilated and eroded
    edge_region = (mask_dilated > 128) & (mask_eroded < 128)
    
    if np.any(edge_region):
        # Convert to LAB for better color comparison
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        bg_color_lab = cv2.cvtColor(bg_color_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
        
        # Calculate color distance in LAB space
        color_diff = np.sqrt(np.sum((frame_lab.astype(np.float32) - bg_color_lab.astype(np.float32))**2, axis=2))
        
        # Pixels similar to background color (within threshold)
        # Lower threshold = more aggressive removal
        color_threshold = 50  # Adjust based on how aggressive you want to be
        similar_to_bg = color_diff < color_threshold
        
        # Remove edge pixels that are similar to background
        contaminated = edge_region & similar_to_bg
        mask[contaminated] = 0
        
        # Also reduce alpha for pixels somewhat similar to background
        semi_contaminated = edge_region & (color_diff < color_threshold * 1.5) & ~contaminated
        mask[semi_contaminated] = mask[semi_contaminated] * 0.5
    
    return mask


def process_video_decontaminated(input_video, background_video, output_path, 
                                max_duration=5.0):
    """
    Replace background with advanced edge decontamination.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process (seconds)
    """
    print("Starting background replacement with edge decontamination...")
    print("This removes the thin white/colored layer at edges")
    
    # Open videos
    cap_input = cv2.VideoCapture(str(input_video))
    cap_bg = cv2.VideoCapture(str(background_video))
    
    # Get video properties
    fps = cap_input.get(cv2.CAP_PROP_FPS)
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(int(fps * max_duration), int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    # Create rembg session
    print("Initializing background removal model...")
    session = new_session('u2netp')  # Use high quality model
    
    # Detect background color from first frame
    print("Detecting original background color...")
    ret, first_frame = cap_input.read()
    if ret:
        # Get initial mask
        first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_pil = Image.fromarray(first_rgb)
        first_output = remove(first_pil, session=session, only_mask=True)
        first_mask = np.array(first_output)
        if len(first_mask.shape) == 3:
            first_mask = first_mask[:, :, 0]
        
        # Detect background color
        bg_color = detect_background_color(first_frame, first_mask)
        print(f"Detected background color (BGR): {bg_color}")
        
        # Reset video
        cap_input.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        bg_color = np.array([255, 255, 255], dtype=np.uint8)  # Default white
    
    # Process frames
    print(f"Processing {total_frames} frames with decontamination...")
    
    for frame_idx in tqdm(range(total_frames), desc="Removing backgrounds"):
        # Read frames
        ret_input, frame_input = cap_input.read()
        ret_bg, frame_bg = cap_bg.read()
        
        if not ret_input:
            break
            
        # Loop background video if needed
        if not ret_bg:
            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_bg, frame_bg = cap_bg.read()
        
        # Resize background to match input
        frame_bg_resized = cv2.resize(frame_bg, (width, height))
        
        # Convert BGR to RGB for rembg
        frame_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Remove background with alpha matting
        output_rgba = remove(pil_image, session=session, 
                           alpha_matting=True,
                           alpha_matting_foreground_threshold=270,
                           alpha_matting_background_threshold=0,
                           alpha_matting_erode_size=0)  # No erosion yet
        
        # Get mask and foreground
        output_np = np.array(output_rgba)
        if output_np.shape[2] == 4:
            mask = output_np[:, :, 3]
            foreground_rgb = output_np[:, :, :3]
            foreground_bgr = cv2.cvtColor(foreground_rgb, cv2.COLOR_RGB2BGR)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
            foreground_bgr = frame_input
        
        # CRITICAL: Decontaminate edges
        mask = decontaminate_edges(frame_input, mask, bg_color, edge_thickness=20)
        
        # Final cleanup
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Fill tiny holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Very slight erosion to ensure clean edges
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.erode(mask, kernel_tiny, iterations=1)
        
        # Minimal blur for smoothing
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
        
        # Create alpha channel
        alpha = mask.astype(np.float32) / 255.0
        alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
        
        # Composite using decontaminated foreground
        composite = (alpha_3d * foreground_bgr.astype(np.float32) + 
                    (1.0 - alpha_3d) * frame_bg_resized.astype(np.float32))
        
        # Convert back to uint8
        composite_final = np.clip(composite, 0, 255).astype(np.uint8)
        
        # Write frame
        out.write(composite_final)
    
    # Clean up
    cap_input.release()
    cap_bg.release()
    out.release()
    
    print("Encoding final video with H.264...")
    # Convert to H.264
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output.name,
        "-i", str(input_video),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-map", "0:v",
        "-map", "1:a?",
        "-movflags", "+faststart",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)
    
    # Clean up temp file
    Path(temp_output.name).unlink()
    
    print(f"Output saved to: {output_path}")


def main():
    """Test decontaminated background replacement."""
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1_test_5sec.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_decontaminated_bg.mp4")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Process video
    process_video_decontaminated(
        input_video,
        background_video,
        output_video,
        max_duration=5.0
    )
    
    # Open result
    subprocess.run(["open", str(output_video)])


if __name__ == "__main__":
    main()