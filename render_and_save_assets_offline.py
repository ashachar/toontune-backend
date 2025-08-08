#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import glob
from pathlib import Path
from sklearn.cluster import MeanShift, estimate_bandwidth
import math
from dotenv import load_dotenv
from supabase import create_client, Client
import tempfile
from typing import Optional, Tuple, List, Dict
from PIL import Image
import cairosvg
import io

# Import all the functions from generate_drawing_video.py
sys.path.append('video-processing')
from generate_drawing_video import (
    load_hand_image,
    overlay_hand,
    is_white_or_transparent,
    segment_image_with_meanshift,
    get_cluster_info,
    order_clusters_clockwise,
    find_cluster_boundary,
    order_boundary_clockwise,
    offset_boundary_inward,
    smooth_trajectory,
    map_pixels_to_trajectory,
    create_edge_following_path
)

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('VITE_SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Error: Supabase credentials not found in .env file")
    sys.exit(1)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Constants
ASSETS_DIR = "../app/backend/uploads/assets"
HAND_IMAGE_PATH = "images/hand.png"  # We'll need to find or create this
BUCKET_NAME = "rendered-assets"
FPS = 30
HAND_SCALE = 0.12
TOTAL_FRAMES = 90  # 3 seconds for proper drawing animation
SVG_SIZE = 600  # Good size for SVG conversion

def convert_svg_to_png(svg_path: str) -> Optional[str]:
    """Convert SVG to PNG and save as temporary file"""
    try:
        # Read SVG and convert to PNG bytes
        png_bytes = cairosvg.svg2png(url=svg_path, output_width=SVG_SIZE, output_height=SVG_SIZE)
        
        # Save to temporary file
        temp_png = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_png.write(png_bytes)
        temp_png_path = temp_png.name
        temp_png.close()
        
        return temp_png_path
    
    except Exception as e:
        print(f"  Error converting SVG to PNG: {e}")
        return None

def create_drawing_animation_for_asset(asset_path: str, hand_img: Optional[np.ndarray], hand_alpha: Optional[np.ndarray]) -> Optional[str]:
    """
    Create a drawing animation for a single asset using the sophisticated algorithm
    from generate_drawing_video.py
    """
    print(f"\nProcessing asset: {asset_path}")
    
    # Handle SVG files
    temp_png_path = None
    if asset_path.endswith('.svg'):
        print("  Converting SVG to PNG...")
        temp_png_path = convert_svg_to_png(asset_path)
        if temp_png_path is None:
            print("  Failed to convert SVG")
            return None
        working_path = temp_png_path
    else:
        working_path = asset_path
    
    # Segment the image using Mean Shift clustering
    print("  Segmenting image with Mean Shift clustering...")
    cluster_map, img = segment_image_with_meanshift(working_path)
    
    # Clean up temp PNG if we created one
    if temp_png_path:
        try:
            os.remove(temp_png_path)
        except:
            pass
    
    if cluster_map is None:
        print("  Failed to segment image")
        return None
    
    height, width = img.shape[:2]
    
    # Setup video writer
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video_path = temp_video.name
    temp_video.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, FPS, (width, height))
    
    # Initialize drawing canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    reveal_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate pen offset for hand position
    if hand_img is not None:
        pen_offset_x = hand_img.shape[1] // 3
        pen_offset_y = hand_img.shape[0] // 3
    else:
        pen_offset_x = pen_offset_y = 20  # Default offset if no hand
    
    # Analyze clusters
    print("  Analyzing clusters...")
    cluster_info = get_cluster_info(cluster_map)
    
    if not cluster_info:
        print("  No valid clusters found")
        video_writer.release()
        os.remove(temp_video_path)
        return None
    
    print(f"  Found {len(cluster_info)} clusters")
    
    # Order clusters in clockwise direction
    ordered_clusters = order_clusters_clockwise(cluster_info)
    
    # Calculate total pixels and frame allocation
    total_pixels = sum(c['pixel_count'] for c in cluster_info)
    total_frames = TOTAL_FRAMES
    
    # Allocate frames to each cluster
    frame_allocation = []
    remaining_frames = total_frames
    
    for i, cluster in enumerate(ordered_clusters):
        if i == len(ordered_clusters) - 1:
            frames = remaining_frames
        else:
            frames = int((cluster['pixel_count'] / total_pixels) * total_frames)
            frames = max(5, frames)  # Minimum 5 frames per cluster
            remaining_frames -= frames
        frame_allocation.append(frames)
    
    print("  Creating drawing animation with edge-following pattern...")
    
    # Draw each cluster
    frames_used = 0
    
    for cluster_idx, cluster in enumerate(ordered_clusters):
        cluster_id = cluster['id']
        center = cluster['center']
        frames_for_cluster = frame_allocation[cluster_idx]
        
        # Create edge-following path
        trajectory = create_edge_following_path(cluster_map, cluster_id, center)
        
        if not trajectory:
            continue
        
        # Map pixels to trajectory points
        pixel_mapping = map_pixels_to_trajectory(cluster_map, cluster_id, trajectory)
        
        # Calculate points per frame
        points_per_frame = max(1, len(trajectory) // frames_for_cluster)
        
        trajectory_idx = 0
        cluster_frames = 0
        
        while trajectory_idx < len(trajectory) and cluster_frames < frames_for_cluster and frames_used < total_frames:
            frame = canvas.copy()
            
            # Process multiple trajectory points per frame
            for _ in range(points_per_frame):
                if trajectory_idx < len(trajectory):
                    # Get pixels to reveal
                    pixels_to_reveal = pixel_mapping.get(trajectory_idx, [])
                    
                    # Reveal pixels with brush
                    for px, py in pixels_to_reveal:
                        cv2.circle(reveal_mask, (px, py), 12, 255, -1)
                    
                    # Also reveal near trajectory point
                    if trajectory_idx < len(trajectory):
                        traj_x, traj_y = trajectory[trajectory_idx]
                        cv2.circle(reveal_mask, (traj_x, traj_y), 20, 255, -1)
                    
                    trajectory_idx += 1
            
            # Apply revealed portions
            mask_3channel = cv2.cvtColor(reveal_mask, cv2.COLOR_GRAY2BGR) / 255.0
            frame = (frame * (1 - mask_3channel) + img * mask_3channel).astype(np.uint8)
            
            # Add hand or drawing indicator
            if trajectory_idx > 0 and trajectory_idx <= len(trajectory):
                hand_pos = trajectory[min(trajectory_idx - 1, len(trajectory) - 1)]
                
                if hand_img is not None:
                    # Use actual hand image
                    hand_x = hand_pos[0] - pen_offset_x
                    hand_y = hand_pos[1] - pen_offset_y
                    frame = overlay_hand(frame, hand_img, hand_alpha, (hand_x, hand_y))
                else:
                    # Draw a pen/brush indicator
                    # Draw a pencil-like shape
                    pen_tip = hand_pos
                    pen_end = (pen_tip[0] - 30, pen_tip[1] - 30)
                    
                    # Draw pencil body
                    cv2.line(frame, pen_tip, pen_end, (139, 69, 19), 8)  # Brown pencil
                    cv2.line(frame, pen_tip, pen_end, (160, 82, 45), 6)  # Lighter brown center
                    
                    # Draw pencil tip
                    cv2.circle(frame, pen_tip, 4, (50, 50, 50), -1)  # Dark tip
                    cv2.circle(frame, pen_tip, 2, (20, 20, 20), -1)  # Darker center
            
            video_writer.write(frame)
            frames_used += 1
            cluster_frames += 1
    
    # Check for missed pixels and do touch-ups
    missed_pixels = []
    for y in range(height):
        for x in range(width):
            if cluster_map[y, x] >= 0 and reveal_mask[y, x] == 0:
                missed_pixels.append((x, y))
    
    if missed_pixels:
        # Quick touch-up animation
        touch_up_frames = 15
        
        for frame_num in range(touch_up_frames):
            frame = canvas.copy()
            
            # Reveal remaining pixels gradually
            pixels_per_frame = len(missed_pixels) // touch_up_frames + 1
            start_idx = frame_num * pixels_per_frame
            end_idx = min(start_idx + pixels_per_frame, len(missed_pixels))
            
            for px, py in missed_pixels[start_idx:end_idx]:
                cv2.circle(reveal_mask, (px, py), 8, 255, -1)
            
            # Apply revealed portions
            mask_3channel = cv2.cvtColor(reveal_mask, cv2.COLOR_GRAY2BGR) / 255.0
            frame = (frame * (1 - mask_3channel) + img * mask_3channel).astype(np.uint8)
            
            video_writer.write(frame)
    
    # Add final frames showing completed image
    final_frame = img.copy()
    for _ in range(30):  # 1 second at 30fps
        video_writer.write(final_frame)
    
    video_writer.release()
    
    print(f"  Video created: {temp_video_path}")
    return temp_video_path

def upload_to_supabase(video_path: str, asset_name: str) -> bool:
    """Upload video to Supabase storage"""
    try:
        video_filename = f"{asset_name}_drawing.mp4"
        
        # Check if already exists and remove
        try:
            files = supabase.storage.from_(BUCKET_NAME).list()
            if any(f['name'] == video_filename for f in files):
                supabase.storage.from_(BUCKET_NAME).remove([video_filename])
                print(f"  Removed existing: {video_filename}")
        except:
            pass
        
        # Read video file
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        # Upload to Supabase
        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=video_filename,
            file=video_data,
            file_options={"content-type": "video/mp4"}
        )
        
        print(f"  Uploaded to Supabase: {video_filename}")
        return True
    
    except Exception as e:
        print(f"  Error uploading to Supabase: {e}")
        return False

def ensure_bucket_exists():
    """Ensure the Supabase bucket exists"""
    try:
        buckets = supabase.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        
        if BUCKET_NAME not in bucket_names:
            supabase.storage.create_bucket(
                BUCKET_NAME,
                options={"public": True}
            )
            print(f"Created bucket: {BUCKET_NAME}")
        else:
            print(f"Bucket already exists: {BUCKET_NAME}")
    
    except Exception as e:
        print(f"Note: Could not verify bucket (may already exist): {e}")

def create_default_hand_image():
    """Create a simple hand drawing if no hand image is available"""
    # Create a simple hand shape
    hand_img = np.ones((200, 150, 3), dtype=np.uint8) * 255
    
    # Draw a simple hand shape
    # Palm
    cv2.ellipse(hand_img, (75, 120), (40, 50), 0, 0, 360, (255, 220, 177), -1)
    
    # Fingers
    finger_positions = [(50, 70), (65, 60), (75, 55), (85, 60), (100, 70)]
    for x, y in finger_positions:
        cv2.ellipse(hand_img, (x, y), (8, 20), 0, 0, 360, (255, 220, 177), -1)
    
    # Thumb
    cv2.ellipse(hand_img, (40, 100), (15, 25), -30, 0, 360, (255, 220, 177), -1)
    
    # Add some shading
    cv2.ellipse(hand_img, (75, 120), (35, 45), 0, 0, 360, (240, 200, 160), 2)
    
    return hand_img, None

def main():
    """Main function to process all assets"""
    print("Starting asset rendering process with proper drawing animation...")
    
    # Ensure bucket exists
    ensure_bucket_exists()
    
    # Try to load hand image, create default if not found
    hand_img, hand_alpha = load_hand_image(HAND_IMAGE_PATH)
    
    if hand_img is None:
        print("Hand image not found, creating default hand shape...")
        hand_img, hand_alpha = create_default_hand_image()
    
    if hand_img is not None:
        # Scale hand image
        new_width = int(hand_img.shape[1] * HAND_SCALE)
        new_height = int(hand_img.shape[0] * HAND_SCALE)
        hand_img = cv2.resize(hand_img, (new_width, new_height))
        if hand_alpha is not None:
            hand_alpha = cv2.resize(hand_alpha, (new_width, new_height))
    
    # Find all assets
    asset_patterns = [
        os.path.join(ASSETS_DIR, "*.png"),
        os.path.join(ASSETS_DIR, "*.jpg"),
        os.path.join(ASSETS_DIR, "*.jpeg"),
        os.path.join(ASSETS_DIR, "*.svg")
    ]
    
    asset_files = []
    for pattern in asset_patterns:
        asset_files.extend(glob.glob(pattern))
    
    if not asset_files:
        print(f"No assets found in {ASSETS_DIR}")
        return
    
    print(f"Found {len(asset_files)} assets to process")
    
    # Process each asset
    successful = 0
    failed = 0
    
    for asset_path in asset_files:
        asset_name = Path(asset_path).stem
        
        try:
            # Render the asset with proper drawing animation
            video_path = create_drawing_animation_for_asset(asset_path, hand_img, hand_alpha)
            
            if video_path:
                # Upload to Supabase
                if upload_to_supabase(video_path, asset_name):
                    successful += 1
                else:
                    failed += 1
                
                # Clean up temporary file
                try:
                    os.remove(video_path)
                except:
                    pass
            else:
                failed += 1
        
        except Exception as e:
            print(f"  Error processing {asset_name}: {e}")
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nVideos have been uploaded to Supabase bucket: {BUCKET_NAME}")
    print("The videos now show proper hand movement and clustering-based drawing!")

if __name__ == "__main__":
    main()