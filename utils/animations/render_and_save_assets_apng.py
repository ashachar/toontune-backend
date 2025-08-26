#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import glob
import argparse
from pathlib import Path
from sklearn.cluster import MeanShift, estimate_bandwidth
import math
from dotenv import load_dotenv
from supabase import create_client, Client
import tempfile
from typing import Optional, Tuple, List, Dict, Set
from PIL import Image
import cairosvg
import io
from apngasm_python import _apngasm_python as apngasm
import json

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
ASSETS_DIR = "../app/uploads/assets"
HAND_IMAGE_PATH = "../app/uploads/assets/hand.png"  # Correct path to the actual hand image
BUCKET_NAME = "rendered-assets"
FPS = 30
HAND_SCALE = 0.225  # 50% larger than 0.15, for the 1024x1024 hand image (results in ~230x230 pixels)
TOTAL_FRAMES = 90  # 3 seconds for proper drawing animation
SVG_SIZE = 600  # Good size for SVG conversion
FRAME_DELAY = 33  # milliseconds per frame (approximately 30 FPS)

def get_existing_apngs() -> Set[str]:
    """Get a set of asset names that already have APNGs in the bucket"""
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()
        # Extract asset names from APNG filenames (remove _drawing.apng suffix)
        existing = set()
        for f in files:
            if f['name'].endswith('_drawing.apng'):
                asset_name = f['name'].replace('_drawing.apng', '')
                existing.add(asset_name)
        return existing
    except Exception as e:
        print(f"Warning: Could not fetch existing files: {e}")
        return set()

def convert_svg_to_png_with_alpha(svg_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Convert SVG to PNG with alpha channel"""
    try:
        # Read SVG and convert to PNG bytes with transparency
        png_bytes = cairosvg.svg2png(url=svg_path, output_width=SVG_SIZE, output_height=SVG_SIZE)
        
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(png_bytes))
        
        # Ensure RGBA mode
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Split into BGR and alpha
        rgb = img_array[:,:,:3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        alpha = img_array[:,:,3]
        
        return bgr, alpha
    
    except Exception as e:
        print(f"  Error converting SVG to PNG: {e}")
        return None

def load_image_with_alpha(image_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load an image and return BGR and alpha channels"""
    if image_path.endswith('.svg'):
        return convert_svg_to_png_with_alpha(image_path)
    else:
        # Load PNG/JPG with alpha channel support
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        
        if img.shape[2] == 4:
            # Has alpha channel
            bgr = img[:,:,:3]
            alpha = img[:,:,3]
            return bgr, alpha
        else:
            # No alpha channel, create one (255 for all non-white pixels)
            bgr = img
            # Create alpha based on non-white pixels
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            alpha = np.ones_like(gray) * 255
            # Make white pixels transparent
            white_mask = gray > 240
            alpha[white_mask] = 0
            return bgr, alpha

def overlay_with_alpha(background_bgr: np.ndarray, background_alpha: np.ndarray,
                       overlay_bgr: np.ndarray, overlay_alpha: np.ndarray,
                       position: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Overlay an image with alpha onto another image with alpha"""
    x, y = position
    h, w = overlay_bgr.shape[:2]
    bg_h, bg_w = background_bgr.shape[:2]
    
    # Adjust position to fit within bounds
    x = max(0, min(x, bg_w - w))
    y = max(0, min(y, bg_h - h))
    
    if x + w > bg_w:
        w = bg_w - x
    if y + h > bg_h:
        h = bg_h - y
    
    if w <= 0 or h <= 0:
        return background_bgr, background_alpha
    
    # Get ROI
    roi_bgr = background_bgr[y:y+h, x:x+w]
    roi_alpha = background_alpha[y:y+h, x:x+w]
    
    # Crop overlay to fit
    overlay_bgr_crop = overlay_bgr[:h, :w]
    overlay_alpha_crop = overlay_alpha[:h, :w]
    
    # Normalize alpha channels
    overlay_alpha_norm = overlay_alpha_crop.astype(float) / 255
    roi_alpha_norm = roi_alpha.astype(float) / 255
    
    # Composite alpha
    output_alpha = overlay_alpha_norm + roi_alpha_norm * (1 - overlay_alpha_norm)
    
    # Avoid division by zero
    output_alpha_safe = np.where(output_alpha > 0, output_alpha, 1)
    
    # Composite color
    overlay_contrib = overlay_bgr_crop * overlay_alpha_norm[:,:,np.newaxis]
    roi_contrib = roi_bgr * roi_alpha_norm[:,:,np.newaxis] * (1 - overlay_alpha_norm[:,:,np.newaxis])
    output_bgr = (overlay_contrib + roi_contrib) / output_alpha_safe[:,:,np.newaxis]
    
    # Update background
    result_bgr = background_bgr.copy()
    result_alpha = background_alpha.copy()
    
    result_bgr[y:y+h, x:x+w] = output_bgr.astype(np.uint8)
    result_alpha[y:y+h, x:x+w] = (output_alpha * 255).astype(np.uint8)
    
    return result_bgr, result_alpha

def create_default_hand_image_with_alpha():
    """Create a simple hand drawing with alpha channel"""
    # Create transparent background (larger size for better visibility)
    hand_bgr = np.zeros((300, 225, 3), dtype=np.uint8)
    hand_alpha = np.zeros((300, 225), dtype=np.uint8)
    
    # Create a mask for the hand shape (scaled up for larger size)
    mask = np.zeros((300, 225), dtype=np.uint8)
    
    # Draw hand shape on mask (all coordinates scaled by 1.5)
    # Palm
    cv2.ellipse(mask, (112, 180), (60, 75), 0, 0, 360, 255, -1)
    
    # Fingers
    finger_positions = [(75, 105), (97, 90), (112, 82), (127, 90), (150, 105)]
    for x, y in finger_positions:
        cv2.ellipse(mask, (x, y), (12, 30), 0, 0, 360, 255, -1)
    
    # Thumb
    cv2.ellipse(mask, (60, 150), (22, 37), -30, 0, 360, 255, -1)
    
    # Apply skin color where mask is white
    skin_color = (177, 220, 255)  # BGR for skin tone
    hand_bgr[mask > 0] = skin_color
    hand_alpha[mask > 0] = 255
    
    # Add some shading (scaled up)
    shading_mask = np.zeros((300, 225), dtype=np.uint8)
    cv2.ellipse(shading_mask, (112, 180), (52, 67), 0, 0, 360, 255, 3)
    darker_skin = (160, 200, 240)
    hand_bgr[shading_mask > 0] = darker_skin
    
    return hand_bgr, hand_alpha

def create_drawing_animation_apng(asset_path: str, hand_bgr: Optional[np.ndarray], 
                                  hand_alpha: Optional[np.ndarray], verbose: bool = True) -> Optional[Tuple[str, List[Dict]]]:
    """
    Create a drawing animation as APNG with transparency.
    Returns tuple of (apng_path, hand_coordinates_list)
    """
    if verbose:
        print(f"\nProcessing asset: {asset_path}")
    
    # Load image with alpha channel
    result = load_image_with_alpha(asset_path)
    if result is None:
        if verbose:
            print("  Failed to load image")
        return None
    
    img_bgr, img_alpha = result
    
    # For segmentation, create a version without alpha for the existing function
    # Create a white background version for segmentation
    white_bg = np.ones_like(img_bgr) * 255
    alpha_3channel = img_alpha[:,:,np.newaxis].astype(float) / 255
    img_for_segmentation = (img_bgr * alpha_3channel + white_bg * (1 - alpha_3channel)).astype(np.uint8)
    
    # Save temporarily for segmentation
    temp_seg = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_seg_path = temp_seg.name
    temp_seg.close()
    cv2.imwrite(temp_seg_path, img_for_segmentation)
    
    # Segment the image
    if verbose:
        print("  Segmenting image with Mean Shift clustering...")
    cluster_map, _ = segment_image_with_meanshift(temp_seg_path)
    
    # Clean up temp file
    try:
        os.remove(temp_seg_path)
    except:
        pass
    
    if cluster_map is None:
        if verbose:
            print("  Failed to segment image")
        return None
    
    height, width = img_bgr.shape[:2]
    
    # Initialize drawing canvas with transparency
    canvas_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((height, width), dtype=np.uint8)
    
    # List to store hand coordinates for each frame
    hand_coordinates = []
    
    # Create reveal mask
    reveal_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate pen offset for hand position
    if hand_bgr is not None:
        pen_offset_x = hand_bgr.shape[1] // 3
        pen_offset_y = hand_bgr.shape[0] // 3
    else:
        pen_offset_x = pen_offset_y = 20
    
    # Analyze clusters
    if verbose:
        print("  Analyzing clusters...")
    cluster_info = get_cluster_info(cluster_map)
    
    if not cluster_info:
        if verbose:
            print("  No valid clusters found")
        return None
    
    if verbose:
        print(f"  Found {len(cluster_info)} clusters")
    
    # Order clusters
    ordered_clusters = order_clusters_clockwise(cluster_info)
    
    # Calculate frame allocation
    total_pixels = sum(c['pixel_count'] for c in cluster_info)
    total_frames = TOTAL_FRAMES
    
    frame_allocation = []
    remaining_frames = total_frames
    
    for i, cluster in enumerate(ordered_clusters):
        if i == len(ordered_clusters) - 1:
            frames = remaining_frames
        else:
            frames = int((cluster['pixel_count'] / total_pixels) * total_frames)
            frames = max(5, frames)
            remaining_frames -= frames
        frame_allocation.append(frames)
    
    if verbose:
        print("  Creating APNG animation with transparency...")
    
    # Create APNG assembler
    apng = apngasm.APNGAsm()
    
    # Generate frames
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
            # Process trajectory points
            for _ in range(points_per_frame):
                if trajectory_idx < len(trajectory):
                    pixels_to_reveal = pixel_mapping.get(trajectory_idx, [])
                    
                    for px, py in pixels_to_reveal:
                        cv2.circle(reveal_mask, (px, py), 12, 255, -1)
                    
                    if trajectory_idx < len(trajectory):
                        traj_x, traj_y = trajectory[trajectory_idx]
                        cv2.circle(reveal_mask, (traj_x, traj_y), 20, 255, -1)
                    
                    trajectory_idx += 1
            
            # Create frame with transparency
            frame_bgr = canvas_bgr.copy()
            frame_alpha = canvas_alpha.copy()
            
            # Apply revealed portions
            revealed_indices = reveal_mask > 0
            frame_bgr[revealed_indices] = img_bgr[revealed_indices]
            frame_alpha[revealed_indices] = img_alpha[revealed_indices]
            
            # Store hand coordinates for this frame (but don't overlay hand on the image)
            if trajectory_idx > 0 and trajectory_idx <= len(trajectory):
                hand_pos = trajectory[min(trajectory_idx - 1, len(trajectory) - 1)]
                hand_x = hand_pos[0] - pen_offset_x
                hand_y = hand_pos[1] - pen_offset_y
                hand_coordinates.append({
                    "x": int(hand_x),
                    "y": int(hand_y),
                    "visible": True
                })
            else:
                # No hand visible in this frame
                hand_coordinates.append({
                    "x": 0,
                    "y": 0,
                    "visible": False
                })
            
            # Convert frame to RGBA for APNG
            frame_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            frame_rgba[:,:,:3] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgba[:,:,3] = frame_alpha
            
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame_rgba, mode='RGBA')
            
            # Save frame temporarily
            temp_frame = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_frame_path = temp_frame.name
            temp_frame.close()
            pil_frame.save(temp_frame_path)
            
            # Add frame to APNG
            apng.add_frame_from_file(temp_frame_path, delay_num=FRAME_DELAY, delay_den=1000)
            
            # Clean up temp frame
            try:
                os.remove(temp_frame_path)
            except:
                pass
            
            frames_used += 1
            cluster_frames += 1
    
    # Check for missed pixels and do animated touch-ups
    missed_pixels = []
    for y in range(height):
        for x in range(width):
            if cluster_map[y, x] >= 0 and reveal_mask[y, x] == 0:
                missed_pixels.append((x, y))
    
    if missed_pixels and verbose:
        print(f"  Touch-up phase: {len(missed_pixels)} missed pixels")
    
    if missed_pixels:
        # Group missed pixels by proximity for more natural touch-up
        missed_groups = []
        used = set()
        
        for px, py in missed_pixels:
            if (px, py) in used:
                continue
            
            # Find nearby missed pixels
            group = [(px, py)]
            used.add((px, py))
            
            for qx, qy in missed_pixels:
                if (qx, qy) not in used:
                    dist = (px - qx)**2 + (py - qy)**2
                    if dist < 400:  # Within 20 pixels
                        group.append((qx, qy))
                        used.add((qx, qy))
            
            if group:
                # Calculate center of group for hand positioning
                center_x = int(np.mean([p[0] for p in group]))
                center_y = int(np.mean([p[1] for p in group]))
                missed_groups.append((center_x, center_y, group))
        
        # Animate touch-ups (0.5 seconds)
        touch_up_frames = 15
        groups_per_frame = max(1, len(missed_groups) // touch_up_frames)
        
        group_idx = 0
        for frame_num in range(touch_up_frames):
            # Create frame with current reveal state
            frame_bgr = canvas_bgr.copy()
            frame_alpha = canvas_alpha.copy()
            
            # Process some groups in this frame
            for _ in range(groups_per_frame):
                if group_idx < len(missed_groups):
                    center_x, center_y, group = missed_groups[group_idx]
                    
                    # Reveal all pixels in this group
                    for px, py in group:
                        cv2.circle(reveal_mask, (px, py), 8, 255, -1)
                    
                    group_idx += 1
            
            # Apply revealed portions
            revealed_indices = reveal_mask > 0
            frame_bgr[revealed_indices] = img_bgr[revealed_indices]
            frame_alpha[revealed_indices] = img_alpha[revealed_indices]
            
            # Store hand coordinates for touch-up frames (but don't overlay hand on the image)
            if group_idx > 0 and group_idx <= len(missed_groups):
                touch_pos = (missed_groups[min(group_idx - 1, len(missed_groups) - 1)][0],
                           missed_groups[min(group_idx - 1, len(missed_groups) - 1)][1])
                hand_x = touch_pos[0] - pen_offset_x
                hand_y = touch_pos[1] - pen_offset_y
                hand_coordinates.append({
                    "x": int(hand_x),
                    "y": int(hand_y),
                    "visible": True
                })
            else:
                # No hand visible in this frame
                hand_coordinates.append({
                    "x": 0,
                    "y": 0,
                    "visible": False
                })
            
            # Convert frame to RGBA and add to APNG
            frame_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            frame_rgba[:,:,:3] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgba[:,:,3] = frame_alpha
            
            pil_frame = Image.fromarray(frame_rgba, mode='RGBA')
            temp_frame = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_frame_path = temp_frame.name
            temp_frame.close()
            pil_frame.save(temp_frame_path)
            
            apng.add_frame_from_file(temp_frame_path, delay_num=FRAME_DELAY, delay_den=1000)
            
            try:
                os.remove(temp_frame_path)
            except:
                pass
    
    # Add final frames showing completed image
    final_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    final_rgba[:,:,:3] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    final_rgba[:,:,3] = img_alpha
    
    final_pil = Image.fromarray(final_rgba, mode='RGBA')
    temp_final = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_final_path = temp_final.name
    temp_final.close()
    final_pil.save(temp_final_path)
    
    # Add final frame multiple times
    for _ in range(30):  # 1 second at 30fps
        apng.add_frame_from_file(temp_final_path, delay_num=FRAME_DELAY, delay_den=1000)
        # No hand visible in final frames
        hand_coordinates.append({
            "x": 0,
            "y": 0,
            "visible": False
        })
    
    try:
        os.remove(temp_final_path)
    except:
        pass
    
    # Save APNG
    temp_apng = tempfile.NamedTemporaryFile(suffix='.apng', delete=False)
    temp_apng_path = temp_apng.name
    temp_apng.close()
    
    apng.assemble(temp_apng_path)
    
    if verbose:
        print(f"  APNG created: {temp_apng_path}")
    return temp_apng_path, hand_coordinates

def upload_to_supabase(apng_path: str, hand_coords_json_path: str, asset_name: str, force: bool = False) -> bool:
    """Upload APNG and hand coordinates JSON to Supabase storage"""
    try:
        apng_filename = f"{asset_name}_drawing.apng"
        json_filename = f"{asset_name}_drawing_hand_coordinates.json"
        
        # Check if already exists
        if not force:
            try:
                files = supabase.storage.from_(BUCKET_NAME).list()
                if any(f['name'] == apng_filename for f in files) and any(f['name'] == json_filename for f in files):
                    print(f"  Skipping (already exists): {apng_filename} and {json_filename}")
                    return True
            except:
                pass
        
        # Remove existing if force mode
        if force:
            try:
                files = supabase.storage.from_(BUCKET_NAME).list()
                files_to_remove = []
                if any(f['name'] == apng_filename for f in files):
                    files_to_remove.append(apng_filename)
                if any(f['name'] == json_filename for f in files):
                    files_to_remove.append(json_filename)
                if files_to_remove:
                    supabase.storage.from_(BUCKET_NAME).remove(files_to_remove)
                    print(f"  Removed existing: {', '.join(files_to_remove)}")
            except:
                pass
        
        # Read APNG file
        with open(apng_path, 'rb') as f:
            apng_data = f.read()
        
        # Read JSON file
        with open(hand_coords_json_path, 'rb') as f:
            json_data = f.read()
        
        # Upload APNG to Supabase
        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=apng_filename,
            file=apng_data,
            file_options={"content-type": "image/apng"}
        )
        
        # Upload JSON to Supabase
        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=json_filename,
            file=json_data,
            file_options={"content-type": "application/json"}
        )
        
        print(f"  Uploaded to Supabase: {apng_filename} and {json_filename}")
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

def main():
    """Main function to process assets"""
    parser = argparse.ArgumentParser(description='Render and save asset animations as APNG files')
    parser.add_argument('--all', action='store_true', 
                       help='Process all assets, even if they already exist in the bucket')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed processing information')
    parser.add_argument('--assets-dir', default=ASSETS_DIR,
                       help=f'Directory containing assets (default: {ASSETS_DIR})')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Asset Animation Renderer - APNG with Transparency")
    print("=" * 60)
    
    # Ensure bucket exists
    ensure_bucket_exists()
    
    # Get existing APNGs if not in --all mode
    existing_apngs = set()
    if not args.all:
        print("\nChecking existing APNGs in bucket...")
        existing_apngs = get_existing_apngs()
        if existing_apngs:
            print(f"Found {len(existing_apngs)} existing APNGs")
            if args.verbose:
                for name in sorted(existing_apngs):
                    print(f"  - {name}")
    else:
        print("\n--all flag set: Will process all assets")
    
    # Load the hand image
    hand_bgr, hand_alpha = load_hand_image(HAND_IMAGE_PATH)
    
    if hand_bgr is None:
        print(f"\nError: Could not load hand image from {HAND_IMAGE_PATH}")
        print("Please ensure the hand.png file exists at the specified path.")
        sys.exit(1)
    
    if hand_bgr is not None:
        # Scale hand image
        new_width = int(hand_bgr.shape[1] * HAND_SCALE)
        new_height = int(hand_bgr.shape[0] * HAND_SCALE)
        hand_bgr = cv2.resize(hand_bgr, (new_width, new_height))
        if hand_alpha is not None:
            hand_alpha = cv2.resize(hand_alpha, (new_width, new_height))
    
    # Find all assets
    asset_patterns = [
        os.path.join(args.assets_dir, "*.png"),
        os.path.join(args.assets_dir, "*.jpg"),
        os.path.join(args.assets_dir, "*.jpeg"),
        os.path.join(args.assets_dir, "*.svg")
    ]
    
    asset_files = []
    for pattern in asset_patterns:
        asset_files.extend(glob.glob(pattern))
    
    if not asset_files:
        print(f"\nNo assets found in {args.assets_dir}")
        return
    
    print(f"\nFound {len(asset_files)} total assets")
    
    # Filter out existing assets if not in --all mode
    assets_to_process = []
    skipped_assets = []
    
    for asset_path in asset_files:
        asset_name = Path(asset_path).stem
        if not args.all and asset_name in existing_apngs:
            skipped_assets.append(asset_name)
        else:
            assets_to_process.append(asset_path)
    
    if skipped_assets and args.verbose:
        print(f"\nSkipping {len(skipped_assets)} assets that already have APNGs:")
        for name in sorted(skipped_assets):
            print(f"  - {name}")
    
    if not assets_to_process:
        print("\nNo new assets to process!")
        print("Use --all flag to re-process all assets")
        return
    
    print(f"\nProcessing {len(assets_to_process)} assets...")
    print("-" * 40)
    
    # Process each asset
    successful = 0
    failed = 0
    
    for i, asset_path in enumerate(assets_to_process, 1):
        asset_name = Path(asset_path).stem
        
        print(f"\n[{i}/{len(assets_to_process)}] {asset_name}")
        
        try:
            # Render the asset as APNG with transparency and get hand coordinates
            result = create_drawing_animation_apng(asset_path, hand_bgr, hand_alpha, 
                                                     verbose=args.verbose)
            
            if result:
                apng_path, hand_coords = result
                
                # Save hand coordinates to JSON file
                json_temp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                json_temp_path = json_temp.name
                json_temp.close()
                
                with open(json_temp_path, 'w') as f:
                    json.dump(hand_coords, f, indent=2)
                
                # Upload both files to Supabase
                if upload_to_supabase(apng_path, json_temp_path, asset_name, force=args.all):
                    successful += 1
                    print(f"  ✓ Success")
                else:
                    failed += 1
                    print(f"  ✗ Upload failed")
                
                # Clean up temporary files
                try:
                    os.remove(apng_path)
                    os.remove(json_temp_path)
                except:
                    pass
            else:
                failed += 1
                print(f"  ✗ Failed to create APNG")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if skipped_assets:
        print(f"  Skipped (already exist): {len(skipped_assets)}")
    print(f"\nAPNG animations stored in Supabase bucket: {BUCKET_NAME}")
    print("All animations have transparent backgrounds!")
    
    # Return exit code based on success
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()