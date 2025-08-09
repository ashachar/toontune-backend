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
FRAMES_PER_CLUSTER = 15  # Base frames allocated per cluster
MIN_FRAMES_PER_CLUSTER = 10  # Minimum frames for small clusters
FINAL_FRAMES = 30  # 1 second showing completed image
SVG_SIZE = 600  # Good size for SVG conversion
FRAME_DELAY = 33  # milliseconds per frame (approximately 30 FPS)

def get_existing_jsons() -> Set[str]:
    """Get a set of asset names that already have coordinate JSONs in the bucket"""
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()
        # Extract asset names from JSON filenames (remove _drawing_hand_coordinates.json suffix)
        existing = set()
        for f in files:
            if f['name'].endswith('_drawing_hand_coordinates.json'):
                asset_name = f['name'].replace('_drawing_hand_coordinates.json', '')
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

def generate_hand_coordinates(asset_path: str, hand_bgr: Optional[np.ndarray], 
                             hand_alpha: Optional[np.ndarray], verbose: bool = True) -> Optional[List[Dict]]:
    """
    Generate hand coordinates for drawing animation.
    Returns list of hand coordinates (only visible ones).
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
    
    # List to store hand coordinates for each frame (only visible ones)
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
    
    # Calculate dynamic frame allocation based on cluster count and size
    total_pixels = sum(c['pixel_count'] for c in cluster_info)
    
    frame_allocation = []
    
    for cluster in ordered_clusters:
        # Allocate frames proportional to cluster size, with a minimum
        pixel_ratio = cluster['pixel_count'] / total_pixels
        frames = max(MIN_FRAMES_PER_CLUSTER, int(FRAMES_PER_CLUSTER * (1 + pixel_ratio * 2)))
        frame_allocation.append(frames)
    
    # Total frames will be dynamic based on cluster count and sizes
    total_frames = sum(frame_allocation)
    
    if verbose:
        print(f"  Frame allocation: {frame_allocation} (total: {total_frames} frames)")
        print("  Generating hand coordinates...")
    
    # Generate coordinates for frames
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
            
            # Store hand coordinates for this frame
            if trajectory_idx > 0 and trajectory_idx <= len(trajectory):
                hand_pos = trajectory[min(trajectory_idx - 1, len(trajectory) - 1)]
                hand_x = hand_pos[0] - pen_offset_x
                hand_y = hand_pos[1] - pen_offset_y
                hand_coordinates.append({
                    "x": int(hand_x),
                    "y": int(hand_y)
                })
            else:
                # No hand visible in this frame
                hand_coordinates.append({
                    "x": 0,
                    "y": 0
                })
            
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
        
        # Animate touch-ups dynamically based on number of groups
        touch_up_frames = min(30, max(5, len(missed_groups) // 2))
        groups_per_frame = max(1, len(missed_groups) // touch_up_frames)
        
        group_idx = 0
        for frame_num in range(touch_up_frames):
            # Process some groups in this frame
            for _ in range(groups_per_frame):
                if group_idx < len(missed_groups):
                    center_x, center_y, group = missed_groups[group_idx]
                    
                    # Reveal all pixels in this group
                    for px, py in group:
                        cv2.circle(reveal_mask, (px, py), 8, 255, -1)
                    
                    group_idx += 1
            
            # Store hand coordinates for touch-up frames
            if group_idx > 0 and group_idx <= len(missed_groups):
                touch_pos = (missed_groups[min(group_idx - 1, len(missed_groups) - 1)][0],
                           missed_groups[min(group_idx - 1, len(missed_groups) - 1)][1])
                hand_x = touch_pos[0] - pen_offset_x
                hand_y = touch_pos[1] - pen_offset_y
                hand_coordinates.append({
                    "x": int(hand_x),
                    "y": int(hand_y)
                })
            else:
                # No hand visible in this frame
                hand_coordinates.append({
                    "x": 0,
                    "y": 0
                })
    
    # Add final frames showing completed image (hand not visible)
    for _ in range(FINAL_FRAMES):
        hand_coordinates.append({
            "x": 0,
            "y": 0
        })
    
    if verbose:
        print(f"  Generated {len(hand_coordinates)} hand coordinate entries")
    
    return hand_coordinates

def upload_json_to_supabase(json_path: str, asset_name: str, force: bool = False) -> bool:
    """Upload hand coordinates JSON to Supabase storage"""
    try:
        json_filename = f"{asset_name}_drawing_hand_coordinates.json"
        
        # Check if already exists
        if not force:
            try:
                files = supabase.storage.from_(BUCKET_NAME).list()
                if any(f['name'] == json_filename for f in files):
                    print(f"  Skipping (already exists): {json_filename}")
                    return True
            except:
                pass
        
        # Remove existing if force mode
        if force:
            try:
                files = supabase.storage.from_(BUCKET_NAME).list()
                if any(f['name'] == json_filename for f in files):
                    supabase.storage.from_(BUCKET_NAME).remove([json_filename])
                    print(f"  Removed existing: {json_filename}")
            except:
                pass
        
        # Read JSON file
        with open(json_path, 'rb') as f:
            json_data = f.read()
        
        # Upload JSON to Supabase
        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=json_filename,
            file=json_data,
            file_options={"content-type": "application/json"}
        )
        
        print(f"  Uploaded to Supabase: {json_filename}")
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
    parser = argparse.ArgumentParser(description='Generate and save hand coordinates for asset drawing animations')
    parser.add_argument('--all', action='store_true', 
                       help='Process all assets, even if they already exist in the bucket')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed processing information')
    parser.add_argument('--assets-dir', default=ASSETS_DIR,
                       help=f'Directory containing assets (default: {ASSETS_DIR})')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Asset Hand Coordinates Generator")
    print("=" * 60)
    
    # Ensure bucket exists
    ensure_bucket_exists()
    
    # Get existing JSONs if not in --all mode
    existing_jsons = set()
    if not args.all:
        print("\nChecking existing coordinate JSONs in bucket...")
        existing_jsons = get_existing_jsons()
        if existing_jsons:
            print(f"Found {len(existing_jsons)} existing coordinate JSONs")
            if args.verbose:
                for name in sorted(existing_jsons):
                    print(f"  - {name}")
    else:
        print("\n--all flag set: Will process all assets")
    
    # Load the hand image (for calculating offsets)
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
        if not args.all and asset_name in existing_jsons:
            skipped_assets.append(asset_name)
        else:
            assets_to_process.append(asset_path)
    
    if skipped_assets and args.verbose:
        print(f"\nSkipping {len(skipped_assets)} assets that already have coordinate JSONs:")
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
            # Generate hand coordinates
            hand_coords = generate_hand_coordinates(asset_path, hand_bgr, hand_alpha, 
                                                   verbose=args.verbose)
            
            if hand_coords:
                # Save hand coordinates to JSON file
                json_temp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                json_temp_path = json_temp.name
                json_temp.close()
                
                with open(json_temp_path, 'w') as f:
                    json.dump(hand_coords, f, indent=2)
                
                # Upload JSON to Supabase
                if upload_json_to_supabase(json_temp_path, asset_name, force=args.all):
                    successful += 1
                    print(f"  ✓ Success")
                else:
                    failed += 1
                    print(f"  ✗ Upload failed")
                
                # Clean up temporary file
                try:
                    os.remove(json_temp_path)
                except:
                    pass
            else:
                failed += 1
                print(f"  ✗ Failed to generate coordinates")
        
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
    print(f"\nHand coordinate JSONs stored in Supabase bucket: {BUCKET_NAME}")
    print("JSONs contain all hand positions (x:0, y:0 means hand not visible)!")
    
    # Return exit code based on success
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()