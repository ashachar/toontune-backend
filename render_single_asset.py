#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
import tempfile
import cairosvg
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('VITE_SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Error: Supabase credentials not found in .env file")
    sys.exit(1)

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

BUCKET_NAME = "rendered-assets"
FPS = 30

def convert_svg_to_png(svg_path):
    """Convert SVG to PNG using cairosvg"""
    try:
        png_bytes = cairosvg.svg2png(url=svg_path, output_width=400, output_height=400)
        img = Image.open(io.BytesIO(png_bytes))
        img_array = np.array(img)
        
        if img_array.shape[2] == 4:
            bgr = cv2.cvtColor(img_array[:,:,:3], cv2.COLOR_RGB2BGR)
            return bgr
        else:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error converting SVG: {e}")
        return None

def create_simple_drawing_video(asset_path):
    """Create a simple drawing animation"""
    print(f"Processing: {asset_path}")
    
    # Load or convert image
    if asset_path.endswith('.svg'):
        img = convert_svg_to_png(asset_path)
        if img is None:
            return None
    else:
        img = cv2.imread(asset_path)
        if img is None:
            return None
    
    height, width = img.shape[:2]
    
    # Create temporary video file
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video_path = temp_video.name
    temp_video.close()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, FPS, (width, height))
    
    # Simple reveal animation - reveal from top to bottom
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    total_frames = 60  # 2 seconds
    
    for frame_num in range(total_frames):
        frame = canvas.copy()
        
        # Calculate how much to reveal
        reveal_height = int((frame_num / total_frames) * height)
        
        # Copy revealed portion
        if reveal_height > 0:
            frame[:reveal_height] = img[:reveal_height]
        
        # Add a drawing indicator line
        if reveal_height < height:
            cv2.line(frame, (0, reveal_height), (width, reveal_height), (100, 100, 100), 2)
        
        video_writer.write(frame)
    
    # Add final frames showing completed image
    for _ in range(30):
        video_writer.write(img)
    
    video_writer.release()
    
    print(f"  Video created: {temp_video_path}")
    return temp_video_path

def upload_to_supabase(video_path, asset_name):
    """Upload video to Supabase storage"""
    try:
        video_filename = f"{asset_name}_drawing.mp4"
        
        # Check if exists and remove
        try:
            files = supabase.storage.from_(BUCKET_NAME).list()
            if any(f['name'] == video_filename for f in files):
                supabase.storage.from_(BUCKET_NAME).remove([video_filename])
                print(f"  Removed existing: {video_filename}")
        except:
            pass
        
        # Upload new file
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        response = supabase.storage.from_(BUCKET_NAME).upload(
            path=video_filename,
            file=video_data,
            file_options={"content-type": "video/mp4"}
        )
        
        print(f"  Uploaded: {video_filename}")
        return True
        
    except Exception as e:
        print(f"  Upload error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python render_single_asset.py <asset_path>")
        sys.exit(1)
    
    asset_path = sys.argv[1]
    
    if not os.path.exists(asset_path):
        print(f"File not found: {asset_path}")
        sys.exit(1)
    
    asset_name = Path(asset_path).stem
    
    # Create video
    video_path = create_simple_drawing_video(asset_path)
    
    if video_path:
        # Upload to Supabase
        success = upload_to_supabase(video_path, asset_name)
        
        # Clean up
        try:
            os.remove(video_path)
        except:
            pass
        
        if success:
            print("Success!")
        else:
            print("Upload failed")
            sys.exit(1)
    else:
        print("Video creation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()