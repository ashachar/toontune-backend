#!/usr/bin/env python3
"""
Split grid video with first frame fix - different approach
Replace first frame with second frame to avoid initialization artifacts
"""

import subprocess
import json
from pathlib import Path
import sys

def split_grid_with_first_frame_fix(video_path, grid_size, output_dir, image_names):
    """Split grid video and fix first frame by duplicating frame 2"""
    
    cols, rows = grid_size
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    print(f"\n✂️ Splitting {cols}x{rows} grid with first-frame duplication fix...")
    
    # Get video dimensions
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    video_info = json.loads(result.stdout)
    video_width = video_info['streams'][0]['width']
    video_height = video_info['streams'][0]['height']
    
    # Calculate cell dimensions
    cell_width = video_width // cols
    cell_height = video_height // rows
    
    print(f"  Video: {video_width}x{video_height}")
    print(f"  Cell: {cell_width}x{cell_height}")
    print(f"  Strategy: Replace frame 1 with frame 2 to avoid artifacts")
    
    output_files = []
    
    for i, name in enumerate(image_names):
        if i >= cols * rows:
            break
            
        row = i // cols
        col = i % cols
        
        # Calculate crop position
        x = col * cell_width
        y = row * cell_height
        
        # Clean filename
        if isinstance(name, Path):
            name = name.stem
        clean_name = name.replace(' ', '_').replace('.png', '')
        output_file = output_dir / f"{clean_name}_clean.webm"
        
        print(f"  [{i+1}/{min(len(image_names), cols*rows)}] {output_file.name}")
        
        # Complex filter that:
        # 1. Crops the video
        # 2. Duplicates frame 2 as frame 1 (using select and concat)
        # 3. Applies the good chromakey+despill settings
        cmd = [
            'ffmpeg', 
            '-i', str(video_path),
            '-filter_complex', 
            # First, crop the video
            f"[0:v]crop={cell_width}:{cell_height}:{x}:{y}[cropped]; "
            # Extract frame 2 and all frames from 2 onwards
            f"[cropped]select='eq(n,1)'[frame2]; "
            f"[cropped]select='gte(n,1)'[rest]; "
            # Concatenate frame2 + rest (effectively replacing frame 1 with frame 2)
            f"[frame2][rest]concat=n=2:v=1[fixed]; "
            # Apply the good chromakey settings
            f"[fixed]chromakey=green:0.10:0.08[ck]; "
            f"[ck]despill=type=green:mix=0.6[dsp]; "
            f"[dsp]format=yuva420p[out]",
            '-map', '[out]',
            '-c:v', 'libvpx-vp9',
            '-pix_fmt', 'yuva420p',
            '-b:v', '0',
            '-crf', '25',
            str(output_file),
            '-y',
            '-loglevel', 'error'
        ]
        
        subprocess.run(cmd, check=True)
        output_files.append(output_file)
        print(f"    ✓ Created: {output_file.name}")
    
    return output_files

# Alternative simpler approach - just process twice and replace first frame
def split_grid_simple_fix(video_path, grid_size, output_dir, image_names):
    """Simpler approach: process normally then fix first frame in post"""
    
    cols, rows = grid_size
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    print(f"\n✂️ Splitting {cols}x{rows} grid (simple first-frame fix)...")
    
    # Get video dimensions
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    video_info = json.loads(result.stdout)
    video_width = video_info['streams'][0]['width']
    video_height = video_info['streams'][0]['height']
    
    # Calculate cell dimensions
    cell_width = video_width // cols
    cell_height = video_height // rows
    
    print(f"  Video: {video_width}x{video_height}")
    print(f"  Cell: {cell_width}x{cell_height}")
    
    output_files = []
    
    for i, name in enumerate(image_names):
        if i >= cols * rows:
            break
            
        row = i // cols
        col = i % cols
        
        # Calculate crop position
        x = col * cell_width
        y = row * cell_height
        
        # Clean filename
        if isinstance(name, Path):
            name = name.stem
        clean_name = name.replace(' ', '_').replace('.png', '')
        output_file = output_dir / f"{clean_name}_final.webm"
        
        print(f"  [{i+1}/{min(len(image_names), cols*rows)}] {output_file.name}")
        
        # Approach: Use trim to start from frame 1 (skip frame 0), then loop first frame
        cmd = [
            'ffmpeg', 
            '-i', str(video_path),
            '-filter_complex', 
            f"[0:v]crop={cell_width}:{cell_height}:{x}:{y}[cropped]; "
            # Trim to start from frame 1 (skip problematic frame 0)
            f"[cropped]trim=start_frame=1,setpts=PTS-STARTPTS[trimmed]; "
            # Apply chromakey with original good settings
            f"[trimmed]chromakey=green:0.10:0.08[ck]; "
            f"[ck]despill=type=green:mix=0.6[dsp]; "
            # Loop the first (good) frame to maintain duration
            f"[dsp]loop=1:1:0,format=yuva420p[out]",
            '-map', '[out]',
            '-c:v', 'libvpx-vp9',
            '-pix_fmt', 'yuva420p',
            '-b:v', '0',
            '-crf', '25',
            str(output_file),
            '-y',
            '-loglevel', 'error'
        ]
        
        subprocess.run(cmd, check=True)
        output_files.append(output_file)
        print(f"    ✓ Created: {output_file.name}")
    
    return output_files

if __name__ == "__main__":
    # Get the image names from the original folder
    batch_dir = Path("uploads/assets/batch_images_transparent_bg")
    image_files = sorted([f.name for f in batch_dir.glob("*.png")])
    
    # Split with simple fix
    video_path = "output/bulk_batch_images_transparent_bg_20250815_114404/kling_animation.mp4"
    output_dir = Path("output/bulk_batch_images_transparent_bg_20250815_114404")
    
    files = split_grid_simple_fix(
        video_path,
        grid_size=(5, 2),
        output_dir=output_dir,
        image_names=image_files[:10]
    )
    
    print(f"\n✅ Created {len(files)} final WebM files (first frame fixed) in:")
    print(f"   {output_dir}")