#!/usr/bin/env python3
"""
Split grid video into individual videos using FFmpeg chromakey with despill
ORIGINAL GOOD VERSION - just chromakey + despill, no other modifications
"""

import subprocess
import json
from pathlib import Path
import sys

def split_grid_with_ffmpeg_chromakey(video_path, grid_size, output_dir, image_names):
    """Split grid video and apply FFmpeg chromakey with despill - ORIGINAL GOOD VERSION"""
    
    cols, rows = grid_size
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    print(f"\n✂️ Splitting {cols}x{rows} grid with FFmpeg chromakey + despill (ORIGINAL VERSION)...")
    
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
    print(f"  Using ORIGINAL settings that worked well")
    
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
        output_file = output_dir / f"{clean_name}_despill.webm"
        
        print(f"  [{i+1}/{min(len(image_names), cols*rows)}] {output_file.name}")
        
        # ORIGINAL WORKING COMMAND - NO MODIFICATIONS
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-filter_complex', 
            f"[0:v]crop={cell_width}:{cell_height}:{x}:{y}[cropped]; "
            f"[cropped]chromakey=green:0.10:0.08[ck]; "
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

if __name__ == "__main__":
    # Get the image names from the original folder
    batch_dir = Path("uploads/assets/batch_images_transparent_bg")
    image_files = sorted([f.name for f in batch_dir.glob("*.png")])
    
    # Split the despill video
    video_path = "output/bulk_batch_images_transparent_bg_20250815_114404/kling_animation.mp4"
    output_dir = Path("output/bulk_batch_images_transparent_bg_20250815_114404")
    
    files = split_grid_with_ffmpeg_chromakey(
        video_path,
        grid_size=(5, 2),
        output_dir=output_dir,
        image_names=image_files[:10]
    )
    
    print(f"\n✅ Created {len(files)} despill WebM files (ORIGINAL VERSION) in:")
    print(f"   {output_dir}")
    print("\nNote: These may have a slight aura in frame 1, but the rest looks perfect!")