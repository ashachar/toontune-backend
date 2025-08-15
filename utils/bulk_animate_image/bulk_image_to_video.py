#!/usr/bin/env python3
"""
Bulk Image to Video Pipeline with BLIP Captioning
Processes multiple images in a grid to save 75-94% on API costs
Default: 3x3 grid (9 images)
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
import replicate
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set REPLICATE_API_TOKEN from REPLICATE_API_KEY if needed
if not os.getenv('REPLICATE_API_TOKEN') and os.getenv('REPLICATE_API_KEY'):
    os.environ['REPLICATE_API_TOKEN'] = os.getenv('REPLICATE_API_KEY')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "image-cap"))

# Import smart transparency functions
from smart_transparency_image import smart_remove_background
from smart_transparency_video import apply_smart_transparency_to_video

# Import BLIP captioner
from simple_captioner import BlipCaptioner

# Import Replicate balance hooks
from replicate_balance_hooks import pre_inference_balance_check, post_inference_balance_check

def validate_input_folder(folder_path, expected_count=None):
    """Validate input folder and return list of image files"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Error: Folder {folder} does not exist")
        return None
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = sorted([
        f for f in folder.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"❌ Error: No image files found in {folder}")
        return None
    
    print(f"📁 Auto-detected {len(image_files)} images in folder")
    
    # Check expected count if provided
    if expected_count and len(image_files) != expected_count:
        print(f"❌ Error: Expected {expected_count} images but found {len(image_files)}")
        print("  Found images:")
        for img in image_files[:10]:
            print(f"    - {img.name}")
        if len(image_files) > 10:
            print(f"    ... and {len(image_files) - 10} more")
        return None
    
    print(f"✓ Found {len(image_files)} images in {folder}")
    return image_files

def calculate_grid_dimensions(num_images):
    """Calculate optimal grid dimensions for given number of images"""
    if num_images <= 4:
        cols = min(2, num_images)
        rows = math.ceil(num_images / cols)
    elif num_images <= 9:
        cols = min(3, num_images)
        rows = math.ceil(num_images / cols)
    elif num_images <= 16:
        cols = 4
        rows = math.ceil(num_images / cols)
    else:
        # For larger numbers, try to make it as square as possible
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
    
    return (cols, rows)

def sanitize_filename(filename):
    """Replace spaces and special characters in filename"""
    # Keep only alphanumeric, dash, underscore, and dot
    import re
    name = Path(filename).stem
    ext = Path(filename).suffix
    
    # Replace spaces and other chars with underscore
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    return f"{clean_name}{ext}"

def create_image_grid(image_paths, output_path, grid_size=None):
    """Create a grid of images with smart background removal"""
    
    if grid_size is None:
        grid_size = calculate_grid_dimensions(len(image_paths))
    
    cols, rows = grid_size
    print(f"\n📊 Creating {cols}x{rows} grid...")
    
    # Process each image with smart background removal
    processed_images = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n  [{i}/{len(image_paths)}] Processing {Path(img_path).name}")
        processed_img = smart_remove_background(str(img_path), alpha_matting=False)
        processed_images.append(processed_img)
    
    # Find max dimensions
    max_width = max(img.width for img in processed_images)
    max_height = max(img.height for img in processed_images)
    
    # Ensure dimensions are even (required for video encoding)
    cell_width = max_width + (max_width % 2)
    cell_height = max_height + (max_height % 2)
    
    # Create grid
    grid_width = cell_width * cols
    grid_height = cell_height * rows
    
    grid = Image.new('RGBA', (grid_width, grid_height), (0, 0, 0, 0))
    
    # Place images in grid
    for i, img in enumerate(processed_images):
        if i >= cols * rows:
            break
            
        row = i // cols
        col = i % cols
        
        # Center image in cell
        x = col * cell_width + (cell_width - img.width) // 2
        y = row * cell_height + (cell_height - img.height) // 2
        
        grid.paste(img, (x, y), img if img.mode == 'RGBA' else None)
    
    # Save grid
    grid.save(output_path)
    print(f"  ✓ Grid saved: {grid_width}x{grid_height} pixels")
    
    return output_path, (cell_width, cell_height)

def caption_single_image(img_path):
    """Caption a single image using BLIP - must be at module level for pickling"""
    captioner = BlipCaptioner(device="auto")
    caption, _ = captioner.caption_image(str(img_path))
    return caption

def generate_blip_captions_parallel(image_paths, max_workers=4):
    """Generate captions using BLIP in parallel"""
    print(f"\n🤖 Generating captions for {len(image_paths)} images...")
    
    captions = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(caption_single_image, path): path 
            for path in image_paths
        }
        
        completed = 0
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                caption = future.result()
                captions[path] = caption
                completed += 1
                
                # Show progress at key milestones
                if completed <= 3 or completed % 4 == 0 or completed == len(image_paths):
                    print(f"  Captioned {completed}/{len(image_paths)} images...")
                    
            except Exception as e:
                captions[path] = "professional character"
                print(f"  Warning: Caption failed for {path.name}: {e}")
    
    return captions

def build_dynamic_prompt(image_paths, grid_size, captions):
    """Build dynamic prompt based on image captions and positions"""
    cols, rows = grid_size
    
    prompt_parts = [f"{cols}x{rows} grid of characters:"]
    
    # Add descriptions for each grid position
    for i, path in enumerate(image_paths):
        if i >= cols * rows:
            break
            
        row = i // cols + 1
        col = i % cols + 1
        
        # Get caption or use default
        caption = captions.get(path, "professional character")
        
        # Truncate very long captions
        if len(caption) > 80:
            # Try to cut at word boundary
            caption = caption[:77] + "..."
        
        # Remove "a " or "an " from start if present
        if caption.startswith("a "):
            caption = caption[2:]
        elif caption.startswith("an "):
            caption = caption[3:]
        
        prompt_parts.append(f"Position ({row},{col}): {caption}")
    
    # Add movement instructions
    prompt_parts.append("Each character with subtle appropriate movements:")
    prompt_parts.append("office workers gentle typing, paper shuffling")
    prompt_parts.append("uniformed figures slight posture shifts")
    prompt_parts.append("all maintaining natural idle animations, independent panel movements")
    
    return " ".join(prompt_parts)

def generate_kling_animation(image_path, prompt, output_dir, resume=False):
    """Generate animation using Kling API"""
    
    output_path = output_dir / "kling_animation.mp4"
    
    # Check if already exists (resume mode)
    if resume and output_path.exists():
        print("  ✓ Animation already exists (resume mode)")
        return output_path
    
    print("\n🎬 Generating animation with Kling v2.1...")
    print(f"  Prompt ({len(prompt)} chars): {prompt[:200]}...")
    
    # Enhanced negative prompt for stability
    negative_prompt = (
        "morphing, warping, distortion, blinking, flickering, "
        "color change, hue shift, saturation change, brightness change, "
        "face distortion, mouth morphing, eye blinking rapidly, "
        "shape shifting, melting, dissolving, transforming, "
        "sudden movements, jerky motion, glitches, artifacts, "
        "background changes, scene changes, camera movement, "
        "zooming, panning, rotating, perspective change, "
        "style change, art style shift, quality degradation, "
        "blur increase, sharpness loss, detail loss, "
        "character interaction, crossing panels, merging panels, "
        "synchronized movement, dependent movement between panels, "
        "moments without any movement, static frames, frozen animation, "
        "complete stillness, no motion at all, stationary, "
        "interdependent animations, characters looking at each other, "
        "shared movements between different panels"
    )
    
    print(f"  Negative: {len(negative_prompt)} chars (enhanced anti-distortion)")
    
    try:
        # Ensure API token is set
        api_token = os.getenv('REPLICATE_API_TOKEN') or os.getenv('REPLICATE_API_KEY')
        if not api_token:
            print("  ❌ Error: No Replicate API token found in environment")
            return None
        
        # Create client with explicit token
        client = replicate.Client(api_token=api_token)
        
        # Use STANDARD Kling model (not master)
        model_id = "kwaivgi/kling-v2.1"
        duration = 5
        
        # PRE-INFERENCE BALANCE CHECK
        pre_inference_balance_check()
        
        output = client.run(
            model_id,
            input={
                "mode": "standard",  # 720p
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "start_image": open(image_path, "rb"),
                "duration": duration
            }
        )
        
        # POST-INFERENCE BALANCE CHECK
        post_inference_balance_check(model_name=model_id, duration=duration)
        
        # Download result
        if output:
            video_url = str(output)
            print(f"  ✓ Animation generated: {video_url}")
            
            # Download with wget
            cmd = ['wget', '-O', str(output_path), video_url, '--quiet']
            subprocess.run(cmd, check=True)
            
            print(f"  ✓ Downloaded to: {output_path}")
            return output_path
        else:
            print("  ❌ No output from Kling")
            return None
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def split_grid_video(video_path, grid_size, cell_size, image_names, output_dir):
    """Split grid video into individual WebM videos with transparency"""
    cols, rows = grid_size
    cell_width, cell_height = cell_size
    
    print(f"\n✂️ Splitting {cols}x{rows} grid into individual videos...")
    
    # Get actual video dimensions
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
    
    # Calculate actual cell dimensions from video
    actual_cell_width = video_width // cols
    actual_cell_height = video_height // rows
    
    print(f"  Video dimensions: {video_width}x{video_height}")
    print(f"  Cell dimensions: {actual_cell_width}x{actual_cell_height}")
    
    output_files = []
    
    for i, img_name in enumerate(image_names):
        if i >= cols * rows:
            break
            
        row = i // cols
        col = i % cols
        
        # Calculate crop position
        x = col * actual_cell_width
        y = row * actual_cell_height
        
        # Clean filename
        clean_name = sanitize_filename(img_name.stem if hasattr(img_name, 'stem') else Path(img_name).stem)
        output_file = output_dir / f"{clean_name}.webm"
        
        print(f"  [{i+1}/{min(len(image_names), cols*rows)}] Creating {output_file.name}...")
        
        # Create temp MP4 with crop
        temp_mp4 = output_dir / f"temp_{clean_name}.mp4"
        
        # Crop video
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'crop={actual_cell_width}:{actual_cell_height}:{x}:{y}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            str(temp_mp4),
            '-y',
            '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)
        
        # Apply smart transparency and convert to WebM
        grid_image = output_dir / "grid.png"
        apply_smart_transparency_to_video(temp_mp4, output_file, grid_image)
        
        # Clean up temp file
        temp_mp4.unlink()
        
        output_files.append(output_file)
    
    return output_files

def bulk_image_to_video(input_folder, output_name=None, grid_size=None, image_count=None, resume=False, test_mode=False):
    """
    Main pipeline: Folder → Captions → Grid → Animation → Split WebM videos
    
    Args:
        input_folder: Path to folder with images
        output_name: Optional output folder name
        grid_size: Tuple of (cols, rows) or None for auto
        image_count: Expected number of images (for validation)
        resume: Skip completed steps if artifacts exist
        test_mode: Stop before Kling API call
    """
    
    print("=" * 70)
    print("🎬 BULK IMAGE TO VIDEO PIPELINE")
    if test_mode:
        print("   (TEST MODE: Will stop before Kling API call)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Default to 3x3 grid
    if grid_size is None and image_count is None:
        image_count = 9
        grid_size = (3, 3)
        print(f"📐 Using default 3x3 grid configuration")
    
    # Validate input
    image_files = validate_input_folder(input_folder, image_count)
    if not image_files:
        return False
    
    # Auto-calculate grid if needed
    if grid_size is None:
        grid_size = calculate_grid_dimensions(len(image_files))
    
    print(f"📐 Grid configuration: {grid_size[0]}x{grid_size[1]} for {len(image_files)} images")
    
    # Create output directory
    if output_name:
        output_dir = Path(f"output/{output_name}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = Path(input_folder).name
        output_dir = Path(f"output/bulk_{folder_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate captions (always run for dynamic prompts)
    print(f"\n[Step 1/4] Generating image captions for dynamic prompt...")
    
    caption_cache = output_dir / "captions.json"
    if resume and caption_cache.exists():
        print("  ✓ Loading cached captions")
        with open(caption_cache) as f:
            captions = {Path(k): v for k, v in json.load(f).items()}
        print(f"  ✓ Loaded {len(captions)} captions")
    else:
        caption_start = time.time()
        captions = generate_blip_captions_parallel(image_files, max_workers=4)
        caption_time = time.time() - caption_start
        print(f"  ✓ Generated {len(captions)} captions in {caption_time:.1f}s")
        
        # Save captions
        with open(caption_cache, 'w') as f:
            json.dump({str(k): v for k, v in captions.items()}, f, indent=2)
    
    # Step 2: Create grid
    print(f"\n[Step 2/4] Creating image grid...")
    
    grid_path = output_dir / "grid.png"
    if resume and grid_path.exists():
        print("  ✓ Grid already exists (resume mode)")
        # Calculate cell size from existing grid
        grid_img = Image.open(grid_path)
        cell_width = grid_img.width // grid_size[0]
        cell_height = grid_img.height // grid_size[1]
        cell_size = (cell_width, cell_height)
    else:
        grid_path, cell_size = create_image_grid(image_files, grid_path, grid_size)
    
    # Step 3: Generate animation with dynamic prompt
    print(f"\n[Step 3/4] Generating animation with dynamic prompt...")
    
    # Build dynamic prompt from captions
    dynamic_prompt = build_dynamic_prompt(image_files, grid_size, captions)
    print("  Using dynamic prompt based on image analysis")
    print(f"  Prompt ({len(dynamic_prompt)} chars): {dynamic_prompt[:200]}...")
    
    # Enhanced negative prompt
    negative_prompt = (
        "morphing, warping, distortion, blinking, flickering, "
        "color change, hue shift, saturation change, brightness change, "
        "face distortion, mouth morphing, eye blinking rapidly, "
        "shape shifting, melting, dissolving, transforming, "
        "sudden movements, jerky motion, glitches, artifacts, "
        "background changes, scene changes, camera movement, "
        "zooming, panning, rotating, perspective change, "
        "style change, art style shift, quality degradation, "
        "blur increase, sharpness loss, detail loss, "
        "character interaction, crossing panels, merging panels, "
        "synchronized movement, dependent movement between panels, "
        "moments without any movement, static frames, frozen animation, "
        "complete stillness, no motion at all, stationary, "
        "interdependent animations, characters looking at each other, "
        "shared movements between different panels"
    )
    print(f"  Negative: {len(negative_prompt)} chars (enhanced anti-distortion)")
    
    if test_mode:
        print("\n📋 TEST MODE - Not calling Kling API")
        print("\n" + "=" * 70)
        print("GENERATED PROMPT:")
        print("=" * 70)
        print(dynamic_prompt)
        print("=" * 70)
        
        # Print captions
        print("\n📝 Generated Captions:")
        for i, (path, caption) in enumerate(captions.items(), 1):
            print(f"  {i}. {path.stem}: {caption}")
        
        elapsed = time.time() - start_time
        print(f"\n📋 TEST MODE COMPLETE")
        print(f"⏱️  Test completed in {elapsed:.1f}s")
        return True
    
    # Generate animation
    animation_path = generate_kling_animation(grid_path, dynamic_prompt, output_dir, resume)
    
    if not animation_path:
        print("❌ Animation generation failed")
        return False
    
    # Step 4: Split into individual videos
    print(f"\n[Step 4/4] Creating individual transparent WebM videos...")
    
    # Check if WebM files already exist (resume mode)
    expected_webm_count = min(len(image_files), grid_size[0] * grid_size[1])
    existing_webm = list(output_dir.glob("*.webm"))
    
    if resume and len(existing_webm) == expected_webm_count:
        print(f"  ✓ {len(existing_webm)} WebM files already exist (resume mode)")
        webm_files = existing_webm
    else:
        webm_files = split_grid_video(
            animation_path, 
            grid_size, 
            cell_size,
            [f.name for f in image_files],
            output_dir
        )
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ BULK PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Grid size: {grid_size[0]}x{grid_size[1]}")
    print(f"🎬 Processed: {len(image_files)} images → {len(webm_files)} WebM videos")
    print(f"⏱️  Total time: {elapsed:.1f}s")
    print(f"💰 Cost savings: ~{((len(image_files)-1)/len(image_files)*100):.0f}% vs individual processing")
    
    # List output files
    print(f"\n📦 Output files:")
    print(f"  - {grid_path.name} (grid image)")
    print(f"  - {animation_path.name} (Kling animation)")
    for webm in webm_files[:5]:
        print(f"  - {webm.name}")
    if len(webm_files) > 5:
        print(f"  ... and {len(webm_files)-5} more WebM files")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Bulk image to video pipeline')
    parser.add_argument('input_folder', help='Folder containing images')
    parser.add_argument('--output', help='Output folder name')
    parser.add_argument('--grid', help='Grid size (e.g., 3x3, 4x4)')
    parser.add_argument('--count', type=int, help='Expected image count')
    parser.add_argument('--resume', action='store_true', help='Resume from existing artifacts')
    parser.add_argument('--test', action='store_true', help='Test mode - skip Kling API')
    
    args = parser.parse_args()
    
    # Parse grid size
    grid_size = None
    if args.grid:
        parts = args.grid.split('x')
        if len(parts) == 2:
            grid_size = (int(parts[0]), int(parts[1]))
    
    # Run pipeline
    success = bulk_image_to_video(
        args.input_folder,
        output_name=args.output,
        grid_size=grid_size,
        image_count=args.count,
        resume=args.resume,
        test_mode=args.test
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()