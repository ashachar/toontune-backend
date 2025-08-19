#!/usr/bin/env python3
"""
Image Downsampling Utility
Reduces image resolution to save space and processing time
Supports extreme downsampling to as low as 8x8 pixels
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import numpy as np


class ImageDownsampler:
    """Downsample images to lower resolutions"""
    
    # Preset resolutions (width x height)
    PRESETS = {
        'icon': (8, 8),         # Absolute minimum - just color blocks
        'micro': (16, 16),      # Minecraft-like pixels
        'tiny': (32, 32),       # Very pixelated
        'mini': (64, 64),       # Default - recognizable shapes
        'small': (128, 128),    # Good for thumbnails
        'medium': (256, 256),   # Reasonable quality
        'large': (512, 512),    # Good balance
        'hd': (1280, 720),      # HD resolution
        'fhd': (1920, 1080),    # Full HD
        '4k': (3840, 2160),     # 4K resolution
    }
    
    # Resampling algorithms
    RESAMPLE_MODES = {
        'nearest': Image.NEAREST,      # Fastest, blocky (good for pixel art)
        'box': Image.BOX,              # Fast, okay quality
        'bilinear': Image.BILINEAR,    # Good balance
        'hamming': Image.HAMMING,      # Sharp edges
        'bicubic': Image.BICUBIC,      # Smooth gradients
        'lanczos': Image.LANCZOS,      # Best quality (default)
    }
    
    def __init__(self,
                 input_path: str,
                 output_path: Optional[str] = None,
                 resolution: Optional[Tuple[int, int]] = None,
                 preset: str = 'small',
                 maintain_aspect: bool = True,
                 resample_mode: str = 'lanczos',
                 quality: int = 85,
                 format: Optional[str] = None,
                 dither: bool = False,
                 palette_colors: Optional[int] = None):
        """
        Initialize image downsampler
        
        Args:
            input_path: Path to input image
            output_path: Path for output (auto-generated if None)
            resolution: Target resolution as (width, height) tuple
            preset: Preset size ('icon', 'micro', 'tiny', 'mini', 'small', etc.) - default 'small'
            maintain_aspect: Maintain aspect ratio
            resample_mode: Resampling algorithm
            quality: JPEG quality (1-100)
            format: Output format (png, jpg, webp, etc.)
            dither: Apply dithering for better color reduction
            palette_colors: Reduce to N colors (e.g., 16, 256)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.resolution = resolution or self.PRESETS.get(preset, self.PRESETS['mini'])
        self.maintain_aspect = maintain_aspect
        self.resample_mode = self.RESAMPLE_MODES.get(resample_mode, Image.LANCZOS)
        self.quality = quality
        self.format = format
        self.dither = dither
        self.palette_colors = palette_colors
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        # Auto-generate output path if not provided
        if not self.output_path:
            input_name = Path(input_path).stem
            width, height = self.resolution
            ext = f".{format}" if format else Path(input_path).suffix
            self.output_path = f"{input_name}_{width}x{height}_downsampled{ext}"
    
    def calculate_target_size(self, img: Image.Image) -> Tuple[int, int]:
        """Calculate target size maintaining aspect ratio if needed"""
        target_width, target_height = self.resolution
        
        if not self.maintain_aspect:
            return (target_width, target_height)
        
        # Calculate scaling to fit within target dimensions
        width_ratio = target_width / img.width
        height_ratio = target_height / img.height
        ratio = min(width_ratio, height_ratio)
        
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        
        return (new_width, new_height)
    
    def apply_pixel_art_effect(self, img: Image.Image, pixel_size: int = None) -> Image.Image:
        """Apply pixel art effect by downsampling then upsampling"""
        if pixel_size is None:
            # Auto-calculate based on target resolution
            pixel_size = max(1, min(img.width, img.height) // min(self.resolution))
        
        # Calculate temporary small size
        temp_width = img.width // pixel_size
        temp_height = img.height // pixel_size
        
        # Downsample with nearest neighbor
        small = img.resize((temp_width, temp_height), Image.NEAREST)
        
        # Upsample back to original size
        pixelated = small.resize((img.width, img.height), Image.NEAREST)
        
        return pixelated
    
    def reduce_colors(self, img: Image.Image, num_colors: int) -> Image.Image:
        """Reduce color palette to specified number of colors"""
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Quantize image
        if self.dither:
            img = img.quantize(colors=num_colors, dither=1)  # 1 = dither enabled
        else:
            img = img.quantize(colors=num_colors, dither=0)  # 0 = no dither
        
        # Convert back to RGB
        img = img.convert('RGB')
        
        return img
    
    def apply_retro_filter(self, img: Image.Image) -> Image.Image:
        """Apply retro/vintage game console filter"""
        # Common retro color palettes
        retro_palettes = {
            'gameboy': [(155, 188, 15), (139, 172, 15), (48, 98, 48), (15, 56, 15)],
            'nes': [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), 
                    (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
        }
        
        # Convert to numpy array
        arr = np.array(img)
        height, width = arr.shape[:2]
        
        # Simple posterization effect
        levels = 4
        arr = (arr // (256 // levels)) * (256 // levels)
        
        return Image.fromarray(arr.astype('uint8'))
    
    def downsample(self, verbose: bool = True, 
                  pixel_art: bool = False,
                  retro: bool = False) -> bool:
        """
        Perform the downsampling
        
        Args:
            verbose: Print progress information
            pixel_art: Apply pixel art effect
            retro: Apply retro filter
        
        Returns:
            Success status
        """
        try:
            # Open image
            img = Image.open(self.input_path)
            original_size = img.size
            original_mode = img.mode
            
            if verbose:
                print(f"🖼️  Input Image Info:")
                print(f"   Path: {self.input_path}")
                print(f"   Resolution: {img.width}x{img.height}")
                print(f"   Mode: {img.mode}")
                print(f"   Format: {img.format}")
                file_size = os.path.getsize(self.input_path) / 1024  # KB
                print(f"   File Size: {file_size:.2f} KB")
                print()
                
                print(f"🎯 Target Settings:")
                print(f"   Resolution: {self.resolution[0]}x{self.resolution[1]}")
                print(f"   Resample: {[k for k, v in self.RESAMPLE_MODES.items() if v == self.resample_mode][0]}")
                print(f"   Maintain Aspect: {self.maintain_aspect}")
                if self.palette_colors:
                    print(f"   Colors: {self.palette_colors}")
                print()
            
            # Convert RGBA to RGB if saving as JPEG
            if self.format in ['jpg', 'jpeg'] or (not self.format and self.output_path.lower().endswith(('.jpg', '.jpeg'))):
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
            
            # Apply effects before resizing for better quality
            if retro:
                if verbose:
                    print("🎮 Applying retro filter...")
                img = self.apply_retro_filter(img)
            
            if pixel_art:
                if verbose:
                    print("🎨 Applying pixel art effect...")
                img = self.apply_pixel_art_effect(img)
            
            # Reduce colors if specified
            if self.palette_colors:
                if verbose:
                    print(f"🎨 Reducing to {self.palette_colors} colors...")
                img = self.reduce_colors(img, self.palette_colors)
            
            # Calculate target size
            target_size = self.calculate_target_size(img)
            
            if verbose:
                print(f"🔄 Downsampling image...")
                if target_size != self.resolution:
                    print(f"   Adjusted size (maintaining aspect): {target_size[0]}x{target_size[1]}")
            
            # Resize image
            img_resized = img.resize(target_size, self.resample_mode)
            
            # Save image
            save_kwargs = {}
            
            # Set quality for lossy formats
            if self.format in ['jpg', 'jpeg', 'webp'] or \
               (not self.format and self.output_path.lower().endswith(('.jpg', '.jpeg', '.webp'))):
                save_kwargs['quality'] = self.quality
                save_kwargs['optimize'] = True
            
            # Set compression for PNG
            if self.format == 'png' or (not self.format and self.output_path.lower().endswith('.png')):
                save_kwargs['compress_level'] = 9  # Maximum compression
                save_kwargs['optimize'] = True
            
            # Save with specified format
            if self.format:
                save_kwargs['format'] = self.format.upper()
            
            img_resized.save(self.output_path, **save_kwargs)
            
            if verbose:
                # Report results
                output_size = os.path.getsize(self.output_path) / 1024  # KB
                input_size = os.path.getsize(self.input_path) / 1024  # KB
                reduction = (1 - output_size/input_size) * 100 if input_size > 0 else 0
                
                print(f"\n✅ Success!")
                print(f"   Output: {self.output_path}")
                print(f"   Final Resolution: {img_resized.width}x{img_resized.height}")
                print(f"   File Size: {output_size:.2f} KB (reduced by {reduction:.1f}%)")
                
                # Calculate pixel reduction
                original_pixels = original_size[0] * original_size[1]
                new_pixels = img_resized.width * img_resized.height
                pixel_reduction = (1 - new_pixels/original_pixels) * 100
                print(f"   Pixels: {new_pixels:,} (reduced by {pixel_reduction:.1f}%)")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"\n❌ Error downsampling image:")
                print(f"   {str(e)}")
            return False
    
    def batch_downsample(self, input_paths: List[str], verbose: bool = True) -> List[str]:
        """
        Downsample multiple images
        
        Args:
            input_paths: List of input image paths
            verbose: Print progress
        
        Returns:
            List of output paths for successful conversions
        """
        successful = []
        
        for i, input_path in enumerate(input_paths, 1):
            if verbose:
                print(f"\n[{i}/{len(input_paths)}] Processing: {Path(input_path).name}")
                print("-" * 50)
            
            # Create new downsampler for each image
            input_name = Path(input_path).stem
            width, height = self.resolution
            ext = f".{self.format}" if self.format else Path(input_path).suffix
            output_path = f"{input_name}_{width}x{height}_downsampled{ext}"
            
            downsampler = ImageDownsampler(
                input_path=input_path,
                output_path=output_path,
                resolution=self.resolution,
                maintain_aspect=self.maintain_aspect,
                resample_mode=[k for k, v in self.RESAMPLE_MODES.items() if v == self.resample_mode][0],
                quality=self.quality,
                format=self.format,
                dither=self.dither,
                palette_colors=self.palette_colors
            )
            
            if downsampler.downsample(verbose=verbose):
                successful.append(output_path)
        
        if verbose:
            print(f"\n📊 Batch Complete: {len(successful)}/{len(input_paths)} successful")
        
        return successful


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Downsample images to lower resolutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preset sizes:
  icon   : 8x8     - Absolute minimum
  micro  : 16x16   - Minecraft-like
  tiny   : 32x32   - Very pixelated  
  mini   : 64x64   - Default, recognizable
  small  : 128x128 - Thumbnails
  medium : 256x256 - Reasonable quality
  large  : 512x512 - Good balance

Resample modes:
  nearest  : Blocky (good for pixel art)
  bilinear : Good balance
  lanczos  : Best quality (default)

Examples:
  %(prog)s image.png                      # Use default 'small' (128x128)
  %(prog)s image.png --preset mini        # Use 64x64 pixels
  %(prog)s image.png --preset icon        # Use 8x8 pixels (extreme)
  %(prog)s image.png --size 100 100       # Custom 100x100
  %(prog)s image.png --pixel-art          # Apply pixel art effect
  %(prog)s image.png --colors 16          # Reduce to 16 colors
  %(prog)s image.png --retro --colors 8   # Retro game style
  %(prog)s *.jpg --preset small           # Batch process
        """
    )
    
    parser.add_argument('input', nargs='+', help='Input image file(s)')
    parser.add_argument('-o', '--output', help='Output file path (single file only)')
    parser.add_argument('-p', '--preset',
                       choices=ImageDownsampler.PRESETS.keys(),
                       default='small',
                       help='Size preset (default: small)')
    parser.add_argument('-s', '--size', nargs=2, type=int, metavar=('W', 'H'),
                       help='Custom resolution (width height)')
    parser.add_argument('-r', '--resample',
                       choices=ImageDownsampler.RESAMPLE_MODES.keys(),
                       default='lanczos',
                       help='Resampling algorithm (default: lanczos)')
    parser.add_argument('-q', '--quality', type=int, default=85,
                       help='JPEG quality 1-100 (default: 85)')
    parser.add_argument('-f', '--format',
                       choices=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif'],
                       help='Output format')
    parser.add_argument('--no-aspect', action='store_true',
                       help="Don't maintain aspect ratio")
    parser.add_argument('--pixel-art', action='store_true',
                       help='Apply pixel art effect')
    parser.add_argument('--retro', action='store_true',
                       help='Apply retro filter')
    parser.add_argument('--dither', action='store_true',
                       help='Use dithering for color reduction')
    parser.add_argument('--colors', type=int, metavar='N',
                       help='Reduce to N colors')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode')
    
    args = parser.parse_args()
    
    # Determine resolution
    if args.size:
        resolution = tuple(args.size)
    else:
        resolution = None  # Use preset
    
    # Handle batch processing
    if len(args.input) > 1:
        if args.output:
            print("Warning: --output ignored for batch processing")
        
        # Create downsampler with common settings
        downsampler = ImageDownsampler(
            input_path=args.input[0],  # Temporary
            resolution=resolution,
            preset=args.preset,
            maintain_aspect=not args.no_aspect,
            resample_mode=args.resample,
            quality=args.quality,
            format=args.format,
            dither=args.dither,
            palette_colors=args.colors
        )
        
        # Process batch
        successful = downsampler.batch_downsample(
            args.input,
            verbose=not args.quiet
        )
        
        sys.exit(0 if successful else 1)
    
    # Single file processing
    downsampler = ImageDownsampler(
        input_path=args.input[0],
        output_path=args.output,
        resolution=resolution,
        preset=args.preset,
        maintain_aspect=not args.no_aspect,
        resample_mode=args.resample,
        quality=args.quality,
        format=args.format,
        dither=args.dither,
        palette_colors=args.colors
    )
    
    # Perform downsampling
    success = downsampler.downsample(
        verbose=not args.quiet,
        pixel_art=args.pixel_art,
        retro=args.retro
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()