#!/usr/bin/env python3
"""
Final optimized 3D text animation test with quality fixes.
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask


# Override the methods directly in the test
class Text3DImproved(Text3DBehindSegment):
    """Improved 3D text with quality fixes."""
    
    def render_3d_text(
        self, 
        text: str, 
        font: ImageFont.FreeTypeFont,
        scale: float = 1.0,
        alpha: float = 1.0,
        apply_perspective: bool = True
    ) -> Image.Image:
        """Render 3D text with anti-aliasing."""
        
        # Supersampling for anti-aliasing
        ss = 2
        
        # Scaled font
        font_size = int(self.font_size * scale * ss)
        if self.font_path and os.path.exists(self.font_path):
            font = ImageFont.truetype(self.font_path, font_size)
        else:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        # Get dimensions
        temp = Image.new('RGBA', (100, 100))
        draw = ImageDraw.Draw(temp)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # Canvas size
        offset = int(self.depth_offset * scale * ss)
        canvas_w = w + self.depth_layers * offset * 3
        canvas_h = h + self.depth_layers * offset * 3
        
        # Create image
        img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Center position
        cx = (canvas_w - w) // 2
        cy = (canvas_h - h) // 2
        
        # Draw depth layers
        for i in range(self.depth_layers, 0, -1):
            x = cx + i * offset
            y = cy - i * offset
            
            # Color interpolation
            t = (self.depth_layers - i) / max(self.depth_layers - 1, 1)
            t = t * t * (3 - 2 * t)  # Smoothstep
            
            r = int(self.depth_color[0] * (1-t) + self.text_color[0] * t * 0.8)
            g = int(self.depth_color[1] * (1-t) + self.text_color[1] * t * 0.8)
            b = int(self.depth_color[2] * (1-t) + self.text_color[2] * t * 0.8)
            
            draw.text((x, y), text, font=font, fill=(r, g, b, int(255 * alpha)))
        
        # Outline
        for ox in [-ss*2, 0, ss*2]:
            for oy in [-ss*2, 0, ss*2]:
                if ox or oy:
                    draw.text((cx+ox, cy+oy), text, font=font, fill=(0, 0, 0, int(120 * alpha)))
        
        # Main text
        draw.text((cx, cy), text, font=font, fill=(*self.text_color, int(255 * alpha)))
        
        # Perspective
        if apply_perspective and self.perspective_angle > 0:
            arr = np.array(img)
            h, w = arr.shape[:2]
            
            angle = np.radians(self.perspective_angle)
            offset = int(h * np.tan(angle) * 0.2)
            
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([[offset, 0], [w-offset, 0], [w, h], [0, h]])
            
            M = cv2.getPerspectiveTransform(src, dst)
            arr = cv2.warpPerspective(arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
            img = Image.fromarray(arr)
        
        # Shadow
        shadow = img.copy()
        shadow_arr = np.array(shadow)
        shadow_arr[:, :, :3] = 0
        shadow_arr[:, :, 3] = (shadow_arr[:, :, 3] * 0.3).astype(np.uint8)
        shadow = Image.fromarray(shadow_arr)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=ss))
        
        # Composite
        final = Image.new('RGBA', img.size, (0, 0, 0, 0))
        shadow_off = int(self.shadow_offset * scale * ss)
        final.paste(shadow, (shadow_off, shadow_off), shadow)
        final = Image.alpha_composite(final, img)
        
        # Downsample with anti-aliasing
        final = final.filter(ImageFilter.GaussianBlur(radius=0.5))
        final = final.resize((final.width // ss, final.height // ss), Image.Resampling.LANCZOS)
        
        return final
    
    def generate_frame(self, frame_number: int, background: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate frame with center shrinking."""
        # Create frame
        if background is not None:
            frame = background.copy()
            if frame.shape[2] == 3:
                alpha = np.ones((frame.shape[0], frame.shape[1], 1), dtype=np.uint8) * 255
                frame = np.concatenate([frame, alpha], axis=2)
        else:
            frame = np.zeros((*self.resolution[::-1], 4), dtype=np.uint8)
            frame[:, :, 3] = 255
        
        # Animation phase
        if frame_number < self.phase1_frames:
            # Shrinking
            t = frame_number / max(self.phase1_frames - 1, 1)
            t = t * t * (3 - 2 * t)  # Smoothstep
            scale = self.start_scale + (self.end_scale - self.start_scale) * t
            alpha = 1.0
            is_behind = False
            
        elif frame_number < self.phase1_frames + self.phase2_frames:
            # Transition
            t = (frame_number - self.phase1_frames) / max(self.phase2_frames - 1, 1)
            scale = self.end_scale
            alpha = 1.0 - 0.5 * (1 - np.exp(-3 * t)) / (1 - np.exp(-3))
            is_behind = t > 0.5
            
        else:
            # Behind
            scale = self.end_scale
            alpha = 0.5
            is_behind = True
        
        # Render text
        text_img = self.render_3d_text(
            self.text, self.font, 
            scale=scale, alpha=alpha,
            apply_perspective=(frame_number >= self.phase1_frames)
        )
        
        text_arr = np.array(text_img)
        th, tw = text_arr.shape[:2]
        
        # Center position
        cx, cy = self.center_position
        x = cx - tw // 2
        y = cy - th // 2
        
        # Bounds check
        x = max(0, min(x, self.resolution[0] - tw))
        y = max(0, min(y, self.resolution[1] - th))
        
        # Create overlay
        overlay = Image.fromarray(frame)
        
        # Apply mask if behind
        if is_behind and self.segment_mask is not None:
            # Mask the text
            text_masked = text_img.copy()
            text_masked_arr = np.array(text_masked)
            
            # Get mask region
            y1, y2 = y, min(y + th, self.resolution[1])
            x1, x2 = x, min(x + tw, self.resolution[0])
            mask_region = self.segment_mask[y1:y2, x1:x2]
            
            # Apply mask
            h_region = y2 - y1
            w_region = x2 - x1
            text_alpha = text_masked_arr[:h_region, :w_region, 3].astype(float)
            text_alpha *= (1 - mask_region / 255.0)
            text_masked_arr[:h_region, :w_region, 3] = text_alpha.astype(np.uint8)
            
            text_img = Image.fromarray(text_masked_arr)
        
        # Composite
        overlay.paste(text_img, (x, y), text_img)
        
        result = np.array(overlay)
        return result[:, :, :3] if result.shape[2] == 4 else result


def main():
    # Load video
    video_path = "test_element_3sec.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return
    
    print(f"Loading: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    print(f"Loaded: {width}x{height} @ {fps}fps")
    
    # Create mask
    print("Creating mask...")
    mask = extract_foreground_mask(frames[0])
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height))
    
    # Create animation
    print("\nGenerating 3D text animation...")
    anim = Text3DImproved(
        duration=3.0,
        fps=fps,
        resolution=(width, height),
        text="HELLO WORLD",
        segment_mask=mask,
        font_size=140,
        text_color=(255, 220, 0),
        depth_color=(200, 170, 0),
        depth_layers=10,
        depth_offset=3,
        start_scale=2.2,
        end_scale=0.9,
        phase1_duration=1.2,
        phase2_duration=0.6,
        phase3_duration=1.2,
        shadow_offset=6,
        outline_width=2,
        perspective_angle=25
    )
    
    # Generate frames
    output_frames = []
    total = int(3.0 * fps)
    
    for i in range(total):
        if i % 10 == 0:
            print(f"  Frame {i}/{total}...")
        
        bg = frames[i % len(frames)]
        frame = anim.generate_frame(i, bg)
        
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        output_frames.append(frame)
    
    print("  ✓ Done!")
    
    # Save video
    print("\nSaving video...")
    output_path = "text_3d_final.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp = "temp_final.mp4"
    out = cv2.VideoWriter(temp, fourcc, fps, (width, height))
    
    for frame in output_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    
    # Convert to H.264
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', temp,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        output_path
    ]
    subprocess.run(cmd, capture_output=True)
    
    if os.path.exists(temp):
        os.remove(temp)
    
    print(f"  ✓ Saved: {output_path}")
    
    # Save preview
    preview = output_frames[total // 2]
    cv2.imwrite("text_3d_final_preview.png", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    
    print("\n✅ Complete!")
    print(f"Output: {output_path}")
    print("Improvements:")
    print("  • Smooth anti-aliased text")
    print("  • Center-point shrinking") 
    print("  • High-quality 3D depth")


if __name__ == "__main__":
    main()