#!/usr/bin/env python3
"""
Fix for 3D text depth smoothness - make all letters as smooth as 'O'
and reduce depth by 80%
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

# Override the rendering method with better depth smoothing
class Text3DSmoothDepth(Text3DBehindSegment):
    """Improved 3D text with smoother depth for all letters"""
    
    def _render_3d_text_with_anchor(
        self,
        text: str,
        scale: float,
        alpha: float,
        apply_perspective: bool,
    ):
        """Enhanced depth rendering with extra smoothing"""
        ss = self.supersample_factor * 2  # DOUBLE supersampling for depth
        
        # Reduce depth by 80% (make it 20% of original)
        actual_depth_offset = self.depth_offset * 0.2
        
        # More layers with smaller steps = smoother gradient
        actual_depth_layers = self.depth_layers * 2
        
        # Font at higher resolution
        font_px = max(2, int(round(self.font_size * scale * ss)))
        font = self._get_font(font_px)
        
        # Measure text
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        face_w = max(1, bbox[2] - bbox[0])
        face_h = max(1, bbox[3] - bbox[1])
        
        # Canvas with padding
        depth_off = int(round(actual_depth_offset * scale * ss))
        pad = max(depth_off * actual_depth_layers * 2, ss * 8)
        canvas_w = face_w + pad * 2 + depth_off * actual_depth_layers
        canvas_h = face_h + pad * 2 + depth_off * actual_depth_layers
        
        # Create layers for smoother blending
        img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        
        # Front face position
        front_x = (canvas_w - face_w) // 2
        front_y = (canvas_h - face_h) // 2
        
        # Draw depth layers with anti-aliasing per layer
        for i in range(actual_depth_layers, 0, -1):
            # Create a separate image for this depth layer
            layer_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
            layer_draw = ImageDraw.Draw(layer_img)
            
            ox = front_x + i * depth_off
            oy = front_y - i * depth_off
            
            # Smooth gradient
            t = (actual_depth_layers - i) / max(actual_depth_layers - 1, 1)
            t = t * t * t * (3.0 - 2.0 * t - t)  # Smoother curve
            
            r = int(self.depth_color[0] * (1 - t) + self.text_color[0] * t * 0.75)
            g = int(self.depth_color[1] * (1 - t) + self.text_color[1] * t * 0.75)
            b = int(self.depth_color[2] * (1 - t) + self.text_color[2] * t * 0.75)
            
            # Draw with slight transparency for blending
            layer_alpha = int(255 * alpha * (0.9 + 0.1 * t))
            layer_draw.text((ox, oy), text, font=font, fill=(r, g, b, layer_alpha))
            
            # Apply slight blur to this layer for smoothness
            if i > 1:  # Don't blur the frontmost layers
                layer_img = layer_img.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Composite this layer
            img = Image.alpha_composite(img, layer_img)
        
        # High-quality outline
        outline_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        outline_draw = ImageDraw.Draw(outline_img)
        outline_w = max(1, int(self.outline_width * ss))
        
        # Multi-sample outline for smoothness
        for radius in np.linspace(outline_w, 0.5, 5):
            fade = 1.0 - (radius / outline_w) * 0.6
            for ang in range(0, 360, 20):  # More samples
                ox = radius * np.cos(np.radians(ang))
                oy = radius * np.sin(np.radians(ang))
                outline_draw.text(
                    (int(front_x + ox), int(front_y + oy)),
                    text,
                    font=font,
                    fill=(0, 0, 0, int(100 * alpha * fade)),
                )
        
        outline_img = outline_img.filter(ImageFilter.GaussianBlur(radius=max(1, ss // 3)))
        img = Image.alpha_composite(img, outline_img)
        
        # Front face
        face_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(face_img).text(
            (front_x, front_y),
            text,
            font=font,
            fill=(*self.text_color, int(255 * alpha)),
        )
        img = Image.alpha_composite(img, face_img)
        
        # Anchor before perspective
        anchor_x_ss = front_x + face_w / 2.0
        anchor_y_ss = front_y + face_h / 2.0
        
        # Perspective if needed
        if apply_perspective and self.perspective_angle > 0:
            arr = np.array(img)
            H, W = arr.shape[:2]
            
            angle = np.radians(self.perspective_angle)
            offset = int(H * np.tan(angle) * 0.2)
            
            src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            dst = np.float32([[offset, 0], [W - offset, 0], [W, H], [0, H]])
            
            M = cv2.getPerspectiveTransform(src, dst)
            arr = cv2.warpPerspective(
                arr, M, (W, H),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            
            # Transform anchor
            pt = np.array([anchor_x_ss, anchor_y_ss, 1.0], dtype=np.float64)
            M33 = M @ pt
            anchor_x_ss = float(M33[0] / max(M33[2], 1e-6))
            anchor_y_ss = float(M33[1] / max(M33[2], 1e-6))
            
            img = Image.fromarray(arr)
        
        # Shadow
        shadow = np.array(img)
        shadow[:, :, :3] = 0
        shadow[:, :, 3] = (shadow[:, :, 3] * 0.3).astype(np.uint8)
        shadow = Image.fromarray(shadow).filter(ImageFilter.GaussianBlur(radius=max(2, ss)))
        
        final = Image.new("RGBA", img.size, (0, 0, 0, 0))
        shadow_off = int(round(self.shadow_offset * scale * ss * 0.5))  # Smaller shadow too
        final.paste(shadow, (shadow_off, shadow_off), shadow)
        final = Image.alpha_composite(final, img)
        
        # Extra smoothing before downsample
        final = final.filter(ImageFilter.GaussianBlur(radius=ss / 5.0))
        
        # High-quality downsample
        target_w = max(1, final.width // ss)
        target_h = max(1, final.height // ss)
        final = final.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # Return anchor
        anchor_x = anchor_x_ss / ss
        anchor_y = anchor_y_ss / ss
        
        return final, (anchor_x, anchor_y)

# Test it
def main():
    video_path = "test_element_3sec.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return
    
    print("Loading video...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("Extracting mask...")
    mask = extract_foreground_mask(frame_rgb)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    print(f"Creating animation with smooth depth (80% smaller)...")
    anim = Text3DSmoothDepth(
        duration=3.0,
        fps=fps,
        resolution=(W, H),
        text="HELLO WORLD",
        segment_mask=mask,
        font_size=140,
        text_color=(255, 220, 0),
        depth_color=(200, 170, 0),
        depth_layers=10,  # Will be doubled internally
        depth_offset=3,    # Will be reduced by 80% internally
        start_scale=2.2,
        end_scale=0.9,
        phase1_duration=1.2,
        phase2_duration=0.6,
        phase3_duration=1.2,
        center_position=(W//2, H//2),
        shadow_offset=6,
        outline_width=2,
        perspective_angle=25,
        supersample_factor=2,  # Will be doubled for depth
        debug=False,
        perspective_during_shrink=False,
    )
    
    print("Generating frames...")
    output_frames = []
    total = int(3.0 * fps)
    
    for i in range(total):
        if i % 30 == 0:
            print(f"  Frame {i}/{total}...")
        bg = frames[i % len(frames)]
        frame = anim.generate_frame(i, bg)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Save video
    print("Saving...")
    output_path = "text_3d_smooth_depth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp = "temp_smooth.mp4"
    
    out = cv2.VideoWriter(temp, fourcc, fps, (W, H))
    for f in output_frames:
        out.write(f)
    out.release()
    
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', temp,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        output_path
    ]
    subprocess.run(cmd, capture_output=True)
    
    if os.path.exists(temp):
        os.remove(temp)
    
    # Save preview
    preview = output_frames[total // 4]
    cv2.imwrite("smooth_depth_preview.png", preview)
    
    print(f"\n✅ Done!")
    print(f"Video: {output_path}")
    print("Preview: smooth_depth_preview.png")
    print("\nImprovements:")
    print("  • All letters have smooth depth like 'O'")
    print("  • Depth reduced by 80% (more subtle)")
    print("  • Extra anti-aliasing on depth layers")

if __name__ == "__main__":
    main()