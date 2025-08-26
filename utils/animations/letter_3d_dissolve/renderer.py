"""
3D letter rendering utilities.
"""

import os
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter


class Letter3DRenderer:
    """Handles 3D letter rendering with depth layers."""
    
    def __init__(
        self,
        font_size: int = 120,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        depth_color: Tuple[int, int, int] = (200, 170, 0),
        depth_layers: int = 8,
        depth_offset: int = 3,
        supersample_factor: int = 2,
        font_path: Optional[str] = None,
        debug: bool = False
    ):
        self.font_size = font_size
        self.text_color = text_color
        self.depth_color = depth_color
        self.depth_layers = depth_layers
        self.depth_offset = depth_offset
        self.supersample_factor = supersample_factor
        self.font_path = font_path
        self.debug = debug
        
        if self.debug:
            print(f"[TEXT_QUALITY] Supersample factor: {self.supersample_factor}")
    
    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font at specified size - prioritize vector fonts."""
        candidates = []
        if self.font_path:
            candidates.append(self.font_path)
        
        # Environment overrides
        for key in ("T3D_FONT", "TEXT_FONT", "FONT_PATH"):
            p = os.environ.get(key)
            if p:
                candidates.append(p)
        
        # Common cross-OS paths
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        
        for p in candidates:
            try:
                if p and os.path.isfile(p):
                    if self.debug:
                        print(f"[TEXT_QUALITY] Using TTF font: {p}")
                    return ImageFont.truetype(p, size)
            except Exception:
                continue
        
        if self.debug:
            print("[TEXT_QUALITY] WARNING: Falling back to PIL bitmap font (edges may look jagged). "
                  "Install a TTF/OTF and pass font_path parameter.")
        return ImageFont.load_default()
    
    def render_3d_letter(
        self, letter: str, scale: float, alpha: float, depth_scale: float
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """Render a single 3D letter with depth layers."""
        font_px = int(self.font_size * scale * self.supersample_factor)
        font = self.get_font(font_px)

        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), letter, font=font)
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        margin = int(self.depth_offset * self.depth_layers * self.supersample_factor)
        width = bbox_w + 2 * margin
        height = bbox_h + 2 * margin

        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for i in range(self.depth_layers - 1, -1, -1):
            depth_alpha = int(alpha * 255 * (0.3 + 0.7 * (1 - i / self.depth_layers)))
            offset = int(i * self.depth_offset * depth_scale * self.supersample_factor)

            if i == 0:
                color = (*self.text_color, depth_alpha)
            else:
                factor = 0.7 - (i / self.depth_layers) * 0.4
                color = tuple(int(c * factor) for c in self.depth_color) + (depth_alpha,)

            x = -bbox[0] + margin + offset
            y = -bbox[1] + margin + offset
            
            # Add stroke for front layer to improve antialiasing
            if i == 0 and self.supersample_factor >= 4:
                stroke_width = max(1, self.supersample_factor // 8)
                draw.text((x, y), letter, font=font, fill=color, 
                         stroke_width=stroke_width, stroke_fill=color)
            else:
                draw.text((x, y), letter, font=font, fill=color)

        # Apply Gaussian blur for antialiasing before downsampling
        if self.supersample_factor >= 4:
            blur_radius = self.supersample_factor / 5.0
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Progressive downsampling for better quality
        if self.supersample_factor >= 8:
            # Two-step downsampling for very high supersample factors
            intermediate_size = (width // (self.supersample_factor // 2), 
                               height // (self.supersample_factor // 2))
            canvas = canvas.resize(intermediate_size, Image.Resampling.LANCZOS)
            
            final_size = (intermediate_size[0] // 2, intermediate_size[1] // 2)
            canvas = canvas.resize(final_size, Image.Resampling.LANCZOS)
            
            ax = int(round((-bbox[0] + margin) / self.supersample_factor))
            ay = int(round((-bbox[1] + margin) / self.supersample_factor))
        elif self.supersample_factor > 1:
            new_size = (width // self.supersample_factor, height // self.supersample_factor)
            canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
            ax = int(round((-bbox[0] + margin) / self.supersample_factor))
            ay = int(round((-bbox[1] + margin) / self.supersample_factor))
        else:
            ax = -bbox[0] + margin
            ay = -bbox[1] + margin

        return canvas, (ax, ay)