#!/usr/bin/env python3
"""
FFmpeg Filter Builder with proper escaping for complex filtergraphs.
Based on ChatGPT's solution for handling multiple text overlays and effects.
"""

import subprocess
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


class FFmpegFilterBuilder:
    """Build complex FFmpeg filtergraphs with proper escaping."""
    
    @staticmethod
    def escape_filtergraph(s: str) -> str:
        """
        Escape string for FFmpeg filtergraph values.
        Order matters: backslash must be escaped first.
        """
        return (s.replace("\\", "\\\\")
                 .replace(":", "\\:")
                 .replace(",", "\\,")
                 .replace("'", "\\'")
                 .replace(";", "\\;")
                 .replace("[", "\\[")
                 .replace("]", "\\]"))
    
    @staticmethod
    def escape_enable_expr(expr: str) -> str:
        """
        Escape enable expressions specifically.
        Commas in between() must be escaped.
        """
        # Replace commas inside function calls
        import re
        # Find all between(...) expressions and escape their commas
        def escape_between(match):
            content = match.group(0)
            # Replace commas with \,
            return content.replace(",", "\\,")
        
        # Match between(t,num,num) patterns
        pattern = r'between\([^)]+\)'
        result = re.sub(pattern, escape_between, expr)
        
        # Also escape other function commas
        pattern2 = r'(gte|lte|gt|lt|eq)\([^)]+\)'
        result = re.sub(pattern2, escape_between, result)
        
        return result
    
    def create_drawtext_filter(self, text: str, x: int, y: int, 
                              start: float, end: float,
                              fontsize: int = 24,
                              fontcolor: str = "white",
                              box: bool = True,
                              debug_label: Optional[str] = None) -> str:
        """
        Create a drawtext filter with proper escaping.
        
        Args:
            text: Text to display
            x, y: Position
            start, end: Time window in seconds
            fontsize: Font size
            fontcolor: Font color
            box: Whether to add background box
            debug_label: Optional debug label for test mode
        
        Returns:
            Properly escaped drawtext filter string
        """
        # Escape the text content
        text_escaped = self.escape_filtergraph(text)
        
        # Build enable expression with escaped commas
        enable_expr = f"between(t\\,{start}\\,{end})"
        
        # Build filter parts
        parts = [
            f"drawtext=text='{text_escaped}'",
            f"x={x}",
            f"y={y}",
            f"fontsize={fontsize}",
            f"fontcolor={fontcolor}"
        ]
        
        if box:
            parts.extend([
                "box=1",
                "boxcolor=black@0.5",
                "boxborderw=5"
            ])
        
        # Add enable expression
        parts.append(f"enable='{enable_expr}'")
        
        # Join with colons
        return ":".join(parts)
    
    def create_debug_overlay(self, label: str, start: float, end: float,
                            x: int = 10, y: int = 10) -> str:
        """
        Create a debug overlay showing when an effect is active.
        
        Args:
            label: Debug label to show
            start, end: Time window
            x, y: Position for debug text
        
        Returns:
            Debug drawtext filter
        """
        label_escaped = self.escape_filtergraph(label)
        enable_expr = f"between(t\\,{start}\\,{end})"
        
        return (
            f"drawtext=text='{label_escaped}':"
            f"x={x}:y={y}:"
            f"fontsize=14:fontcolor=yellow:"
            f"box=1:boxcolor=red@0.7:boxborderw=3:"
            f"enable='{enable_expr}'"
        )
    
    def create_bloom_effect(self, start: float, end: float, 
                          intensity: float = 1.2) -> str:
        """
        Create a bloom/glow effect using curves.
        
        Args:
            start, end: Time window
            intensity: Bloom intensity
        
        Returns:
            Curves filter for bloom effect
        """
        enable_expr = f"between(t\\,{start}\\,{end})"
        curves_val = f"0/0 0.5/{0.5*intensity} 1/1"
        
        return f"curves=all='{curves_val}':enable='{enable_expr}'"
    
    def create_zoom_effect(self, start: float, end: float,
                         zoom_factor: float = 1.1,
                         width: int = 256, height: int = 144) -> str:
        """
        Create a smooth zoom effect using zoompan.
        
        Args:
            start, end: Time window
            zoom_factor: Maximum zoom level
            width, height: Video dimensions
        
        Returns:
            Zoompan filter string
        """
        # Zoom expression with escaped commas
        zoom_expr = (
            f"z='if(between(t\\,{start}\\,{end})\\,"
            f"min(pzoom+0.001\\,{zoom_factor})\\,1)'"
        )
        
        return (
            f"zoompan={zoom_expr}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:s={width}x{height}"
        )
    
    def create_brightness_effect(self, start: float, end: float,
                                brightness: float = 0.1,
                                saturation: float = 1.1) -> str:
        """
        Create a brightness/saturation adjustment.
        
        Args:
            start, end: Time window
            brightness: Brightness adjustment
            saturation: Saturation adjustment
        
        Returns:
            EQ filter string
        """
        enable_expr = f"between(t\\,{start}\\,{end})"
        
        return (
            f"eq=brightness={brightness}:saturation={saturation}:"
            f"enable='{enable_expr}'"
        )
    
    def create_shake_effect(self, start: float, end: float,
                          intensity: float = 1.0) -> str:
        """
        Create a camera shake effect.
        
        Args:
            start, end: Time window  
            intensity: Shake intensity
        
        Returns:
            Crop + rotate filter for shake
        """
        enable_expr = f"between(t\\,{start}\\,{end})"
        
        return (
            f"crop=in_w:in_h:0:0,"
            f"rotate=PI/180*sin(10*t)*{intensity}:"
            f"enable='{enable_expr}'"
        )
    
    def build_filtergraph(self, filters: List[str]) -> str:
        """
        Build a complete filtergraph from a list of filters.
        
        Args:
            filters: List of filter strings
        
        Returns:
            Complete filtergraph string
        """
        # Join filters with commas for a single chain
        return ",".join(filters)
    
    def build_complex_filtergraph(self, chains: List[Tuple[str, List[str]]]) -> str:
        """
        Build a complex filtergraph with multiple chains and labels.
        
        Args:
            chains: List of (label, filters) tuples
        
        Returns:
            Complex filtergraph with semicolon-separated chains
        """
        chain_strings = []
        
        for label, filters in chains:
            if label:
                # Chain with label
                chain = ",".join(filters)
                chain_strings.append(f"{label}{chain}")
            else:
                # Simple chain
                chain = ",".join(filters)
                chain_strings.append(chain)
        
        # Join chains with semicolons
        return ";".join(chain_strings)


def apply_filters_to_video(video_path: str, output_path: str, 
                          text_overlays: List[Dict],
                          visual_effects: List[Dict],
                          test_mode: bool = False) -> bool:
    """
    Apply text overlays and visual effects to a video.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        text_overlays: List of text overlay dictionaries
        visual_effects: List of visual effect dictionaries
        test_mode: Whether to add debug overlays
    
    Returns:
        True if successful
    """
    builder = FFmpegFilterBuilder()
    filters = []
    
    # Add text overlays
    for overlay in text_overlays:
        text_filter = builder.create_drawtext_filter(
            text=overlay.get("text", ""),
            x=overlay.get("x", 10),
            y=overlay.get("y", 100),
            start=overlay.get("start", 0),
            end=overlay.get("end", 1)
        )
        filters.append(text_filter)
        
        # Add debug overlay in test mode
        if test_mode:
            debug_filter = builder.create_debug_overlay(
                label=f"TEXT: {overlay.get('text', '')[:15]}",
                start=overlay.get("start", 0),
                end=overlay.get("end", 1),
                y=10
            )
            filters.append(debug_filter)
    
    # Add visual effects
    for effect in visual_effects:
        effect_type = effect.get("type", "")
        start = effect.get("start", 0)
        end = effect.get("end", 1)
        
        if effect_type == "bloom":
            effect_filter = builder.create_bloom_effect(start, end)
            filters.append(effect_filter)
            if test_mode:
                filters.append(builder.create_debug_overlay(
                    "EFFECT: BLOOM", start, end, y=30
                ))
        elif effect_type == "zoom":
            effect_filter = builder.create_zoom_effect(start, end)
            filters.append(effect_filter)
            if test_mode:
                filters.append(builder.create_debug_overlay(
                    "EFFECT: ZOOM", start, end, y=30
                ))
        elif effect_type == "brightness":
            effect_filter = builder.create_brightness_effect(start, end)
            filters.append(effect_filter)
            if test_mode:
                filters.append(builder.create_debug_overlay(
                    "EFFECT: BRIGHTNESS", start, end, y=30
                ))
    
    # Build complete filtergraph
    filtergraph = builder.build_filtergraph(filters)
    
    # FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", filtergraph,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]
    
    try:
        print(f"Applying {len(text_overlays)} text overlays and {len(visual_effects)} effects...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        
        return Path(output_path).exists()
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Test the filter builder."""
    builder = FFmpegFilterBuilder()
    
    # Test escaping
    text = "Let's start"
    print(f"Original: {text}")
    print(f"Escaped: {builder.escape_filtergraph(text)}")
    
    # Test drawtext filter
    filter_str = builder.create_drawtext_filter(
        text="Let's start",
        x=10, y=90,
        start=2.779, end=3.579
    )
    print(f"\nDrawtext filter:\n{filter_str}")
    
    # Test enable expression escaping
    enable = "between(t,2.779,3.579)"
    print(f"\nOriginal enable: {enable}")
    print(f"Escaped enable: {builder.escape_enable_expr(enable)}")


if __name__ == "__main__":
    main()