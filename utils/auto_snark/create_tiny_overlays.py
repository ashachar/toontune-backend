#!/usr/bin/env python3
"""
Create properly sized overlays for the tiny 256x116 video
"""

import subprocess
import os

def create_proper_overlays():
    """Create overlays that fit the tiny video dimensions"""
    
    input_video = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/downsampled/scene_001.mp4"
    output_video = "do_re_mi_proper_overlays.mp4"
    
    # Much shorter text for tiny video
    snarks = [
        (3.8, "Another musical..."),
        (16.8, "ABC so complex!"), 
        (25.5, "We get it..."),
        (41.8, "Teaching 101?"),
        (48.0, "Deer = deer!"),
        (52.5, "Narcissism alert")
    ]
    
    filter_parts = []
    
    # Tiny title (8px font)
    filter_parts.append(
        "drawtext=text='CYNICAL MODE':"
        "fontsize=8:fontcolor=red:borderw=1:bordercolor=white:"
        "x=(w-text_w)/2:y=5:enable='between(t,0,3)'"
    )
    
    # Add each snark with appropriate sizing
    for time, text in snarks:
        escaped = text.replace(':', '\\:').replace(',', '\\,')
        
        # Small red background bar at bottom
        filter_parts.append(
            f"drawbox=x=0:y=h-20:w=w:h=20:"
            f"color=red@0.7:thickness=fill:"
            f"enable='between(t,{time},{time+3})'"
        )
        
        # Tiny text (10px font)
        filter_parts.append(
            f"drawtext=text='{escaped}':"
            f"fontsize=10:fontcolor=white:borderw=1:bordercolor=black:"
            f"x=(w-text_w)/2:y=h-15:"
            f"enable='between(t,{time},{time+3})'"
        )
    
    filter_complex = ",".join(filter_parts)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", filter_complex,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_video
    ]
    
    print(f"🎬 Creating properly sized overlays for 256x116 video...")
    print(f"📐 Video dimensions: 256x116 pixels")
    print(f"🔤 Font size: 8-10px (appropriate for tiny video)")
    print(f"\n😈 CYNICAL SNARKS (shortened for space):")
    for time, text in snarks:
        print(f"  {time:5.1f}s: \"{text}\"")
    
    print(f"\n⏳ Processing...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n✅ Success! Created: {output_video}")
        print(f"📊 File size: {size_mb:.2f} MB")
        print(f"✨ Text now properly fits in the 256x116 frame!")
        return output_video
    else:
        print(f"❌ Error: {result.stderr[:200]}")
        return None

if __name__ == "__main__":
    create_proper_overlays()