#!/usr/bin/env python3
"""
Create a visual demonstration video showing snark overlays
This simulates what the final output would look like with TTS
"""

import subprocess
import os

def create_visual_snark_demo():
    """Create a video showing snark text overlays at detected beats"""
    
    output_file = "demo_snark_visual.mp4"
    
    # Snark insertions that would be generated
    snarks = [
        (2.7, "Bold choice. Not judging. Okay, maybe a little."),
        (8.1, "Plot twist no one asked for."),
        (13.3, "Ah yes, the professional approach."),
        (19.2, "Narrator: that did not go as planned."),
        (29.5, "Confidence level: unverified."),
        (37.4, "We're calling this... creative efficiency."),
        (48.0, "If you blinked, you missed the logic."),
        (59.1, "Ten out of ten for commitment. Evidence pending.")
    ]
    
    # Build the complex filter for all snarks
    filter_parts = []
    
    # Main title
    filter_parts.append("""
    drawtext=text='AUTO-SNARK NARRATOR DEMO':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=42:
        fontcolor=white:borderw=3:bordercolor=black:
        x=(w-text_w)/2:y=50""")
    
    # Tutorial content overlays
    filter_parts.append("""
    drawtext=text='Welcome to this tutorial...':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=32:
        fontcolor=white:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=200:
        enable='between(t,0.1,2.6)'""")
    
    filter_parts.append("""
    drawtext=text='Actually\, let me show you something...':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=32:
        fontcolor=white:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=200:
        enable='between(t,2.8,7.9)'""")
    
    filter_parts.append("""
    drawtext=text='But wait\, there is more...':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=32:
        fontcolor=white:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=200:
        enable='between(t,9.2,13.1)'""")
    
    # Add snark overlays with red background boxes
    for time, text in snarks:
        # Escape special characters for FFmpeg
        escaped_text = text.replace(':', '\\:').replace(',', '\\,').replace("'", "\\'")
        
        # Add red box background
        filter_parts.append(f"""
    drawbox=x=40:y=h-180:w=w-80:h=100:
        color=red@0.8:thickness=fill:
        enable='between(t,{time},{time+3})'""")
        
        # Add snark text
        filter_parts.append(f"""
    drawtext=text='🎙️ SNARK\\: {escaped_text}':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=28:
        fontcolor=white:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=h-150:
        enable='between(t,{time},{time+3})'""")
        
        # Add timestamp
        filter_parts.append(f"""
    drawtext=text='[{time:.1f}s]':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=20:
        fontcolor=yellow:borderw=1:bordercolor=black:
        x=50:y=h-120:
        enable='between(t,{time},{time+3})'""")
    
    # Add beat detection indicators
    filter_parts.append("""
    drawtext=text='🔍 BEAT DETECTED (pause)':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=18:
        fontcolor=lime:borderw=1:bordercolor=black:
        x=50:y=100:
        enable='between(t,2.6,2.9)+between(t,7.9,8.2)+between(t,13.1,13.4)'""")
    
    filter_parts.append("""
    drawtext=text='🔍 BEAT DETECTED (discourse marker)':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=18:
        fontcolor=cyan:borderw=1:bordercolor=black:
        x=50:y=100:
        enable='between(t,5,5.3)+between(t,17,17.3)+between(t,23,23.3)'""")
    
    # Join all filter parts
    filter_complex = "[0:v]" + ",".join(filter_parts)
    
    # Create the video
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"gradients=size=1280x720:duration=70:speed=0.05:nb_colors=3:c0=0x222222:c1=0x333333:c2=0x444444",
        "-f", "lavfi",
        "-i", "anoisesrc=d=70:c=pink:r=44100:a=0.01",  # Very subtle audio
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "96k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_file
    ]
    
    print(f"🎬 Creating visual snark demonstration: {output_file}")
    print("  This shows where snarky comments would be inserted...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"\n✅ Visual demo created: {output_file}")
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
        
        print("\n🎯 SNARK INSERTIONS VISUALIZED:")
        for time, text in snarks:
            print(f"  • {time:5.1f}s: \"{text}\"")
        
        print("\n📝 FEATURES DEMONSTRATED:")
        print("  ✅ Beat detection indicators (pauses & markers)")
        print("  ✅ Snark overlay timing with 3-second duration")
        print("  ✅ Red box highlighting for narrator comments")
        print("  ✅ Timestamp display for each insertion")
        print("  ✅ Professional tutorial-style presentation")
        
        return output_file
    else:
        print(f"❌ Error: {result.stderr[:500]}")
        return None

if __name__ == "__main__":
    create_visual_snark_demo()