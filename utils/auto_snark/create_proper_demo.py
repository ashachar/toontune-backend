#!/usr/bin/env python3
"""
Create a proper demo video with actual visual content
Using simpler, more reliable FFmpeg commands
"""

import subprocess
import os

def create_proper_tutorial_video():
    """Create a tutorial video with test patterns and text"""
    
    output_file = "tutorial_demo_proper.mp4"
    
    # Use testsrc2 for a more interesting visual pattern
    # Add text overlays that simulate a tutorial
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=duration=70:size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=70:sample_rate=44100",
        "-vf", (
            "drawtext=text='TUTORIAL - How to Build Amazing Things':"
            "fontsize=48:fontcolor=white:borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=100:enable='between(t,0.1,2.6)',"
            
            "drawtext=text='Step 1 - Getting Started':"
            "fontsize=36:fontcolor=yellow:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=300:enable='between(t,2.8,7.9)',"
            
            "drawtext=text='Actually... let me show you a better way':"
            "fontsize=32:fontcolor=cyan:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=400:enable='between(t,4,7.9)',"
            
            "drawtext=text='But wait... there is more!':"
            "fontsize=36:fontcolor=lime:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=300:enable='between(t,9.2,13.1)',"
            
            "drawtext=text='Anyway... back to the main topic':"
            "fontsize=32:fontcolor=orange:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=300:enable='between(t,15.5,19)',"
            
            "drawtext=text='OK! Here comes the important part':"
            "fontsize=36:fontcolor=red:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=300:enable='between(t,21.2,25.2)',"
            
            "drawtext=text='Wait! I forgot something crucial':"
            "fontsize=32:fontcolor=yellow:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=300:enable='between(t,31.8,35.6)',"
            
            "drawtext=text='Seriously... pay attention now':"
            "fontsize=36:fontcolor=red:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=300:enable='between(t,37.2,41.5)',"
            
            "drawtext=text='Moving on to implementation':"
            "fontsize=32:fontcolor=lime:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=300:enable='between(t,49.5,53.2)',"
            
            "drawtext=text='THE END':"
            "fontsize=64:fontcolor=white:borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=300:enable='between(t,61.2,65.3)'"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_file
    ]
    
    print(f"🎬 Creating proper tutorial video: {output_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Tutorial video created: {output_file}")
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
        
        # Also create a simpler color bars version
        create_color_bars_version()
        
        return output_file
    else:
        print(f"❌ Error: {result.stderr[:500]}")
        return None

def create_color_bars_version():
    """Create a simpler version with color bars"""
    
    output_file = "tutorial_demo_colorbars.mp4"
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "smptebars=duration=70:size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=70",
        "-vf", (
            "drawtext=text='Tutorial Video with Natural Pauses':"
            "fontsize=42:fontcolor=white:borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=50,"
            
            "drawtext=text='[Pause here - 2.2 seconds]':"
            "fontsize=24:fontcolor=yellow:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=600:enable='between(t,2.6,4.8)',"
            
            "drawtext=text='[Discourse marker: Actually]':"
            "fontsize=24:fontcolor=cyan:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=600:enable='between(t,4.8,7)',"
            
            "drawtext=text='[Pause here - 2.3 seconds]':"
            "fontsize=24:fontcolor=yellow:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=600:enable='between(t,7.9,10.2)',"
            
            "drawtext=text='[Discourse marker: But]':"
            "fontsize=24:fontcolor=cyan:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=600:enable='between(t,10.2,13)'"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-t", "30",  # Just 30 seconds for this one
        output_file
    ]
    
    print(f"\n🎨 Creating color bars version: {output_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Color bars video created: {output_file}")
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")

def create_animated_demo():
    """Create an animated demo with moving elements"""
    
    output_file = "tutorial_demo_animated.mp4"
    
    # Create video with animated color background
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x2C3E50:s=1280x720:d=30:r=30",
        "-f", "lavfi", "-i", "anoisesrc=d=30:c=pink:r=44100:a=0.02",
        "-filter_complex", (
            "[0:v]"
            "drawbox=x=0:y=0:w=iw:h=ih:color=0x34495E@0.8:t=fill,"
            "drawtext=text='AUTO-SNARK NARRATOR DEMO':"
            "fontsize=48:fontcolor=white:borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=100,"
            
            "drawbox=x=100:y=250:w=1080:h=100:"
            "color=red@0.3:t=fill:enable='between(t,2,5)',"
            "drawtext=text='🎙️ SNARK: Bold choice. Not judging.':"
            "fontsize=32:fontcolor=white:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=280:enable='between(t,2,5)',"
            
            "drawbox=x=100:y=250:w=1080:h=100:"
            "color=blue@0.3:t=fill:enable='between(t,8,11)',"
            "drawtext=text='🎙️ SNARK: Plot twist no one asked for.':"
            "fontsize=32:fontcolor=white:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=280:enable='between(t,8,11)',"
            
            "drawbox=x=100:y=250:w=1080:h=100:"
            "color=green@0.3:t=fill:enable='between(t,14,17)',"
            "drawtext=text='🎙️ SNARK: Ah yes, the professional approach.':"
            "fontsize=32:fontcolor=white:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=280:enable='between(t,14,17)',"
            
            "drawtext=text='Beat Detection: PAUSE (2.2s gap)':"
            "fontsize=20:fontcolor=yellow:borderw=1:bordercolor=black:"
            "x=50:y=500:enable='between(t,2,2.5)',"
            
            "drawtext=text='Beat Detection: DISCOURSE MARKER (actually)':"
            "fontsize=20:fontcolor=cyan:borderw=1:bordercolor=black:"
            "x=50:y=500:enable='between(t,8,8.5)',"
            
            "drawtext=text='Beat Detection: PAUSE + MARKER':"
            "fontsize=20:fontcolor=lime:borderw=1:bordercolor=black:"
            "x=50:y=500:enable='between(t,14,14.5)'"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "96k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_file
    ]
    
    print(f"\n🎞️ Creating animated demo: {output_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Animated demo created: {output_file}")
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
        
        print("\n📋 DEMO FEATURES:")
        print("  • Test pattern background (testsrc2)")
        print("  • Color bars (SMPTE)")
        print("  • Animated snark overlays")
        print("  • Beat detection indicators")
        print("  • Tutorial-style text progression")
        print("  • Audio track (sine wave)")

if __name__ == "__main__":
    # Create all three demo versions
    create_proper_tutorial_video()
    create_animated_demo()
    
    print("\n✨ ALL DEMOS CREATED!")
    print("\nYou can now run the snark narrator on these videos:")
    print("  python snark_narrator.py --video tutorial_demo_proper.mp4 --transcript test_transcript.json --out final_with_snark.mp4")