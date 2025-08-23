#!/usr/bin/env python3
"""
Create a demo video with actual visual content and text overlays
to demonstrate the auto-snark narrator capability
"""

import subprocess
import json
import os

def create_demo_video_with_content():
    """Create a demo video with actual tutorial-like content"""
    
    output_file = "demo_tutorial.mp4"
    
    # Create a video with text overlays simulating a tutorial
    # Using FFmpeg's drawtext filter to add tutorial-like text
    
    filter_complex = """
    [0:v]
    drawtext=text='TUTORIAL\: How to Build Amazing Things':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=48:
        fontcolor=white:borderw=3:bordercolor=black:
        x=(w-text_w)/2:y=100:
        enable='between(t,0.1,2.6)',
    drawtext=text='Step 1\: Getting Started':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=36:
        fontcolor=yellow:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=300:
        enable='between(t,2.8,7.9)',
    drawtext=text='Actually... let me show you a better way':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=32:
        fontcolor=cyan:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=400:
        enable='between(t,4,7.9)',
    drawtext=text='But wait... there is more!':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=36:
        fontcolor=lime:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=300:
        enable='between(t,9.2,13.1)',
    drawtext=text='Anyway... back to the main topic':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=32:
        fontcolor=orange:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=300:
        enable='between(t,15.5,19)',
    drawtext=text='OK! Here comes the important part':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=36:
        fontcolor=red:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=300:
        enable='between(t,21.2,25.2)',
    drawtext=text='Loading advanced features...':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=28:
        fontcolor=white:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=500:
        enable='between(t,25.5,29.3)',
    drawtext=text='Wait! I forgot something crucial':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=32:
        fontcolor=yellow:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=300:
        enable='between(t,31.8,35.6)',
    drawtext=text='Seriously... pay attention now':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=36:
        fontcolor=red:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=300:
        enable='between(t,37.2,41.5)',
    drawtext=text='Moving on to implementation':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=32:
        fontcolor=lime:borderw=2:bordercolor=black:
        x=(w-text_w)/2:y=300:
        enable='between(t,49.5,53.2)',
    drawtext=text='THE END':
        fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=64:
        fontcolor=white:borderw=3:bordercolor=black:
        x=(w-text_w)/2:y=300:
        enable='between(t,61.2,65.3)'
    """
    
    # Create gradient background with animated colors
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", 
        "-i", f"gradients=size=1280x720:duration=70:speed=0.1:nb_colors=4:c0=0x1e3c72:c1=0x2a5298:c2=0x7e8aa2:c3=0xffffff",
        "-f", "lavfi", 
        "-i", "anoisesrc=d=70:c=pink:r=44100:a=0.02",  # Subtle background audio
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_file
    ]
    
    print(f"🎬 Creating demo tutorial video: {output_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Demo video created: {output_file}")
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
        return output_file
    else:
        print(f"❌ Error creating video: {result.stderr}")
        return None

def run_snark_narrator_on_demo():
    """Run the snark narrator on the demo video"""
    
    print("\n🎙️ Running Auto-Snark Narrator on demo video...")
    
    # Use the test transcript we already have
    cmd = [
        "python", "snark_narrator.py",
        "--video", "demo_tutorial.mp4",
        "--transcript", "test_transcript.json",
        "--out", "demo_tutorial_snarked.mp4",
        "--no-elevenlabs",  # Use local TTS
        "--max-snarks", "8",
        "--min-gap", "8",
        "--style", "wry",
        "--log-level", "INFO"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    # Check if report was generated
    report_file = "demo_tutorial_snarked.mp4.report.json"
    if os.path.exists(report_file):
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        print("\n📊 SNARK REPORT:")
        print(f"  • Candidates found: {report['counts']['candidates']}")
        print(f"  • Snarks inserted: {report['counts']['selected']}")
        print(f"  • Total characters: {report['estimates']['tts_total_chars']}")
        print(f"  • Estimated cost: ${report['estimates']['approx_cost_usd']:.5f}")
        
        if report['inserts']:
            print("\n  🎯 Snark insertions:")
            for ins in report['inserts']:
                print(f"    - {ins['time_s']:.1f}s: \"{ins['text'][:50]}...\"")

if __name__ == "__main__":
    # Create the demo video
    video = create_demo_video_with_content()
    
    if video:
        # Run snark narrator
        run_snark_narrator_on_demo()
        
        print("\n✨ DEMO COMPLETE!")
        print("Files created:")
        print("  • demo_tutorial.mp4 - Original tutorial video")
        print("  • demo_tutorial_snarked.mp4 - Video with snarky commentary")
        print("  • demo_tutorial_snarked.mp4.report.json - Detailed report")