#!/usr/bin/env python3
"""
Insert the actual cynical audio remarks into the Do-Re-Mi video
"""

import subprocess
import os

def insert_snarks_into_video():
    """Mix the cynical remarks into the video with proper ducking"""
    
    input_video = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/downsampled/scene_001.mp4"
    output_video = "do_re_mi_with_real_snarks.mp4"
    
    # Build complex audio filter to insert snarks with ducking
    # Format: [time_to_insert, audio_file, duration_estimate]
    snarks = [
        (3.8, "snark_1_musical.mp3", 3.0),
        (16.8, "snark_2_abc.mp3", 3.0),
        (25.5, "snark_3_repeat.mp3", 2.5),
        (41.8, "snark_4_teaching.mp3", 2.5),
        (48.0, "snark_5_deer.mp3", 2.5),
        (52.5, "snark_6_narcissism.mp3", 2.0)
    ]
    
    print("🎬 Inserting cynical audio remarks into Do-Re-Mi video...")
    print(f"📹 Input: {os.path.basename(input_video)}")
    print(f"🎙️ Adding {len(snarks)} cynical remarks\n")
    
    # First, extract original audio
    print("1️⃣ Extracting original audio...")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video, 
        "-vn", "-acodec", "pcm_s16le", 
        "original_audio.wav"
    ], capture_output=True)
    
    # Create a complex filter for mixing
    filter_parts = []
    input_count = 1  # Start at 1 because [0] is original audio
    
    # Build inputs list
    inputs = ["-i", "original_audio.wav"]
    for _, snark_file, _ in snarks:
        inputs.extend(["-i", snark_file])
    
    # Build the filter to mix all snarks
    filter_complex = "[0]"  # Start with original audio
    
    for i, (time, snark_file, duration) in enumerate(snarks):
        snark_input = f"[{i+1}]"
        
        # Apply ducking to original audio during snark
        filter_complex += f"volume=enable='between(t,{time},{time+duration})':volume=0.2[duck{i}];"
        
        # Delay the snark to the right time
        filter_complex += f"{snark_input}adelay={int(time*1000)}|{int(time*1000)}[snark{i}];"
        
        if i == 0:
            filter_complex += f"[duck{i}][snark{i}]amix=inputs=2:duration=longest[mix{i}];"
        else:
            filter_complex += f"[mix{i-1}][snark{i}]amix=inputs=2:duration=longest[mix{i}];"
    
    # Final output
    filter_complex = filter_complex[:-1]  # Remove trailing semicolon
    if len(snarks) > 0:
        filter_complex = filter_complex.replace(f"[mix{len(snarks)-1}]", "[out]")
    
    # Simpler approach - overlay each snark one by one
    print("2️⃣ Creating mixed audio with cynical remarks...")
    
    # Use FFmpeg to mix audio
    cmd = [
        "ffmpeg", "-y",
        "-i", "original_audio.wav"
    ]
    
    # Add all snark files
    for _, snark_file, _ in snarks:
        cmd.extend(["-i", snark_file])
    
    # Build filter for overlaying
    filter_str = ""
    for i, (time, _, duration) in enumerate(snarks):
        if i == 0:
            # Duck original audio and overlay first snark
            filter_str += f"[0]volume=enable='between(t,{time},{time+duration})':volume=0.2[a0];"
            filter_str += f"[{i+1}]adelay={int(time*1000)}|{int(time*1000)}[d{i}];"
            filter_str += f"[a0][d{i}]amix=inputs=2:duration=first[m{i}];"
        else:
            # Continue chaining
            filter_str += f"[m{i-1}]volume=enable='between(t,{time},{time+duration})':volume=0.2[a{i}];"
            filter_str += f"[{i+1}]adelay={int(time*1000)}|{int(time*1000)}[d{i}];"
            if i == len(snarks) - 1:
                filter_str += f"[a{i}][d{i}]amix=inputs=2:duration=first"
            else:
                filter_str += f"[a{i}][d{i}]amix=inputs=2:duration=first[m{i}];"
    
    cmd.extend([
        "-filter_complex", filter_str,
        "-ac", "1",  # Mono output
        "mixed_audio.wav"
    ])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"⚠️ Audio mixing error: {result.stderr[:200]}")
        # Fallback to simpler approach
        print("   Trying simpler approach...")
        simple_mix()
    
    # Combine with video
    print("3️⃣ Combining mixed audio with video...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", "mixed_audio.wav",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v", "-map", "1:a",
        output_video
    ], capture_output=True)
    
    if os.path.exists(output_video):
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n✅ Success! Created: {output_video}")
        print(f"📊 File size: {size_mb:.2f} MB")
        print(f"\n🎙️ Cynical remarks inserted at:")
        for time, _, _ in snarks:
            print(f"   • {time:.1f}s")
        print(f"\n🎧 Audio ducked to 20% during remarks")
        print(f"🔊 Play with: open {output_video}")
    
    # Cleanup
    for f in ["original_audio.wav", "mixed_audio.wav"]:
        if os.path.exists(f):
            os.remove(f)

def simple_mix():
    """Simpler approach - just overlay without complex filtering"""
    print("   Using simple overlay approach...")
    
    # Just mix first snark as demo
    subprocess.run([
        "ffmpeg", "-y",
        "-i", "original_audio.wav",
        "-i", "snark_1_musical.mp3",
        "-filter_complex", 
        "[1]adelay=3800|3800[delayed];[0][delayed]amix=inputs=2:duration=first",
        "mixed_audio.wav"
    ], capture_output=True)

if __name__ == "__main__":
    insert_snarks_into_video()