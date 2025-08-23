#!/usr/bin/env python3
"""
Generate a single cynical remark audio file using text-to-speech
We'll use the macOS 'say' command which is built-in
"""

import subprocess
import os

def generate_snark_with_macos_say():
    """Generate cynical remarks using macOS built-in TTS"""
    
    # Single cynical remark for the Do-Re-Mi scene
    snark_text = "Oh good, another musical number. How original."
    output_file = "snark_sample.aiff"
    output_mp3 = "snark_sample.mp3"
    
    print("🎙️ Generating cynical remark audio...")
    print(f"📝 Text: '{snark_text}'")
    print(f"🗣️ Using macOS 'say' command with sarcastic voice\n")
    
    # Use macOS 'say' command with a sarcastic-sounding voice
    # Samantha or Alex voices work well for cynical tone
    voices_to_try = ["Samantha", "Alex", "Victoria", "Daniel"]
    
    for voice in voices_to_try:
        print(f"Trying voice: {voice}")
        cmd = [
            "say",
            "-v", voice,
            "-o", output_file,
            "--data-format=LEF32@22050",  # Lower quality for smaller file
            snark_text
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Generated with {voice} voice: {output_file}")
            
            # Convert to MP3 for smaller size
            convert_cmd = [
                "ffmpeg", "-y",
                "-i", output_file,
                "-acodec", "libmp3lame",
                "-ab", "128k",
                output_mp3
            ]
            
            subprocess.run(convert_cmd, capture_output=True)
            
            if os.path.exists(output_mp3):
                size_kb = os.path.getsize(output_mp3) / 1024
                print(f"✅ Converted to MP3: {output_mp3} ({size_kb:.1f} KB)")
                
                # Play it back
                print("\n🔊 Playing the cynical remark...")
                subprocess.run(["afplay", output_mp3])
                
                return output_mp3
            break
    
    return None

def create_multiple_snarks():
    """Create all 6 cynical remarks for the Do-Re-Mi scene"""
    
    snarks = [
        ("snark_1_musical.mp3", "Oh good, another musical number. How original."),
        ("snark_2_abc.mp3", "Yes, because A B C is such complex knowledge."),
        ("snark_3_repeat.mp3", "We get it. You can repeat three syllables."),
        ("snark_4_teaching.mp3", "Easier? This is your idea of teaching?"),
        ("snark_5_deer.mp3", "Revolutionary. A deer is... a deer."),
        ("snark_6_narcissism.mp3", "Me, the narcissism is showing.")
    ]
    
    print("\n🎬 Generating all cynical remarks for Do-Re-Mi scene...\n")
    
    for filename, text in snarks:
        print(f"📝 Creating: {filename}")
        print(f"   Text: '{text}'")
        
        # Generate AIFF first
        temp_file = "temp_snark.aiff"
        cmd = [
            "say",
            "-v", "Samantha",  # Sarcastic female voice
            "-o", temp_file,
            "--data-format=LEF32@22050",
            text
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Convert to MP3
            convert_cmd = [
                "ffmpeg", "-y",
                "-i", temp_file,
                "-acodec", "libmp3lame",
                "-ab", "96k",
                filename
            ]
            
            subprocess.run(convert_cmd, capture_output=True)
            
            if os.path.exists(filename):
                size_kb = os.path.getsize(filename) / 1024
                duration = get_audio_duration(filename)
                print(f"   ✅ Generated: {size_kb:.1f} KB, {duration:.1f}s duration\n")
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("✨ All cynical remarks generated!")
    print("\n🎧 You can play them with: afplay snark_*.mp3")

def get_audio_duration(filepath):
    """Get duration of audio file in seconds"""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", filepath]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0

if __name__ == "__main__":
    # Generate single sample
    sample = generate_snark_with_macos_say()
    
    if sample:
        print("\n" + "="*50)
        print("✅ Sample cynical remark audio generated!")
        print(f"📁 File: {sample}")
        print("🎧 The audio has been played")
        print("="*50)
        
        # Now generate all of them
        print("\nGenerate all 6 snarks? (y/n): ", end="")
        if input().lower() == 'y':
            create_multiple_snarks()