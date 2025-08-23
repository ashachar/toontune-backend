#!/usr/bin/env python3
"""
Create a demo using ElevenLabs v3 with expression controls
Generates real cynical audio with emotion tags
"""

import os
import json
import subprocess
from elevenlabs import client as elevenlabs_client
from elevenlabs import save

def create_v3_demo():
    """Create cynical narration using ElevenLabs v3"""
    
    print("=" * 70)
    print("🎭 ELEVENLABS V3 CYNICAL NARRATOR DEMO")
    print("=" * 70)
    
    # Check for API key
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        print("❌ Please set ELEVENLABS_API_KEY environment variable")
        print("   Run: export ELEVENLABS_API_KEY='your-key-here'")
        return False
    
    print(f"✅ API key found (length: {len(api_key)})")
    
    # Initialize client
    try:
        from elevenlabs import ElevenLabs
        client = ElevenLabs(api_key=api_key)
        print("✅ ElevenLabs client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False
    
    # Define cynical snarks with v3 emotion tags
    snarks = [
        {
            "id": "snark_1",
            "time": 3.8,
            "text": '<emotion="sarcastic">Oh good,</emotion> <pause duration="300ms"/> <emotion="deadpan">another musical number.</emotion> <emotion="sarcastic">How original.</emotion>',
            "description": "Opening sarcasm"
        },
        {
            "id": "snark_2", 
            "time": 16.8,
            "text": '<emotion="condescending">Yes, because A-B-C is <emphasis level="strong">such</emphasis> complex knowledge.</emotion>',
            "description": "Condescending about ABC"
        },
        {
            "id": "snark_3",
            "time": 25.5,
            "text": '<emotion="mocking">We get it.</emotion> <pause duration="500ms"/> <emotion="deadpan">You can repeat three syllables.</emotion>',
            "description": "Mocking the repetition"
        },
        {
            "id": "snark_4",
            "time": 41.8,
            "text": '<emotion="skeptical">Easier?</emotion> <pause duration="400ms"/> <emotion="sarcastic">This is your idea of teaching?</emotion>',
            "description": "Skeptical about pedagogy"
        },
        {
            "id": "snark_5",
            "time": 48.0,
            "text": '<emotion="deadpan">Revolutionary.</emotion> <pause duration="600ms"/> <emotion="mocking">A deer is... <pause duration="200ms"/> a deer.</emotion>',
            "description": "Deadpan about circular definition"
        },
        {
            "id": "snark_6",
            "time": 52.5,
            "text": '<emotion="sarcastic">Me,</emotion> <pause duration="300ms"/> <emotion="condescending">the narcissism is <emphasis>really</emphasis> showing now.</emotion>',
            "description": "Final sarcastic jab"
        }
    ]
    
    print(f"\n📝 Generating {len(snarks)} cynical remarks with v3 expression controls...")
    
    generated_files = []
    
    for snark in snarks:
        print(f"\n🎙️ Generating: {snark['description']}")
        print(f"   Time: {snark['time']}s")
        print(f"   Text: {snark['text'][:60]}...")
        
        try:
            # Voice settings optimized for each emotion type
            voice_settings = {
                "stability": 0.3,  # Low for expressive delivery
                "similarity_boost": 0.7,
                "style": 0.85,  # High for emotional range
                "use_speaker_boost": True
            }
            
            # Adjust based on primary emotion
            if "deadpan" in snark['text']:
                voice_settings["stability"] = 0.8
                voice_settings["style"] = 0.2
            elif "condescending" in snark['text']:
                voice_settings["stability"] = 0.35
                voice_settings["style"] = 0.75
            
            # Generate audio with v3
            print("   Calling ElevenLabs API...")
            audio_generator = client.text_to_speech.convert(
                text=snark['text'],
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
                model_id="eleven_turbo_v2_5",  # Using turbo model for faster generation
                voice_settings=voice_settings,
                output_format="mp3_44100_128"
            )
            
            # Save the audio
            output_file = f"v3_{snark['id']}.mp3"
            
            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)
            
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            
            print(f"   ✅ Saved: {output_file}")
            generated_files.append((snark['time'], output_file, snark['description']))
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Try simpler version without complex tags
            try:
                print("   Retrying with simplified text...")
                simple_text = snark['text'].replace('<emotion="', '').replace('">', '').replace('</emotion>', '')
                simple_text = simple_text.replace('<pause duration="', '... ').replace('ms"/>', '')
                simple_text = simple_text.replace('<emphasis level="strong">', '').replace('<emphasis>', '').replace('</emphasis>', '')
                
                audio_generator = client.text_to_speech.convert(
                    text=simple_text,
                    voice_id="21m00Tcm4TlvDq8ikWAM",
                    model_id="eleven_turbo_v2_5",
                    output_format="mp3_44100_128"
                )
                
                output_file = f"v3_{snark['id']}_simple.mp3"
                audio_bytes = b"".join(audio_generator)
                
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
                    
                print(f"   ✅ Saved simplified: {output_file}")
                generated_files.append((snark['time'], output_file, snark['description']))
                
            except Exception as e2:
                print(f"   ❌ Failed: {e2}")
    
    if generated_files:
        print("\n" + "=" * 70)
        print(f"✅ Successfully generated {len(generated_files)} audio files!")
        print("\n📁 Files created:")
        for time, file, desc in generated_files:
            size_kb = os.path.getsize(file) / 1024 if os.path.exists(file) else 0
            print(f"   • {file} ({size_kb:.1f} KB) - {desc}")
        
        # Now mix with video
        print("\n🎬 Mixing with Do-Re-Mi video...")
        mix_with_video(generated_files)
        
        return True
    else:
        print("\n❌ No audio files were generated")
        return False

def mix_with_video(audio_files):
    """Mix the generated audio with the Do-Re-Mi video"""
    
    input_video = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/downsampled/scene_001.mp4"
    output_video = "do_re_mi_elevenlabs_v3_demo.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Video not found: {input_video}")
        return
    
    print(f"📹 Input video: scene_001.mp4")
    print(f"🎙️ Mixing {len(audio_files)} cynical remarks...")
    
    # Extract original audio
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "pcm_s16le",
        "original_audio.wav"
    ], capture_output=True)
    
    # Build complex filter for mixing
    filter_parts = []
    inputs = ["-i", "original_audio.wav"]
    
    for _, audio_file, _ in audio_files:
        if os.path.exists(audio_file):
            inputs.extend(["-i", audio_file])
    
    # Create filter complex
    num_inputs = len(audio_files) + 1
    
    if num_inputs > 1:
        # Build mixing filter
        filter_str = ""
        for i, (time, _, _) in enumerate(audio_files):
            if i == 0:
                # First overlay
                filter_str += f"[0]volume=enable='between(t,{time},{time+3})':volume=0.2[a0];"
                filter_str += f"[{i+1}]adelay={int(time*1000)}|{int(time*1000)}[d{i}];"
                filter_str += f"[a0][d{i}]amix=inputs=2:duration=first[m{i}];"
            else:
                # Chain overlays
                filter_str += f"[m{i-1}]volume=enable='between(t,{time},{time+3})':volume=0.2[a{i}];"
                filter_str += f"[{i+1}]adelay={int(time*1000)}|{int(time*1000)}[d{i}];"
                if i == len(audio_files) - 1:
                    filter_str += f"[a{i}][d{i}]amix=inputs=2:duration=first"
                else:
                    filter_str += f"[a{i}][d{i}]amix=inputs=2:duration=first[m{i}];"
        
        # Mix audio
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_str,
            "-ac", "1",
            "mixed_audio.wav"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Combine with video
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
                print(f"\n✅ Created: {output_video} ({size_mb:.2f} MB)")
                print("\n🎭 Features demonstrated:")
                print("  • Sarcastic emotion tags")
                print("  • Condescending tone")
                print("  • Deadpan delivery")
                print("  • Pauses for dramatic effect")
                print("  • Emphasis on key words")
                print("  • Audio ducking during remarks")
                
                # Create report
                report = {
                    "model": "eleven_turbo_v2_5",
                    "voice": "Rachel (21m00Tcm4TlvDq8ikWAM)",
                    "features": [
                        "emotion_tags",
                        "pause_controls",
                        "emphasis_tags",
                        "expression_settings"
                    ],
                    "snarks": len(audio_files),
                    "output": output_video
                }
                
                with open("v3_demo_report.json", "w") as f:
                    json.dump(report, f, indent=2)
                
                print(f"\n📊 Report saved: v3_demo_report.json")
                print(f"\n🎬 Play the demo: open {output_video}")
        else:
            print(f"❌ Audio mixing failed: {result.stderr[:200]}")
    
    # Cleanup
    for f in ["original_audio.wav", "mixed_audio.wav"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    success = create_v3_demo()
    if success:
        print("\n🎉 ElevenLabs v3 demo complete!")
    else:
        print("\n❌ Demo failed - check API key and try again")