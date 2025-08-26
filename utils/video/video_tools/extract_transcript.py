#!/usr/bin/env python3
"""
Video Transcript Extractor

Extracts audio from video, compresses it, and uses Gemini API to generate
a transcript with millisecond-precision timestamps.
"""

import argparse
import subprocess
import json
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()


def extract_audio(video_path, output_dir):
    """
    Extract and compress audio from video
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save audio file
    
    Returns:
        Path to extracted audio file
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create small audio file (mono, low bitrate for smaller size)
    audio_path = output_dir / f"{video_path.stem}_audio.mp3"
    
    print(f"[{time.strftime('%H:%M:%S')}] Extracting audio from video...")
    
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vn',  # No video
        '-ac', '1',  # Mono
        '-ar', '16000',  # 16kHz sample rate
        '-ab', '32k',  # 32 kbps bitrate
        '-f', 'mp3',
        '-y', str(audio_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error extracting audio: {result.stderr}")
        return None
    
    # Get file size
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"✅ Audio extracted: {audio_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    return audio_path


def get_transcript_from_gemini(audio_path, model=None):
    """
    Get transcript with timestamps using Gemini API
    
    Args:
        audio_path: Path to audio file
        model: Gemini model instance
    
    Returns:
        Transcript data as dictionary
    """
    if model is None:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            print("❌ No GEMINI_API_KEY found in .env")
            return None
        
        genai.configure(api_key=gemini_key)
        
        # Use the strongest model if available, then fast expensive, then default to flash
        model_name = os.environ.get("GEMINI_MODEL_STRONG") or os.environ.get("GEMINI_MODEL_FAST_EXPENSIVE", "gemini-1.5-flash")
        print(f"   Using model: {model_name}")
        model = genai.GenerativeModel(model_name)
    
    print(f"[{time.strftime('%H:%M:%S')}] Uploading audio to Gemini...")
    
    # Upload the audio file to Gemini
    try:
        # Upload file using the Files API
        uploaded_file = genai.upload_file(path=str(audio_path), mime_type='audio/mpeg')
        print(f"   Uploaded: {uploaded_file.name}")
        
        # Wait for file to be processed
        import time as time_module
        time_module.sleep(1)
        
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        print("   Trying direct approach...")
        
        # Alternative: Read file and send as blob
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Create a Part object for the audio
        from google.generativeai.types import content_types
        audio_part = content_types.Part(
            inline_data=content_types.Blob(
                mime_type='audio/mpeg',
                data=audio_bytes
            )
        )
        uploaded_file = audio_part
    
    # Get actual audio duration first
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            audio_duration_s = float(result.stdout.strip())
            audio_duration_ms = int(audio_duration_s * 1000)
            print(f"   Audio duration: {audio_duration_s:.1f} seconds")
        else:
            audio_duration_ms = 0
    except:
        audio_duration_ms = 0
    
    # Create specific prompt for transcript extraction
    prompt = f"""Analyze this entire audio file ({audio_duration_s:.1f} seconds) and provide a COMPLETE transcript with precise timestamps.

IMPORTANT: 
- The audio is {audio_duration_s:.1f} seconds long. Please transcribe the ENTIRE duration.
- Continue transcribing until the end of the audio, even if there are pauses or music.
- Return ONLY valid JSON in the exact format below, with no additional text before or after.

Return the transcript in this exact JSON format:
{{
  "transcript": [
    {{
      "start_ms": <start time in milliseconds>,
      "end_ms": <end time in milliseconds>,
      "text": "<spoken text>",
      "speaker": "<speaker identifier if multiple speakers, otherwise 'narrator'>"
    }}
  ],
  "metadata": {{
    "total_duration_ms": {audio_duration_ms},
    "total_segments": <number of transcript segments>,
    "has_speech": <true if speech detected, false if silent/music only>
  }}
}}

Guidelines:
1. Create a new segment for each distinct phrase or sentence
2. Include ALL spoken words, even if unclear (mark with [unclear] if needed)
3. Use millisecond precision for timestamps (1 second = 1000 milliseconds)
4. Do not include non-speech sounds (music, effects) unless they're important
5. If there are long silences (>2 seconds), end the current segment and start a new one after the silence
6. Keep segments generally under 10 seconds (10000 ms) unless it would break mid-sentence
7. If no speech is detected, return an empty transcript array with has_speech: false

Return ONLY the JSON object, no markdown formatting, no explanations."""
    
    print(f"[{time.strftime('%H:%M:%S')}] Requesting transcript from Gemini...")
    
    try:
        # Send request to Gemini with the audio file
        response = model.generate_content([prompt, uploaded_file])
        result_text = response.text.strip()
        
        print(f"[{time.strftime('%H:%M:%S')}] Received response from Gemini")
        
        # Clean up response if it has markdown
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        transcript_data = json.loads(result_text)
        
        print(f"✅ Transcript extracted: {len(transcript_data.get('transcript', []))} segments")
        
        # Validate and add missing fields if needed
        if 'metadata' not in transcript_data:
            transcript_data['metadata'] = {}
        
        if 'total_segments' not in transcript_data['metadata']:
            transcript_data['metadata']['total_segments'] = len(transcript_data.get('transcript', []))
        
        if 'has_speech' not in transcript_data['metadata']:
            transcript_data['metadata']['has_speech'] = len(transcript_data.get('transcript', [])) > 0
        
        return transcript_data
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing Gemini response as JSON: {e}")
        print(f"Response was: {result_text[:500]}...")
        return None
    except Exception as e:
        print(f"❌ Error getting transcript from Gemini: {e}")
        return None


def save_transcript(transcript_data, output_dir, video_name):
    """
    Save transcript to JSON file
    
    Args:
        transcript_data: Transcript dictionary
        output_dir: Directory to save transcript
        video_name: Name of the video (for filename)
    
    Returns:
        Path to saved transcript file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transcript_path = output_dir / f"{video_name}_transcript.json"
    
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Transcript saved: {transcript_path}")
    
    # Also save a readable text version
    text_path = output_dir / f"{video_name}_transcript.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript for {video_name}\n")
        f.write("=" * 50 + "\n\n")
        
        if transcript_data and 'transcript' in transcript_data:
            for segment in transcript_data['transcript']:
                start_s = segment['start_ms'] / 1000
                end_s = segment['end_ms'] / 1000
                text = segment['text']
                speaker = segment.get('speaker', 'narrator')
                
                f.write(f"[{start_s:.2f}s - {end_s:.2f}s] {speaker}: {text}\n")
        else:
            f.write("No speech detected in audio.\n")
    
    print(f"📝 Text version saved: {text_path}")
    
    return transcript_path


def process_video(video_path):
    """
    Main processing function
    
    Args:
        video_path: Path to input video
    
    Returns:
        Dictionary with paths to generated files
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return None
    
    # Create organized folder structure
    video_name = video_path.stem
    base_dir = video_path.parent / video_name
    soundtrack_dir = base_dir / "soundtrack"
    
    print(f"\n🎬 Processing: {video_path.name}")
    print(f"📁 Output directory: {base_dir}")
    
    # Extract audio
    audio_path = extract_audio(video_path, soundtrack_dir)
    if not audio_path:
        return None
    
    # Get transcript from Gemini
    transcript_data = get_transcript_from_gemini(audio_path)
    if not transcript_data:
        print("⚠️ Could not extract transcript, saving empty transcript file")
        transcript_data = {
            "transcript": [],
            "metadata": {
                "total_duration_ms": 0,
                "total_segments": 0,
                "has_speech": False,
                "error": "Could not extract transcript from audio"
            }
        }
    
    # Save transcript
    transcript_path = save_transcript(transcript_data, soundtrack_dir, video_name)
    
    print(f"\n✅ Processing complete!")
    print(f"   Audio: {audio_path}")
    print(f"   Transcript: {transcript_path}")
    
    return {
        'audio_path': audio_path,
        'transcript_path': transcript_path,
        'transcript_data': transcript_data,
        'base_dir': base_dir
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio and transcript from video using Gemini API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
  - GEMINI_API_KEY must be set in .env file
  - ffmpeg must be installed

Examples:
  # Extract transcript from video
  python extract_transcript.py video.mp4
  
  # Process multiple videos
  python extract_transcript.py video1.mp4 video2.mp4
        """
    )
    
    parser.add_argument('videos', nargs='+', help='Video file(s) to process')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        sys.exit(1)
    
    # Process each video
    results = []
    for video_path in args.videos:
        result = process_video(video_path)
        if result:
            results.append(result)
        print("\n" + "=" * 60 + "\n")
    
    # Summary
    if results:
        print(f"📊 Processed {len(results)} video(s) successfully")
        return 0
    else:
        print(f"❌ No videos were processed successfully")
        return 1


if __name__ == '__main__':
    sys.exit(main())