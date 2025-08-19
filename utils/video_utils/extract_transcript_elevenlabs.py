#!/usr/bin/env python3
"""
Video Transcript Extractor using ElevenLabs API

Extracts audio from video and uses ElevenLabs transcription API to generate
a transcript with word-by-word timestamps.
"""

import argparse
import subprocess
import json
import sys
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

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
    
    # Create audio file in MP3 format for ElevenLabs
    audio_path = output_dir / f"{video_path.stem}_audio.mp3"
    
    print(f"[{time.strftime('%H:%M:%S')}] Extracting audio from video...")
    
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vn',  # No video
        '-ac', '1',  # Mono
        '-ar', '22050',  # 22kHz sample rate
        '-ab', '128k',  # 128 kbps bitrate for good quality
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


def get_transcript_from_elevenlabs(audio_path):
    """
    Get transcript with word-level timestamps using ElevenLabs API
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Transcript data as dictionary
    """
    # Get API key from environment
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ No ELEVENLABS_API_KEY found in .env")
        return None
    
    print(f"[{time.strftime('%H:%M:%S')}] Uploading audio to ElevenLabs...")
    
    # ElevenLabs API endpoint for dubbing/transcription
    # Note: ElevenLabs dubbing API includes transcription with word-level timestamps
    url = "https://api.elevenlabs.io/v1/dubbing"
    
    headers = {
        "xi-api-key": api_key
    }
    
    # Open and upload the audio file
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {
                'audio': (audio_path.name, audio_file, 'audio/mpeg')
            }
            
            # Request parameters for detailed transcription
            data = {
                'model': 'eleven_turbo_v2',  # Latest model for best accuracy
                'language': 'en',  # English language
                'output_format': 'json_verbose',  # Get detailed output with word timestamps
            }
            
            print(f"[{time.strftime('%H:%M:%S')}] Requesting transcription from ElevenLabs...")
            
            response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code != 200:
                print(f"❌ Error from ElevenLabs API: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
            
            print(f"[{time.strftime('%H:%M:%S')}] Received response from ElevenLabs")
            
            # Parse the response
            result = response.json()
            
            # Convert ElevenLabs format to our standard format
            transcript_segments = []
            word_timestamps = []
            
            # Process segments if available
            if 'segments' in result:
                for segment in result['segments']:
                    transcript_segments.append({
                        "start_ms": int(segment.get('start', 0) * 1000),
                        "end_ms": int(segment.get('end', 0) * 1000),
                        "text": segment.get('text', '').strip(),
                        "speaker": segment.get('speaker', 'narrator')
                    })
            
            # Process word-level timestamps if available
            if 'words' in result:
                for word_data in result['words']:
                    word_timestamps.append({
                        "word": word_data.get('word', ''),
                        "start_ms": int(word_data.get('start', 0) * 1000),
                        "end_ms": int(word_data.get('end', 0) * 1000),
                        "confidence": word_data.get('confidence', 1.0)
                    })
            
            # If no segments but we have words, create segments from words
            if not transcript_segments and word_timestamps:
                # Group words into sentence-like segments
                current_segment = {
                    "start_ms": word_timestamps[0]["start_ms"],
                    "end_ms": word_timestamps[0]["end_ms"],
                    "text": word_timestamps[0]["word"],
                    "speaker": "narrator"
                }
                
                for word in word_timestamps[1:]:
                    # Check if this word starts a new sentence (after punctuation or long pause)
                    gap = word["start_ms"] - current_segment["end_ms"]
                    ends_with_punctuation = current_segment["text"].rstrip().endswith(('.', '!', '?'))
                    
                    if gap > 1000 or ends_with_punctuation:  # 1 second gap or sentence end
                        # Save current segment and start new one
                        transcript_segments.append(current_segment)
                        current_segment = {
                            "start_ms": word["start_ms"],
                            "end_ms": word["end_ms"],
                            "text": word["word"],
                            "speaker": "narrator"
                        }
                    else:
                        # Add word to current segment
                        current_segment["text"] += " " + word["word"]
                        current_segment["end_ms"] = word["end_ms"]
                
                # Add the last segment
                if current_segment["text"]:
                    transcript_segments.append(current_segment)
            
            # Get total duration
            total_duration_ms = 0
            if transcript_segments:
                total_duration_ms = max(seg['end_ms'] for seg in transcript_segments)
            elif word_timestamps:
                total_duration_ms = max(word['end_ms'] for word in word_timestamps)
            
            transcript_data = {
                "transcript": transcript_segments,
                "metadata": {
                    "total_duration_ms": total_duration_ms,
                    "total_segments": len(transcript_segments),
                    "total_words": len(word_timestamps),
                    "has_speech": len(transcript_segments) > 0,
                    "language": result.get('language', 'en'),
                    "model": "elevenlabs_turbo_v2"
                }
            }
            
            # Include word timestamps
            if word_timestamps:
                transcript_data['word_timestamps'] = word_timestamps
            
            print(f"✅ Transcript extracted: {len(transcript_segments)} segments, {len(word_timestamps)} words")
            
            return transcript_data
            
    except Exception as e:
        print(f"❌ Error getting transcript from ElevenLabs: {e}")
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
    
    transcript_path = output_dir / f"{video_name}_transcript_elevenlabs.json"
    
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Transcript saved: {transcript_path}")
    
    # Also save a readable text version
    text_path = output_dir / f"{video_name}_transcript_elevenlabs.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript for {video_name} (ElevenLabs)\n")
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
        
        # Add word-level timestamps if available
        if 'word_timestamps' in transcript_data and transcript_data['word_timestamps']:
            f.write("\n" + "=" * 50 + "\n")
            f.write("Word-level timestamps:\n\n")
            
            current_line = []
            for word_data in transcript_data['word_timestamps']:
                word = word_data['word']
                start_s = word_data['start_ms'] / 1000
                confidence = word_data.get('confidence', 1.0)
                
                # Format with confidence if less than perfect
                if confidence < 0.95:
                    current_line.append(f"{word}[{start_s:.2f}s,{confidence:.2f}]")
                else:
                    current_line.append(f"{word}[{start_s:.2f}s]")
                
                # Add line break after punctuation
                if word.rstrip().endswith(('.', '!', '?', ',')):
                    f.write(" ".join(current_line) + "\n")
                    current_line = []
            
            # Write any remaining words
            if current_line:
                f.write(" ".join(current_line) + "\n")
    
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
    
    # Get transcript from ElevenLabs
    transcript_data = get_transcript_from_elevenlabs(audio_path)
    if not transcript_data:
        print("⚠️ Could not extract transcript, saving empty transcript file")
        transcript_data = {
            "transcript": [],
            "metadata": {
                "total_duration_ms": 0,
                "total_segments": 0,
                "total_words": 0,
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
        description='Extract audio and transcript from video using ElevenLabs API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
  - ELEVENLABS_API_KEY must be set in .env file
  - ffmpeg must be installed
  - ElevenLabs account with transcription access

Examples:
  # Extract transcript from video
  python extract_transcript_elevenlabs.py video.mp4
  
  # Process multiple videos
  python extract_transcript_elevenlabs.py video1.mp4 video2.mp4
        """
    )
    
    parser.add_argument('videos', nargs='+', help='Video file(s) to process')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("❌ Error: ELEVENLABS_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        print("   Get your API key from: https://elevenlabs.io/app/settings/api-keys")
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