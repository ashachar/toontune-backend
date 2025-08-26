#!/usr/bin/env python3
"""
Video Transcript Extractor using OpenAI Whisper

Extracts audio from video and uses OpenAI Whisper API to generate
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
import openai

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
    
    # Create audio file in a format Whisper supports well (MP3)
    audio_path = output_dir / f"{video_path.stem}_audio.mp3"
    
    print(f"[{time.strftime('%H:%M:%S')}] Extracting audio from video...")
    
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vn',  # No video
        '-ac', '1',  # Mono
        '-ar', '16000',  # 16kHz sample rate (Whisper's preferred)
        '-ab', '64k',  # 64 kbps bitrate
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


def get_transcript_from_whisper(audio_path):
    """
    Get transcript with timestamps using OpenAI Whisper API
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Transcript data as dictionary
    """
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found in .env")
        return None
    
    openai.api_key = api_key
    client = openai.OpenAI(api_key=api_key)
    
    print(f"[{time.strftime('%H:%M:%S')}] Transcribing with Whisper...")
    
    # Open the audio file
    with open(audio_path, 'rb') as audio_file:
        try:
            # Use Whisper API with detailed timestamps
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",  # Get detailed output with segments
                timestamp_granularities=["segment", "word"]  # Get both segment and word timestamps
            )
            
            print(f"[{time.strftime('%H:%M:%S')}] Received response from Whisper")
            
            # Convert Whisper segments to our format
            transcript_segments = []
            
            if hasattr(response, 'segments'):
                for segment in response.segments:
                    transcript_segments.append({
                        "start_ms": int(segment.start * 1000),
                        "end_ms": int(segment.end * 1000),
                        "text": segment.text.strip(),
                        "speaker": "narrator"  # Whisper doesn't do speaker diarization by default
                    })
            
            # Get total duration
            total_duration_ms = 0
            if transcript_segments:
                total_duration_ms = max(seg['end_ms'] for seg in transcript_segments)
            
            transcript_data = {
                "transcript": transcript_segments,
                "metadata": {
                    "total_duration_ms": total_duration_ms,
                    "total_segments": len(transcript_segments),
                    "has_speech": len(transcript_segments) > 0,
                    "language": response.language if hasattr(response, 'language') else "unknown",
                    "model": "whisper-1"
                }
            }
            
            # Also include word-level timestamps if available
            if hasattr(response, 'words') and response.words:
                word_timestamps = []
                for word in response.words:
                    word_timestamps.append({
                        "word": word.word,
                        "start_ms": int(word.start * 1000),
                        "end_ms": int(word.end * 1000)
                    })
                transcript_data['word_timestamps'] = word_timestamps
            
            print(f"✅ Transcript extracted: {len(transcript_segments)} segments")
            
            return transcript_data
            
        except Exception as e:
            print(f"❌ Error getting transcript from Whisper: {e}")
            return None


def save_transcript(transcript_data, output_dir, video_name):
    """
    Save transcript to multiple JSON files for different use cases
    
    Args:
        transcript_data: Transcript dictionary
        output_dir: Directory to save transcript
        video_name: Name of the video (for filename)
    
    Returns:
        Dictionary with paths to saved transcript files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save sentence-level transcript (for scene splitting)
    sentence_transcript_path = output_dir / f"{video_name}_transcript_sentences.json"
    sentence_data = {
        "transcript": transcript_data.get("transcript", []),
        "metadata": transcript_data.get("metadata", {})
    }
    with open(sentence_transcript_path, 'w', encoding='utf-8') as f:
        json.dump(sentence_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Sentence transcript saved: {sentence_transcript_path}")
    
    # 2. Save word-level transcript (for word-by-word display)
    word_transcript_path = output_dir / f"{video_name}_transcript_words.json"
    word_data = {
        "words": transcript_data.get("word_timestamps", []),
        "metadata": {
            "total_words": len(transcript_data.get("word_timestamps", [])),
            "total_duration_ms": transcript_data.get("metadata", {}).get("total_duration_ms", 0),
            "language": transcript_data.get("metadata", {}).get("language", "english"),
            "model": "whisper-1"
        }
    }
    with open(word_transcript_path, 'w', encoding='utf-8') as f:
        json.dump(word_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Word transcript saved: {word_transcript_path}")
    
    # 3. Save combined transcript (original format with everything)
    combined_transcript_path = output_dir / f"{video_name}_transcript_whisper.json"
    with open(combined_transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Combined transcript saved: {combined_transcript_path}")
    
    # Also save a readable text version
    text_path = output_dir / f"{video_name}_transcript_whisper.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript for {video_name} (Whisper)\n")
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
                
                current_line.append(f"{word}[{start_s:.2f}s]")
                
                # Add line break after punctuation
                if word.rstrip().endswith(('.', '!', '?', ',')):
                    f.write(" ".join(current_line) + "\n")
                    current_line = []
            
            # Write any remaining words
            if current_line:
                f.write(" ".join(current_line) + "\n")
    
    print(f"📝 Text version saved: {text_path}")
    
    return {
        'sentence_transcript': sentence_transcript_path,
        'word_transcript': word_transcript_path,
        'combined_transcript': combined_transcript_path,
        'text_transcript': text_path
    }


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
    
    # Get transcript from Whisper
    transcript_data = get_transcript_from_whisper(audio_path)
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
    transcript_paths = save_transcript(transcript_data, soundtrack_dir, video_name)
    
    print(f"\n✅ Processing complete!")
    print(f"   Audio: {audio_path}")
    print(f"   Sentence transcript: {transcript_paths['sentence_transcript']}")
    print(f"   Word transcript: {transcript_paths['word_transcript']}")
    
    return {
        'audio_path': audio_path,
        'transcript_paths': transcript_paths,
        'transcript_data': transcript_data,
        'base_dir': base_dir
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio and transcript from video using OpenAI Whisper API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
  - OPENAI_API_KEY must be set in .env file
  - ffmpeg must be installed
  - Supports audio files up to 25MB

Examples:
  # Extract transcript from video
  python extract_transcript_whisper.py video.mp4
  
  # Process multiple videos
  python extract_transcript_whisper.py video1.mp4 video2.mp4
        """
    )
    
    parser.add_argument('videos', nargs='+', help='Video file(s) to process')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
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