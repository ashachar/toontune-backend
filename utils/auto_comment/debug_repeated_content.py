#!/usr/bin/env python3
"""
Debug repeated content in videos by analyzing segments and finding duplications.
"""

import cv2
import numpy as np
import subprocess
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import sys


class RepeatedContentDebugger:
    """Debug tool to find and visualize repeated content in videos."""
    
    def __init__(self, video_path: str, problem_ranges: List[Tuple[float, float]] = None):
        """
        Initialize debugger.
        
        Args:
            video_path: Path to video file
            problem_ranges: List of (start, end) tuples for problem areas
        """
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.output_dir = Path(f"debug_output_{self.video_name}")
        self.output_dir.mkdir(exist_ok=True)
        
        # Default problem ranges if not specified
        self.problem_ranges = problem_ranges or [
            (23.0, 30.0),  # "no longer a question of" repetition
            (45.0, 52.0),  # Frame duplication area
        ]
        
        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🔍 Debugging: {self.video_path.name}")
        print(f"📊 Video: {self.fps:.2f} fps, {self.total_frames} frames")
        print(f"🎯 Problem ranges: {self.problem_ranges}")
    
    def extract_audio_segments(self, start: float, end: float) -> Path:
        """Extract audio segment for analysis."""
        output_path = self.output_dir / f"audio_{start:.1f}_{end:.1f}.wav"
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start),
            "-i", str(self.video_path),
            "-t", str(end - start),
            "-vn", "-acodec", "pcm_s16le",
            str(output_path)
        ]
        subprocess.run(cmd, check=True)
        return output_path
    
    def extract_frames_in_range(self, start: float, end: float, 
                                interval: float = 0.2) -> Dict[float, np.ndarray]:
        """Extract frames at intervals in time range."""
        frames = {}
        
        for t in np.arange(start, end, interval):
            frame_num = int(t * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if ret:
                # Resize for faster processing
                small = cv2.resize(frame, (320, 180))
                frames[t] = small
        
        return frames
    
    def find_duplicate_frames(self, frames: Dict[float, np.ndarray]) -> List[Dict]:
        """Find duplicate frames using perceptual hashing."""
        frame_hashes = {}
        duplicates = []
        
        for time, frame in frames.items():
            # Create hash
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_hash = hashlib.md5(gray.tobytes()).hexdigest()
            
            if frame_hash in frame_hashes:
                duplicates.append({
                    "time": time,
                    "duplicate_of": frame_hashes[frame_hash],
                    "hash": frame_hash
                })
            else:
                frame_hashes[frame_hash] = time
        
        return duplicates
    
    def create_visual_timeline(self, start: float, end: float) -> Path:
        """Create visual timeline showing frames."""
        frames = self.extract_frames_in_range(start, end, interval=0.5)
        
        # Create montage
        cols = 10
        rows = (len(frames) + cols - 1) // cols
        
        frame_list = list(frames.values())
        times = list(frames.keys())
        
        # Pad if needed
        while len(frame_list) < rows * cols:
            frame_list.append(np.zeros_like(frame_list[0]))
            times.append(0)
        
        # Create grid
        grid_rows = []
        for r in range(rows):
            row_frames = frame_list[r*cols:(r+1)*cols]
            row = np.hstack(row_frames)
            grid_rows.append(row)
        
        grid = np.vstack(grid_rows)
        
        # Add timestamps
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, t in enumerate(times[:len(frames)]):
            y = (i // cols) * 180 + 20
            x = (i % cols) * 320 + 10
            cv2.putText(grid, f"{t:.1f}s", (x, y), font, 0.5, (0, 255, 0), 1)
        
        # Save
        output_path = self.output_dir / f"timeline_{start:.1f}_{end:.1f}.jpg"
        cv2.imwrite(str(output_path), grid)
        
        return output_path
    
    def analyze_segment_files(self, work_dir: Path = None):
        """Analyze segment files if available."""
        if not work_dir:
            # Try to find most recent work directory
            tmp_dirs = list(Path("/tmp").glob(f"precise_gap_pipeline_{self.video_name}_*"))
            if tmp_dirs:
                work_dir = sorted(tmp_dirs)[-1]
                print(f"📁 Found work directory: {work_dir}")
        
        if not work_dir or not work_dir.exists():
            print("❌ No work directory found")
            return
        
        segments_dir = work_dir / "segments"
        segment_files = sorted(segments_dir.glob("segment_*.mp4"))
        
        print(f"\n📊 Analyzing {len(segment_files)} segments:")
        
        segment_info = []
        for seg_file in segment_files:
            # Get duration
            cmd = ["ffprobe", "-v", "error", "-show_entries", 
                   "format=duration", "-of", "json", str(seg_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            duration = float(data["format"]["duration"])
            
            # Extract first frame
            cap = cv2.VideoCapture(str(seg_file))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Create hash
                small = cv2.resize(frame, (160, 90))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                frame_hash = hashlib.md5(gray.tobytes()).hexdigest()[:8]
            else:
                frame_hash = "error"
            
            segment_info.append({
                "file": seg_file.name,
                "duration": duration,
                "hash": frame_hash
            })
        
        # Check for duplicate hashes
        hash_counts = {}
        for info in segment_info:
            h = info["hash"]
            if h not in hash_counts:
                hash_counts[h] = []
            hash_counts[h].append(info["file"])
        
        # Report
        for info in segment_info[:20]:  # Show first 20
            print(f"  {info['file']:25s} {info['duration']:6.3f}s  hash:{info['hash']}")
        
        print("\n🔍 Duplicate content detection:")
        for h, files in hash_counts.items():
            if len(files) > 1:
                print(f"  ⚠️ Hash {h} appears in: {', '.join(files)}")
    
    def transcribe_segment(self, start: float, end: float) -> str:
        """Transcribe audio in segment using Whisper."""
        try:
            import whisper
            
            # Extract audio
            audio_path = self.extract_audio_segments(start, end)
            
            # Transcribe
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path))
            
            return result["text"]
        except ImportError:
            return "Whisper not available"
    
    def run_diagnosis(self):
        """Run complete diagnosis."""
        print("\n" + "="*70)
        print("🔬 STARTING REPEATED CONTENT DIAGNOSIS")
        print("="*70)
        
        for start, end in self.problem_ranges:
            print(f"\n📍 Analyzing range {start:.1f}s - {end:.1f}s")
            print("-" * 50)
            
            # Extract frames
            frames = self.extract_frames_in_range(start, end)
            
            # Find duplicates
            duplicates = self.find_duplicate_frames(frames)
            if duplicates:
                print(f"🔴 Found {len(duplicates)} duplicate frames:")
                for dup in duplicates[:5]:
                    print(f"   {dup['time']:.2f}s duplicates {dup['duplicate_of']:.2f}s")
            
            # Create visual timeline
            timeline_path = self.create_visual_timeline(start, end)
            print(f"📸 Visual timeline saved: {timeline_path.name}")
            
            # Transcribe audio
            transcript = self.transcribe_segment(start, end)
            print(f"📝 Transcript: {transcript[:100]}...")
            
            # Check for repeated words
            words = transcript.lower().split()
            repeated = []
            for i in range(len(words) - 3):
                phrase = " ".join(words[i:i+3])
                if phrase in " ".join(words[i+4:]):
                    repeated.append(phrase)
            
            if repeated:
                print(f"🔁 Repeated phrases: {set(repeated)}")
        
        # Analyze segment files
        self.analyze_segment_files()
        
        print("\n" + "="*70)
        print("📊 DIAGNOSIS COMPLETE")
        print(f"📁 Results saved in: {self.output_dir}")
        print("="*70)
        
        self.cap.release()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python debug_repeated_content.py <video_path> [start1,end1] [start2,end2]")
        print("\nExample:")
        print("  python debug_repeated_content.py video.mp4 23,30 45,52")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Parse problem ranges if provided
    problem_ranges = []
    for arg in sys.argv[2:]:
        if "," in arg:
            start, end = map(float, arg.split(","))
            problem_ranges.append((start, end))
    
    # Run debugger
    debugger = RepeatedContentDebugger(video_path, problem_ranges)
    debugger.run_diagnosis()


if __name__ == "__main__":
    main()