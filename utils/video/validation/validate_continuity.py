#!/usr/bin/env python3
"""
Validate video continuity by detecting duplicate frames.
Excludes uniform frames (black/white screens) based on pixel variance.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import hashlib
from collections import defaultdict


class VideoContinuityValidator:
    """Detect duplicate frames in video, excluding uniform screens."""
    
    def __init__(self, video_path: str, variance_threshold: float = 100.0, 
                 sample_rate: int = 1, resize_width: int = 320):
        """
        Initialize validator.
        
        Args:
            video_path: Path to video file
            variance_threshold: Minimum variance to consider frame non-uniform
            sample_rate: Check every Nth frame (1 = every frame, 5 = every 5th)
            resize_width: Resize frames to this width for faster processing
        """
        self.video_path = Path(video_path)
        self.variance_threshold = variance_threshold
        self.sample_rate = sample_rate
        self.resize_width = resize_width
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"🎬 Video: {self.video_path.name}")
        print(f"📊 Properties: {self.width}x{self.height} @ {self.fps:.2f} fps")
        print(f"📁 Total frames: {self.total_frames}")
        print(f"🔍 Checking every {self.sample_rate} frame(s)")
        print(f"📏 Variance threshold: {self.variance_threshold}")
    
    def compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute hash of frame for exact comparison."""
        # Resize for faster processing
        if self.resize_width and frame.shape[1] > self.resize_width:
            aspect = frame.shape[0] / frame.shape[1]
            new_height = int(self.resize_width * aspect)
            frame = cv2.resize(frame, (self.resize_width, new_height))
        
        # Convert to bytes and hash
        frame_bytes = frame.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()
    
    def is_uniform_frame(self, frame: np.ndarray) -> bool:
        """Check if frame is uniform (low variance, like black/white screen)."""
        # Convert to grayscale for variance calculation
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate variance
        variance = np.var(gray)
        
        # Also check if it's mostly one color
        unique_colors = len(np.unique(gray))
        
        # Uniform if low variance OR very few unique colors
        is_uniform = variance < self.variance_threshold or unique_colors < 10
        
        return is_uniform, variance
    
    def find_duplicate_frames(self) -> Dict:
        """Find all duplicate frames in the video."""
        print("\n" + "=" * 70)
        print("🔍 SCANNING FOR DUPLICATE FRAMES...")
        print("=" * 70)
        
        frame_hashes = {}  # hash -> list of (frame_num, timestamp)
        uniform_frames = []
        duplicates = []
        
        frame_count = 0
        processed = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Sample frames based on rate
            if frame_count % self.sample_rate == 0:
                timestamp = frame_count / self.fps
                
                # Check if uniform
                is_uniform, variance = self.is_uniform_frame(frame)
                
                if is_uniform:
                    uniform_frames.append({
                        'frame': frame_count,
                        'time': timestamp,
                        'variance': variance
                    })
                else:
                    # Compute hash for non-uniform frames
                    frame_hash = self.compute_frame_hash(frame)
                    
                    if frame_hash in frame_hashes:
                        # Found duplicate!
                        original = frame_hashes[frame_hash][0]
                        duplicates.append({
                            'frame': frame_count,
                            'time': timestamp,
                            'original_frame': original['frame'],
                            'original_time': original['time'],
                            'time_diff': timestamp - original['time'],
                            'hash': frame_hash
                        })
                        frame_hashes[frame_hash].append({'frame': frame_count, 'time': timestamp})
                    else:
                        frame_hashes[frame_hash] = [{'frame': frame_count, 'time': timestamp}]
                
                processed += 1
                
                # Progress update
                if processed % 100 == 0:
                    progress = (frame_count / self.total_frames) * 100
                    print(f"   Progress: {progress:.1f}% - Frame {frame_count}/{self.total_frames}")
            
            frame_count += 1
        
        self.cap.release()
        
        return {
            'total_frames': frame_count,
            'frames_checked': processed,
            'unique_frames': len(frame_hashes),
            'uniform_frames': uniform_frames,
            'duplicates': duplicates,
            'frame_hashes': frame_hashes
        }
    
    def print_report(self, results: Dict):
        """Print detailed report of findings."""
        print("\n" + "=" * 70)
        print("📊 CONTINUITY VALIDATION REPORT")
        print("=" * 70)
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total frames: {results['total_frames']}")
        print(f"   Frames checked: {results['frames_checked']}")
        print(f"   Unique content frames: {results['unique_frames']}")
        print(f"   Uniform frames (skipped): {len(results['uniform_frames'])}")
        print(f"   Duplicate frames found: {len(results['duplicates'])}")
        
        # Report uniform frames
        if results['uniform_frames']:
            print(f"\n⬜ UNIFORM FRAMES (first 5):")
            for uf in results['uniform_frames'][:5]:
                print(f"   Frame {uf['frame']} at {uf['time']:.2f}s (variance: {uf['variance']:.2f})")
        
        # Report duplicates
        if results['duplicates']:
            print(f"\n❌ DUPLICATE FRAMES DETECTED:")
            print("-" * 70)
            
            # Group duplicates by time difference
            near_duplicates = []  # < 1 second apart
            far_duplicates = []   # > 1 second apart
            
            for dup in results['duplicates']:
                if dup['time_diff'] < 1.0:
                    near_duplicates.append(dup)
                else:
                    far_duplicates.append(dup)
            
            if far_duplicates:
                print(f"\n🚨 PROBLEMATIC DUPLICATES (>1s apart):")
                for i, dup in enumerate(far_duplicates[:20], 1):  # Show max 20
                    print(f"\n   {i}. Frame {dup['frame']} at {dup['time']:.2f}s")
                    print(f"      Duplicate of frame {dup['original_frame']} at {dup['original_time']:.2f}s")
                    print(f"      Time difference: {dup['time_diff']:.2f}s")
                    print(f"      ⚠️ Content repeats after {dup['time_diff']:.1f} seconds!")
            
            if near_duplicates:
                print(f"\n⚡ NEAR DUPLICATES (<1s apart, possibly intentional):")
                print(f"   Found {len(near_duplicates)} frames")
                if len(near_duplicates) > 0:
                    example = near_duplicates[0]
                    print(f"   Example: Frame {example['frame']} duplicates {example['original_frame']} ({example['time_diff']:.3f}s apart)")
        else:
            print("\n✅ NO DUPLICATE FRAMES FOUND!")
            print("   The video has perfect continuity with no repeated content.")
        
        # Find frames that appear multiple times
        multi_duplicates = defaultdict(list)
        for hash_val, frames in results['frame_hashes'].items():
            if len(frames) > 2:  # Appears 3+ times
                multi_duplicates[hash_val] = frames
        
        if multi_duplicates:
            print(f"\n🔄 FRAMES APPEARING 3+ TIMES:")
            for hash_val, frames in list(multi_duplicates.items())[:5]:
                print(f"   Content appears {len(frames)} times:")
                for f in frames[:5]:
                    print(f"      - Frame {f['frame']} at {f['time']:.2f}s")
        
        print("\n" + "=" * 70)
        
        # Final verdict
        if results['duplicates']:
            problematic = [d for d in results['duplicates'] if d['time_diff'] > 1.0]
            if problematic:
                print("❌ VALIDATION FAILED: Video contains repeated content!")
                print(f"   Found {len(problematic)} duplicate frames >1s apart")
                print("   This indicates the video jumps back to previous content.")
            else:
                print("⚠️ VALIDATION WARNING: Minor duplicates found")
                print("   All duplicates are <1s apart (might be intentional pauses)")
        else:
            print("✅ VALIDATION PASSED: Video has perfect continuity!")
    
    def validate(self) -> bool:
        """Run validation and return True if no issues found."""
        results = self.find_duplicate_frames()
        self.print_report(results)
        
        # Return True if no far duplicates found
        far_duplicates = [d for d in results['duplicates'] if d['time_diff'] > 1.0]
        return len(far_duplicates) == 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_continuity.py <video_path> [options]")
        print("\nOptions:")
        print("  --variance-threshold N  Minimum variance for non-uniform frames (default: 100)")
        print("  --sample-rate N         Check every Nth frame (default: 1)")
        print("  --resize-width N        Resize frames to width for faster processing (default: 320)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Parse optional arguments
    variance_threshold = 100.0
    sample_rate = 1
    resize_width = 320
    
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--variance-threshold" and i + 1 < len(sys.argv):
            variance_threshold = float(sys.argv[i + 1])
        elif sys.argv[i] == "--sample-rate" and i + 1 < len(sys.argv):
            sample_rate = int(sys.argv[i + 1])
        elif sys.argv[i] == "--resize-width" and i + 1 < len(sys.argv):
            resize_width = int(sys.argv[i + 1])
    
    # Run validation
    validator = VideoContinuityValidator(
        video_path, 
        variance_threshold=variance_threshold,
        sample_rate=sample_rate,
        resize_width=resize_width
    )
    
    is_valid = validator.validate()
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()