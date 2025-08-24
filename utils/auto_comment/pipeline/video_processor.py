"""Video processing with speed adjustments."""

import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from utils.auto_comment.pipeline.config import DEFAULT_FPS, DEFAULT_VIDEO_QUALITY, DEFAULT_AUDIO_QUALITY


class VideoProcessor:
    """Handles video speed adjustments and final video creation."""
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._get_video_info()
    
    def _get_video_info(self):
        """Get video information."""
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", str(self.video_path)
        ], capture_output=True, text=True)
        
        video_info = json.loads(result.stdout)
        self.width = video_info["streams"][0]["width"]
        self.height = video_info["streams"][0]["height"]
        fps_parts = video_info["streams"][0]["r_frame_rate"].split("/")
        self.fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else DEFAULT_FPS
        
        # Get duration
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.video_path)
        ], capture_output=True, text=True)
        self.duration = float(result.stdout.strip())
    
    def create_video_with_speed_adjustments(
        self, 
        snarks_with_pausing: List[Dict], 
        mixed_audio_path: Path, 
        output_path: Path
    ):
        """Create video with speed adjustments during remarks."""
        print("  🎬 Applying speed adjustments...")
        
        # Separate remarks by whether they need slowdown
        all_remarks = sorted(snarks_with_pausing, key=lambda x: x["snark"].time)
        
        # Check if any need slowdown
        if not any(s["needs_slowdown"] for s in all_remarks):
            # No slowdowns needed, just remux
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-i", str(mixed_audio_path),
                "-map", "0:v", "-map", "1:a",
                "-c:v", DEFAULT_VIDEO_QUALITY, "-preset", "fast", "-crf", "23",
                "-c:a", DEFAULT_AUDIO_QUALITY, "-b:a", "256k",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ], capture_output=True)
            return
        
        # Process all remarks for reporting
        self._print_speed_report(all_remarks)
        
        # Build video with speed adjustments
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create debug directory for segments
        debug_dir = Path("/tmp/auto_comment_segments")
        debug_dir.mkdir(exist_ok=True)
        print(f"  💾 Saving segments to {debug_dir} for debugging")
        
        try:
            segments = self._create_video_segments(all_remarks, temp_dir)
            
            if segments:
                video_adjusted = self._concatenate_segments(segments, temp_dir)
                if video_adjusted:
                    self._combine_with_audio(video_adjusted, mixed_audio_path, output_path)
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _print_speed_report(self, all_remarks: List[Dict]):
        """Print speed adjustment report."""
        print(f"\n  📊 Speed Adjustment Report:")
        print(f"  {'='*60}")
        
        for item in all_remarks:
            snark = item["snark"]
            gap_duration = item["gap_duration"]
            remark_duration = item["duration"]
            speed_factor = item["speed_factor"]
            
            print(f"  • Remark '{snark.text}':")
            print(f"    - Gap: {gap_duration:.2f}s, Remark: {remark_duration:.2f}s")
            
            if speed_factor < 1.0:
                speed_pct = speed_factor * 100
                print(f"    - Speed: {speed_pct:.1f}% (slowed by {100-speed_pct:.1f}%)")
                print(f"    - Video at {snark.time:.2f}s slowed to stretch {gap_duration:.2f}s → {remark_duration:.2f}s")
            else:
                print(f"    - Speed: 100% (normal - remark fits in gap)")
        
        print(f"  {'='*60}\n")
    
    def _create_video_segments(self, all_remarks: List[Dict], temp_dir: Path) -> List[str]:
        """Create video segments with speed adjustments."""
        segments = []
        current_pos = 0
        segment_idx = 0
        output_time = 0  # Track output video position
        
        print(f"\n  🎆 Building video segments...")
        print(f"  📹 Video duration: {self.duration:.2f}s")
        print(f"  {'='*60}")
        
        for i, item in enumerate(all_remarks):
            snark = item["snark"]
            gap_start = snark.time
            gap_duration = item["gap_duration"]
            speed_factor = item["speed_factor"]
            remark_duration = item["duration"]
            
            # Add normal speed segment before the gap
            if current_pos < gap_start:
                seg_duration = gap_start - current_pos
                print(f"  📦 Segment {segment_idx:02d} [NORMAL]:")
                print(f"     Source: [{current_pos:.3f}s → {gap_start:.3f}s] = {seg_duration:.3f}s")
                print(f"     Output: [{output_time:.3f}s → {output_time + seg_duration:.3f}s]")
                
                from utils.auto_comment.pipeline.video_utils import create_normal_segment
                segment_file = create_normal_segment(
                    self.video_path, current_pos, gap_start, segment_idx, temp_dir
                )
                if segment_file:
                    segments.append(segment_file)
                    # Copy to debug directory
                    import shutil
                    debug_file = Path(f"/tmp/auto_comment_segments/seg_{segment_idx:03d}_normal.mp4")
                    shutil.copy2(segment_file, debug_file)
                    print(f"        💾 Saved to {debug_file}")
                    segment_idx += 1
                    output_time += seg_duration
            
            # Add segment for the gap (slowed or normal)
            if speed_factor < 1.0:
                output_duration = gap_duration / speed_factor
                print(f"  📦 Segment {segment_idx:02d} [SLOWED]:")
                print(f"     Source: [{gap_start:.3f}s for {gap_duration:.3f}s]")
                print(f"     Output: {output_duration:.3f}s @ {speed_factor*100:.1f}% speed")
                print(f"     Output: [{output_time:.3f}s → {output_time + output_duration:.3f}s]")
                print(f"     Remark: '{snark.text}' ({remark_duration:.3f}s)")
                output_time += output_duration
            else:
                print(f"  📦 Segment {segment_idx:02d} [GAP]:")
                print(f"     Source: [{gap_start:.3f}s for {gap_duration:.3f}s]")
                print(f"     Output: [{output_time:.3f}s → {output_time + gap_duration:.3f}s]")
                print(f"     Remark: '{snark.text}' ({remark_duration:.3f}s)")
                output_time += gap_duration
            
            from utils.auto_comment.pipeline.video_utils import create_gap_segment
            gap_segment = create_gap_segment(
                self.video_path, gap_start, gap_duration, speed_factor,
                segment_idx, temp_dir, self.fps
            )
            if gap_segment:
                segments.append(gap_segment)
                # Copy to debug directory
                import shutil
                suffix = "slowed" if speed_factor < 1.0 else "gap"
                debug_file = Path(f"/tmp/auto_comment_segments/seg_{segment_idx:03d}_{suffix}.mp4")
                shutil.copy2(gap_segment, debug_file)
                print(f"        💾 Saved to {debug_file}")
                segment_idx += 1
            
            current_pos = gap_start + gap_duration
            print(f"     Next source position: {current_pos:.3f}s")
            print(f"  {'-'*60}")
        
        # Add remaining video at normal speed
        if current_pos < self.duration:
            from utils.auto_comment.pipeline.video_utils import create_normal_segment
            final_segment = create_normal_segment(
                self.video_path, current_pos, self.duration, segment_idx, temp_dir
            )
            if final_segment:
                segments.append(final_segment)
        
        return segments
    
    
    def _concatenate_segments(self, segments: List[str], temp_dir: Path) -> Optional[Path]:
        """Concatenate video segments."""
        from utils.auto_comment.pipeline.video_utils import concatenate_segments
        return concatenate_segments(segments, temp_dir)
    
    def _combine_with_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        """Combine video with audio."""
        from utils.auto_comment.pipeline.video_utils import combine_with_audio
        return combine_with_audio(video_path, audio_path, output_path)