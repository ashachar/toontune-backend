"""
Person to Character Animation Pipeline
Orchestrates the transformation of a person video into an animated character video
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import from same directory since they're now in pipelines/person_animation/
from nano_banana import NanoBananaGenerator
from runway import RunwayActTwo


class PersonAnimationPipeline:
    """
    Main pipeline for person to character animation
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the pipeline
        
        Args:
            output_dir: Directory to save all outputs (will be set dynamically based on input video)
        """
        self.output_dir = output_dir  # Will be set in process_video
        
        # Initialize the components
        self.nano_banana = NanoBananaGenerator()
        self.act_two = RunwayActTwo()
        
        print(f"Pipeline initialized. Output directory: {self.output_dir}")
    
    def extract_frame(self, video_path: str, timestamp: float, output_path: str) -> str:
        """
        Extract a frame from video at specific timestamp
        
        Args:
            video_path: Path to input video
            timestamp: Time in seconds
            output_path: Path to save the frame
            
        Returns:
            Path to extracted frame
        """
        print(f"Extracting frame at {timestamp}s from {video_path}...")
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract frame: {result.stderr}")
        
        print(f"Frame extracted to: {output_path}")
        return output_path
    
    def extract_video_segment(self, 
                            video_path: str, 
                            start_time: float, 
                            end_time: float,
                            output_path: str) -> str:
        """
        Extract a video segment
        
        Args:
            video_path: Path to input video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save the segment
            
        Returns:
            Path to extracted segment
        """
        duration = end_time - start_time
        print(f"Extracting video segment from {start_time}s to {end_time}s...")
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract video segment: {result.stderr}")
        
        print(f"Video segment extracted to: {output_path}")
        return output_path
    
    def replace_video_segment(self,
                            original_video: str,
                            replacement_video: str,
                            start_time: float,
                            end_time: float,
                            output_path: str) -> str:
        """
        Replace a segment in the original video with the generated video
        
        Args:
            original_video: Path to original video
            replacement_video: Path to replacement video
            start_time: Start time of segment to replace
            end_time: End time of segment to replace
            output_path: Path to save final video
            
        Returns:
            Path to final video
        """
        print(f"Replacing video segment from {start_time}s to {end_time}s...")
        
        # Create temp files for the three segments
        before_segment = os.path.join(self.output_dir, "temp_before.mp4")
        after_segment = os.path.join(self.output_dir, "temp_after.mp4")
        concat_list = os.path.join(self.output_dir, "concat_list.txt")
        
        # Extract segment before replacement
        if start_time > 0:
            cmd_before = [
                'ffmpeg', '-y',
                '-i', original_video,
                '-t', str(start_time),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'copy',
                before_segment
            ]
            subprocess.run(cmd_before, capture_output=True, check=True)
        
        # Get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            original_video
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        total_duration = float(result.stdout.strip())
        
        # Extract segment after replacement
        if end_time < total_duration:
            cmd_after = [
                'ffmpeg', '-y',
                '-ss', str(end_time),
                '-i', original_video,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'copy',
                after_segment
            ]
            subprocess.run(cmd_after, capture_output=True, check=True)
        
        # Process replacement video to remove its audio and use original audio
        replacement_processed = os.path.join(self.output_dir, "temp_replacement.mp4")
        
        # Check if original video has audio
        probe_audio = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=nokey=1:noprint_wrappers=1',
            original_video
        ]
        result = subprocess.run(probe_audio, capture_output=True, text=True)
        has_audio = bool(result.stdout.strip())
        
        if has_audio:
            # Extract audio from the original video segment
            original_audio = os.path.join(self.output_dir, "temp_audio.aac")
            cmd_audio = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', original_video,
                '-t', str(end_time - start_time),
                '-vn',
                '-c:a', 'aac',
                original_audio
            ]
            subprocess.run(cmd_audio, capture_output=True, check=True)
            
            # Combine replacement video with original audio
            cmd_replace = [
                'ffmpeg', '-y',
                '-i', replacement_video,
                '-i', original_audio,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                replacement_processed
            ]
            subprocess.run(cmd_replace, capture_output=True, check=True)
        else:
            # Just copy the replacement video without audio processing
            cmd_replace = [
                'ffmpeg', '-y',
                '-i', replacement_video,
                '-c:v', 'copy',
                '-an',  # Remove any audio from replacement
                replacement_processed
            ]
            subprocess.run(cmd_replace, capture_output=True, check=True)
        
        # Create concat list
        with open(concat_list, 'w') as f:
            if start_time > 0:
                f.write(f"file '{before_segment}'\n")
            f.write(f"file '{replacement_processed}'\n")
            if end_time < total_duration:
                f.write(f"file '{after_segment}'\n")
        
        # Check for accompanying MP3 audio file
        video_dir = os.path.dirname(original_video)
        video_name = os.path.splitext(os.path.basename(original_video))[0]
        
        # Try different audio file naming patterns
        possible_audio_files = [
            os.path.join(video_dir, f"{video_name}_audio.mp3"),
            os.path.join(video_dir, f"{video_name.replace('_input', '_audio')}.mp3"),
            os.path.join(video_dir, f"{video_name.replace('input', 'audio')}.mp3"),
        ]
        
        audio_file = None
        for audio_path in possible_audio_files:
            if os.path.exists(audio_path):
                audio_file = audio_path
                print(f"Found audio file: {audio_file}")
                break
        
        # If no exact match, look for any MP3 in the same directory
        if not audio_file:
            mp3_files = [f for f in os.listdir(video_dir) if f.endswith('.mp3')]
            if mp3_files:
                audio_file = os.path.join(video_dir, mp3_files[0])
                print(f"Using audio file: {audio_file}")
        
        # Concatenate all video segments first
        temp_video_no_audio = os.path.join(self.output_dir, "temp_video_no_audio.mp4")
        cmd_concat = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-an',  # No audio for now
            temp_video_no_audio
        ]
        
        result = subprocess.run(cmd_concat, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to concatenate segments: {result.stderr}")
        
        # If audio file exists, add it to the video
        if audio_file and os.path.exists(audio_file):
            print(f"Adding audio from: {audio_file}")
            
            # Get video duration
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                temp_video_no_audio
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            video_duration = float(result.stdout.strip())
            
            # Add audio to video, trimmed to video length
            cmd_add_audio = [
                'ffmpeg', '-y',
                '-i', temp_video_no_audio,
                '-i', audio_file,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-t', str(video_duration),  # Trim audio to video length
                '-map', '0:v:0',
                '-map', '1:a:0',
                output_path
            ]
            
            result = subprocess.run(cmd_add_audio, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Failed to add audio: {result.stderr}")
                # Fallback: just copy video without audio
                import shutil
                shutil.copy2(temp_video_no_audio, output_path)
            
            # Clean up temp video
            if os.path.exists(temp_video_no_audio):
                os.remove(temp_video_no_audio)
        else:
            # No audio file found, just rename the temp video
            import shutil
            shutil.move(temp_video_no_audio, output_path)
            if not has_audio:
                print("No audio track in original video and no MP3 file found")
        
        # Clean up temp files
        temp_files = [before_segment, after_segment, replacement_processed, concat_list]
        if has_audio:
            temp_files.append(original_audio)
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"Final video saved to: {output_path}")
        return output_path
    
    def process_video(self, 
                     video_path: str,
                     character_description: str = "friendly meerkat") -> str:
        """
        Process the entire video pipeline
        
        Args:
            video_path: Path to input video
            character_description: Description of character to generate
            
        Returns:
            Path to final processed video
        """
        # Set output directory to same folder as input video
        if self.output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(video_path))
        
        print(f"\n{'='*60}")
        print(f"Starting Person Animation Pipeline")
        print(f"Input: {video_path}")
        print(f"Character: {character_description}")
        print(f"Output folder: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract frame at 0.5s
        frame_path = os.path.join(self.output_dir, "frame0.png")
        if os.path.exists(frame_path):
            print(f"Frame already exists, skipping extraction: {frame_path}")
        else:
            self.extract_frame(video_path, 0.5, frame_path)
        
        # Step 2: Extract video segment from 0.5s to 3.5s
        subvideo_path = os.path.join(self.output_dir, "runway_subvideo.mp4")
        if os.path.exists(subvideo_path):
            print(f"Subvideo already exists, skipping extraction: {subvideo_path}")
        else:
            self.extract_video_segment(video_path, 0.5, 3.55, subvideo_path)  # 3.05 seconds to ensure > 3s
        
        # Step 3: Generate character image using Nano Banana
        character_image = os.path.join(self.output_dir, "runway_character.png")
        if os.path.exists(character_image):
            print(f"\nCharacter image already exists, skipping generation: {character_image}")
        else:
            print(f"\nGenerating character sketch for: {character_description}")
            character_image = self.nano_banana.generate_character(
                frame_path,
                character_description,
                self.output_dir
            )
        
        # Step 4: Generate character performance using Act-Two
        generated_video = os.path.join(self.output_dir, "runway_act_two_output.mp4")
        if os.path.exists(generated_video):
            print(f"\nAct-Two video already exists, skipping generation: {generated_video}")
        else:
            print(f"\nGenerating character performance with Act-Two...")
            generated_video = self.act_two.generate_character_performance(
                subvideo_path,
                character_image,
                self.output_dir,
                duration=3  # 3 seconds
            )
        
        # Step 5: Replace segment in original video
        final_video_path = os.path.join(self.output_dir, "final_character_video.mp4")
        if os.path.exists(final_video_path):
            print(f"\nFinal video already exists, skipping replacement: {final_video_path}")
        else:
            self.replace_video_segment(
                video_path,
                generated_video,
                0.5,
                3.55,
                final_video_path
            )
        
        # Step 6: Open the final video
        print(f"\nOpening final video...")
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", final_video_path])
        elif sys.platform == "win32":  # Windows
            subprocess.run(["start", "", final_video_path], shell=True)
        else:  # Linux
            subprocess.run(["xdg-open", final_video_path])
        
        print(f"\n{'='*60}")
        print(f"Pipeline Complete!")
        print(f"Final video: {final_video_path}")
        print(f"All artifacts saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return final_video_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Person to Character Animation Pipeline")
    parser.add_argument(
        "video",
        nargs="?",
        default="nano_video_input.mp4",
        help="Path to input video (default: nano_video_input.mp4)"
    )
    parser.add_argument(
        "--character",
        default="friendly meerkat",
        help="Character description (default: friendly meerkat)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for all artifacts (default: same as input video)"
    )
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.video):
        print(f"Error: Input video not found: {args.video}")
        sys.exit(1)
    
    # Run the pipeline
    pipeline = PersonAnimationPipeline(output_dir=args.output_dir)
    
    try:
        final_video = pipeline.process_video(args.video, args.character)
        print(f"Success! Final video: {final_video}")
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()