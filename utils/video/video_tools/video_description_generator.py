#!/usr/bin/env python3
"""
Video Description Generator with Gemini 2.5 Pro

This script analyzes videos and generates detailed editing plans in JSON format.
It extracts effects documentation, builds prompts, and sends them to Gemini 2.5 Pro.

Usage:
    python video_description_generator.py <video_path> [--output-dir <dir>]
"""

import sys
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import cv2
import yaml
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

class VideoDescriptionGenerator:
    """Generates detailed video descriptions and editing plans using Gemini 2.5 Pro"""
    
    def __init__(self):
        """Initialize the generator with Gemini API"""
        self.setup_gemini()
        self.effects_documentation = None
        self.prompts_file = Path(__file__).parent.parent / "prompts.yaml"
        self.transcript_data = None
        
    def setup_gemini(self):
        """Setup Gemini 2.5 Pro API"""
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            print("❌ No GEMINI_API_KEY found in .env")
            sys.exit(1)
        
        # Use Gemini 2.5 Pro (the strong model)
        genai.configure(api_key=gemini_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini 2.5 Pro configured")
    
    def load_transcript_for_scene(self, video_path: Path) -> Dict[str, Any]:
        """Load transcript data relevant to the video/scene"""
        
        # Check if this is a scene file from do_re_mi_with_music
        if "do_re_mi_with_music" in str(video_path) and "scenes" in str(video_path):
            # Extract scene number from filename (e.g., scene_001.mp4 -> 1)
            scene_name = video_path.stem  # e.g., "scene_001"
            scene_num = int(scene_name.split('_')[1]) if '_' in scene_name else None
            
            if scene_num:
                # Load the full transcript
                transcript_dir = video_path.parent.parent / "soundtrack"
                words_file = transcript_dir / "do_re_mi_with_music_transcript_words.json"
                index_file = video_path.parent / "scene_index_transcript.txt"
                
                if words_file.exists():
                    with open(words_file, 'r') as f:
                        all_words = json.load(f)['words']
                    
                    # Parse scene timings from index file
                    scene_timings = {}
                    if index_file.exists():
                        with open(index_file, 'r') as f:
                            content = f.read()
                            
                        # Extract timing for this scene
                        import re
                        pattern = rf"Scene {scene_num:03d}:.*?Time: ([\d.]+)s - ([\d.]+)s"
                        match = re.search(pattern, content, re.DOTALL)
                        
                        if match:
                            start_time = float(match.group(1))
                            end_time = float(match.group(2))
                            
                            # Filter words for this scene's time range
                            scene_words = []
                            for word in all_words:
                                word_start = word['start_ms'] / 1000.0  # Convert to seconds
                                word_end = word['end_ms'] / 1000.0
                                
                                if start_time <= word_start < end_time:
                                    # Adjust timing relative to scene start
                                    scene_words.append({
                                        'word': word['word'],
                                        'start_seconds': f"{word_start - start_time:.3f}",
                                        'end_seconds': f"{word_end - start_time:.3f}",
                                        'duration_ms': word['end_ms'] - word['start_ms']
                                    })
                            
                            return {
                                'scene_number': scene_num,
                                'total_words': len(scene_words),
                                'duration_seconds': end_time - start_time,
                                'words': scene_words
                            }
        
        return None
    
    def extract_effects_documentation(self) -> str:
        """Extract effects documentation using the existing script"""
        print("📚 Extracting effects documentation...")
        
        # Import the existing effects extraction script
        from utils.get_existing_effects import get_all_effects
        
        documentation = get_all_effects()
        self.effects_documentation = documentation
        
        # Count effects for verification
        lines = documentation.split('\n')
        for line in lines:
            if "Total Editing Effects:" in line or "Total Animation Classes:" in line:
                print(f"   {line}")
        
        return documentation
    
    def build_prompt(self, video_path: Path = None) -> str:
        """Build the complete prompt with injected effects documentation and transcript"""
        
        if not self.effects_documentation:
            self.effects_documentation = self.extract_effects_documentation()
        
        # Load transcript if available
        transcript_section = ""
        if video_path:
            self.transcript_data = self.load_transcript_for_scene(video_path)
            if self.transcript_data:
                transcript_section = f"""

TRANSCRIPT TIMING DATA:
The following JSON contains word-by-word timing for the audio in this video. Use this to place text naturally in the scene.
{json.dumps(self.transcript_data, indent=2)}

"""
        
        prompt_template = """You are an expert video editor and senior director. Your task is to analyze the provided video and produce a detailed editing plan in a specific JSON format. Your analysis must be based only on the visual information in the video, without making any external assumptions about the plot, characters, or film title.

CRITICAL INSTRUCTION FOR TEXT OVERLAYS: When suggesting text overlay placements, you MUST provide EXACT PIXEL COORDINATES for positioning. Specify the top-left corner (x, y) and bottom-right corner (x, y) in pixels. Consider the actual frame dimensions and available space. Many frames have subjects that fill most of the screen, leaving very limited safe areas for text. Be realistic about coordinates - ensure text doesn't occlude faces, bodies, or key action.

The output MUST be a single, valid JSON object structured as follows:

characters_in_video: A top-level list of objects. Identify every distinct character or logical group of characters from the entire video. For each, create an object with:

name: A descriptive, generic name (e.g., "Adult Woman", "Younger Boy", "Children Group").

description: A brief visual description.
These names must be used consistently throughout the rest of the JSON.

video_description: A top-level string containing a concise, high-level summary of the video's narrative and visual progression.

sound_effects: A top-level list of suggested sound effects to enhance the video. Each sound effect object must contain exactly:

sound: A simple, clear sound name (e.g., "whoosh", "bell", "footsteps", "splash", "chime", "pop", "swoosh", "ding").

timestamp: The precise string timestamp when the sound should play (e.g., "12.500").

scenes: A top-level list where the video is broken down into distinct scenes based on changes in camera angle, subject, or action. Each scene object in the list must contain:

start_seconds and end_seconds: Highly accurate timestamps for the scene's duration, formatted as a string with microseconds (e.g., "43.251").

scene_description: A nested object containing the following fields:

characters: A list of the character name(s) present in the scene.

most_prominent_figure: The main visual focus of the scene.

character_actions: An object detailing what each character or group is doing.

background: A description of the environment.

camera_angle: The specific cinematographic term for the shot (e.g., "Long Shot", "Medium Close-Up", "High-Angle Shot").

camera_movement: (Only if movement exists) A nested object with direction and pixels_per_second (as an integer speed). Omit this field entirely if the camera is static.

suggested_effects: This is the most critical part. As a senior director, suggest one or more effects to enhance the scene. This must be a list of objects. Each object represents a specific, callable function from the reference documentation below and must contain:

effect_fn_name: The exact string name of the function to be called (e.g., "apply_smooth_zoom").

effect_timestamp: The precise string timestamp (e.g., "27.000") within the scene when the effect should trigger.

effect_fn_params: A nested object containing the exact parameter names and appropriate values for the chosen function.

top_left_pixels: An object with x and y coordinates in pixels for the top-left corner of the effect/overlay (e.g., {"x": 50, "y": 100}).

bottom_right_pixels: An object with x and y coordinates in pixels for the bottom-right corner of the effect/overlay (e.g., {"x": 250, "y": 250}).

text_overlays: (REQUIRED if transcript data is provided) A list of text placement suggestions for lyrics/dialogue. Each overlay object must contain:

word: The text to display (from the transcript).

start_seconds: When the text appears (matching transcript timing).

end_seconds: When the text disappears.

top_left_pixels: CRITICAL - An object with x and y pixel coordinates for the top-left corner of the text overlay. Must be realistic based on frame dimensions and avoid occluding subjects. Example: {"x": 10, "y": 10} for top-left corner placement.

bottom_right_pixels: CRITICAL - An object with x and y pixel coordinates for the bottom-right corner of the text overlay. Together with top_left_pixels, this defines the text bounding box. Example: {"x": 90, "y": 40} for an 80x30 pixel text box.

IMPORTANT COORDINATE GUIDELINES:
  - Consider actual frame dimensions (commonly 1920x1080, 1280x720, or downsampled like 256x114)
  - NEVER place text over faces or main subjects - check x,y coordinates carefully
  - For a 256x114 frame with subject at center, safe areas might be: top-left (0-50, 0-20), top-right (200-256, 0-20), bottom corners
  - Ensure bottom_right is greater than top_left in both x and y
  - Keep text boxes reasonably sized (e.g., 80x30 pixels for single words in small frames)

text_effect: The animation effect for the text itself, chosen from the text effects in the documentation (e.g., "Typewriter", "WordBuildup", "SplitText").

text_effect_params: Parameters for the text effect.

interaction_style: How the text interacts with the scene (e.g., "text_behind_object", "floating_with_character", "anchored_to_background", "emerging_from_water").

Here is a short example of the expected output format:

{
  "characters_in_video": [
    {
      "name": "Man in Red Hat",
      "description": "A man wearing a red baseball cap and a blue jacket."
    },
    {
      "name": "Dog",
      "description": "A golden retriever."
    }
  ],
  "video_description": "A man plays fetch with his dog in a park on a sunny day.",
  "sound_effects": [
    {
      "sound": "whoosh",
      "timestamp": "2.800"
    },
    {
      "sound": "bark",
      "timestamp": "5.200"
    },
    {
      "sound": "footsteps",
      "timestamp": "6.100"
    }
  ],
  "scenes": [
    {
      "start_seconds": "0.000",
      "end_seconds": "5.120",
      "scene_description": {
        "characters": [
          "Man in Red Hat"
        ],
        "most_prominent_figure": "Man in Red Hat",
        "character_actions": {
          "Man in Red Hat": "He winds up and throws a yellow tennis ball off-screen to the right."
        },
        "background": "A green park with trees in the distance.",
        "camera_angle": "Medium Shot",
        "suggested_effects": [
          {
            "effect_fn_name": "apply_smooth_zoom",
            "effect_timestamp": "2.500",
            "effect_fn_params": {
              "zoom_factor": 1.2,
              "zoom_type": "in",
              "easing": "ease_out"
            },
            "top_left_pixels": {
              "x": 0,
              "y": 0
            },
            "bottom_right_pixels": {
              "x": 1920,
              "y": 1080
            }
          }
        ],
        "text_overlays": [
          {
            "word": "fetch",
            "start_seconds": "1.200",
            "end_seconds": "2.000",
            "top_left_pixels": {
              "x": 850,
              "y": 400
            },
            "bottom_right_pixels": {
              "x": 970,
              "y": 440
            },
            "text_effect": "WordBuildup",
            "text_effect_params": {
              "buildup_mode": "fade",
              "word_delay": 5
            },
            "interaction_style": "floating_with_character"
          }
        ]
      }
    },
    {
      "start_seconds": "5.121",
      "end_seconds": "10.500",
      "scene_description": {
        "characters": [
          "Dog"
        ],
        "most_prominent_figure": "Dog",
        "character_actions": {
          "Dog": "The dog runs from left to right across the field, chasing the ball."
        },
        "background": "A large, open grassy field.",
        "camera_angle": "Long Shot",
        "camera_movement": {
          "direction": "Pan Right",
          "pixels_per_second": 250
        },
        "suggested_effects": [
          {
            "effect_fn_name": "apply_speed_ramp",
            "effect_timestamp": "7.000",
            "effect_fn_params": {
              "speed_points": [
                [
                  1.8,
                  1.0
                ],
                [
                  2.0,
                  0.5
                ],
                [
                  2.5,
                  1.5
                ]
              ],
              "interpolation": "smooth"
            }
          }
        ]
      }
    }
  ]
}

IMPORTANT INSTRUCTIONS FOR TEXT PLACEMENT WITH PIXEL COORDINATES:
- Each word from the transcript must be positioned with EXACT pixel coordinates
- Provide top_left_pixels (x, y) and bottom_right_pixels (x, y) for every text overlay
- Consider the actual frame dimensions when setting coordinates
- For 1920x1080 frames: safe text areas might be (10,10)-(200,50) for top-left, (1720,10)-(1910,50) for top-right
- For 256x114 downsampled frames: safe areas might be (5,5)-(85,35) for top-left, (170,5)-(250,35) for top-right
- VERIFY coordinates don't place text over faces or main subjects
- The box defined by coordinates should be appropriately sized for the word (80x30 for short words, 120x40 for longer)
- Use the text animation classes (Typewriter, WordBuildup, SplitText) for text appearance
- Consider using apply_text_behind_subject or apply_video_in_text for creative text integration

IMPORTANT INSTRUCTIONS FOR COORDINATE CALCULATIONS:
- Always ensure bottom_right.x > top_left.x and bottom_right.y > top_left.y
- Width = bottom_right.x - top_left.x (should be 80-200 pixels for single words)
- Height = bottom_right.y - top_left.y (should be 30-60 pixels for text)
- For effects covering full frame: top_left=(0,0), bottom_right=(frame_width, frame_height)
- Consider visual hierarchy - important words should have larger bounding boxes
- Leave padding around text - don't place coordinates at exact frame edges

IMPORTANT INSTRUCTIONS FOR SOUND EFFECTS:
- Suggest sound effects that enhance key moments and actions
- Keep sound names simple and universal (e.g., "whoosh", "ding", "pop")
- Time sounds precisely to match visual events
- Consider both diegetic sounds (from the scene) and non-diegetic sounds (for emphasis)
- Typical sounds: whoosh, swoosh, pop, ding, chime, bell, click, snap, splash, thud, bounce, sparkle
- Place 5-10 sound effects throughout the video for natural enhancement

""" + transcript_section + """

Use the following comprehensive library of functions to make your choices for the suggested_effects and text_overlays fields. You must use the exact function names and parameters as defined here.

""" + self.effects_documentation
        
        return prompt_template
    
    def extract_video_frames(self, video_path: Path, num_frames: int = 10) -> list:
        """Extract frames from video for analysis"""
        print(f"🎬 Extracting {num_frames} frames from video...")
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        print(f"   Extracted {len(frames)} frames")
        return frames
    
    def analyze_video(self, video_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Analyze video and generate description using Gemini"""
        print(f"🔍 Analyzing video: {video_path.name}")
        
        # Build the prompt with video path for transcript loading
        prompt = self.build_prompt(video_path)
        
        if dry_run:
            print("🔸 DRY RUN MODE - Skipping Gemini API call")
            print(f"   Prompt length: {len(prompt)} characters")
            print(f"   Would process video: {video_path.name}")
            if self.transcript_data:
                print(f"   Transcript words included: {self.transcript_data.get('total_words', 0)}")
            return {
                "prompt": prompt,
                "response": None,
                "raw_response": None,
                "dry_run": True,
                "has_transcript": self.transcript_data is not None
            }
        
        # Create video file for upload
        print("📤 Uploading video to Gemini...")
        video_file = genai.upload_file(str(video_path))
        
        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            print("   Processing video...")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")
        
        print("✅ Video uploaded successfully")
        
        # Send to Gemini for analysis
        print("🤖 Generating video description with Gemini 2.5 Pro...")
        
        try:
            response = self.model.generate_content([
                video_file,
                prompt
            ])
            
            # Parse the JSON response
            response_text = response.text
            
            # Clean up the response (remove markdown code blocks if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            print("✅ Video analysis complete")
            return {
                "prompt": prompt,
                "response": result,
                "raw_response": response_text
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"Raw response: {response_text[:500]}...")
            return {
                "prompt": prompt,
                "response": None,
                "raw_response": response_text,
                "error": str(e)
            }
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return {
                "prompt": prompt,
                "response": None,
                "raw_response": None,
                "error": str(e)
            }
    
    def save_results(self, video_path: Path, results: Dict[str, Any], output_dir: Optional[Path] = None, clean_folder: bool = True):
        """Save prompt and results to the video folder structure"""
        
        # Determine output directory
        if output_dir:
            prompts_dir = output_dir / "prompts"
        else:
            prompts_dir = video_path.parent / "prompts"
        
        # Clean the prompts folder if requested (only for dry-run files)
        if clean_folder and prompts_dir.exists() and results.get('dry_run'):
            # Check if this is scene_001 to clean once for all scenes
            if 'scene_001' in str(video_path):
                count = 0
                for file in prompts_dir.glob("*_dryrun.txt"):
                    file.unlink()
                    count += 1
                if count > 0:
                    print(f"🧹 Cleaned {count} previous dry-run files from prompts folder")
        
        prompts_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = video_path.stem
        
        # Add dry_run suffix if in dry run mode
        suffix = "_dryrun" if results.get('dry_run') else ""
        
        # Save prompt
        prompt_file = prompts_dir / f"{video_name}_prompt_{timestamp}{suffix}.txt"
        with open(prompt_file, 'w') as f:
            f.write(results['prompt'])
        print(f"📝 Prompt saved to: {prompt_file}")
        
        # Save response
        if results.get('response'):
            response_file = prompts_dir / f"{video_name}_response_{timestamp}.json"
            with open(response_file, 'w') as f:
                json.dump(results['response'], f, indent=2)
            print(f"📊 Response saved to: {response_file}")
        
        # Save raw response if different
        if results.get('raw_response'):
            raw_file = prompts_dir / f"{video_name}_raw_{timestamp}.txt"
            with open(raw_file, 'w') as f:
                f.write(results['raw_response'])
            print(f"📄 Raw response saved to: {raw_file}")
        
        # Update prompts.yaml
        self.update_prompts_yaml(video_name, results)
        
        return prompts_dir
    
    def update_prompts_yaml(self, video_name: str, results: Dict[str, Any]):
        """Update the central prompts.yaml file"""
        
        # Load existing prompts if file exists
        if self.prompts_file.exists():
            with open(self.prompts_file, 'r') as f:
                prompts_data = yaml.safe_load(f) or {}
        else:
            prompts_data = {}
        
        # Ensure structure exists
        if 'prompts' not in prompts_data:
            prompts_data['prompts'] = {}
        
        if 'video_description' not in prompts_data['prompts']:
            prompts_data['prompts']['video_description'] = {
                'description': 'Video analysis and editing plan generation',
                'model': 'gemini-2.5-pro',
                'instances': []
            }
        
        # Add this instance
        instance = {
            'video': video_name,
            'timestamp': datetime.now().isoformat(),
            'prompt_length': len(results['prompt']),
            'response_valid': results.get('response') is not None,
            'effects_suggested': 0
        }
        
        # Count suggested effects
        if results.get('response') and 'scenes' in results['response']:
            for scene in results['response']['scenes']:
                if 'scene_description' in scene and 'suggested_effects' in scene['scene_description']:
                    instance['effects_suggested'] += len(scene['scene_description']['suggested_effects'])
        
        prompts_data['prompts']['video_description']['instances'].append(instance)
        
        # Save updated prompts.yaml
        with open(self.prompts_file, 'w') as f:
            yaml.dump(prompts_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"📚 Updated prompts.yaml with new instance")


def main():
    """Main function"""
    
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python video_description_generator.py <video_path> [options]")
        print("\nOptions:")
        print("  --output-dir <dir>  Specify output directory for results")
        print("  --dry-run          Extract effects and build prompt without calling Gemini API")
        print("  --no-clean         Don't clean previous dry-run files from prompts folder")
        print("  -h, --help         Show this help message")
        sys.exit(0 if '--help' in sys.argv or '-h' in sys.argv else 1)
    
    video_path = Path(sys.argv[1])
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    # Parse arguments
    output_dir = None
    dry_run = False
    clean_folder = True
    
    for i, arg in enumerate(sys.argv):
        if arg == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = Path(sys.argv[i + 1])
            output_dir.mkdir(exist_ok=True)
        elif arg == '--dry-run':
            dry_run = True
        elif arg == '--no-clean':
            clean_folder = False
    
    print("=" * 80)
    print("VIDEO DESCRIPTION GENERATOR")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Output: {output_dir or video_path.parent}")
    print(f"Mode: {'DRY RUN' if dry_run else 'FULL EXECUTION'}")
    print("=" * 80)
    
    # Initialize generator
    generator = VideoDescriptionGenerator()
    
    # Analyze video
    results = generator.analyze_video(video_path, dry_run=dry_run)
    
    # Save results
    output_path = generator.save_results(video_path, results, output_dir, clean_folder=clean_folder)
    
    print("=" * 80)
    print("✅ GENERATION COMPLETE")
    print(f"📁 Results saved in: {output_path}")
    print("=" * 80)
    
    # Print summary if successful
    if results.get('response'):
        response = results['response']
        print("\n📊 SUMMARY:")
        print(f"   Characters: {len(response.get('characters_in_video', []))}")
        print(f"   Scenes: {len(response.get('scenes', []))}")
        
        total_effects = 0
        for scene in response.get('scenes', []):
            if 'scene_description' in scene and 'suggested_effects' in scene['scene_description']:
                total_effects += len(scene['scene_description']['suggested_effects'])
        print(f"   Effects suggested: {total_effects}")


if __name__ == "__main__":
    main()