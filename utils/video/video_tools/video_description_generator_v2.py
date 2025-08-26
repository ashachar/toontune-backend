#!/usr/bin/env python3
"""
Video Description Generator V2 with Key Phrases and Cartoon Characters
=======================================================================

This version replaces word-by-word overlays with:
1. Key phrases (max 4 words) appearing maximum once every 20 seconds
2. Cartoon character suggestions related to the video content
"""

import sys
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
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

class VideoDescriptionGeneratorV2:
    """Generates video descriptions with key phrases and cartoon characters"""
    
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
        print("✅ Gemini 2.5 Pro configured for V2 prompts")
    
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
                            
                            # Get full text for this scene
                            scene_words = []
                            scene_text = []
                            for word in all_words:
                                word_start = word['start_ms'] / 1000.0  # Convert to seconds
                                word_end = word['end_ms'] / 1000.0
                                
                                if start_time <= word_start < end_time:
                                    # Adjust timing relative to scene start
                                    scene_words.append({
                                        'word': word['word'],
                                        'start_seconds': word_start - start_time,
                                        'end_seconds': word_end - start_time,
                                    })
                                    scene_text.append(word['word'])
                            
                            return {
                                'scene_number': scene_num,
                                'total_words': len(scene_words),
                                'duration_seconds': end_time - start_time,
                                'full_text': ' '.join(scene_text),
                                'words_with_timing': scene_words
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
        """Build the complete prompt with new key phrases and cartoon character approach"""
        
        if not self.effects_documentation:
            self.effects_documentation = self.extract_effects_documentation()
        
        # Load transcript if available
        transcript_section = ""
        if video_path:
            self.transcript_data = self.load_transcript_for_scene(video_path)
            if self.transcript_data:
                transcript_section = f"""

TRANSCRIPT FOR THIS SCENE:
The following is the complete text/lyrics for this video segment. Use this to identify KEY PHRASES (not individual words) that should appear on screen.
Full text: "{self.transcript_data.get('full_text', '')}"
Duration: {self.transcript_data.get('duration_seconds', 0):.1f} seconds

WORD TIMINGS (for reference only - DO NOT suggest overlay for every word):
{json.dumps(self.transcript_data.get('words_with_timing', [])[:10], indent=2)}
... (showing first 10 words for reference)

"""
        elif self.transcript_data:
            # Use pre-loaded transcript data
            transcript_section = f"""

TRANSCRIPT FOR THIS SCENE:
Full text: "{' '.join([w['word'] for w in self.transcript_data.get('words', [])])}"
Duration: {self.transcript_data.get('duration_seconds', 0):.1f} seconds

"""
        
        prompt_template = """You are an expert video editor and senior director. Your task is to analyze the provided video and produce a detailed editing plan in a specific JSON format. Your analysis must be based only on the visual information in the video, without making any external assumptions about the plot, characters, or film title.

CRITICAL NEW INSTRUCTIONS FOR OVERLAYS:

1. KEY PHRASES (not individual words):
   - Identify ONLY the most impactful phrases (maximum 4 words each)
   - Suggest KEY PHRASES maximum once every 20 seconds (if a 60-second scene, maximum 3 key phrases)
   - These should be the most emotionally important or thematically central phrases
   - Place them strategically where they won't occlude important action
   - Examples: "Do Re Mi", "climb every mountain", "favorite things", "so long farewell"

2. CARTOON CHARACTERS:
   - Suggest cute, animated cartoon characters that are 100% related to what's being said/shown
   - Characters should enhance the narrative, not distract from it
   - Examples:
     * If lyrics mention "raindrops on roses" → suggest a cute animated rose with dewdrops
     * If showing mountains → suggest a happy mountain goat character
     * If mentioning food → suggest an animated version of that food with a face
   - Characters should appear briefly (2-4 seconds) and be positioned to not block main subjects
   - Maximum 1 cartoon character every 20 seconds (same frequency as key phrases)
   - CRITICAL: NEVER show a cartoon character at the same time as a key phrase - they must be temporally separated by at least 3 seconds

The output MUST be a single, valid JSON object structured as follows:

characters_in_video: A top-level list of objects identifying every distinct character from the video.

video_description: A concise summary of the video's narrative and visual progression.

sound_effects: A list of suggested sound effects with precise timestamps.

scenes: A list breaking down the video into distinct scenes. Each scene must contain:

  start_seconds and end_seconds: Accurate timestamps as strings.
  
  scene_description: An object containing:
    - characters: List of character names present
    - most_prominent_figure: Main visual focus
    - character_actions: What each character is doing
    - background: Environment description
    - camera_angle: Cinematographic term for the shot
    - camera_movement: (if exists) Direction and speed
  
  suggested_effects: Visual effects from the documentation
  
  key_phrases: (NEW - REPLACES text_overlays) A list of the most important phrases to display. Each object must contain:
    - phrase: The key phrase (1-4 words maximum) that captures the essence of this moment
    - start_seconds: When the phrase appears
    - duration_seconds: How long it stays (typically 3-5 seconds)
    - top_left_pixels: Pixel coordinates for placement
    - bottom_right_pixels: Pixel coordinates for placement
    - style: Visual style like "elegant_fade", "playful_bounce", "dramatic_reveal"
    - importance: "critical", "high", or "medium" to indicate narrative importance
  
  cartoon_characters: (NEW) A list of cartoon character suggestions. Each object must contain:
    - character_type: What the character is (e.g., "dancing_musical_note", "happy_mountain", "singing_flower")
    - related_to: What lyric/action this relates to
    - start_seconds: When the character appears
    - duration_seconds: How long it stays (2-4 seconds typically)
    - position_pixels: Center position {"x": 100, "y": 200}
    - size_pixels: Size of character {"width": 80, "height": 80}
    - animation_style: How it moves (e.g., "bounce_in_place", "float_across", "spin_and_fade")
    - interaction: How it relates to the scene (e.g., "follows_character", "reacts_to_music", "mimics_action")

Here is an example of the expected output format:

{
  "characters_in_video": [
    {
      "name": "Woman in Blue Dress",
      "description": "A woman with brown hair wearing a blue dress, appears to be singing"
    },
    {
      "name": "Children Group",
      "description": "Seven children of various ages in traditional clothing"
    }
  ],
  "video_description": "A musical scene in a hillside meadow where a woman teaches children to sing using musical notes.",
  "sound_effects": [
    {
      "sound": "birds_chirping",
      "timestamp": "0.500"
    }
  ],
  "scenes": [
    {
      "start_seconds": "0.000",
      "end_seconds": "30.000",
      "scene_description": {
        "characters": ["Woman in Blue Dress", "Children Group"],
        "most_prominent_figure": "Woman in Blue Dress",
        "character_actions": {
          "Woman in Blue Dress": "Gesturing and teaching, mouth moving as if singing",
          "Children Group": "Standing in a line, following along with the lesson"
        },
        "background": "Green hillside meadow with mountains in the distance",
        "camera_angle": "Wide Shot",
        "suggested_effects": [
          {
            "effect_fn_name": "apply_gentle_vignette",
            "effect_timestamp": "5.000",
            "effect_fn_params": {
              "intensity": 0.3
            }
          }
        ]
      },
      "key_phrases": [
        {
          "phrase": "Do Re Mi",
          "start_seconds": "8.500",
          "duration_seconds": 4.0,
          "top_left_pixels": {"x": 50, "y": 50},
          "bottom_right_pixels": {"x": 200, "y": 100},
          "style": "playful_bounce",
          "importance": "critical"
        },
        {
          "phrase": "very beginning",
          "start_seconds": "25.000",
          "duration_seconds": 3.5,
          "top_left_pixels": {"x": 100, "y": 400},
          "bottom_right_pixels": {"x": 300, "y": 440},
          "style": "elegant_fade",
          "importance": "high"
        }
      ],
      "cartoon_characters": [
        {
          "character_type": "dancing_musical_note",
          "related_to": "Do Re Mi singing lesson",
          "start_seconds": "10.000",
          "duration_seconds": 3.0,
          "position_pixels": {"x": 150, "y": 150},
          "size_pixels": {"width": 60, "height": 80},
          "animation_style": "bounce_in_place",
          "interaction": "appears_near_singer"
        },
        {
          "character_type": "happy_deer",
          "related_to": "Doe a deer lyric",
          "start_seconds": "22.000",
          "duration_seconds": 2.5,
          "position_pixels": {"x": 500, "y": 300},
          "size_pixels": {"width": 100, "height": 120},
          "animation_style": "hop_across",
          "interaction": "crosses_background"
        }
      ]
    }
  ]
}

IMPORTANT GUIDELINES:
- For a 60-second scene: Maximum 3 key phrases and maximum 3 cartoon characters
- Both key phrases and cartoon characters appear maximum once every 20 seconds
- CRITICAL RULE: Never display a cartoon character and key phrase simultaneously
  * If key phrase at 10s-14s, cartoon must be before 7s or after 17s
  * Maintain at least 3 seconds separation between them
- Key phrases should be the MOST important/memorable parts of the lyrics
- Cartoon characters must be directly related to what's being said or shown
- Both should enhance, not distract from, the main action
- Consider the visual composition - don't cover faces or important action
- Keep suggestions tasteful and appropriate for all audiences

""" + transcript_section + """

Use the following comprehensive library of effects for your suggested_effects selections:

""" + self.effects_documentation + """

Remember: Focus on KEY PHRASES (not every word) and RELEVANT CARTOON CHARACTERS that enhance the storytelling!"""

        return prompt_template
    
    def generate_description(self, video_path: Path, output_dir: Path = None) -> Dict[str, Any]:
        """Generate video description using Gemini 2.5 Pro"""
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Load transcript if available
        self.transcript_data = self.load_transcript_for_scene(video_path)
        
        # Build prompt
        prompt = self.build_prompt(video_path)
        
        # Extract frames from video
        print(f"📹 Extracting frames from: {video_path}")
        frames = self.extract_video_frames(video_path)
        print(f"   Extracted {len(frames)} frames")
        
        # Prepare multimodal input
        print("🤖 Sending to Gemini 2.5 Pro...")
        
        # Create input with video frames and prompt
        input_parts = []
        
        # Add frames
        for i, frame_data in enumerate(frames):
            input_parts.append({
                "mime_type": "image/jpeg",
                "data": frame_data
            })
        
        # Add text prompt
        input_parts.append(prompt)
        
        # Generate response
        response = self.model.generate_content(input_parts)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            response_text = response.text
            
            # Find JSON content (between first { and last })
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Save if output directory provided
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / f"{video_path.stem}_description_v2.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"✅ Description saved to: {output_file}")
                
                return result
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            print(f"❌ Error parsing response: {e}")
            print(f"Response text: {response_text[:500]}...")
            return None
    
    def extract_video_frames(self, video_path: Path, num_frames: int = 8) -> List[str]:
        """Extract frames from video for analysis"""
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                # Convert to base64
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames.append(frame_b64)
        
        cap.release()
        return frames


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate video descriptions with key phrases and cartoon characters')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output-dir', help='Output directory for JSON', default='output')
    
    args = parser.parse_args()
    
    generator = VideoDescriptionGeneratorV2()
    result = generator.generate_description(Path(args.video), Path(args.output_dir))
    
    if result:
        print("\n📊 Summary:")
        print(f"  Characters: {len(result.get('characters_in_video', []))}")
        print(f"  Scenes: {len(result.get('scenes', []))}")
        
        # Count key phrases and cartoon characters
        total_phrases = 0
        total_characters = 0
        for scene in result.get('scenes', []):
            total_phrases += len(scene.get('key_phrases', []))
            total_characters += len(scene.get('cartoon_characters', []))
        
        print(f"  Key Phrases: {total_phrases} (was individual words before)")
        print(f"  Cartoon Characters: {total_characters} (new feature)")


if __name__ == "__main__":
    main()