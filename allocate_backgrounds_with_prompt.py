#!/usr/bin/env python3
"""
Allocate background themes to video timestamps using AI prompt.
This script demonstrates how to use the transcript_background_allocation prompt.
"""

import json
import yaml
from pathlib import Path
import subprocess


def load_prompt_template(prompt_name):
    """Load prompt template from prompts.yaml."""
    with open('prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)
    
    if prompt_name in prompts:
        return prompts[prompt_name]
    raise ValueError(f"Prompt '{prompt_name}' not found in prompts.yaml")


def mock_llm_background_allocation(transcript_text, duration):
    """
    Mock LLM response for background allocation without API.
    Analyzes transcript to identify themes and allocate backgrounds.
    """
    # Simple keyword-based analysis for demonstration
    segments = []
    
    # Analyze content in chunks
    words = transcript_text.lower().split()
    chunk_size = len(words) // 8  # Roughly 8 segments
    
    theme_keywords = {
        "abstract_tech": ["ai", "artificial", "intelligence", "technology", "digital", "computer", "model"],
        "mathematics": ["math", "calculus", "derivative", "integral", "equation", "theorem", "axiom", "formula"],
        "data_visualization": ["data", "trend", "analysis", "pattern", "statistics", "visualization"],
        "research": ["research", "study", "experiment", "discover", "investigate", "science"],
        "innovation": ["new", "create", "invent", "innovation", "build", "develop"],
        "education": ["learn", "teach", "understand", "explain", "student", "knowledge"],
        "cosmic": ["universe", "philosophy", "exist", "fundamental", "nature", "reality"],
        "minimal": ["detail", "specific", "technical", "complex", "intricate"]
    }
    
    current_time = 0.0
    segment_duration = duration / 8
    
    for i in range(8):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_text = ' '.join(chunk_words)
        
        # Score each theme based on keyword matches
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = sum(1 for keyword in keywords if keyword in chunk_text)
            if score > 0:
                theme_scores[theme] = score
        
        # Select best theme or default
        if theme_scores:
            best_theme = max(theme_scores, key=theme_scores.get)
        else:
            # Default based on position
            if i < 2:
                best_theme = "abstract_tech"
            elif i < 4:
                best_theme = "mathematics"
            elif i < 6:
                best_theme = "data_visualization"
            else:
                best_theme = "innovation"
        
        # Get keywords for selected theme
        keywords = theme_keywords.get(best_theme, ["general"])
        
        segment = {
            "start_time": round(current_time, 1),
            "end_time": round(min(current_time + segment_duration, duration), 1),
            "theme": best_theme,
            "keywords": keywords[:4],  # Limit to 4 keywords
            "reason": f"Section {i+1}: Content focuses on {best_theme.replace('_', ' ')}"
        }
        
        segments.append(segment)
        current_time += segment_duration
    
    return segments


def allocate_backgrounds_for_video(video_path, use_llm=False):
    """
    Allocate background themes for a video based on its transcript.
    
    Args:
        video_path: Path to the video file
        use_llm: Whether to use actual LLM API (False = use mock)
    """
    video_path = Path(video_path)
    project_folder = video_path.parent / video_path.stem
    transcript_path = project_folder / "transcript.json"
    
    print(f"📝 Analyzing transcript: {transcript_path}")
    
    # Load transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    transcript_text = transcript_data['text']
    
    # Get video duration
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    duration = float(subprocess.check_output(cmd).decode().strip())
    
    print(f"⏱️  Video duration: {duration:.1f} seconds")
    
    if use_llm:
        # Load prompt template
        prompt_config = load_prompt_template('transcript_background_allocation')
        
        # Format the prompt
        user_prompt = prompt_config['user'].format(
            full_transcript=transcript_text,
            duration_seconds=duration
        )
        
        print("🤖 Using LLM for background allocation...")
        print("\nPrompt preview:")
        print("-" * 50)
        print(user_prompt[:500] + "...")
        print("-" * 50)
        
        # Here you would call your LLM API
        # segments = call_llm_api(system_prompt, user_prompt)
        segments = []  # Placeholder
        
    else:
        print("🎭 Using mock allocation (no API)")
        segments = mock_llm_background_allocation(transcript_text, duration)
    
    # Save allocation
    output_path = project_folder / "background_allocation.json"
    with open(output_path, 'w') as f:
        json.dump(segments, f, indent=2)
    
    print(f"\n✅ Background allocation saved: {output_path}")
    print("\n📊 Allocated segments:")
    print("-" * 60)
    
    for i, seg in enumerate(segments, 1):
        duration = seg['end_time'] - seg['start_time']
        print(f"{i}. [{seg['start_time']:6.1f}s - {seg['end_time']:6.1f}s] "
              f"({duration:5.1f}s) : {seg['theme']:20s}")
        print(f"   Keywords: {', '.join(seg['keywords'])}")
        print(f"   Reason: {seg['reason']}")
    
    print("-" * 60)
    print(f"Total segments: {len(segments)}")
    
    return segments


def main():
    """Demo background allocation for the AI math video."""
    
    video_path = Path("uploads/assets/videos/ai_math1.mp4")
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print("=" * 60)
    print("Background Theme Allocation Using AI Prompt")
    print("=" * 60)
    print()
    
    # Allocate backgrounds
    segments = allocate_backgrounds_for_video(video_path, use_llm=False)
    
    # Show how this integrates with the pipeline
    print("\n🔗 Integration with pipeline:")
    print("1. This allocation determines when backgrounds change")
    print("2. The pipeline searches for stock videos matching each theme")
    print("3. Videos are downloaded/cached and applied at specified times")
    print("4. Smooth transitions occur at theme boundaries")


if __name__ == "__main__":
    main()