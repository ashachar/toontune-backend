#!/usr/bin/env python3
"""
Coverr Background Video Manager

Manages downloading and caching of background videos from Coverr API.
Implements intelligent caching to avoid re-downloading videos.
"""

import os
import json
import hashlib
import requests
import yaml
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse
import re
from openai import OpenAI


class CoverrManager:
    """Manages Coverr video downloads and caching."""
    
    def __init__(self, demo_mode=False):
        self.api_base = "https://api.coverr.co"
        self.cache_dir = Path("assets/videos/coverr")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.demo_mode = demo_mode
        
        if not demo_mode:
            # Load API key from .env
            self.api_key = self._load_api_key()
        else:
            self.api_key = None
        
        # Load prompts configuration
        self.prompts_config = self._load_prompts_config()
        
        # Initialize OpenAI client for video selection
        self.openai_client = self._init_openai_client()
        
    def _load_api_key(self) -> str:
        """Load Coverr API key from .env file."""
        env_path = Path(".env")
        if not env_path.exists():
            raise ValueError("No .env file found. Please add COVERR_KEY to .env")
        
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("COVERR_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        
        raise ValueError("COVERR_KEY not found in .env file")
    
    def _load_prompts_config(self) -> Dict:
        """Load prompts configuration from prompts.yaml."""
        prompts_path = Path("prompts.yaml")
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _init_openai_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client for AI-based video selection."""
        try:
            env_path = Path(".env")
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith("OPENAI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            return OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
        return None
    
    def extract_keywords_from_transcript(self, transcript: str) -> List[str]:
        """
        Extract search keywords from a video transcript.
        Uses AI to identify relevant visual themes.
        """
        # For now, use a simple keyword extraction
        # In production, this would call an LLM with the prompt from prompts.yaml
        
        # Common visual keywords mapping - prioritize more generic visuals
        keyword_map = {
            "AI": ["futuristic", "tech", "neon", "circuit", "digital"],
            "artificial intelligence": ["robot", "future", "tech", "brain"],
            "math": ["abstract", "geometric", "lines", "grid", "numbers"],
            "calculus": ["graph", "wave", "curve", "abstract"],
            "computer": ["code", "screen", "keyboard", "tech"],
            "science": ["space", "particles", "abstract", "laboratory"],
            "invention": ["lightbulb", "creative", "innovation"],
            "discovery": ["space", "exploration", "nature"],
            "theory": ["abstract", "space", "particles"],
            "model": ["3d", "structure", "abstract"],
            "data": ["network", "dots", "visualization", "graph"],
            "language": ["text", "typing", "books"],
            "learn": ["books", "study", "classroom"],
            "equation": ["formula", "blackboard", "numbers"],
            "algorithm": ["flowchart", "network", "abstract"]
        }
        
        # Extract keywords based on transcript content
        transcript_lower = transcript.lower()
        keywords = []
        keyword_scores = {}
        
        # Score keywords by frequency of related terms
        for term, visual_keywords in keyword_map.items():
            if term in transcript_lower:
                count = transcript_lower.count(term)
                for keyword in visual_keywords:
                    if keyword not in keyword_scores:
                        keyword_scores[keyword] = 0
                    keyword_scores[keyword] += count
        
        # Sort by score and get top keywords
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [k for k, v in sorted_keywords[:8]]  # Get more keywords
        
        # If not enough keywords, add some safe defaults for tech/science content
        if len(keywords) < 3:
            defaults = ["abstract", "technology", "futuristic", "digital", "space"]
            for default in defaults:
                if default not in keywords:
                    keywords.append(default)
                if len(keywords) >= 5:
                    break
        
        return keywords[:5]
    
    def search_videos(self, keywords: List[str], page_size: int = 10) -> List[Dict]:
        """
        Search Coverr for videos matching keywords.
        
        Args:
            keywords: List of search keywords
            page_size: Number of results per page
            
        Returns:
            List of video metadata dictionaries
        """
        search_query = " ".join(keywords)
        
        params = {
            "api_key": self.api_key,
            "query": search_query,
            "page_size": page_size,
            "urls": "true",
            "sort": "popular"
        }
        
        response = requests.get(f"{self.api_base}/videos", params=params)
        
        if response.status_code != 200:
            print(f"Error searching Coverr: {response.status_code}")
            if response.status_code == 401:
                print("Authentication failed. Please check your COVERR_KEY in .env")
                print(f"Current API key (first 8 chars): {self.api_key[:8] if len(self.api_key) >= 8 else self.api_key}...")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Response text: {response.text[:200]}")
            return []
        
        data = response.json()
        return data.get("hits", [])
    
    def _get_cached_filename(self, project_name: str, start_time: float, 
                           end_time: float, video_id: str) -> str:
        """Generate the cached filename for a background video."""
        start_str = f"{int(start_time)}_{int((start_time % 1) * 10)}"
        end_str = f"{int(end_time)}_{int((end_time % 1) * 10)}" 
        return f"{project_name}_background_{start_str}_{end_str}_{video_id}.mp4"
    
    def check_cache(self, project_name: str, project_folder: Path, 
                   start_time: float, end_time: float) -> Optional[Path]:
        """
        Check if a background video already exists in cache.
        
        Args:
            project_name: Name of the project (e.g., "ai_math1")
            project_folder: Path to project folder
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            
        Returns:
            Path to cached video if exists, None otherwise
        """
        # Check project folder for existing background videos
        pattern = f"{project_name}_background_{int(start_time)}_{int(end_time)}_*.mp4"
        
        for file in project_folder.glob(pattern):
            if file.exists():
                print(f"Found cached background: {file}")
                return file
        
        return None
    
    def download_video(self, video_data: Dict, project_name: str, 
                      project_folder: Path, start_time: float, 
                      end_time: float) -> Path:
        """
        Download a video from Coverr and cache it.
        
        Args:
            video_data: Video metadata from Coverr API
            project_name: Name of the project
            project_folder: Path to project folder
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Path to downloaded video
        """
        video_id = video_data["id"]
        video_url = video_data["urls"]["mp4_download"]
        
        # Generate filenames
        cached_name = self._get_cached_filename(project_name, start_time, 
                                               end_time, video_id)
        
        # Save locations
        coverr_cache_path = self.cache_dir / cached_name
        project_cache_path = project_folder / cached_name
        
        # Check if already exists in Coverr cache
        if coverr_cache_path.exists():
            print(f"Found in Coverr cache: {coverr_cache_path}")
            # Copy to project folder
            import shutil
            shutil.copy2(coverr_cache_path, project_cache_path)
            return project_cache_path
        
        # Download the video
        print(f"Downloading video: {video_data['title']}")
        response = requests.get(video_url, stream=True)
        
        if response.status_code == 200:
            # Save to both locations
            content = response.content
            
            # Save to Coverr cache
            with open(coverr_cache_path, 'wb') as f:
                f.write(content)
            
            # Save to project folder
            with open(project_cache_path, 'wb') as f:
                f.write(content)
            
            # Register download with Coverr stats
            self._register_download(video_id)
            
            print(f"Downloaded and cached: {project_cache_path}")
            return project_cache_path
        else:
            raise Exception(f"Failed to download video: {response.status_code}")
    
    def _register_download(self, video_id: str):
        """Register a download with Coverr statistics."""
        url = f"{self.api_base}/videos/{video_id}/stats/downloads"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.patch(url, headers=headers)
        if response.status_code != 204:
            print(f"Warning: Failed to register download stats: {response.status_code}")
    
    def select_best_video(self, videos: List[Dict], transcript: str) -> Dict:
        """
        Use AI to select the most appropriate video from a list.
        
        Args:
            videos: List of video metadata from Coverr
            transcript: Video transcript for context
            
        Returns:
            Selected video metadata
        """
        if not videos:
            return None
        
        # If only one video, return it
        if len(videos) == 1:
            return videos[0]
        
        # Try to use AI selection if available
        if self.openai_client and self.prompts_config.get('coverr_video_selection'):
            try:
                # Prepare content summary
                content_summary = "Educational video about AI and mathematics, discussing how AI can create new mathematical theories and improve itself"
                
                # Get first 200 chars of transcript
                transcript_excerpt = transcript[:200] if len(transcript) > 200 else transcript
                
                # Format video list
                video_list = "\n".join([
                    f"{i+1}. {video['title']} - {video.get('description', 'No description')[:100]}"
                    for i, video in enumerate(videos[:10])
                ])
                
                # Get prompt template
                prompt_config = self.prompts_config['coverr_video_selection']
                
                # Call OpenAI
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt_config['system']},
                        {"role": "user", "content": prompt_config['user'].format(
                            content_summary=content_summary,
                            transcript_excerpt=transcript_excerpt,
                            video_list=video_list
                        )}
                    ],
                    temperature=0.3,
                    max_tokens=10
                )
                
                # Parse response
                selection = response.choices[0].message.content.strip()
                try:
                    index = int(selection) - 1
                    if 0 <= index < len(videos):
                        selected = videos[index]
                        print(f"AI selected video #{index+1}: {selected['title']}")
                        return selected
                except ValueError:
                    print(f"AI selection invalid: {selection}")
                    
            except Exception as e:
                print(f"AI selection failed: {e}")
        
        # Fallback: Use heuristic selection
        # Prefer videos with certain keywords in title/description
        preferred_keywords = ['abstract', 'digital', 'tech', 'futuristic', 'particles', 
                            'visualization', 'network', 'cosmic', 'data', 'code']
        
        scored_videos = []
        for video in videos[:10]:  # Limit to first 10
            score = 0
            title_lower = video.get('title', '').lower()
            desc_lower = video.get('description', '').lower()
            
            for keyword in preferred_keywords:
                if keyword in title_lower:
                    score += 2
                if keyword in desc_lower:
                    score += 1
            
            # Penalize certain keywords
            avoid_keywords = ['party', 'dinner', 'casual', 'girl', 'boy', 'people', 'face', 'portrait']
            for keyword in avoid_keywords:
                if keyword in title_lower:
                    score -= 3
                if keyword in desc_lower:
                    score -= 1
            
            scored_videos.append((video, score))
        
        # Sort by score
        scored_videos.sort(key=lambda x: x[1], reverse=True)
        selected = scored_videos[0][0]
        print(f"Heuristic selected: {selected['title']} (score: {scored_videos[0][1]})")
        
        return selected
    
    def create_demo_background(self, output_path: Path, duration: float = 5.0) -> Path:
        """
        Create a demo background video with animated gradient.
        
        Args:
            output_path: Path to save the demo video
            duration: Duration in seconds
            
        Returns:
            Path to created video
        """
        print("Creating demo background video...")
        
        # Create animated gradient background using FFmpeg
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"gradients=size=1920x1080:duration={duration}:speed=0.1",
            "-vf", "hue=H=2*PI*t/10,scale=1920:1080",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            # Fallback to simpler gradient
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=blue:size=1920x1080:duration={duration}",
                "-vf", "geq=r='X/W*155+100':g='Y/H*155+100':b='128+127*sin(2*PI*T)',scale=1920:1080",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]
            subprocess.run(cmd, check=True)
        
        print(f"Created demo background: {output_path}")
        return output_path
    
    def get_background_for_video(self, project_name: str, transcript_path: Path,
                                start_time: float = 0, end_time: float = 5) -> Path:
        """
        Get a background video for a project, using cache if available.
        
        Args:
            project_name: Name of the project (e.g., "ai_math1")
            transcript_path: Path to transcript file
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            
        Returns:
            Path to background video file
        """
        # Setup project folder
        project_folder = Path(f"uploads/assets/videos/{project_name}")
        project_folder.mkdir(parents=True, exist_ok=True)
        
        # Check cache first
        cached = self.check_cache(project_name, project_folder, start_time, end_time)
        if cached:
            return cached
        
        # If in demo mode, create a demo background
        if self.demo_mode:
            demo_path = project_folder / f"{project_name}_background_{int(start_time)}_{int(end_time)}_demo.mp4"
            duration = end_time - start_time
            return self.create_demo_background(demo_path, duration)
        
        # Load transcript
        with open(transcript_path, 'r') as f:
            transcript = f.read()
        
        # Extract keywords
        keywords = self.extract_keywords_from_transcript(transcript)
        print(f"Search keywords: {keywords}")
        
        # Search for videos
        videos = self.search_videos(keywords)
        
        if not videos:
            print("No videos found with primary keywords, trying each keyword individually...")
            # Try each keyword separately
            for keyword in keywords:
                videos = self.search_videos([keyword])
                if videos:
                    print(f"Found videos with keyword: {keyword}")
                    break
        
        if not videos:
            print("Still no videos, trying broader search...")
            # Try different combinations of generic terms
            fallback_searches = [
                ["abstract"],
                ["technology"], 
                ["space"],
                ["futuristic"],
                ["particles"],
                ["digital"],
                ["network"]
            ]
            for search_terms in fallback_searches:
                videos = self.search_videos(search_terms)
                if videos:
                    print(f"Found videos with fallback terms: {search_terms}")
                    break
        
        if not videos:
            print("API authentication failed. Falling back to demo mode...")
            # Fallback to demo mode
            demo_path = project_folder / f"{project_name}_background_{int(start_time)}_{int(end_time)}_demo.mp4"
            duration = end_time - start_time
            return self.create_demo_background(demo_path, duration)
        
        # Select best video using AI or heuristics
        video = self.select_best_video(videos, transcript)
        if not video:
            print("No suitable video found. Using demo background...")
            demo_path = project_folder / f"{project_name}_background_{int(start_time)}_{int(end_time)}_demo.mp4"
            duration = end_time - start_time
            return self.create_demo_background(demo_path, duration)
        
        print(f"Selected video: {video['title']}")
        
        # Download and cache
        return self.download_video(video, project_name, project_folder, 
                                  start_time, end_time)


def main():
    """Test the Coverr manager."""
    manager = CoverrManager()
    
    # Test with ai_math1
    project_name = "ai_math1"
    transcript_path = Path(f"uploads/assets/videos/{project_name}/{project_name}_whisper_transcript.txt")
    
    if transcript_path.exists():
        background_path = manager.get_background_for_video(
            project_name, 
            transcript_path,
            start_time=0,
            end_time=5
        )
        print(f"Background video ready: {background_path}")
    else:
        print(f"Transcript not found: {transcript_path}")


if __name__ == "__main__":
    main()