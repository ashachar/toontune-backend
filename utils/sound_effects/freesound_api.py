#!/usr/bin/env python3
"""
Freesound API integration for searching and managing sound effects.
Supports caching to avoid repeated API calls.
"""

import os
import json
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import quote
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class FreesoundAPI:
    """Freesound API client for searching and downloading sound effects."""
    
    BASE_URL = "https://freesound.org/apiv2"
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "sound_effects/cache"):
        """
        Initialize Freesound API client.
        
        Args:
            api_key: Freesound API key (or from env FREESOUND_API_KEY)
            cache_dir: Directory for caching API responses
        """
        self.api_key = api_key or os.environ.get("FREESOUND_API_KEY")
        if not self.api_key:
            raise ValueError("Missing FREESOUND_API_KEY environment variable or api_key parameter")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers = {"Authorization": f"Token {self.api_key}"}
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Load cached response for a query."""
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                print(f"✓ Loaded from cache: {query}")
                return data
            except Exception as e:
                print(f"Warning: Failed to load cache for {query}: {e}")
        
        return None
    
    def _save_to_cache(self, query: str, data: Dict[str, Any]):
        """Save response to cache."""
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved to cache: {query}")
        except Exception as e:
            print(f"Warning: Failed to save cache for {query}: {e}")
    
    def search_sound(self, query: str, page_size: int = 10, use_cache: bool = True, 
                    max_duration: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Search for sound effects by name.
        
        Args:
            query: Search query (e.g., "door creak", "swoosh")
            page_size: Number of results to search through
            use_cache: Whether to use cached results
            max_duration: Maximum duration in seconds (default 0.5s)
        
        Returns:
            Sound effect information with preview URL and attribution
        """
        # Check cache first
        cache_key_with_duration = f"{query}_max{max_duration}"
        if use_cache:
            cached = self._load_from_cache(cache_key_with_duration)
            if cached:
                return cached
        
        # Build API request - get more results to filter
        url = f"{self.BASE_URL}/search/text/"
        params = {
            "query": query,
            "filter": f'license:"Creative Commons 0" OR license:"Attribution" AND duration:[0 TO {max_duration}]',
            "sort": "score",
            "page_size": str(page_size),
            "fields": "id,name,username,license,previews,url,duration,download"
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("results"):
                print(f"No results found for: {query} (duration < {max_duration}s)")
                # Try without duration filter
                params["filter"] = 'license:"Creative Commons 0" OR license:"Attribution"'
                response = requests.get(url, params=params, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("results"):
                    print(f"No results found for: {query}")
                    return None
            
            # Find first result under max_duration
            selected = None
            for result in data["results"]:
                duration = result.get("duration", 999)
                if duration <= max_duration:
                    selected = result
                    print(f"✓ Found short sound: {result['name']} ({duration:.2f}s)")
                    break
            
            # If no short sound found, use first result but warn
            if not selected:
                selected = data["results"][0]
                duration = selected.get("duration", 0)
                print(f"⚠ Warning: No sound under {max_duration}s, using {selected['name']} ({duration:.2f}s)")
            
            # Get preview URL (prefer high quality)
            preview_url = (
                selected.get("previews", {}).get("preview-hq-mp3") or
                selected.get("previews", {}).get("preview-lq-mp3") or
                selected.get("previews", {}).get("preview-hq-ogg") or
                selected.get("previews", {}).get("preview-lq-ogg")
            )
            
            result = {
                "id": selected["id"],
                "name": selected["name"],
                "author": selected["username"],
                "license": selected["license"],
                "duration": selected.get("duration", 0),
                "preview_url": preview_url,
                "download_url": selected.get("download"),
                "page_url": selected["url"],
                "attribution": self._build_attribution(selected),
                "query": query
            }
            
            # Save to cache
            if use_cache:
                self._save_to_cache(cache_key_with_duration, result)
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching Freesound for '{query}': {e}")
            return None
    
    def _build_attribution(self, sound_info: Dict[str, Any]) -> str:
        """Build attribution string for a sound."""
        page_url = f"https://freesound.org/s/{sound_info['id']}/"
        return (f'"{sound_info["name"]}" by {sound_info["username"]} '
                f'({page_url}) — License: {sound_info["license"]}')
    
    def download_sound(self, sound_info: Dict[str, Any], output_dir: str = "sound_effects/downloaded") -> Optional[str]:
        """
        Download a sound effect preview.
        
        Args:
            sound_info: Sound information from search_sound()
            output_dir: Directory to save downloaded files
        
        Returns:
            Path to downloaded file or None if failed
        """
        if not sound_info or not sound_info.get("preview_url"):
            print("No preview URL available")
            return None
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        safe_name = "".join(c for c in sound_info["name"] if c.isalnum() or c in "._- ")[:50]
        ext = ".mp3" if "mp3" in sound_info["preview_url"] else ".ogg"
        filename = f"{sound_info['id']}_{safe_name}{ext}"
        filepath = output_path / filename
        
        # Check if already downloaded
        if filepath.exists():
            print(f"✓ Already downloaded: {filename}")
            return str(filepath)
        
        try:
            print(f"Downloading: {sound_info['name']}...")
            response = requests.get(sound_info["preview_url"], stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Downloaded: {filename}")
            
            # Save attribution info
            attribution_file = filepath.with_suffix('.txt')
            with open(attribution_file, 'w') as f:
                f.write(sound_info["attribution"])
            
            return str(filepath)
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading sound: {e}")
            return None
    
    def search_and_download(self, query: str) -> Optional[str]:
        """
        Search for a sound and download it.
        
        Args:
            query: Search query
        
        Returns:
            Path to downloaded file or None if failed
        """
        sound_info = self.search_sound(query)
        if sound_info:
            return self.download_sound(sound_info)
        return None


def main():
    """CLI for testing Freesound API."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python freesound_api.py <search_query>")
        print("Example: python freesound_api.py 'door creak'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    # Initialize API
    api = FreesoundAPI()
    
    # Search and download
    filepath = api.search_and_download(query)
    if filepath:
        print(f"\nSound effect saved to: {filepath}")
    else:
        print(f"\nFailed to download sound effect for: {query}")


if __name__ == "__main__":
    main()