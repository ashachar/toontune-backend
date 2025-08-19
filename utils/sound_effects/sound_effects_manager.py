#!/usr/bin/env python3
"""
Sound Effects Manager for handling sound effect downloads, caching, and lookups.
Checks local files before downloading new ones.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from .freesound_api import FreesoundAPI


@dataclass
class SoundEffect:
    """Represents a sound effect with metadata."""
    name: str
    timestamp: float
    filepath: str
    query: str
    attribution: str
    downloaded_at: str
    freesound_id: Optional[int] = None
    duration: Optional[float] = None


class SoundEffectsManager:
    """Manages sound effects with local caching and smart lookups."""
    
    def __init__(self, download_dir: str = "sound_effects/downloaded", 
                 metadata_file: str = "sound_effects/sound_effects_registry.json"):
        """
        Initialize Sound Effects Manager.
        
        Args:
            download_dir: Directory for downloaded sound effects
            metadata_file: JSON file storing metadata about all sound effects
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = Path(metadata_file)
        self.metadata = self._load_metadata()
        
        self.api = FreesoundAPI()
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        try:
            # Ensure parent directory exists
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")
    
    def find_existing_sound(self, sound_name: str) -> Optional[str]:
        """
        Find an existing sound effect by name.
        
        Args:
            sound_name: Name of the sound effect (e.g., "ding", "swoosh")
        
        Returns:
            Path to existing sound file or None
        """
        # Check metadata first
        if sound_name in self.metadata:
            filepath = self.metadata[sound_name].get("filepath")
            if filepath and Path(filepath).exists():
                print(f"✓ Found existing sound effect: {sound_name} -> {filepath}")
                return filepath
        
        # Check download directory for matching files
        pattern = f"*{sound_name}*"
        for file in self.download_dir.glob(pattern):
            if file.suffix in ['.mp3', '.ogg', '.wav']:
                print(f"✓ Found matching file: {sound_name} -> {file}")
                return str(file)
        
        return None
    
    def get_or_download_sound(self, sound_name: str, query: Optional[str] = None) -> Optional[str]:
        """
        Get a sound effect, downloading if necessary.
        
        Args:
            sound_name: Name identifier for the sound (e.g., "ding", "swoosh")
            query: Search query for Freesound (defaults to sound_name)
        
        Returns:
            Path to sound file or None if failed
        """
        # Check for existing sound first
        existing = self.find_existing_sound(sound_name)
        if existing:
            return existing
        
        # Use sound_name as query if not provided
        if query is None:
            query = sound_name
        
        print(f"Searching for new sound effect: {sound_name} (query: {query})")
        
        # Search and download from Freesound
        sound_info = self.api.search_sound(query)
        if not sound_info:
            print(f"No sound found for: {query}")
            return None
        
        filepath = self.api.download_sound(sound_info, str(self.download_dir))
        if filepath:
            # Update metadata
            self.metadata[sound_name] = {
                "filepath": filepath,
                "query": query,
                "attribution": sound_info["attribution"],
                "freesound_id": sound_info["id"],
                "duration": sound_info.get("duration"),
                "downloaded_at": datetime.now().isoformat()
            }
            self._save_metadata()
            
        return filepath
    
    def process_sound_effects_list(self, sound_effects: List[Dict]) -> Dict[str, str]:
        """
        Process a list of sound effects from video metadata.
        
        Args:
            sound_effects: List of dicts with 'sound' and 'timestamp' keys
        
        Returns:
            Dictionary mapping sound names to file paths
        """
        result = {}
        
        # Define better search queries for common sound effects
        sound_queries = {
            "ding": "bell ding bright",
            "swoosh": "whoosh swoosh transition",
            "chime": "chime bell magical",
            "sparkle": "sparkle magic shimmer",
            "pop": "pop bubble cartoon"
        }
        
        for effect in sound_effects:
            sound_name = effect.get("sound", "").lower()
            timestamp = effect.get("timestamp")
            
            if not sound_name:
                continue
            
            # Get custom query if available
            query = sound_queries.get(sound_name, sound_name)
            
            print(f"\nProcessing: {sound_name} at {timestamp}s")
            filepath = self.get_or_download_sound(sound_name, query)
            
            if filepath:
                result[sound_name] = filepath
            else:
                print(f"⚠ Could not obtain sound effect: {sound_name}")
        
        return result
    
    def get_attributions(self) -> List[str]:
        """Get all attributions for downloaded sounds."""
        attributions = []
        for sound_name, info in self.metadata.items():
            if "attribution" in info:
                attributions.append(f"{sound_name}: {info['attribution']}")
        return attributions
    
    def export_sound_mapping(self, output_file: str = "sound_effects/sound_mapping.json"):
        """Export sound name to filepath mapping."""
        mapping = {}
        for sound_name, info in self.metadata.items():
            if "filepath" in info and Path(info["filepath"]).exists():
                mapping[sound_name] = info["filepath"]
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"✓ Exported sound mapping to: {output_file}")
        return mapping


def main():
    """Test the Sound Effects Manager."""
    import sys
    
    # Test sound effects from the video
    test_effects = [
        {"sound": "ding", "timestamp": "3.579"},
        {"sound": "swoosh", "timestamp": "13.050"},
        {"sound": "chime", "timestamp": "17.700"},
        {"sound": "sparkle", "timestamp": "40.119"},
        {"sound": "pop", "timestamp": "44.840"}
    ]
    
    manager = SoundEffectsManager()
    
    if len(sys.argv) > 1:
        # Single sound test
        sound_name = sys.argv[1]
        filepath = manager.get_or_download_sound(sound_name)
        if filepath:
            print(f"\n✓ Sound ready: {filepath}")
    else:
        # Process all test effects
        print("Processing test sound effects...")
        results = manager.process_sound_effects_list(test_effects)
        
        print("\n" + "="*50)
        print("Sound Effects Mapping:")
        print("="*50)
        for name, path in results.items():
            print(f"{name:15} -> {path}")
        
        # Export mapping
        manager.export_sound_mapping()
        
        # Show attributions
        print("\n" + "="*50)
        print("Attributions:")
        print("="*50)
        for attr in manager.get_attributions():
            print(attr)


if __name__ == "__main__":
    main()