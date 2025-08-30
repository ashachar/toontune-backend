#!/usr/bin/env python3
"""
Background Cache Manager
Manages a centralized cache of background videos that can be reused across projects.
"""

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import subprocess


class BackgroundCacheManager:
    """Manages cached background videos with metadata and search capabilities."""
    
    def __init__(self, cache_dir: Path = None):
        """
        Initialize the background cache manager.
        
        Args:
            cache_dir: Path to cache directory (default: uploads/assets/backgrounds)
        """
        self.cache_dir = cache_dir or Path("uploads/assets/backgrounds")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load or initialize cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "backgrounds": {},
                "last_updated": datetime.now().isoformat()
            }
            self.save_metadata()
    
    def save_metadata(self):
        """Save cache metadata to disk."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_video_hash(self, video_path: Path) -> str:
        """Generate unique hash for a video file."""
        stats = video_path.stat()
        hash_input = f"{video_path.name}_{stats.st_size}_{stats.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def add_background(self, 
                       video_path: Path,
                       theme: str,
                       keywords: List[str],
                       source: str = "unknown",
                       source_id: str = None) -> Path:
        """
        Add a background video to the cache.
        
        Args:
            video_path: Path to the video file
            theme: Primary theme (e.g., "abstract_tech", "nature", etc.)
            keywords: List of keywords describing the video
            source: Source of the video (e.g., "coverr", "pexels", "local")
            source_id: Original ID from the source
            
        Returns:
            Path to the cached video
        """
        video_hash = self.get_video_hash(video_path)
        
        # Create filename with theme and hash
        cached_filename = f"{theme}_{video_hash}.mp4"
        cached_path = self.cache_dir / cached_filename
        
        # Copy if not already cached
        if not cached_path.exists():
            shutil.copy2(video_path, cached_path)
            print(f"✅ Cached: {cached_filename}")
        else:
            print(f"♻️  Already cached: {cached_filename}")
        
        # Update metadata
        self.metadata["backgrounds"][video_hash] = {
            "filename": cached_filename,
            "theme": theme,
            "keywords": keywords,
            "source": source,
            "source_id": source_id,
            "added_date": datetime.now().isoformat(),
            "file_size": cached_path.stat().st_size,
            "duration": self.get_video_duration(cached_path)
        }
        
        self.save_metadata()
        return cached_path
    
    def get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", 
                   "format=duration", "-of", 
                   "default=noprint_wrappers=1:nokey=1", str(video_path)]
            output = subprocess.check_output(cmd).decode().strip()
            return float(output)
        except:
            return 0.0
    
    def search_by_theme(self, theme: str) -> List[Dict]:
        """
        Search for backgrounds by theme.
        
        Args:
            theme: Theme to search for (exact or partial match)
            
        Returns:
            List of matching background entries
        """
        matches = []
        theme_lower = theme.lower()
        
        for hash_id, bg_data in self.metadata["backgrounds"].items():
            if theme_lower in bg_data["theme"].lower():
                bg_data["hash_id"] = hash_id
                bg_data["path"] = self.cache_dir / bg_data["filename"]
                matches.append(bg_data)
        
        return matches
    
    def search_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """
        Search for backgrounds by keywords.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of matching backgrounds sorted by relevance
        """
        matches = []
        keywords_lower = [k.lower() for k in keywords]
        
        for hash_id, bg_data in self.metadata["backgrounds"].items():
            # Calculate relevance score
            score = 0
            bg_keywords_lower = [k.lower() for k in bg_data.get("keywords", [])]
            bg_theme_lower = bg_data["theme"].lower()
            
            for keyword in keywords_lower:
                # Check keywords
                if keyword in bg_keywords_lower:
                    score += 2
                # Check theme
                if keyword in bg_theme_lower:
                    score += 1
                # Partial matches in keywords
                for bg_keyword in bg_keywords_lower:
                    if keyword in bg_keyword or bg_keyword in keyword:
                        score += 0.5
            
            if score > 0:
                bg_data["hash_id"] = hash_id
                bg_data["path"] = self.cache_dir / bg_data["filename"]
                bg_data["relevance_score"] = score
                matches.append(bg_data)
        
        # Sort by relevance
        matches.sort(key=lambda x: x["relevance_score"], reverse=True)
        return matches
    
    def get_best_match(self, theme: str = None, keywords: List[str] = None) -> Optional[Path]:
        """
        Get the best matching background for given criteria.
        
        Args:
            theme: Preferred theme
            keywords: Keywords to match
            
        Returns:
            Path to best matching video or None
        """
        candidates = []
        
        # Search by theme
        if theme:
            candidates.extend(self.search_by_theme(theme))
        
        # Search by keywords
        if keywords:
            keyword_matches = self.search_by_keywords(keywords)
            # Avoid duplicates
            existing_hashes = {c.get("hash_id") for c in candidates}
            for match in keyword_matches:
                if match.get("hash_id") not in existing_hashes:
                    candidates.append(match)
        
        if not candidates:
            return None
        
        # Return the first match (highest relevance if from keyword search)
        return candidates[0]["path"]
    
    def list_all_backgrounds(self) -> List[Dict]:
        """List all cached backgrounds with their metadata."""
        backgrounds = []
        for hash_id, bg_data in self.metadata["backgrounds"].items():
            bg_data["hash_id"] = hash_id
            bg_data["path"] = self.cache_dir / bg_data["filename"]
            backgrounds.append(bg_data)
        return backgrounds
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        backgrounds = self.list_all_backgrounds()
        
        # Count by theme
        themes = {}
        for bg in backgrounds:
            theme = bg["theme"]
            themes[theme] = themes.get(theme, 0) + 1
        
        # Count by source
        sources = {}
        for bg in backgrounds:
            source = bg.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        # Total size
        total_size = sum(bg.get("file_size", 0) for bg in backgrounds)
        
        return {
            "total_backgrounds": len(backgrounds),
            "total_size_mb": total_size / (1024 * 1024),
            "themes": themes,
            "sources": sources,
            "last_updated": self.metadata.get("last_updated")
        }
    
    def import_from_coverr_cache(self, coverr_dir: Path = None):
        """
        Import existing videos from Coverr cache into centralized cache.
        
        Args:
            coverr_dir: Path to Coverr cache (default: assets/videos/coverr)
        """
        coverr_dir = coverr_dir or Path("assets/videos/coverr")
        
        if not coverr_dir.exists():
            print(f"⚠️ Coverr cache not found: {coverr_dir}")
            return
        
        imported = 0
        for video_file in coverr_dir.glob("*.mp4"):
            # Try to extract info from filename
            # Expected format: theme_keywords_id.mp4 or just id.mp4
            name_parts = video_file.stem.split("_")
            
            # Default theme and keywords
            theme = "general"
            keywords = []
            
            if len(name_parts) > 1:
                theme = name_parts[0]
                keywords = name_parts[1:-1] if len(name_parts) > 2 else []
            
            # Add to cache
            self.add_background(
                video_file,
                theme=theme,
                keywords=keywords,
                source="coverr",
                source_id=name_parts[-1] if name_parts else video_file.stem
            )
            imported += 1
        
        print(f"✅ Imported {imported} videos from Coverr cache")
    
    def cleanup_missing_files(self):
        """Remove metadata entries for files that no longer exist."""
        to_remove = []
        
        for hash_id, bg_data in self.metadata["backgrounds"].items():
            file_path = self.cache_dir / bg_data["filename"]
            if not file_path.exists():
                to_remove.append(hash_id)
                print(f"🗑️ Removing missing: {bg_data['filename']}")
        
        for hash_id in to_remove:
            del self.metadata["backgrounds"][hash_id]
        
        if to_remove:
            self.save_metadata()
            print(f"✅ Cleaned up {len(to_remove)} missing entries")


def main():
    """Demo the background cache manager."""
    
    manager = BackgroundCacheManager()
    
    print("=" * 60)
    print("Background Cache Manager")
    print("=" * 60)
    
    # Show stats
    stats = manager.get_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"  Total backgrounds: {stats['total_backgrounds']}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Themes: {stats['themes']}")
    print(f"  Sources: {stats['sources']}")
    
    # Import from existing Coverr cache if available
    print("\n🔄 Checking for existing Coverr cache...")
    manager.import_from_coverr_cache()
    
    # Example searches
    print("\n🔍 Example Searches:")
    
    # Search by theme
    print("\n1. Search by theme 'tech':")
    tech_results = manager.search_by_theme("tech")
    for result in tech_results[:3]:
        print(f"  - {result['filename']} ({result['theme']})")
    
    # Search by keywords
    print("\n2. Search by keywords ['data', 'visualization']:")
    keyword_results = manager.search_by_keywords(["data", "visualization"])
    for result in keyword_results[:3]:
        print(f"  - {result['filename']} (score: {result['relevance_score']:.1f})")
    
    # Get best match
    print("\n3. Get best match for theme='abstract' keywords=['particle', 'flow']:")
    best = manager.get_best_match(theme="abstract", keywords=["particle", "flow"])
    if best:
        print(f"  Best match: {best.name}")
    else:
        print("  No matches found")
    
    # Cleanup
    print("\n🧹 Cleaning up missing files...")
    manager.cleanup_missing_files()


if __name__ == "__main__":
    main()