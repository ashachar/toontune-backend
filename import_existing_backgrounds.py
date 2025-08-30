#!/usr/bin/env python3
"""
Import existing background videos into the centralized cache with proper metadata.
"""

from pathlib import Path
from utils.video.background.background_cache_manager import BackgroundCacheManager


def import_project_backgrounds():
    """Import backgrounds from project folders into centralized cache."""
    
    manager = BackgroundCacheManager()
    
    # Define known backgrounds with their themes based on content
    known_backgrounds = [
        {
            "pattern": "*_background_7_2_9_5_*.mp4",
            "theme": "mathematics",
            "keywords": ["math", "geometry", "cube", "drawing", "education"],
            "source": "coverr"
        },
        {
            "pattern": "*_background_27_9_31_6_*.mp4", 
            "theme": "ai_visualization",
            "keywords": ["ai", "artificial", "intelligence", "art", "generation"],
            "source": "coverr"
        },
        {
            "pattern": "*_background_112_7_127_2_*.mp4",
            "theme": "data_analytics",
            "keywords": ["data", "notebook", "analytics", "writing", "research"],
            "source": "coverr"
        },
        {
            "pattern": "*_background_145_2_152_0_*.mp4",
            "theme": "financial_trends",
            "keywords": ["crypto", "trends", "finance", "trading", "charts"],
            "source": "coverr"
        },
        {
            "pattern": "*_background_65_5_70_8_*.mp4",
            "theme": "technology",
            "keywords": ["tech", "digital", "innovation", "future"],
            "source": "coverr"
        },
        {
            "pattern": "*_background_0_0_5_0_*.mp4",
            "theme": "abstract_intro",
            "keywords": ["abstract", "opening", "introduction", "minimal"],
            "source": "coverr"
        }
    ]
    
    # Search in project folders
    projects_dir = Path("uploads/assets/videos")
    imported_count = 0
    
    print("🔍 Searching for existing backgrounds in project folders...")
    
    for project_dir in projects_dir.glob("*/"):
        if not project_dir.is_dir():
            continue
        
        print(f"\n📁 Checking {project_dir.name}/")
        
        for bg_info in known_backgrounds:
            for video_file in project_dir.glob(bg_info["pattern"]):
                print(f"  Found: {video_file.name}")
                
                # Extract source ID from filename
                parts = video_file.stem.split("_")
                source_id = parts[-1] if parts else None
                
                # Add to centralized cache
                cached_path = manager.add_background(
                    video_file,
                    theme=bg_info["theme"],
                    keywords=bg_info["keywords"],
                    source=bg_info["source"],
                    source_id=source_id
                )
                
                imported_count += 1
    
    print(f"\n✅ Imported {imported_count} backgrounds into centralized cache")
    
    # Show updated stats
    stats = manager.get_stats()
    print(f"\n📊 Updated Cache Statistics:")
    print(f"  Total backgrounds: {stats['total_backgrounds']}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Themes: {dict(stats['themes'])}")
    
    # Test search
    print("\n🔍 Testing search capabilities:")
    
    # Search for math-related backgrounds
    print("\n1. Searching for 'mathematics' theme:")
    math_results = manager.search_by_theme("mathematics")
    for result in math_results:
        print(f"  ✓ {result['filename']}")
    
    # Search by AI keywords
    print("\n2. Searching for AI-related keywords:")
    ai_results = manager.search_by_keywords(["ai", "artificial", "intelligence"])
    for result in ai_results[:3]:
        print(f"  ✓ {result['filename']} (relevance: {result['relevance_score']:.1f})")
    
    # Search for data/trends
    print("\n3. Searching for data/trends keywords:")
    data_results = manager.search_by_keywords(["data", "trends", "analytics"])
    for result in data_results[:3]:
        print(f"  ✓ {result['filename']} (relevance: {result['relevance_score']:.1f})")


def main():
    """Import and organize existing backgrounds."""
    print("=" * 60)
    print("Import Existing Backgrounds to Centralized Cache")
    print("=" * 60)
    
    import_project_backgrounds()
    
    print("\n" + "=" * 60)
    print("✅ Import complete! Backgrounds are now cached at:")
    print("   uploads/assets/backgrounds/")
    print("=" * 60)


if __name__ == "__main__":
    main()