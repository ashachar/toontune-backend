#!/usr/bin/env python3
"""
Analysis of burning letter VFX assets scraped from productioncrate.com
Documents the types of effects available for photorealistic text burning animations.
"""

import os
from pathlib import Path

def analyze_vfx_results():
    """Analyze the scraped VFX assets and provide recommendations."""
    
    screenshots_dir = Path("vfx_screenshots")
    
    if not screenshots_dir.exists():
        print("No screenshots directory found. Run scrape_burning_vfx.py first.")
        return
    
    # Count files by type
    all_files = list(screenshots_dir.glob("*.png"))
    vfx_items = [f for f in all_files if f.name.startswith("vfx_item_")]
    video_previews = [f for f in all_files if f.name.startswith("video_preview_")]
    overview_shots = [f for f in all_files if "full_page" in f.name or "viewport" in f.name or "scrolled" in f.name]
    
    print("=== BURNING LETTER VFX ANALYSIS ===\n")
    
    print(f"📊 SCRAPING RESULTS:")
    print(f"   • Total screenshots: {len(all_files)}")
    print(f"   • Individual VFX items: {len(vfx_items)}")
    print(f"   • Video previews: {len(video_previews)}")
    print(f"   • Overview pages: {len(overview_shots)}")
    print()
    
    print("🔥 VFX ASSET CATEGORIES IDENTIFIED:")
    print("   Based on the full page screenshot, ProductionCrate offers:")
    print()
    
    # Categories observed from the screenshot
    categories = [
        {
            "name": "Smoke/Steam Effects", 
            "description": "Wispy smoke trails and steam effects",
            "best_for": "Text dissolving/fading effects, mysterious reveals"
        },
        {
            "name": "Fire Plumes", 
            "description": "Tall flame columns and fire bursts",
            "best_for": "Dramatic text burning, intense fire letters"
        },
        {
            "name": "Explosion/Combustion", 
            "description": "Explosive burning effects with debris",
            "best_for": "Text destruction, dramatic reveals"
        },
        {
            "name": "Ember/Ash Effects", 
            "description": "Glowing particles and floating ash",
            "best_for": "Subtle text burning, elegant transitions"
        },
        {
            "name": "Dense Smoke Clouds", 
            "description": "Heavy, billowing smoke formations",
            "best_for": "Text obscuring/revealing, atmospheric effects"
        }
    ]
    
    for i, category in enumerate(categories, 1):
        print(f"   {i}. {category['name']}")
        print(f"      • Description: {category['description']}")
        print(f"      • Best for: {category['best_for']}")
        print()
    
    print("⭐ RECOMMENDATIONS FOR PHOTOREALISTIC TEXT BURNING:")
    print()
    
    recommendations = [
        "🎯 **Fire Plume Assets** - Most suitable for letter-by-letter burning",
        "🎨 **Ember/Particle Effects** - Perfect for glowing letter edges", 
        "💨 **Light Smoke Trails** - Ideal for post-burn wisps",
        "💥 **Small Explosions** - Great for dramatic letter ignition",
        "🌫️  **Ash Effects** - Perfect for letter dissolution"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    print()
    
    print("🛠️  IMPLEMENTATION STRATEGY:")
    print("   1. Layer multiple effects for realism:")
    print("      • Base fire effect (main burn)")
    print("      • Ember particles (glowing edges)")
    print("      • Smoke trail (aftermath)")
    print("      • Ash particles (dissolution)")
    print()
    print("   2. Timing sequence:")
    print("      • Ember glow appears first")
    print("      • Fire ignites from ignition point")
    print("      • Smoke develops during burn")
    print("      • Ash particles as text disappears")
    print()
    print("   3. Blending considerations:")
    print("      • Use 'screen' or 'add' blend modes for fire")
    print("      • Use 'multiply' for smoke shadows")
    print("      • Use 'normal' with alpha for particles")
    print()
    
    print("📁 FILE STRUCTURE:")
    print(f"   All screenshots saved to: {screenshots_dir.absolute()}")
    print("   Key files to review:")
    print("   • burning_letters_full_page.png - Complete asset overview")
    print("   • vfx_item_XX_XX.png - Individual VFX cards (20 samples)")
    print("   • video_preview_XX.png - Animated preview thumbnails")
    print()
    
    print("🚀 NEXT STEPS:")
    print("   1. Review individual VFX cards to select specific assets")
    print("   2. Download chosen VFX files from ProductionCrate")
    print("   3. Test compositing with sample text")
    print("   4. Create burning text animation pipeline")
    print("   5. Implement timing and sequencing logic")

if __name__ == "__main__":
    analyze_vfx_results()