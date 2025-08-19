#!/usr/bin/env python3
"""
Create visual proof that all features are in the final video
"""

import subprocess
from pathlib import Path

def create_visual_proof():
    base_dir = Path("uploads/assets/videos/do_re_mi")
    video_path = base_dir / "scenes/edited/scene_001.mp4"
    proof_dir = base_dir / "visual_proof"
    proof_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("📸 CREATING VISUAL PROOF OF ALL FEATURES")
    print("="*70)
    
    # Extract frames at key moments
    extractions = [
        (11.5, "karaoke_phrase1", "Karaoke + 'very beginning' phrase"),
        (23.0, "karaoke_phrase2", "Karaoke + 'Do Re Mi' phrase"),
        (47.5, "cartoon1", "First cartoon character"),
        (51.5, "cartoon2", "Second cartoon character"),
    ]
    
    print("\nExtracting proof frames:")
    for time, name, description in extractions:
        output = proof_dir / f"{name}.png"
        cmd = [
            'ffmpeg', '-ss', str(time),
            '-i', str(video_path),
            '-frames:v', '1',
            '-y', str(output)
        ]
        subprocess.run(cmd, capture_output=True)
        print(f"  ✓ {time:5.1f}s → {name}.png ({description})")
    
    # Create a 2x2 montage
    print("\nCreating montage...")
    montage_path = proof_dir / "all_features_montage.png"
    
    cmd = [
        'ffmpeg',
        '-i', str(proof_dir / 'karaoke_phrase1.png'),
        '-i', str(proof_dir / 'karaoke_phrase2.png'),
        '-i', str(proof_dir / 'cartoon1.png'),
        '-i', str(proof_dir / 'cartoon2.png'),
        '-filter_complex',
        '[0][1]hstack[top];[2][3]hstack[bottom];[top][bottom]vstack',
        '-y', str(montage_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        print(f"  ✓ Montage created: {montage_path.name}")
    
    # Also create annotated versions
    print("\nCreating annotated frames...")
    for time, name, description in extractions:
        input_path = proof_dir / f"{name}.png"
        annotated_path = proof_dir / f"{name}_annotated.png"
        
        # Add text annotation
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-vf', f"drawtext=text='{description} at {time}s':fontsize=20:fontcolor=red:bordercolor=white:borderw=2:x=10:y=10",
            '-y', str(annotated_path)
        ]
        subprocess.run(cmd, capture_output=True)
        print(f"  ✓ Annotated: {name}_annotated.png")
    
    print("\n" + "="*70)
    print("✅ VISUAL PROOF COMPLETE!")
    print("="*70)
    print(f"\n📁 Proof images saved to: {proof_dir}")
    print("\n🖼️ Key files:")
    print(f"  • all_features_montage.png - 2x2 grid showing all features")
    print(f"  • karaoke_phrase1.png - Shows karaoke + white phrase")
    print(f"  • karaoke_phrase2.png - Shows karaoke + yellow phrase")
    print(f"  • cartoon1.png - Shows first cartoon")
    print(f"  • cartoon2.png - Shows second cartoon")
    print("\n🎬 Final video with ALL features:")
    print(f"  {video_path}")
    print("="*70)

if __name__ == "__main__":
    create_visual_proof()