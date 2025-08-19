#!/usr/bin/env python3
"""
Single-pass pipeline - Prepare all overlays, then combine ONCE
===============================================================

Instead of re-encoding video multiple times, we:
1. Prepare karaoke subtitle file
2. Prepare phrase overlay definitions  
3. Prepare cartoon overlay definitions
4. Run ONE FFmpeg command that applies everything
"""

import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


class SinglePassPipeline:
    """Pipeline that prepares all overlays then combines in one pass."""
    
    def __init__(self, video_dir):
        self.video_dir = Path(video_dir)
        self.scenes_dir = self.video_dir / "scenes"
        self.edited_dir = self.scenes_dir / "edited"
        self.inferences_dir = self.video_dir / "inferences"
        self.transcripts_dir = self.video_dir / "transcripts"
        
        # Overlay definitions to build up
        self.overlays = {
            'karaoke': None,
            'phrases': [],
            'cartoons': []
        }
        
    def prepare_karaoke(self, scene_num=1):
        """Prepare karaoke subtitle file (don't apply yet)."""
        print("\n📝 PREPARING KARAOKE...")
        
        # Load transcript
        words_file = self.transcripts_dir / 'transcript_words.json'
        if not words_file.exists():
            print("  ❌ No transcript found")
            return False
            
        with open(words_file) as f:
            words = json.load(f)['words'][:50]  # First 50 words for testing
        
        # Create simple ASS subtitle file
        ass_path = self.edited_dir / f"scene_{scene_num:03d}_karaoke_prepared.ass"
        
        # Simple subtitle content
        ass_content = """[Script Info]
Title: Karaoke
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Group words into simple lines
        current_time = 0.0
        for i in range(0, len(words), 6):  # 6 words per line
            chunk = words[i:i+6]
            if not chunk:
                break
                
            text = ' '.join(w['word'] for w in chunk).upper()
            start = chunk[0].get('start', current_time)
            end = chunk[-1].get('end', start + 3)
            
            # Format times
            def fmt_time(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = t % 60
                return f"{h}:{m:02d}:{s:05.2f}"
            
            ass_content += f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},Default,,0,0,0,,{text}\n"
            current_time = end
        
        with open(ass_path, 'w') as f:
            f.write(ass_content)
        
        self.overlays['karaoke'] = ass_path
        print(f"  ✅ Karaoke subtitle prepared: {ass_path.name}")
        return True
        
    def prepare_phrases(self, scene_num=1):
        """Prepare phrase overlay definitions (don't apply yet)."""
        print("\n📝 PREPARING PHRASES...")
        
        # Load inference
        inference_file = self.inferences_dir / f"scene_{scene_num:03d}_inference.json"
        if not inference_file.exists():
            print("  ❌ No inference found")
            return False
            
        with open(inference_file) as f:
            inference = json.load(f)
        
        phrases = inference['scenes'][0].get('key_phrases', [])
        
        for p in phrases:
            self.overlays['phrases'].append({
                'text': p['phrase'],
                'x': p.get('top_left_pixels', {}).get('x', 100),
                'y': p.get('top_left_pixels', {}).get('y', 100),
                'start': float(p['start_seconds']),
                'end': float(p['start_seconds']) + float(p['duration_seconds']),
                'color': 'yellow' if 'playful' in p.get('style', '') else 'white',
                'size': 30
            })
            
        print(f"  ✅ Prepared {len(phrases)} phrase overlays")
        return True
        
    def prepare_cartoons(self, scene_num=1):
        """Prepare cartoon overlay definitions (don't apply yet)."""
        print("\n📝 PREPARING CARTOONS...")
        
        # Load inference
        inference_file = self.inferences_dir / f"scene_{scene_num:03d}_inference.json"
        if not inference_file.exists():
            return False
            
        with open(inference_file) as f:
            inference = json.load(f)
        
        cartoons = inference['scenes'][0].get('cartoon_characters', [])
        
        # Find cartoon asset
        cartoon_asset = None
        for asset_dir in [Path("cartoon-test"), Path("uploads/assets/batch_images_transparent_bg")]:
            if asset_dir.exists():
                spring = asset_dir / "spring.png"
                if spring.exists():
                    cartoon_asset = spring
                    break
        
        if not cartoon_asset:
            print("  ❌ No cartoon assets found")
            return False
        
        for i, c in enumerate(cartoons):
            self.overlays['cartoons'].append({
                'asset': cartoon_asset,
                'x': 200 + (i * 250),
                'y': 250,
                'width': 80,
                'height': 100,
                'start': float(c['start_seconds']),
                'end': float(c['start_seconds']) + float(c.get('duration_seconds', 3))
            })
        
        print(f"  ✅ Prepared {len(cartoons)} cartoon overlays")
        return True
        
    def combine_all_overlays(self, scene_num=1):
        """Combine all overlays in a SINGLE FFmpeg pass."""
        print("\n🎬 COMBINING ALL OVERLAYS IN SINGLE PASS...")
        
        input_video = self.scenes_dir / "original" / f"scene_{scene_num:03d}.mp4"
        output_video = self.edited_dir / f"scene_{scene_num:03d}.mp4"
        
        if not input_video.exists():
            print(f"  ❌ No input video found: {input_video}")
            return False
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-i', str(input_video)]
        
        # Add cartoon images as inputs
        cartoon_inputs = []
        for c in self.overlays['cartoons']:
            if c['asset'] not in cartoon_inputs:
                cmd.extend(['-i', str(c['asset'])])
                cartoon_inputs.append(c['asset'])
        
        # Build complex filter
        filters = []
        
        # 1. Add karaoke subtitles first (base layer)
        current_stream = "0:v"
        if self.overlays['karaoke']:
            filters.append(f"[{current_stream}]subtitles={self.overlays['karaoke']}[with_karaoke]")
            current_stream = "with_karaoke"
        
        # 2. Add phrase overlays
        for i, p in enumerate(self.overlays['phrases']):
            text = p['text'].replace("'", "\\'").replace(":", "\\:")
            next_stream = f"with_phrase{i}"
            
            filters.append(
                f"[{current_stream}]drawtext="
                f"text='{text}':"
                f"fontsize={p['size']}:"
                f"fontcolor={p['color']}:"
                f"bordercolor=black:borderw=2:"
                f"x={p['x']}:y={p['y']}:"
                f"enable='between(t,{p['start']},{p['end']})'"
                f"[{next_stream}]"
            )
            current_stream = next_stream
        
        # 3. Add cartoon overlays
        for i, c in enumerate(self.overlays['cartoons']):
            # Find input index for this cartoon's asset
            input_idx = cartoon_inputs.index(c['asset']) + 1
            
            # Scale cartoon
            filters.append(f"[{input_idx}:v]scale={c['width']}:{c['height']}[cartoon{i}]")
            
            # Overlay it
            next_stream = f"with_cartoon{i}" if i < len(self.overlays['cartoons'])-1 else "final"
            filters.append(
                f"[{current_stream}][cartoon{i}]overlay="
                f"x={c['x']}:y={c['y']}:"
                f"enable='between(t,{c['start']},{c['end']})'"
                f"[{next_stream}]"
            )
            current_stream = next_stream
        
        # If no filters, just copy
        if not filters:
            print("  ⚠️ No overlays to apply")
            shutil.copy(input_video, output_video)
            return True
        
        # Join filters
        filter_complex = ";".join(filters)
        
        # Complete command
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', f'[{current_stream}]',  # Use final stream
            '-map', '0:a?',  # Copy audio
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            '-y', str(output_video)
        ])
        
        print(f"  Running single-pass FFmpeg...")
        print(f"  Applying:")
        print(f"    • Karaoke: {self.overlays['karaoke'] is not None}")
        print(f"    • Phrases: {len(self.overlays['phrases'])}")
        print(f"    • Cartoons: {len(self.overlays['cartoons'])}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            size_mb = output_video.stat().st_size / (1024*1024)
            print(f"  ✅ SUCCESS! Output: {output_video.name} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ❌ FFmpeg failed: {result.stderr[:500]}")
            return False
            
    def run(self):
        """Run the complete single-pass pipeline."""
        print("="*70)
        print("🚀 SINGLE-PASS PIPELINE")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
        
        # Prepare all overlays
        self.prepare_karaoke()
        self.prepare_phrases()
        self.prepare_cartoons()
        
        # Combine everything in ONE pass
        if self.combine_all_overlays():
            print("\n" + "="*70)
            print("✨ COMPLETE!")
            print("="*70)
            print(f"Final video: {self.edited_dir / 'scene_001.mp4'}")
            print("All features applied in a single encoding pass!")
            print("="*70)
        else:
            print("\n❌ Pipeline failed")


def extract_verification_frames():
    """Extract frames to verify all features are present."""
    from pathlib import Path
    import subprocess
    
    video = Path("uploads/assets/videos/do_re_mi/scenes/edited/scene_001.mp4")
    verify_dir = Path("uploads/assets/videos/do_re_mi/single_pass_verify")
    verify_dir.mkdir(exist_ok=True)
    
    print("\n📸 Extracting verification frames...")
    
    test_times = [
        (8.0, "karaoke"),
        (11.5, "phrase1"),
        (23.0, "phrase2"),
        (47.5, "cartoon1"),
        (51.5, "cartoon2")
    ]
    
    for time, name in test_times:
        frame = verify_dir / f"{name}_{time}s.png"
        cmd = ['ffmpeg', '-ss', str(time), '-i', str(video),
               '-frames:v', '1', '-y', str(frame)]
        subprocess.run(cmd, capture_output=True)
        print(f"  ✓ {time}s → {name}")


if __name__ == "__main__":
    pipeline = SinglePassPipeline("uploads/assets/videos/do_re_mi")
    pipeline.run()
    extract_verification_frames()