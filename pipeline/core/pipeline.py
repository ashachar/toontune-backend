"""
Main Pipeline Orchestrator
===========================

Coordinates all pipeline steps and manages state.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from .config import PipelineConfig
from ..steps.step_1_downsample import DownsampleStep
from ..steps.step_2_transcripts import TranscriptsStep
from ..steps.step_3_scenes import SceneSplitStep
from ..steps.step_4_prompts import PromptsStep
from ..steps.step_5_inference import InferenceStep
from ..steps.step_6_edit_videos import VideoEditStep
from ..steps.step_7_karaoke import KaraokeStep


class UnifiedVideoPipeline:
    """Unified pipeline for video processing."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.video_path = Path(config.video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {config.video_path}")
        
        # Set up directory structure
        self.video_name = self.video_path.stem
        self.base_dir = Path(config.output_base_dir) / self.video_name
        
        # Create all necessary directories
        self.dirs = {
            'base': self.base_dir,
            'transcripts': self.base_dir / 'transcripts',
            'scenes': self.base_dir / 'scenes',
            'scenes_original': self.base_dir / 'scenes' / 'original',
            'scenes_downsampled': self.base_dir / 'scenes' / 'downsampled',
            'scenes_edited': self.base_dir / 'scenes' / 'edited',
            'prompts': self.base_dir / 'prompts',
            'inferences': self.base_dir / 'inferences',
            'metadata': self.base_dir / 'metadata'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Pipeline state
        self.pipeline_state = {
            'start_time': datetime.now().isoformat(),
            'video_path': str(self.video_path),
            'video_name': self.video_name,
            'config': asdict(config),
            'steps_completed': []
        }
        
        # Initialize all steps
        self.step_1 = DownsampleStep(self.pipeline_state, self.dirs, config)
        self.step_2 = TranscriptsStep(self.pipeline_state, self.dirs, config)
        self.step_3 = SceneSplitStep(self.pipeline_state, self.dirs, config)
        self.step_4 = PromptsStep(self.pipeline_state, self.dirs, config)
        self.step_5 = InferenceStep(self.pipeline_state, self.dirs, config)
        self.step_6 = VideoEditStep(self.pipeline_state, self.dirs, config)
        self.step_6.video_name = self.video_name  # Set video name for editing step
        self.step_7 = KaraokeStep(self.pipeline_state, self.dirs, config)
    
    def save_pipeline_state(self):
        """Save the current pipeline state to metadata."""
        state_file = self.dirs['metadata'] / 'pipeline_state.json'
        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2)
        print(f"  ✓ Pipeline state saved to: {state_file}")
    
    def run(self):
        """Run the complete pipeline."""
        print("="*70)
        print("UNIFIED VIDEO PROCESSING PIPELINE")
        print("="*70)
        print(f"Video: {self.video_path}")
        print(f"Output: {self.base_dir}")
        print(f"Mode: {'DRY RUN (no LLM inference)' if self.config.dry_run else 'FULL PROCESSING'}")
        print()
        
        # Step 1: Create downsampled version of full video
        if self.config.downsample_video:
            self.step_1.run(self.video_path, self.video_name)
        else:
            print("\n[SKIPPING] Video downsampling (--no-downsample)")
        
        # Step 2: Generate transcripts
        if self.config.generate_transcripts:
            self.step_2.run(self.video_path)
        else:
            print("\n[SKIPPING] Transcript generation (--no-transcript)")
        
        # Step 3: Split into scenes
        if self.config.split_scenes:
            self.step_3.run(self.video_path)
        else:
            print("\n[SKIPPING] Scene splitting (--no-scenes)")
        
        # Step 4: Generate prompts for each scene
        if self.config.generate_prompts:
            self.step_4.run()
        else:
            print("\n[SKIPPING] Prompt generation (--no-prompts)")
        
        # Step 5: Run LLM inference (or dry run)
        if self.config.run_inference:
            self.step_5.run()
        else:
            print("\n[SKIPPING] LLM inference (--dry-run or --no-inference)")
        
        # Step 6: Edit videos based on inference
        if self.config.edit_videos:
            self.step_6.run()
        else:
            print("\n[SKIPPING] Video editing (--no-editing)")
        
        # Step 7: Generate karaoke captions (optional)
        if self.config.generate_karaoke:
            self.step_7.run()
        else:
            print("\n[SKIPPING] Karaoke caption generation (--no-karaoke)")
        
        # Save final state
        self.save_pipeline_state()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"All outputs saved to: {self.base_dir}")
    
    def get_summary(self):
        """Get a summary of the pipeline execution."""
        summary = {
            "video": self.video_name,
            "base_directory": str(self.base_dir),
            "steps_completed": self.pipeline_state['steps_completed'],
            "directories_created": list(self.dirs.keys()),
            "mode": "DRY_RUN" if self.config.dry_run else "FULL"
        }
        return summary