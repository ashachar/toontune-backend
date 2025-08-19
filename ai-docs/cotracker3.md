# CoTracker Implementation Guide for M3 Max

## Complete Setup & Development Instructions

### System Requirements
- macOS with Apple Silicon (M3 Max)
- Python 3.8+
- 48GB RAM (you have this ✓)
- ~10GB free disk space

### Step 1: Environment Setup

```bash
# Create virtual environment
python3 -m venv cotracker_env
source cotracker_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Core Dependencies

```bash
# Install PyTorch with MPS support (Apple Silicon)
pip install torch torchvision torchaudio

# Install CoTracker and dependencies
pip install git+https://github.com/facebookresearch/co-tracker.git

# Install additional required packages
pip install imageio[ffmpeg] matplotlib flow_vis tqdm tensorboard opencv-python
pip install gradio spaces  # For demo interface
```

### Step 3: Verify MPS Support

```python
# test_mps.py
import torch

if torch.backends.mps.is_available():
    print("✅ MPS (Metal Performance Shaders) is available!")
    device = torch.device("mps")
    print(f"Device: {device}")
    
    # Test tensor operation
    x = torch.randn(1, 3, 224, 224).to(device)
    print(f"Tensor shape on MPS: {x.shape}")
else:
    print("❌ MPS not available, using CPU")
```

### Step 4: Basic Video Tracking Implementation

```python
# track_video.py
import torch
import imageio.v3 as iio
import numpy as np
from pathlib import Path

class VideoTracker:
    def __init__(self, model_type="cotracker3_online"):
        """Initialize CoTracker for M3 Max"""
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = torch.hub.load(
            "facebookresearch/co-tracker", 
            model_type
        ).to(self.device)
        
    def process_video(self, video_path, grid_size=5, output_path="output.mp4"):
        """Process video with point tracking"""
        
        # Load video
        print(f"Loading video: {video_path}")
        frames = iio.imread(video_path, plugin="FFMPEG")
        
        # Convert to tensor
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float()
        video = video.to(self.device)
        
        print(f"Video shape: {video.shape}")
        print(f"Duration: {video.shape[1]/30:.2f} seconds at 30fps")
        
        # Track points
        print(f"Tracking with grid_size={grid_size}")
        
        # Initialize
        self.model(video_chunk=video, is_first_step=True, grid_size=grid_size)
        
        # Process video in chunks
        all_tracks = []
        all_visibility = []
        
        for ind in range(0, video.shape[1] - self.model.step, self.model.step):
            pred_tracks, pred_visibility = self.model(
                video_chunk=video[:, ind : ind + self.model.step * 2]
            )
            all_tracks.append(pred_tracks)
            all_visibility.append(pred_visibility)
        
        # Concatenate results
        final_tracks = torch.cat(all_tracks, dim=1)
        final_visibility = torch.cat(all_visibility, dim=1)
        
        print(f"Tracked {final_tracks.shape[2]} points")
        
        # Visualize results
        self.visualize_tracks(video, final_tracks, final_visibility, output_path)
        
        return final_tracks, final_visibility
    
    def visualize_tracks(self, video, tracks, visibility, output_path):
        """Save visualization video"""
        from cotracker.utils.visualizer import Visualizer
        
        vis = Visualizer(
            save_dir=str(Path(output_path).parent),
            pad_value=120,
            linewidth=3
        )
        vis.visualize(video, tracks, visibility)
        print(f"Saved visualization to: {output_path}")

    def track_specific_points(self, video_path, points):
        """Track specific points instead of grid"""
        
        # Load video
        frames = iio.imread(video_path, plugin="FFMPEG")
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float()
        video = video.to(self.device)
        
        # Create queries (frame_idx, x, y)
        queries = torch.tensor(points).float().to(self.device)
        
        # Track
        pred_tracks, pred_visibility = self.model(
            video, 
            queries=queries[None]
        )
        
        return pred_tracks, pred_visibility
```

### Step 5: Optimized Script for Low-Resolution Videos

```python
# fast_track_lowres.py
import torch
import imageio.v3 as iio
import time
import cv2

def fast_track_480p(video_path, target_width=640):
    """
    Optimized tracking for low-res videos on M3 Max
    Perfect for 5-second 480p videos
    """
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load and resize video if needed
    print("Loading video...")
    frames = iio.imread(video_path, plugin="FFMPEG")
    
    # Resize if larger than target
    h, w = frames.shape[1:3]
    if w > target_width:
        scale = target_width / w
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (new_w, new_h))
            resized_frames.append(resized)
        frames = np.array(resized_frames)
        print(f"Resized to: {new_w}x{new_h}")
    
    # Convert to tensor
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
    
    # Load model (online mode for speed)
    print("Loading CoTracker model...")
    model = torch.hub.load(
        "facebookresearch/co-tracker", 
        "cotracker3_online"
    ).to(device)
    
    # Track with small grid for speed
    grid_size = 3  # Small grid for speed
    
    print(f"Tracking {grid_size}x{grid_size} grid...")
    start_time = time.time()
    
    # Initialize
    model(video_chunk=video, is_first_step=True, grid_size=grid_size)
    
    # Process
    tracks_list = []
    vis_list = []
    
    for ind in range(0, video.shape[1] - model.step, model.step):
        tracks, visibility = model(
            video_chunk=video[:, ind : ind + model.step * 2]
        )
        tracks_list.append(tracks)
        vis_list.append(visibility)
    
    # Combine results
    all_tracks = torch.cat(tracks_list, dim=1)
    all_visibility = torch.cat(vis_list, dim=1)
    
    elapsed = time.time() - start_time
    fps = video.shape[1] / elapsed
    
    print(f"✅ Tracking complete!")
    print(f"   Time: {elapsed:.2f} seconds")
    print(f"   Speed: {fps:.1f} fps")
    print(f"   Points tracked: {all_tracks.shape[2]}")
    
    return all_tracks, all_visibility, video

# Usage example
if __name__ == "__main__":
    tracks, visibility, video = fast_track_480p("my_video.mp4")
```

### Step 6: Interactive Demo with Gradio

```python
# demo_app.py
import gradio as gr
import torch
import imageio.v3 as iio
import numpy as np
from cotracker.utils.visualizer import Visualizer
import tempfile

def track_video_gradio(video_file, grid_size, tracking_mode):
    """Gradio interface for CoTracker"""
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load video
    frames = iio.imread(video_file, plugin="FFMPEG")
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
    
    # Select model based on mode
    model_name = "cotracker3_online" if tracking_mode == "Fast (Online)" else "cotracker3_offline"
    model = torch.hub.load("facebookresearch/co-tracker", model_name).to(device)
    
    # Track
    if tracking_mode == "Fast (Online)":
        model(video_chunk=video, is_first_step=True, grid_size=grid_size)
        tracks, vis = [], []
        for ind in range(0, video.shape[1] - model.step, model.step):
            t, v = model(video_chunk=video[:, ind : ind + model.step * 2])
            tracks.append(t)
            vis.append(v)
        pred_tracks = torch.cat(tracks, dim=1)
        pred_visibility = torch.cat(vis, dim=1)
    else:
        pred_tracks, pred_visibility = model(video, grid_size=grid_size)
    
    # Visualize
    with tempfile.TemporaryDirectory() as tmpdir:
        vis = Visualizer(save_dir=tmpdir, pad_value=120, linewidth=3)
        vis.visualize(video, pred_tracks, pred_visibility)
        output_path = f"{tmpdir}/video.mp4"
        
        return output_path, f"Tracked {pred_tracks.shape[2]} points successfully!"

# Create Gradio interface
iface = gr.Interface(
    fn=track_video_gradio,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Slider(3, 10, value=5, step=1, label="Grid Size"),
        gr.Radio(["Fast (Online)", "Accurate (Offline)"], value="Fast (Online)", label="Tracking Mode")
    ],
    outputs=[
        gr.Video(label="Tracked Video"),
        gr.Textbox(label="Status")
    ],
    title="CoTracker on M3 Max",
    description="Track any point in your video! Optimized for Apple Silicon."
)

if __name__ == "__main__":
    iface.launch(share=True)
```

### Step 7: Testing & Benchmarking

```python
# benchmark.py
import torch
import time
import numpy as np

def benchmark_cotracker():
    """Benchmark CoTracker on M3 Max"""
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test different configurations
    configs = [
        {"resolution": (360, 640), "frames": 150, "grid": 3},  # 360p, 5 sec
        {"resolution": (480, 854), "frames": 150, "grid": 5},  # 480p, 5 sec
        {"resolution": (720, 1280), "frames": 150, "grid": 10}, # 720p, 5 sec
    ]
    
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    
    for config in configs:
        h, w = config["resolution"]
        frames = config["frames"]
        grid = config["grid"]
        
        # Create dummy video
        video = torch.randn(1, frames, 3, h, w).to(device)
        
        # Warm up
        model(video_chunk=video[:, :32], is_first_step=True, grid_size=grid)
        
        # Benchmark
        torch.mps.synchronize() if device == 'mps' else None
        start = time.time()
        
        model(video_chunk=video[:, :32], is_first_step=True, grid_size=grid)
        for ind in range(0, frames - model.step, model.step):
            model(video_chunk=video[:, ind : ind + model.step * 2])
        
        torch.mps.synchronize() if device == 'mps' else None
        elapsed = time.time() - start
        
        fps = frames / elapsed
        print(f"\n{w}x{h}, {frames} frames, {grid}x{grid} grid:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.1f}")
        print(f"  Points: {grid * grid}")

if __name__ == "__main__":
    benchmark_cotracker()
```

### Step 8: Troubleshooting

```bash
# If ffmpeg issues:
brew install ffmpeg

# If MPS errors:
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Memory issues:
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

# Check installation:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Quick Start Command

```bash
# One-liner to get started
curl -fsSL https://raw.githubusercontent.com/facebookresearch/co-tracker/main/demo.py | python - --grid_size 5
```

### Performance Tips for M3 Max

1. **Use Online Mode** - 2-3x faster than offline
2. **Reduce Grid Size** - grid_size=3 is 4x faster than grid_size=6
3. **Lower Resolution** - 360p is 3x faster than 720p
4. **Batch Processing** - Process multiple videos in sequence, not parallel

### Expected Performance on Your M3 Max

| Video Spec | Mode | Grid | Expected Time |
|------------|------|------|---------------|
| 480p, 5 sec | Online | 3×3 | ~2-3 seconds |
| 480p, 5 sec | Online | 5×5 | ~3-5 seconds |
| 360p, 5 sec | Online | 3×3 | ~1-2 seconds |
| 720p, 5 sec | Online | 5×5 | ~8-12 seconds |

---

**Ready to track! 🚀** Your M3 Max will handle this beautifully for rapid prototyping.