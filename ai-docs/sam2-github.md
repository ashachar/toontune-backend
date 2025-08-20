# SAM2 Installation & Usage Guide for MacBook

## Prerequisites

- **Python**: ≥3.10
- **PyTorch**: ≥2.5.1
- **TorchVision**: ≥0.20.1
- **CUDA**: Not required for MacBook (will use MPS/CPU)

## Installation

### 1. Setup Environment

```bash
# Create new conda environment
conda create -n sam2 python=3.10
conda activate sam2

# Install PyTorch for Mac (with MPS support)
pip install torch torchvision torchaudio
```

### 2. Clone and Install SAM2

```bash
# Clone repository
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# Install SAM2
pip install -e .

# For notebooks support (optional)
pip install -e ".[notebooks]"
```

**Note**: CUDA kernel compilation warnings can be ignored on MacBook.

## Download Model Checkpoints

### Quick Download (All Models)
```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

### Individual Downloads
- **Tiny** (39M): `sam2.1_hiera_tiny.pt`
- **Small** (46M): `sam2.1_hiera_small.pt`
- **Base+** (81M): `sam2.1_hiera_base_plus.pt`
- **Large** (224M): `sam2.1_hiera_large.pt`

## Quick Start Usage

### Image Segmentation

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Setup (use 'mps' for Mac GPU or 'cpu')
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"  # Use tiny for faster inference
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Initialize predictor
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

# Run inference
with torch.inference_mode():
    predictor.set_image(your_image)
    masks, scores, logits = predictor.predict(
        point_coords=your_points,  # [[x, y]]
        point_labels=your_labels,   # [1] for positive, [0] for negative
    )
```

### Video Tracking

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

# Setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Initialize video predictor
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

# Process video
with torch.inference_mode():
    state = predictor.init_state(your_video_frames)
    
    # Add prompts
    frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
        state, 
        frame_idx=0,
        obj_id=1,
        points=[[x, y]],
        labels=[1]
    )
    
    # Propagate through video
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # Process each frame's masks
        pass
```

### Using Hugging Face Models

```python
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Direct HF loading
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
```

## MacBook-Specific Optimizations

### 1. Model Selection
- **Recommended**: Use `sam2.1_hiera_tiny` or `sam2.1_hiera_small` for better performance
- Large models may be slow without GPU acceleration

### 2. Device Configuration
```python
# Auto-detect best available device
if torch.backends.mps.is_available():
    device = 'mps'  # Apple Silicon GPU
elif torch.cuda.is_available():
    device = 'cuda'  # Unlikely on Mac
else:
    device = 'cpu'
```

### 3. Memory Management
```python
# For limited RAM, process in batches
with torch.no_grad():
    # Process frames/images one at a time
    for batch in data_loader:
        result = predictor.predict(batch)
        # Clear cache if needed
        if device == 'mps':
            torch.mps.empty_cache()
```

### 4. Performance Tips
- Disable autocast on CPU/MPS: Remove `torch.autocast()` calls
- Use smaller input resolutions when possible
- Consider batch processing for multiple images
- Use `torch.inference_mode()` for inference

## Common Issues & Solutions

### Issue: MPS Backend Errors
```python
# Fallback to CPU if MPS fails
try:
    model.to('mps')
except:
    model.to('cpu')
```

### Issue: Out of Memory
- Switch to smaller model (`tiny` or `small`)
- Reduce input image resolution
- Process in smaller batches

### Issue: Slow Performance
- Ensure you're using native ARM Python (not x86 via Rosetta)
- Check Activity Monitor for thermal throttling
- Consider cloud GPU for production workloads

## Example: Complete Pipeline

```python
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def segment_image(image_path, points):
    """Quick segmentation function for MacBook"""
    
    # Setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    # Load image
    image = np.array(Image.open(image_path))
    
    # Initialize and run
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    
    with torch.inference_mode():
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.ones(len(points))
        )
    
    return masks

# Usage
masks = segment_image('photo.jpg', [[100, 200], [150, 250]])
```

## Resources

- [Official Repository](https://github.com/facebookresearch/sam2)
- [Paper](https://arxiv.org/abs/2408.00714)
- [Interactive Demo](https://sam2.metademolab.com/demo)
- [Colab Notebooks](https://github.com/facebookresearch/sam2/tree/main/notebooks)