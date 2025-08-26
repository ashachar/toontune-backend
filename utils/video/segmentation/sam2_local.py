#!/usr/bin/env python3
"""
SAM2 Local Implementation for MacBook
Implements Segment Anything Model 2 with MPS/CPU support
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add SAM2 to path if installed locally
SAM2_PATH = Path.home() / "sam2"  # Adjust if installed elsewhere
if SAM2_PATH.exists():
    sys.path.insert(0, str(SAM2_PATH))

try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("SAM2 not found. Please install it first:")
    print("git clone https://github.com/facebookresearch/sam2.git")
    print("cd sam2 && pip install -e .")
    sys.exit(1)


class SAM2Local:
    """Local SAM2 implementation optimized for MacBook"""
    
    def __init__(
        self,
        model_size: str = "tiny",
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize SAM2 model
        
        Args:
            model_size: Model size - 'tiny', 'small', 'base_plus', or 'large'
            checkpoint_dir: Path to checkpoint directory
            device: Device to use ('mps', 'cpu', or None for auto-detect)
        """
        self.model_size = model_size
        self.device = self._setup_device(device)
        self.checkpoint_dir = self._setup_checkpoint_dir(checkpoint_dir)
        self.predictor = None
        self.video_predictor = None
        
        # Model configurations
        self.model_configs = {
            'tiny': ('sam2.1_hiera_tiny.pt', 'sam2.1/sam2.1_hiera_t.yaml'),
            'small': ('sam2.1_hiera_small.pt', 'sam2.1/sam2.1_hiera_s.yaml'),
            'base_plus': ('sam2.1_hiera_base_plus.pt', 'sam2.1/sam2.1_hiera_b+.yaml'),
            'large': ('sam2.1_hiera_large.pt', 'sam2.1/sam2.1_hiera_l.yaml')
        }
        
        # Initialize model
        self._init_model()
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Auto-detect best available device"""
        if device:
            return device
            
        if torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon GPU)")
            return 'mps'
        elif torch.cuda.is_available():
            print("Using CUDA GPU")
            return 'cuda'
        else:
            print("Using CPU")
            return 'cpu'
    
    def _setup_checkpoint_dir(self, checkpoint_dir: Optional[str]) -> Path:
        """Setup checkpoint directory"""
        if checkpoint_dir:
            return Path(checkpoint_dir)
        
        # Check common locations
        possible_dirs = [
            Path.home() / "sam2" / "checkpoints",
            Path("./checkpoints"),
            Path("../sam2/checkpoints"),
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                print(f"Found checkpoints at: {dir_path}")
                return dir_path
        
        print("Warning: Checkpoint directory not found. Please specify path.")
        return Path("./checkpoints")
    
    def _init_model(self):
        """Initialize SAM2 model"""
        if self.model_size not in self.model_configs:
            raise ValueError(f"Invalid model size. Choose from: {list(self.model_configs.keys())}")
        
        checkpoint_file, config_file = self.model_configs[self.model_size]
        checkpoint_path = self.checkpoint_dir / checkpoint_file
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found at {checkpoint_path}")
            print(f"Download it from: https://github.com/facebookresearch/sam2")
            return
        
        print(f"Loading {self.model_size} model from {checkpoint_path}")
        
        try:
            # Build image predictor - config_file is relative to sam2 package
            sam2_model = build_sam2(config_file, str(checkpoint_path), device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            if self.device == 'mps':
                print("Falling back to CPU...")
                self.device = 'cpu'
                sam2_model = build_sam2(config_file, str(checkpoint_path), device=self.device)
                self.predictor = SAM2ImagePredictor(sam2_model)
    
    def segment_image(
        self,
        image: np.ndarray,
        points: Optional[List[List[float]]] = None,
        labels: Optional[List[int]] = None,
        boxes: Optional[List[List[float]]] = None,
        multimask_output: bool = True
    ) -> Dict[str, Any]:
        """
        Segment an image using SAM2
        
        Args:
            image: Input image as numpy array (H, W, 3)
            points: List of [x, y] coordinates for point prompts
            labels: List of labels (1 for positive, 0 for negative)
            boxes: List of [x1, y1, x2, y2] bounding boxes
            multimask_output: Whether to return multiple masks
        
        Returns:
            Dictionary with masks, scores, and logits
        """
        if self.predictor is None:
            raise RuntimeError("Model not initialized")
        
        with torch.inference_mode():
            self.predictor.set_image(image)
            
            # Prepare inputs
            point_coords = np.array(points) if points else None
            point_labels = np.array(labels) if labels else None
            box = np.array(boxes[0]) if boxes else None
            
            # Generate masks
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output
            )
            
            return {
                'masks': masks,
                'scores': scores,
                'logits': logits
            }
    
    def automatic_segmentation(
        self,
        image: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        crop_n_layers: int = 0,
        min_mask_region_area: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate masks for the entire image
        
        Args:
            image: Input image
            points_per_side: Number of points to sample per side
            pred_iou_thresh: IoU threshold for filtering
            stability_score_thresh: Stability score threshold
            crop_n_layers: Number of crop layers
            min_mask_region_area: Minimum mask region area
        
        Returns:
            List of mask dictionaries
        """
        if self.predictor is None:
            raise RuntimeError("Model not initialized")
        
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        # Create automatic mask generator
        mask_generator = SAM2AutomaticMaskGenerator(
            self.predictor.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            min_mask_region_area=min_mask_region_area
        )
        
        with torch.inference_mode():
            masks = mask_generator.generate(image)
        
        return masks
    
    def visualize_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        scores: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize masks with colors
        
        Args:
            image: Original image
            masks: Binary masks (N, H, W) or (H, W)
            scores: Optional scores for each mask
            save_path: Optional path to save visualization
            alpha: Transparency for overlay
        
        Returns:
            Visualization as numpy array
        """
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, ...]
        
        # Create figure
        fig, axes = plt.subplots(1, len(masks) + 1, figsize=(15, 5))
        
        # Show original
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Colors for masks
        colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
        
        # Show each mask
        for idx, (mask, color) in enumerate(zip(masks, colors)):
            ax = axes[idx + 1]
            
            # Create colored overlay
            overlay = image.copy()
            mask_overlay = np.zeros_like(image)
            mask_overlay[mask > 0] = (color[:3] * 255).astype(np.uint8)
            
            # Blend
            result = cv2.addWeighted(overlay, 1 - alpha, mask_overlay, alpha, 0)
            
            ax.imshow(result)
            title = f"Mask {idx + 1}"
            if scores is not None:
                title += f" (score: {scores[idx]:.3f})"
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        # Convert figure to array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return vis_array
    
    def segment_video_frame(
        self,
        video_path: str,
        frame_idx: int = 0,
        points: Optional[List[List[float]]] = None,
        automatic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract and segment a single frame from video
        
        Args:
            video_path: Path to video file
            frame_idx: Frame index to extract
            points: Optional points for segmentation
            automatic: Use automatic segmentation
        
        Returns:
            Tuple of (frame, segmentation_results)
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Segment
        if automatic:
            results = self.automatic_segmentation(frame_rgb)
        else:
            # Use center point if no points provided
            if points is None:
                h, w = frame_rgb.shape[:2]
                points = [[w // 2, h // 2]]
                labels = [1]
            else:
                labels = [1] * len(points)
            
            results = self.segment_image(frame_rgb, points, labels)
        
        return frame_rgb, results


def main():
    """Test SAM2 on a video frame"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM2 Local Segmentation")
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("--frame", type=int, default=30, help="Frame index for video")
    parser.add_argument("--model", default="tiny", help="Model size")
    parser.add_argument("--automatic", action="store_true", help="Use automatic segmentation")
    parser.add_argument("--points", nargs="+", type=float, help="Points as x1 y1 x2 y2...")
    parser.add_argument("--output", help="Output path for visualization")
    
    args = parser.parse_args()
    
    # Initialize SAM2
    sam2 = SAM2Local(model_size=args.model)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video input
        print(f"Processing frame {args.frame} from {input_path}")
        frame, results = sam2.segment_video_frame(
            str(input_path),
            frame_idx=args.frame,
            points=[[args.points[i], args.points[i+1]] for i in range(0, len(args.points), 2)] if args.points else None,
            automatic=args.automatic
        )
    else:
        # Image input
        frame = np.array(Image.open(input_path))
        if args.automatic:
            results = sam2.automatic_segmentation(frame)
        else:
            points = [[args.points[i], args.points[i+1]] for i in range(0, len(args.points), 2)] if args.points else None
            results = sam2.segment_image(frame, points=points)
    
    # Visualize
    if args.automatic:
        # Create combined mask visualization for automatic segmentation
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for idx, mask_data in enumerate(results[:10]):  # Show top 10 masks
            mask = mask_data['segmentation']
            combined_mask[mask] = (idx + 1) * 25  # Different value for each mask
        
        # Create colored visualization
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(1, 11):
            mask = combined_mask == idx * 25
            color = plt.cm.tab20(idx / 10)[:3]
            colored[mask] = (color * 255).astype(np.uint8)
        
        # Overlay on original
        overlay = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)
        
        # Save or show
        output_path = args.output or "sam2_automatic_segments.png"
        Image.fromarray(overlay).save(output_path)
        print(f"Saved automatic segmentation to {output_path}")
    else:
        # Visualize masks with colors
        output_path = args.output or "sam2_segments.png"
        vis = sam2.visualize_masks(
            frame,
            results['masks'],
            results['scores'],
            save_path=output_path
        )
    
    print("Segmentation complete!")


if __name__ == "__main__":
    main()