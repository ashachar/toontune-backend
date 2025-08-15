#!/usr/bin/env python3
"""
Simple BLIP Image Captioner
Fast and reliable image captioning using Salesforce BLIP
"""

import os
import time
from pathlib import Path
import argparse
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class BlipCaptioner:
    def __init__(self, device="auto"):
        """Initialize BLIP model and processor"""
        self.model = None
        self.processor = None
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model_name = "Salesforce/blip-image-captioning-large"
        
    def load_model(self):
        """Load BLIP model and processor"""
        if self.model is not None:
            return
        
        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        
        # Load model with appropriate dtype
        if self.device == "cpu":
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            )
        else:
            # Use float16 for GPU/MPS
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )
        
        # Move model to device
        if self.device == "mps":
            self.model = self.model.to("mps")
        elif self.device == "cuda":
            self.model = self.model.cuda()
        
        self.model.eval()
        
    def caption_image(self, image_path, max_length=50):
        """Generate caption for an image"""
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = self.processor(image, return_tensors="pt")
        
        # Move to device
        if self.device == "mps":
            inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        elif self.device == "cuda":
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=max_length)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption, 0

def main():
    parser = argparse.ArgumentParser(description='BLIP image captioner')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu', 'mps'])
    parser.add_argument('--max-length', type=int, default=50)
    
    args = parser.parse_args()
    
    # Process image
    print(f"🔧 Using device: {args.device if args.device != 'auto' else 'auto-detect'}")
    print(f"\n📷 Processing: {args.input}")
    
    start_time = time.time()
    
    # Initialize and run captioner
    captioner = BlipCaptioner(device=args.device)
    
    print("📥 Loading BLIP model...")
    captioner.load_model()
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.1f}s")
    
    # Generate caption
    gen_start = time.time()
    caption, _ = captioner.caption_image(args.input, max_length=args.max_length)
    gen_time = time.time() - gen_start
    
    print(f"\n✅ Caption generated!")
    print(f"📄 Caption: {caption}")
    print(f"⏱️  Generation time: {gen_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    # Save caption
    input_path = Path(args.input)
    caption_file = input_path.parent / f"{input_path.stem}.txt"
    caption_file.write_text(caption)
    print(f"💾 Saved to: {caption_file}")

if __name__ == "__main__":
    main()