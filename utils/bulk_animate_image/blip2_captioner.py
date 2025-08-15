#!/usr/bin/env python3
"""
BLIP-2 Image Captioner - Enhanced version with fixes
Uses Salesforce BLIP-2 for improved captioning
"""

import os
import time
from pathlib import Path
import argparse
from PIL import Image
import torch

# Try different BLIP-2 imports based on availability
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    BLIP2_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False
    print("⚠️ BLIP-2 not available, falling back to BLIP")

# Fallback to BLIP if BLIP-2 fails
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

class EnhancedCaptioner:
    def __init__(self, device="auto", model_type="blip2", model_size="base"):
        """
        Initialize with BLIP-2 or fallback to BLIP
        
        model_type: "blip2" or "blip"
        model_size for BLIP-2: "base" (2.7b), "large" (6.7b), "flan" (flan-t5)
        """
        self.model = None
        self.processor = None
        self.model_type = model_type
        
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
        
        print(f"🔧 Using device: {self.device}")
        
        # Try BLIP-2 first, fallback to BLIP
        if model_type == "blip2" and BLIP2_AVAILABLE:
            self.setup_blip2(model_size)
        else:
            self.setup_blip()
    
    def setup_blip2(self, model_size):
        """Setup BLIP-2 model"""
        try:
            # BLIP-2 model configurations
            model_configs = {
                "base": "Salesforce/blip2-opt-2.7b",
                "large": "Salesforce/blip2-opt-6.7b",
                "flan": "Salesforce/blip2-flan-t5-xl"
            }
            
            self.model_name = model_configs.get(model_size, model_configs["base"])
            print(f"📦 Attempting BLIP-2: {self.model_name}")
            
            # Try to load with error handling
            try:
                self.processor = Blip2Processor.from_pretrained(self.model_name)
                
                # Load model with appropriate dtype
                if self.device == "cpu":
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                else:
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                
                # Move to device
                if self.device == "mps":
                    self.model = self.model.to("mps")
                elif self.device == "cuda":
                    self.model = self.model.cuda()
                
                self.model.eval()
                self.model_type = "blip2"
                print("✅ BLIP-2 loaded successfully")
                
            except Exception as e:
                print(f"⚠️ BLIP-2 loading failed: {e}")
                print("   Falling back to BLIP...")
                self.setup_blip()
                
        except Exception as e:
            print(f"⚠️ BLIP-2 setup failed: {e}")
            self.setup_blip()
    
    def setup_blip(self):
        """Setup original BLIP model as fallback"""
        if not BLIP_AVAILABLE:
            raise ImportError("Neither BLIP-2 nor BLIP are available. Please install transformers.")
        
        self.model_name = "Salesforce/blip-image-captioning-large"
        print(f"📦 Using BLIP: {self.model_name}")
        
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        
        # Load model
        if self.device == "cpu":
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            )
        else:
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )
        
        # Move to device
        if self.device == "mps":
            self.model = self.model.to("mps")
        elif self.device == "cuda":
            self.model = self.model.cuda()
        
        self.model.eval()
        self.model_type = "blip"
        print("✅ BLIP loaded successfully")
    
    def load_model(self):
        """Ensure model is loaded"""
        if self.model is None:
            if self.model_type == "blip2":
                self.setup_blip2("base")
            else:
                self.setup_blip()
    
    def caption_image(self, image_path, prompt=None, max_length=100, num_beams=5):
        """
        Generate caption for an image
        
        Args:
            image_path: Path to image
            prompt: Optional prompt (BLIP-2 feature)
            max_length: Maximum caption length
            num_beams: Beam search width
        """
        if self.model is None:
            self.load_model()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs based on model type
        if self.model_type == "blip2" and prompt:
            # BLIP-2 with prompt
            inputs = self.processor(image, text=prompt, return_tensors="pt")
        else:
            # BLIP or BLIP-2 without prompt
            inputs = self.processor(image, return_tensors="pt")
        
        # Move to device
        if self.device == "mps":
            inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        elif self.device == "cuda":
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generate caption
        start = time.time()
        with torch.no_grad():
            if self.model_type == "blip":
                # BLIP generation
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams
                )
            else:
                # BLIP-2 generation
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=0.9,
                    do_sample=False
                )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Clean up caption
            caption = caption.strip()
            if prompt and caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
        
        elapsed = time.time() - start
        
        return caption, elapsed
    
    def detailed_caption(self, image_path, max_length=150):
        """Generate a detailed caption"""
        if self.model_type == "blip2":
            # Use BLIP-2's prompt feature
            prompt = "This is a detailed description of the image:"
            return self.caption_image(image_path, prompt=prompt, max_length=max_length)
        else:
            # Regular BLIP caption with higher max_length
            return self.caption_image(image_path, max_length=max_length)
    
    def answer_question(self, image_path, question, max_length=50):
        """Answer a question about the image (BLIP-2 feature)"""
        if self.model_type == "blip2":
            prompt = f"Question: {question} Answer:"
            return self.caption_image(image_path, prompt=prompt, max_length=max_length)
        else:
            # Fallback for BLIP - just generate caption
            print("⚠️ Q&A not supported in BLIP, generating caption instead")
            return self.caption_image(image_path, max_length=max_length)

def main():
    parser = argparse.ArgumentParser(description='Enhanced BLIP/BLIP-2 captioner')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu', 'mps'])
    parser.add_argument('--model', default='blip2', choices=['blip2', 'blip'])
    parser.add_argument('--size', default='base', choices=['base', 'large', 'flan'])
    parser.add_argument('--max-length', type=int, default=100)
    parser.add_argument('--prompt', help='Optional prompt for guided generation')
    parser.add_argument('--question', help='Ask a question about the image')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed caption')
    
    args = parser.parse_args()
    
    # Initialize captioner
    print(f"\n📷 Processing: {args.input}")
    start_total = time.time()
    
    captioner = EnhancedCaptioner(
        device=args.device, 
        model_type=args.model,
        model_size=args.size
    )
    
    # Process image
    if args.question:
        # Q&A mode
        answer, gen_time = captioner.answer_question(args.input, args.question)
        print(f"\n❓ Question: {args.question}")
        print(f"💬 Answer: {answer}")
    elif args.detailed:
        # Detailed caption mode
        caption, gen_time = captioner.detailed_caption(args.input, max_length=args.max_length)
        print(f"\n📄 Detailed Caption: {caption}")
    else:
        # Regular caption mode
        caption, gen_time = captioner.caption_image(
            args.input, 
            prompt=args.prompt,
            max_length=args.max_length
        )
        print(f"\n📄 Caption: {caption}")
    
    print(f"⏱️  Generation time: {gen_time:.2f}s")
    
    total_time = time.time() - start_total
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    # Save caption
    input_path = Path(args.input)
    model_suffix = "blip2" if captioner.model_type == "blip2" else "blip"
    caption_file = input_path.parent / f"{input_path.stem}_{model_suffix}.txt"
    
    if args.question:
        caption_file.write_text(f"Q: {args.question}\nA: {answer}")
    else:
        caption_file.write_text(caption if 'caption' in locals() else answer)
    
    print(f"💾 Saved to: {caption_file}")

if __name__ == "__main__":
    main()