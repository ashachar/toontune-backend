#!/usr/bin/env python3
"""Generate RVM mask for full video using Replicate."""

import os
import sys
import replicate
from dotenv import load_dotenv

load_dotenv()

def generate_rvm_mask(input_video, output_path):
    """Generate RVM mask using Replicate API."""
    
    print(f"Generating RVM mask for: {input_video}")
    print("This will take a few minutes...")
    
    # Run RVM model on Replicate
    output = replicate.run(
        "arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
        input={
            "input_video": open(input_video, "rb"),
            "output_type": "green-screen",
        }
    )
    
    # Download result
    import requests
    response = requests.get(output)
    
    # Save to file
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    print(f"✅ RVM mask saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    input_video = "uploads/assets/videos/ai_math1.mp4"
    output_mask = "uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_mask), exist_ok=True)
    
    # Check input exists
    if not os.path.exists(input_video):
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)
    
    # Generate mask
    try:
        generate_rvm_mask(input_video, output_mask)
    except Exception as e:
        print(f"Error generating RVM mask: {e}")
        sys.exit(1)