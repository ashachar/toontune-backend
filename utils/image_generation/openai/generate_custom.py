#!/usr/bin/env python3
"""
Generate a custom image using provided JSON parameters
"""
import os
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def generate_custom_image(params):
    # Load .env from current dir and parent directories
    load_dotenv()  # Try current directory first
    load_dotenv(Path(__file__).parent.parent.parent.parent / '.env')  # Load from backend root
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it in your shell and try again.")

    client = OpenAI(api_key=api_key)

    print("▶️  Generating custom image...")
    
    try:
        # Use the gpt-image-1 model directly as specified
        # Note: gpt-image-1 might not support all parameters
        resp = client.images.generate(
            model="gpt-image-1",  # New state-of-the-art image generation model
            prompt=params["prompt"],
            size=params.get("size", "1024x1024"),  # Supports 1024x1536 directly
            quality=params.get("quality", "standard"),  # Supports low/medium/high
            n=params.get("n", 1)
            # response_format not supported by gpt-image-1
        )
        
        # Handle the response - it might be URL or b64_json
        if hasattr(resp.data[0], 'b64_json'):
            b64_json = resp.data[0].b64_json
        elif hasattr(resp.data[0], 'url'):
            # If URL, download the image
            import requests
            response = requests.get(resp.data[0].url)
            image_data = response.content
            b64_json = base64.b64encode(image_data).decode('utf-8')
        else:
            raise Exception("Unknown response format from API")
        out_dir = Path(__file__).parent / "out"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "girl_telescope.png"
        
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64_json))
        
        print(f"✅ Saved {out_path}")
        
        # Return the revised prompt if available
        if hasattr(resp.data[0], 'revised_prompt'):
            print(f"\n📝 Revised prompt: {resp.data[0].revised_prompt}")
        
        return str(out_path)
            
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        raise

def main():
    # Your provided JSON parameters
    params = {
        "model": "gpt-image-1",
        "prompt": "A clean, crisp, high-resolution 2D cartoon-style illustration of a young girl (about 7–9 years old) with light skin and long chestnut-brown hair. She wears a coral-orange short-sleeve t-shirt, a knee-length blue skirt, red sneakers with white laces, and light pink ankle socks. She stands at the right, faces left, and peers into a silver-gray telescope mounted on a sturdy three-legged black tripod. Expression: curious, engaged; large round black cartoon eyes; gentle smile. Hair parted to the side with a pink headband; smooth flow behind shoulders. Style: bold black outlines, saturated flat color fills, clean vector-like look; consistent with polished children's educational illustrations. Lighting: flat and even (no gradients, no complex shading). Composition: full character and full tripod visible. Background: fully transparent (alpha channel), no scenery, no props beyond the telescope and tripod.\n\nNEGATIVE (avoid explicitly): photorealism, 3D render, gradients, soft/global shadowing, complex or nontransparent background, extra props, text, watermark, logos, noise, blur, artifacts, anime/chibi, abstract/sketch lines, cross-hatching, rough texture, muted/desaturated colors, misaligned outlines, distorted anatomy.",
        "size": "1024x1536",
        "quality": "high",
        "background": "transparent",
        "n": 1,
        "format": "png",
        "response_format": "b64_json"
    }
    
    image_path = generate_custom_image(params)
    
    # Open the image
    import platform
    import subprocess
    
    system = platform.system()
    if system == "Darwin":  # macOS
        subprocess.run(["open", image_path])
    elif system == "Windows":
        subprocess.run(["start", image_path], shell=True)
    else:  # Linux
        subprocess.run(["xdg-open", image_path])

if __name__ == "__main__":
    main()