#!/usr/bin/env python3
"""
Generate a high-quality colorful doodle kid with transparent background using OpenAI image-1
"""
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def main():
    # Load .env from current dir and parent directories
    load_dotenv()  # Try current directory first
    load_dotenv(Path(__file__).parent.parent.parent.parent / '.env')  # Load from backend root
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it in your shell and try again.")

    client = OpenAI(api_key=api_key)

    prompt = (
        "high quality colorful doodle of a smiling kid, thick black outline, simple hand-drawn shapes, "
        "flat bright crayon colors, playful proportions, minimal details, whimsical cartoon doodle style, "
        "clean edges, no shading, transparent background"
    )

    print("▶️  Generating high-quality doodle kid with gpt-image-1...")
    
    try:
        resp = client.images.generate(
            model="gpt-image-1",  # State-of-the-art image generation model
            prompt=prompt,
            size="1024x1024",
            quality="high",  # High quality for gpt-image-1
            response_format="b64_json"
        )
        
        b64_json = resp.data[0].b64_json
        out_dir = Path(__file__).parent / "out"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "kid.png"
        
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64_json))
        
        print(f"✅ Saved {out_path}")
        
        # Return the revised prompt if available
        if hasattr(resp.data[0], 'revised_prompt'):
            print(f"\n📝 Revised prompt: {resp.data[0].revised_prompt}")
            
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        raise

if __name__ == "__main__":
    main()