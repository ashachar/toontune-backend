#!/usr/bin/env python3
"""
Check available SAM2 models on Replicate
"""

import os
import replicate
from dotenv import load_dotenv

# Load environment variables
load_dotenv('./backend/.env')

# Set API token
api_key = os.environ.get("REPLICATE_API_KEY") or os.environ.get("REPLICATE_API_TOKEN")
if api_key:
    os.environ["REPLICATE_API_TOKEN"] = api_key

print("Checking SAM2 models on Replicate...")

try:
    # Try different model names
    model_names = [
        "meta/sam-2",
        "meta/sam-2-video", 
        "zsxkib/segment-anything-2",
        "chenxwh/sam-2",
        "datasette/sam2"
    ]
    
    for model_name in model_names:
        try:
            print(f"\nTrying: {model_name}")
            model = replicate.models.get(model_name)
            print(f"✅ Found: {model_name}")
            print(f"   Latest version: {model.latest_version.id}")
            print(f"   Description: {model.description[:100]}..." if model.description else "")
        except Exception as e:
            print(f"❌ Not found or error: {e}")
    
    # Search for SAM models
    print("\n\nSearching for SAM models...")
    # Note: This might not work if search is not available
    
except Exception as e:
    print(f"Error: {e}")