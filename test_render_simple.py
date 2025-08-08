#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
import glob

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('VITE_SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Error: Supabase credentials not found in .env file")
    sys.exit(1)

print(f"Supabase URL: {SUPABASE_URL}")

# Initialize Supabase client
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("Supabase client created successfully")
except Exception as e:
    print(f"Error creating Supabase client: {e}")
    sys.exit(1)

# Constants
ASSETS_DIR = "../app/backend/uploads/assets"
BUCKET_NAME = "rendered-assets"

# Find all assets
asset_patterns = [
    os.path.join(ASSETS_DIR, "*.png"),
    os.path.join(ASSETS_DIR, "*.jpg"),
    os.path.join(ASSETS_DIR, "*.jpeg"),
    os.path.join(ASSETS_DIR, "*.svg")
]

asset_files = []
for pattern in asset_patterns:
    asset_files.extend(glob.glob(pattern))

print(f"Found {len(asset_files)} assets:")
for asset in asset_files:
    print(f"  - {Path(asset).name}")

# Test bucket access
try:
    buckets = supabase.storage.list_buckets()
    print(f"\nAvailable buckets: {[b.name for b in buckets]}")
except Exception as e:
    print(f"Error listing buckets: {e}")

# Test if our bucket exists
try:
    files = supabase.storage.from_(BUCKET_NAME).list()
    print(f"\nFiles in {BUCKET_NAME}: {len(files)} files")
    for f in files[:5]:  # Show first 5 files
        print(f"  - {f.get('name', 'unknown')}")
except Exception as e:
    print(f"Error accessing bucket {BUCKET_NAME}: {e}")
    # Try to create it
    try:
        supabase.storage.create_bucket(BUCKET_NAME, options={"public": True})
        print(f"Created bucket: {BUCKET_NAME}")
    except Exception as e2:
        print(f"Error creating bucket: {e2}")