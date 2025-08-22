#!/usr/bin/env bash
set -euo pipefail

echo "[LAMBDA_ANIM] Building Lambda Layer with rembg using Docker..."

# Use Amazon Linux 2 image that matches Lambda runtime
docker run --rm \
  -v "$(pwd)":/build \
  -w /build \
  public.ecr.aws/lambda/python:3.11 \
  bash -c "
    echo '[LAMBDA_ANIM] Installing packages for Lambda Linux environment...'
    pip install --no-cache-dir \
                --platform manylinux2014_x86_64 \
                --implementation cp \
                --python-version 311 \
                --only-binary=:all: \
                --upgrade \
                -t python \
                -r requirements.txt
    
    # Download U2Net model to include in layer (optional but speeds up cold starts)
    echo '[LAMBDA_ANIM] Pre-downloading U2Net model...'
    python -c \"
from rembg import new_session
import os
os.environ['U2NET_HOME'] = '/build/python/.u2net'
session = new_session('u2net')
print('[LAMBDA_ANIM] U2Net model downloaded successfully')
\" || echo '[LAMBDA_ANIM] Warning: Could not pre-download model'
  "

# Package the layer
echo "[LAMBDA_ANIM] Creating layer.zip..."
zip -r9q layer.zip python

# Get file size
SIZE=$(du -h layer.zip | cut -f1)
echo "[LAMBDA_ANIM] Layer built: $(pwd)/layer.zip (size: $SIZE)"
echo "[LAMBDA_ANIM] Ready to upload to AWS Lambda Layers"