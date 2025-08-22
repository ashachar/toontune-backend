#!/usr/bin/env bash
set -euo pipefail

echo "[LAMBDA_ANIM] Building Lambda Layer with rembg dependencies..."

PYV=python3.11                          # match Lambda runtime
LAYER_DIR="$(pwd)/python"               # final layer content must be under /python
rm -rf "$LAYER_DIR"
mkdir -p "$LAYER_DIR"

echo "[LAMBDA_ANIM] Installing packages to $LAYER_DIR..."

# Use platform-specific wheels for Lambda (Linux x86_64)
pip install --platform manylinux2014_x86_64 \
            --implementation cp \
            --python-version 311 \
            --only-binary=:all: \
            --upgrade \
            -t "$LAYER_DIR" \
            -r requirements.txt

# Package the layer
cd "$(dirname "$LAYER_DIR")"
echo "[LAMBDA_ANIM] Creating layer.zip..."
zip -r9q layer.zip python

# Get file size
SIZE=$(du -h layer.zip | cut -f1)
echo "[LAMBDA_ANIM] Layer built: $(pwd)/layer.zip (size: $SIZE)"
echo "[LAMBDA_ANIM] Ready to upload to AWS Lambda Layers"