#!/bin/bash

# Generate random ID for this session
RANDOM_ID=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)

# Get current timestamp
TIMESTAMP=$(date +%s)

# Create timestamp file in project root
echo "$TIMESTAMP" > "inference_start_timestamped_${RANDOM_ID}.txt"

# Also store the random ID for the post hook to find
echo "$RANDOM_ID" > .current_inference_id

# Make sure the file is created
if [ -f "inference_start_timestamped_${RANDOM_ID}.txt" ]; then
    exit 0
else
    exit 1
fi