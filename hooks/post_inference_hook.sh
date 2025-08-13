#!/bin/bash

# Get the random ID from the current inference
if [ ! -f .current_inference_id ]; then
    exit 0
fi

RANDOM_ID=$(cat .current_inference_id)
TIMESTAMP_FILE="inference_start_timestamped_${RANDOM_ID}.txt"

# Check if timestamp file exists
if [ ! -f "$TIMESTAMP_FILE" ]; then
    rm -f .current_inference_id
    exit 0
fi

# Get the timestamp
TIMESTAMP=$(cat "$TIMESTAMP_FILE")

# Convert timestamp to date format for find command (BSD/macOS compatible)
# Find all files modified after the timestamp
echo "Changed files:"
find . -type f -newer "$TIMESTAMP_FILE" 2>/dev/null | grep -v "inference_start_timestamped_" | grep -v ".current_inference_id" | grep -v ".git/" | sort

# Clean up
rm -f "$TIMESTAMP_FILE"
rm -f .current_inference_id