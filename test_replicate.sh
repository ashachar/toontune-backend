#!/bin/bash

# Test AnimateDiff on Replicate using cURL
# You need to set your API token first:
# export REPLICATE_API_TOKEN='r8_your_token_here'

if [ -z "$REPLICATE_API_TOKEN" ]; then
    echo "❌ Please set REPLICATE_API_TOKEN environment variable"
    echo "Get your token at: https://replicate.com/account/api-tokens"
    echo ""
    echo "Example:"
    echo "export REPLICATE_API_TOKEN='r8_your_token_here'"
    exit 1
fi

echo "🎭 Testing AnimateDiff on Replicate API"
echo "========================================"

# Using stable-diffusion-animation model
MODEL_VERSION="0359ebf6a1e4c4b8c59e0b01b86b7ae3e35c3fce2b1b4dbe7c9e2a21c2e7bff6"

# Create prediction
RESPONSE=$(curl -s -X POST \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "'$MODEL_VERSION'",
    "input": {
      "prompt": "A graceful cartoon ballet dancer girl with brown hair in a bun, wearing black leotard, performing pirouette spin and arabesque, smooth ballet movements, cartoon animation style",
      "num_frames": 16,
      "num_inference_steps": 25,
      "guidance_scale": 7.5,
      "width": 512,
      "height": 512,
      "seed": 42
    }
  }' \
  https://api.replicate.com/v1/predictions)

# Extract prediction ID and status URL
PREDICTION_ID=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))")
STATUS_URL=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('urls', {}).get('get', ''))")

if [ -z "$PREDICTION_ID" ]; then
    echo "❌ Failed to create prediction"
    echo "Response: $RESPONSE"
    exit 1
fi

echo "✅ Prediction created: $PREDICTION_ID"
echo "⏳ Waiting for generation (10-30 seconds)..."

# Poll for results
while true; do
    sleep 3
    
    STATUS_RESPONSE=$(curl -s -H "Authorization: Token $REPLICATE_API_TOKEN" "$STATUS_URL")
    STATUS=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))")
    
    echo "   Status: $STATUS"
    
    if [ "$STATUS" = "succeeded" ]; then
        OUTPUT_URL=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('output', ''))")
        echo ""
        echo "✅ Animation generated!"
        echo "🎬 Output URL: $OUTPUT_URL"
        
        # Download the animation
        mkdir -p test_output
        curl -s "$OUTPUT_URL" -o test_output/dancer_replicate.gif
        echo "💾 Saved to: test_output/dancer_replicate.gif"
        
        # Get metrics
        PREDICT_TIME=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('metrics', {}).get('predict_time', 0))")
        COST=$(echo "$PREDICT_TIME * 0.00055" | bc -l 2>/dev/null || echo "0.002")
        
        echo ""
        echo "💰 Estimated cost: \$$COST"
        echo "⏱️  Generation time: ${PREDICT_TIME}s"
        break
        
    elif [ "$STATUS" = "failed" ]; then
        ERROR=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', 'Unknown error'))")
        echo "❌ Failed: $ERROR"
        exit 1
        
    elif [ "$STATUS" = "canceled" ]; then
        echo "⚠️ Canceled"
        exit 1
    fi
done

echo ""
echo "🎉 Test complete! View your animation at: test_output/dancer_replicate.gif"