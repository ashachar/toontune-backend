#!/usr/bin/env python3
"""
Simple example of calling the text animation Lambda function.
"""

import json
import requests
import time

# Lambda function URL
LAMBDA_URL = "https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/"

def call_text_animation_lambda(video_url, text):
    """
    Call the Lambda function to add text animation to a video.
    
    Args:
        video_url: S3 URL or public URL to the video
        text: Text to animate (will be converted to uppercase)
    
    Returns:
        dict: Response from Lambda with video_url, s3_key, etc.
    """
    
    # Prepare the request payload
    payload = {
        "video_url": video_url,
        "text": text
    }
    
    print(f"Calling Lambda with:")
    print(f"  Video: {video_url}")
    print(f"  Text: {text}")
    
    try:
        # Make the HTTP POST request
        response = requests.post(
            LAMBDA_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minute timeout
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        print(f"\n✓ Success!")
        print(f"  Output video: {result['video_url']}")
        print(f"  S3 key: {result['s3_key']}")
        print(f"  Size: {result['output_size_mb']} MB")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error calling Lambda: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"\n✗ Error parsing response: {e}")
        return None


def main():
    """Example usage."""
    
    # Example 1: Using an S3 URL
    print("=" * 60)
    print("Example 1: S3 URL")
    print("=" * 60)
    
    s3_video_url = "s3://toontune-text-animations/test-videos/do_re_mi.mov"
    result1 = call_text_animation_lambda(s3_video_url, "HELLO")
    
    if result1:
        print(f"\nYou can download the video from:")
        print(f"{result1['video_url']}")
    
    # Example 2: Using a public URL (if you have one)
    # print("\n" + "=" * 60)
    # print("Example 2: Public URL")
    # print("=" * 60)
    # 
    # public_video_url = "https://example.com/my-video.mp4"
    # result2 = call_text_animation_lambda(public_video_url, "WORLD")
    
    # Example 3: Processing multiple texts
    print("\n" + "=" * 60)
    print("Example 3: Multiple texts")
    print("=" * 60)
    
    texts = ["START", "MIDDLE", "END"]
    for text in texts:
        print(f"\nProcessing text: {text}")
        result = call_text_animation_lambda(s3_video_url, text)
        if result:
            print(f"  → {result['s3_key']}")
        time.sleep(1)  # Be nice to the Lambda


if __name__ == "__main__":
    main()