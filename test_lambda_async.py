#!/usr/bin/env python3
"""
Async example of calling the Lambda function with concurrent requests.
"""

import asyncio
import aiohttp
import json

LAMBDA_URL = "https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/"

async def call_lambda_async(session, video_url, text):
    """Async Lambda call."""
    payload = {
        "video_url": video_url,
        "text": text
    }
    
    try:
        async with session.post(LAMBDA_URL, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✓ {text}: {result['s3_key']}")
                return result
            else:
                print(f"✗ {text}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"✗ {text}: {e}")
        return None


async def process_multiple_videos():
    """Process multiple videos concurrently."""
    
    # List of videos to process
    tasks_data = [
        ("s3://toontune-text-animations/test-videos/do_re_mi.mov", "HELLO"),
        ("s3://toontune-text-animations/test-videos/do_re_mi.mov", "WORLD"),
        ("s3://toontune-text-animations/test-videos/do_re_mi.mov", "ASYNC"),
    ]
    
    async with aiohttp.ClientSession() as session:
        # Create tasks for concurrent execution
        tasks = [
            call_lambda_async(session, video_url, text)
            for video_url, text in tasks_data
        ]
        
        # Wait for all tasks to complete
        print("Processing videos concurrently...")
        results = await asyncio.gather(*tasks)
        
        # Print results
        print("\nAll tasks completed!")
        for i, result in enumerate(results):
            if result:
                print(f"  Task {i+1}: {result['video_url']}")


if __name__ == "__main__":
    # Run the async function
    asyncio.run(process_multiple_videos())