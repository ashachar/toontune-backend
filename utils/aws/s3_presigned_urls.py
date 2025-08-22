#!/usr/bin/env python3
"""
Utility for generating pre-signed URLs for S3 objects.
This is an alternative to public access for more secure, time-limited access.
"""

import boto3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

class S3PresignedURLGenerator:
    """Generate pre-signed URLs for S3 objects."""
    
    def __init__(self, bucket_name: str = 'toontune-text-animations'):
        """Initialize the generator with a bucket name."""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
    
    def generate_url(self, 
                    object_key: str, 
                    expiration: int = 3600,
                    http_method: str = 'get_object') -> Optional[str]:
        """
        Generate a pre-signed URL for an S3 object.
        
        Args:
            object_key: The S3 object key (path within bucket)
            expiration: URL expiration time in seconds (default: 1 hour)
            http_method: The HTTP method (get_object or put_object)
        
        Returns:
            Pre-signed URL string or None if error
        """
        try:
            url = self.s3_client.generate_presigned_url(
                http_method,
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"Error generating pre-signed URL: {e}")
            return None
    
    def generate_upload_url(self,
                           object_key: str,
                           content_type: str = 'video/mp4',
                           expiration: int = 3600) -> Optional[Dict[str, Any]]:
        """
        Generate a pre-signed URL for uploading to S3.
        
        Args:
            object_key: The S3 object key (path within bucket)
            content_type: MIME type of the content
            expiration: URL expiration time in seconds
        
        Returns:
            Dict with URL and fields for form upload, or None if error
        """
        try:
            response = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=object_key,
                Fields={'Content-Type': content_type},
                Conditions=[
                    {'Content-Type': content_type},
                    ['content-length-range', 0, 500 * 1024 * 1024]  # Max 500MB
                ],
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            print(f"Error generating upload URL: {e}")
            return None
    
    def generate_batch_urls(self, 
                          object_keys: list,
                          expiration: int = 3600) -> Dict[str, str]:
        """
        Generate pre-signed URLs for multiple objects.
        
        Args:
            object_keys: List of S3 object keys
            expiration: URL expiration time in seconds
        
        Returns:
            Dict mapping object keys to their pre-signed URLs
        """
        urls = {}
        for key in object_keys:
            url = self.generate_url(key, expiration)
            if url:
                urls[key] = url
        return urls
    
    def generate_time_based_url(self,
                               object_key: str,
                               days: int = 0,
                               hours: int = 1,
                               minutes: int = 0) -> Optional[str]:
        """
        Generate a pre-signed URL with time-based expiration.
        
        Args:
            object_key: The S3 object key
            days: Number of days until expiration
            hours: Number of hours until expiration
            minutes: Number of minutes until expiration
        
        Returns:
            Pre-signed URL string or None if error
        """
        total_seconds = days * 86400 + hours * 3600 + minutes * 60
        return self.generate_url(object_key, total_seconds)

def generate_video_urls(video_paths: list, expiration_days: int = 7) -> Dict[str, str]:
    """
    Convenience function to generate URLs for video files.
    
    Args:
        video_paths: List of video paths in S3
        expiration_days: Number of days until URL expires
    
    Returns:
        Dict mapping video paths to pre-signed URLs
    """
    generator = S3PresignedURLGenerator()
    return generator.generate_batch_urls(
        video_paths,
        expiration=expiration_days * 24 * 3600
    )

def main():
    """Example usage and testing."""
    import sys
    
    generator = S3PresignedURLGenerator()
    
    if len(sys.argv) > 1:
        # Generate URL for specific file
        object_key = sys.argv[1]
        expiration_days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        
        url = generator.generate_time_based_url(
            object_key, 
            days=expiration_days
        )
        
        if url:
            print(f"✅ Pre-signed URL generated (expires in {expiration_days} days):")
            print(f"📎 {url}")
            
            # Also show upload URL if it's a new file
            if 'processed/' in object_key or 'assets/' in object_key:
                upload_info = generator.generate_upload_url(object_key)
                if upload_info:
                    print(f"\n📤 Upload URL and fields:")
                    print(json.dumps(upload_info, indent=2))
        else:
            print("❌ Failed to generate URL")
    else:
        # Test with example files
        print("🧪 Testing pre-signed URL generation...")
        
        test_files = [
            'processed/test-video.mp4',
            'assets/cartoon-character.png',
            'scenes/edited/scene-001.mp4'
        ]
        
        urls = generator.generate_batch_urls(test_files, expiration=3600)
        
        print(f"\n📎 Generated {len(urls)} URLs (1 hour expiration):")
        for key, url in urls.items():
            print(f"\n{key}:")
            print(f"  {url[:100]}...")
        
        # Test time-based URL
        long_url = generator.generate_time_based_url(
            'processed/long-term-video.mp4',
            days=30
        )
        print(f"\n📅 30-day URL for long-term access:")
        print(f"  {long_url[:100]}...")

if __name__ == "__main__":
    main()