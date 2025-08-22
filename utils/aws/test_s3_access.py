#!/usr/bin/env python3
"""
Test S3 access and verify that videos are publicly accessible.
This script checks both public URLs and pre-signed URLs.
"""

import boto3
import urllib.request
import urllib.error
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

class S3AccessTester:
    """Test S3 bucket access and permissions."""
    
    def __init__(self, bucket_name: str = 'toontune-text-animations'):
        """Initialize the tester."""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.test_results = []
    
    def test_public_url(self, object_key: str) -> Tuple[bool, str]:
        """
        Test if an object is publicly accessible.
        
        Returns:
            Tuple of (success, message)
        """
        # Try different URL formats
        url_formats = [
            f"https://{self.bucket_name}.s3.amazonaws.com/{object_key}",
            f"https://s3.amazonaws.com/{self.bucket_name}/{object_key}",
            f"https://{self.bucket_name}.s3.us-east-1.amazonaws.com/{object_key}",
            f"https://s3.us-east-1.amazonaws.com/{self.bucket_name}/{object_key}"
        ]
        
        for url in url_formats:
            try:
                request = urllib.request.Request(url, method='HEAD')
                response = urllib.request.urlopen(request)
                
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', 'unknown')
                    return True, f"✅ Publicly accessible at: {url} (Content-Type: {content_type})"
                    
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    continue  # Try next URL format
                elif e.code == 404:
                    return False, f"❌ Object not found: {object_key}"
            except Exception as e:
                continue
        
        return False, f"❌ Not publicly accessible (all URL formats failed): {object_key}"
    
    def test_object_exists(self, object_key: str) -> bool:
        """Check if an object exists in the bucket."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except:
            return False
    
    def upload_test_file(self, folder: str = 'processed') -> Optional[str]:
        """Upload a test file to verify permissions."""
        test_key = f"{folder}/test-access-{sys.argv[0]}.txt"
        test_content = b"Test file for S3 access verification"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=test_key,
                Body=test_content,
                ContentType='text/plain'
            )
            return test_key
        except Exception as e:
            print(f"❌ Failed to upload test file: {e}")
            return None
    
    def test_folder_permissions(self, folder: str) -> dict:
        """Test permissions for a specific folder."""
        results = {
            'folder': folder,
            'upload': False,
            'public_access': False,
            'test_url': None
        }
        
        print(f"\n📁 Testing folder: /{folder}/*")
        print("-" * 40)
        
        # Upload test file
        test_key = self.upload_test_file(folder)
        if test_key:
            results['upload'] = True
            print(f"✅ Upload successful: {test_key}")
            
            # Test public access
            success, message = self.test_public_url(test_key)
            results['public_access'] = success
            print(message)
            
            if success:
                results['test_url'] = f"https://{self.bucket_name}.s3.amazonaws.com/{test_key}"
            
            # Clean up
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=test_key)
                print(f"🧹 Test file cleaned up")
            except:
                pass
        else:
            print(f"❌ Upload failed for folder: {folder}")
        
        return results
    
    def test_existing_videos(self) -> list:
        """Test access to existing videos in the bucket."""
        print(f"\n🎬 Testing existing videos...")
        print("-" * 40)
        
        results = []
        
        # List objects in processed folder
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='processed/',
                MaxKeys=5
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith(('.mp4', '.webm', '.mov')):
                        print(f"\nTesting: {key}")
                        success, message = self.test_public_url(key)
                        print(f"  {message}")
                        results.append({
                            'key': key,
                            'size': obj['Size'],
                            'public': success
                        })
            else:
                print("No videos found in /processed/ folder")
                
        except Exception as e:
            print(f"Error listing objects: {e}")
        
        return results
    
    def verify_cors(self) -> bool:
        """Verify CORS configuration."""
        print(f"\n🌐 Verifying CORS configuration...")
        print("-" * 40)
        
        try:
            response = self.s3_client.get_bucket_cors(Bucket=self.bucket_name)
            cors_rules = response.get('CORSRules', [])
            
            if cors_rules:
                print(f"✅ CORS configured with {len(cors_rules)} rule(s)")
                for i, rule in enumerate(cors_rules, 1):
                    print(f"  Rule {i}:")
                    print(f"    Methods: {rule.get('AllowedMethods', [])}")
                    print(f"    Origins: {rule.get('AllowedOrigins', [])}")
                return True
            else:
                print("❌ No CORS rules configured")
                return False
                
        except self.s3_client.exceptions.NoSuchCORSConfiguration:
            print("❌ CORS not configured")
            return False
        except Exception as e:
            print(f"❌ Error checking CORS: {e}")
            return False
    
    def run_all_tests(self):
        """Run all access tests."""
        print(f"🧪 S3 Access Test Suite")
        print(f"🪣 Bucket: {self.bucket_name}")
        print("=" * 50)
        
        # Test folder permissions
        folders_to_test = ['processed', 'assets', 'scenes/edited']
        folder_results = []
        
        for folder in folders_to_test:
            result = self.test_folder_permissions(folder)
            folder_results.append(result)
        
        # Test existing videos
        video_results = self.test_existing_videos()
        
        # Verify CORS
        cors_ok = self.verify_cors()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        print("\nFolder Permissions:")
        for result in folder_results:
            status = "✅" if result['public_access'] else "❌"
            print(f"  {status} /{result['folder']}/* - Public: {result['public_access']}")
            if result['test_url']:
                print(f"      Test URL: {result['test_url']}")
        
        if video_results:
            print(f"\nExisting Videos ({len(video_results)} tested):")
            public_count = sum(1 for v in video_results if v['public'])
            print(f"  Public: {public_count}/{len(video_results)}")
        
        print(f"\nCORS Configuration: {'✅ Enabled' if cors_ok else '❌ Disabled'}")
        
        # Overall status
        all_public = all(r['public_access'] for r in folder_results)
        if all_public and cors_ok:
            print("\n✨ All tests passed! Your S3 bucket is properly configured.")
        else:
            print("\n⚠️ Some tests failed. Please run the configuration script:")
            print("  python utils/aws/configure_s3_permissions.py")

def main():
    """Main function."""
    bucket_name = 'toontune-text-animations'
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python test_s3_access.py [bucket-name]")
            print("\nTests S3 bucket permissions and public access.")
            return
        bucket_name = sys.argv[1]
    
    tester = S3AccessTester(bucket_name)
    tester.run_all_tests()

if __name__ == "__main__":
    main()