"""
Runway Act-Two API Interface for Character Performance Animation
Transforms a driver video and character image into an animated character video
"""

import os
import time
import requests
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
import subprocess

# Check if RunwayML SDK is available
try:
    import runwayml
    from runwayml import RunwayML, TaskFailedError
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("RunwayML SDK not installed. Using REST API directly.")

class RunwayActTwo:
    """
    Interface to Runway's Act-Two API for character performance
    """
    
    def __init__(self):
        """Initialize the Runway API client"""
        load_dotenv()
        
        # Get the API key - try both possible env var names
        self.api_key = os.getenv('RUNWAYML_API_SECRET') or os.getenv('RUNWAY_API_KEY')
        if not self.api_key:
            raise ValueError("RUNWAY_API_KEY or RUNWAYML_API_SECRET not found in .env file")
        
        # Initialize client if SDK is available
        if SDK_AVAILABLE:
            self.client = RunwayML(api_key=self.api_key)
        else:
            self.api_base = "https://api.runwayml.com/v1"
            self.headers = {
                "Authorization": f"{self.api_key}",  # Try without Bearer prefix
                "Content-Type": "application/json",
                "X-Runway-Version": "2024-11-06"  # Add API version header
            }
        
        print("Initialized Runway Act-Two API client")
    
    def generate_character_performance(self,
                                      driver_video_path: str,
                                      character_image_path: str,
                                      output_dir: Optional[str] = None,
                                      duration: Optional[int] = None) -> str:
        """
        Generate a character performance video using Act-Two
        
        Args:
            driver_video_path: Path to the driver video
            character_image_path: Path to the character image
            output_dir: Directory to save output (defaults to same as driver video)
            duration: Duration in seconds (defaults to auto-detect from driver video)
            
        Returns:
            Path to the generated video
        """
        if SDK_AVAILABLE:
            return self._generate_with_sdk(driver_video_path, character_image_path, output_dir, duration)
        else:
            return self._generate_with_rest(driver_video_path, character_image_path, output_dir, duration)
    
    def _generate_with_sdk(self,
                          driver_video_path: str,
                          character_image_path: str,
                          output_dir: Optional[str] = None,
                          duration: Optional[int] = None) -> str:
        """Generate using the RunwayML SDK"""
        print("Using RunwayML SDK for Act-Two generation...")
        
        # Upload files as data URIs
        driver_video_uri = self._file_to_data_uri(driver_video_path, "video/mp4")
        character_image_uri = self._file_to_data_uri(character_image_path, "image/png")
        
        # Auto-detect duration if not provided
        if duration is None:
            duration = self._get_video_duration(driver_video_path)
            duration = min(int(duration), 10)  # Cap at 10 seconds
        
        print(f"Creating Act-Two task with {duration}s duration...")
        
        try:
            # Create the character performance task
            # Note: Act-Two uses different parameter names and structure
            task = self.client.character_performance.create(
                model='act_two',
                character={
                    'type': 'image',
                    'uri': character_image_uri
                },
                reference={
                    'type': 'video',
                    'uri': driver_video_uri
                },
                ratio='1280:720'  # Use landscape ratio for now
            )
            
            print(f"Task created with ID: {task.id}")
            print("Waiting for task to complete...")
            
            # Wait for the task to complete
            completed_task = task.wait_for_task_output(timeout=10 * 60 * 1000)  # 10 minutes
            
            if completed_task.status == 'SUCCEEDED':
                # Download the output video
                video_url = completed_task.output[0]
                
                # Determine output path
                if output_dir is None:
                    output_dir = os.path.dirname(driver_video_path)
                
                output_path = os.path.join(output_dir, "runway_act_two_output.mp4")
                
                # Download the video
                print(f"Downloading video from: {video_url[:50]}...")
                response = requests.get(video_url)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Generated video saved to: {output_path}")
                return output_path
            else:
                raise RuntimeError(f"Task failed with status: {completed_task.status}")
                
        except TaskFailedError as e:
            print(f"Task failed: {e.taskDetails}")
            raise
        except Exception as e:
            print(f"Error during generation: {e}")
            raise
    
    def _generate_with_rest(self,
                           driver_video_path: str,
                           character_image_path: str,
                           output_dir: Optional[str] = None,
                           duration: Optional[int] = None) -> str:
        """Generate using REST API directly"""
        print("Using REST API for Act-Two generation...")
        
        # Upload files as data URIs
        driver_video_uri = self._file_to_data_uri(driver_video_path, "video/mp4")
        character_image_uri = self._file_to_data_uri(character_image_path, "image/png")
        
        # Auto-detect duration if not provided
        if duration is None:
            duration = self._get_video_duration(driver_video_path)
            duration = min(int(duration), 10)  # Cap at 10 seconds
        
        print(f"Creating Act-Two task with {duration}s duration...")
        
        # Create the task
        create_url = f"{self.api_base}/character_performance"
        payload = {
            "model": "act_two",
            "driverVideo": driver_video_uri,
            "characterImage": character_image_uri,
            "duration": duration
        }
        
        response = requests.post(create_url, json=payload, headers=self.headers)
        response.raise_for_status()
        
        task_data = response.json()
        task_id = task_data['id']
        print(f"Task created with ID: {task_id}")
        
        # Poll for completion
        poll_url = f"{self.api_base}/tasks/{task_id}"
        max_attempts = 120  # 10 minutes with 5 second intervals
        
        for attempt in range(max_attempts):
            time.sleep(5)  # Poll every 5 seconds
            
            response = requests.get(poll_url, headers=self.headers)
            response.raise_for_status()
            
            task_status = response.json()
            status = task_status['status']
            
            print(f"Task status: {status} (attempt {attempt + 1}/{max_attempts})")
            
            if status == 'SUCCEEDED':
                # Download the output video
                video_url = task_status['output'][0]
                
                # Determine output path
                if output_dir is None:
                    output_dir = os.path.dirname(driver_video_path)
                
                output_path = os.path.join(output_dir, "runway_act_two_output.mp4")
                
                # Download the video
                print(f"Downloading video...")
                response = requests.get(video_url)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Generated video saved to: {output_path}")
                return output_path
            
            elif status == 'FAILED':
                raise RuntimeError(f"Task failed: {task_status.get('error', 'Unknown error')}")
            
            elif status == 'CANCELED':
                raise RuntimeError("Task was canceled")
        
        raise TimeoutError(f"Task did not complete within {max_attempts * 5} seconds")
    
    def _file_to_data_uri(self, file_path: str, mime_type: str) -> str:
        """Convert a file to a data URI"""
        import base64
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        base64_data = base64.b64encode(file_data).decode('utf-8')
        return f"data:{mime_type};base64,{base64_data}"
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get the duration of a video in seconds"""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True, text=True, check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return 5.0  # Default to 5 seconds


if __name__ == "__main__":
    # Test the Act-Two interface
    act_two = RunwayActTwo()
    
    # Test with sample files if they exist
    driver_video = "runway_subvideo.mp4"
    character_image = "runway_character.png"
    
    if os.path.exists(driver_video) and os.path.exists(character_image):
        result = act_two.generate_character_performance(
            driver_video,
            character_image
        )
        print(f"Generated: {result}")
    else:
        print(f"Test files not found: {driver_video}, {character_image}")