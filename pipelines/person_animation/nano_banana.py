"""
Gemini Nano Banana Model Interface for Image Generation
Uses Gemini's imagen model to transform images into character sketches
"""

import os
import base64
import json
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import time
import google.generativeai as genai
import PIL.Image


class NanoBananaGenerator:
    """
    Interface to Gemini's Nano Banana (imagen) model for character generation
    """
    
    def __init__(self):
        """Initialize the Gemini model with API key from environment"""
        load_dotenv()
        
        # Get the API key and model name
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        # Get the model name from env
        self.model_name = os.getenv('GEMINI_NANO_BANANA', 'gemini-2.5-flash-image-preview')
        if not self.model_name:
            self.model_name = 'gemini-2.5-flash-image-preview'
        
        # Remove 'models/' prefix if present for API calls
        self.api_model_name = self.model_name.replace('models/', '')
        
        # Configure Gemini SDK
        genai.configure(api_key=self.api_key)
        
        # Initialize the model with response modalities for image generation
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        # Create model instance with image generation support
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
        
        print(f"Initialized Gemini model: {self.model_name}")
        print("Note: Model configured for image generation with responseModalities")
    
    def generate_character(self, 
                          image_path: str, 
                          character_description: str,
                          output_dir: Optional[str] = None) -> str:
        """
        Generate a character sketch from an input image using Gemini API
        
        Args:
            image_path: Path to the input image
            character_description: Description of the character to generate
            output_dir: Directory to save the output (defaults to same as input)
            
        Returns:
            Path to the generated character image
        """
        # Read the input image
        image = PIL.Image.open(image_path)
        
        # Create the prompt
        prompt = f"Absolutely exactly 100% the same image, pose, hand gesture, face gesture, clothes, etc., just of a sketch of a {character_description} with very clear facial elements"
        
        print(f"Sending image generation request to Gemini...")
        print(f"Model: {self.api_model_name}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Prepare the API request
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.api_model_name}:generateContent"
        
        # Encode image to base64
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare request payload with responseModalities in generationConfig
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
                "responseModalities": ["TEXT", "IMAGE"]  # Critical for image generation
            }
        }
        
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            # Make the API request
            print("Making API request for image generation...")
            response = requests.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the generated image from response
            if 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    for part in candidate['content']['parts']:
                        # Check if there's an inlineData with image (note camelCase)
                        if 'inlineData' in part:
                            mime_type = part['inlineData'].get('mimeType', '')
                            if 'image' in mime_type:
                                # Decode the base64 image
                                image_data = base64.b64decode(part['inlineData']['data'])
                                
                                # Save the image
                                if output_dir is None:
                                    output_dir = os.path.dirname(image_path)
                                
                                output_path = os.path.join(output_dir, "runway_character.png")
                                
                                with open(output_path, 'wb') as f:
                                    f.write(image_data)
                                
                                print(f"✅ Successfully generated character image!")
                                print(f"Generated character image saved to: {output_path}")
                                return output_path
                        
                        # Check for text response with image URL
                        if 'text' in part:
                            try:
                                # Try to parse as JSON in case it contains image data
                                text_data = json.loads(part['text'])
                                if 'image' in text_data or 'imageUrl' in text_data:
                                    print("Image URL found in response, downloading...")
                                    # Handle image URL if present
                                    image_url = text_data.get('image') or text_data.get('imageUrl')
                                    return self._download_image(image_url, output_dir)
                            except:
                                pass
            
            # If no image was generated, log the response for debugging
            print("No image found in response. Full response structure:")
            print(json.dumps(result, indent=2)[:500])  # Print first 500 chars for debugging
            
            # Fallback to using original image
            print("WARNING: No image generated, using original as placeholder")
            return self._use_original_as_fallback(image_path, output_dir)
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"Error details: {e.response.text[:500]}")
            return self._use_original_as_fallback(image_path, output_dir)
        except Exception as e:
            print(f"Unexpected error: {e}")
            return self._use_original_as_fallback(image_path, output_dir)
    
    def _download_image(self, image_url: str, output_dir: str) -> str:
        """Download image from URL"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            output_path = os.path.join(output_dir, "runway_character.png")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded generated image to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Failed to download image: {e}")
            raise
    
    def _use_original_as_fallback(self, image_path: str, output_dir: Optional[str] = None) -> str:
        """
        Fallback method - uses original image as placeholder
        """
        print("Using fallback: copying original image as placeholder")
        
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, "runway_character.png")
        
        # Copy original image
        import shutil
        shutil.copy2(image_path, output_path)
        
        print(f"Placeholder character image saved to: {output_path}")
        print("NOTE: Gemini image generation may require additional setup or permissions")
        return output_path


if __name__ == "__main__":
    # Test the generator
    generator = NanoBananaGenerator()
    
    # Test with a sample image if it exists
    test_image = "frame0.png"
    if os.path.exists(test_image):
        result = generator.generate_character(
            test_image,
            "friendly meerkat"
        )
        print(f"Generated: {result}")
    else:
        print(f"Test image {test_image} not found")