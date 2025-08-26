"""
Segment extraction utilities using rembg.
"""

import numpy as np
from PIL import Image
from rembg import remove, new_session


# Create a global session for efficiency
session = new_session('u2net')


def extract_foreground_mask(image, model='u2net'):
    """
    Extract foreground mask from an image using rembg.
    
    Parameters:
    ----------
    image : np.ndarray or PIL.Image
        Input image (RGB)
    model : str
        Model to use for segmentation (default 'u2net')
        
    Returns:
    -------
    np.ndarray
        Binary mask where 255 = foreground, 0 = background
    """
    # Convert numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Remove background to get RGBA image
    output = remove(image, session=session)
    
    # Extract alpha channel as mask
    output_np = np.array(output)
    if output_np.shape[2] == 4:
        mask = output_np[:, :, 3]
    else:
        # If no alpha channel, create a mask based on non-black pixels
        mask = np.any(output_np != 0, axis=2).astype(np.uint8) * 255
    
    # Ensure binary mask
    mask = (mask > 128).astype(np.uint8) * 255
    
    return mask


def extract_foreground_rgba(image, model='u2net'):
    """
    Extract foreground with alpha channel from an image.
    
    Parameters:
    ----------
    image : np.ndarray or PIL.Image
        Input image (RGB)
    model : str
        Model to use for segmentation (default 'u2net')
        
    Returns:
    -------
    np.ndarray
        RGBA image with transparent background
    """
    # Convert numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Remove background to get RGBA image
    output = remove(image, session=session)
    
    return np.array(output)