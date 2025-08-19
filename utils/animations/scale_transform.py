"""
ScaleTransformAnimation - Base class for animations that transform element size.
"""

from typing import Tuple, Optional, List
from .animate import Animation


class ScaleTransformAnimation(Animation):
    """
    Base class for animations that involve scaling or resizing elements.
    
    This class extends Animation to provide size transformation capabilities
    for effects like zoom, stretch, grow, shrink, etc.
    
    Additional Parameters:
    ---------------------
    start_width : int, optional
        Starting width in pixels (None = use original)
    start_height : int, optional  
        Starting height in pixels (None = use original)
    end_width : int, optional
        Ending width in pixels (None = maintain start_width)
    end_height : int, optional
        Ending height in pixels (None = maintain start_height)
    maintain_aspect_ratio : bool
        Whether to maintain aspect ratio during scaling (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        start_width: Optional[int] = None,
        start_height: Optional[int] = None,
        end_width: Optional[int] = None,
        end_height: Optional[int] = None,
        maintain_aspect_ratio: bool = True,
        direction: float = 0,
        start_frame: int = 0,
        animation_start_frame: int = 0,
        path: Optional[List[Tuple[int, int, int]]] = None,
        fps: int = 30,
        duration: float = 7.0,
        temp_dir: Optional[str] = None
    ):
        # Initialize parent class
        super().__init__(
            element_path=element_path,
            background_path=background_path,
            position=position,
            direction=direction,
            start_frame=start_frame,
            animation_start_frame=animation_start_frame,
            path=path,
            fps=fps,
            duration=duration,
            temp_dir=temp_dir
        )
        
        # Store size transformation parameters
        self.start_width = start_width
        self.start_height = start_height
        self.end_width = end_width if end_width is not None else start_width
        self.end_height = end_height if end_height is not None else start_height
        self.maintain_aspect_ratio = maintain_aspect_ratio
        
        # Will store original dimensions after loading
        self.original_width = None
        self.original_height = None
    
    def calculate_dimensions_at_frame(self, frame: int) -> Tuple[int, int]:
        """
        Calculate the element dimensions at a specific frame.
        
        Parameters:
        -----------
        frame : int
            The frame number to calculate dimensions for
            
        Returns:
        --------
        Tuple[int, int]
            Width and height in pixels at the specified frame
        """
        if frame < self.animation_start_frame:
            # Before animation starts, use start dimensions
            width = self.start_width if self.start_width else self.original_width
            height = self.start_height if self.start_height else self.original_height
        elif frame >= self.animation_start_frame + self.total_frames:
            # After animation ends, use end dimensions
            width = self.end_width if self.end_width else self.start_width
            height = self.end_height if self.end_height else self.start_height
        else:
            # During animation, interpolate
            progress = (frame - self.animation_start_frame) / self.total_frames
            
            start_w = self.start_width if self.start_width else self.original_width
            start_h = self.start_height if self.start_height else self.original_height
            end_w = self.end_width if self.end_width else start_w
            end_h = self.end_height if self.end_height else start_h
            
            width = int(start_w + (end_w - start_w) * progress)
            height = int(start_h + (end_h - start_h) * progress)
        
        if self.maintain_aspect_ratio and self.original_width and self.original_height:
            # Adjust to maintain aspect ratio
            original_ratio = self.original_width / self.original_height
            current_ratio = width / height if height > 0 else original_ratio
            
            if abs(current_ratio - original_ratio) > 0.01:  # Tolerance for floating point
                # Adjust height to maintain ratio
                height = int(width / original_ratio)
        
        return width, height
    
    def get_scale_factor_at_frame(self, frame: int) -> Tuple[float, float]:
        """
        Calculate the scale factors at a specific frame.
        
        Parameters:
        -----------
        frame : int
            The frame number to calculate scale for
            
        Returns:
        --------
        Tuple[float, float]
            Scale factors (x_scale, y_scale) relative to original size
        """
        width, height = self.calculate_dimensions_at_frame(frame)
        
        if self.original_width and self.original_height:
            x_scale = width / self.original_width
            y_scale = height / self.original_height
        else:
            x_scale = 1.0
            y_scale = 1.0
        
        return x_scale, y_scale