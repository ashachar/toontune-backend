"""Main 3D letter dissolve animation class."""
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Any
from .timing import TimingCalculator
from .renderer import Letter3DRenderer
from .sprite_manager import SpriteManager
from .occlusion import OcclusionHandler
from .frame_renderer import FrameRenderer
from .handoff import HandoffHandler


class Letter3DDissolve:
    """3D letter-by-letter dissolve animation with frame-accurate timing."""
    
    def __init__(
        self,
        duration: float = 1.5,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = "HELLO",
        font_size: int = 120,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        depth_color: Tuple[int, int, int] = (200, 170, 0),
        depth_layers: int = 8,
        depth_offset: int = 3,
        initial_scale: float = 0.9,
        initial_position: Optional[Tuple[int, int]] = None,
        stable_duration: float = 0.2,
        stable_alpha: float = 0.3,
        dissolve_duration: float = 0.8,
        dissolve_stagger: float = 0.1,
        float_distance: float = 50,
        max_dissolve_scale: float = 1.3,
        randomize_order: bool = False,
        segment_mask: Optional[np.ndarray] = None,
        is_behind: bool = False,
        shadow_offset: int = 5,  # Kept for compatibility but not used
        outline_width: int = 2,  # Kept for compatibility but not used
        supersample_factor: int = 2,
        post_fade_seconds: float = 0.10,
        pre_dissolve_hold_frames: int = 1,
        ensure_no_gap: bool = True,
        font_path: Optional[str] = None,
        debug: bool = False,
    ):
        # Store parameters
        self.duration = duration
        self.fps = fps
        self.total_frames = int(round(duration * fps))
        self.resolution = resolution
        self.text = text
        self.font_size = font_size
        self.initial_scale = initial_scale
        self.initial_position = initial_position or (resolution[0] // 2, resolution[1] // 2)
        self.stable_duration = stable_duration
        self.stable_alpha = max(0.0, min(1.0, stable_alpha))
        self.dissolve_duration = dissolve_duration
        self.dissolve_stagger = dissolve_stagger
        self.float_distance = float_distance
        self.max_dissolve_scale = max(1.0, max_dissolve_scale)
        self.randomize_order = randomize_order
        self.segment_mask = segment_mask
        self.is_behind = is_behind
        self.post_fade_seconds = max(0.0, post_fade_seconds)
        self.pre_dissolve_hold_frames = max(0, int(pre_dissolve_hold_frames))
        self.ensure_no_gap = ensure_no_gap
        self.debug = debug
        
        # Initialize components
        self.renderer = Letter3DRenderer(
            font_size=font_size,
            text_color=text_color,
            depth_color=depth_color,
            depth_layers=depth_layers,
            depth_offset=depth_offset,
            supersample_factor=supersample_factor,
            font_path=font_path,
            debug=debug
        )
        
        self.sprite_manager = SpriteManager(self.renderer, debug)
        self.timing_calc = TimingCalculator(fps, self.total_frames, debug)
        self.occlusion = OcclusionHandler(debug)
        self.frame_renderer = FrameRenderer(debug)
        self.handoff = HandoffHandler(self.sprite_manager, self.timing_calc, debug)
        
        # Store handoff alpha for opacity fix
        self._handoff_sprite_alpha: Optional[float] = None
    
    def set_initial_state(
        self, 
        scale: float, 
        position: Tuple[int, int], 
        alpha: float = None,
        is_behind: bool = None, 
        segment_mask: np.ndarray = None,
        rendered_text: Image.Image = None, 
        text_topleft: Tuple[int, int] = None,
        letter_sprites: List[Any] = None
    ):
        """Handle handoff from motion animation."""
        self.initial_scale = scale
        self.initial_position = position
        
        result = self.handoff.handle_initial_state(
            scale, position, alpha, is_behind, segment_mask, letter_sprites,
            self.text, self.randomize_order, self.stable_duration,
            self.dissolve_stagger, self.dissolve_duration, self.post_fade_seconds,
            self.pre_dissolve_hold_frames, self.ensure_no_gap
        )
        
        if result['stable_alpha'] is not None:
            self.stable_alpha = result['stable_alpha']
        if result['is_behind'] is not None:
            self.is_behind = result['is_behind']
        if result['segment_mask'] is not None:
            self.segment_mask = result['segment_mask']
        self._handoff_sprite_alpha = result['handoff_sprite_alpha']
    
    def generate_frame(self, frame_number: int, background: np.ndarray) -> np.ndarray:
        """Generate a single frame of the dissolve animation."""
        # Debug logging
        if frame_number % 5 == 0:
            print(f"\n[MASK_DEBUG] Frame {frame_number}: generate_frame called, is_behind={self.is_behind}")
        
        # Ensure sprites exist
        if not self.sprite_manager.letter_sprites:
            self.sprite_manager.prepare_letter_sprites(self.text, self.initial_position, self.initial_scale)
            self.sprite_manager.init_dissolve_order(self.text, self.randomize_order)
            self.timing_calc.build_frame_timeline(
                self.sprite_manager.dissolve_order,
                self.stable_duration,
                self.dissolve_stagger,
                self.dissolve_duration,
                self.post_fade_seconds,
                self.pre_dissolve_hold_frames,
                self.ensure_no_gap
            )
        
        # Prepare frame
        frame = background.copy()
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        
        # Extract fresh mask if needed
        current_mask = None
        if self.is_behind:
            current_mask = self.occlusion.extract_fresh_mask(background, frame_number, self.resolution)
        
        canvas = Image.fromarray(frame)
        
        # Process each letter
        timeline = self.timing_calc.get_timeline()
        hold_logged = self.timing_calc.get_hold_logged()
        entered_dissolve_logged = self.timing_calc.get_entered_dissolve_logged()
        
        for idx in self.sprite_manager.dissolve_order:
            sprite = self.sprite_manager.letter_sprites[idx]
            if sprite.sprite_3d is None:
                continue
            
            timing = timeline[idx]
            
            # Compute letter state
            phase, target_alpha, scale, float_y, add_holes = self.frame_renderer.compute_letter_state(
                frame_number, timing, self.stable_alpha,
                self.max_dissolve_scale, self.float_distance
            )
            
            # CRITICAL FIX: Don't skip "gone" letters if they need occlusion processing
            # They should still be rendered (with alpha=0) to update their masks
            if phase == "gone" and not self.is_behind:
                # Only skip if not behind foreground
                continue
            elif phase == "gone" and self.is_behind:
                # Force invisible but still process for occlusion updates
                target_alpha = 0.0
                if self.debug and frame_number % 10 == 0:
                    print(f"[FIX_DEBUG] Frame {frame_number}: Processing 'gone' letter '{sprite.char}' for occlusion update")
            
            # Log phase transitions
            if phase == "hold" and not hold_logged.get(idx):
                if self.debug:
                    print(f"[JUMP_CUT] '{sprite.char}' enters HOLD at frame {frame_number}")
                hold_logged[idx] = True
            elif phase == "dissolve" and not entered_dissolve_logged.get(idx):
                if self.debug:
                    print(f"[JUMP_CUT] '{sprite.char}' begins DISSOLVE at frame {frame_number}")
                entered_dissolve_logged[idx] = True
            
            # Add dissolve holes if needed (skip for gone phase)
            if phase != "gone":
                if add_holes and phase == "dissolve":
                    letter_t = (frame_number - timing.hold_end) / max(1, (timing.end - timing.hold_end))
                    self.sprite_manager.add_dissolve_holes(idx, letter_t)
                elif phase == "fade" and idx not in self.sprite_manager.letter_kill_masks:
                    # Ensure it's "holey" in fade
                    self.sprite_manager.letter_kill_masks[idx] = np.ones(
                        (sprite.sprite_3d.height, sprite.sprite_3d.width), dtype=np.uint8
                    )
            
            # Transform sprite
            sprite_img, (pos_x, pos_y) = self.frame_renderer.transform_sprite(
                sprite, scale, float_y, frame_number
            )
            
            sprite_array = np.array(sprite_img)
            
            # Apply alpha and kill mask
            kill_mask = self.sprite_manager.letter_kill_masks.get(idx)
            sprite_array = self.frame_renderer.apply_alpha_and_kill_mask(
                sprite_array, target_alpha, kill_mask,
                (sprite_img.width, sprite_img.height),
                self._handoff_sprite_alpha,
                frame_number, sprite.char, phase, timing.start
            )
            
            # Apply occlusion if behind
            if self.is_behind and current_mask is not None:
                sprite_array = self.occlusion.apply_occlusion(
                    sprite_array, (pos_x, pos_y), current_mask,
                    self.resolution, frame_number, sprite.char
                )
            
            sprite_img = Image.fromarray(sprite_array)
            canvas.paste(sprite_img, (int(pos_x), int(pos_y)), sprite_img)
        
        result = np.array(canvas)
        return result[:, :, :3] if result.shape[2] == 4 else result