#!/usr/bin/env python3
"""
Apply 3D text motion + dissolve to any video with **high-quality edges**.

Key improvements:
- Always use a vector font (via --font) or robust auto-discovery. Logs with [TEXT_QUALITY].
- Lossless PNG intermediate to avoid mid-pipeline compression loss.
- Final encode in 4:4:4 (yuv444p) or RGB (libx264rgb) to preserve colored edges.
- Debug logs to verify quality choices.

Usage examples:
  python apply_3d_text_animation.py input.mp4 --text "AWESOME" \
      --font "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" \
      --pixfmt yuv444p --crf 18 --preset slow --supersample 12 --debug

  python apply_3d_text_animation.py input.mp4 --text "AI" --pixfmt rgb24 \
      --font "/Library/Fonts/Arial.ttf" --supersample 12
"""

import cv2
import numpy as np
import subprocess
import argparse
import shutil
import os
from pathlib import Path
from .text_3d_motion import Text3DMotion
from .letter_3d_dissolve import Letter3DDissolve
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_placement.optimal_position_finder import OptimalTextPositionFinder, estimate_text_size


def parse_position(position_str, width, height):
    positions = {
        'center': (width // 2, height // 2),
        'top': (width // 2, height // 4),
        'bottom': (width // 2, 3 * height // 4),
        'left': (width // 4, height // 2),
        'right': (3 * width // 4, height // 2),
        'top-left': (width // 4, height // 4),
        'top-right': (3 * width // 4, height // 4),
        'bottom-left': (width // 4, 3 * height // 4),
        'bottom-right': (3 * width // 4, 3 * height // 4),
    }
    
    # Handle tuple input (e.g., (x, y))
    if isinstance(position_str, tuple) and len(position_str) == 2:
        try:
            x, y = int(position_str[0]), int(position_str[1])
            return (x, y)
        except (ValueError, TypeError):
            print(f"[JUMP_CUT] Warning: Invalid position tuple '{position_str}', using center")
            return positions['center']
    
    # Handle string input
    if position_str in positions:
        return positions[position_str]
    try:
        x, y = map(int, position_str.split(','))
        return (x, y)
    except Exception:
        print(f"[JUMP_CUT] Warning: Invalid position '{position_str}', using center")
        return positions['center']


def extract_foreground_mask_safe(frame):
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from video.segmentation.segment_extractor import extract_foreground_mask
        mask = extract_foreground_mask(frame)
        print(f"[JUMP_CUT] Foreground mask extracted: {mask.shape}")
        return mask
    except Exception as e:
        print(f"[JUMP_CUT] No foreground mask (text won't go behind objects): {e}")
        return None


def apply_animation_to_video(
    video_path,
    text="HELLO WORLD",
    output_path=None,
    motion_duration=0.75,
    dissolve_duration=1.5,
    position='center',
    final_opacity=0.5,
    font_size=140,
    text_color=(255, 220, 0),
    start_scale=2.0,
    end_scale=1.0,
    final_scale=0.9,
    loop=False,
    extract_mask=True,
    supersample=12,
    font_path=None,
    pixfmt="yuv444p",        # yuv444p | yuv420p | yuv422p | rgb24
    crf=18,
    preset="slow",
    tune=None,               # e.g., "animation" or None
    keep_frames=False,
    auto_position=False,    # NEW: automatically find optimal position
    debug=False
):
    """
    Apply 3D text animation to a video file and encode with crisp edges.

    Returns: Path to final mp4 file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    print(f"[JUMP_CUT] Input video: {video_path}")
    print(f"[JUMP_CUT] Resolution: {width}x{height}, FPS: {fps}")
    print(f"[JUMP_CUT] Duration: {video_duration:.2f}s ({total_frames} frames)")
    print(f"[JUMP_CUT] Text: '{text}'")
    print(f"[TEXT_QUALITY] Supersample requested: {supersample}")
    if font_path:
        print(f"[TEXT_QUALITY] Font path requested: {font_path}")

    # Determine text position
    if auto_position:
        print(f"[TEXT_PLACEMENT] Finding optimal position for text placement...")
        
        # Estimate text size for analysis
        text_width, text_height = estimate_text_size(text, font_size, final_scale)
        motion_frames = int(round(motion_duration * fps))
        
        # Find optimal position
        position_finder = OptimalTextPositionFinder(
            text_width=text_width,
            text_height=text_height,
            motion_frames=motion_frames,
            sample_rate=3,
            grid_divisions=6,  # 6x6 grid for faster processing
            debug=debug
        )
        
        try:
            text_position = position_finder.find_optimal_position(
                video_path,
                prefer_center=True,
                center_weight=0.15  # Slight preference for center
            )
            print(f"[TEXT_PLACEMENT] Optimal position found: {text_position}")
        except Exception as e:
            print(f"[TEXT_PLACEMENT] Failed to find optimal position: {e}")
            print(f"[TEXT_PLACEMENT] Falling back to center position")
            text_position = (width // 2, height // 2)
    else:
        text_position = parse_position(position, width, height)
    
    print(f"[JUMP_CUT] Text position: {text_position}")

    motion_frames = int(round(motion_duration * fps))
    dissolve_frames = int(round(dissolve_duration * fps))
    animation_frames = motion_frames + dissolve_frames
    animation_duration = animation_frames / fps

    print(f"[JUMP_CUT] Animation: {animation_duration:.2f}s ({animation_frames} frames)")
    print(f"[JUMP_CUT]   Motion: {motion_duration:.2f}s ({motion_frames} frames)")
    print(f"[JUMP_CUT]   Dissolve: {dissolve_duration:.2f}s ({dissolve_frames} frames)")

    # Optional foreground mask from first frame
    segment_mask = None
    if extract_mask:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if ret:
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            segment_mask = extract_foreground_mask_safe(first_frame_rgb)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create animations (pass font_path to ensure vector font usage)
    motion = Text3DMotion(
        duration=motion_duration, fps=fps, resolution=(width, height),
        text=text, segment_mask=segment_mask, font_size=font_size,
        text_color=text_color, depth_color=tuple(int(c * 0.8) for c in text_color),
        depth_layers=8, depth_offset=3,
        start_scale=start_scale, end_scale=end_scale, final_scale=final_scale,
        start_position=text_position, end_position=text_position,
        shrink_duration=motion_duration * 0.8, settle_duration=motion_duration * 0.2,
        final_alpha=final_opacity, shadow_offset=6, outline_width=2,
        perspective_angle=0, supersample_factor=supersample,
        glow_effect=True, font_path=font_path, debug=debug,
    )

    dissolve = Letter3DDissolve(
        duration=dissolve_duration, fps=fps, resolution=(width, height),
        text=text, font_size=font_size,
        text_color=text_color, depth_color=tuple(int(c * 0.8) for c in text_color),
        depth_layers=8, depth_offset=3,
        initial_scale=final_scale, initial_position=text_position,
        stable_duration=0.1, stable_alpha=final_opacity,
        dissolve_duration=0.5, dissolve_stagger=0.1,
        float_distance=40, max_dissolve_scale=1.3,
        randomize_order=False, segment_mask=segment_mask, is_behind=True,
        shadow_offset=6, outline_width=2, supersample_factor=supersample,
        post_fade_seconds=0.10, pre_dissolve_hold_frames=1, ensure_no_gap=True,
        font_path=font_path, debug=debug,
    )

    # Prepare frames directory for lossless PNG intermediate
    if output_path is None:
        input_path = Path(video_path)
        output_path = input_path.parent / f"{input_path.stem}_3d_text.mp4"
    output_path = Path(output_path)

    frames_dir = output_path.parent / f"{output_path.stem}_frames_tmp"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TEXT_QUALITY] Using lossless PNG intermediate: {frames_dir}")

    print(f"[JUMP_CUT] Rendering frames...")
    frame_count = 0
    animation_cycle = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if loop:
            anim_frame = frame_count % animation_frames
            if anim_frame == 0 and frame_count > 0:
                animation_cycle += 1
                print(f"[JUMP_CUT] Starting animation cycle {animation_cycle + 1}")
        else:
            anim_frame = frame_count if frame_count < animation_frames else -1

        if anim_frame >= 0:
            if anim_frame < motion_frames:
                frame_rgb = motion.generate_frame(anim_frame, frame_rgb)
                if anim_frame == motion_frames - 1:
                    final_state = motion.get_final_state()
                    if final_state and (frame_count < motion_frames or loop):
                        if debug:
                            print(f"[JUMP_CUT] Handoff at frame {frame_count}: "
                                  f"center={final_state.center_position}, scale={final_state.scale:.3f}, "
                                  f"alpha={final_state.alpha:.3f}")
                        dissolve.set_initial_state(
                            scale=final_state.scale,
                            position=final_state.center_position,
                            alpha=final_state.alpha,  # Use the alpha from motion's final state
                            is_behind=final_state.is_behind,
                            segment_mask=segment_mask,
                            letter_sprites=final_state.letter_sprites  # Pass the letter sprites!
                        )
            else:
                dissolve_frame = anim_frame - motion_frames
                frame_rgb = dissolve.generate_frame(dissolve_frame, frame_rgb)

        # Save as PNG (lossless)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        png_path = frames_dir / f"frame_{frame_count:06d}.png"
        cv2.imwrite(str(png_path), frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        frame_count += 1
        if frame_count % (fps * 5) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"[JUMP_CUT] Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

    cap.release()

    # Final encode from PNG sequence
    final_out = output_path.parent / f"{output_path.stem}_hq.mp4"
    codec = "libx264rgb" if pixfmt.lower() == "rgb24" else "libx264"
    pixfmt_arg = "rgb24" if pixfmt.lower() == "rgb24" else pixfmt

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-s", f"{width}x{height}",
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", pixfmt_arg,
        "-movflags", "+faststart",
        str(final_out)
    ]
    if tune:
        # insert after codec args
        cmd.insert(cmd.index("-crf"), "-tune")
        cmd.insert(cmd.index("-tune") + 1, tune)

    print(f"[TEXT_QUALITY] Final encode: codec={codec} pix_fmt={pixfmt_arg} crf={crf} preset={preset} "
          f"{'(tune=' + tune + ')' if tune else ''}")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode("utf-8", errors="ignore"))
        raise

    # Cleanup
    if not keep_frames:
        try:
            shutil.rmtree(frames_dir)
        except Exception:
            pass

    print(f"[JUMP_CUT] ✅ Final video: {final_out}")
    print(f"[TEXT_QUALITY] To verify pixel format, run:")
    print(f"  ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt -of default=nw=1:nk=1 {final_out}")
    return str(final_out)


def main():
    parser = argparse.ArgumentParser(
        description="Apply 3D text animation to any video (high-quality edges).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4
  %(prog)s video.mp4 --text "AWESOME" --font "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
  %(prog)s video.mp4 --text "SUBSCRIBE" --position bottom --loop
  %(prog)s video.mp4 --text "2024" --color 255,0,0 --size 200 --pixfmt yuv444p
  %(prog)s video.mp4 --position 100,200 --motion-duration 2.0 --supersample 12
        """
    )

    parser.add_argument('video', help='Input video file path')
    parser.add_argument('--text', default='HELLO WORLD', help='Text to animate (default: HELLO WORLD)')
    parser.add_argument('--output', '-o', help='Output video path base name (we add _hq.mp4)')

    # Timing
    parser.add_argument('--motion-duration', type=float, default=0.75, help='Motion phase duration (sec)')
    parser.add_argument('--dissolve-duration', type=float, default=1.5, help='Dissolve phase duration (sec)')
    parser.add_argument('--loop', action='store_true', help='Loop animation throughout video')

    # Position
    parser.add_argument('--position', '-p', default='center',
                        help='Text position: center, top, bottom, left, right, or x,y (default: center)')
    parser.add_argument('--auto-position', action='store_true',
                        help='Automatically find optimal position for maximum visibility')

    # Appearance
    parser.add_argument('--size', type=int, default=140, help='Font size (default: 140)')
    parser.add_argument('--color', default='255,220,0', help='Text color as R,G,B (default: 255,220,0)')
    parser.add_argument('--opacity', type=float, default=0.5, help='Final text opacity 0.0-1.0 (default: 0.5)')
    parser.add_argument('--start-scale', type=float, default=2.0, help='Initial scale of text')
    parser.add_argument('--end-scale', type=float, default=1.0, help='Scale at end of motion')
    parser.add_argument('--final-scale', type=float, default=0.9, help='Final scale during dissolve')

    # Quality / system options
    parser.add_argument('--no-mask', action='store_true',
                        help="Disable foreground mask extraction (text won't go behind objects)")
    parser.add_argument('--supersample', type=int, default=12,
                        help='Supersampling factor for text quality, higher=better (default: 12)')
    parser.add_argument('--font', help='Path to a TTF/OTF font (recommended).')
    parser.add_argument('--pixfmt', default='yuv444p', choices=['yuv444p', 'yuv422p', 'yuv420p', 'rgb24'],
                        help='Pixel format for final encode (yuv444p recommended).')
    parser.add_argument('--crf', type=int, default=18, help='CRF for x264 (lower=better). Default: 18')
    parser.add_argument('--preset', default='slow', help='x264 preset (ultrafast..placebo). Default: slow')
    parser.add_argument('--tune', default=None, help='x264 tune, e.g., animation, film, grain (optional)')
    parser.add_argument('--keep-frames', action='store_true', help='Keep PNG frames (debugging).')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    try:
        color = tuple(map(int, args.color.split(',')))
        if len(color) != 3 or any(c < 0 or c > 255 for c in color):
            raise ValueError
    except Exception:
        print(f"[JUMP_CUT] Warning: Invalid color '{args.color}', using default")
        color = (255, 220, 0)

    try:
        output = apply_animation_to_video(
            video_path=args.video,
            text=args.text,
            output_path=args.output,
            motion_duration=args.motion_duration,
            dissolve_duration=args.dissolve_duration,
            position=args.position,
            final_opacity=args.opacity,
            font_size=args.size,
            text_color=color,
            start_scale=args.start_scale,
            end_scale=args.end_scale,
            final_scale=args.final_scale,
            loop=args.loop,
            extract_mask=not args.no_mask,
            supersample=args.supersample,
            font_path=args.font,
            pixfmt=args.pixfmt,
            crf=args.crf,
            preset=args.preset,
            tune=args.tune,
            keep_frames=args.keep_frames,
            auto_position=args.auto_position,
            debug=args.debug
        )
        print(f"\n✅ Success! Output saved to: {output}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())