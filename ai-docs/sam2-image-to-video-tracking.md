# SAM2 Auto-Track Pipeline (Images → Video) — Step‑by‑Step Guide

> **Goal:** Use **SAM2 (images)** to propose masks on the **first frame** of your video, convert selected masks into robust **seed click points**, and then feed those into **SAM2 (video)** to track the objects across the entire clip.
>
> This guide gives you: environment setup, full runnable code, and production tips for cost, speed, and reliability.

---

## 1) Prerequisites & Environment Check

### What you need
- **Python 3.9+**
- A **Replicate API token** (GPU runs happen on Replicate’s side).
- (Optional) **FFmpeg** if you want true alpha-output workflows later (WebM VP9/AV1 with alpha or ProRes 4444).

### Verify your tools
```bash
python --version
pip --version
ffmpeg -version   # optional; only needed for alpha video muxing
```

### Set your Replicate API token
macOS/Linux:
```bash
export REPLICATE_API_TOKEN="YOUR_TOKEN_HERE"
```
Windows (PowerShell):
```powershell
setx REPLICATE_API_TOKEN "YOUR_TOKEN_HERE"
$env:REPLICATE_API_TOKEN  # confirm in current session if needed
```

---

## 2) Install Dependencies

Create a clean virtual environment (recommended) and install packages.

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

Create **`requirements.txt`** with:
```txt
replicate==0.32.1
opencv-python==4.10.0.84
numpy==1.26.4
requests==2.32.3
Pillow==10.4.0
scikit-image==0.24.0
```
Install:
```bash
pip install -r requirements.txt
```

---

## 3) How the Pipeline Works (High-Level)

1. **Extract frame 0** from your **target video** **without resizing** (to keep coordinates aligned).
2. Send that frame to **`meta/sam-2`** (images) which proposes a set of masks (no human clicks required).
3. **Filter** & **select** the objects you care about; compute **multiple interior seed points** per chosen object:
   - We place seeds at the centroid and interior peaks (via distance transform) to make masks robust.
4. Build the four strings required by **`meta/sam-2-video`**:
   - `click_coordinates` — `"[x,y],[x,y],..."`
   - `click_labels` — `"1,1,..."` (foreground) and, if needed, `"0"` for background.
   - `click_frames` — `"0,0,..."` (we seed on frame 0).
   - `click_object_ids` — `"obj_1,obj_1,obj_2,..."` (group seed points into distinct objects).
5. Run **SAM2 (video)** once to get masks tracked through the full video.
6. (Optional) Post-process masks to create **true-alpha** videos.

> **Key rule:** Keep frame 0 extraction and click generation in the *same pixel space* as the video you pass to SAM2-video (no scaling, padding, or letterboxing in between).

---

## 4) End-to-End Script (Complete)

Save the script below as **`sam2_autotrack_pipeline.py`**.

```python
#!/usr/bin/env python3
'''
Auto-bootstrap clicks for SAM2-video using masks from SAM2-image on frame 0.

Workflow:
  1) Read frame 0 from the input video (no resizing).
  2) Call meta/sam-2 (images) to get mask proposals.
  3) Filter + dedupe masks, then compute multiple interior seed points per object.
  4) Build click strings for meta/sam-2-video.
  5) Call meta/sam-2-video to track objects through the video.

Usage:
  export REPLICATE_API_TOKEN=...
  python sam2_autotrack_pipeline.py --video input.mp4 --out_dir out       --max_objects 5 --seeds_per_object 2       --mask_type binary --output_video true --video_fps 25

Outputs:
  - out/frame0.png                 : exact first frame used for seeding
  - out/masks_selected/*.png       : selected masks used to make clicks
  - out/clicks.json                : the four click strings (for reuse)
  - out/video_result.json          : JSON/URLs returned by SAM2-video
'''
import os
import io
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
import requests
from PIL import Image
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

import replicate

# --------------------------
# Helpers
# --------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_frame0_noresize(video_path: str) -> Tuple[np.ndarray, int, int]:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame 0 from video.")
    h, w = frame.shape[:2]
    return frame, w, h

def save_image(path: Path, img_bgr: np.ndarray):
    cv2.imwrite(str(path), img_bgr)

def download_to(path: Path, url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)

def binarize_mask_from_image(path: Path, expected_wh: Tuple[int, int]) -> np.ndarray:
    '''Load a mask image (PNG/WEBP/JPG) into a uint8 {0,1} array of shape HxW, resized if necessary.'''
    w_expect, h_expect = expected_wh
    img = Image.open(path).convert('L')
    if img.size != (w_expect, h_expect):
        img = img.resize((w_expect, h_expect), Image.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    # Heuristic: consider non-zero as foreground. If the mask is an overlay, threshold higher.
    _, thresh = cv2.threshold(arr, 1, 1, cv2.THRESH_BINARY)
    return thresh

def filter_and_pick_masks(masks: List[np.ndarray], min_area: int, max_objects: int) -> List[np.ndarray]:
    '''Filter small masks, dedupe by IoU, and pick up to max_objects by area.'''
    # Remove small blobs inside each mask
    cleaned = []
    for m in masks:
        lab = label(m.astype(bool))
        # Keep components > min_area
        keep = np.zeros_like(m, dtype=bool)
        for region in regionprops(lab):
            if region.area >= min_area:
                keep[tuple(zip(*region.coords))] = True
        cleaned.append(keep.astype(np.uint8))

    # Sort by area desc
    areas = [int(x.sum()) for x in cleaned]
    order = np.argsort(areas)[::-1]
    cleaned = [cleaned[i] for i in order]

    # Greedy dedupe by IoU
    picked = []
    for cand in cleaned:
        is_dup = False
        for p in picked:
            inter = (cand & p).sum()
            union = cand.sum() + p.sum() - inter
            iou = inter / (union + 1e-6)
            if iou > 0.9:
                is_dup = True
                break
        if not is_dup:
            picked.append(cand)
        if len(picked) == max_objects:
            break

    return picked

def seeds_for_mask(mask: np.ndarray, seeds_per_object: int) -> List[Tuple[int,int]]:
    '''Return up to N interior seed points (x,y) for one object mask using centroid + distance peaks.'''
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    # Centroid (integer)
    cx = int(xs.mean())
    cy = int(ys.mean())

    seeds = [(cx, cy)]

    # Distance transform for inner-most points
    dist = distance_transform_edt(mask.astype(np.uint8))
    # Find top K peaks by distance (avoid the centroid duplicate)
    # Sample a grid of candidate points to keep it robust and O(K)
    num_candidates = 500
    h, w = mask.shape
    rng = np.random.default_rng(42)
    cand_x = rng.integers(0, w, size=num_candidates)
    cand_y = rng.integers(0, h, size=num_candidates)
    cand_scores = dist[cand_y, cand_x] * mask[cand_y, cand_x]

    idxs = np.argsort(cand_scores)[::-1]
    for idx in idxs:
        x, y = int(cand_x[idx]), int(cand_y[idx])
        if mask[y, x] == 0:
            continue
        # keep points not too close to existing seeds
        too_close = False
        for sx, sy in seeds:
            if (sx - x)**2 + (sy - y)**2 < 25**2:
                too_close = True
                break
        if not too_close:
            seeds.append((x, y))
        if len(seeds) >= seeds_per_object:
            break

    return seeds[:seeds_per_object]

# --------------------------
# SAM2 calls
# --------------------------

def call_sam2_image(client: replicate.Client, frame0_path: Path,
                    points_per_side: int, pred_iou_thresh: float,
                    stability_score_thresh: float, use_m2m: bool) -> Any:
    print('[SAM2-image] Requesting masks...')
    with open(frame0_path, 'rb') as f:
        prediction = client.run(
            'meta/sam-2',
            input={
                'image': f,
                'points_per_side': points_per_side,
                'pred_iou_thresh': pred_iou_thresh,
                'stability_score_thresh': stability_score_thresh,
                'use_m2m': use_m2m,
            },
        )
    print('[SAM2-image] Got response.')
    return prediction

def extract_mask_urls_or_images(pred: Any) -> List[str]:
    '''
    Normalize SAM2-image output into a list of URLs (http/https) we can download.
    Some variants return list[str] of urls; others return list[dict] with 'mask' or 'image' fields.
    '''
    if isinstance(pred, dict) and 'output' in pred:
        pred = pred['output']
    urls = []
    if isinstance(pred, list):
        for item in pred:
            if isinstance(item, str) and item.startswith(('http://', 'https://')):
                urls.append(item)
            elif isinstance(item, dict):
                for key in ('mask', 'image', 'output', 'url'):
                    val = item.get(key)
                    if isinstance(val, str) and val.startswith(('http://', 'https://')):
                        urls.append(val)
    return urls

def call_sam2_video(client: replicate.Client, video_path: Path,
                    click_coordinates: str, click_labels: str,
                    click_frames: str, click_object_ids: str,
                    mask_type: str, output_video: bool,
                    video_fps: int, output_format: str,
                    output_quality: int, output_frame_interval: int) -> Any:
    print('[SAM2-video] Tracking with seeded clicks...')
    with open(video_path, 'rb') as f:
        prediction = client.run(
            'meta/sam-2-video',
            input={
                'input_video': f,
                'click_coordinates': click_coordinates,
                'click_labels': click_labels,
                'click_frames': click_frames,
                'click_object_ids': click_object_ids,
                'mask_type': mask_type,
                'output_video': output_video,
                'video_fps': video_fps,
                'output_format': output_format,
                'output_quality': output_quality,
                'output_frame_interval': output_frame_interval,
            },
        )
    print('[SAM2-video] Done.')
    return prediction

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='Path to input video (used for both frame0 and SAM2-video)')
    ap.add_argument('--out_dir', default='out', help='Output directory')
    ap.add_argument('--points_per_side', type=int, default=32)
    ap.add_argument('--pred_iou_thresh', type=float, default=0.88)
    ap.add_argument('--stability_score_thresh', type=float, default=0.95)
    ap.add_argument('--use_m2m', type=bool, default=True)

    ap.add_argument('--min_area', type=int, default=500)        # ignore tiny specks
    ap.add_argument('--max_objects', type=int, default=3)       # pick top N masks by area
    ap.add_argument('--seeds_per_object', type=int, default=2)  # seed points per object

    ap.add_argument('--mask_type', default='binary', choices=['binary', 'highlighted', 'greenscreen'])
    ap.add_argument('--output_video', type=lambda s: s.lower()=='true', default=True)
    ap.add_argument('--video_fps', type=int, default=25)
    ap.add_argument('--output_format', default='png')           # used when output_video=False
    ap.add_argument('--output_quality', type=int, default=80)   # jpg/webp quality
    ap.add_argument('--output_frame_interval', type=int, default=1)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    masks_dir = out_dir / 'masks_all'
    ensure_dir(masks_dir)
    sel_dir = out_dir / 'masks_selected'
    ensure_dir(sel_dir)

    # 1) Extract frame 0
    frame0_bgr, w, h = read_frame0_noresize(args.video)
    frame0_path = out_dir / 'frame0.png'
    save_image(frame0_path, frame0_bgr)
    print(f'[INFO] Saved frame 0 to {frame0_path} ({w}x{h}).')

    # 2) Call SAM2-image
    client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
    if client.api_token is None:
        raise RuntimeError('REPLICATE_API_TOKEN not set.')
    pred = call_sam2_image(
        client, frame0_path,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        use_m2m=args.use_m2m,
    )

    # 3) Materialize masks
    urls = extract_mask_urls_or_images(pred)
    mask_arrays: List[np.ndarray] = []
    for i, url in enumerate(urls):
        dst = masks_dir / f'mask_{i:03d}.png'
        try:
            download_to(dst, url)
            m = binarize_mask_from_image(dst, (w, h))
            if m.sum() > 0:
                mask_arrays.append(m)
        except Exception as e:
            print(f'[WARN] Failed to fetch/parse mask {i} from {url}: {e}')

    if not mask_arrays:
        raise RuntimeError('No usable masks returned by SAM2-image. Try relaxing thresholds or different content.')

    # 4) Filter, pick top objects, and compute seeds
    picked = filter_and_pick_masks(mask_arrays, min_area=args.min_area, max_objects=args.max_objects)
    if not picked:
        raise RuntimeError('No masks survived filtering. Consider lowering --min_area or increasing --max_objects.')

    for i, m in enumerate(picked):
        cv2.imwrite(str(sel_dir / f'obj_{i+1:02d}.png'), m * 255)

    click_coords: List[Tuple[int,int]] = []
    click_labels: List[int] = []
    click_frames: List[int] = []
    click_obj_ids: List[str] = []

    for idx, m in enumerate(picked, start=1):
        seeds = seeds_for_mask(m, seeds_per_object=args.seeds_per_object)
        obj_id = f'obj_{idx}'
        for (x, y) in seeds:
            click_coords.append((x, y))
            click_labels.append(1)   # foreground seed
            click_frames.append(0)   # frame 0
            click_obj_ids.append(obj_id)

    if not click_coords:
        raise RuntimeError('No seed points computed. Increase --seeds_per_object or check masks.')

    # Build strings in Replicate's expected formats
    coords_str = ','.join([f'[{x},{y}]' for (x,y) in click_coords])
    labels_str = ','.join([str(v) for v in click_labels])
    frames_str = ','.join([str(v) for v in click_frames])
    ids_str    = ','.join(click_obj_ids)

    clicks_json = {
        'click_coordinates': coords_str,
        'click_labels': labels_str,
        'click_frames': frames_str,
        'click_object_ids': ids_str,
        'num_points': len(click_coords),
    }
    (out_dir / 'clicks.json').write_text(json.dumps(clicks_json, indent=2))
    print('[INFO] Seed clicks written to', out_dir / 'clicks.json')

    # 5) Call SAM2-video with seeded clicks
    vid_pred = call_sam2_video(
        client, Path(args.video),
        click_coordinates=coords_str,
        click_labels=labels_str,
        click_frames=frames_str,
        click_object_ids=ids_str,
        mask_type=args.mask_type,
        output_video=args.output_video,
        video_fps=args.video_fps,
        output_format=args.output_format,
        output_quality=args.output_quality,
        output_frame_interval=args.output_frame_interval,
    )
    (out_dir / 'video_result.json').write_text(json.dumps(vid_pred, indent=2, default=str))
    print('[INFO] SAM2-video result written to', out_dir / 'video_result.json')

if __name__ == '__main__':
    main()
```

---

## 5) Run the Pipeline

```bash
# 1) Activate your venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2) Ensure your token is set
export REPLICATE_API_TOKEN="YOUR_TOKEN_HERE"

# 3) Run
python sam2_autotrack_pipeline.py   --video input.mp4   --out_dir out   --max_objects 5   --seeds_per_object 2   --mask_type binary   --output_video true   --video_fps 25
```

**Outputs**
- `out/frame0.png` – the exact first frame used to seed clicks (verify W×H matches your video).
- `out/masks_selected/*.png` – binary masks actually used to produce seeds.
- `out/clicks.json` – the 4 strings (you can reuse them to re-run SAM2-video quickly).
- `out/video_result.json` – URLs/metadata from the SAM2-video call (preview or sequence).

---

## 6) Producing a True‑Alpha Video (Optional)

If you chose `mask_type: "binary"` + **image sequence** from SAM2-video, you can **alphamerge** with the original video:

**WebM VP9 (alpha)**
```bash
# Assume:
# - input.mp4 (original video)
# - masks/frame_%05d.png (1‑channel binary masks, one per frame)
ffmpeg -i input.mp4 -framerate 30 -i masks/frame_%05d.png   -filter_complex "[1:v]format=gray,scale=iw:ih[mask];[0:v][mask]alphamerge"   -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 0 -crf 23 output_alpha.webm
```

**ProRes 4444 (alpha)**
```bash
ffmpeg -i input.mp4 -framerate 30 -i masks/frame_%05d.png   -filter_complex "[1:v]format=gray,scale=iw:ih[mask];[0:v][mask]alphamerge,format=rgba"   -c:v prores_ks -profile:v 4 output_alpha_prores.mov
```

If you chose `mask_type: "greenscreen"` + `output_video: true`, you can key it in your editor. The **true alpha** path above yields cleaner compositing for production.

---

## 7) Tuning, Costs, and Speed

- **Reduce compute/cost for previews**:
  - `--output_frame_interval 2` (use every other frame) or even `3/4` for quick checks.
  - Lower `--video_fps` during exploration.
- **Improve robustness**:
  - Increase `--seeds_per_object` to 3 (centroid + 2 interior peaks).
  - If a mask “leaks”, regenerate with an added **background click** (label `0`) near the leak and re-run.
- **Handle tiny or noisy proposals**:
  - Raise `--min_area` (e.g., 1500 or 3000) or lower it if objects are small.
- **Throughput**:
  - Batch multiple videos; be mindful of rate limits. Cache `frame0.png` and `clicks.json` to avoid recompute.
  - Parallelize at the **video** level (1 process per video).

---

## 8) Troubleshooting Checklist

- **Masks don’t align with the video**: Ensure you extracted **frame 0** from the **same file** passed to SAM2-video—no resizing or re-encoding between steps.
- **No masks returned**: Relax SAM2-image thresholds: lower `--pred_iou_thresh` and/or `--stability_score_thresh`.
- **Wrong object(s) selected**: Increase `--max_objects`, then pick by area (you can add logic to filter by position/color if needed).
- **Tracking drifts after occlusion**: Increase `--seeds_per_object`; add extra seeds on frame 0 that cover distinct parts of the object (head/torso/wheels, etc.).
- **Multiple similar objects**: Distinct **`object_ids`** are already handled in our script (`obj_1`, `obj_2`, …). If you need per‑class IDs, add a detector to pre-label classes and map them to IDs.

---

## 9) Going Fully Automatic (No Manual Selection)

To avoid *any* hand clicking:
1. Run a **detector** (YOLOv8/NAS) on frame 0 (or first few frames).  
2. Convert each box center into one or more seed points; map tracker IDs (ByteTrack/OC‑SORT) to stable `object_ids`.  
3. Feed those generated clicks into SAM2-video (use this same script as a base—replace the mask‑driven seed step with detector‑driven seeds).

This gives you a scalable, repeatable pipeline for large volumes of videos.

---

## 10) Security & Compliance Notes

- Only process videos you **own** or have rights to process.
- Be careful with **identifiable individuals** and follow applicable privacy laws.
- Store outputs and logs securely if videos contain sensitive data.

---

## 11) Summary

- Use **SAM2 (images)** on **frame 0** to get masks **without manual clicks**.
- Convert those masks into **seed points** grouped by **object IDs**.
- Run **SAM2 (video)** once to **track** those objects across the clip.
- For production compositing, prefer **binary masks + alphamerge** for a true alpha channel.

If you want, you can drop this script into your repo and we can extend it to:
- integrate a detector for automatic object selection,
- export clean per-object matte sequences,
- or generate final alpha videos in one go.
