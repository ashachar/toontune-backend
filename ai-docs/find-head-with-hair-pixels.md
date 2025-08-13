# Cartoon/Anime Face + Hair Pixel Mask — One-File Pipeline (Detector + SAM2)

This guide gives you a **single Python script** that outputs a **pixel-accurate mask of the face *including hair*** from a colorful cartoon/anime image (e.g., `woman.png`).

It combines:

* **hysts/anime-face-detector** (mmdet+mmpose) to get a **face box + 28 landmarks**, tuned for anime/cartoons. ([GitHub][1])
* **Segment Anything 2 (SAM2)** to turn that prompt (box + points) into a **full-resolution pixel mask**. It supports Linux/WSL, requires Python ≥ 3.10, and provides ready-made 2.1 checkpoints. ([GitHub][2])
* **Fallback**: if the detector can't run/doesn't find a face, we use the classic **LBP cascade for anime faces** to produce a box, and still run SAM2. ([GitHub][3])

You'll end up with:

* `face_hair_mask.png` (binary mask)
* `face_hair_cutout.png` (RGBA cutout, transparent background)
* `overlay.png` (QA overlay)

## 1) Requirements (tested approach)

**OS**: Ubuntu 22.04 (native or **Windows via WSL** is recommended for SAM2). ([GitHub][2])  
**Python**: 3.10+  
**GPU**: Optional. SAM2 benefits from CUDA; CPU is okay for single-image use.

> The `anime-face-detector` repo ships install steps via `openmim` (MMCV, MMDetection, MMPose) and notes it's **tested only on Ubuntu**, so Linux/WSL is the easiest path. ([GitHub][1])

## 2) Create a clean environment

```bash
# Ubuntu / WSL
sudo apt-get update
sudo apt-get install -y python3.10-venv python3-pip git wget

# Project folder
mkdir cartoon-mask && cd cartoon-mask
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If you have an NVIDIA GPU, first install a matching **PyTorch + CUDA** per the official instructions; otherwise install the CPU wheel. (SAM2 requires torch ≥ 2.5.1.) ([GitHub][2])

```bash
# Example (CPU-only):
pip install "torch>=2.5.1" "torchvision>=0.20.1"
```

## 3) Install SAM2

Follow Meta's SAM2 install guidance (clone and editable install). ([GitHub][2])

```bash
# From your project folder
git clone https://github.com/facebookresearch/sam2.git
pip install -e sam2
```

Download a **SAM2.1** checkpoint. This script defaults to **small**; you can switch to `base_plus`/`large` if you've got the VRAM. ([GitHub][2])

```bash
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/checkpoints/sam2.1_hiera_small.pt
```

> SAM2's README also lists other checkpoints: `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_base_plus.pt`, `sam2.1_hiera_large.pt`. ([GitHub][2])

## 4) Install the anime face detector (and deps)

Use the exact steps from the repo (OpenMMLab via `mim`, then the detector). ([GitHub][1])

```bash
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmpose

pip install anime-face-detector
```

> The repo explicitly highlights **Ubuntu** as the tested platform, and provides a minimal usage snippet (`create_detector('yolov3')`, returns `bbox` + 28 `keypoints`). Pretrained models download automatically at first use. ([GitHub][1])

## 5) (Optional fallback) Download the LBP anime cascade

If `anime-face-detector` can't be installed/run on your machine, we'll fall back to this classic detector for a coarse face box. ([GitHub][3])

```bash
wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
```

## 6) Put your test image here

Copy `woman.png` into the project folder (same directory where you'll run the script).

## 7) The single-file script

Save as `cartoon_face_mask.py` (no edits needed):

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cartoon_face_mask.py
Purpose: Produce a pixel-accurate mask of an ANIME/CARTOON face INCLUDING HAIR
from a colorful drawing (e.g., woman.png) using:
  - Stage A: hysts/anime-face-detector (mmdet+mmpose) for face bbox + 28 landmarks
  - Stage B: Segment Anything 2 (SAM2) for pixel-wise segmentation via box+points prompt
  - Fallback: OpenCV LBP cascade for anime faces (bbox) -> SAM2

Outputs in CWD:
  - face_hair_mask.png   (binary mask, white=face+hair)
  - face_hair_cutout.png (RGBA cutout with transparent bg)
  - overlay.png          (quality-check overlay)
  - debug_bbox.png       (optional visualization of the expanded head box)

Notes & Sources:
- hysts/anime-face-detector: mmdet + mmpose; usage via create_detector('yolov3');
  installation tested on Ubuntu; pretrained models auto-download. (MIT) 
  https://github.com/hysts/anime-face-detector
- SAM2 (Meta): promptable segmentation; requires python>=3.10, torch>=2.5.1; 
  checkpoints (e.g., sam2.1_hiera_small.pt) provided in the README. (Apache-2.0)
  https://github.com/facebookresearch/sam2
- Fallback detector: nagadomi/lbpcascade_animeface (OpenCV LBP cascade, single XML).
  https://github.com/nagadomi/lbpcascade_animeface
"""

import os
import sys
import urllib.request
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image

# ------------------ Device selection (PyTorch used internally by SAM2) ------------------
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                            else "cpu"))
print(f"[INFO] Using device: {DEVICE}")

# ------------------ SAM2 imports ------------------
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as e:
    print("[ERR] SAM2 is not installed. Please follow the README section 'Install SAM2'.")
    raise

# ------------------ anime-face-detector (optional, primary detector) ------------------
def try_anime_face_detector(image_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Use hysts/anime-face-detector to get bbox + 28 landmarks.
    Returns (bbox_xyxy, keypoints_xy) or None on failure.
    """
    try:
        from anime_face_detector import create_detector  # installed via pip
    except Exception as e:
        print(f"[WARN] anime-face-detector not available: {e}")
        return None
    try:
        det = create_detector('yolov3')   # downloads weights automatically
        preds = det(image_bgr)
        if not preds:
            print("[WARN] No anime faces found by anime-face-detector.")
            return None
        p = preds[0]
        bbox = p['bbox'].astype(np.float32)  # [x0, y0, x1, y1, score]
        x0, y0, x1, y1 = bbox[:4]
        bbox_xyxy = np.array([x0, y0, x1, y1], dtype=np.float32)
        kps = p.get('keypoints', None)
        if kps is not None and len(kps) > 0:
            kps = kps[:, :2].astype(np.float32)  # Nx2
        else:
            kps = None
        return bbox_xyxy, kps
    except Exception as e:
        print(f"[WARN] anime-face-detector inference failed: {e}")
        return None

# ------------------ Fallback detector: LBP cascade ------------------
CASCADE_PATH = os.path.join(os.getcwd(), "lbpcascade_animeface.xml")
CASCADE_URL  = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"

def ensure_lbp_cascade():
    if not os.path.isfile(CASCADE_PATH):
        print("[INFO] Downloading lbpcascade_animeface.xml ...")
        with urllib.request.urlopen(CASCADE_URL) as r, open(CASCADE_PATH, "wb") as f:
            f.write(r.read())

def fallback_lbp_detector(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns bbox_xyxy from OpenCV LBP cascade detector, or None.
    """
    ensure_lbp_cascade()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24,24))
    if len(faces) == 0:
        print("[WARN] LBP cascade found no faces.")
        return None
    x, y, w, h = faces[0]
    return np.array([x, y, x+w, y+h], dtype=np.float32)

# ------------------ Head box expansion (to include HAIR) ------------------
def clip(val, lo, hi):
    return max(lo, min(hi, val))

def expand_head_box(bbox_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Expand the face bbox to cover hair.
      - expand top by +60% face height (anime hair is often tall)
      - expand left/right by +20%
      - expand bottom by +20% (chin/neck margin)
    """
    x0, y0, x1, y1 = bbox_xyxy
    w = x1 - x0
    h = y1 - y0
    nx0 = clip(x0 - 0.20 * w, 0, img_w - 1)
    nx1 = clip(x1 + 0.20 * w, 0, img_w - 1)
    ny0 = clip(y0 - 0.60 * h, 0, img_h - 1)
    ny1 = clip(y1 + 0.20 * h, 0, img_h - 1)
    return np.array([nx0, ny0, nx1, ny1], dtype=np.float32)

def keypoints_to_positive_points(kps: Optional[np.ndarray], head_box: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Build SAM2 positive point prompts from landmarks (if available),
    plus 3 synthetic points near hairline to encourage hair inclusion.
    """
    if kps is None or len(kps) == 0:
        return None
    pts = kps.copy()
    x0, y0, x1, y1 = head_box
    cx = 0.5 * (x0 + x1)
    top_y = y0 + 0.15 * (y1 - y0)
    left_x = x0 + 0.25 * (x1 - x0)
    right_x = x0 + 0.75 * (x1 - x0)
    extra = np.array([[cx, top_y], [left_x, top_y], [right_x, top_y]], dtype=np.float32)
    pts = np.vstack([pts, extra])
    labels = np.ones((pts.shape[0],), dtype=np.int32)  # 1=positive
    return pts, labels

# ------------------ SAM2 predictor ------------------
def find_sam2_config_path() -> str:
    """
    Locate 'sam2.1_hiera_s.yaml' inside the installed sam2 package.
    """
    import sam2, os
    cand = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2.1", "sam2.1_hiera_s.yaml")
    if os.path.isfile(cand):
        return cand
    # Fallback to relative path if running from a clone next to this script
    cand2 = os.path.join("sam2", "configs", "sam2.1", "sam2.1_hiera_s.yaml")
    if os.path.isfile(cand2):
        return cand2
    raise FileNotFoundError("Could not find SAM2 config 'sam2.1_hiera_s.yaml'.")

def run_sam2(image_bgr: np.ndarray, head_box_xyxy: np.ndarray,
             points_labels: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             ckpt_path: str = "checkpoints/sam2.1_hiera_small.pt") -> np.ndarray:
    """
    Run SAM2 with a box prompt (+ optional positive points) to get a binary mask for face+hair.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found at: {ckpt_path}")
    cfg_path = find_sam2_config_path()
    print(f"[INFO] Loading SAM2: cfg={cfg_path}, ckpt={ckpt_path}")
    sam2_model = build_sam2(cfg_path, ckpt_path)
    predictor = SAM2ImagePredictor(sam2_model)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    box = head_box_xyxy.astype(np.float32)

    if points_labels is not None:
        pts, labs = points_labels
        masks, scores, _ = predictor.predict(point_coords=pts, point_labels=labs, box=box, multimask_output=True)
    else:
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)

    # choose mask with highest IoU to head box (tends to prefer whole head)
    best_iou, best_mask = -1.0, None
    hb = np.zeros(masks[0].shape, dtype=bool)
    x0, y0, x1, y1 = box.astype(int)
    hb[y0:y1+1, x0:x1+1] = True
    for m in masks:
        m_bool = m.astype(bool)
        inter = np.logical_and(m_bool, hb).sum()
        union = np.logical_or(m_bool, hb).sum()
        iou = (inter / union) if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_mask = iou, m

    mask = (best_mask * 255).astype(np.uint8)

    # light post-process to smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (0,0), 0.8)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask

# ------------------ IO helpers ------------------
def imread(path, flags=1):
    img = cv2.imread(path, flags)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def imwrite(path, img):
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write: {path}")

def save_outputs(image_bgr: np.ndarray, mask: np.ndarray, head_box: np.ndarray):
    imwrite("face_hair_mask.png", mask)

    overlay = image_bgr.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.6 + np.array([0, 0, 255]) * 0.4).astype(np.uint8)
    imwrite("overlay.png", overlay)

    dbg = image_bgr.copy()
    x0, y0, x1, y1 = head_box.astype(int)
    cv2.rectangle(dbg, (x0, y0), (x1, y1), (0,255,0), 2)
    imwrite("debug_bbox.png", dbg)

    rgba = np.dstack([cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), mask])
    Image.fromarray(rgba, mode="RGBA").save("face_hair_cutout.png")

    coverage = (mask > 0).mean() * 100.0
    print(f"[OK] Saved face_hair_mask.png, overlay.png, face_hair_cutout.png (coverage {coverage:.2f}%).")

# ------------------ main ------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python cartoon_face_mask.py woman.png")
        sys.exit(1)
    in_path = sys.argv[1]
    image_bgr = imread(in_path, 1)
    H, W = image_bgr.shape[:2]

    # Detector A (preferred): anime-face-detector
    det = try_anime_face_detector(image_bgr)
    if det is not None:
        bbox_xyxy, kps = det
    else:
        # Fallback: LBP cascade
        fb = fallback_lbp_detector(image_bgr)
        if fb is None:
            print("[ERR] No face detected by both detectors.")
            sys.exit(2)
        bbox_xyxy, kps = fb, None

    head_box = expand_head_box(bbox_xyxy, W, H)
    prompts = keypoints_to_positive_points(kps, head_box) if kps is not None else None

    # SAM2 segmentation
    mask = run_sam2(image_bgr, head_box, prompts)

    # Save products
    save_outputs(image_bgr, mask, head_box)

if __name__ == "__main__":
    main()
```

**What the script does (logic):**

1. Try `anime-face-detector` → get **bbox + 28 landmarks**; if missing, fallback to LBP cascade (bbox). ([GitHub][1])
2. Expand that face box upward/sideways to a **"head box"** so hair is included.
3. Run **SAM2** with the box and (if we have them) **positive points** (landmarks + synthetic hairline hints) to get a high-quality **face+hair mask**. ([GitHub][2])

## 8) How to run

```bash
# From the project folder (venv active)
python cartoon_face_mask.py woman.png
```

You'll see:

* ✅ `face_hair_mask.png` — white=face+hair
* ✅ `face_hair_cutout.png` — the original subject with transparent background
* ✅ `overlay.png` — mask visualization over your input
* ✅ `debug_bbox.png` — the expanded head box we used to prompt SAM2

## 9) Troubleshooting & tips

* **Windows**: Prefer **WSL (Ubuntu)** for SAM2, as recommended by Meta. ([GitHub][2])
* **mmdet/mmpose errors** (installing `anime-face-detector`): install exactly as in the repo (`openmim` + `mmcv-full` + `mmdet` + `mmpose`), and keep it on Ubuntu/WSL. ([GitHub][1])
* **No face found**: Make sure the face is near-frontal (the detector is trained for that). The fallback LBP is more basic but often enough to seed SAM2. ([GitHub][1])
* **More hair coverage**: increase the top expansion in `expand_head_box` from `0.60` to `0.75`.
* **Higher quality masks**: swap the checkpoint to `sam2.1_hiera_base_plus.pt` (more VRAM/time). Update the file path in the script and download it from the SAM2 README. ([GitHub][2])

## 10) Why this works well for cartoons

* `anime-face-detector` is **domain-tuned** to anime/cartoons, returning a face box + **28 landmarks**. We leverage those to prompt SAM2 precisely. ([GitHub][1])
* **SAM2** is a strong general segmenter. Given a good head box (and optional positive points), it produces robust **pixel-level masks** even in stylized domains. Its docs include ready-to-use checkpoints and clear install instructions for Python ≥ 3.10 / Torch ≥ 2.5.1. ([GitHub][2])

---

If you want, I can also give you a **batch version** that processes a whole folder and saves **COCO polygons** alongside the PNG masks for animation pipelines.

[1]: https://github.com/hysts/anime-face-detector "GitHub - hysts/anime-face-detector: Anime Face Detector using mmdet and mmpose"
[2]: https://github.com/facebookresearch/sam2 "GitHub - facebookresearch/sam2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."
[3]: https://github.com/nagadomi/lbpcascade_animeface "nagadomi/lbpcascade_animeface: A Face detector for anime faces"