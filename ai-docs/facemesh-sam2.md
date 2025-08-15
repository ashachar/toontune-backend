# FaceMesh → SAM2 Face Pixel Mask API (Cloud Run Ready)

**Goal:** Per‑pixel segmentation of a **western cartoon face** (not a bounding box), *without* training.
We combine **MediaPipe Face Mesh** (468 facial landmarks) to generate precise prompts, and call your **existing SAM2 Cloud Run API** to get a crisp face mask. The result is returned as Base64 PNGs.

> This service is a thin adapter you deploy to Cloud Run. It receives an image, computes face landmarks, builds SAM2 prompts (points + optional box), calls your SAM2 service, and returns a clean face mask + overlay + RGBA cutout.

---

## Directory Layout

```
facesam2-api/
├─ Dockerfile
├─ requirements.txt
└─ main.py
```

---

## 1) `requirements.txt`

```txt
fastapi==0.111.0
uvicorn[standard]==0.30.0
pillow==10.4.0
numpy==1.26.4
opencv-python-headless==4.10.0.84
mediapipe==0.10.14
requests==2.32.3
python-multipart==0.0.9
pydantic==2.8.2
```

> Notes
> - `mediapipe` provides the 468 landmark **Face Mesh**.  
> - We keep dependencies CPU‑friendly for cheap Cloud Run usage.

---

## 2) `main.py` — FastAPI Service (complete)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, base64, math, random
from typing import List, Tuple, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import requests

import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# =========================
# Config via ENV
# =========================
SAM2_URL = os.environ.get("SAM2_URL", "https://sam2-your-service-uc.a.run.app/v1/segment")
SAM2_API_KEY = os.environ.get("SAM2_API_KEY", "")
SAM2_IMAGE_FIELD = os.environ.get("SAM2_IMAGE_FIELD", "image_base64")
SAM2_POINTS_FIELD = os.environ.get("SAM2_POINTS_FIELD", "points")
SAM2_BOX_FIELD    = os.environ.get("SAM2_BOX_FIELD", "box")
SAM2_MULTIMASK    = os.environ.get("SAM2_MULTIMASK", "multimask")
SAM2_RETURN_PROB  = os.environ.get("SAM2_RETURN_PROB", "return_prob")

app = FastAPI(title="FaceMesh → SAM2 Face Pixel Mask API",
    description="Computes MediaPipe Face Mesh landmarks, builds SAM2 prompts, calls your SAM2 Cloud Run API, and returns a per-pixel face mask.",
    version="1.0.0")

class MaskResponse(BaseModel):
    width: int
    height: int
    boxes: List[List[int]]
    mask_png_base64: str
    face_rgba_png_base64: str
    overlay_png_base64: str

# =========================
# MediaPipe Face Mesh
# =========================
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
FACE_OVAL_EDGES = mp_face_mesh.FACEMESH_FACE_OVAL

def image_from_upload(file: UploadFile) -> Image.Image:
    data = file.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img

def landmarks_to_xy(landmarks, w: int, h: int) -> np.ndarray:
    pts = []
    for lm in landmarks.landmark:
        x = min(max(lm.x * w, 0), w-1)
        y = min(max(lm.y * h, 0), h-1)
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)

def ordered_face_oval(pts: np.ndarray) -> np.ndarray:
    idxs = set()
    for e in FACE_OVAL_EDGES:
        idxs.add(e[0]); idxs.add(e[1])
    oval = np.array([pts[i] for i in idxs], dtype=np.float32)
    c = oval.mean(axis=0)
    angles = np.arctan2(oval[:,1]-c[1], oval[:,0]-c[0])
    order = np.argsort(angles)
    return oval[order]

def polygon_mask(h: int, w: int, polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask

def sample_interior_points(polygon: np.ndarray, num_points: int = 16) -> List[Tuple[int,int]]:
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    res = []
    tries = 0
    while len(res) < num_points and tries < num_points*50:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if cv2.pointPolygonTest(polygon.astype(np.int32), (x, y), False) >= 0:
            res.append((int(x), int(y)))
        tries += 1
    if not res:
        c = polygon.mean(axis=0)
        res = [(int(c[0]), int(c[1]))]
    return res

def negative_points_outside(polygon: np.ndarray, w: int, h: int, pad: int = 20) -> List[Tuple[int,int]]:
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    x_min = max(int(x_min - pad), 0); y_min = max(int(y_min - pad), 0)
    x_max = min(int(x_max + pad), w-1); y_max = min(int(y_max + pad), h-1)
    candidates = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
    return candidates

def bbox_from_polygon(polygon: np.ndarray, w: int, h: int, expand: float = 0.15) -> List[int]:
    x1, y1 = polygon.min(axis=0)
    x2, y2 = polygon.max(axis=0)
    bw = x2 - x1; bh = y2 - y1
    cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
    nx1 = max(0, int(cx - (1+2*expand)*bw/2))
    nx2 = min(w-1, int(cx + (1+2*expand)*bw/2))
    ny1 = max(0, int(cy - (1+2*expand)*bh/2))
    ny2 = min(h-1, int(cy + (1+2*expand)*bh/2))
    return [nx1, ny1, nx2, ny2]

def call_sam2(image_b64: str, points_labeled: List[List[int]], box: List[int], multimask: bool = True):
    headers = {"Content-Type": "application/json"}
    if SAM2_API_KEY:
        headers["Authorization"] = f"Bearer {SAM2_API_KEY}"
    payload = {
        SAM2_IMAGE_FIELD: image_b64,
        SAM2_POINTS_FIELD: points_labeled,
        SAM2_BOX_FIELD: box,
        SAM2_MULTIMASK: bool(multimask),
        SAM2_RETURN_PROB: False
    }
    r = requests.post(SAM2_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def parse_sam2_mask(resp, h: int, w: int) -> np.ndarray:
    if "mask_png_base64" in resp:
        png_b64 = resp["mask_png_base64"]
        arr = np.frombuffer(base64.b64decode(png_b64), dtype=np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8) * 255
    if "mask_rle" in resp:
        rle = resp["mask_rle"]
        if isinstance(rle, dict) and "counts" in rle and "size" in rle:
            counts = rle["counts"]; size = rle["size"]
        elif isinstance(rle, list) and len(rle) == 2:
            counts, size = rle
        else:
            raise ValueError("Unsupported RLE format")
        H, W = size
        flat = np.zeros(H*W, dtype=np.uint8)
        idx = 0; val = 0
        for c in counts:
            flat[idx:idx+c] = val
            idx += c
            val = 255 - val
        mask = flat.reshape((H, W), order='F')
        if (H, W) != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.uint8)
    if "masks" in resp:
        masks = resp["masks"]
        best = None; best_area = -1
        for m in masks:
            arr = np.array(m)
            if arr.dtype != np.uint8:
                arr = (arr > 0.5).astype(np.uint8)*255
            area = int(arr.sum() // 255)
            if area > best_area:
                best_area = area; best = arr
        mask = best
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8)*255
    raise ValueError("Could not parse SAM2 response")

def rgba_from_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.dstack([image_rgb, (mask > 0).astype(np.uint8)*255])

def overlay_from_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[mask == 255] = (0.6*overlay[mask==255] + 0.4*np.array([0,255,0])).astype(np.uint8)
    return overlay

def np_to_b64_png(arr: np.ndarray, mode: str) -> str:
    if mode == "L":
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    elif mode == "RGBA":
        img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

@app.post("/v1/face-mask-sam2", response_model=MaskResponse)
async def face_mask_sam2(
    image: UploadFile = File(..., description="Cartoon image (png/jpg/webp)"),
    multimask: bool = Form(True, description="Let SAM2 return multiple candidates; we auto-pick largest area"),
    expand: float = Form(0.15, description="Box expansion around face oval for SAM2"),
    pos_points: int = Form(16, description="Number of positive points sampled inside face oval"),
):
    pil = image_from_upload(image)
    img_rgb = np.array(pil)
    H, W = img_rgb.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=5) as fm:
        res = fm.process(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    if not res.multi_face_landmarks:
        empty = np.zeros((H, W), dtype=np.uint8)
        return JSONResponse(status_code=200, content={
            "width": W, "height": H, "boxes": [],
            "mask_png_base64": np_to_b64_png(empty, "L"),
            "face_rgba_png_base64": np_to_b64_png(np.dstack([img_rgb, empty]), "RGBA"),
            "overlay_png_base64": np_to_b64_png(img_rgb, "RGB")
        })

    full_mask = np.zeros((H, W), dtype=np.uint8)
    boxes_out = []

    for face_lms in res.multi_face_landmarks:
        pts = landmarks_to_xy(face_lms, W, H)
        oval_poly = ordered_face_oval(pts)

        box = bbox_from_polygon(oval_poly, W, H, expand=expand)
        boxes_out.append(box)

        pos = sample_interior_points(oval_poly, num_points=int(pos_points))
        neg = negative_points_outside(oval_poly, W, H, pad=20)

        points_labeled = [[int(x), int(y), 1] for (x,y) in pos] + \
                         [[int(x), int(y), 0] for (x,y) in neg]

        buf = io.BytesIO(); Image.fromarray(img_rgb).save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        try:
            sam2_resp = call_sam2(image_b64, points_labeled, box, multimask=multimask)
            mask = parse_sam2_mask(sam2_resp, H, W)
        except Exception:
            mask = polygon_mask(H, W, oval_poly)

        full_mask = np.maximum(full_mask, mask)

    rgba = rgba_from_mask(img_rgb, full_mask)
    overlay = overlay_from_mask(img_rgb, full_mask)

    return {
        "width": W,
        "height": H,
        "boxes": boxes_out,
        "mask_png_base64": np_to_b64_png(full_mask, "L"),
        "face_rgba_png_base64": np_to_b64_png(rgba, "RGBA"),
        "overlay_png_base64": np_to_b64_png(overlay, "RGB"),
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
```

---

## 3) `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

---

## 4) Deploy to Google Cloud Run

Authenticate and select your project:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Build the container (Cloud Build):
```bash
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/facesam2-api:1
```

Deploy to Cloud Run:
```bash
gcloud run deploy facesam2-api \
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/facesam2-api:1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 2Gi \
  --max-instances 5 \
  --concurrency 1
```

Set your SAM2 endpoint and (optional) API key:
```bash
gcloud run services update facesam2-api \
  --region us-central1 \
  --update-env-vars SAM2_URL=https://sam2-your-service-uc.a.run.app/v1/segment,SAM2_API_KEY=YOUR_TOKEN
```

---

## 5) Test with `curl`

```bash
curl -X POST \
  -F "image=@/path/to/cartoon.png" \
  -F "multimask=true" \
  -F "expand=0.15" \
  -F "pos_points=16" \
  "https://YOUR_FACESAM2_URL/v1/face-mask-sam2" > out.json
```

Save results locally:
```python
# save_outputs.py
import json, base64

with open("out.json","r") as f:
    data = json.load(f)

for key in ["mask_png_base64", "face_rgba_png_base64", "overlay_png_base64"]:
    with open(key.replace("_base64","") + ".png", "wb") as out:
        out.write(base64.b64decode(data[key]))
print("Saved: mask_png.png, face_rgba_png.png, overlay_png.png")
```

---

## 6) How it works (summary)

1. **Landmarks**: MediaPipe Face Mesh predicts **468** 3D face landmarks.  
2. **Oval Polygon**: We extract the **face oval** subset and build an ordered polygon, then sample **positive points** inside and **negative points** outside the polygon.  
3. **SAM2 prompts**: We POST the original image + prompts to your **SAM2** Cloud Run API (`points` + `box`).  
4. **Mask parsing**: We accept multiple common SAM2 server formats (`mask_png_base64`, `mask_rle`, or `masks[]`).  
5. **Outputs**: Binary mask PNG, overlay visualization, and RGBA cutout.

---

## 7) Tunables

- `pos_points` (default 16): More points → stronger guidance for SAM2 on stylized cartoons.  
- `expand` (default 0.15): Slightly enlarges the box around the face oval to ensure full coverage of cheeks/forehead without hair.  
- If your SAM2 server prefers only **point prompts** (no box), you can set `box=None` in `call_sam2` and adjust payload fields.

---

## 8) Security & Hardening (optional)

- Put this behind **API Gateway** or require a custom header.  
- Limit upload size via Cloud Run + FastAPI middleware.  
- Set `--min-instances` to mitigate cold starts for low latency.

---

## 9) Compatibility

- This adapter expects your SAM2 endpoint to accept **image + points (+ optional box)** and return a **mask**. If your contract is different, adjust the ENV field names at the top of `main.py` (no code changes required).
