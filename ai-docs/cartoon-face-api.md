# Cartoon Face Pixel Mask API (Cloud Run Ready)

Detect per-pixel **face masks** (not bounding boxes) from cartoon/anime images via a lightweight, CPU-friendly pipeline. The API returns:
- `mask_png_base64` – binary face mask (white=face, black=background).
- `face_rgba_png_base64` – original image cutout with transparency outside the face mask.
- `overlay_png_base64` – visualization overlay for quick QA.
- `boxes` – detected face bounding boxes (for info only).

**Pipeline (CPU-first):** LBP AnimeFace (face detection) → SegFormer (face parsing) → optional DenseCRF refinement.
Designed to run cheaply on **Google Cloud Run (CPU)** with no internet dependency at runtime (assets pre-cached in the image).

---

## Directory Layout

```
cartoon-face-api/
├─ Dockerfile
├─ requirements.txt
├─ download_assets.py
└─ main.py
```

---

## 1) requirements.txt

> DenseCRF is optional. If you want sharper edges, add `pydensecrf` to the list below and set `use_crf=true` on requests.

```txt
fastapi==0.111.0
uvicorn[standard]==0.30.0
pillow==10.4.0
numpy==1.26.4
opencv-python-headless==4.10.0.84
torch==2.3.1
torchvision==0.18.1
transformers==4.42.3
accelerate==0.33.0
python-multipart==0.0.9
pydantic==2.8.2
```

---

## 2) download_assets.py

Downloads the LBP cascade and pre-caches the face parsing model at **build time**, so runtime is offline.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, pathlib, urllib.request

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

ASSETS_DIR = pathlib.Path("/app/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

LBP_XML = ASSETS_DIR / "lbpcascade_animeface.xml"
LBP_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"

print("[download] LBP AnimeFace cascade …")
if not LBP_XML.exists():
    urllib.request.urlretrieve(LBP_URL, LBP_XML.as_posix())

# Pre-cache HF model into HF_HOME
HF_HOME = os.environ.get("HF_HOME", "/models/hfcache")
os.makedirs(HF_HOME, exist_ok=True)
MODEL_ID = "jonathandinu/face-parsing"

print(f"[download] Pre-caching face-parsing model into {HF_HOME} …")
# This will download config, image processor, and weights
_ = SegformerImageProcessor.from_pretrained(MODEL_ID, cache_dir=HF_HOME)
_ = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, cache_dir=HF_HOME)

print("[done] Assets ready.")
```

---

## 3) main.py — FastAPI Service (Complete)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, base64
from typing import List
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# =========================
# Performance & Offline
# =========================
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # prefer offline (we pre-cached)

if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

DEVICE = "cpu"  # Cloud Run CPU by default
HF_HOME = os.environ.get("HF_HOME", "/models/hfcache")
ASSETS_DIR = Path("/app/assets")
LBP_XML = ASSETS_DIR / "lbpcascade_animeface.xml"

# =========================
# Load models on startup
# =========================
print("[load] Loading LBP cascade …")
LBP = cv2.CascadeClassifier(LBP_XML.as_posix())
if LBP.empty():
    raise RuntimeError("LBP cascade not found or failed to load")

print("[load] Loading face-parsing model (SegFormer) …")
FP_MODEL_ID = "jonathandinu/face-parsing"
FP_LABELS = {
    0:"background",1:"skin",2:"nose",3:"eye_g",4:"l_eye",5:"r_eye",6:"l_brow",7:"r_brow",
    8:"l_ear",9:"r_ear",10:"mouth",11:"u_lip",12:"l_lip",13:"hair",14:"hat",15:"ear_r",
    16:"neck_l",17:"neck",18:"cloth"
}
FACE_LABEL_IDS_DEFAULT = [1,2,3,4,5,6,7,8,9,10,11,12]  # no hair/hat/cloth/neck by default

fp_processor = SegformerImageProcessor.from_pretrained(FP_MODEL_ID, cache_dir=HF_HOME, local_files_only=True)
fp_model = SegformerForSemanticSegmentation.from_pretrained(FP_MODEL_ID, cache_dir=HF_HOME, local_files_only=True).to(DEVICE)
fp_model.eval()

# =========================
# Helpers
# =========================
def image_from_upload(file: UploadFile) -> Image.Image:
    data = file.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img

def detect_face_boxes(img_rgb: np.ndarray) -> List[List[int]]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = LBP.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24,24))
    boxes = []
    for (x,y,w,h) in faces:
        boxes.append([int(x), int(y), int(x+w), int(y+h)])
    return boxes

def resize_if_needed(img_rgb: np.ndarray, max_side: int):
    H, W = img_rgb.shape[:2]
    if max(H, W) <= max_side:
        return img_rgb, 1.0
    scale = max_side / float(max(H, W))
    out = cv2.resize(img_rgb, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    return out, scale

def face_parsing_probs(roi_rgb: np.ndarray, max_size: int):
    H, W = roi_rgb.shape[:2]
    roi_small, scale = resize_if_needed(roi_rgb, max_size)
    inputs = fp_processor(images=Image.fromarray(roi_small), return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = fp_model(**inputs).logits  # [1,C,h/4,w/4]
        up = F.interpolate(logits, size=roi_small.shape[:2], mode='bilinear', align_corners=False)  # [1,C,Hs,Ws]
        probs_small = up.softmax(dim=1)[0].cpu().numpy()  # CxHsxWs
    if scale != 1.0:
        C = probs_small.shape[0]
        probs = np.zeros((C, H, W), dtype=np.float32)
        for c in range(C):
            probs[c] = cv2.resize(probs_small[c], (W, H), interpolation=cv2.INTER_LINEAR)
        return probs
    else:
        return probs_small

def build_face_prob(probs: np.ndarray, include_hair: bool) -> np.ndarray:
    ids = FACE_LABEL_IDS_DEFAULT.copy()
    if include_hair:
        ids += [13]  # hair
    face_prob = np.clip(np.sum(probs[ids, :, :], axis=0), 0.0, 1.0)
    return face_prob

def expand_box(x1, y1, x2, y2, W, H, expand: float):
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    bw, bh = (x2-x1), (y2-y1)
    nx1 = max(0, int(cx - (1+2*expand)*bw/2))
    nx2 = min(W, int(cx + (1+2*expand)*bw/2))
    ny1 = max(0, int(cy - (1+2*expand)*bh/2))
    ny2 = min(H, int(cy + (1+2*expand)*bh/2))
    return nx1, ny1, nx2, ny2

def np_to_b64_png(arr: np.ndarray, mode: str = "L") -> str:
    # mode "L" for mask, "RGBA" for cutout, "RGB" for overlay
    if mode == "L":
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    elif mode == "RGBA":
        img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    else:  # "RGB"
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Cartoon Face Pixel Mask API",
    description="Detects cartoon/anime faces and returns a per-pixel mask of the face (not a bounding box).",
    version="1.0.0",
)

class MaskResponse(BaseModel):
    width: int
    height: int
    boxes: List[List[int]]
    mask_png_base64: str
    face_rgba_png_base64: str
    overlay_png_base64: str

@app.post("/v1/face-mask", response_model=MaskResponse)
async def face_mask(
    image: UploadFile = File(..., description="Cartoon/Anime image file (png/jpg/webp)"),
    include_hair: bool = Form(False, description="Include hair in 'face' mask"),
    use_crf: bool = Form(False, description="Enable CRF refinement (requires pydensecrf)"),
    max_size: int = Form(448, description="Max side for face parsing ROI (CPU-friendly)"),
    expand: float = Form(0.15, description="ROI expansion around detected face"),
    threshold: float = Form(0.5, description="Probability threshold for face mask [0..1]")
):
    # Read image
    pil = image_from_upload(image)
    img_rgb = np.array(pil)
    H, W = img_rgb.shape[:2]

    # 1) face detection
    boxes = detect_face_boxes(img_rgb)
    if len(boxes) == 0:
        # Return empty mask the same size
        empty = np.zeros((H, W), dtype=np.uint8)
        rgba = np.dstack([img_rgb, empty])
        overlay = img_rgb.copy()
        return JSONResponse(status_code=200, content={
            "width": W,
            "height": H,
            "boxes": [],
            "mask_png_base64": np_to_b64_png(empty, "L"),
            "face_rgba_png_base64": np_to_b64_png(rgba, "RGBA"),
            "overlay_png_base64": np_to_b64_png(overlay, "RGB"),
        })

    full_mask = np.zeros((H, W), dtype=np.uint8)

    # Optional CRF (only if installed and requested)
    HAS_CRF = False
    if use_crf:
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
            HAS_CRF = True
        except Exception:
            HAS_CRF = False

    for (x1,y1,x2,y2) in boxes:
        nx1, ny1, nx2, ny2 = expand_box(x1, y1, x2, y2, W, H, expand)
        roi = img_rgb[ny1:ny2, nx1:nx2].copy()
        if roi.size == 0:
            continue

        probs = face_parsing_probs(roi, max_size=max_size)
        face_prob = build_face_prob(probs, include_hair=include_hair)

        if HAS_CRF:
            # refine with CRF
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
            H2, W2 = face_prob.shape
            probs2 = np.stack([1.0 - face_prob, face_prob], axis=0).astype(np.float32)
            d = dcrf.DenseCRF2D(W2, H2, 2)
            U = unary_from_softmax(probs2)
            d.setUnaryEnergy(U)
            d.addPairwiseEnergy(create_pairwise_gaussian(sdims=(3,3), shape=(H2,W2)), compat=3)
            d.addPairwiseEnergy(create_pairwise_bilateral(sdims=(60,60), schan=(20,20,20), img=roi, chdim=2), compat=5)
            Q = d.inference(5)
            res = np.array(Q).reshape((2, H2, W2))
            mask = (res[1] > res[0]).astype(np.uint8) * 255
        else:
            mask = (face_prob >= float(threshold)).astype(np.uint8) * 255

        # paste back
        full_mask[ny1:ny2, nx1:nx2] = np.maximum(full_mask[ny1:ny2, nx1:nx2], mask)

    # Build outputs
    overlay = img_rgb.copy()
    overlay[full_mask == 255] = (0.5*overlay[full_mask==255] + 0.5*np.array([0,255,0])).astype(np.uint8)
    rgba = np.dstack([img_rgb, (full_mask>0).astype(np.uint8)*255])

    return {
        "width": W,
        "height": H,
        "boxes": boxes,
        "mask_png_base64": np_to_b64_png(full_mask, "L"),
        "face_rgba_png_base64": np_to_b64_png(rgba, "RGBA"),
        "overlay_png_base64": np_to_b64_png(overlay, "RGB"),
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
```

---

## 4) Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Performance/env defaults
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    HF_HOME=/models/hfcache \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY download_assets.py .
COPY main.py .
RUN mkdir -p /app/assets /models/hfcache

# Pre-download models & assets into the image (offline-friendly at runtime)
RUN python download_assets.py

# Cloud Run expects the service to listen on $PORT
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

---

## Deploy to Google Cloud Run

Authenticate and select project:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Build the container (Cloud Build):
```bash
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/cartoon-face-api:1
```

Deploy to Cloud Run:
```bash
gcloud run deploy cartoon-face-api \
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/cartoon-face-api:1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 2Gi \
  --max-instances 3 \
  --concurrency 1
```
> Increase `--memory` to `4Gi` for very large images. Keep `--concurrency 1` initially to bound RAM per instance; raise after measuring.

Optional: keep one warm instance to reduce cold starts:
```bash
gcloud run services update cartoon-face-api \
  --region us-central1 \
  --min-instances 1
```

---

## Test with cURL

Send an image and optional params. Receive JSON with Base64 fields:
```bash
curl -X POST \
  -F "image=@/path/to/cartoon.png" \
  -F "include_hair=false" \
  -F "use_crf=false" \
  -F "max_size=448" \
  -F "expand=0.15" \
  -F "threshold=0.5" \
  "https://YOUR_CLOUD_RUN_URL/v1/face-mask" > out.json
```

Save outputs locally:
```python
# save_mask.py
import json, base64
with open("out.json","r") as f:
    data = json.load(f)

for k in ["mask_png_base64", "face_rgba_png_base64", "overlay_png_base64"]:
    with open(k.replace("_base64","") + ".png", "wb") as out:
        out.write(base64.b64decode(data[k]))
print("saved: mask_png.png, face_rgba_png.png, overlay_png.png")
```

---

## Performance & Cost Tips (Cloud Run, CPU)

- Keep `max_size=448` (or even `384`) for the face parsing ROI to reduce latency and memory.
- If clients upload huge images (e.g., 4K), downscale on the client or gateway to ~1600 px max side before sending.
- Start with `--concurrency 1` to cap RAM; then measure and increase as needed.
- If you want sharper boundaries, add `pydensecrf` to `requirements.txt` and pass `use_crf=true` (adds CPU time/RAM).

---

## Security & Hardening (Optional)

- Put the service behind **API Gateway** or Cloud Endpoints (ESPv2) and require an API key or JWT.
- Add size limits to uploads and validate MIME types.
- Consider setting `max-instances` and request timeouts appropriate for your traffic.

---

## Notes

- This implementation avoids heavy SAM models to keep CPU costs low. If you later need even cleaner edges, you can integrate MobileSAM or run GPU-backed inference; the REST contract remains unchanged.
- All assets (LBP cascade + model weights) are pre-fetched during the build to avoid network calls at runtime.
