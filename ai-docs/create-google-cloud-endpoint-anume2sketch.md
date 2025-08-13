# Anime2Sketch Remote API Deployment Guide

This guide walks you through setting up a remote API for Anime2Sketch using **Google Cloud Run**, so you can upload images and get sketch outputs via a public HTTP endpoint.

---

## 0) Prerequisites

- Google Cloud Project with billing enabled.
- `gcloud` CLI installed and authenticated (use `gcloud auth login`).
- Optional: GPU usage (L4) requires a region that supports Cloud Run GPU, and quota approval.

---

## 1) Set Project Variables & Enable APIs

```bash
PROJECT_ID=$(gcloud config get-value project)
REGION=europe-west4

# Enable required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
````

---

## 2) Create Artifact Registry (Docker Repo)

```bash
gcloud artifacts repositories create containers \
  --repository-format=docker --location=$REGION
```

---

## 3) Prepare Deployment Files

```bash
mkdir anime2sketch-api && cd anime2sketch-api
```

### `server.py`

```python
import os, uuid, shutil, subprocess
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

BASE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.join(BASE, "Anime2Sketch")
TMP = os.path.join(BASE, "tmp")
os.makedirs(TMP, exist_ok=True)

def repo_ready():
    if not os.path.isdir(REPO):
        return False, "Repository missing."
    wd = os.path.join(REPO, "weights")
    if not os.path.isdir(wd) or not os.listdir(wd):
        return False, "Weights missing."
    return True, "ok"

@app.get("/healthz")
def health():
    ok, msg = repo_ready()
    return {"status": "ok" if ok else "not_ready", "msg": msg}

@app.post("/infer")
async def infer(file: UploadFile = File(...), load_size: int = 512):
    ok, msg = repo_ready()
    if not ok:
        return JSONResponse({"error": msg}, status_code=500)
    rid = str(uuid.uuid4())
    idir = os.path.join(TMP, f"in_{rid}")
    odir = os.path.join(TMP, f"out_{rid}")
    os.makedirs(idir); os.makedirs(odir)
    path = os.path.join(idir, file.filename)
    with open(path, "wb") as f: f.write(await file.read())
    try:
        subprocess.check_call([
            "python", "test.py", "--dataroot", idir,
            "--load_size", str(load_size), "--output_dir", odir
        ], cwd=REPO)
        outs = [os.path.join(odir, p) for p in os.listdir(odir)]
        if not outs:
            return JSONResponse({"error": "no output"}, status_code=500)
        return FileResponse(outs[0], media_type="image/png", filename=os.path.basename(outs[0]))
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": f"model failed: {e}"}, status_code=500)
    finally:
        shutil.rmtree(idir, ignore_errors=True)
        shutil.rmtree(odir, ignore_errors=True)
```

### `Dockerfile`

```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y git wget unzip && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/Mukosame/Anime2Sketch.git
RUN pip install --no-cache-dir -r Anime2Sketch/requirements.txt fastapi uvicorn[standard] gdown

RUN mkdir -p Anime2Sketch/weights && \
    gdown --folder "https://drive.google.com/drive/folders/1Srf-WYUixK0wiUddc9y3pNKHHno5PN6R?usp=sharing" -O Anime2Sketch/weights || true

COPY server.py /app/server.py

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8080"]
```

---

## 4) Build & Push Container

```bash
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/containers/anime2sketch:latest"
gcloud builds submit --tag "$IMAGE"
```

---

## 5A) Deploy to Cloud Run with GPU

```bash
gcloud run deploy anime2sketch \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --cpu 4 \
  --memory 16Gi \
  --no-cpu-throttling \
  --concurrency 1 \
  --allow-unauthenticated
```

## 5B) Or CPU-only (fallback)

```bash
gcloud run deploy anime2sketch \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --cpu 2 \
  --memory 4Gi \
  --allow-unauthenticated
```

---

## 6) Test the Endpoint

```bash
URL=$(gcloud run services describe anime2sketch --region "$REGION" --format='value(status.url)')
curl -s "$URL/healthz" | jq .

curl -X POST "$URL/infer" \
  -F "file=@test_samples/madoka.jpg" \
  --output sketch.png
```

---

## 7) Logs & Cleanup

```bash
gcloud logs read "run.googleapis.com%2Frequests" --limit=100 | jq .

# To update:
gcloud builds submit --tag "$IMAGE"
gcloud run deploy anime2sketch --image "$IMAGE" --region "$REGION"

# To delete service:
# gcloud run services delete anime2sketch --region "$REGION"
```

---

### Notes

* Works both on **GPU (fast)** and **CPU (slower)**.
* Weights are downloaded at build time from the official Google Drive folder.
* Cloud Run GPU handles drivers and CUDA for you—no manual setup needed.

---

That's it! Save this file and run it in Cloud Shell or share it with Claude Code—they can simply execute line by line to expose the Anime2Sketch API remotely.
