#!/usr/bin/env bash
# ==============================================================================
# LAMBDA CONTAINER MANIFEST RUNBOOK — rembg-enabled text animation
# Purpose: Ship the working container to AWS Lambda by forcing a **Docker V2
#          (schema 2)** image manifest (and single-arch), with clear
#          verification and rollback steps. Adds runtime debug logs prefixed:
#          [LAMBDA_MANIFEST] ...
#
# How to use:
#   1) Copy this entire block into a file named: manifest_runbook.sh
#   2) Run:  bash manifest_runbook.sh
#
# This script is intentionally verbose and self-checking, optimized for a
# junior operator. It creates tiny helper files (debug logger + Dockerfiles),
# builds & pushes the image with the correct manifest, verifies, deploys, tests,
# and shows rollback.
#
# Notes:
#   • We **enforce Docker V2 schema 2** by either:
#       A) Using the legacy builder (DOCKER_BUILDKIT=0), or
#       B) Using buildx but exporting with docker media types (oci-mediatypes=false)
#   • We **enforce a single architecture** (linux/amd64) so Lambda won't reject
#     a multi-arch index.
#   • We add runtime debug logs via `sitecustomize.py` so you'll see
#     [LAMBDA_MANIFEST] lines even before your code runs.
# ==============================================================================

set -euo pipefail

# ----------------------------
# 0) Configuration (edit if needed)
# ----------------------------
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID="562404437786"
export ECR_REPO="text-animation-lambda"
export IMAGE_TAG="v5-final"
export LAMBDA_FUNCTION="toontune-text-animation"
export PLATFORM="linux/amd64"          # Lambda x86_64. Use linux/arm64 if your function is arm64.

# Paths in this repo (keep defaults to match your tree)
export DOCKERFILE_INCREMENTAL="lambda/Dockerfile.v5-incremental"
export DOCKERFILE_FULL="lambda/Dockerfile"

# Derived vars
export REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export IMAGE_URI="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

# ----------------------------
# Logging helpers
# ----------------------------
log()  { echo "[LAMBDA_MANIFEST] $*"; }
fail() { echo "[LAMBDA_MANIFEST] FATAL: $*" >&2; exit 1; }

# ----------------------------
# 1) Pre-flight checks
# ----------------------------
log "Pre-flight: checking tools and AWS identity..."
command -v docker >/dev/null 2>&1 || fail "Docker is not installed."
command -v aws    >/dev/null 2>&1 || fail "AWS CLI is not installed."
if ! command -v jq >/dev/null 2>&1; then
  log "jq not found; manifest JSON will be shown raw. (Optional)"
fi

aws sts get-caller-identity >/dev/null || fail "AWS CLI not authenticated."

log "Docker version: $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'unknown')"
log "AWS account asserted: ${AWS_ACCOUNT_ID}"
log "Target ECR repo: ${ECR_REPO}"
log "Target region: ${AWS_REGION}"
log "Lambda function: ${LAMBDA_FUNCTION}"
log "Target platform: ${PLATFORM}"

# ----------------------------
# 2) Ensure ECR repo exists & login
# ----------------------------
if ! aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  log "ECR repo ${ECR_REPO} not found; creating..."
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null
  log "ECR repo created."
fi

log "Logging into ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}" >/dev/null
log "ECR login OK."

# ----------------------------
# 3) Add runtime debug logs (sitecustomize.py)
#    This prints [LAMBDA_MANIFEST] lines when Python starts in Lambda.
# ----------------------------
mkdir -p lambda/python
cat > lambda/python/sitecustomize.py << 'PY'
import os, platform, sys
print(f"[LAMBDA_MANIFEST] Python {sys.version.split()[0]} on {platform.system()} {platform.release()} arch={platform.machine()}")
print(f"[LAMBDA_MANIFEST] AWS_EXECUTION_ENV={os.environ.get('AWS_EXECUTION_ENV')}")
print(f"[LAMBDA_MANIFEST] LAMBDA_TASK_ROOT={os.environ.get('LAMBDA_TASK_ROOT')}")
print(f"[LAMBDA_MANIFEST] PATH={os.environ.get('PATH')}")
print(f"[LAMBDA_MANIFEST] REMBG_SESSION(default)=u2net (init happens in code)")
PY
log "Injected lambda/python/sitecustomize.py for early runtime diagnostics."

# ----------------------------
# 4) Write Dockerfiles (full & incremental) that COPY sitecustomize.py
#    These are complete files; safe to overwrite/update in-place.
# ----------------------------
mkdir -p lambda/utils lambda/python utils/animations

# --- Full Dockerfile (lambda/Dockerfile)
cat > "${DOCKERFILE_FULL}" << 'DOCKER'
# AWS Lambda Python 3.11 runtime container
FROM public.ecr.aws/lambda/python:3.11

# Diagnostics during build
RUN echo "[LAMBDA_MANIFEST] Base image: public.ecr.aws/lambda/python:3.11" && \
    python3 --version

# System deps for OpenCV & build
RUN yum update -y && \
    yum install -y mesa-libGL mesa-libGLU libgomp gcc gcc-c++ make && \
    yum clean all && rm -rf /var/cache/yum

# Python deps
COPY lambda/python/requirements-lambda.txt /tmp/requirements-lambda.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lambda.txt && rm /tmp/requirements-lambda.txt

# rembg + pinned deps (compatibility)
RUN pip install --no-cache-dir \
      numpy==1.24.3 \
      rembg==2.0.59 \
      onnxruntime==1.16.3 \
      opencv-python-headless==4.8.1.78 \
      imageio-ffmpeg==0.5.1 \
      pillow==10.4.0 && \
    python - <<'PY' || true
from rembg import new_session
new_session('u2net')
print('[LAMBDA_MANIFEST] U2Net model cached during build')
PY

# Copy code & animations
COPY lambda/python/lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY lambda/python/text_animation_processor.py ${LAMBDA_TASK_ROOT}/
COPY utils/animations/ ${LAMBDA_TASK_ROOT}/utils/animations/

# Early runtime debug (auto-imported by Python)
COPY lambda/python/sitecustomize.py ${LAMBDA_TASK_ROOT}/sitecustomize.py

# Helpful envs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Handler
CMD ["lambda_handler.lambda_handler"]
DOCKER
log "Wrote ${DOCKERFILE_FULL}"

# --- Incremental Dockerfile (lambda/Dockerfile.v5-incremental)
cat > "${DOCKERFILE_INCREMENTAL}" << 'DOCKER'
# Start from the working v4 image (existing in your ECR)
FROM 562404437786.dkr.ecr.us-east-1.amazonaws.com/text-animation-lambda:v4

# Build deps
RUN yum install -y gcc gcc-c++ make && \
    yum clean all && rm -rf /var/cache/yum

# rembg + pinned deps (compatibility)
RUN pip install --no-cache-dir \
      numpy==1.24.3 \
      rembg==2.0.59 \
      onnxruntime==1.16.3 \
      opencv-python-headless==4.8.1.78 \
      imageio-ffmpeg==0.5.1 \
      pillow==10.4.0 && \
    python - <<'PY' || true
from rembg import new_session
new_session('u2net')
print('[LAMBDA_MANIFEST] U2Net model cached during build (incremental)')
PY

# Updated processor & animations
COPY lambda/python/text_animation_processor.py ${LAMBDA_TASK_ROOT}/
COPY utils/animations/ ${LAMBDA_TASK_ROOT}/utils/animations/

# Early runtime debug (auto-imported by Python)
COPY lambda/python/sitecustomize.py ${LAMBDA_TASK_ROOT}/sitecustomize.py

# Handler unchanged
CMD ["lambda_handler.lambda_handler"]
DOCKER
log "Wrote ${DOCKERFILE_INCREMENTAL}"

# ----------------------------
# 5) BUILD OPTION A — Legacy builder (forces Docker V2 manifest)
#    This is the simplest & most reliable path for Lambda.
# ----------------------------
log "Building with legacy builder (DOCKER_BUILDKIT=0) to force Docker V2 manifest..."
export DOCKER_BUILDKIT=0
docker build --platform "${PLATFORM}" -t "${ECR_REPO}:${IMAGE_TAG}" -f "${DOCKERFILE_INCREMENTAL}" . \
  || fail "Legacy build failed."

# Tag & push
log "Tagging & pushing ${IMAGE_URI} ..."
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}" >/dev/null || fail "Push failed."

# ----------------------------
# 6) VERIFY manifest type & single-arch
# ----------------------------
log "Verifying manifest from local Docker (client must be logged in to ECR)..."
if command -v jq >/dev/null 2>&1; then
  docker manifest inspect "${IMAGE_URI}" | jq '{mediaType,platforms:.manifests[].platform?} // .' || true
else
  docker manifest inspect "${IMAGE_URI}" || true
fi

log "Verifying manifest via ECR (imageManifestMediaType)..."
aws ecr describe-images \
  --repository-name "${ECR_REPO}" \
  --image-ids imageTag="${IMAGE_TAG}" \
  --region "${AWS_REGION}" \
  --query 'imageDetails[0].imageManifestMediaType' \
  --output text

log "Expected mediaType: application/vnd.docker.distribution.manifest.v2+json (single arch)."
log "If you see oci.* or manifest.list/index.*, Lambda may reject it. Use OPTION B below or conversion fallback."

# ----------------------------
# 7) DEPLOY to Lambda
# ----------------------------
log "Updating Lambda function to use ${IMAGE_URI} ..."
aws lambda update-function-code \
  --function-name "${LAMBDA_FUNCTION}" \
  --image-uri "${IMAGE_URI}" \
  --region "${AWS_REGION}" \
  --publish >/dev/null || fail "Lambda update-function-code failed."

log "✓ Lambda updated. New version published."

# ----------------------------
# 8) (Optional) Quick invoke smoke-test (no video processing)
#    This confirms the container boots and prints [LAMBDA_MANIFEST] lines.
# ----------------------------
if command -v python3 >/dev/null 2>&1; then
python3 - <<'PY' || true
import json, boto3, os
print("[LAMBDA_MANIFEST] Invoking quick smoke-test (no payload)...")
client = boto3.client("lambda", region_name=os.environ.get("AWS_REGION","us-east-1"))
resp = client.invoke(FunctionName=os.environ.get("LAMBDA_FUNCTION","toontune-text-animation"),
                     InvocationType="RequestResponse",
                     Payload=b'{}')
print("[LAMBDA_MANIFEST] StatusCode:", resp.get("StatusCode"))
print("[LAMBDA_MANIFEST] Log tail (base64):", resp.get("LogResult","")[:128], "...")
print("[LAMBDA_MANIFEST] Done.")
PY
else
  log "Python3 not found; skipping quick invoke."
fi

# ----------------------------
# 9) Alternative BUILD OPTION B — buildx with Docker media types
#    Use if you must keep BuildKit enabled:
# ----------------------------
log "If deployment still fails, try Buildx with Docker media types (oci-mediatypes=false)."
log "Running Buildx push (this will overwrite the tag with a Docker V2 manifest):"
if docker buildx version >/dev/null 2>&1; then
  docker buildx build \
    --platform "${PLATFORM}" \
    -t "${IMAGE_URI}" \
    -f "${DOCKERFILE_INCREMENTAL}" \
    --provenance=false \
    --sbom=false \
    --output=type=registry,oci-mediatypes=false \
    . || log "Buildx push failed; you can skip if OPTION A worked."
  log "Re-verify manifest after buildx push (same commands as step 6)."
else
  log "Buildx not available; OPTION A should already have worked."
fi

# ----------------------------
# 10) Fallback: convert an existing OCI/index image to Docker V2 (tools optional)
# ----------------------------
cat <<'TXT'
[LAMBDA_MANIFEST] Fallback tools (use only if needed):
  # Using regctl (https://github.com/regclient/regclient)
  regctl image mod ${IMAGE_URI} --to-docker --replace

  # Using skopeo (https://github.com/containers/skopeo)
  # 1) Pull locally (if not), or copy directly between registries:
  #    skopeo copy --format v2s2 docker://${IMAGE_URI} docker://${IMAGE_URI}
TXT

# ----------------------------
# 11) Rollback (if needed)
# ----------------------------
cat <<'TXT'
[LAMBDA_MANIFEST] Rollback example:
  # Find previous image digest:
  aws ecr describe-images \
    --repository-name '"${ECR_REPO}"' \
    --region '"${AWS_REGION}"' \
    --query 'reverse(sort_by(imageDetails,&imagePushedAt))[*].{tag:imageTags[0],digest:imageDigest}' \
    --output table

  # Deploy by digest (replace with a known-good digest):
  aws lambda update-function-code \
    --function-name '"${LAMBDA_FUNCTION}"' \
    --image-uri '"${REGISTRY}/${ECR_REPO}@sha256:YOUR_OLD_DIGEST"' \
    --region '"${AWS_REGION}"' \
    --publish
TXT

# ----------------------------
# 12) What you should see next
# ----------------------------
log "Expect Lambda logs to include lines like:"
log "  [LAMBDA_MANIFEST] Python 3.11.x on Linux ... arch=x86_64"
log "  [LAMBDA_MANIFEST] U2Net model cached during build (incremental)"
log "If the animation still misbehaves, your code's existing [LAMBDA_ANIM] logs + these [LAMBDA_MANIFEST] logs will pinpoint issues."

# ----------------------------
# 13) Extra: Add two targeted debug lines to your Python (optional)
#     (Safe: no behavior change, only logs)
# ----------------------------
cat <<'PY'
# --- OPTIONAL PATCH HINT (manual) ---
# In lambda/python/text_animation_processor.py, near rembg init:
# Replace/augment the prints so you also emit [LAMBDA_MANIFEST] lines:

try:
    from rembg import remove, new_session
    from PIL import Image
    print("[LAMBDA_MANIFEST] rembg imported successfully")
    print("[LAMBDA_MANIFEST] Preparing rembg session (u2net)")
except ImportError as e:
    print(f"[LAMBDA_MANIFEST] FATAL: rembg import failed: {e}")
    sys.exit(1)

try:
    REMBG_SESSION = new_session('u2net')
    print("[LAMBDA_MANIFEST] rembg session initialized (u2net)")
except Exception as e:
    print(f"[LAMBDA_MANIFEST] FATAL: rembg session init failed: {e}")
    sys.exit(1)
PY

# ----------------------------
# 14) References (read these if you're curious)
# ----------------------------
cat <<'REFS'
[LAMBDA_MANIFEST] References:
  • AWS Lambda: Create a function using a container image
    https://docs.aws.amazon.com/lambda/latest/dg/images-create.html
    (Lambda supports Docker image manifest V2, schema 2 and OCI specs v1.0+)

  • Amazon ECR: Container image manifest format support
    https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-manifest-formats.html

  • Docker Buildx exporters (oci-mediatypes flag)
    https://docs.docker.com/build/exporters/
    https://docs.docker.com/build/exporters/oci-docker/

  • Docker manifest v2, schema 2 spec
    https://distribution.github.io/distribution/spec/manifest-v2-2/

  • (Optional tooling) regctl / skopeo
    https://github.com/regclient/regclient
    https://github.com/containers/skopeo
REFS

log "Runbook completed."