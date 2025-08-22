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
