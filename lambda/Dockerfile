# AWS Lambda Python 3.11 runtime container
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for OpenCV
RUN yum update -y && \
    yum install -y \
    mesa-libGL \
    mesa-libGLU \
    libgomp \
    && yum clean all && \
    rm -rf /var/cache/yum

# Copy requirements and install Python packages
COPY lambda/python/requirements-lambda.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-lambda.txt && \
    rm /tmp/requirements-lambda.txt

# Copy the Lambda handler and processor
COPY lambda/python/lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY lambda/python/text_animation_processor.py ${LAMBDA_TASK_ROOT}/

# Copy animation modules
COPY utils/animations/ ${LAMBDA_TASK_ROOT}/utils/animations/

# Set the handler
CMD ["lambda_handler.lambda_handler"]