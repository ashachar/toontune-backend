# Use AWS Lambda Python runtime as base
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum update -y && \
    yum install -y \
    libGL \
    libgthread-2.0 \
    libSM \
    libXext \
    libXrender \
    libgomp \
    wget \
    && yum clean all

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy animation utilities
COPY utils/ ${LAMBDA_TASK_ROOT}/utils/

# Copy Lambda handler
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Set the handler
CMD ["lambda_handler.lambda_handler"]