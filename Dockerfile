FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY cast-openenv/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY cast-openenv/ ./

# Environment variables (injected at runtime by OpenEnv)
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

# Run inference
CMD ["python", "inference.py"]
