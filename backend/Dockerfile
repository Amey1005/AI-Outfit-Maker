# Build from the repo root:  docker build -f backend/Dockerfile -t outfit-api .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/srv:/srv/backend HF_HOME=/srv/.hf U2NET_HOME=/srv/.u2net
WORKDIR /srv

# CPU-only PyTorch keeps the image far smaller than the default CUDA wheels.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Model weights are NOT baked in: FashionCLIP downloads from Hugging Face on first start.
# A trained checkpoint, if present, is copied from ml/checkpoints/ (see docs/DEPLOY.md).
COPY ml ml
COPY backend backend

WORKDIR /srv/backend
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
