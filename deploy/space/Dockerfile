# Hugging Face Space (Docker SDK): the website and the API in one container, on port 7860.
# Build context is the repo root:  docker build -f deploy/space/Dockerfile .
# (deploy/push_space.py uploads the same layout to the Space, plus the trained checkpoint.)

FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci || npm install
COPY frontend/ ./
# empty API address = same address as the website
RUN echo "VITE_API_URL=" > .env.production.local && npm run build

FROM python:3.11-slim
# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
ENV PYTHONUNBUFFERED=1 HOME=/home/user PYTHONPATH=/srv:/srv/backend \
    HF_HOME=/home/user/.cache/huggingface U2NET_HOME=/home/user/.u2net \
    PORT=7860 MEDIA_STORE=db FRONTEND_DIR=/srv/web CORS_ORIGINS="" \
    DATABASE_URL=sqlite:////home/user/data/outfit.db \
    COMPAT_CHECKPOINT=/srv/ml/checkpoints/compat.pt
WORKDIR /srv

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=user ml ml
COPY --chown=user backend backend
COPY --chown=user --from=web /web/dist web

USER user
# Download FashionCLIP and the background-removal model at build time, so a restart starts quickly.
RUN mkdir -p /home/user/data \
 && python -c "from transformers import CLIPModel, CLIPProcessor; m = 'patrickjohncyh/fashion-clip'; CLIPModel.from_pretrained(m); CLIPProcessor.from_pretrained(m)" \
 && python -c "from rembg import new_session; new_session('u2netp')"

WORKDIR /srv/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
