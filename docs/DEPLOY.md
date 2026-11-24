# Deploying

## Shape of the deployment

* **Frontend** -> Vercel: import the repo, set root directory `frontend/`, env var `VITE_API_URL=https://<your-api-host>`.
* **Backend** -> Render / Railway / Hugging Face Space (Docker), built from `backend/Dockerfile` with the **repo root as build context** (`docker build -f backend/Dockerfile .`). Set env vars:
  * `DATABASE_URL` - hosted Postgres (Neon, or the platform's own; check current free-tier limits)
  * `CORS_ORIGINS` - your Vercel URL
  * `COMPAT_CHECKPOINT` - path of the trained checkpoint (see below)
  * `EMBED_BACKEND=fashionclip`
* The platform provides `PORT`; the container honours it.

## Memory warning

Free backend tiers often give ~512 MB RAM. FashionCLIP + PyTorch + rembg may not fit. Options, cheapest first:

1. Use the `u2netp` background model (already the default) or set `USE_REMBG=0` to use the lightweight border-mask fallback.
2. Host the backend as a Hugging Face Space (Docker) - more RAM on the free CPU tier.
3. Run a smaller demo mode (`EMBED_BACKEND=lite`) and say so clearly in the README, plus a recorded demo GIF of the full version.

Free instances sleep; the first request after idle is slow - say so in the README.

## Model files

Do not commit weights. FashionCLIP downloads from Hugging Face at startup. Host the small trained checkpoint (`compat.pt`, a few MB) on the Hugging Face Hub or commit it to `ml/checkpoints/` (it is git-ignored by default; remove the ignore line to ship it) so the Dockerfile's `COPY ml ml` includes it.

## Media storage

Uploaded photos are written to `MEDIA_DIR`. Most free tiers have ephemeral disks, so photos disappear on redeploy - fine for a demo, but mention it. For persistence, mount a volume (as `docker-compose.yml` does) or move storage to object storage.

## Checklist

- [ ] `docker compose up --build` works locally
- [ ] CI green on GitHub (`.github/workflows/ci.yml`)
- [ ] Frontend deployed and pointing at the backend
- [ ] Checkpoint present: `POST /outfits/suggest` response says `"scorer": "compatibility_model"`
- [ ] README has the live link, demo GIF (ScreenToGif) and architecture diagram (excalidraw / draw.io)
