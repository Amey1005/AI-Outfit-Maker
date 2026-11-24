# Deploying

## Recommended: one free Hugging Face Space (website + API at one address)

Free web hosts usually give ~512 MB of memory; FashionCLIP + PyTorch need several GB. A free Hugging Face Space
(Docker, CPU basic: 2 vCPU, 16 GB RAM) fits, and serves the built website from the same container as the API, so
there is one public link and no CORS setup.

1. Put the trained checkpoint at `ml/checkpoints/compat.pt` (it is not in git; download it from Google Drive).
2. Create a Hugging Face token with **Write** access (huggingface.co -> Settings -> Access Tokens) and log in:
   `hf auth login`
3. From the repo root: `python deploy/push_space.py --dry-run` (shows what will be uploaded), then
   `python deploy/push_space.py`.
4. Watch the build at `https://huggingface.co/spaces/<you>/outfit-maker` (about 10-15 minutes the first time).
   The app is then at `https://<you>-outfit-maker.hf.space`.

What the Space image does (`deploy/space/Dockerfile`):
* builds the React app with an empty `VITE_API_URL`, so it calls the API on its own address;
* installs CPU-only PyTorch and downloads FashionCLIP and the `u2netp` background model at build time, so a restart
  is quick;
* stores photos **in the database** (`MEDIA_STORE=db`), because a Space's disk is wiped on restart;
* uses SQLite in the container by default.

### Keeping data across restarts

With the default SQLite database, wardrobes are lost whenever the Space restarts (after a rebuild or a long sleep).
To keep them, create a free Postgres database (for example on neon.tech), copy its connection string, and add it in
the Space -> Settings -> **Variables and secrets** -> New secret, named `DATABASE_URL`. Photos are stored in that
database too.

### Notes

* A free Space sleeps after a period without visitors; the first visit after that takes about a minute.
* The checkpoint is uploaded to the Space with the code (about 20 MB); your local photos and database are not.
* `.github/workflows/ci.yml` builds the Space image on every push, so a broken Dockerfile shows up on GitHub first.

## Alternative: separate frontend and backend

* **Frontend** on Vercel: import the repo, root directory `frontend/`, env var `VITE_API_URL=https://<your-api-host>`.
* **Backend** from `backend/Dockerfile` (repo root as build context) on any host with enough memory. Set
  `DATABASE_URL`, `CORS_ORIGINS` (your Vercel URL), `MEDIA_STORE=db` if its disk is not persistent, and
  `COMPAT_CHECKPOINT`.
* Do not commit model weights. FashionCLIP downloads from Hugging Face; ship the small checkpoint with the image.
