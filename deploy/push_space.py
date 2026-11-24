"""Upload the app to a Hugging Face Space (Docker) - the website and API at one public address.

    hf auth login                                   # once; paste a token with WRITE access
    python deploy/push_space.py                     # creates/updates <your-username>/outfit-maker
    python deploy/push_space.py --dry-run           # only show what would be uploaded

Run it from the repo root. It uploads the code, deploy/space/Dockerfile and the trained checkpoint
ml/checkpoints/compat.pt (not in git). Your photos and local database are never uploaded.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"node_modules", "dist", "media", "__pycache__", ".pytest_cache", ".ruff_cache", "tests", "notebooks"}
SKIP_SUFFIXES = {".db", ".pyc"}

SPACE_README = """---
title: Outfit Maker
emoji: 👕
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: AI wardrobe assistant - outfits that go together
---

# Outfit Maker

Upload photos of your clothes; the app removes the background, tags each item, suggests outfits for an occasion and
today's weather, and tells you what to buy next. Source code and results: {github}

This free Space may sleep when idle - the first visit after that takes a minute to wake up. Photos are kept in the
Space's database; without a `DATABASE_URL` secret that database resets when the Space restarts.
"""


def _copy_tree(src: Path, dst: Path):
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(part in SKIP_DIRS for part in rel.parts) or p.suffix in SKIP_SUFFIXES or p.name.endswith(".backup.db"):
            continue
        if p.is_file():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)


def assemble(out: Path, github: str) -> list[Path]:
    for d in ("backend", "ml", "frontend"):
        _copy_tree(ROOT / d, out / d)
    ck = ROOT / "ml" / "checkpoints" / "compat.pt"
    (out / "ml" / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ck, out / "ml" / "checkpoints" / "compat.pt")
    # Unix line endings: a Windows checkout may hold CRLF, which breaks multi-line RUN commands
    dockerfile = (ROOT / "deploy" / "space" / "Dockerfile").read_bytes().replace(b"\r\n", b"\n")
    (out / "Dockerfile").write_bytes(dockerfile)
    (out / "README.md").write_text(SPACE_README.format(github=github), encoding="utf8")
    return sorted(p for p in out.rglob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=None, help="<user>/<name>; default <your-username>/outfit-maker")
    ap.add_argument("--github", default="https://github.com/Amey1005/AI-Outfit-Maker")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not (ROOT / "ml" / "checkpoints" / "compat.pt").exists():
        sys.exit("ml/checkpoints/compat.pt is missing - download it from Google Drive first (see docs/DEPLOY.md).")

    with tempfile.TemporaryDirectory() as tmp:
        files = assemble(Path(tmp), a.github)
        total = sum(f.stat().st_size for f in files)
        print(f"{len(files)} files, {total / 1e6:.1f} MB")
        for f in files:
            rel = f.relative_to(tmp)
            if f.stat().st_size > 1e6 or rel.parts[0] not in ("backend", "ml", "frontend"):
                print(f"  {rel}  ({f.stat().st_size / 1e6:.1f} MB)")
        if a.dry_run:
            print("dry run - nothing uploaded")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        try:
            user = api.whoami()["name"]
        except Exception:
            sys.exit("Not logged in to Hugging Face. Run:  hf auth login   (use a token with WRITE access)")
        space = a.space or f"{user}/outfit-maker"
        api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space", commit_message="Deploy Outfit Maker")
        owner, name = space.split("/")
        host = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
        print(f"\nUploaded. Build progress: https://huggingface.co/spaces/{space}")
        print(f"The app (once the build finishes, ~10-15 min): https://{host}.hf.space")


if __name__ == "__main__":
    main()
