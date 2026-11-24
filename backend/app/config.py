import os
import sys
from pathlib import Path

# The shared `ml` package lives at the repo root (or /srv/ml in Docker).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class Settings:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", f"sqlite:///{_ROOT / 'backend' / 'outfit.db'}")
        if self.database_url.startswith("postgres://"):  # some hosts hand out the legacy scheme
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
        self.media_dir = Path(os.getenv("MEDIA_DIR", _ROOT / "backend" / "media"))
        # "fashionclip" (real tagging + embeddings) or "lite" (no-download color-histogram fallback for dev/tests)
        self.embed_backend = os.getenv("EMBED_BACKEND", "fashionclip")
        self.checkpoint = Path(os.getenv("COMPAT_CHECKPOINT", _ROOT / "ml" / "checkpoints" / "compat.pt"))
        self.cors_origins = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o]
        self.max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
        self.use_rembg = os.getenv("USE_REMBG", "1") == "1"


settings = Settings()
