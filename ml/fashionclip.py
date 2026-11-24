"""Thin FashionCLIP wrapper (Hugging Face `patrickjohncyh/fashion-clip`) shared by ML scripts and backend."""
from __future__ import annotations

import numpy as np

MODEL_ID = "patrickjohncyh/fashion-clip"


def _as_tensor(out):
    # Newer transformers versions return an output object; use its pooler_output (512-d).
    return out.pooler_output if hasattr(out, "pooler_output") else out


class FashionClip:
    def __init__(self, device: str | None = None, fp16: bool = True):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(self.device, dtype=self.dtype).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self._text_cache: dict[str, np.ndarray] = {}

    def embed_images(self, images) -> np.ndarray:
        """list[PIL.Image] -> L2-normalised float32 array [N,512]."""
        torch = self.torch
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.no_grad():
            f = _as_tensor(self.model.get_image_features(pixel_values=inputs["pixel_values"])).float()
        f = torch.nn.functional.normalize(f, dim=-1)
        return f.cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in texts if t not in self._text_cache]
        if todo:
            torch = self.torch
            inputs = self.processor(text=todo, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                f = _as_tensor(self.model.get_text_features(**inputs)).float()
            f = torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float32)
            for t, v in zip(todo, f):
                self._text_cache[t] = v
        return np.stack([self._text_cache[t] for t in texts])

    def zero_shot(self, image_emb: np.ndarray, labels: list[str], templates=("a photo of a {}", "a product image of a {}")):
        """Pick the best label by averaged-prompt cosine similarity. Returns (label, softmax prob, all probs)."""
        text = np.stack([self.embed_texts([t.format(l) for t in templates]).mean(0) for l in labels])
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        logits = 100.0 * (text @ image_emb)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        i = int(p.argmax())
        return labels[i], float(p[i]), p
