"""Draw a small synthetic wardrobe (flat-colored garment silhouettes on a plain background).

Used for offline demos, README screenshots and tests. These are NOT real photos - accuracy numbers must come from
your own wardrobe photos.

    python scripts/make_demo_wardrobe.py demo_wardrobe
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

W = 400
SHAPES = {
    "t-shirt": [(120, 60), (170, 50), (230, 50), (280, 60), (350, 130), (310, 170), (280, 140), (280, 340), (120, 340), (120, 140), (90, 170), (50, 130)],
    "shirt": [(120, 50), (170, 40), (200, 70), (230, 40), (280, 50), (360, 150), (330, 330), (290, 300), (280, 350), (120, 350), (110, 300), (70, 330), (40, 150)],
    "hoodie": [(110, 70), (170, 40), (230, 40), (290, 70), (360, 160), (330, 320), (290, 290), (285, 350), (115, 350), (110, 290), (70, 320), (40, 160)],
    "jacket": [(110, 50), (200, 40), (290, 50), (370, 160), (340, 340), (290, 320), (280, 360), (120, 360), (110, 320), (60, 340), (30, 160)],
    "jeans": [(120, 40), (280, 40), (300, 360), (215, 360), (200, 150), (185, 360), (100, 360)],
    "trousers": [(125, 40), (275, 40), (290, 360), (215, 360), (200, 140), (185, 360), (110, 360)],
    "shorts": [(110, 80), (290, 80), (310, 230), (215, 230), (200, 150), (185, 230), (90, 230)],
    "skirt": [(130, 80), (270, 80), (340, 300), (60, 300)],
    "dress": [(160, 40), (240, 40), (260, 140), (330, 360), (70, 360), (140, 140)],
    "sneakers": [(40, 200), (200, 170), (260, 220), (360, 230), (370, 280), (40, 280)],
    "formal shoes": [(40, 210), (180, 190), (250, 230), (350, 250), (360, 280), (40, 280)],
    "boots": [(120, 60), (240, 60), (250, 200), (360, 240), (370, 290), (120, 290)],
    "sandals": [(40, 250), (360, 250), (370, 280), (40, 280)],
}
SHAPES["track pants"] = SHAPES["trousers"]
SHAPES["track jacket"] = SHAPES["jacket"]
COLORS = {"navy": (25, 40, 90), "white": (240, 240, 240), "black": (25, 25, 25), "beige": (215, 195, 160),
          "red": (190, 35, 45), "blue": (50, 100, 200), "grey": (130, 130, 130), "green": (50, 130, 70),
          "brown": (105, 70, 45), "pink": (235, 150, 175)}
# (fine type, color) - a plausible 24-item wardrobe
WARDROBE = [("t-shirt", "white"), ("t-shirt", "black"), ("t-shirt", "blue"), ("t-shirt", "red"), ("shirt", "white"),
            ("shirt", "blue"), ("hoodie", "grey"), ("hoodie", "navy"), ("jacket", "black"), ("jacket", "beige"),
            ("jeans", "blue"), ("jeans", "black"), ("trousers", "beige"), ("trousers", "navy"), ("shorts", "grey"),
            ("skirt", "black"), ("dress", "red"), ("dress", "navy"), ("sneakers", "white"), ("sneakers", "black"),
            ("formal shoes", "brown"), ("formal shoes", "black"), ("boots", "brown"), ("sandals", "beige")]


def draw_item(fine: str, color: str) -> Image.Image:
    img = Image.new("RGB", (W, W), (236, 233, 228))          # plain paper-like background
    ImageDraw.Draw(img).polygon(SHAPES[fine], fill=COLORS[color])
    return img


def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, (fine, color) in enumerate(WARDROBE):
        draw_item(fine, color).save(out / f"{fine.replace(' ', '_')}_{color}{k:02d}.png")
    print(f"wrote {len(WARDROBE)} images to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_wardrobe")
