"""Shared vocabulary: coarse outfit types, fine clothing labels, patterns, materials, occasions."""

COARSE_TYPES = ["top", "bottom", "shoes", "outerwear", "allbody"]
COARSE_INDEX = {t: i for i, t in enumerate(COARSE_TYPES)}

# Polyvore `semantic_category` -> coarse type. Accessories, bags, jewellery etc. are ignored.
SEMANTIC_TO_COARSE = {
    "tops": "top",
    "bottoms": "bottom",
    "shoes": "shoes",
    "outerwear": "outerwear",
    "all-body": "allbody",
}

# Fine labels used for zero-shot tagging of wardrobe photos -> coarse type.
FINE_TO_COARSE = {
    "t-shirt": "top",
    "shirt": "top",
    "hoodie": "top",
    "sweater": "top",
    "jacket": "outerwear",
    "coat": "outerwear",
    "track jacket": "outerwear",
    "jeans": "bottom",
    "trousers": "bottom",
    "track pants": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "dress": "allbody",
    "sneakers": "shoes",
    "formal shoes": "shoes",
    "sandals": "shoes",
    "boots": "shoes",
}
FINE_LABELS = list(FINE_TO_COARSE)

PATTERNS = ["solid", "striped", "checked", "floral", "printed"]
MATERIALS = ["cotton", "denim", "knit", "leather", "suede", "canvas"]
OCCASIONS = ["college", "party", "formal", "gym"]

OCCASION_PROMPTS = {
    "college": ["a casual college outfit", "a relaxed everyday student outfit"],
    "party": ["a stylish party outfit", "a fashionable night out outfit"],
    "formal": ["a formal office outfit", "a smart business outfit"],
    "gym": ["a sporty gym workout outfit", "comfortable athletic wear"],
}

# Who an item is cut for, chosen when it is uploaded. Menswear items are never tagged as these types, and one
# outfit never mixes menswear with womenswear. The wardrobe's mix decides the style of "buy next" suggestions.
WEAR = ["menswear", "womenswear", "unisex"]
STYLE_PREFS = ["all", "menswear", "womenswear"]
WOMENSWEAR_ONLY = {"dress", "skirt"}


def labels_for(style: str | None) -> list[str]:
    return [l for l in FINE_LABELS if not (style == "menswear" and l in WOMENSWEAR_ONLY)]


# Occasion rules on fine types. An occasion with an ALLOWED set may use only those types (gym clothes are a
# narrow category); otherwise BANNED types are excluded outright and DISCOURAGED ones only lower the score.
OCCASION_ALLOWED = {
    "gym": {"t-shirt", "hoodie", "sweater", "track jacket", "track pants", "shorts", "sneakers"},
}
OCCASION_BANNED = {
    "college": set(),
    "party": {"track jacket", "track pants"},
    "formal": {"hoodie", "track jacket", "track pants", "shorts", "sandals"},
}
OCCASION_DISCOURAGED = {
    "formal": {"t-shirt", "sneakers"},
}


def occasion_allows(fine_label: str, occasion: str) -> bool:
    if occasion in OCCASION_ALLOWED:
        return fine_label in OCCASION_ALLOWED[occasion]
    return fine_label not in OCCASION_BANNED.get(occasion, set())


# Outfit templates for candidate generation: lists of coarse types (optional slots handled in recommender).
REQUIRED_TEMPLATES = [("top", "bottom", "shoes"), ("allbody", "shoes")]


def coarse_of(fine_label: str) -> str:
    return FINE_TO_COARSE.get(fine_label, "top")
