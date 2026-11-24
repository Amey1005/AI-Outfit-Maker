"""Weather lookup (Open-Meteo, no API key) and the clothing rules it triggers."""
from __future__ import annotations

import time

import httpx

CACHE_SECONDS = 30 * 60
_cache: dict[tuple, tuple[float, dict]] = {}

# Rules live in config so they are easy to tune.
WEATHER_RULES = {
    "hot_above_c": 28,
    "cold_below_c": 15,
    "hot_banned_fine": {"jacket", "coat", "hoodie", "sweater", "track jacket"},
    "cold_banned_fine": {"shorts", "sandals"},
    "cold_required_fine": {"jacket", "coat", "hoodie", "sweater", "track jacket"},   # at least one when cold
    "rain_banned_fine": {"sandals"},
    "rain_banned_material": {"suede", "canvas"},
}
RAIN_CODES = set(range(51, 68)) | set(range(80, 83)) | {95, 96, 99}


def fetch_weather(lat: float, lon: float, client: httpx.Client | None = None) -> dict | None:
    """-> {"temp_c", "rain"} or None if the service is unreachable (no weather filtering then)."""
    key = (round(lat, 2), round(lon, 2))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    try:
        c = client or httpx.Client(timeout=5)
        r = c.get("https://api.open-meteo.com/v1/forecast",
                  params={"latitude": lat, "longitude": lon, "current": "temperature_2m,precipitation,weathercode"})
        r.raise_for_status()
        cur = r.json()["current"]
        data = {"temp_c": float(cur["temperature_2m"]),
                "rain": float(cur.get("precipitation") or 0) > 0.1 or int(cur.get("weathercode") or 0) in RAIN_CODES}
    except Exception:
        return None
    _cache[key] = (time.time(), data)
    return data


def condition(weather: dict | None) -> str:
    if not weather:
        return "unknown"
    t = weather["temp_c"]
    parts = ["hot" if t > WEATHER_RULES["hot_above_c"] else "cold" if t < WEATHER_RULES["cold_below_c"] else "mild"]
    if weather.get("rain"):
        parts.append("rain")
    return "+".join(parts)


def outfit_allowed(items, weather: dict | None, require_layer: bool = True) -> bool:
    """`items` need .fine_type and .material. True when the outfit passes every active weather rule.

    `require_layer=False` drops only the "cold needs a warm layer" rule (used when the user owns no warm layer
    that suits the occasion; the caller then tells them to wear one)."""
    if not weather:
        return True
    r, t = WEATHER_RULES, weather["temp_c"]
    fines = {i.fine_type for i in items}
    mats = {i.material for i in items}
    if t > r["hot_above_c"] and (fines & r["hot_banned_fine"] or "knit" in mats):
        return False
    if t < r["cold_below_c"] and (fines & r["cold_banned_fine"] or (require_layer and not fines & r["cold_required_fine"])):
        return False
    if weather.get("rain") and (fines & r["rain_banned_fine"] or mats & r["rain_banned_material"]):
        return False
    return True
