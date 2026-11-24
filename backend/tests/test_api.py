import numpy as np
from conftest import png

from app.services import weather as wx


def test_upload_tags_item_and_lists(client, user):
    r = client.post("/wardrobe/items", headers=user, files={"file": ("n.png", png("t-shirt", "navy"), "image/png")}, data={"type": "t-shirt"})
    assert r.status_code == 201
    item = r.json()
    assert item["type"] == "t-shirt" and item["coarse_type"] == "top"
    assert item["colors"][0]["name"] == "navy"
    assert client.get("/wardrobe/items", headers=user).json()[0]["id"] == item["id"]
    assert client.get(item["image_url"]).status_code == 200


def test_bad_inputs(client, user):
    assert client.get("/wardrobe/items").status_code == 422                       # missing header
    assert client.get("/wardrobe/items", headers={"X-User-Id": "!!"}).status_code == 400
    bad = client.post("/wardrobe/items", headers=user, files={"file": ("n.txt", b"not an image", "text/plain")})
    assert bad.status_code == 400
    r = client.post("/wardrobe/items", headers=user, files={"file": ("n.png", png("jeans", "blue"), "image/png")}, data={"type": "spaceship"})
    assert r.status_code == 400


def test_users_are_isolated(client, user, stocked):
    other = {"X-User-Id": "11111111-2222-3333-4444-555555555555"}
    assert client.get("/wardrobe/items", headers=other).json() == []
    some_id = next(iter(stocked[1].values()))
    assert client.delete(f"/wardrobe/items/{some_id}", headers=other).status_code == 404


def test_suggest_returns_valid_diverse_outfits(client, stocked):
    user, ids = stocked
    r = client.post("/outfits/suggest", headers=user, json={"occasion": "college", "top_k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["scorer"] == "color_rules"          # no checkpoint in tests
    outfits = body["outfits"]
    assert len(outfits) == 10
    counts = {}
    for o in outfits:
        types = [i["coarse_type"] for i in o["items"]]
        assert ("top" in types and "bottom" in types and "shoes" in types) or ("allbody" in types and "shoes" in types)
        for i in o["item_ids"]:
            counts[i] = counts.get(i, 0) + 1
    assert max(counts.values()) <= 2                  # diversity rule
    scores = [o["score"] for o in outfits]
    assert scores == sorted(scores, reverse=True)


def test_weather_rules(client, stocked):
    user, ids = stocked
    hot = client.post("/outfits/suggest", headers=user, json={"occasion": "college", "temp_c": 35}).json()
    banned = {"jacket", "coat", "hoodie", "sweater"}
    assert hot["weather"]["condition"] == "hot" and hot["outfits"]
    assert all(i["type"] not in banned for o in hot["outfits"] for i in o["items"])
    cold = client.post("/outfits/suggest", headers=user, json={"occasion": "college", "temp_c": 5}).json()
    assert cold["outfits"] and all(any(i["type"] in banned for i in o["items"]) for o in cold["outfits"])
    assert all(i["type"] not in {"shorts", "sandals"} for o in cold["outfits"] for i in o["items"])
    rain = client.post("/outfits/suggest", headers=user, json={"occasion": "college", "temp_c": 20, "rain": True}).json()
    assert all(i["type"] != "sandals" for o in rain["outfits"] for i in o["items"])


def test_occasion_changes_ranking(client, stocked):
    user, _ = stocked
    formal = client.post("/outfits/suggest", headers=user, json={"occasion": "formal", "top_k": 3}).json()["outfits"]
    gym = client.post("/outfits/suggest", headers=user, json={"occasion": "gym", "top_k": 3}).json()["outfits"]
    assert {i["type"] for i in formal[0]["items"]} & {"formal shoes", "trousers", "shirt"}
    assert {i["type"] for i in gym[0]["items"]} & {"sneakers", "shorts", "t-shirt"}
    assert client.post("/outfits/suggest", headers=user, json={"occasion": "wedding"}).status_code == 422


def test_matches(client, stocked):
    user, ids = stocked
    m = client.get(f"/wardrobe/items/{ids[('jeans', 'blue')]}/matches", headers=user).json()
    assert len(m) == 8 and all(x["item"]["coarse_type"] != "bottom" for x in m)
    assert [x["score"] for x in m] == sorted((x["score"] for x in m), reverse=True)


def test_personalization_shifts_toward_blue(client, stocked):
    """Simulated user who only likes blue outfits and dislikes the rest: top-10 should get bluer."""
    user, ids = stocked
    def blue_share():
        out = client.post("/outfits/suggest", headers=user, json={"occasion": "college", "top_k": 10}).json()["outfits"]
        names = [i["colors"][0]["name"] for o in out for i in o["items"]]
        return names.count("blue") / len(names)
    before = blue_share()
    blue = [ids[("t-shirt", "blue")], ids[("jeans", "blue")]]
    other = [ids[("t-shirt", "red")], ids[("trousers", "beige")]]
    for _ in range(8):
        client.post("/feedback", headers=user, json={"item_ids": blue, "liked": True})
        client.post("/feedback", headers=user, json={"item_ids": other, "liked": False})
    assert blue_share() > before


def test_gap_analysis(client, stocked):
    user, _ = stocked
    g = client.get("/wardrobe/gaps", headers=user).json()
    assert len(g["suggestions"]) == 5
    assert g["suggestions"][0]["new_outfits"] >= g["suggestions"][-1]["new_outfits"]


def test_delete_my_data(client, stocked):
    user, ids = stocked
    url = client.get("/wardrobe/items", headers=user).json()[0]["image_url"]
    assert client.get(url).status_code == 200
    assert client.post("/feedback", headers=user, json={"item_ids": list(ids.values())[:2], "liked": True}).status_code == 201
    assert client.delete("/users/me/data", headers=user).status_code == 204
    assert client.get("/wardrobe/items", headers=user).json() == []
    assert client.get(url).status_code == 404


def test_latency_metrics(client, stocked):
    user, _ = stocked
    client.post("/outfits/suggest", headers=user, json={"occasion": "party"})
    m = client.get("/metrics/latency").json()
    assert m["POST /outfits/suggest"]["p95_ms"] >= m["POST /outfits/suggest"]["p50_ms"] > 0


def test_weather_fetch_parses_and_caches():
    calls = []

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"current": {"temperature_2m": 31.2, "precipitation": 0.0, "weathercode": 1}}

    class Client:
        def get(self, *a, **k):
            calls.append(1)
            return Resp()

    wx._cache.clear()
    assert wx.fetch_weather(12.97, 77.59, Client()) == {"temp_c": 31.2, "rain": False}
    wx.fetch_weather(12.97, 77.59, Client())
    assert len(calls) == 1                                         # second call served from the 30-minute cache
    assert wx.condition({"temp_c": 31.2, "rain": True}) == "hot+rain"
    assert np.isclose(wx.WEATHER_RULES["hot_above_c"], 28)


def _add(client, user, fine, color, wear="unisex"):
    r = client.post("/wardrobe/items", headers=user, files={"file": ("x.png", png(fine, color), "image/png")},
                    data={"type": fine, "wear": wear})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_gym_never_uses_a_party_jacket_even_when_cold(client, stocked):
    user, _ = stocked
    _add(client, user, "track pants", "black")                    # the demo wardrobe's only gym bottoms are shorts
    body = client.post("/outfits/suggest", headers=user, json={"occasion": "gym", "temp_c": 8}).json()
    assert body["outfits"]
    for o in body["outfits"]:
        types = {i["type"] for i in o["items"]}
        assert not types & {"jacket", "coat", "shirt", "jeans", "trousers", "formal shoes", "boots"}
        assert types & {"hoodie", "sweater", "track jacket"}       # cold: a gym-suitable warm layer


def test_gym_without_gym_bottoms_explains_why(client, user):
    for fine, color in [("t-shirt", "white"), ("jeans", "blue"), ("sneakers", "white")]:
        _add(client, user, fine, color)
    body = client.post("/outfits/suggest", headers=user, json={"occasion": "gym"}).json()
    assert body["outfits"] == []
    assert "bottoms" in body["notes"][0] and "track pants" in body["notes"][0]


def test_cold_without_a_suitable_layer_still_suggests_with_a_note(client, user):
    for fine, color in [("t-shirt", "white"), ("track pants", "black"), ("sneakers", "white"), ("jacket", "beige")]:
        _add(client, user, fine, color)
    body = client.post("/outfits/suggest", headers=user, json={"occasion": "gym", "temp_c": 5}).json()
    assert body["outfits"] and all("jacket" not in {i["type"] for i in o["items"]} for o in body["outfits"])
    assert "warm layer" in body["notes"][0]


def test_single_pair_of_shoes_does_not_cap_results(client, user):
    for fine, color in [("t-shirt", "white"), ("t-shirt", "black"), ("t-shirt", "blue"), ("jeans", "blue"),
                        ("jeans", "black"), ("trousers", "beige"), ("sneakers", "white")]:
        _add(client, user, fine, color)
    body = client.post("/outfits/suggest", headers=user, json={"occasion": "college", "top_k": 6}).json()
    assert len(body["outfits"]) == 6


def test_fixing_an_item_type(client, user):
    item = _add(client, user, "trousers", "black")
    r = client.patch(f"/wardrobe/items/{item}", headers=user, json={"type": "track pants"})
    assert r.status_code == 200 and r.json()["type"] == "track pants" and r.json()["coarse_type"] == "bottom"
    assert client.patch(f"/wardrobe/items/{item}", headers=user, json={"type": "spaceship"}).status_code == 400
    other = {"X-User-Id": "99999999-2222-3333-4444-555555555555"}
    assert client.patch(f"/wardrobe/items/{item}", headers=other, json={"type": "jeans"}).status_code == 404


def test_gap_analysis_reports_every_category(client, stocked):
    user, _ = stocked
    g = client.get("/wardrobe/gaps", headers=user).json()
    cats = {c["category"] for c in g["by_category"]}
    assert {"top", "bottom", "shoes"} <= cats
    for c in g["by_category"]:
        names = [p["name"] for p in c["picks"]]
        assert len(names) == len(set(names))                      # distinct styles, not one style in 3 colors


def test_fixing_an_item_color(client, user):
    item = _add(client, user, "jeans", "navy")
    r = client.patch(f"/wardrobe/items/{item}", headers=user, json={"color": "blue"})
    assert r.status_code == 200
    colors = r.json()["colors"]
    assert colors[0]["name"] == "blue" and [c["name"] for c in colors].count("blue") == 1
    assert client.get("/wardrobe/items", headers=user).json()[0]["colors"][0]["name"] == "blue"
    assert r.json()["type"] == "jeans"                                   # a color change keeps the type
    assert client.patch(f"/wardrobe/items/{item}", headers=user, json={"color": "chartreuse"}).status_code == 400


def test_wear_is_stored_and_outfits_never_mix_menswear_with_womenswear(client, user):
    for fine, color in [("t-shirt", "white"), ("jeans", "blue"), ("sneakers", "white")]:
        _add(client, user, fine, color, "menswear")
    dress = _add(client, user, "dress", "red", "womenswear")
    heels = _add(client, user, "formal shoes", "black", "womenswear")
    assert {i["wear"] for i in client.get("/wardrobe/items", headers=user).json()} == {"menswear", "womenswear"}
    outfits = client.post("/outfits/suggest", headers=user, json={"occasion": "party", "top_k": 20}).json()["outfits"]
    assert outfits
    for o in outfits:
        assert len({i["wear"] for i in o["items"]} - {"unisex"}) == 1
    assert any(dress in o["item_ids"] and heels in o["item_ids"] for o in outfits)
    assert client.post("/wardrobe/items", headers=user, files={"file": ("x.png", png("jeans", "blue"), "image/png")},
                       data={"wear": "aliens"}).status_code == 400
    r = client.patch(f"/wardrobe/items/{dress}", headers=user, json={"wear": "unisex"})
    assert r.status_code == 200 and r.json()["wear"] == "unisex"


def test_buy_next_follows_the_wardrobe(client, user):
    for fine, color in [("t-shirt", "white"), ("jeans", "blue"), ("sneakers", "white"), ("shirt", "blue")]:
        _add(client, user, fine, color, "menswear")
    g = client.get("/wardrobe/gaps", headers=user).json()
    assert g["style"] == "menswear"
    names = {p["name"] for c in g["by_category"] for p in c["picks"]}
    assert not names & {"dress", "midi skirt", "blouse", "crop top", "heels", "ballet flats"}
    assert all(c["category"] != "allbody" for c in g["by_category"])
    assert client.get("/wardrobe/gaps?style=robots", headers=user).status_code == 422


def test_buy_suggestions_have_shopping_links(client, stocked):
    user, _ = stocked
    g = client.get("/wardrobe/gaps?style=menswear", headers=user).json()
    pick = g["by_category"][0]["picks"][0]
    assert pick["shop_url"].startswith("https://www.google.com/search?tbm=shop&q=")
    assert pick["shop_url"].endswith("+men") and pick["name"].split()[0] in pick["shop_url"]


def test_upload_keeps_original_and_shows_a_cutout(client, user):
    from app.config import settings

    item = client.post("/wardrobe/items", headers=user, files={"file": ("x.png", png("jeans", "blue"), "image/png")},
                       data={"type": "jeans"}).json()
    assert item["image_url"].endswith("_cut.jpg")
    shown = settings.media_dir / item["image_url"].removeprefix("/media/")
    original = shown.with_name(shown.name.replace("_cut.jpg", ".jpg"))
    assert shown.exists() and original.exists()
    from PIL import Image

    with Image.open(shown) as im:
        assert im.size == (640, 640) and im.getpixel((5, 5)) == (255, 255, 255)   # background removed to white
    assert client.delete(f"/wardrobe/items/{item['id']}", headers=user).status_code == 204
    assert not shown.exists() and not original.exists()


def test_photos_can_live_in_the_database(client, user, monkeypatch):
    """MEDIA_STORE=db (used on hosts whose disk is wiped on restart): nothing is written to disk."""
    from app.config import settings

    monkeypatch.setattr(settings, "media_store", "db")
    item = client.post("/wardrobe/items", headers=user, files={"file": ("x.png", png("shirt", "blue"), "image/png")},
                       data={"type": "shirt"}).json()
    assert not (settings.media_dir / item["image_url"].removeprefix("/media/")).exists()
    r = client.get(item["image_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    assert client.delete(f"/wardrobe/items/{item['id']}", headers=user).status_code == 204
    assert client.get(item["image_url"]).status_code == 404

    item = client.post("/wardrobe/items", headers=user, files={"file": ("x.png", png("shirt", "blue"), "image/png")},
                       data={"type": "shirt"}).json()
    assert client.delete("/users/me/data", headers=user).status_code == 204
    assert client.get(item["image_url"]).status_code == 404


def test_media_route_refuses_paths_outside_the_media_folder(client):
    assert client.get("/media/../../app/config.py").status_code == 404
    assert client.get("/media/%2e%2e/%2e%2e/app/config.py").status_code == 404
