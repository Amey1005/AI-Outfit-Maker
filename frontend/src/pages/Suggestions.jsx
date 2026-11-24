import { useState } from "react";
import { api, imageSrc } from "../api.js";

const OCCASIONS = ["college", "party", "formal", "gym"];

export default function Suggestions() {
  const [occasion, setOccasion] = useState("college");
  const [temp, setTemp] = useState("");
  const [coords, setCoords] = useState(null);
  const [data, setData] = useState(null);
  const [voted, setVoted] = useState({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function locate() {
    navigator.geolocation?.getCurrentPosition(
      (p) => setCoords({ lat: p.coords.latitude, lon: p.coords.longitude }),
      () => setError("Location was blocked — type a temperature instead.")
    );
  }

  async function suggest() {
    setBusy(true);
    setError("");
    setVoted({});
    const json = { occasion, top_k: 6 };
    if (temp !== "") json.temp_c = Number(temp);
    else if (coords) Object.assign(json, coords);
    try { setData(await api("/outfits/suggest", { method: "POST", json })); } catch (e) { setError(e.message); }
    setBusy(false);
  }

  async function vote(k, outfit, liked) {
    await api("/feedback", { method: "POST", json: { item_ids: outfit.item_ids, liked, occasion } });
    setVoted((v) => ({ ...v, [k]: liked ? "👍" : "👎" }));
  }
  async function wore(k, outfit) {
    await api("/wear", { method: "POST", json: { item_ids: outfit.item_ids } });
    setVoted((v) => ({ ...v, [k]: "worn" }));
  }

  return (
    <section className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        {OCCASIONS.map((o) => (
          <button key={o} onClick={() => setOccasion(o)} aria-pressed={occasion === o} className={`chip ${occasion === o ? "chip-on" : "chip-off"}`}>{o}</button>
        ))}
        <input value={temp} onChange={(e) => setTemp(e.target.value)} placeholder="Temp °C (optional)" type="number" className="field w-40 px-3" />
        <button onClick={locate} className="link text-sm">{coords ? "Location set ✓" : "Use my location"}</button>
        <button onClick={suggest} disabled={busy} className="btn-primary ml-auto">{busy ? "Thinking…" : "Suggest outfits"}</button>
      </div>
      {error && <p className="error">{error}</p>}
      {data && (
        <p className="flex flex-wrap items-center gap-2 text-sm text-charcoal/60">
          Scored by <b className="text-charcoal">{data.scorer.replace("_", " ")}</b>
          {data.weather && (
            <span className="rounded-full bg-peach px-2.5 py-0.5 text-xs font-medium text-charcoal">
              {data.weather.temp_c.toFixed(0)}°C · {data.weather.condition}
            </span>
          )}
        </p>
      )}
      {data?.notes?.map((n, i) => <p key={i} className="note">{n}</p>)}
      <ul className="grid gap-5 sm:grid-cols-2">
        {data?.outfits.map((o, k) => (
          <li key={k} className="card card-hover p-4">
            <div className="flex gap-2">
              {o.items.map((it) => (
                <div key={it.id} className="group min-w-0 flex-1 overflow-hidden rounded-xl ring-1 ring-warmgrey">
                  <img src={imageSrc(it.image_url)} alt={it.type} title={`${it.colors[0]?.name} ${it.type}`} className="photo" />
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-charcoal/60">match {o.compatibility.toFixed(2)} · occasion fit {o.occasion.toFixed(2)}</p>
            <div className="mt-2 flex items-center gap-2 text-sm">
              {voted[k] ? <span className="text-charcoal/70">{voted[k] === "worn" ? "Marked as worn" : `You voted ${voted[k]}`}</span> : (
                <>
                  <button onClick={() => vote(k, o, true)} aria-label="Like" className="rounded-full px-3 py-1 ring-1 ring-warmgrey transition hover:bg-peach/40 hover:ring-peach active:scale-95">👍</button>
                  <button onClick={() => vote(k, o, false)} aria-label="Dislike" className="rounded-full px-3 py-1 ring-1 ring-warmgrey transition hover:bg-peach/40 hover:ring-peach active:scale-95">👎</button>
                  <button onClick={() => wore(k, o)} className="link ml-auto">I wore this</button>
                </>
              )}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
