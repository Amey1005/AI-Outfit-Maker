import { useEffect, useState } from "react";
import { api, COLORS, imageSrc, TYPES, WEAR } from "../api.js";

export default function Wardrobe({ goTo }) {
  const [items, setItems] = useState(null);
  const [selected, setSelected] = useState(null);
  const [matches, setMatches] = useState([]);
  const [gaps, setGaps] = useState(null);
  const [gapsBusy, setGapsBusy] = useState(false);
  const [bulkBusy, setBulkBusy] = useState(false);
  const [error, setError] = useState("");

  const load = () => api("/wardrobe/items").then(setItems).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);
  useEffect(() => {
    if (selected) document.getElementById("item-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selected?.id]);

  async function pick(item) {
    setSelected(item);
    setMatches([]);
    try { setMatches(await api(`/wardrobe/items/${item.id}/matches`)); } catch (e) { setError(e.message); }
  }
  async function update(item, change) {
    setError("");
    try {
      const updated = await api(`/wardrobe/items/${item.id}`, { method: "PATCH", json: change });
      setItems((all) => all.map((i) => (i.id === updated.id ? updated : i)));
      setGaps(null);
      pick(updated);
    } catch (e) { setError(e.message); }
  }
  async function remove(item) {
    await api(`/wardrobe/items/${item.id}`, { method: "DELETE" });
    setSelected(null);
    load();
  }
  async function findGaps() {
    setError("");
    setGapsBusy(true);
    try { setGaps(await api("/wardrobe/gaps")); } catch (e) { setError(e.message); }
    setGapsBusy(false);
  }
  async function markAll(wear) {
    setError("");
    setBulkBusy(true);
    try {
      for (const it of items.filter((i) => i.wear === "unisex")) {
        await api(`/wardrobe/items/${it.id}`, { method: "PATCH", json: { wear } });
      }
      setGaps(null);
      await load();
    } catch (e) { setError(e.message); }
    setBulkBusy(false);
  }

  if (items === null) return <p className="text-charcoal/60">{error || "Loading…"}</p>;
  if (items.length === 0)
    return (
      <div className="card p-10 text-center">
        <p className="text-lg font-semibold">Your wardrobe is empty</p>
        <p className="mt-1 text-sm text-charcoal/60">Snap a few clothes and the app will tag them for you.</p>
        <button onClick={() => goTo("upload")} className="btn-primary mt-5">Add your first items</button>
      </div>
    );

  const unisex = items.filter((i) => i.wear === "unisex").length;

  return (
    <section className="space-y-6">
      {error && <p className="error">{error}</p>}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-charcoal/60">{items.length} items — click one to see what matches it.</p>
        <button onClick={findGaps} disabled={gapsBusy} className="btn-primary">{gapsBusy ? "Thinking…" : "What should I buy next?"}</button>
      </div>

      {unisex > 0 && (
        <div className="note flex flex-wrap items-center gap-2">
          <span>{unisex} item{unisex > 1 ? "s are" : " is"} not marked as menswear or womenswear. Mark {unisex > 1 ? "them all" : "it"} as</span>
          <button disabled={bulkBusy} onClick={() => markAll("menswear")} className="btn-secondary">Menswear</button>
          <button disabled={bulkBusy} onClick={() => markAll("womenswear")} className="btn-secondary">Womenswear</button>
          {bulkBusy && <span className="text-charcoal/60">Saving…</span>}
        </div>
      )}

      {gaps && (
        <div className="rounded-2xl bg-peach/35 p-5 ring-1 ring-peach">
          <p className="text-sm font-semibold">What to buy next — the number is how many new good outfits it would make with what you own</p>
          <p className="mb-4 mt-0.5 text-xs text-charcoal/60">"shop" opens a shopping search for that item; it is not one specific product.</p>
          <div className="grid gap-3 sm:grid-cols-2">
            {gaps.by_category.map((c) => (
              <div key={c.category} className="rounded-xl bg-white p-4 ring-1 ring-warmgrey">
                <p className="text-xs font-semibold uppercase tracking-wide text-tomato">
                  {c.label} <span className="font-normal normal-case text-charcoal/50">(you own {c.owned})</span>
                </p>
                <ol className="mt-2 space-y-1.5 text-sm">
                  {c.picks.map((s, i) => (
                    <li key={i} className="flex items-baseline justify-between gap-3">
                      <span>
                        <b className="capitalize">{s.color} {s.name}</b> <span className="text-charcoal/60">+{s.new_outfits}</span>
                      </span>
                      <a href={s.shop_url} target="_blank" rel="noopener noreferrer" className="link shrink-0 text-xs">shop</a>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
        </div>
      )}

      {selected && (
        <div id="item-panel" className="card scroll-mt-24 p-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="font-semibold capitalize">Goes well with your {selected.colors[0]?.name} {selected.type}</p>
              <div className="mt-2 flex flex-wrap gap-3 text-sm text-charcoal/70">
                <label>
                  Wrong type?{" "}
                  <select value={selected.type} onChange={(e) => update(selected, { type: e.target.value })} className="field ml-1">
                    {TYPES.map((t) => <option key={t}>{t}</option>)}
                  </select>
                </label>
                <label>
                  Wrong color?{" "}
                  <select value={selected.colors[0]?.name} onChange={(e) => update(selected, { color: e.target.value })} className="field ml-1">
                    {COLORS.map((c) => <option key={c}>{c}</option>)}
                  </select>
                </label>
                <label>
                  Made for{" "}
                  <select value={selected.wear} onChange={(e) => update(selected, { wear: e.target.value })} className="field ml-1">
                    {WEAR.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
                  </select>
                </label>
              </div>
            </div>
            <div className="flex gap-4 text-sm">
              <button onClick={() => remove(selected)} className="font-medium text-tomato hover:underline">Delete item</button>
              <button onClick={() => setSelected(null)} className="text-charcoal/60 hover:text-charcoal">Close</button>
            </div>
          </div>
          <ul className="mt-4 grid grid-cols-4 gap-3 sm:grid-cols-6 md:grid-cols-8">
            {matches.map((m) => (
              <li key={m.item_id} title={`${m.item.type} (${m.score.toFixed(2)})`} className="group">
                <div className="overflow-hidden rounded-xl ring-1 ring-warmgrey">
                  <img src={imageSrc(m.item.image_url)} alt={m.item.type} className="photo" />
                </div>
                <p className="mt-1 text-center text-xs text-charcoal/60">{m.score.toFixed(2)}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      <ul className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
        {items.map((it) => (
          <li key={it.id}>
            <button
              onClick={() => pick(it)}
              className={`card card-hover group block w-full text-left ${selected?.id === it.id ? "ring-2 ring-tomato" : ""}`}
            >
              <div className="overflow-hidden">
                <img src={imageSrc(it.image_url)} alt={it.type} className="photo" />
              </div>
              <span className="block truncate px-2.5 py-2 text-xs capitalize text-charcoal/70">{it.colors[0]?.name} {it.type}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
