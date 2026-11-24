import { useEffect, useState } from "react";
import { api, imageSrc, TYPES, WEAR } from "../api.js";

const WEAR_HINT = {
  menswear: "Never tagged as a dress or skirt",
  womenswear: "Can be tagged as any type",
  unisex: "Fits into any outfit",
};

export default function Upload({ goTo }) {
  const [type, setType] = useState("");
  const [busy, setBusy] = useState(false);
  const [added, setAdded] = useState([]);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(null); // files waiting for the menswear/womenswear answer

  useEffect(() => {
    if (!pending) return;
    const onKey = (e) => e.key === "Escape" && setPending(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pending]);

  async function upload(files, wear) {
    setPending(null);
    setBusy(true);
    setError("");
    for (const file of files) {
      const form = new FormData();
      form.append("file", file);
      if (type) form.append("type", type);
      form.append("wear", wear);
      try {
        const item = await api("/wardrobe/items", { method: "POST", form });
        setAdded((a) => [item, ...a]);
      } catch (e) {
        setError(`${file.name}: ${e.message}`);
      }
    }
    setBusy(false);
  }

  return (
    <section className="space-y-6">
      <div className="note">
        <b>Tip:</b> lay each item flat on a plain background (a sheet or a table), in daylight, shot from directly above — one
        item per photo. Messy backgrounds make tagging less accurate.
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-sm text-charcoal/70">
          Item type{" "}
          <select value={type} onChange={(e) => setType(e.target.value)} className="field ml-1">
            <option value="">Auto-detect</option>
            {TYPES.map((t) => <option key={t}>{t}</option>)}
          </select>
        </label>
        <label className={`btn-primary cursor-pointer ${busy ? "pointer-events-none opacity-60" : ""}`}>
          {busy ? "Tagging…" : "Choose photos"}
          <input
            type="file"
            accept="image/*"
            multiple
            hidden
            disabled={busy}
            onChange={(e) => {
              const files = [...e.target.files];
              e.target.value = "";
              if (files.length) setPending(files);
            }}
          />
        </label>
        {added.length > 0 && (
          <button onClick={() => goTo("wardrobe")} className="link text-sm">
            View wardrobe →
          </button>
        )}
      </div>
      {error && <p className="error">{error}</p>}
      <ul className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
        {added.map((it) => (
          <li key={it.id} className="card card-hover group">
            <div className="overflow-hidden">
              <img src={imageSrc(it.image_url)} alt={it.type} className="photo" />
            </div>
            <div className="p-3 text-sm">
              <div className="flex items-center justify-between gap-2">
                <p className="font-semibold capitalize">{it.type}</p>
                <span className="badge">{it.wear}</span>
              </div>
              <p className="text-charcoal/60">
                {it.colors.map((c) => c.name).join(", ")} · {it.pattern} · {it.material}
              </p>
              {it.type_confidence < 0.5 && <p className="mt-1 text-xs font-medium text-tomato">Low confidence — check the type.</p>}
            </div>
          </li>
        ))}
      </ul>

      {pending && (
        <div className="fixed inset-0 z-30 flex items-center justify-center bg-charcoal/40 p-4" onClick={() => setPending(null)}>
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="wear-title"
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-md rounded-2xl bg-cream p-6 ring-1 ring-warmgrey"
          >
            <p id="wear-title" className="text-lg font-semibold">Who is this for?</p>
            <p className="mt-1 text-sm text-charcoal/70">
              {pending.length === 1 ? "1 photo" : `${pending.length} photos`} — the answer applies to all of them. Outfits never mix menswear
              with womenswear.
            </p>
            <div className="mt-5 grid gap-2">
              {WEAR.map(([value, label], i) => (
                <button
                  key={value}
                  autoFocus={i === 0}
                  onClick={() => upload(pending, value)}
                  className="group flex items-center justify-between rounded-xl bg-white px-4 py-3 text-left ring-1 ring-warmgrey transition duration-200 hover:ring-2 hover:ring-tomato focus:ring-2 focus:ring-tomato focus:outline-none"
                >
                  <span>
                    <span className="block font-semibold">{label}</span>
                    <span className="block text-xs text-charcoal/60">{WEAR_HINT[value]}</span>
                  </span>
                  <span className="text-tomato opacity-0 transition group-hover:opacity-100">→</span>
                </button>
              ))}
            </div>
            <button onClick={() => setPending(null)} className="mt-4 w-full text-sm text-charcoal/60 hover:text-charcoal">
              Cancel
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
