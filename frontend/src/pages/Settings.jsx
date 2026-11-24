import { useState } from "react";
import { api } from "../api.js";

export default function Settings() {
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");

  async function wipe() {
    if (!confirm("Delete all your photos, tags and feedback? This cannot be undone.")) return;
    try {
      await api("/users/me/data", { method: "DELETE" });
      localStorage.removeItem("outfit-user-id");
      setDone(true);
    } catch (e) { setError(e.message); }
  }

  return (
    <section className="card max-w-xl space-y-4 p-6">
      <h2 className="text-lg font-semibold">Privacy</h2>
      <p className="text-sm text-charcoal/70">Your photos are personal. Everything is tied to a random ID stored in this browser — no email or password. You can erase it all at any time.</p>
      {done ? <p className="note">All your data was deleted. A fresh ID is created next time you visit.</p>
        : <button onClick={wipe} className="btn-primary">Delete my data</button>}
      {error && <p className="error">{error}</p>}
    </section>
  );
}
