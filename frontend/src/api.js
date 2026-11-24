// Empty string = the API is on the same address as the website (the deployed single-URL setup).
export const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// Simple identity: a UUID generated on first visit, sent as X-User-Id. There is no login.
export function userId() {
  let id = localStorage.getItem("outfit-user-id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("outfit-user-id", id);
  }
  return id;
}

export const imageSrc = (path) => `${API_URL}${path}`;

export const TYPES = ["t-shirt", "shirt", "hoodie", "sweater", "jacket", "coat", "track jacket", "jeans", "trousers",
  "track pants", "shorts", "skirt", "dress", "sneakers", "formal shoes", "sandals", "boots"];

export const COLORS = ["black", "white", "grey", "beige", "brown", "red", "maroon", "orange", "yellow", "green", "teal",
  "blue", "light blue", "navy", "purple", "pink"];

// Who an item is cut for; asked on upload, editable on the Wardrobe page.
export const WEAR = [
  ["menswear", "Menswear"],
  ["womenswear", "Womenswear"],
  ["unisex", "Unisex"],
];

export async function api(path, { method = "GET", json, form } = {}) {
  const headers = { "X-User-Id": userId() };
  let body;
  if (json) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form) {
    body = form;
  }
  const res = await fetch(`${API_URL}${path}`, { method, headers, body });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const d = await res.json();
      detail = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}
