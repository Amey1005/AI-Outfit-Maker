import { useState } from "react";
import Upload from "./pages/Upload.jsx";
import Wardrobe from "./pages/Wardrobe.jsx";
import Suggestions from "./pages/Suggestions.jsx";
import Settings from "./pages/Settings.jsx";

const TABS = [
  ["wardrobe", "Wardrobe", Wardrobe],
  ["upload", "Add clothes", Upload],
  ["suggest", "Outfits", Suggestions],
  ["settings", "Settings", Settings],
];

export default function App() {
  const [tab, setTab] = useState("wardrobe");
  const Page = TABS.find(([k]) => k === tab)[2];
  return (
    <>
      <header className="sticky top-0 z-20 border-b border-warmgrey bg-cream/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight text-charcoal">
            <span aria-hidden="true" className="inline-block h-3 w-3 rounded-full bg-tomato" />
            Outfit Maker
          </h1>
          <nav className="flex w-full gap-1 rounded-full bg-white p-1 ring-1 ring-warmgrey sm:w-auto">
            {TABS.map(([k, label]) => (
              <button
                key={k}
                onClick={() => setTab(k)}
                aria-current={tab === k ? "page" : undefined}
                className={`flex-1 whitespace-nowrap rounded-full px-2 py-1.5 text-sm font-medium transition duration-200 sm:flex-none sm:px-4 ${
                  tab === k ? "bg-tomato text-white" : "text-charcoal/70 hover:bg-peach/40 hover:text-charcoal"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 pb-16 pt-6">
        <Page goTo={setTab} />
      </main>
    </>
  );
}
