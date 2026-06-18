"use client";

import { useState } from "react";
import { api, type Hit, type SearchResp } from "@/lib/api";
import { ResultCard } from "./ResultCard";
import { MediaPlayer } from "./MediaPlayer";

const MODALITIES = [
  { id: "video_frames", label: "Video" },
  { id: "audio_chunks", label: "Audio" },
  { id: "text_descriptions", label: "Text" },
];

export function Search({ disabled }: { disabled: boolean }) {
  const [query, setQuery] = useState("");
  const [auto, setAuto] = useState(true);
  const [assetRecall, setAssetRecall] = useState(false);
  const [mods, setMods] = useState<string[]>([]);
  const [weights, setWeights] = useState({ text_descriptions: 0.5, video_frames: 0.3, audio_chunks: 0.2 });
  const [showFilters, setShowFilters] = useState(false);
  const [resp, setResp] = useState<SearchResp | null>(null);
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    if (!query.trim() || busy) return;
    setBusy(true); setError(""); setResp(null);
    try {
      const r = await api.search({
        query,
        modalities: mods.length ? mods : null,
        weights: auto ? null : weights,
        asset_recall: assetRecall,
      });
      setResp(r); setSelected(0);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const toggleMod = (id: string) =>
    setMods((m) => (m.includes(id) ? m.filter((x) => x !== id) : [...m, id]));

  const hits: Hit[] = resp?.hits ?? [];
  const sel = hits[Math.min(selected, hits.length - 1)];

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
      {/* Left: query + results */}
      <div>
        <div className="glass-strong p-5">
          <div className="flex gap-2">
            <input
              className="field"
              value={query}
              disabled={disabled}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run()}
              placeholder="Find the chart showing the budget increase right when the crowd cheers…"
            />
            <button className="btn-primary" onClick={run} disabled={disabled || busy || !query.trim()}>
              {busy ? "Searching…" : "Search"}
            </button>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-4 text-sm">
            <Toggle label="Auto weights" checked={auto} onChange={setAuto} />
            <Toggle label="MaxSim recall" checked={assetRecall} onChange={setAssetRecall} />
            <button className="ml-auto text-clay-600 hover:underline" onClick={() => setShowFilters((s) => !s)}>
              {showFilters ? "Hide filters" : "Filters & weights"}
            </button>
          </div>

          {showFilters && (
            <div className="mt-4 animate-fade-up space-y-4 border-t border-line pt-4">
              <div className="flex flex-wrap gap-2">
                {MODALITIES.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => toggleMod(m.id)}
                    className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
                      mods.includes(m.id)
                        ? "border-clay bg-clay text-white"
                        : "border-line-strong bg-white/60 text-ink-muted hover:border-clay"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
              {!auto && (
                <div className="space-y-2">
                  {(Object.keys(weights) as (keyof typeof weights)[]).map((k) => (
                    <label key={k} className="flex items-center gap-3 text-xs text-ink-muted">
                      <span className="w-32">{k.split("_")[0]}</span>
                      <input
                        type="range" min={0} max={1} step={0.05} value={weights[k]}
                        onChange={(e) => setWeights((w) => ({ ...w, [k]: +e.target.value }))}
                        className="flex-1 accent-clay"
                      />
                      <span className="w-8 font-mono">{weights[k].toFixed(2)}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          )}

          {resp && !resp.blocked && (
            <p className="mt-3 text-xs text-ink-muted">
              intent <b className="text-clay-600">{resp.intent}</b> · fusion{" "}
              <b className="text-clay-600">{resp.fusion}</b>
            </p>
          )}
        </div>

        {error && (
          <div className="mt-4 rounded-2xl border border-clay/40 bg-clay/5 p-4 text-sm text-clay-700">
            {error}
          </div>
        )}
        {resp?.blocked && (
          <div className="mt-4 rounded-2xl border border-clay/40 bg-clay/5 p-4 text-sm text-clay-700">
            🛑 Query blocked: {resp.message}
          </div>
        )}

        <div className="mt-4 space-y-3">
          {busy && <Skeleton />}
          {!busy && resp && !resp.blocked && hits.length === 0 && (
            <p className="text-sm text-ink-muted">No matches. Attach media in Chat (📎) or rephrase.</p>
          )}
          {hits.map((h, i) => (
            <ResultCard key={h.point_id} hit={h} index={i} active={i === selected}
              onSelect={() => setSelected(i)} />
          ))}
        </div>
      </div>

      {/* Right: player (revealed when a result is selected) */}
      <div className="space-y-4 lg:sticky lg:top-4 lg:self-start">
        {sel ? (
          <MediaPlayer hit={sel} />
        ) : (
          <div className="glass flex h-64 items-center justify-center px-6 text-center text-sm text-ink-muted">
            Select a result to play it at the exact moment. For answers &amp; counting, use the
            <b className="mx-1 text-clay-600">Chat</b> tab.
          </div>
        )}
      </div>
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button onClick={() => onChange(!checked)} className="flex items-center gap-2">
      <span className={`relative h-5 w-9 rounded-full transition ${checked ? "bg-clay" : "bg-ink/20"}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${checked ? "left-4" : "left-0.5"}`} />
      </span>
      <span className="text-ink">{label}</span>
    </button>
  );
}

function Skeleton() {
  return (
    <div className="space-y-3">
      {[0, 1, 2].map((i) => (
        <div key={i} className="glass relative h-20 overflow-hidden">
          <div className="absolute inset-0 -translate-x-full animate-shimmer
            bg-gradient-to-r from-transparent via-white/60 to-transparent" />
        </div>
      ))}
    </div>
  );
}
