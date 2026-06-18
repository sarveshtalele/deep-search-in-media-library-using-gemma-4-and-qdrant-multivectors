"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type Asset, type Health, type Stats } from "@/lib/api";
import type { View } from "@/app/page";

const MOD_ICON: Record<string, string> = {
  video_frames: "🎬", audio_chunks: "🎧", text_descriptions: "📝",
};

function assetIcon(a: Asset) {
  if (a.modalities.includes("video_frames")) return "🎬";
  if (a.modalities.includes("audio_chunks")) return "🎧";
  return "📝";
}

export function Sidebar({
  view, setView, assets, stats, onHealth, onDelete, onRefresh,
}: {
  view: View;
  setView: (v: View) => void;
  assets: Asset[];
  stats: Stats | null;
  onHealth: (ok: boolean) => void;
  onDelete: (id: string) => void;
  onRefresh: () => void;
}) {
  const [health, setHealth] = useState<Health | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);

  const check = useCallback(async () => {
    try { const h = await api.health(); setHealth(h); onHealth(h.ok); }
    catch { setHealth(null); onHealth(false); }
  }, [onHealth]);

  useEffect(() => { check(); const t = setInterval(check, 20000); return () => clearInterval(t); }, [check]);

  const ok = !!health?.ok;

  return (
    <aside className="sticky top-0 flex h-screen w-72 flex-none flex-col gap-4 border-r border-line
      bg-cream-deep/60 p-4 backdrop-blur-glass">
      {/* Brand */}
      <div>
        <div className="text-lg font-extrabold tracking-tight text-ink">
          Deep Search <span className="text-clay">·</span>
        </div>
        <div className="text-xs text-ink-muted">Gemma 4 + Qdrant · 100% local</div>
      </div>

      {/* Health */}
      <button onClick={check}
        className="glass flex items-center gap-2 px-3 py-2 text-left text-xs">
        <span className={`h-2.5 w-2.5 rounded-full ${ok ? "bg-emerald-500" : "bg-clay"}`} />
        <span className="min-w-0 flex-1 truncate font-semibold text-ink">
          {ok ? "Gemma 4 ready" : "Gemma 4 unavailable"}
        </span>
        <span className="text-ink-muted">↻</span>
      </button>
      {!ok && health?.detail && (
        <p className="-mt-2 text-[11px] leading-snug text-clay-700">{health.detail}</p>
      )}

      {/* Nav */}
      <nav className="flex flex-col gap-1">
        {([["chat", "💬", "Chat"], ["search", "🔎", "Search"], ["library", "📚", "Library"]] as
          [View, string, string][]).map(([id, icon, label]) => (
          <button key={id} onClick={() => setView(id)}
            className={`flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-semibold transition ${
              view === id ? "bg-clay text-white shadow-glass" : "text-ink-muted hover:bg-white/60 hover:text-ink"
            }`}>
            <span>{icon}</span>{label}
          </button>
        ))}
      </nav>

      {/* Library list with delete */}
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="mb-2 flex items-center justify-between text-xs font-semibold text-ink-muted">
          <span>LIBRARY · {assets.length}</span>
          <button onClick={onRefresh} className="hover:text-clay-600">↻</button>
        </div>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
          {assets.length === 0 && (
            <p className="text-xs text-ink-muted">No media yet. Attach a file in chat (📎).</p>
          )}
          {assets.map((a) => (
            <div key={a.asset_id}
              className="group flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-white/60">
              <span>{assetIcon(a)}</span>
              <span className="min-w-0 flex-1 truncate text-ink" title={a.asset_name}>{a.asset_name}</span>
              {a.has_audio && <span className="text-[10px]" title="has audio">🎧</span>}
              {confirmId === a.asset_id ? (
                <span className="flex gap-1">
                  <button onClick={() => { onDelete(a.asset_id); setConfirmId(null); }}
                    className="text-xs font-bold text-clay">Yes</button>
                  <button onClick={() => setConfirmId(null)} className="text-xs text-ink-muted">No</button>
                </span>
              ) : (
                <button onClick={() => setConfirmId(a.asset_id)}
                  className="text-ink-muted opacity-0 transition group-hover:opacity-100 hover:text-clay"
                  title="Delete">🗑</button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="border-t border-line pt-3 text-xs text-ink-muted">
        {stats ? (
          <>
            {stats.assets} assets · {stats.fragments} fragments
            <div className="mt-1 flex gap-2">
              {Object.entries(stats.by_modality).map(([k, v]) => (
                <span key={k}>{MOD_ICON[k] ?? k} {v}</span>
              ))}
            </div>
          </>
        ) : "index idle"}
      </div>
    </aside>
  );
}
