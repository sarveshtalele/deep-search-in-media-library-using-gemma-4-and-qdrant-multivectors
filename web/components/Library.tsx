"use client";

import { useState } from "react";
import type { Asset, Stats } from "@/lib/api";
import { IconAudio, IconLibrary, IconRefresh, IconTrash } from "./icons";

const VIDEO = [".mp4", ".mov", ".mkv", ".webm", ".m4v"];
const AUDIO = [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"];
const IMAGE = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"];
const ext = (p: string) => { const i = p.lastIndexOf("."); return i >= 0 ? p.slice(i).toLowerCase() : ""; };

export function Library({
  assets, stats, onDelete, onRefresh,
}: {
  assets: Asset[]; stats: Stats | null; onDelete: (id: string) => void; onRefresh: () => void;
}) {
  const [confirm, setConfirm] = useState<string | null>(null);
  const max = stats ? Math.max(1, ...Object.values(stats.by_modality)) : 1;

  return (
    <div className="space-y-5">
      <div className="glass-strong p-5">
        <div className="flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-lg font-bold"><IconLibrary size={18} /> Library</h2>
          <button className="btn-ghost inline-flex items-center gap-1 px-3 py-1 text-xs" onClick={onRefresh}>
            <IconRefresh size={13} /> Refresh
          </button>
        </div>
        {stats && (
          <>
            <div className="mt-4 grid grid-cols-3 gap-3">
              <Stat label="Assets" value={stats.assets} />
              <Stat label="Fragments" value={stats.fragments} />
              <Stat label="Modalities" value={Object.keys(stats.by_modality).length} />
            </div>
            <div className="mt-4 space-y-2">
              {Object.entries(stats.by_modality).map(([k, v]) => (
                <div key={k} className="flex items-center gap-3 text-sm">
                  <span className="w-24 text-ink-muted">{k.split("_")[0]}</span>
                  <div className="h-3 flex-1 overflow-hidden rounded-full bg-ink/10">
                    <div className="h-full rounded-full bg-clay" style={{ width: `${(v / max) * 100}%` }} />
                  </div>
                  <span className="w-10 text-right font-mono">{v}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {assets.length === 0 ? (
        <div className="glass p-8 text-center text-sm text-ink-muted">
          No media yet. Go to <b className="text-clay-600">Chat</b> and attach a file.
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          {assets.map((a) => {
            const e = ext(a.file_path);
            return (
              <div key={a.asset_id} className="glass overflow-hidden p-3">
                <div className="mb-2 flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate font-semibold text-ink">{a.asset_name}</span>
                  {a.has_audio && <IconAudio size={14} className="flex-none text-ink-muted" />}
                  {confirm === a.asset_id ? (
                    <span className="flex gap-1 text-xs">
                      <button className="font-bold text-clay" onClick={() => { onDelete(a.asset_id); setConfirm(null); }}>Delete</button>
                      <button className="text-ink-muted" onClick={() => setConfirm(null)}>Cancel</button>
                    </span>
                  ) : (
                    <button className="text-ink-muted hover:text-clay" title="Delete" onClick={() => setConfirm(a.asset_id)}><IconTrash size={15} /></button>
                  )}
                </div>
                {VIDEO.includes(e) ? (
                  <video src={a.media_url} controls className="w-full rounded-xl bg-black" style={{ maxHeight: 200 }} />
                ) : AUDIO.includes(e) ? (
                  <audio src={a.media_url} controls className="w-full" />
                ) : IMAGE.includes(e) ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={a.media_url} alt={a.asset_name} className="w-full rounded-xl" />
                ) : (
                  <div className="rounded-xl bg-white/60 p-4 text-center text-xs text-ink-muted">text asset</div>
                )}
                <div className="mt-2 flex flex-wrap gap-1">
                  {a.modalities.map((m) => (
                    <span key={m} className="rounded-full bg-amber-soft px-2 py-0.5 text-[10px] text-clay-600">
                      {m.split("_")[0]}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="glass p-3 text-center">
      <div className="text-2xl font-extrabold text-clay-600">{value}</div>
      <div className="text-xs text-ink-muted">{label}</div>
    </div>
  );
}
