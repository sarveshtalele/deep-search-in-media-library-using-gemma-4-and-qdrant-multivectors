"use client";

import type { Hit } from "@/lib/api";
import { ModalityIcon } from "./icons";

const MOD_LABEL: Record<string, string> = {
  video_frames: "video", audio_chunks: "audio", text_descriptions: "text",
};

export function ResultCard({
  hit, index, active, onSelect,
}: {
  hit: Hit; index: number; active: boolean; onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={`glass animate-fade-up w-full text-left transition hover:shadow-lift ${
        active ? "ring-2 ring-clay" : ""
      }`}
    >
      <div className="flex gap-3 p-3">
        {hit.thumbnail_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={hit.thumbnail_url} alt="" className="h-16 w-24 flex-none rounded-xl object-cover" />
        ) : (
          <div className="flex h-16 w-24 flex-none items-center justify-center rounded-xl bg-amber-soft text-clay-600">
            <ModalityIcon modality={hit.modality} size={24} />
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-bold text-clay">#{index + 1}</span>
            <span className="font-semibold text-clay-600">{hit.score.toFixed(3)}</span>
            <span className="truncate font-semibold text-ink">{hit.asset_name}</span>
            <span className="ml-auto font-mono text-xs font-bold text-amber-700">{hit.timestamp_label}</span>
          </div>
          <div className="mt-0.5 text-[10px] font-semibold uppercase tracking-wide text-ink-muted">
            {MOD_LABEL[hit.modality] ?? hit.modality}
          </div>
          <p className="mt-1 line-clamp-2 text-sm text-ink/75">{hit.text.slice(0, 180)}</p>
        </div>
      </div>
    </button>
  );
}
