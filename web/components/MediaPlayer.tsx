"use client";

import { useEffect, useRef } from "react";
import type { Hit } from "@/lib/api";

const VIDEO = [".mp4", ".mov", ".mkv", ".webm", ".m4v"];
const AUDIO = [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"];
const IMAGE = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"];

function ext(path: string) {
  const i = path.lastIndexOf(".");
  return i >= 0 ? path.slice(i).toLowerCase() : "";
}

// Decide the player from the actual file extension (not just the modality), so a
// video asset always renders a <video> even when the matching fragment is a frame.
export function MediaPlayer({ hit }: { hit: Hit }) {
  const ref = useRef<HTMLVideoElement | HTMLAudioElement>(null);
  const e = ext(hit.file_path);
  const start = Math.floor(hit.start_s);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const seek = () => { try { el.currentTime = start; } catch {} };
    el.addEventListener("loadedmetadata", seek);
    seek();
    return () => el.removeEventListener("loadedmetadata", seek);
  }, [hit.media_url, start]);

  return (
    <div className="glass-strong overflow-hidden p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="truncate font-semibold text-ink">{hit.asset_name}</div>
        <span className="font-mono text-sm font-bold text-amber-700">▶ {hit.timestamp_label}</span>
      </div>

      {VIDEO.includes(e) ? (
        <video
          ref={ref as React.RefObject<HTMLVideoElement>}
          src={`${hit.media_url}#t=${start}`}
          controls
          autoPlay
          className="w-full rounded-2xl bg-black"
          style={{ maxHeight: 360 }}
        />
      ) : AUDIO.includes(e) ? (
        <audio ref={ref as React.RefObject<HTMLAudioElement>} src={`${hit.media_url}#t=${start}`}
          controls autoPlay className="w-full" />
      ) : IMAGE.includes(e) ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={hit.media_url} alt={hit.asset_name} className="w-full rounded-2xl" />
      ) : (
        <div className="rounded-2xl bg-white/60 p-6 text-center text-sm text-ink-muted">
          Text asset — no media playback.
        </div>
      )}

      <p className="mt-3 text-sm leading-relaxed text-ink/80">{hit.text.slice(0, 420)}</p>
    </div>
  );
}
