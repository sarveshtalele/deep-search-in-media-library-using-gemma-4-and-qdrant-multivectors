"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, type AskSource, type IngestEvent } from "@/lib/api";

type Turn = {
  role: "user" | "assistant" | "system";
  content: string;
  sources?: AskSource[];
};

type Upload = { name: string; kind?: string; step: string; pct: number; phase: "up" | "ingest" };

const SUGGESTIONS = [
  "Which images did I upload? Describe them.",
  "How many times is “finance” mentioned?",
  "Summarise what my videos are about.",
  "What is said in the audio?",
];

export function Chat({ disabled, onUploaded }: { disabled: boolean; onUploaded: () => void }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [upload, setUpload] = useState<Upload | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [turns, busy, upload]);

  const ask = async (q: string) => {
    const question = q.trim();
    if (!question || busy || disabled) return;
    const history = turns
      .filter((t) => t.role !== "system")
      .map((t) => ({ role: t.role, content: t.content }));
    // Push the user turn + an empty assistant turn we stream into.
    setTurns((t) => [...t, { role: "user", content: question }, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    const patchLast = (fn: (t: Turn) => Turn) =>
      setTurns((ts) => { const n = [...ts]; n[n.length - 1] = fn(n[n.length - 1]); return n; });
    try {
      await api.askStream(
        { question, history },
        (tok) => patchLast((t) => ({ ...t, content: t.content + tok })),
        (src) => patchLast((t) => ({ ...t, sources: src })),
      );
    } catch (e) {
      patchLast((t) => ({ ...t, content: (t.content || "") + `\n\n⚠️ ${(e as Error).message}` }));
    } finally { setBusy(false); }
  };

  const handleFile = async (file: File) => {
    if (disabled) return;
    setUpload({ name: file.name, step: "Uploading…", pct: 0, phase: "up" });
    try {
      const up = await api.upload(file);
      if (!up.supported) {
        setTurns((t) => [...t, { role: "system", content: `⚠️ ${file.name}: unsupported file type.` }]);
        setUpload(null);
        return;
      }
      setUpload({ name: up.filename, kind: up.kind, step: "Indexing…", pct: 0, phase: "ingest" });
      await api.ingest(up.filename, "chat", (ev: IngestEvent) => {
        if (ev.stage === "result") {
          const by = ev.by_modality || {};
          const parts = Object.entries(by).map(([k, v]) => `${k.split("_")[0]} ${v}`).join(", ");
          setTurns((t) => [...t, {
            role: "system",
            content: ev.error
              ? `⚠️ ${up.filename}: ${ev.error}`
              : `✅ Added **${up.filename}** to your library (${parts}). Ask me anything about it.`,
          }]);
          setUpload(null);
          onUploaded();
        } else if (ev.stage === "error") {
          setTurns((t) => [...t, { role: "system", content: `⚠️ ${up.filename}: ${ev.message}` }]);
          setUpload(null);
        } else {
          setUpload((u) => u && {
            ...u, step: ev.message || u.step,
            pct: ev.total ? Math.min(1, (ev.current || 0) / ev.total) : u.pct,
          });
        }
      });
    } catch (e) {
      setTurns((t) => [...t, { role: "system", content: `⚠️ Upload failed: ${(e as Error).message}` }]);
      setUpload(null);
    }
  };

  return (
    <div className="glass-strong flex h-[calc(100vh-3rem)] flex-col p-5 sm:p-6">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-lg font-bold">💬 Ask your library</span>
        <span className="text-xs text-ink-muted">grounded in your indexed media</span>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto pr-1">
        {turns.length === 0 && !upload && (
          <div className="animate-fade-up">
            <p className="text-sm text-ink-muted">
              Attach media with <b>📎</b> (it’s saved to your library and indexed automatically),
              then ask anything — answers are grounded in your media, with the exact moments as sources.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="chip hover:bg-amber/20" disabled={disabled} onClick={() => ask(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}

        {turns.map((t, i) => <Bubble key={i} turn={t} />)}

        {upload && (
          <div className="animate-fade-up">
            <div className="glass max-w-[92%] rounded-2xl px-4 py-3 text-sm">
              <div className="flex items-center gap-2 font-semibold text-ink">
                <Spinner /> {upload.phase === "up" ? "Uploading" : "Indexing"} {upload.name}
              </div>
              <div className="mt-1 text-xs text-ink-muted">{upload.step}</div>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-ink/10">
                <div className="h-full rounded-full bg-clay transition-all"
                  style={{ width: `${Math.round(upload.pct * 100)}%` }} />
              </div>
            </div>
          </div>
        )}

        <div ref={endRef} />
      </div>

      {/* Composer with attach */}
      <div className="mt-4 flex items-center gap-2">
        <button
          className="btn-ghost h-11 w-11 flex-none rounded-full p-0 text-lg"
          title="Attach media (auto-saved to library)"
          disabled={disabled || !!upload}
          onClick={() => fileRef.current?.click()}
        >📎</button>
        <input ref={fileRef} type="file" className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.currentTarget.value = ""; }} />
        <input
          className="field"
          value={input}
          disabled={disabled || busy}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask(input)}
          placeholder={disabled ? "Backend not ready — check status in the sidebar" : "Ask about your media…"}
        />
        <button className="btn-primary" onClick={() => ask(input)} disabled={disabled || busy || !input.trim()}>Ask</button>
      </div>
    </div>
  );
}

function Bubble({ turn }: { turn: Turn }) {
  if (turn.role === "user")
    return (
      <div className="flex animate-fade-up justify-end">
        <div className="max-w-[85%] rounded-2xl bg-clay px-4 py-2.5 text-sm text-white">{turn.content}</div>
      </div>
    );
  if (turn.role === "system")
    return (
      <div className="animate-fade-up text-center">
        <span className="inline-block rounded-full bg-amber-soft px-3 py-1 text-xs text-clay-700">
          {turn.content.replace(/\*\*/g, "")}
        </span>
      </div>
    );
  return (
    <div className="animate-fade-up max-w-[92%]">
      <div className="glass rounded-2xl px-4 py-3 text-sm text-ink">
        {turn.content ? <Markdown>{turn.content}</Markdown> : <Thinking />}
      </div>
      {turn.sources && turn.sources.length > 0 && <Sources sources={turn.sources} />}
    </div>
  );
}

// Product-grade markdown rendering (bold, lists, headings, code, links).
function Markdown({ children }: { children: string }) {
  return (
    <div className="chat-md">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}

function Thinking() {
  return (
    <span className="inline-flex items-center gap-1 text-ink-muted">
      <span className="h-2 w-2 animate-bounce rounded-full bg-clay/60 [animation-delay:-0.3s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-clay/60 [animation-delay:-0.15s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-clay/60" />
    </span>
  );
}

function Sources({ sources }: { sources: AskSource[] }) {
  const [open, setOpen] = useState(false);
  const [play, setPlay] = useState<AskSource | null>(null);
  return (
    <div className="mt-2">
      <button onClick={() => setOpen((o) => !o)} className="text-xs font-semibold text-clay-600 hover:underline">
        {open ? "▾ Hide sources" : `▸ ${sources.length} source${sources.length > 1 ? "s" : ""}`}
      </button>
      {open && (
        <div className="mt-2 flex flex-wrap gap-2">
          {sources.map((s, i) => (
            <button key={i} onClick={() => setPlay(s)}
              className="rounded-full border border-line-strong bg-white/70 px-3 py-1 text-xs text-ink hover:border-clay hover:text-clay-600">
              {s.modality === "audio_chunks" ? "🎧" : s.modality === "text_descriptions" ? "📝" : "🎬"}{" "}
              {s.asset_name} <span className="font-mono text-amber-700">{s.timestamp_label}</span>
            </button>
          ))}
        </div>
      )}
      {open && play && <SourcePlayer source={play} />}
    </div>
  );
}

function SourcePlayer({ source }: { source: AskSource }) {
  const ref = useRef<HTMLVideoElement | HTMLAudioElement>(null);
  const start = Math.floor(source.start_s);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const seek = () => { try { el.currentTime = start; } catch {} };
    el.addEventListener("loadedmetadata", seek); seek();
    return () => el.removeEventListener("loadedmetadata", seek);
  }, [source.media_url, start]);
  return (
    <div className="glass mt-3 p-3">
      <div className="mb-2 text-xs font-semibold text-ink">
        {source.asset_name} · <span className="font-mono text-amber-700">{source.timestamp_label}</span>
      </div>
      {source.modality === "audio_chunks" ? (
        <audio ref={ref as React.RefObject<HTMLAudioElement>} src={`${source.media_url}#t=${start}`} controls autoPlay className="w-full" />
      ) : (
        <video ref={ref as React.RefObject<HTMLVideoElement>} src={`${source.media_url}#t=${start}`} controls autoPlay
          className="w-full rounded-xl bg-black" style={{ maxHeight: 300 }} />
      )}
    </div>
  );
}

function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-clay/30 border-t-clay" />;
}
