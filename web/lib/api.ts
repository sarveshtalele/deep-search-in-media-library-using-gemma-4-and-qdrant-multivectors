// Typed client for the FastAPI backend. All paths are same-origin (/api/*)
// thanks to the Next.js rewrite to http://localhost:8000.

export type Health = {
  ok: boolean;
  backend: string;
  stub_mode: boolean;
  ollama_host: string;
  gemma_model: string;
  embed_model: string;
  ollama_reachable: boolean;
  gemma_present: boolean;
  embed_present: boolean;
  models: string[];
  detail: string;
};

export type UploadResp = {
  filename: string;
  kind: "video" | "audio" | "image" | "text" | "unknown";
  size_bytes: number;
  supported: boolean;
};

export type Hit = {
  point_id: string;
  asset_id: string;
  asset_name: string;
  file_path: string;
  media_url: string;
  thumbnail_url: string | null;
  modality: string;
  text: string;
  start_s: number;
  end_s: number;
  timestamp_label: string;
  score: number;
  contributions: Record<string, number>;
};

export type SearchResp = {
  query: string;
  intent: string;
  weights: Record<string, number>;
  fusion: string;
  blocked: boolean;
  message: string;
  hits: Hit[];
};

export type Stats = { assets: number; fragments: number; by_modality: Record<string, number> };

export type Asset = {
  asset_id: string;
  asset_name: string;
  file_path: string;
  media_url: string;
  modalities: string[];
  has_audio: boolean;
};

export type AskSource = {
  asset_id: string;
  asset_name: string;
  media_url: string;
  modality: string;
  start_s: number;
  timestamp_label: string;
};

export type AskResp = { answer: string; sources: AskSource[] };

export type IngestEvent = {
  stage: string;
  message?: string;
  current?: number;
  total?: number;
  // final "result" event:
  error?: string | null;
  fragments?: number;
  by_modality?: Record<string, number>;
  asset_id?: string;
  asset_name?: string;
};

async function jsonOrThrow(res: Response) {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: (): Promise<Health> => fetch("/api/health").then(jsonOrThrow),

  stats: (): Promise<Stats> => fetch("/api/stats").then(jsonOrThrow),

  assets: (): Promise<Asset[]> => fetch("/api/assets").then(jsonOrThrow),

  deleteAsset: (id: string): Promise<{ ok: boolean }> =>
    fetch(`/api/asset/${encodeURIComponent(id)}`, { method: "DELETE" }).then(jsonOrThrow),

  upload: (file: File): Promise<UploadResp> => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/upload", { method: "POST", body: fd }).then(jsonOrThrow);
  },

  search: (body: {
    query: string;
    modalities?: string[] | null;
    categories?: string[] | null;
    weights?: Record<string, number> | null;
    asset_recall?: boolean;
  }): Promise<SearchResp> =>
    fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(jsonOrThrow),

  ask: (body: {
    question: string;
    history: { role: string; content: string }[];
  }): Promise<AskResp> =>
    fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(jsonOrThrow),

  // Streams the answer token-by-token (SSE), then the sources.
  askStream: async (
    body: { question: string; history: { role: string; content: string }[] },
    onToken: (t: string) => void,
    onSources: (s: AskSource[]) => void,
  ): Promise<void> => {
    const res = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      throw new Error((await res.json().catch(() => ({})))?.detail || `Ask failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() || "";
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "token") onToken(ev.data as string);
        else if (ev.type === "sources") onSources(ev.data as AskSource[]);
        else if (ev.type === "error") throw new Error(ev.data as string);
      }
    }
  },

  // Streams Server-Sent-Events from POST /api/ingest. Calls onEvent per event.
  ingest: async (
    filename: string,
    category: string,
    onEvent: (ev: IngestEvent) => void,
  ): Promise<void> => {
    const res = await fetch("/api/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, category }),
    });
    if (!res.ok || !res.body) {
      throw new Error((await res.json().catch(() => ({})))?.detail || `Ingest failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() || "";
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (line) onEvent(JSON.parse(line.slice(6)));
      }
    }
  },
};
