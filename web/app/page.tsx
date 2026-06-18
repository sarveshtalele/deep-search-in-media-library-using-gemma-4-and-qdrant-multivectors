"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type Asset, type Stats } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar";
import { Chat } from "@/components/Chat";
import { Search } from "@/components/Search";
import { Library } from "@/components/Library";

export type View = "chat" | "search" | "library";

export default function Home() {
  const [view, setView] = useState<View>("chat");
  const [healthOk, setHealthOk] = useState(false);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);

  const refresh = useCallback(async () => {
    try { setStats(await api.stats()); } catch { setStats(null); }
    try { setAssets(await api.assets()); } catch { setAssets([]); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const onDelete = useCallback(async (id: string) => {
    try { await api.deleteAsset(id); } catch {}
    refresh();
  }, [refresh]);

  return (
    <div className="flex min-h-screen">
      <Sidebar
        view={view} setView={setView}
        assets={assets} stats={stats}
        onHealth={setHealthOk} onDelete={onDelete} onRefresh={refresh}
      />
      <main className="flex-1 px-4 py-6 sm:px-8">
        <div className="mx-auto max-w-4xl">
          {view === "chat" && <Chat disabled={!healthOk} onUploaded={refresh} />}
          {view === "search" && <Search disabled={!healthOk} />}
          {view === "library" && (
            <Library assets={assets} stats={stats} onDelete={onDelete} onRefresh={refresh} />
          )}
        </div>
      </main>
    </div>
  );
}
