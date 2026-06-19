"use client";

import { useState, useEffect } from "react";
import type { RankingEntry } from "@/lib/api";
import { getRankings, REGION_COLORS } from "@/lib/api";
import { TEAM_LOGOS } from "@/lib/teamLogos";
import SectionHeading from "@/components/SectionHeading";

const REGIONS = ["All", "Americas", "EMEA", "Pacific", "China"];

function RankBadge({ rank }: { rank: number }) {
  const colors: Record<number, string> = { 1: "#C9AA71", 2: "#B0B8C1", 3: "#CD7F32" };
  const color = colors[rank] ?? "#6B7E8F";
  return (
    <span className="font-display text-base font-extrabold w-6 text-right" style={{ color }}>
      {rank}
    </span>
  );
}

function EloChange({ change }: { change: number }) {
  if (Math.abs(change) < 0.5) {
    return <span className="font-sans text-xs text-vmuted">—</span>;
  }
  const positive = change > 0;
  return (
    <span
      className="font-display text-xs font-bold"
      style={{ color: positive ? "#00C87A" : "#FF4655" }}
    >
      {positive ? "+" : ""}{Math.round(change)}
    </span>
  );
}

export default function EloRankings() {
  const [rankings, setRankings] = useState<RankingEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [regionFilter, setRegionFilter] = useState("All");

  useEffect(() => {
    getRankings()
      .then(setRankings)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load rankings"))
      .finally(() => setLoading(false));
  }, []);

  const filtered =
    regionFilter === "All" ? rankings : rankings.filter((r) => r.region === regionFilter);


  return (
    <div>
      <SectionHeading title="Elo Rankings" subtitle="Current standings based on match history" />

      {/* Region filters */}
      <div className="mb-6 flex gap-2 flex-wrap">
        {REGIONS.map((r) => (
          <button
            key={r}
            onClick={() => setRegionFilter(r)}
            className={[
              "border px-4 py-1.5 font-display text-[10px] font-bold uppercase tracking-[0.22em] transition-colors",
              regionFilter === r
                ? "border-accent bg-accent/10 text-vcream"
                : "border-vborder bg-vcard text-vmuted hover:text-vcream",
            ].join(" ")}
          >
            {r !== "All" && (
              <span
                className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full"
                style={{ background: REGION_COLORS[r] ?? "#6B7E8F" }}
              />
            )}
            {r}
          </button>
        ))}
      </div>

      {loading && (
        <div className="flex justify-center py-20">
          <span className="inline-block h-8 w-8 animate-spin rounded-full border-2 border-accent border-t-transparent" />
        </div>
      )}

      {error && <p className="text-center font-sans text-sm text-red-400">{error}</p>}

      {!loading && !error && (
        <div className="overflow-x-auto rounded border border-vborder bg-vcard">
          {/* Header */}
          <div className="grid min-w-[560px] grid-cols-[40px_1fr_80px_80px_80px_60px_80px] gap-2 border-b border-vborder bg-velev px-4 py-2.5 text-[9px] font-bold uppercase tracking-[0.28em] text-vmuted">
            <span>#</span>
            <span>Team</span>
            <span className="text-right">Elo</span>
            <span className="text-right">30d</span>
            <span className="text-right">W–L</span>
            <span className="text-right">Win%</span>
            <span>Win Rate</span>
          </div>

          {filtered.map((row) => {
            const regionColor = REGION_COLORS[row.region] ?? "#6B7E8F";
            const logoSrc = TEAM_LOGOS[row.team];

            return (
              <div
                key={row.team}
                className="grid min-w-[560px] grid-cols-[40px_1fr_80px_80px_80px_60px_80px] items-center gap-2 border-b border-vborder/50 px-4 py-3 last:border-0 hover:bg-velev/60 transition-colors"
              >
                <RankBadge rank={row.rank} />

                <div className="flex items-center gap-2.5 min-w-0">
                  {logoSrc && (
                    <span className="inline-flex shrink-0 items-center justify-center rounded-sm bg-white/15"
                      style={{ width: 26, height: 26 }}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={logoSrc}
                        alt={row.team}
                        width={20}
                        height={20}
                        className="object-contain"
                        onError={(e) => { (e.currentTarget.parentElement as HTMLElement).style.display = "none"; }}
                      />
                    </span>
                  )}
                  <div className="min-w-0">
                    <p className="truncate font-display text-sm font-bold text-vcream">{row.team}</p>
                    <p className="font-display text-[9px] uppercase tracking-[0.25em]" style={{ color: regionColor }}>
                      {row.region}
                    </p>
                  </div>
                </div>

                <span className="text-right font-display text-sm font-bold text-vcream">
                  {Math.round(row.elo)}
                </span>

                <span className="text-right">
                  <EloChange change={row.elo_change} />
                </span>

                <span className="text-right font-sans text-xs text-vmuted">
                  {row.wins}–{row.losses}
                </span>

                <span className="text-right font-display text-xs font-bold text-vcream">
                  {(row.win_rate * 100).toFixed(0)}%
                </span>

                {/* Win rate bar */}
                <div className="h-1 rounded-full bg-vborder overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${row.win_rate * 100}%`, background: regionColor }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
