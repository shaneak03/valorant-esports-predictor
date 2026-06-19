"use client";

import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend, Cell, ReferenceLine,
} from "recharts";
import type { StatsData, EdaFullData } from "@/lib/api";
import { getStats, getEdaFull, REGION_COLORS, TEAM_COLORS } from "@/lib/api";
import { TEAM_LOGOS } from "@/lib/teamLogos";
import { MAP_IMAGES } from "@/lib/mapImages";
import SectionHeading from "@/components/SectionHeading";

// ── helpers ───────────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Fmt = (v: any, name: any) => [string | number, string];

const TOOLTIP_STYLE = {
  contentStyle: { background: "#0F1419", border: "1px solid #1C2A38", borderRadius: 4, fontSize: 11 },
  labelStyle: { color: "#ECE8E1", fontSize: 11 },
  itemStyle: { fontSize: 11 },
};

const REGIONS = ["All", "Americas", "EMEA", "Pacific", "China"];

const REGION_TEAMS: Record<string, string[]> = {
  Americas: ["G2 Esports", "LEVIATÁN", "NRG"],
  EMEA:     ["Team Heretics", "Team Vitality", "FUT Esports"],
  Pacific:  ["Paper Rex", "FULL SENSE", "Global Esports"],
  China:    ["EDward Gaming", "Xi Lai Gaming", "Dragon Ranger Gaming"],
};

const ALL_TEAMS = [
  "G2 Esports", "LEVIATÁN", "NRG",
  "Team Heretics", "Team Vitality", "FUT Esports",
  "Paper Rex", "FULL SENSE", "Global Esports",
  "EDward Gaming", "Xi Lai Gaming", "Dragon Ranger Gaming",
];

const SHORT: Record<string, string> = {
  "G2 Esports": "G2", "LEVIATÁN": "LEVI", "NRG": "NRG",
  "Team Heretics": "HER", "Team Vitality": "VIT", "FUT Esports": "FUT",
  "Paper Rex": "PRX", "FULL SENSE": "FSN", "Global Esports": "GE",
  "EDward Gaming": "EDG", "Xi Lai Gaming": "XLG", "Dragon Ranger Gaming": "DRG",
};

function wrBg(wr: number | null): string {
  if (wr === null) return "rgba(28,42,56,0.4)";
  const t = Math.max(0, Math.min(1, (wr - 0.25) / 0.5));
  if (t < 0.5) return `rgba(255,70,85,${((0.5 - t) * 2 * 0.55).toFixed(2)})`;
  return `rgba(0,200,122,${((t - 0.5) * 2 * 0.55).toFixed(2)})`;
}

function wrText(wr: number | null): string {
  if (wr === null) return "#6B7E8F";
  if (wr >= 0.65 || wr <= 0.35) return "#ECE8E1";
  return "#9AAFC0";
}

function TeamLogo({ team, size = 16 }: { team: string; size?: number }) {
  const src = TEAM_LOGOS[team];
  if (!src) return null;
  return (
    <span className="inline-flex shrink-0 items-center justify-center rounded-sm bg-white/15"
      style={{ width: size + 4, height: size + 4 }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt="" width={size} height={size} className="object-contain"
        onError={(e) => { (e.currentTarget.parentElement as HTMLElement).style.display = "none"; }} />
    </span>
  );
}

// ── sub-section tabs ───────────────────────────────────────────────────────────

type Section = "elo" | "overview" | "mapwr" | "h2h" | "form" | "breakdown";
const SECTIONS: { id: Section; label: string }[] = [
  { id: "elo",       label: "Elo Stats"     },
  { id: "overview",  label: "Overview"      },
  { id: "mapwr",     label: "Map Win Rates" },
  { id: "h2h",       label: "Head-to-Head"  },
  { id: "form",      label: "Recent Form"   },
  { id: "breakdown", label: "Team Breakdown"},
];

// ── section components ─────────────────────────────────────────────────────────

function EloStats({ data }: { data: StatsData }) {
  const [timelineRegion, setTimelineRegion] = useState("All");

  const timelineTeams =
    timelineRegion === "All" ? ALL_TEAMS : (REGION_TEAMS[timelineRegion] ?? []);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Elo Distribution */}
        <div className="rounded border border-vborder bg-vcard p-5">
          <h4 className="mb-4 font-display text-xs font-bold uppercase tracking-[0.25em] text-vmuted">
            Elo Distribution
          </h4>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              layout="vertical"
              data={[...data.elo_distribution].sort((a, b) => b.elo - a.elo)}
              margin={{ top: 0, right: 16, bottom: 0, left: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1C2A38" horizontal={false} />
              <XAxis type="number" domain={["auto", "auto"]}
                tick={{ fill: "#6B7E8F", fontSize: 10 }} tickLine={false}
                axisLine={{ stroke: "#1C2A38" }} />
              <YAxis type="category" dataKey="team" width={160}
                tick={{ fill: "#ECE8E1", fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip {...TOOLTIP_STYLE}
                formatter={((v: number) => [Math.round(v), "Elo"]) as Fmt} />
              <Bar dataKey="elo" radius={[0, 2, 2, 0]} maxBarSize={14}>
                {[...data.elo_distribution].sort((a, b) => b.elo - a.elo).map((e) => (
                  <Cell key={e.team} fill={TEAM_COLORS[e.team] ?? "#FF4655"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Win Rate by Region */}
        <div className="rounded border border-vborder bg-vcard p-5">
          <h4 className="mb-4 font-display text-xs font-bold uppercase tracking-[0.25em] text-vmuted">
            Win Rate by Region
          </h4>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.win_rates_by_region}
              margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1C2A38" vertical={false} />
              <XAxis dataKey="region" tick={{ fill: "#ECE8E1", fontSize: 11 }}
                tickLine={false} axisLine={{ stroke: "#1C2A38" }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{ fill: "#6B7E8F", fontSize: 10 }} tickLine={false} axisLine={false} width={40} />
              <Tooltip {...TOOLTIP_STYLE}
                formatter={((v: number) => [`${(v * 100).toFixed(1)}%`, "Win Rate"]) as Fmt} />
              <Bar dataKey="win_rate" radius={[2, 2, 0, 0]} maxBarSize={48}>
                {data.win_rates_by_region.map((e) => (
                  <Cell key={e.region} fill={REGION_COLORS[e.region] ?? "#6B7E8F"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Elo Timeline */}
      <div className="rounded border border-vborder bg-vcard p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h4 className="font-display text-xs font-bold uppercase tracking-[0.25em] text-vmuted">
            Elo Timeline
          </h4>
          <div className="flex gap-1.5">
            {REGIONS.map((r) => (
              <button key={r} onClick={() => setTimelineRegion(r)}
                className={["border px-2.5 py-1 font-display text-[9px] font-bold uppercase tracking-[0.22em] transition-colors",
                  timelineRegion === r ? "border-accent bg-accent/10 text-vcream" : "border-vborder bg-velev text-vmuted hover:text-vcream",
                ].join(" ")}>
                {r}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data.elo_timeline} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1C2A38" />
            <XAxis dataKey="date" tick={{ fill: "#6B7E8F", fontSize: 10 }}
              tickLine={false} axisLine={{ stroke: "#1C2A38" }} interval="preserveStartEnd" />
            <YAxis domain={["auto", "auto"]} tick={{ fill: "#6B7E8F", fontSize: 10 }}
              tickLine={false} axisLine={false} width={50} />
            <Tooltip {...TOOLTIP_STYLE}
              formatter={((v: number, name: string) => [Math.round(v), name]) as Fmt} />
            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 12 }}
              formatter={(value) => (
                <span style={{ color: TEAM_COLORS[value] ?? "#ECE8E1", fontSize: 10 }}>{value}</span>
              )} />
            {timelineTeams.map((team) => (
              <Line key={team} type="monotone" dataKey={team}
                stroke={TEAM_COLORS[team] ?? "#6B7E8F"} strokeWidth={1.5}
                dot={false} activeDot={{ r: 3 }} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Wins vs Losses */}
      <div className="rounded border border-vborder bg-vcard p-5">
        <h4 className="mb-4 font-display text-xs font-bold uppercase tracking-[0.25em] text-vmuted">
          Wins vs Losses
        </h4>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.elo_distribution} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1C2A38" vertical={false} />
            <XAxis dataKey="team" tick={{ fill: "#ECE8E1", fontSize: 9 }}
              tickLine={false} axisLine={{ stroke: "#1C2A38" }}
              interval={0} angle={-30} textAnchor="end" height={55} />
            <YAxis tick={{ fill: "#6B7E8F", fontSize: 10 }} tickLine={false} axisLine={false} width={30} />
            <Tooltip {...TOOLTIP_STYLE}
              formatter={((v: number, name: string) => [v, name === "wins" ? "Wins" : "Losses"]) as Fmt} />
            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8 }}
              formatter={(value) => (
                <span style={{ color: value === "wins" ? "#00C87A" : "#FF4655", fontSize: 10 }}>
                  {value === "wins" ? "Wins" : "Losses"}
                </span>
              )} />
            <Bar dataKey="wins"   stackId="a" fill="#00C87A" radius={[0, 0, 0, 0]} maxBarSize={32} />
            <Bar dataKey="losses" stackId="a" fill="#FF4655" radius={[2, 2, 0, 0]} maxBarSize={32} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Overview({ data }: { data: EdaFullData }) {
  return (
    <div className="rounded border border-vborder bg-vcard overflow-x-auto">
      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-vborder bg-velev">
            {["Team", "Region", "Series", "Win%", "Last 5", "Streak", "Best Map", "Worst Map", "Likely Ban"].map((h) => (
              <th key={h} className="px-4 py-2.5 font-display text-[9px] font-bold uppercase tracking-[0.28em] text-vmuted whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.overall_stats.map((row) => {
            const rc = REGION_COLORS[row.region] ?? "#6B7E8F";
            return (
              <tr key={row.team} className="border-b border-vborder/50 last:border-0 hover:bg-velev/50 transition-colors">
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <TeamLogo team={row.team} size={16} />
                    <span className="font-display text-sm font-bold text-vcream whitespace-nowrap">{row.team}</span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <span className="font-display text-[10px] font-bold uppercase tracking-[0.2em]" style={{ color: rc }}>{row.region}</span>
                </td>
                <td className="px-4 py-3 text-right font-display text-sm font-bold text-vcream">{row.series}</td>
                <td className="px-4 py-3 text-right font-display text-sm font-bold"
                  style={{ color: row.win_pct >= 0.5 ? "#00C87A" : "#FF4655" }}>
                  {(row.win_pct * 100).toFixed(0)}%
                </td>
                <td className="px-4 py-3">
                  <div className="flex gap-0.5">
                    {row.last5.split("").map((r, i) => (
                      <span key={i} className="inline-flex h-5 w-5 items-center justify-center rounded font-display text-[10px] font-bold"
                        style={{
                          background: r === "W" ? "rgba(0,200,122,0.15)" : "rgba(255,70,85,0.15)",
                          color: r === "W" ? "#00C87A" : "#FF4655",
                        }}>
                        {r}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="px-4 py-3 font-display text-xs font-bold"
                  style={{ color: row.streak.startsWith("W") ? "#00C87A" : "#FF4655" }}>
                  {row.streak}
                </td>
                <td className="px-4 py-3 font-sans text-xs text-vcream whitespace-nowrap">
                  {row.best_map ? `${row.best_map} (${((row.best_map_wr ?? 0) * 100).toFixed(0)}%)` : "—"}
                </td>
                <td className="px-4 py-3 font-sans text-xs text-vcream whitespace-nowrap">
                  {row.worst_map ? `${row.worst_map} (${((row.worst_map_wr ?? 0) * 100).toFixed(0)}%)` : "—"}
                </td>
                <td className="px-4 py-3 font-sans text-xs text-vmuted">{row.likely_ban ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MapWinRates({ data }: { data: EdaFullData }) {
  const [hovered, setHovered] = useState<{ team: string; map: string } | null>(null);

  return (
    <div className="space-y-3">
      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 rounded border border-vborder bg-velev px-4 py-2.5">
        <span className="font-display text-[9px] font-bold uppercase tracking-[0.25em] text-vmuted">Legend:</span>
        {[
          { bg: "rgba(0,200,122,0.45)", label: "Strong (≥65% win rate)" },
          { bg: "rgba(108,126,143,0.15)", label: "Even (35–65%)" },
          { bg: "rgba(255,70,85,0.45)", label: "Weak (≤35% win rate)" },
          { bg: "rgba(28,42,56,0.4)", label: "Too few games played (<3)" },
        ].map(({ bg, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <span className="inline-block h-3.5 w-3.5 rounded-sm border border-vborder/50" style={{ background: bg }} />
            <span className="font-sans text-[10px] text-vmuted">{label}</span>
          </div>
        ))}
      </div>

      <div className="overflow-x-auto rounded border border-vborder bg-vcard">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-vborder bg-velev">
              {/* Team column header */}
              <th className="px-4 py-3 text-left font-display text-[9px] font-bold uppercase tracking-[0.28em] text-vmuted">
                Team
              </th>
              {data.map_pool.map((mapName) => {
                const imgSrc = MAP_IMAGES[mapName];
                return (
                  <th key={mapName} className="px-1 py-2 text-center">
                    <div className="flex flex-col items-center gap-1" style={{ minWidth: 72 }}>
                      {/* Map image — reserve a fixed slot; shows when user drops image */}
                      <div className="h-10 w-16 overflow-hidden rounded border border-vborder/40 bg-velev">
                        {imgSrc && (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={imgSrc} alt={mapName}
                            className="h-full w-full object-cover"
                            onError={(e) => { e.currentTarget.style.display = "none"; }} />
                        )}
                      </div>
                      <span className="font-display text-[9px] font-bold uppercase tracking-[0.22em] text-vmuted">
                        {mapName}
                      </span>
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {data.teams.map((team) => {
              const regionKey = Object.keys(REGION_TEAMS).find(r => REGION_TEAMS[r].includes(team)) ?? "";
              const regionColor = REGION_COLORS[regionKey] ?? "#6B7E8F";
              return (
                <tr key={team} className="border-b border-vborder/40 last:border-0 hover:bg-velev/40 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2 whitespace-nowrap">
                      <TeamLogo team={team} size={16} />
                      <div>
                        <p className="font-display text-[11px] font-bold text-vcream">{team}</p>
                        <p className="font-display text-[8px] uppercase tracking-[0.2em]" style={{ color: regionColor }}>
                          {regionKey}
                        </p>
                      </div>
                    </div>
                  </td>
                  {data.map_pool.map((mapName) => {
                    const stat = data.map_stats[team]?.[mapName];
                    const wr = stat?.win_rate ?? null;
                    const played = stat?.played ?? 0;
                    const wins = stat?.wins ?? 0;
                    const isHovered = hovered?.team === team && hovered?.map === mapName;

                    return (
                      <td key={mapName} className="px-1 py-1.5 text-center">
                        <div
                          className="relative mx-auto flex flex-col items-center justify-center rounded cursor-default transition-all duration-150"
                          style={{
                            background: wrBg(wr),
                            minWidth: 72,
                            height: 52,
                            outline: isHovered ? "1px solid rgba(255,255,255,0.2)" : "none",
                          }}
                          onMouseEnter={() => setHovered({ team, map: mapName })}
                          onMouseLeave={() => setHovered(null)}
                        >
                          {wr !== null ? (
                            <>
                              <span className="font-display text-sm font-extrabold leading-tight" style={{ color: wrText(wr) }}>
                                {(wr * 100).toFixed(0)}%
                              </span>
                              <span className="font-sans text-[9px] leading-tight" style={{ color: wrText(wr), opacity: 0.75 }}>
                                {wins}W / {played - wins}L
                              </span>
                              <span className="font-sans text-[8px] leading-tight" style={{ color: wrText(wr), opacity: 0.5 }}>
                                {played} maps played
                              </span>
                            </>
                          ) : (
                            <div className="flex flex-col items-center">
                              <span className="font-sans text-[10px] font-bold text-vmuted">
                                {played === 0 ? "No data" : `${played} played`}
                              </span>
                              {played > 0 && (
                                <span className="font-sans text-[8px] text-vmuted/60">
                                  Need 3+ for stats
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function H2HMatrix({ data }: { data: EdaFullData }) {
  return (
    <div className="overflow-x-auto rounded border border-vborder bg-vcard">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-vborder bg-velev">
            <th className="px-3 py-2.5 text-left font-display text-[9px] font-bold uppercase tracking-[0.28em] text-vmuted">
              ↓ vs →
            </th>
            {data.teams.map((t) => (
              <th key={t} className="px-2 py-2.5 text-center font-display text-[9px] font-bold text-vmuted">
                {SHORT[t] ?? t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.teams.map((rowTeam) => (
            <tr key={rowTeam} className="border-b border-vborder/40 last:border-0">
              <td className="px-3 py-2">
                <div className="flex items-center gap-1.5 whitespace-nowrap">
                  <TeamLogo team={rowTeam} size={13} />
                  <span className="font-display text-[10px] font-bold text-vcream">{SHORT[rowTeam]}</span>
                </div>
              </td>
              {data.teams.map((colTeam) => {
                if (rowTeam === colTeam) {
                  return (
                    <td key={colTeam} className="px-2 py-2 text-center">
                      <div className="flex h-9 w-12 items-center justify-center rounded bg-vborder/30 text-vmuted">—</div>
                    </td>
                  );
                }
                const entry = data.h2h[rowTeam]?.[colTeam];
                const wr = entry?.win_rate ?? null;
                const total = entry?.total ?? 0;
                return (
                  <td key={colTeam} className="px-2 py-2 text-center">
                    <div className="flex h-9 w-12 flex-col items-center justify-center rounded"
                      style={{ background: wrBg(wr) }}>
                      {wr !== null ? (
                        <>
                          <span className="font-display text-[11px] font-bold" style={{ color: wrText(wr) }}>
                            {(wr * 100).toFixed(0)}%
                          </span>
                          <span className="font-sans text-[8px]" style={{ color: wrText(wr), opacity: 0.6 }}>
                            ({total})
                          </span>
                        </>
                      ) : (
                        <span className="font-sans text-[9px] text-vmuted">—</span>
                      )}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t border-vborder px-4 py-2 font-sans text-[10px] text-vmuted">
        Row team win rate vs column team · Green = row dominates · Red = column dominates · — = never played
      </p>
    </div>
  );
}

function RecentForm({ data }: { data: EdaFullData }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {data.teams.map((team) => {
        const points = data.recent_form[team] ?? [];
        const regionKey = Object.keys(REGION_TEAMS).find(r => REGION_TEAMS[r].includes(team)) ?? "";
        const color = TEAM_COLORS[team] ?? REGION_COLORS[regionKey] ?? "#6B7E8F";
        const results = points.map((p) => p.result);
        const w = results.filter(Boolean).length;
        const n = results.length;
        const winPct = n > 0 ? w / n : 0;

        return (
          <div key={team} className="rounded border border-vborder bg-vcard p-4">
            <div className="mb-2 flex items-center gap-2">
              <TeamLogo team={team} size={16} />
              <span className="font-display text-xs font-bold text-vcream">{team}</span>
              <span className="ml-auto font-display text-xs font-bold"
                style={{ color: winPct >= 0.5 ? "#00C87A" : "#FF4655" }}>
                {w}W–{n - w}L
              </span>
            </div>
            {points.length < 3 ? (
              <p className="py-4 text-center font-sans text-xs text-vmuted">Insufficient data</p>
            ) : (
              <ResponsiveContainer width="100%" height={110}>
                <LineChart data={points} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1C2A38" />
                  <XAxis dataKey="index" hide />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fill: "#6B7E8F", fontSize: 9 }} tickLine={false} axisLine={false} width={36} />
                  <ReferenceLine y={0.5} stroke="#1C2A38" strokeDasharray="4 2" />
                  <Tooltip {...TOOLTIP_STYLE}
                    formatter={((v: number) => [`${(v * 100).toFixed(0)}%`, "Win rate"]) as Fmt} />
                  <Line type="monotone" dataKey="rolling_wr" stroke={color} strokeWidth={1.5}
                    dot={false} activeDot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        );
      })}
    </div>
  );
}

function TeamBreakdown({ data }: { data: EdaFullData }) {
  const [team, setTeam] = useState(data.teams[0] ?? "");

  const regionKey = Object.keys(REGION_TEAMS).find(r => REGION_TEAMS[r].includes(team)) ?? "";
  const teamColor = TEAM_COLORS[team] ?? "#FF4655";
  const regionColor = REGION_COLORS[regionKey] ?? "#6B7E8F";

  const mapStatsList = data.map_pool.map((m) => ({
    map: m,
    ...(data.map_stats[team]?.[m] ?? { played: 0, wins: 0, losses: 0, win_rate: null, avg_atk: null, avg_def: null, play_freq: 0 }),
  }));

  const atkDefData = mapStatsList
    .filter((s) => s.played >= 3 && s.avg_atk !== null && s.avg_def !== null)
    .map((s) => ({ map: s.map, Attack: +(s.avg_atk! * 100).toFixed(1), Defence: +(s.avg_def! * 100).toFixed(1) }));

  const permaBanData = [...mapStatsList]
    .sort((a, b) => a.play_freq - b.play_freq)
    .map((s) => ({
      map: s.map,
      "Play Freq": +(s.play_freq * 100).toFixed(1),
      played: s.played,
    }));

  return (
    <div className="space-y-5">
      {/* Team selector */}
      <div className="flex flex-wrap gap-2">
        {data.teams.map((t) => (
          <button key={t} onClick={() => setTeam(t)}
            className={["flex items-center gap-1.5 border px-3 py-1.5 font-display text-[10px] font-bold uppercase tracking-[0.18em] transition-colors",
              team === t ? "border-accent bg-accent/10 text-vcream" : "border-vborder bg-vcard text-vmuted hover:text-vcream",
            ].join(" ")}>
            <TeamLogo team={t} size={14} />
            {SHORT[t] ?? t}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-3">
        <TeamLogo team={team} size={28} />
        <div>
          <h4 className="font-display text-lg font-extrabold uppercase tracking-[0.12em] text-vcream">{team}</h4>
          <span className="font-display text-[9px] font-bold uppercase tracking-[0.3em]" style={{ color: regionColor }}>{regionKey}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* Attack vs Defence */}
        <div className="rounded border border-vborder bg-vcard p-5">
          <h5 className="mb-4 font-display text-[10px] font-bold uppercase tracking-[0.25em] text-vmuted">
            Attack vs Defence (round win rate per map)
          </h5>
          {atkDefData.length === 0 ? (
            <p className="py-8 text-center font-sans text-xs text-vmuted">Insufficient map data (&lt;3 played)</p>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={atkDefData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1C2A38" vertical={false} />
                <XAxis dataKey="map" tick={{ fill: "#ECE8E1", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "#1C2A38" }} />
                <YAxis domain={[0, 55]} tickFormatter={(v) => `${v}%`}
                  tick={{ fill: "#6B7E8F", fontSize: 10 }} tickLine={false} axisLine={false} width={38} />
                <ReferenceLine y={25} stroke="#1C2A38" strokeDasharray="4 2" label={{ value: "avg", fill: "#6B7E8F", fontSize: 9 }} />
                <Tooltip {...TOOLTIP_STYLE}
                  formatter={((v: number, name: string) => [`${v.toFixed(1)}%`, name]) as Fmt} />
                <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8 }}
                  formatter={(value) => (
                    <span style={{ color: value === "Attack" ? "#FF8080" : "#74B9FF", fontSize: 10 }}>{value}</span>
                  )} />
                <Bar dataKey="Attack"  fill="#FF8080" radius={[2, 2, 0, 0]} maxBarSize={24} />
                <Bar dataKey="Defence" fill="#74B9FF" radius={[2, 2, 0, 0]} maxBarSize={24} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Permaban inference */}
        <div className="rounded border border-vborder bg-vcard p-5">
          <h5 className="mb-4 font-display text-[10px] font-bold uppercase tracking-[0.25em] text-vmuted">
            Permaban Inference (shorter = rarely played)
          </h5>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart layout="vertical" data={permaBanData}
              margin={{ top: 0, right: 40, bottom: 0, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1C2A38" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`}
                tick={{ fill: "#6B7E8F", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "#1C2A38" }} />
              <YAxis type="category" dataKey="map" width={64}
                tick={{ fill: "#ECE8E1", fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip {...TOOLTIP_STYLE}
                formatter={((v: number, name: string, item: { payload: { played: number } }) =>
                  [`${v.toFixed(1)}% (${item.payload.played} maps)`, "Play frequency"]) as Fmt} />
              <Bar dataKey="Play Freq" radius={[0, 2, 2, 0]} maxBarSize={18}>
                {permaBanData.map((entry, i) => (
                  <Cell key={entry.map}
                    fill={i === 0 ? "#FF4655" : i === permaBanData.length - 1 ? "#00C87A" : teamColor}
                    fillOpacity={0.7 + i * 0.04} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-2 font-sans text-[9px] text-vmuted">
            Red = shortest bar = likely permaban · Values are maps-played per series
          </p>
        </div>
      </div>
    </div>
  );
}

// ── main component ─────────────────────────────────────────────────────────────

export default function EDA() {
  const [statsData, setStatsData] = useState<StatsData | null>(null);
  const [edaData, setEdaData] = useState<EdaFullData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [section, setSection] = useState<Section>("elo");

  useEffect(() => {
    Promise.all([getStats(), getEdaFull()])
      .then(([s, e]) => { setStatsData(s); setEdaData(e); })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load data"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center py-24">
        <span className="inline-block h-8 w-8 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      </div>
    );
  }

  if (error) return <p className="py-20 text-center font-sans text-sm text-red-400">{error}</p>;
  if (!statsData || !edaData) return null;

  return (
    <div>
      <SectionHeading title="EDA" subtitle="Exploratory data analysis" />

      {/* Sub-section nav */}
      <div className="mb-7 flex flex-wrap items-center gap-1.5 border-b border-vborder pb-4">
        {SECTIONS.map(({ id, label }) => (
          <button key={id} onClick={() => setSection(id)}
            className={["border px-4 py-1.5 font-display text-[10px] font-bold uppercase tracking-[0.2em] transition-colors",
              section === id ? "border-accent bg-accent/10 text-vcream" : "border-vborder bg-vcard text-vmuted hover:text-vcream",
            ].join(" ")}>
            {label}
          </button>
        ))}
        {/* Contextual scope badge — only shown for sections that use the date filter */}
        {section !== "elo" && (
          <span className="ml-auto font-display text-[9px] font-bold uppercase tracking-[0.25em] text-vmuted">
            Match data since {edaData.min_date}
          </span>
        )}
        {section === "elo" && (
          <span className="ml-auto font-display text-[9px] font-bold uppercase tracking-[0.25em] text-vmuted">
            All-time Elo data
          </span>
        )}
      </div>

      {section === "elo"       && <EloStats data={statsData} />}
      {section === "overview"  && <Overview data={edaData} />}
      {section === "mapwr"     && <MapWinRates data={edaData} />}
      {section === "h2h"       && <H2HMatrix data={edaData} />}
      {section === "form"      && <RecentForm data={edaData} />}
      {section === "breakdown" && <TeamBreakdown data={edaData} />}
    </div>
  );
}
