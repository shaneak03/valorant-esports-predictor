"use client";

import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import type { TeamStats } from "@/lib/api";
import { getTeamStats, REGION_COLORS, TEAM_COLORS } from "@/lib/api";
import { TEAM_LOGOS } from "@/lib/teamLogos";
import SectionHeading from "@/components/SectionHeading";

const ALL_TEAMS = [
  "G2 Esports", "LEVIATÁN", "NRG",
  "Team Heretics", "Team Vitality", "FUT Esports",
  "Paper Rex", "FULL SENSE", "Global Esports",
  "EDward Gaming", "Xi Lai Gaming", "Dragon Ranger Gaming",
];

function TeamLogo({ team, size = 32 }: { team: string; size?: number }) {
  const src = TEAM_LOGOS[team];
  if (!src) return null;
  return (
    <img
      src={src}
      alt={team}
      width={size}
      height={size}
      className="object-contain"
      onError={(e) => { e.currentTarget.style.display = "none"; }}
    />
  );
}

export default function TeamAnalysis() {
  const [selected, setSelected] = useState("");
  const [stats, setStats] = useState<TeamStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSelect(team: string) {
    setSelected(team);
    setError(null);
    setStats(null);
    if (!team) return;
    setLoading(true);
    try {
      setStats(await getTeamStats(team));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load team data");
    } finally {
      setLoading(false);
    }
  }

  const teamColor = selected ? (TEAM_COLORS[selected] ?? "#FF4655") : "#FF4655";
  const regionColor = stats ? (REGION_COLORS[stats.region] ?? "#ECE8E1") : "#ECE8E1";

  return (
    <div>
      <SectionHeading title="Team Analysis" subtitle="Deep dive into team stats and Elo history" />

      {/* Team picker */}
      <div className="mb-8 flex flex-wrap gap-2">
        {ALL_TEAMS.map((t) => (
          <button
            key={t}
            onClick={() => handleSelect(t)}
            className={[
              "flex items-center gap-2 border px-3 py-1.5 font-display text-[11px] font-bold uppercase tracking-[0.18em] transition-colors",
              selected === t
                ? "border-accent bg-accent/10 text-vcream"
                : "border-vborder bg-vcard text-vmuted hover:text-vcream hover:border-vmuted",
            ].join(" ")}
          >
            <TeamLogo team={t} size={16} />
            {t}
          </button>
        ))}
      </div>

      {loading && (
        <div className="flex justify-center py-20">
          <span className="inline-block h-8 w-8 animate-spin rounded-full border-2 border-accent border-t-transparent" />
        </div>
      )}

      {error && <p className="text-center font-sans text-sm text-red-400">{error}</p>}

      {stats && !loading && (
        <div className="space-y-6 animate-fade-in-up">
          {/* Profile card */}
          <div className="rounded border border-vborder bg-vcard p-6">
            <div className="flex items-center gap-4">
              <TeamLogo team={stats.team} size={56} />
              <div className="flex-1">
                <h3 className="font-display text-3xl font-extrabold uppercase tracking-[0.12em] text-vcream">
                  {stats.team}
                </h3>
                <span
                  className="mt-1 inline-block font-display text-[10px] font-bold uppercase tracking-[0.3em]"
                  style={{ color: regionColor }}
                >
                  {stats.region}
                </span>
              </div>
              <div className="text-right">
                <p className="font-display text-4xl font-extrabold" style={{ color: teamColor }}>
                  {Math.round(stats.elo)}
                </p>
                <p className="font-display text-[9px] uppercase tracking-[0.3em] text-vmuted">Elo Rating</p>
              </div>
            </div>

            {/* Stats row */}
            <div className="mt-5 grid grid-cols-3 divide-x divide-vborder border-t border-vborder pt-5">
              {[
                { label: "Wins",     value: stats.wins },
                { label: "Losses",   value: stats.losses },
                { label: "Win Rate", value: `${(stats.win_rate * 100).toFixed(1)}%` },
              ].map(({ label, value }) => (
                <div key={label} className="px-4 text-center">
                  <p className="font-display text-2xl font-bold text-vcream">{value}</p>
                  <p className="font-display text-[9px] uppercase tracking-[0.3em] text-vmuted">{label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Recent form */}
          <div className="rounded border border-vborder bg-vcard p-5">
            <h4 className="mb-3 font-display text-xs font-bold uppercase tracking-[0.25em] text-vmuted">
              Recent Form (last 10)
            </h4>
            <div className="flex gap-1.5">
              {stats.recent_form.map((r, i) => (
                <span
                  key={i}
                  className="flex h-7 w-7 items-center justify-center rounded font-display text-xs font-bold"
                  style={{
                    background: r === "W" ? "rgba(0,200,122,0.15)" : "rgba(255,70,85,0.15)",
                    color: r === "W" ? "#00C87A" : "#FF4655",
                    border: `1px solid ${r === "W" ? "rgba(0,200,122,0.3)" : "rgba(255,70,85,0.3)"}`,
                  }}
                >
                  {r}
                </span>
              ))}
            </div>
          </div>

          {/* Elo history chart */}
          <div className="rounded border border-vborder bg-vcard p-5">
            <h4 className="mb-4 font-display text-xs font-bold uppercase tracking-[0.25em] text-vmuted">
              Elo History
            </h4>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={stats.elo_history} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1C2A38" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "#6B7E8F", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "#1C2A38" }}
                  interval="preserveStartEnd"
                />
                <YAxis
                  domain={["auto", "auto"]}
                  tick={{ fill: "#6B7E8F", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  width={50}
                />
                <Tooltip
                  contentStyle={{ background: "#0F1419", border: "1px solid #1C2A38", borderRadius: 4 }}
                  labelStyle={{ color: "#ECE8E1", fontSize: 11 }}
                  itemStyle={{ color: teamColor, fontSize: 11 }}
                />
                <Line
                  type="monotone"
                  dataKey="elo"
                  stroke={teamColor}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: teamColor }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Recent matches */}
          <div className="rounded border border-vborder bg-vcard p-5">
            <h4 className="mb-3 font-display text-xs font-bold uppercase tracking-[0.25em] text-vmuted">
              Recent Matches
            </h4>
            <div className="divide-y divide-vborder">
              {stats.recent_matches.map((m, i) => (
                <div key={i} className="flex items-center gap-3 py-2.5">
                  <span
                    className="w-6 shrink-0 text-center font-display text-xs font-bold"
                    style={{ color: m.result === "W" ? "#00C87A" : "#FF4655" }}
                  >
                    {m.result}
                  </span>
                  <span className="flex-1 font-sans text-sm text-vcream">{m.opponent}</span>
                  <span className="font-sans text-xs text-vmuted">{m.date}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
