"use client";

import { useEffect, useState } from "react";
import type { PredictionResult as PR } from "@/lib/api";
import { TEAM_LOGOS } from "@/lib/teamLogos";

function ProbBar({ probability, color, delayMs = 0 }: { probability: number; color: string; delayMs?: number }) {
  const pct = Math.round(probability * 100);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => setWidth(pct), delayMs);
    return () => clearTimeout(t);
  }, [pct, delayMs]);

  return (
    <div className="h-1.5 w-full overflow-hidden bg-velev">
      <div
        className="h-full transition-[width] duration-700 ease-out"
        style={{ width: `${width}%`, backgroundColor: color }}
      />
    </div>
  );
}

function CountUp({ target, delayMs = 0 }: { target: number; delayMs?: number }) {
  const [val, setVal] = useState(0);

  useEffect(() => {
    let n = 0;
    const step = Math.max(1, Math.ceil(target / 40));
    const t = setTimeout(() => {
      const id = setInterval(() => {
        n = Math.min(n + step, target);
        setVal(n);
        if (n >= target) clearInterval(id);
      }, 18);
      return () => clearInterval(id);
    }, delayMs);
    return () => clearTimeout(t);
  }, [target, delayMs]);

  return <>{val}</>;
}

function TeamLogo({ team, size = 32 }: { team: string; size?: number }) {
  const src = TEAM_LOGOS[team];
  if (!src) return null;
  return (
    <span className="inline-flex shrink-0 items-center justify-center rounded bg-white/15"
      style={{ width: size + 6, height: size + 6 }}>
      <img
        src={src}
        alt={team}
        width={size}
        height={size}
        className="object-contain"
        onError={(e) => { (e.currentTarget.parentElement as HTMLElement).style.display = "none"; }}
      />
    </span>
  );
}

export default function PredictionResult({ result }: { result: PR }) {
  const isAWinner = result.winner === result.team_a;

  return (
    <div className="border border-vborder bg-vcard">
      <div className="flex items-center gap-3 border-b border-vborder px-8 py-4">
        <span className="h-4 w-[3px] bg-accent" />
        <span className="font-display text-xs font-bold uppercase tracking-[0.3em] text-vmuted">
          Prediction Result
        </span>
      </div>

      <div className="px-8 py-8">
        <div className="space-y-7">
          {/* Team A */}
          <div className={isAWinner ? "rounded border border-accent/20 bg-accent/5 p-4 -mx-4" : ""}>
            <div className="mb-2.5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <TeamLogo team={result.team_a} size={28} />
                <span
                  className={[
                    "font-display text-xl font-bold uppercase tracking-wide",
                    isAWinner ? "text-accent" : "text-vcream",
                  ].join(" ")}
                >
                  {result.team_a}
                </span>
                {isAWinner && (
                  <span className="font-display text-[10px] font-semibold uppercase tracking-[0.3em] text-accent">
                    ◈ Winner
                  </span>
                )}
              </div>
              <span
                className={[
                  "font-display text-4xl font-extrabold tabular-nums",
                  isAWinner ? "text-accent" : "text-vmuted",
                ].join(" ")}
              >
                <CountUp target={Math.round(result.prob_a * 100)} delayMs={150} />%
              </span>
            </div>
            <ProbBar probability={result.prob_a} color={isAWinner ? "#FF4655" : "#6B7E8F"} delayMs={150} />
          </div>

          <div className="h-px bg-vborder" />

          {/* Team B */}
          <div className={!isAWinner ? "rounded border border-accent/20 bg-accent/5 p-4 -mx-4" : ""}>
            <div className="mb-2.5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <TeamLogo team={result.team_b} size={28} />
                <span
                  className={[
                    "font-display text-xl font-bold uppercase tracking-wide",
                    !isAWinner ? "text-accent" : "text-vcream",
                  ].join(" ")}
                >
                  {result.team_b}
                </span>
                {!isAWinner && (
                  <span className="font-display text-[10px] font-semibold uppercase tracking-[0.3em] text-accent">
                    ◈ Winner
                  </span>
                )}
              </div>
              <span
                className={[
                  "font-display text-4xl font-extrabold tabular-nums",
                  !isAWinner ? "text-accent" : "text-vmuted",
                ].join(" ")}
              >
                <CountUp target={Math.round(result.prob_b * 100)} delayMs={350} />%
              </span>
            </div>
            <ProbBar probability={result.prob_b} color={!isAWinner ? "#FF4655" : "#6B7E8F"} delayMs={350} />
          </div>
        </div>

        {/* Elo strip */}
        <div className="mt-8 border-t border-vborder pt-7">
          <p className="mb-4 font-display text-[10px] font-bold uppercase tracking-[0.35em] text-vmuted">
            Current Elo Rating
          </p>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <TeamLogo team={result.team_a} size={24} />
              <p className="mt-1 font-display text-3xl font-extrabold tabular-nums text-vcream">
                {result.elo_a}
              </p>
              <p className="mt-1 font-display text-[10px] uppercase tracking-[0.2em] text-vmuted">
                {result.team_a}
              </p>
            </div>
            <div className="flex items-center justify-center">
              <span className="font-display text-xs uppercase tracking-[0.3em] text-vborder">vs</span>
            </div>
            <div className="text-center">
              <TeamLogo team={result.team_b} size={24} />
              <p className="mt-1 font-display text-3xl font-extrabold tabular-nums text-vcream">
                {result.elo_b}
              </p>
              <p className="mt-1 font-display text-[10px] uppercase tracking-[0.2em] text-vmuted">
                {result.team_b}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
