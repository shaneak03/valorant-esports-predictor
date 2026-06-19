"use client";

import type { Tab } from "@/lib/api";

const TABS: { id: Tab; label: string }[] = [
  { id: "predictor", label: "Match Predictor" },
  { id: "analysis",  label: "Team Analysis"   },
  { id: "rankings",  label: "Elo Rankings"    },
  { id: "eda",       label: "EDA"             },
];

interface Props {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
  apiOnline: boolean;
}

export default function Navbar({ activeTab, onTabChange, apiOnline }: Props) {
  return (
    <nav className="fixed left-0 right-0 top-0 z-50 flex flex-col">
      {/* ── Red accent strip ──────────────────────────────────────────────── */}
      <div className="flex h-8 min-w-0 items-center justify-between bg-accent px-3 sm:px-5">
        <div className="flex min-w-0 items-center gap-2">
          {/* Drop vct-logo.png in frontend/public/ */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/vct_masters.png"
            alt="VCT"
            width={18}
            height={18}
            className="shrink-0 object-contain"
            onError={(e) => { e.currentTarget.style.display = "none"; }}
          />
          <span className="truncate font-display text-[9px] font-bold uppercase tracking-[0.3em] text-white sm:text-[10px] sm:tracking-[0.35em]">
            <span className="hidden sm:inline">Valorant Champions Tour · </span>Masters London 2026
          </span>
        </div>
        <span className="hidden shrink-0 font-display text-[9px] font-semibold uppercase tracking-[0.3em] text-white/60 sm:block">
          ML Match Predictor
        </span>
      </div>

      {/* ── Dark tab strip ────────────────────────────────────────────────── */}
      <div
        className="flex h-[46px] items-stretch overflow-x-auto border-b border-vborder"
        style={{ background: "rgba(10,12,16,0.97)", backdropFilter: "blur(12px)" }}
      >
        <div className="flex shrink-0 items-stretch">
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={[
                "border-b-2 px-3 font-display text-[10px] font-bold uppercase tracking-[0.18em] transition-colors duration-150 sm:px-5 sm:text-[11px] sm:tracking-[0.22em]",
                activeTab === id
                  ? "border-accent text-vcream"
                  : "border-transparent text-vmuted hover:text-vcream",
              ].join(" ")}
            >
              {label}
            </button>
          ))}
        </div>

        {/* API status */}
        <div className="ml-auto flex shrink-0 items-center gap-2 border-l border-vborder px-3 sm:px-5">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ background: apiOnline ? "#00C87A" : "#FF4655" }}
          />
          <span className="hidden font-display text-[9px] font-semibold uppercase tracking-[0.28em] text-vmuted sm:block">
            {apiOnline ? "API Online" : "API Offline"}
          </span>
        </div>
      </div>
    </nav>
  );
}
