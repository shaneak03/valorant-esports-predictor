"use client";

import { useEffect, useRef, useState } from "react";
import { TEAM_LOGOS } from "@/lib/teamLogos";

interface Props {
  label: string;
  value: string;
  onChange: (team: string) => void;
  exclude: string;
  teamsByRegion: Record<string, string[]>;
}

const REGION_COLORS: Record<string, string> = {
  Americas: "#FF7A00",
  EMEA:     "#B6FF00",
  Pacific:  "#00E5FF",
  China:    "#FF3B3B",
};

export default function TeamSelector({ label, value, onChange, exclude, teamsByRegion }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const selectedRegion = value
    ? Object.keys(teamsByRegion).find((r) => teamsByRegion[r].includes(value)) ?? null
    : null;
  const regionColor = selectedRegion ? REGION_COLORS[selectedRegion] : undefined;
  const logoSrc = value ? TEAM_LOGOS[value] : undefined;

  return (
    <div ref={ref} className="relative">
      <p className="mb-3 font-display text-xs font-semibold uppercase tracking-[0.3em] text-vmuted">
        {label}
      </p>

      {/* Trigger card */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={[
          "relative w-full cursor-pointer border bg-vcard p-6 text-left transition-all duration-200",
          value
            ? "border-accent shadow-[0_0_24px_rgba(255,70,85,0.12)]"
            : "border-vborder hover:border-accent/50",
        ].join(" ")}
      >
        {value && <span className="absolute inset-y-0 left-0 w-[3px] bg-accent" aria-hidden />}

        <span className="block pl-2">
          {value ? (
            <span className="flex items-center gap-3">
              {logoSrc && (
                <span className="inline-flex shrink-0 items-center justify-center rounded bg-white/15"
                  style={{ width: 44, height: 44 }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={logoSrc}
                    alt={value}
                    width={36}
                    height={36}
                    className="object-contain"
                    onError={(e) => { (e.currentTarget.parentElement as HTMLElement).style.display = "none"; }}
                  />
                </span>
              )}
              <span>
                <span className="block font-display text-xl font-extrabold uppercase tracking-wide text-vcream sm:text-3xl">
                  {value}
                </span>
                <span className="mt-1 flex items-center gap-2">
                  <span
                    className="inline-block h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: regionColor }}
                  />
                  <span
                    className="font-display text-xs font-semibold uppercase tracking-[0.25em]"
                    style={{ color: regionColor }}
                  >
                    {selectedRegion}
                  </span>
                </span>
              </span>
            </span>
          ) : (
            <span className="font-display text-lg font-semibold uppercase tracking-wider text-vmuted sm:text-2xl">
              Select Team
            </span>
          )}
        </span>

        <span
          className={[
            "absolute right-5 top-1/2 -translate-y-1/2 text-vmuted transition-transform duration-200",
            open ? "rotate-180" : "",
          ].join(" ")}
          aria-hidden
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </span>
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute left-0 right-0 top-full z-50 mt-1 max-h-80 overflow-y-auto border border-vborder bg-vcard shadow-2xl">
          {Object.entries(teamsByRegion).map(([region, teams]) => (
            <div key={region}>
              <div
                className="border-b border-vborder bg-velev px-4 py-2 font-display text-xs font-bold uppercase tracking-[0.3em]"
                style={{ color: REGION_COLORS[region] }}
              >
                {region}
              </div>

              {teams.map((team) => {
                const isExcluded = team === exclude;
                const isSelected = team === value;
                const tLogo = TEAM_LOGOS[team];
                return (
                  <button
                    key={team}
                    type="button"
                    disabled={isExcluded}
                    onClick={() => { onChange(team); setOpen(false); }}
                    className={[
                      "flex w-full items-center gap-3 border-l-2 px-4 py-3 text-left transition-all duration-150",
                      isExcluded
                        ? "cursor-not-allowed border-transparent opacity-30"
                        : isSelected
                          ? "border-accent bg-accent/10"
                          : "border-transparent hover:border-accent/40 hover:bg-velev",
                    ].join(" ")}
                  >
                    {tLogo && (
                      <span className="inline-flex shrink-0 items-center justify-center rounded-sm bg-white/15"
                        style={{ width: 26, height: 26 }}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={tLogo}
                          alt={team}
                          width={20}
                          height={20}
                          className="object-contain"
                          onError={(e) => { (e.currentTarget.parentElement as HTMLElement).style.display = "none"; }}
                        />
                      </span>
                    )}
                    <span
                      className="h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{ backgroundColor: REGION_COLORS[region] }}
                    />
                    <span
                      className={[
                        "font-display text-lg font-semibold tracking-wide",
                        isSelected ? "text-accent" : "text-vcream",
                      ].join(" ")}
                    >
                      {team}
                    </span>
                    {isExcluded && (
                      <span className="ml-auto font-sans text-xs text-vmuted">Selected</span>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
