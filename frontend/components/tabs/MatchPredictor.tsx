"use client";

import { useState, useEffect } from "react";
import type { PredictionResult as PR } from "@/lib/api";
import { getTeams, predict } from "@/lib/api";
import TeamSelector from "@/components/TeamSelector";
import PredictionResult from "@/components/PredictionResult";
import SectionHeading from "@/components/SectionHeading";

export default function MatchPredictor() {
  const [teamsByRegion, setTeamsByRegion] = useState<Record<string, string[]>>({});
  const [teamA, setTeamA] = useState("");
  const [teamB, setTeamB] = useState("");
  const [result, setResult] = useState<PR | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getTeams().then(setTeamsByRegion).catch(() => {});
  }, []);

  const canPredict = !!(teamA && teamB && teamA !== teamB);

  async function handlePredict() {
    if (!canPredict) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predict(teamA, teamB));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col items-center">
      <div className="mb-10 text-center">
        <SectionHeading
          title="Match Predictor"
          subtitle="Select two teams to get an ML-powered win probability"
        />
      </div>

      <div className="w-full max-w-4xl">
        <div className="grid grid-cols-1 items-start gap-6 sm:grid-cols-[1fr_56px_1fr]">
          <TeamSelector
            label="Team A"
            value={teamA}
            onChange={setTeamA}
            exclude={teamB}
            teamsByRegion={teamsByRegion}
          />

          <div className="hidden flex-col items-center justify-center gap-3 pt-10 sm:flex">
            <div className="h-10 w-px bg-vborder" />
            <span className="font-display text-2xl font-bold tracking-widest text-vmuted">VS</span>
            <div className="h-10 w-px bg-vborder" />
          </div>

          <TeamSelector
            label="Team B"
            value={teamB}
            onChange={setTeamB}
            exclude={teamA}
            teamsByRegion={teamsByRegion}
          />
        </div>
      </div>

      <div className="mt-10 flex flex-col items-center gap-4">
        <button
          onClick={handlePredict}
          disabled={!canPredict || loading}
          className={[
            "font-display px-14 py-4 text-base font-bold uppercase tracking-[0.25em] transition-all duration-200",
            canPredict && !loading
              ? "cursor-pointer bg-accent text-white hover:bg-red-400 animate-pulse-glow"
              : "cursor-not-allowed border border-vborder bg-velev text-vmuted",
          ].join(" ")}
        >
          {loading ? (
            <span className="flex items-center gap-3">
              <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              Predicting…
            </span>
          ) : (
            "Predict Match →"
          )}
        </button>

        {error && (
          <p className="max-w-md text-center font-sans text-sm text-red-400">{error}</p>
        )}
      </div>

      {result && (
        <div className="mt-12 w-full max-w-4xl animate-fade-in-up">
          <PredictionResult result={result} />
        </div>
      )}

      <p className="mt-16 font-sans text-xs text-vmuted/50">
        Predictions based on historical VCT match data · For entertainment purposes
      </p>
    </div>
  );
}
