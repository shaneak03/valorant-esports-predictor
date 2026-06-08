"use client";

import { useState, useEffect } from "react";
import type { Tab } from "@/lib/api";
import { getTeams } from "@/lib/api";
import Navbar from "@/components/Navbar";
import MatchPredictor from "@/components/tabs/MatchPredictor";
import TeamAnalysis from "@/components/tabs/TeamAnalysis";
import EloRankings from "@/components/tabs/EloRankings";
import EDA from "@/components/tabs/EDA";

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("predictor");
  const [apiOnline, setApiOnline] = useState(false);

  useEffect(() => {
    getTeams()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  return (
    <>
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} apiOnline={apiOnline} />
      <main className="mx-auto max-w-6xl px-4 py-8">
        {activeTab === "predictor" && <MatchPredictor />}
        {activeTab === "analysis"  && <TeamAnalysis />}
        {activeTab === "rankings"  && <EloRankings />}
        {activeTab === "eda"       && <EDA />}
      </main>
    </>
  );
}
