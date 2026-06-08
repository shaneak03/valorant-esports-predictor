export const API_BASE = "http://localhost:5000";

export type Tab = "predictor" | "analysis" | "rankings" | "eda";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface PredictionResult {
  team_a: string;
  team_b: string;
  prob_a: number;
  prob_b: number;
  winner: string;
  elo_a: number;
  elo_b: number;
}

export interface RankingEntry {
  rank: number;
  team: string;
  region: string;
  elo: number;
  elo_change: number;
  wins: number;
  losses: number;
  win_rate: number;
}

export interface MatchEntry {
  date: string;
  opponent: string;
  result: "W" | "L";
}

export interface EloPoint {
  date: string;
  elo: number;
}

export interface TeamStats {
  team: string;
  region: string;
  elo: number;
  wins: number;
  losses: number;
  win_rate: number;
  recent_form: string[];
  recent_matches: MatchEntry[];
  elo_history: EloPoint[];
}

export interface EloDistEntry {
  team: string;
  region: string;
  elo: number;
  wins: number;
  losses: number;
}

export interface RegionWinRate {
  region: string;
  win_rate: number;
  wins: number;
  total: number;
}

// Each object has "date" + one key per Masters London team
export type EloTimelinePoint = { date: string } & Record<string, number>;

export interface MapStat {
  played: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  avg_atk: number | null;
  avg_def: number | null;
  play_freq: number;
}

export interface OverallStatRow {
  team: string;
  region: string;
  series: number;
  win_pct: number;
  last5: string;
  streak: string;
  best_map: string | null;
  best_map_wr: number | null;
  worst_map: string | null;
  worst_map_wr: number | null;
  likely_ban: string | null;
}

export interface H2HEntry {
  wins: number;
  total: number;
  win_rate: number | null;
}

export interface RecentFormPoint {
  index: number;
  rolling_wr: number;
  result: number;
}

export interface EdaFullData {
  teams: string[];
  map_pool: string[];
  map_stats: Record<string, Record<string, MapStat>>;
  overall_stats: OverallStatRow[];
  h2h: Record<string, Record<string, H2HEntry>>;
  recent_form: Record<string, RecentFormPoint[]>;
  min_date: string;
}

export interface StatsData {
  elo_distribution: EloDistEntry[];
  win_rates_by_region: RegionWinRate[];
  elo_timeline: EloTimelinePoint[];
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function getTeams(): Promise<Record<string, string[]>> {
  return apiFetch("/api/teams");
}

export function predict(teamA: string, teamB: string): Promise<PredictionResult> {
  return apiFetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ team_a: teamA, team_b: teamB }),
  });
}

export function getRankings(): Promise<RankingEntry[]> {
  return apiFetch("/api/rankings");
}

export function getTeamStats(name: string): Promise<TeamStats> {
  return apiFetch(`/api/team/${encodeURIComponent(name)}`);
}

export function getStats(): Promise<StatsData> {
  return apiFetch("/api/stats");
}

export function getEdaFull(): Promise<EdaFullData> {
  return apiFetch("/api/eda/full");
}

// ── Constants ─────────────────────────────────────────────────────────────────

export const REGION_COLORS: Record<string, string> = {
  Americas: "#3D9BFF",
  EMEA:     "#FF9F3A",
  Pacific:  "#00C87A",
  China:    "#FF4655",
};

// 12 distinct colours, 3 per region (base, darker, lighter)
export const TEAM_COLORS: Record<string, string> = {
  "G2 Esports":            "#3D9BFF",
  "LEVIATÁN":              "#1A7EDD",
  "NRG":                   "#88C8FF",
  "Team Heretics":         "#FF9F3A",
  "Team Vitality":         "#E07800",
  "FUT Esports":           "#FFB96A",
  "Paper Rex":             "#00C87A",
  "FULL SENSE":            "#009A5C",
  "Global Esports":        "#4DE8A0",
  "EDward Gaming":         "#FF4655",
  "Xi Lai Gaming":         "#CC2535",
  "Dragon Ranger Gaming":  "#FF7F8A",
};
