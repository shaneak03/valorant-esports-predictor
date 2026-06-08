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
  Americas: "#FF7A00",
  EMEA:     "#B6FF00",
  Pacific:  "#00E5FF",
  China:    "#FF3B3B",
};

// Each team gets a distinct hue — within a region the 3 teams use clearly
// different parts of the colour wheel so lines are readable on the Elo timeline.
export const TEAM_COLORS: Record<string, string> = {
  // Americas — base orange, + gold, + deep pink
  "G2 Esports":            "#FF7A00",
  "LEVIATÁN":              "#FFD700",
  "NRG":                   "#FF2D78",
  // EMEA — base lime, + violet, + sky blue
  "Team Heretics":         "#B6FF00",
  "Team Vitality":         "#BF5FFF",
  "FUT Esports":           "#00BFFF",
  // Pacific — base cyan, + spring green, + blue
  "Paper Rex":             "#00E5FF",
  "FULL SENSE":            "#00FF7A",
  "Global Esports":        "#4D80FF",
  // China — base red, + orchid pink, + amber
  "EDward Gaming":         "#FF3B3B",
  "Xi Lai Gaming":         "#FF7FD4",
  "Dragon Ranger Gaming":  "#FF9500",
};
