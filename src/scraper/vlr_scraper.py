"""
VLR.gg scraper.

Scrapes match listings, individual match pages, and team match histories.
All HTML responses are cached to disk; real fetches are rate-limited to
1 request per 2 seconds to be respectful to VLR.gg.
"""

import re
import time
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .cache import DiskCache
from .rate_limiter import RateLimiter

log = logging.getLogger(__name__)

BASE_URL = "https://www.vlr.gg"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Tier classification is based purely on tournament name prefix.
#
# All official Riot-run VCT events (franchised leagues, Kickoff, Masters, Champions)
# are named "VCT 20XX: ..." — no other tournament type uses this prefix.
#
# Tier 2 : VCT Masters / Champions / LOCK//IN  (international LAN events)
# Tier 1 : VCT regional leagues (Americas, EMEA, Pacific, China) + Kickoff
# Tier 0 : Everything else (Challengers, Game Changers, EWC qualifiers, etc.)

TIER2_KEYWORDS = ["masters", "champions", "lock//in", "lockin"]

STAGE_KEYWORDS = {
    2: ["grand final", "grand-final"],
    1: ["upper final", "lower final", "upper semi", "lower semi",
        "winner's", "winner", "decider", "elimination", "playoff"],
    0: ["group stage", "swiss stage", "regular phase", "opening", "round", "day"],
}


def _tier_from_tournament(tournament_name: str) -> int:
    """
    Returns 2 for Masters/Champions, 1 for VCT league play, 0 for everything else.
    """
    lower = tournament_name.lower().strip()
    # Must start with "vct" to be a franchised Riot event
    if not lower.startswith("vct"):
        return 0
    # Within VCT events, distinguish Masters/Champions from league play
    if any(k in lower for k in TIER2_KEYWORDS):
        return 2
    return 1


def _stage_from_event(event_name: str) -> int:
    lower = event_name.lower()
    for stage, keywords in STAGE_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return stage
    return 0


def _parse_utc_ts(value: str) -> Optional[str]:
    """
    Parse data-utc-ts attribute to a YYYY-MM-DD string.
    VLR.gg uses "2026-05-22 04:00:00" format (datetime string, not Unix int).
    """
    if not value:
        return None
    value = value.strip()
    # Try datetime string format first: "2026-05-22 04:00:00"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Fallback: try Unix timestamp integer
    try:
        return datetime.utcfromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        pass
    return None


class VLRScraper:
    def __init__(self, cache_dir: str = "data/raw", requests_per_second: float = 0.5):
        self._cache = DiskCache(cache_dir)
        self._limiter = RateLimiter(requests_per_second)
        self._session = requests.Session()
        self._session.headers.update(HEADERS)

    # ------------------------------------------------------------------
    # Low-level fetch
    # ------------------------------------------------------------------

    def _fetch(self, url: str, force: bool = False, retries: int = 3) -> str:
        if not force and self._cache.has(url):
            return self._cache.get(url)
        for attempt in range(1, retries + 1):
            self._limiter.wait()
            try:
                log.info("GET %s", url)
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
                self._cache.set(url, resp.text)
                return resp.text
            except (requests.Timeout, requests.ConnectionError) as e:
                wait = 2 ** attempt
                if attempt < retries:
                    log.warning("Attempt %d failed (%s). Retrying in %ds...", attempt, e, wait)
                    time.sleep(wait)
                else:
                    log.error("All %d attempts failed for %s", retries, url)
                    raise

    def _soup(self, url: str, force: bool = False) -> BeautifulSoup:
        return BeautifulSoup(self._fetch(url, force=force), "lxml")

    # ------------------------------------------------------------------
    # Match listing — paginated
    # ------------------------------------------------------------------

    def get_match_ids_page(self, page: int = 1, force: bool = False) -> list[int]:
        """Return match IDs from a results listing page.

        force=True bypasses the cache — use for recent pages so new matches
        added since the last scrape are discovered.
        """
        url = f"{BASE_URL}/matches/results/?page={page}"
        soup = self._soup(url, force=force)
        ids = []
        for a in soup.select("a.match-item"):
            href = a.get("href", "")
            m = re.match(r"/(\d+)/", href)
            if m:
                ids.append(int(m.group(1)))
        return ids

    def get_all_match_ids(self, max_pages: int = 500, force_pages: int = 5) -> list[int]:
        """Paginate through results pages and collect all match IDs.

        force_pages: re-fetch the first N listing pages even if cached.
        New matches always appear at the front, so forcing a small window
        guarantees you pick up anything added since the last scrape.
        """
        all_ids = []
        for page in range(1, max_pages + 1):
            force = (page <= force_pages)
            ids = self.get_match_ids_page(page, force=force)
            if not ids:
                log.info("No more matches at page %d, stopping.", page)
                break
            all_ids.extend(ids)
            log.info("Page %d: %d matches collected so far", page, len(all_ids))
        return all_ids

    # ------------------------------------------------------------------
    # Individual match page
    # ------------------------------------------------------------------

    def get_match(self, match_id: int) -> Optional[dict]:
        """
        Parse a single match page and return a dict with series-level stats.
        Returns None if the match is incomplete or unparseable.
        """
        url = f"{BASE_URL}/{match_id}/"
        try:
            soup = self._soup(url)
        except Exception as e:
            log.warning("Failed to fetch match %d: %s", match_id, e)
            return None

        # --- Teams (team A = mod-1 / left, team B = mod-2 / right) ---
        team_names = [
            el.get_text(strip=True)
            for el in soup.select(".match-header-link-name .wf-title-med")
        ]
        if len(team_names) < 2:
            return None

        # --- Series score ---
        # Use winner/loser spans which are reliable across VLR layouts.
        # Then determine which team (A=left/mod-1 or B=right/mod-2) is the winner
        # by comparing against per-map map-wins.
        winner_span = soup.select_one(".match-header-vs-score-winner")
        loser_span = soup.select_one(".match-header-vs-score-loser")
        if not winner_span or not loser_span:
            return None
        try:
            win_score = int(winner_span.get_text(strip=True))
            lose_score = int(loser_span.get_text(strip=True))
        except ValueError:
            return None

        if win_score == lose_score:
            return None  # tie / incomplete

        # Determine winner side (0=team_a won, 1=team_b won) from per-map blocks
        maps_data = self._parse_maps(soup)
        team_a_map_wins = sum(1 for m in maps_data if m.get("winner_side") == "a")
        team_b_map_wins = sum(1 for m in maps_data if m.get("winner_side") == "b")

        if team_a_map_wins == 0 and team_b_map_wins == 0:
            # Fallback: assume the winner score belongs to team_a
            # (VLR historically shows left-team score first)
            score_a, score_b = win_score, lose_score
        elif team_a_map_wins >= team_b_map_wins:
            score_a, score_b = win_score, lose_score
        else:
            score_a, score_b = lose_score, win_score

        winner = 0 if score_a > score_b else 1

        # --- Date ---
        date_el = soup.select_one(".match-header-date .moment-tz-convert")
        match_date = None
        if date_el:
            match_date = _parse_utc_ts(date_el.get("data-utc-ts", ""))

        # --- Event / Tournament ---
        event_series_el = soup.select_one(".match-header-event-series")
        event_name = " ".join(event_series_el.get_text().split()) if event_series_el else ""

        # Tournament name is in the bold div inside the event link
        # Structure: <a class="match-header-event"><img/><div><div style="font-weight:700">NAME</div>...</div></a>
        tournament_name = ""
        bold_el = soup.select_one(".match-header-event div[style*='font-weight']")
        if bold_el:
            tournament_name = " ".join(bold_el.get_text().split())

        tournament_tier = _tier_from_tournament(tournament_name)
        bracket_stage = _stage_from_event(event_name)

        # --- Team IDs from href links ---
        team_links = soup.select(".match-header-link")
        team_ids = []
        for link in team_links[:2]:
            href = link.get("href", "")
            m = re.match(r"/team/(\d+)/", href)
            team_ids.append(int(m.group(1)) if m else None)

        return {
            "match_id": match_id,
            "date": match_date,
            "team_a": team_names[0],
            "team_b": team_names[1],
            "team_a_id": team_ids[0] if len(team_ids) > 0 else None,
            "team_b_id": team_ids[1] if len(team_ids) > 1 else None,
            "score_a": score_a,
            "score_b": score_b,
            "maps_played": score_a + score_b,
            "winner": winner,
            "event": event_name,
            "tournament": tournament_name,
            "tournament_tier": tournament_tier,
            "bracket_stage": bracket_stage,
            "maps": maps_data,
        }

    # ------------------------------------------------------------------
    # Per-map stats
    # ------------------------------------------------------------------

    def _parse_maps(self, soup: BeautifulSoup) -> list[dict]:
        """Extract per-map round stats from a match page."""
        maps = []
        for map_el in soup.select(".vm-stats-game"):
            # --- Map name ---
            map_name_el = map_el.select_one(".map span")
            if not map_name_el:
                map_name_el = map_el.select_one(".map div")
            map_name = map_name_el.get_text(strip=True) if map_name_el else "unknown"
            # Strip "PICK"/"BAN" suffixes that VLR appends
            map_name = re.sub(r"\s*(PICK|BAN|REMAIN|LEFT)\s*$", "", map_name, flags=re.I).strip()
            if not map_name or map_name.lower() in ("map", ""):
                continue

            # --- Round scores (left team = team A, right team = team B) ---
            # Left team: .team (no mod-right), right team: .team.mod-right
            left_team = map_el.select_one(".vm-stats-game-header .team:not(.mod-right)")
            right_team = map_el.select_one(".vm-stats-game-header .team.mod-right")

            rounds_a = self._parse_team_score(left_team)
            rounds_b = self._parse_team_score(right_team)
            total_rounds = rounds_a + rounds_b

            if total_rounds == 0:
                continue

            # Determine map winner
            winner_side = "a" if rounds_a > rounds_b else "b"

            # --- Side stats: T-side and CT-side rounds ---
            # Structure: <span class="mod-t">11</span> / <span class="mod-ct">2</span>
            atk_a, def_a = self._parse_side_rounds(left_team)
            atk_b, def_b = self._parse_side_rounds(right_team)

            # --- Pistol rounds (rounds 1 and 13) ---
            pistol_win_rate_a, pistol_win_rate_b = self._parse_pistol_rounds(map_el)

            # --- Max consecutive round win streaks ---
            max_streak_a, max_streak_b = self._parse_round_streaks(map_el)

            maps.append({
                "map": map_name,
                "rounds_a": rounds_a,
                "rounds_b": rounds_b,
                "total_rounds": total_rounds,
                "winner_side": winner_side,
                "round_win_rate_a": rounds_a / total_rounds,
                "round_win_rate_b": rounds_b / total_rounds,
                "atk_win_rate_a": atk_a / total_rounds if total_rounds else 0.0,
                "def_win_rate_a": def_a / total_rounds if total_rounds else 0.0,
                "atk_win_rate_b": atk_b / total_rounds if total_rounds else 0.0,
                "def_win_rate_b": def_b / total_rounds if total_rounds else 0.0,
                "pistol_win_rate_a": pistol_win_rate_a,
                "pistol_win_rate_b": pistol_win_rate_b,
                "max_round_streak_a": max_streak_a,
                "max_round_streak_b": max_streak_b,
            })
        return maps

    @staticmethod
    def _parse_team_score(team_el) -> int:
        """Extract total round score from a team div (the .score element)."""
        if team_el is None:
            return 0
        score_el = team_el.select_one(".score")
        if not score_el:
            return 0
        try:
            return int(score_el.get_text(strip=True))
        except ValueError:
            return 0

    @staticmethod
    def _parse_side_rounds(team_el) -> tuple[int, int]:
        """
        Return (attack_rounds, defense_rounds) for a team from their T/CT span counts.
        Structure: <span class="mod-t">N</span> / <span class="mod-ct">M</span>
        """
        if team_el is None:
            return 0, 0
        t_el = team_el.select_one("span.mod-t")
        ct_el = team_el.select_one("span.mod-ct")
        try:
            t_rounds = int(t_el.get_text(strip=True)) if t_el else 0
            ct_rounds = int(ct_el.get_text(strip=True)) if ct_el else 0
            return t_rounds, ct_rounds  # T = attack, CT = defense
        except ValueError:
            return 0, 0

    @staticmethod
    def _parse_pistol_rounds(map_el) -> tuple[float, float]:
        """
        Parse pistol round win rates from round-by-round icons.
        Round 1 and round 13 (0-indexed: positions 0 and 12 in the round list).
        A round belongs to team A if the top .rnd-sq has mod-win.
        """
        round_cols = map_el.select(".vlr-rounds-row-col")
        # First col is the label col (team names), actual rounds start at index 1
        round_cols = [c for c in round_cols if c.select_one(".rnd-num")]

        if not round_cols:
            return 0.0, 0.0

        pistol_indices = [i for i in (0, 12) if i < len(round_cols)]
        if not pistol_indices:
            return 0.0, 0.0

        wins_a = 0
        total = 0
        for idx in pistol_indices:
            col = round_cols[idx]
            # Two .rnd-sq per column: first = team A row, second = team B row
            rnd_sqs = col.select(".rnd-sq")
            if len(rnd_sqs) >= 2:
                total += 1
                if "mod-win" in rnd_sqs[0].get("class", []):
                    wins_a += 1

        if total == 0:
            return 0.0, 0.0
        rate_a = wins_a / total
        return rate_a, 1.0 - rate_a

    @staticmethod
    def _parse_round_streaks(map_el) -> tuple[int, int]:
        """
        Parse full round sequence and return:
          (max_round_streak_a, max_round_streak_b)
        where streak = longest consecutive round wins for that team.
        """
        round_cols = [c for c in map_el.select(".vlr-rounds-row-col")
                      if c.select_one(".rnd-num")]
        if not round_cols:
            return 0, 0

        seq_a = []
        for col in round_cols:
            rnd_sqs = col.select(".rnd-sq")
            if len(rnd_sqs) >= 2:
                seq_a.append(1 if "mod-win" in rnd_sqs[0].get("class", []) else 0)

        if not seq_a:
            return 0, 0

        seq_b = [1 - r for r in seq_a]

        def max_streak(seq):
            best = cur = 0
            for r in seq:
                cur = cur + 1 if r == 1 else 0
                best = max(best, cur)
            return best

        return max_streak(seq_a), max_streak(seq_b)

    # ------------------------------------------------------------------
    # Team history
    # ------------------------------------------------------------------

    def get_team_match_ids(self, team_id: int, max_pages: int = 10) -> list[int]:
        """Return match IDs from a team's match history pages."""
        ids = []
        for page in range(1, max_pages + 1):
            url = f"{BASE_URL}/team/{team_id}/matches/?page={page}"
            try:
                soup = self._soup(url)
            except Exception as e:
                log.warning("Team %d page %d failed: %s", team_id, page, e)
                break
            page_ids = []
            for a in soup.select("a.match-item"):
                href = a.get("href", "")
                m = re.match(r"/(\d+)/", href)
                if m:
                    page_ids.append(int(m.group(1)))
            if not page_ids:
                break
            ids.extend(page_ids)
        return ids
