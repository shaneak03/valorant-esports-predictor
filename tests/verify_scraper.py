import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.vlr_scraper import VLRScraper

s = VLRScraper(cache_dir="data/raw")

for mid in [675204, 681753, 680956]:
    m = s.get_match(mid)
    ta, tb = m["team_a"], m["team_b"]
    sa, sb = m["score_a"], m["score_b"]
    print(f"{ta} {sa}-{sb} {tb}")
    print(f"  tournament : {m['tournament']}")
    print(f"  event      : {m['event']}")
    print(f"  date       : {m['date']}  tier={m['tournament_tier']}  stage={m['bracket_stage']}")
    for mp in m["maps"]:
        name = mp["map"]
        ra, rb = mp["rounds_a"], mp["rounds_b"]
        rwr = mp["round_win_rate_a"]
        atk = mp["atk_win_rate_a"]
        df  = mp["def_win_rate_a"]
        pis = mp["pistol_win_rate_a"]
        print(f"  {name:10s} {ra}-{rb}  rwr_a={rwr:.2f}  atk_a={atk:.2f}  def_a={df:.2f}  pistol_a={pis:.2f}")
    print()
