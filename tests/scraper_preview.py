"""
Quick preview: scrape 1 results page, then parse the first 3 matches in full.
Shows exactly what get_match() returns before committing to a full scrape.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.scraper.vlr_scraper import VLRScraper

scraper = VLRScraper(cache_dir="data/raw", requests_per_second=0.5)

# Step 1 — get match IDs from page 1 of results
print("\n" + "="*60)
print("STEP 1: Collecting match IDs from page 1 of results...")
print("="*60)
ids = scraper.get_match_ids_page(page=1)
print(f"Found {len(ids)} match IDs: {ids}\n")

# Step 2 — parse first 3 matches in full
print("="*60)
print("STEP 2: Parsing first 3 match pages in full...")
print("="*60)

parsed = []
for mid in ids[:3]:
    print(f"\n--- Match ID: {mid} ---")
    match = scraper.get_match(mid)
    if match is None:
        print("  (skipped — incomplete/no data)")
        continue
    parsed.append(match)
    print(json.dumps(match, indent=2))

print("\n" + "="*60)
print(f"Summary: {len(parsed)} / 3 matches parsed successfully")
if parsed:
    m = parsed[0]
    print(f"\nSample: {m['team_a']} {m['score_a']}–{m['score_b']} {m['team_b']}")
    print(f"  Date:  {m['date']}")
    print(f"  Event: {m['event']}")
    print(f"  Tier:  {m['tournament_tier']}")
    print(f"  Maps played: {len(m['maps'])}")
    for mp in m['maps']:
        print(f"    {mp['map']:10s}  {mp['rounds_a']}–{mp['rounds_b']}  "
              f"(atk_a={mp['atk_win_rate_a']:.2f} def_a={mp['def_win_rate_a']:.2f} "
              f"pistol_a={mp['pistol_win_rate_a']:.2f})")
print("="*60)
