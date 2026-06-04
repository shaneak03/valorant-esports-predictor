"""Inspect real VLR.gg HTML structure to verify CSS selectors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.vlr_scraper import VLRScraper

scraper = VLRScraper(cache_dir="data/raw")
soup = scraper._soup("https://www.vlr.gg/681753/")

# 1. Score selectors
print("=== SCORE ELEMENTS (.match-header-vs-score .js-spoiler) ===")
score_els = soup.select(".match-header-vs-score .js-spoiler")
print(f"Found {len(score_els)} elements:")
for el in score_els:
    classes = " ".join(el.get("class", []))
    text = el.get_text(strip=True)[:80]
    print(f"  classes=[{classes}]  text={repr(text)}")

print()
print("=== WINNER / LOSER SPANS ===")
for span in soup.select(".match-header-vs-score-winner, .match-header-vs-score-loser"):
    cls = " ".join(span.get("class", []))
    print(f"  [{cls}] -> {repr(span.get_text(strip=True))}")

print()
print("=== DATE ELEMENT ===")
date_el = soup.select_one(".match-header-date .moment-tz-convert")
if date_el:
    print(f"  data-utc-ts = {repr(date_el.get('data-utc-ts'))}")

print()
print("=== VM-STATS-GAME BLOCKS ===")
maps = soup.select(".vm-stats-game")
print(f"Found {len(maps)} map blocks")
if maps:
    # Print the first 2500 chars of the first map block to see its structure
    print(str(maps[0])[:2500])
