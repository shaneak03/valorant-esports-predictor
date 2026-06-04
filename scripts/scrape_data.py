"""
CLI: Scrape VLR.gg match data and save raw match dicts to a JSON file.

Usage:
    python scripts/scrape_data.py
    python scripts/scrape_data.py --max-pages 500 --min-tier 1

If interrupted, re-run the exact same command — it resumes from the checkpoint
and skips any match pages already cached to disk.

Run build_dataset.py afterward to produce training-ready splits.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.vlr_scraper import VLRScraper
from src.data.team_filter import is_franchised_match, involves_masters_london_team, resolve_team

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT = "data/processed/scrape_checkpoint.json"


def load_checkpoint(path: str) -> tuple[list, set]:
    cp = Path(path)
    if cp.exists():
        with open(cp) as f:
            data = json.load(f)
        matches = data.get("matches", [])
        done_ids = set(data.get("done_ids", []))
        log.info("Resuming from checkpoint: %d matches saved, %d match IDs already processed",
                 len(matches), len(done_ids))
        return matches, done_ids
    return [], set()


def save_checkpoint(path: str, matches: list, done_ids: set) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"matches": matches, "done_ids": list(done_ids)}, f)


def main():
    parser = argparse.ArgumentParser(description="Scrape VLR.gg match data")
    parser.add_argument("--max-pages", type=int, default=500,
                        help="Max result listing pages to paginate (~20 match IDs per page)")
    parser.add_argument("--cache-dir", default="data/raw",
                        help="Directory for cached HTML (already-fetched pages are free)")
    parser.add_argument("--out", default="data/processed/raw_matches.json",
                        help="Output JSON path")
    parser.add_argument("--rate", type=float, default=0.5,
                        help="Requests per second (default 0.5 = 1 req/2s)")
    parser.add_argument("--min-tier", type=int, default=0, choices=[0, 1, 2],
                        help="Min tournament tier: 0=all (default)  1=VCT leagues+intl  2=Masters/Champions only")
    parser.add_argument("--franchised-only", action="store_true",
                        help="Keep only matches where both teams are franchised VCT teams.")
    parser.add_argument("--masters-london-any", action="store_true",
                        help="Keep only matches involving at least one Masters London qualifier.")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="Checkpoint file path for resuming interrupted scrapes")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N match pages processed (default 50)")
    parser.add_argument("--force-listing-pages", type=int, default=5, metavar="N",
                        help="Re-fetch the first N listing pages even if cached (default 5). "
                             "New matches always appear on the first few pages, so this is "
                             "how you pick up matches added since the last scrape.")
    args = parser.parse_args()

    scraper = VLRScraper(cache_dir=args.cache_dir, requests_per_second=args.rate)

    # Load checkpoint so we can resume if interrupted
    matches, done_ids = load_checkpoint(args.checkpoint)

    log.info("Collecting match IDs (max %d listing pages, forcing first %d)...",
             args.max_pages, args.force_listing_pages)
    match_ids = scraper.get_all_match_ids(
        max_pages=args.max_pages,
        force_pages=args.force_listing_pages,
    )
    log.info("Found %d match IDs total.", len(match_ids))

    # Filter out already-processed IDs
    remaining = [mid for mid in match_ids if str(mid) not in done_ids]
    log.info("%d match IDs remaining to process (%d already done).",
             len(remaining), len(done_ids))

    skipped_tier = 0
    for i, mid in enumerate(remaining):
        try:
            match = scraper.get_match(mid)
        except Exception as e:
            log.warning("Skipping match %d due to error: %s", mid, e)
            done_ids.add(str(mid))
            continue

        done_ids.add(str(mid))

        if match is None:
            continue
        if match["tournament_tier"] < args.min_tier:
            skipped_tier += 1
            continue

        if args.franchised_only and not is_franchised_match(match["team_a"], match["team_b"]):
            continue
        if args.masters_london_any and not involves_masters_london_team(match["team_a"], match["team_b"]):
            continue

        matches.append(match)

        if (i + 1) % args.save_every == 0:
            save_checkpoint(args.checkpoint, matches, done_ids)
            log.info("Progress: %d/%d processed | %d kept | %d skipped low-tier",
                     i + 1, len(remaining), len(matches), skipped_tier)

    # Final save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(matches, f, indent=2)

    # Clean up checkpoint now that we're done
    cp = Path(args.checkpoint)
    if cp.exists():
        cp.unlink()

    log.info("Done. Saved %d matches to %s (skipped %d below tier %d)",
             len(matches), out_path, skipped_tier, args.min_tier)


if __name__ == "__main__":
    main()
