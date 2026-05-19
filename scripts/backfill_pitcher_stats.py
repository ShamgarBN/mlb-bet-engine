"""Re-extract per-pitcher stats from cached boxscore payloads.

Since every boxscore we fetched is already in the HTTP cache, re-running
``fetch_boxscore`` is instant (cache hit). This script:

  1. Lists every game_pk that already has a team_boxscore row but no
     entries in ``pitcher_game_stats``.
  2. Re-fetches the boxscore payload (cached) and extracts per-pitcher
     stats via ``normalize_pitcher_stats``.
  3. Bulk upserts.

This adds the highest-impact missing feature (starting pitcher rolling
stats) without any new network calls.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from mlb_model.data.sources import mlb_statsapi
from mlb_model.data.warehouse import init_schema, query, upsert_dataframe
from mlb_model.logging import configure_logging, get_logger

log = get_logger("backfill_pitcher_stats")


def main() -> None:
    configure_logging()
    init_schema()

    pks_df = query(
        """
        SELECT DISTINCT b.game_pk
        FROM team_boxscores b
        LEFT JOIN pitcher_game_stats p ON p.game_pk = b.game_pk
        WHERE p.game_pk IS NULL
        ORDER BY b.game_pk
        """
    )
    pks = pks_df["game_pk"].astype(int).tolist()
    log.info("pitcher_stats.backfill.start", pending=len(pks))
    if not pks:
        return

    rows = []

    def _one(pk: int) -> pd.DataFrame:
        try:
            payload = mlb_statsapi.fetch_boxscore(pk)
        except Exception as exc:  # noqa: BLE001
            log.warning("boxscore.cache.miss", game_pk=pk, error=str(exc))
            return pd.DataFrame()
        return mlb_statsapi.normalize_pitcher_stats(pk, payload)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_one, pk): pk for pk in pks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            df = fut.result()
            if not df.empty:
                rows.append(df)
            if done % 500 == 0:
                log.info("pitcher_stats.progress", done=done, total=len(pks))

    if not rows:
        log.warning("pitcher_stats.no_rows")
        return
    big = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["game_pk", "pitcher_id"], keep="last"
    )
    n = upsert_dataframe(big, "pitcher_game_stats", key_columns=["game_pk", "pitcher_id"])
    log.info("pitcher_stats.backfill.done", rows=n)


if __name__ == "__main__":
    main()
