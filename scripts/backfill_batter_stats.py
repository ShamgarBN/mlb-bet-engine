"""Re-extract per-batter and per-pitcher stats from cached boxscores.

Every boxscore we fetched is in the HTTP cache, so this is fast: it just
parses the cached JSON in parallel and bulk-upserts batter + pitcher rows.

Run this after schema changes that add new tables sourced from boxscores.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from mlb_model.data.sources import mlb_statsapi
from mlb_model.data.warehouse import init_schema, query, upsert_dataframe
from mlb_model.logging import configure_logging, get_logger

log = get_logger("backfill_batter_stats")


def main() -> None:
    configure_logging()
    init_schema()

    pks_df = query(
        """
        SELECT DISTINCT b.game_pk
        FROM team_boxscores b
        LEFT JOIN batter_game_stats bg ON bg.game_pk = b.game_pk
        WHERE bg.game_pk IS NULL
        ORDER BY b.game_pk
        """
    )
    pks = pks_df["game_pk"].astype(int).tolist()
    log.info("batter_stats.backfill.start", pending=len(pks))
    if not pks:
        return

    batter_rows: list[pd.DataFrame] = []
    pitcher_rows: list[pd.DataFrame] = []

    def _one(pk: int):
        try:
            payload = mlb_statsapi.fetch_boxscore(pk)
        except Exception as exc:  # noqa: BLE001
            log.warning("boxscore.cache.miss", game_pk=pk, error=str(exc))
            return None
        return (
            mlb_statsapi.normalize_batter_stats(pk, payload),
            mlb_statsapi.normalize_pitcher_stats(pk, payload),
        )

    def _flush() -> None:
        nonlocal batter_rows, pitcher_rows
        if batter_rows:
            big = pd.concat(batter_rows, ignore_index=True).drop_duplicates(
                subset=["game_pk", "batter_id"], keep="last"
            )
            upsert_dataframe(big, "batter_game_stats", key_columns=["game_pk", "batter_id"])
        if pitcher_rows:
            big = pd.concat(pitcher_rows, ignore_index=True).drop_duplicates(
                subset=["game_pk", "pitcher_id"], keep="last"
            )
            upsert_dataframe(big, "pitcher_game_stats", key_columns=["game_pk", "pitcher_id"])
        batter_rows = []
        pitcher_rows = []

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_one, pk): pk for pk in pks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result is None:
                continue
            b_df, p_df = result
            if not b_df.empty:
                batter_rows.append(b_df)
            if not p_df.empty:
                pitcher_rows.append(p_df)
            if done % 1000 == 0:
                log.info("batter_stats.progress", done=done, total=len(pks))
                _flush()

    _flush()
    log.info("batter_stats.backfill.done")


if __name__ == "__main__":
    main()
