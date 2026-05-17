"""
Data loading, preprocessing, and opponent lookup utilities.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
PBP_FILES = ["s24_pbp.parquet", "s25_pbp.parquet", "s26_pbp.parquet"]
STATS_FILES = ["S24.csv", "S25.csv", "S26.csv"]


@st.cache_data(show_spinner="Loading and processing data…")
def load_data():
    """Load all seasons, build per-quarter scoring, and join with game context."""
    missing = [f for f in PBP_FILES + STATS_FILES if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {DATA_DIR}/: {', '.join(missing)}"
        )

    pbp = pd.concat([pd.read_parquet(DATA_DIR / f) for f in PBP_FILES], ignore_index=True)
    stats = pd.concat([pd.read_csv(DATA_DIR / f) for f in STATS_FILES], ignore_index=True)

    # Normalize gameId for joining (strip leading zeros)
    pbp["gameId_clean"] = pbp["gameId"].astype(str).str.lstrip("0")
    stats["GAME_ID"] = stats["GAME_ID"].astype(str).str.lstrip("0")

    # Attach opponent + opponent DEF_RATING to every player-game row.
    stats = attach_opponent_def_rating(stats)

    q_pivot = build_quarter_scoring(pbp)

    join_cols = [
        "GAME_ID", "PLAYER_ID", "PLAYER_NAME", "SEASON_YEAR",
        "TEAM_ABBREVIATION", "OPP_ABBREVIATION",
        "DEF_RATING", "OPP_DEF_RATING",
        "GAME_TOTAL", "TEAM_SPREAD", "STARTING", "GAME_DATE",
    ]
    join_cols = [c for c in join_cols if c in stats.columns]

    # Left-merge FROM stats so games where the player played but didn't score
    # are still present (with zero-filled quarter columns). The old inner
    # merge silently dropped scoreless appearances.
    merged = stats[join_cols].merge(
        q_pivot, on=["GAME_ID", "PLAYER_ID"], how="left"
    )
    for q in ["Q1", "Q2", "Q3", "Q4", "OT"]:
        if q not in merged.columns:
            merged[q] = 0
        merged[q] = merged[q].fillna(0).astype(int)

    if "GAME_DATE" in merged.columns:
        merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"], errors="coerce")

    return merged, stats


def attach_opponent_def_rating(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Player gamelogs carry the player's OWN team DEF_RATING for the game.
    For matchup-similarity we want the OPPONENT's defensive rating.

    1. Average DEF_RATING across all players on a team in a given game →
       team-game DEF_RATING.
    2. For each player-row, look up the OTHER team in the same game and
       attach that team's team-game DEF_RATING as OPP_DEF_RATING.
    """
    team_game_def = (
        stats.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)["DEF_RATING"]
        .mean()
        .rename(columns={"DEF_RATING": "TEAM_DEF_RATING"})
    )

    # Pair each (game, team) with the opposing team in that game.
    team_pairs = stats[["GAME_ID", "TEAM_ABBREVIATION"]].drop_duplicates()
    team_pairs = team_pairs.merge(team_pairs, on="GAME_ID", suffixes=("", "_opp"))
    team_pairs = team_pairs[
        team_pairs["TEAM_ABBREVIATION"] != team_pairs["TEAM_ABBREVIATION_opp"]
    ].rename(columns={"TEAM_ABBREVIATION_opp": "OPP_ABBREVIATION"})

    stats = stats.merge(team_pairs, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    stats = stats.merge(
        team_game_def.rename(
            columns={
                "TEAM_ABBREVIATION": "OPP_ABBREVIATION",
                "TEAM_DEF_RATING": "OPP_DEF_RATING",
            }
        ),
        on=["GAME_ID", "OPP_ABBREVIATION"],
        how="left",
    )
    return stats


def build_quarter_scoring(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-player, per-game Q1–Q4 and OT point totals from play-by-play."""
    fg = pbp[pbp["actionType"] == "Made Shot"]
    fg_pts = (
        fg.groupby(["gameId_clean", "personId", "period"])["shotValue"]
        .sum()
        .reset_index()
        .rename(columns={"shotValue": "pts"})
    )

    ft = pbp[
        (pbp["actionType"] == "Free Throw")
        & (~pbp["description"].astype(str).str.startswith("MISS"))
    ].copy()
    ft["pts"] = 1
    ft_pts = (
        ft.groupby(["gameId_clean", "personId", "period"])["pts"].sum().reset_index()
    )

    all_pts = pd.concat([fg_pts, ft_pts], ignore_index=True)
    q_pts = (
        all_pts.groupby(["gameId_clean", "personId", "period"])["pts"].sum().reset_index()
    )

    # Separate regular quarters and overtime
    q_pts_regular = q_pts[q_pts["period"] <= 4].copy()
    q_pts_ot = q_pts[q_pts["period"] >= 5].copy()

    # Aggregate all OT periods into a single "OT" entry
    if not q_pts_ot.empty:
        q_pts_ot["period"] = "OT"
        q_pts_ot = q_pts_ot.groupby(["gameId_clean", "personId", "period"])["pts"].sum().reset_index()

    # Combine regular quarters with OT
    q_pts_combined = pd.concat([q_pts_regular, q_pts_ot], ignore_index=True)

    q_pivot = q_pts_combined.pivot_table(
        index=["gameId_clean", "personId"],
        columns="period",
        values="pts",
        fill_value=0,
    ).reset_index()

    for q in [1, 2, 3, 4]:
        if q not in q_pivot.columns:
            q_pivot[q] = 0

    col_order = ["gameId_clean", "personId", 1, 2, 3, 4]
    if "OT" in q_pivot.columns:
        col_order.append("OT")

    q_pivot = q_pivot[col_order]
    new_cols = ["GAME_ID", "PLAYER_ID", "Q1", "Q2", "Q3", "Q4"]
    if "OT" in q_pivot.columns:
        new_cols.append("OT")
    q_pivot.columns = new_cols
    return q_pivot


def get_opponent_def_rating(stats_df: pd.DataFrame, opp_abbrev: str, recent_n: int = 10) -> float:
    """Mean DEF_RATING of the opponent team across its most recent `recent_n` games."""
    team_games = stats_df[stats_df["TEAM_ABBREVIATION"] == opp_abbrev].copy()
    if team_games.empty:
        return float(stats_df["DEF_RATING"].mean())

    if "GAME_DATE" in team_games.columns:
        team_games["GAME_DATE"] = pd.to_datetime(team_games["GAME_DATE"], errors="coerce")
        per_game = (
            team_games.groupby("GAME_ID", as_index=False)
            .agg(DEF_RATING=("DEF_RATING", "mean"),
                 GAME_DATE=("GAME_DATE", "max"))
            .sort_values("GAME_DATE", ascending=False)
        )
    else:
        per_game = team_games.groupby("GAME_ID", as_index=False)["DEF_RATING"].mean()

    per_game = per_game.head(recent_n)
    if per_game.empty:
        return float(stats_df["DEF_RATING"].mean())
    return float(per_game["DEF_RATING"].mean())
