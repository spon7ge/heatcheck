"""
NBA Player Scoring Prop Predictor
A Markov chain Monte Carlo simulator for over/under points props.

Expected data layout (relative to this file):
    data/s24_pbp.parquet, data/s25_pbp.parquet, data/s26_pbp.parquet
    data/S24.csv,         data/S25.csv,         data/S26.csv
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from helper import (
    DATA_DIR,
    PBP_FILES,
    STATS_FILES,
    load_data,
    get_opponent_def_rating,
)
from markov import (
    RECENT_GAMES_TOP_N,
    N_SIMS_DEFAULT,
    setup_player,
    compute_weights,
    build_transition_matrix,
    build_q1_prior,
    adjust_prior,
    simulate,
)


def main():
    st.set_page_config(page_title="HeatCheck", page_icon="🏀", layout="wide")
    st.title("🏀 HeatCheck")
    st.caption("Markov chain + Monte Carlo over per-quarter scoring states.")

    try:
        merged, stats = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            f"Drop your season files into `{DATA_DIR}/` and refresh.\n\n"
            f"**Play-by-play (parquet):** {', '.join(PBP_FILES)}\n\n"
            f"**Stats (csv):** {', '.join(STATS_FILES)}"
        )
        return

    with st.sidebar:
        st.header("Inputs")
        player = st.selectbox("Player", sorted(merged["PLAYER_NAME"].unique()))
        opponent = st.selectbox("Opponent", sorted(stats["TEAM_ABBREVIATION"].unique()))
        game_total = st.number_input("Game Total", value=230.0, step=0.5)
        team_spread = st.number_input(
            "Team Spread (negative = favorite)", value=-3.5, step=0.5
        )
        line = st.number_input("Points Line", value=24.5, step=0.5)
        n_sims = st.slider("Simulations", 2_000, 50_000, N_SIMS_DEFAULT, step=1_000)
        run = st.button("Run Model", type="primary", use_container_width=True)

    if not run:
        st.info("Configure inputs in the sidebar and hit **Run Model**.")
        return

    player_games = merged[merged["PLAYER_NAME"] == player].copy()
    if len(player_games) < 5:
        st.error(
            f"Only {len(player_games)} games found for {player}. "
            "Need at least ~5 to build a usable model."
        )
        return

    opp_def_rating = get_opponent_def_rating(stats, opponent)
    player_games, edges, pts_map = setup_player(player_games)
    weights = compute_weights(player_games, opp_def_rating, game_total, team_spread)

    T1 = build_transition_matrix(player_games, weights, "sQ1", "sQ2")
    T2 = build_transition_matrix(player_games, weights, "sQ2", "sQ3")
    T3 = build_transition_matrix(player_games, weights, "sQ3", "sQ4")
    prior = adjust_prior(build_q1_prior(player_games, weights), game_total, team_spread)

    totals, p_over, p_under = simulate(
        T1, T2, T3, prior, pts_map, game_total, team_spread, line, n_sims=n_sims
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(Over)", f"{p_over:.1%}")
    c2.metric("P(Under)", f"{p_under:.1%}")
    c3.metric("Projected", f"{totals.mean():.1f} pts")
    hist_cols = ['Q1', 'Q2', 'Q3', 'Q4']
    if 'OT' in player_games.columns:
        hist_cols.append('OT')
    c4.metric("Historical Avg", f"{player_games[hist_cols].sum(axis=1).mean():.1f} pts")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(totals, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(line, color="red", linewidth=2, label=f"Line: {line}")
    ax.axvline(totals.mean(), color="orange", linewidth=2, linestyle="--",
               label=f"Mean: {totals.mean():.1f}")
    ax.set_xlabel("Simulated Points")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{player} vs {opponent} — Simulated Point Distribution")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Model internals"):
        st.write(f"**Upcoming opponent DEF_RATING (last 10 games avg):** {opp_def_rating:.2f}")
        if "OPP_DEF_RATING" in player_games.columns:
            st.write(
                f"**Player's historical mean OPP_DEF_RATING:** "
                f"{player_games['OPP_DEF_RATING'].mean():.2f}"
            )
        st.write(f"**State bin edges (player-specific):** {[round(e, 2) for e in edges[:-1]] + ['∞']}")
        st.write(f"**Median points per state:** {pts_map}")
        st.write(f"**Q1 prior (adjusted):** {prior.round(3).tolist()}")
        st.write(f"**Games used:** {len(player_games)}")
        scoreless_cols = ["Q1", "Q2", "Q3", "Q4"]
        if "OT" in player_games.columns:
            scoreless_cols.append("OT")
        zero_pt_games = int((player_games[scoreless_cols].sum(axis=1) == 0).sum())
        st.write(f"**Scoreless games included:** {zero_pt_games}")

        if "GAME_DATE" in player_games.columns and player_games["GAME_DATE"].notna().any():
            rank = player_games["GAME_DATE"].rank(method="first", ascending=False).values - 1
            top10_weight = weights[rank < RECENT_GAMES_TOP_N].sum()
            st.write(
                f"**Weight on last {RECENT_GAMES_TOP_N} games:** "
                f"{top10_weight:.1%} of total"
            )

        st.write("**Transition matrices** (rows = from state, cols = to state)")
        labels = ["Cold", "Avg", "Hot", "Fire"]
        for name, T in [("Q1→Q2", T1), ("Q2→Q3", T2), ("Q3→Q4", T3)]:
            st.write(f"_{name}_")
            st.dataframe(
                pd.DataFrame(T.round(3), index=labels, columns=labels),
                use_container_width=False,
            )


if __name__ == "__main__":
    main()
