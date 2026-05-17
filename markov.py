"""
Markov chain Monte Carlo engine for NBA per-quarter scoring simulation.
"""

import numpy as np
import pandas as pd

RECENT_GAMES_TOP_N = 10
RECENCY_DECAY_SCALE = 15.0
NEUTRAL_TOTAL = 230.0
N_SIMS_DEFAULT = 15_000
LAPLACE_ALPHA = 0.5


def setup_player(player_games: pd.DataFrame):
    """Bin each quarter into Cold/Average/Hot/On-Fire and build a points map."""
    player_games = player_games.copy()
    all_q_pts = pd.concat(
        [player_games["Q1"], player_games["Q2"], player_games["Q3"], player_games["Q4"]]
    ).values

    q25, q50, q75 = np.percentile(all_q_pts, [25, 50, 75])
    edges = sorted({0.0, float(q25), float(q50), float(q75)})
    if len(edges) < 4:
        base = float(q50) if q50 > 0 else 5.0
        edges = [0.0, base * 0.5, base, base * 1.5]
    edges = edges + [np.inf]

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        player_games[f"s{q}"] = (
            pd.cut(
                player_games[q],
                bins=edges,
                labels=[0, 1, 2, 3],
                right=False,
                include_lowest=True,
            )
            .astype("Int64")
            .fillna(0)
            .astype(int)
        )

    all_labeled = pd.concat(
        [
            player_games[["Q1"]].rename(columns={"Q1": "pts"}).assign(state=player_games["sQ1"]),
            player_games[["Q2"]].rename(columns={"Q2": "pts"}).assign(state=player_games["sQ2"]),
            player_games[["Q3"]].rename(columns={"Q3": "pts"}).assign(state=player_games["sQ3"]),
            player_games[["Q4"]].rename(columns={"Q4": "pts"}).assign(state=player_games["sQ4"]),
        ],
        ignore_index=True,
    )
    median_by_state = all_labeled.groupby("state")["pts"].median().to_dict()

    fallback = {0: 0.0, 1: 3.5, 2: 7.5, 3: 12.5}
    pts_map = {s: float(median_by_state.get(s, fallback[s])) for s in range(4)}

    return player_games, edges, pts_map


def compute_weights(player_games, opp_def_rating, game_total, team_spread,
                    top_n: int = RECENT_GAMES_TOP_N, decay_scale: float = RECENCY_DECAY_SCALE):
    """
    Combine context similarity (Gaussian kernel over OPP_DEF_RATING/
    GAME_TOTAL/TEAM_SPREAD) with game-rank recency: the most recent
    `top_n` games get full weight, older games decay exponentially.
    """
    candidate = ["OPP_DEF_RATING", "GAME_TOTAL", "TEAM_SPREAD"]
    context_cols = [c for c in candidate if c in player_games.columns]
    new_vals = {"OPP_DEF_RATING": opp_def_rating,
                "GAME_TOTAL": game_total,
                "TEAM_SPREAD": team_spread}

    if not context_cols:
        sim_w = np.ones(len(player_games))
    else:
        X = player_games[context_cols].values.astype(float)
        x_new = np.array([new_vals[c] for c in context_cols], dtype=float)

        # Robustness: fill any missing context values with the column mean.
        col_means = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), col_means, X)

        mu, sigma = X.mean(0), X.std(0) + 1e-8
        X_norm = (X - mu) / sigma
        x_norm = (x_new - mu) / sigma

        dists = np.linalg.norm(X_norm - x_norm, axis=1)
        sim_w = np.exp(-0.5 * (dists / 1.2) ** 2)

    # Game-rank recency: 0 = most recent.
    if "GAME_DATE" in player_games.columns and player_games["GAME_DATE"].notna().any():
        rank = player_games["GAME_DATE"].rank(method="first", ascending=False).values - 1
    else:
        season_order = (player_games["SEASON_YEAR"]
                        .map({"2025-26": 0, "2024-25": 1, "2023-24": 2})
                        .fillna(2).values)
        rank = season_order * 25

    rec_w = np.where(rank < top_n, 1.0, np.exp(-(rank - top_n) / decay_scale))

    final_w = sim_w * rec_w
    total = final_w.sum()
    if total == 0:
        return np.ones(len(player_games)) / len(player_games)
    return final_w / total


def build_transition_matrix(player_games, weights, from_q, to_q,
                            alpha: float = LAPLACE_ALPHA):
    """
    Weighted-count transition matrix with Laplace smoothing.

    Weights from compute_weights are normalized to sum to 1 across the whole
    sample, so a raw `T` would sum to 1 too — and adding alpha=0.5 to each
    of 16 cells would drown the data (~89% prior). We scale weights by N
    first so they look like effective counts; then `alpha` is interpretable
    as a pseudo-game count per cell, and smoothing scales sensibly with
    sample size.
    """
    N = len(player_games)
    T = np.zeros((4, 4))
    from_states = player_games[from_q].values.astype(int)
    to_states = player_games[to_q].values.astype(int)
    for i in range(N):
        T[from_states[i], to_states[i]] += weights[i]

    T = T * N + alpha
    T /= T.sum(axis=1, keepdims=True)
    return T


def build_q1_prior(player_games, weights):
    prior = np.zeros(4)
    q1_states = player_games["sQ1"].values.astype(int)
    for i in range(len(player_games)):
        prior[q1_states[i]] += weights[i]
    if prior.sum() == 0:
        return np.ones(4) / 4
    return prior / prior.sum()


def adjust_prior(base_prior, game_total, team_spread, alpha=0.25):
    FAST = np.array([0.10, 0.30, 0.40, 0.20])
    SLOW = np.array([0.35, 0.40, 0.20, 0.05])
    NEUT = np.array([0.25, 0.35, 0.30, 0.10])

    archetype = FAST if game_total >= 230 else SLOW if game_total <= 210 else NEUT
    shifted = (1 - alpha) * base_prior + alpha * archetype

    if team_spread <= -7:
        shifted = shifted + np.array([-0.01, +0.02, -0.01, 0.00])

    shifted = np.clip(shifted, 0.001, None)
    return shifted / shifted.sum()


def simulate(T1, T2, T3, prior, pts_map, game_total, team_spread, line, n_sims=N_SIMS_DEFAULT):
    pace = game_total / NEUTRAL_TOTAL
    p_full_q4 = float(np.clip(0.82 + np.tanh(team_spread / 14.0) * 0.20, 0.40, 0.97))

    rng = np.random.default_rng()
    totals = np.empty(n_sims)
    pts_arr = np.array([pts_map[s] for s in range(4)])

    for sim in range(n_sims):
        state = rng.choice(4, p=prior)
        total = 0.0

        for T in (T1, T2, T3):
            pts = pts_arr[state] * pace
            total += max(0.0, pts + rng.normal(0, 2.0))
            state = rng.choice(4, p=T[state])

        q4_pts = pts_arr[state] * pace
        q4_full = max(0.0, q4_pts + rng.normal(0, 1.5))
        # Bernoulli sample for full vs. shortened Q4 instead of blending the
        # expected value — preserves variance around the line.
        if rng.random() < p_full_q4:
            total += q4_full
        else:
            total += q4_full * (4 / 12)

        totals[sim] = total

    p_over = float((totals > line).mean())
    p_under = float((totals < line).mean())
    return totals, round(p_over, 4), round(p_under, 4)
