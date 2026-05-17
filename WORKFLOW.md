# HeatCheck — Workflow

## Overview

A Markov chain Monte Carlo (MCMC) simulator that estimates the probability a player scores over/under a betting line, using per-quarter scoring states and context-weighted historical games.

---

## Pipeline

```mermaid
flowchart TD
    A["Raw Data\n(parquet PBP + CSV stats\nS24, S25, S26)"]
    --> B["helper.load_data()\n• Build Q1–Q4 pts from play-by-play\n• Attach opponent DEF_RATING per game\n• Left-merge stats ← quarter scoring"]

    B --> C["User Inputs (Streamlit sidebar)\n• Player, Opponent\n• Game Total, Team Spread, Points Line\n• # Simulations"]

    C --> D["markov.setup_player()\n• Compute player-specific bin edges\n  (25th / 50th / 75th percentile)\n• Label every quarter: Cold=0 / Avg=1 / Hot=2 / Fire=3\n• Map each state → median points"]

    C --> E["helper.get_opponent_def_rating()\n• Average opponent's last 10-game DEF_RATING"]

    D & E --> F["markov.compute_weights()\n• Gaussian similarity over\n  OPP_DEF_RATING, GAME_TOTAL, TEAM_SPREAD\n• Recency decay: last 10 games = full weight,\n  older games decay exponentially"]

    F --> G["markov.build_transition_matrix()\n• Weighted-count 4×4 matrices:\n  T1: Q1→Q2  T2: Q2→Q3  T3: Q3→Q4\n• Laplace smoothing (α = 0.5)"]

    F --> H["markov.build_q1_prior() + adjust_prior()\n• Weighted Q1 state distribution\n• Nudge by game pace (FAST/SLOW/NEUT)\n  and whether team is a big favorite"]

    G & H --> I["markov.simulate()  ×N_SIMS (default 15,000)\nFor each simulation:\n  1. Sample Q1 state from prior\n  2. Lookup pts × pace + noise → add to total\n  3. Transition to next quarter via T1, T2, T3\n  4. Q4: Bernoulli draw for full vs. shortened\n     (garbage time effect)"]

    I --> J["Output\n• P(Over), P(Under)\n• Mean projected points\n• Distribution histogram\n• Model internals expander"]
```

---

## State System

Each quarter is labeled with one of four performance states based on the player's own historical percentiles:

| State | Label   | Meaning                      |
|-------|---------|------------------------------|
| 0     | Cold    | ≤ 25th percentile            |
| 1     | Average | 25th – 50th percentile       |
| 2     | Hot     | 50th – 75th percentile       |
| 3     | On-Fire | > 75th percentile            |

---

## Key Design Decisions

**Context-weighted games** — Historical games are not treated equally. Games played against similar opponents (by DEF_RATING), in similar pace environments (game total), and with similar spreads are weighted higher using a Gaussian kernel.

**Recency decay** — The 10 most recent games always get full weight. Games beyond that decay exponentially so the model stays current without ignoring older sample.

**Pace scaling** — Simulated quarter points are multiplied by `game_total / 230` so the model naturally projects more in fast-paced games.

**Shortened Q4** — There's a Bernoulli draw each sim for whether Q4 is played in full. The probability is derived from the team spread: big favorites are more likely to see garbage time (partial Q4 scoring).

**Laplace smoothing** — Transition matrices use α = 0.5 pseudo-count per cell to avoid zero-probability transitions from thin samples.

---

## File Map

| File        | Role                                                        |
|-------------|-------------------------------------------------------------|
| `app.py`    | Streamlit UI — inputs, result display, histogram            |
| `markov.py` | Core engine — state binning, weights, matrices, simulation  |
| `helper.py` | Data loading — PBP parsing, quarter scoring, DEF_RATING join|
| `data/`     | Season files: `s24_pbp.parquet`, `S24.csv`, etc.           |
