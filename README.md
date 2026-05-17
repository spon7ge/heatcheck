# HeatCheck

A Markov chain Monte Carlo simulator for NBA player points props. Given a player, opponent, game total, and spread, it estimates the probability of going over or under a scoring line.

## How It Works

Each historical quarter is labeled as one of four performance states (Cold / Average / Hot / On-Fire) based on the player's own scoring percentiles. A weighted Markov chain models transitions between states quarter-to-quarter. Monte Carlo simulation (default 15,000 runs) walks the chain to produce a full distribution of projected game totals.

Historical games are weighted by:
- **Context similarity** — how closely the opponent's DEF rating, game total, and spread match the upcoming game
- **Recency** — the last 10 games get full weight; older games decay exponentially

See [WORKFLOW.md](WORKFLOW.md) for a detailed pipeline diagram.

## Setup

### Requirements

- Python 3.10+
- Dependencies listed in `req.txt`

```bash
pip install -r req.txt
```

### Data

Place the following files in the `data/` directory:

| File | Description |
|------|-------------|
| `s24_pbp.parquet` | 2023-24 play-by-play |
| `s25_pbp.parquet` | 2024-25 play-by-play |
| `s26_pbp.parquet` | 2025-26 play-by-play |
| `S24.csv` | 2023-24 player game logs with context stats |
| `S25.csv` | 2024-25 player game logs with context stats |
| `S26.csv` | 2025-26 player game logs with context stats |

Stats CSVs should include columns: `GAME_ID`, `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ABBREVIATION`, `DEF_RATING`, `GAME_TOTAL`, `TEAM_SPREAD`, `GAME_DATE`.

## Running the App

```bash
streamlit run app.py
```

Then open the sidebar to configure:

| Input | Description |
|-------|-------------|
| **Player** | NBA player name |
| **Opponent** | Opposing team abbreviation |
| **Game Total** | Vegas over/under for the full game |
| **Team Spread** | Negative = favorite (e.g. `-3.5`) |
| **Points Line** | The prop line to evaluate |
| **Simulations** | Number of Monte Carlo runs (2,000–50,000) |

Click **Run Model** to simulate.

## Output

- **P(Over) / P(Under)** — probability the player exceeds or falls short of the line
- **Projected points** — mean of the simulated distribution
- **Historical average** — baseline from the loaded game data
- **Distribution histogram** — full simulated point distribution with line and mean markers
- **Model internals** — opponent DEF rating, state bin edges, transition matrices, and weight breakdown

## Project Structure

```
markov_chain_pts/
├── app.py        # Streamlit UI
├── markov.py     # Markov chain engine and Monte Carlo simulation
├── helper.py     # Data loading, quarter scoring, DEF_RATING joins
├── requirements.txt       # Python dependencies
├── WORKFLOW.md   # Pipeline diagram and design notes
└── data/         # Season parquet and CSV files (not included)
```
