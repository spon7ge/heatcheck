# How HeatCheck Works (Like You're 5)

You're trying to answer one question: **"Will this NBA player score more or less than X points tonight?"**

Instead of just averaging his past games, the app simulates 15,000 fake versions of tonight's game and sees what happens most often.

---

## Step 1 — Label Every Quarter He's Ever Played

Look at every quarter this player has ever played. Sort those quarters into 4 buckets based on how many points he scored:

| State | Label | Percentile |
|-------|-------|------------|
| 0 | Cold | Bottom 25% of his quarters |
| 1 | Average | 25th–50th percentile |
| 2 | Hot | 50th–75th percentile |
| 3 | On-Fire | Top 25% |

The cutoffs are personal to each player — "Hot" for a bench player is different than "Hot" for a star.

---

## Step 2 — Build the "What Comes Next?" Table (This IS the Markov Chain)

A **Markov chain** asks one simple question: **"Given where I am right now, where am I likely to go next?"**

The app builds a 4×4 table from real history:

```
             Next Quarter →
             Cold  Avg  Hot  Fire
If Q1 was  Cold [ 50%  30%  15%   5% ]
           Avg  [ 20%  40%  30%  10% ]
           Hot  [ 10%  25%  45%  20% ]
           Fire [  5%  15%  35%  45% ]
```

If a player was Hot in Q1, how often was he Hot/Cold/etc. in Q2? That's all it is — a lookup table of "next-state probabilities."

> **The key Markov insight:** the next quarter only depends on the current quarter, not everything that happened before. Q3 only "looks back" at Q2.

---

## Step 3 — Weight Old Games by How Similar Tonight's Matchup Is

Not all past games count equally. A game vs. a weak defense in a high-scoring game should matter more when predicting a similar night. Each historical game gets a weight based on:

- **How similar was the opponent's defense?** (DEF rating)
- **How similar was the game total (pace)?**
- **How similar was the spread?**
- **How recent was it?** — last 10 games get full credit; older games slowly fade

The transition table from Step 2 is built using these weights, so tonight's specific matchup shapes the chain.

---

## Step 4 — Run 15,000 Fake Games (Monte Carlo Simulation)

Simulate one fake game:

1. Roll the dice to pick a Q1 state (Cold/Avg/Hot/Fire), weighted by his typical Q1 starts
2. Look up points for that state (e.g. Hot = 8 pts), add a tiny random wobble
3. Roll the dice again using the transition table to pick Q2's state
4. Repeat for Q3 and Q4
5. Add up all 4 quarters = one simulated game total

Do that **15,000 times**. You get a full distribution of possible point totals.

---

## Step 5 — Count the Results

```
P(Over 24.5)  =  how many of the 15,000 simulations landed above 24.5
P(Under 24.5) =  the rest
```

The output also gives you the mean of all simulations as the projected points total, plus a histogram so you can see the full shape of the distribution.

---

## One-Sentence Summary

HeatCheck watches how a player tends to stay hot or cool down quarter-by-quarter, weights past games toward similar matchups, then rolls the dice 15,000 times using those patterns to estimate whether he'll hit tonight's prop line.
