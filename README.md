<img width="1501" height="745" alt="Screenshot 2026-04-15 at 1 53 47 PM" src="https://github.com/user-attachments/assets/9353a922-0666-4ecd-a64f-a8f1138431cf" />

# PortalMatch — CBB Transfer Portal Matching Engine

A full-stack college basketball analytics tool that matches transfer portal players to programs (and vice versa) using shot profile similarity, opportunity fit, gap analysis, and shooting efficiency. Includes AI-generated scouting reports, an agentic analyst chat, NIL valuation models, and real-time filtering.

## Features

- **Team → Player matching**: Find portal players whose shot profiles fit your system, ranked by a weighted composite score
- **Player → Team matching**: Find programs that fit a player's game, filtered by conference
- **Player/Team match comparison**: Select a player and a team and see how they match
- **Player/Team smart search**: Natural language querying to find players and teams
- **Fit scoring**: Cosine shot similarity, histogram intersection opportunity fit, MPG-weighted gap profile, normalized eFG%
- **Adjustable weights**: Drag sliders to reweight the four scoring dimensions live on the client
- **NIL valuation**: LightGBM regression model (separate for men's and women's) predicts NIL dollar value and tier (Low / Mid / High) for every portal player
- **AI scouting reports**: Gemini-generated player scouting reports and team program overviews
- **Agentic analyst chat**: Ask any question about players or teams — backed by a Gemini function-calling agent with access to live stats
- **Men's & Women's**: Full support for both, dynamically resolved from the competitions API each season
- **Filters**: NIL budget (two-way slider), class year, position, conference

## Tech Stack

**Backend**
- Python / Flask
- pandas, numpy, scikit-learn
- LightGBM (NIL valuation models)
- Mistral AI API (`mistralai`) — scouting reports + agentic chat
- Data from cbbanalytics.com API

**Frontend**
- React (Vite)
- Recharts (radar chart)
- Custom dark sports-app UI

## Project Structure

```
├── backend/
│   ├── similarity_server.py   # Flask API
│   ├── requirements.txt
│   └── models/
│       ├── nil_regressor_male.pkl
│       ├── nil_regressor_female.pkl
│       ├── nil_meta_male.json
│       └── nil_meta_female.json
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── App.css
│   └── index.html
├── nil_model.ipynb            # NIL model training
└── full_model.ipynb
```

## NIL Model

Trained on historical NIL deal data merged with per-game and advanced stats. Key decisions:

- Log-transform on target (NIL values are right-skewed)
- LightGBM feature importance used for feature selection
- Conference tier derived from NIL medians in training data
- Tier thresholds (Low / Mid / High) set at 33rd/67th percentile of training values
- Separate models for men's and women's — different market sizes and conference structures

## Scoring Model

Each player-team match is scored across four dimensions:

| Dimension | Method | Default Weight |
|---|---|---|
| Shot Fit | Cosine similarity of shot-type distributions | 45% |
| Gap Fill | Cosine similarity vs team's departing player profile | 25% |
| Efficiency | Normalized TS% | 30% |
Weights are adjustable in the UI per search.
