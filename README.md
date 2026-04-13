<<<<<<< HEAD
# PortalMatch — CBB Transfer Portal Matching Engine
=======
<img width="1506" height="745" alt="Screenshot 2026-04-12 at 9 42 28 PM" src="https://github.com/user-attachments/assets/7cd974c3-c35b-4164-9b73-0f5266337301" />

## PortalMatch — CBB Transfer Portal Matching Engine
>>>>>>> d80ee7d6c5754259e5e12a2e45bfa7cc0c30ad3c

A full-stack college basketball analytics tool that matches transfer portal players to programs (and vice versa) using shot profile similarity, opportunity fit, gap analysis, and shooting efficiency. Includes AI-generated scouting reports, an agentic analyst chat, NIL valuation models, and real-time filtering.

## Features

- **Team → Player matching**: Find portal players whose shot profiles fit your system, ranked by a weighted composite score
- **Player → Team matching**: Find programs that fit a player's game, filtered by conference
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
- Google Gemini API (`google-genai`) — scouting reports + agentic chat
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

<<<<<<< HEAD
## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:

```
GEMINI_API_KEY=your_key_here
API_BASE_URL=https://api.cbbanalytics.com
```

Start the server:

```bash
python similarity_server.py
```

The server loads data for both Men's and Women's competitions at startup (~15–20 seconds) and runs on `http://localhost:5002`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /get_team_fit/<team_name>?gender=MALE` | Top 50 portal players for a team |
| `GET /get_player_fit/<player_name>?gender=MALE` | Best team fits for a portal player |
| `GET /get_player_overview/<name>?gender=MALE` | AI scouting report |
| `GET /get_team_overview/<name>?gender=MALE` | AI program overview |
| `GET /get_teams?gender=MALE` | All team names |
| `GET /get_players?gender=MALE` | All portal player names |
| `POST /chat` | Agentic analyst chat (`{message, history, gender}`) |

=======
>>>>>>> d80ee7d6c5754259e5e12a2e45bfa7cc0c30ad3c
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
| Shot Fit | Cosine similarity of shot-type distributions | 35% |
| Opportunity Fit | Histogram intersection of shot profiles | 25% |
| Gap Fill | Cosine similarity vs team's departing player profile | 20% |
| Efficiency | Normalized eFG% (5th–95th percentile) | 20% |

Weights are adjustable in the UI per search.
<<<<<<< HEAD

## Notes

- `.env` and `nil_data.csv` are gitignored — you must supply your own credentials and training data to retrain the NIL models
- Trained model artifacts (`.pkl`) are also gitignored; run the notebook to generate them before starting the backend
- The backend must be running before the frontend will load any data
=======
>>>>>>> d80ee7d6c5754259e5e12a2e45bfa7cc0c30ad3c
