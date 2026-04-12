import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from google import genai
from google.genai import types as genai_types
app = Flask(__name__)
CORS(app)

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "models/gemini-3.1-flash-lite-preview"
_overview_cache = {}

# -----------------------------
# LOAD DATA
# -----------------------------
team_url   = os.getenv("API_BASE_URL") + "/api/gs/team-agg-pbp-stats?competitionId=41097&divisionId=1&scope=season"
player_url = os.getenv("API_BASE_URL") + "/api/gs/player-agg-pbp-stats?competitionId=41097&divisionId=1&scope=season"
stats_url  = os.getenv("API_BASE_URL") + "/api/gs/player-agg-stats-public?competitionId=41097&divisionId=1&scope=season"

teamdf = pd.DataFrame(requests.get(team_url).json())
teamdf = teamdf[teamdf["isOffense"] == True]
playerdf_all = pd.DataFrame(requests.get(player_url).json())

STAT_COLS = ["ptsScoredPg", "astPg", "rebPg", "blkPg", "stlPg", "tovPg",
             "fgPct", "fg2Pct", "fg3Pct", "ftPct"]
statsdf = pd.DataFrame(requests.get(stats_url).json())
statsdf = statsdf[["playerId"] + [c for c in STAT_COLS if c in statsdf.columns]]

# minutes
playerdf_all["mpg"] = playerdf_all["minsPbp"] / playerdf_all["gpPbp"]
playerdf_all["mpg"] = playerdf_all["mpg"].replace([np.inf, -np.inf], 0).fillna(0)

# portal players for matching pool
playerdf = playerdf_all[
    (playerdf_all.get("inPortalAfterSeason", False) == True) 
    &
    (playerdf_all["mpg"] >= playerdf_all["mpg"].mean()) &
    (playerdf_all["gpPbp"] >= playerdf_all["gpPbp"].mean())
].copy()

teamdf["fullName"] = teamdf["teamMarket"] + " " + teamdf["teamName"]
playerdf["fullName"] = playerdf["fullName"]
playerdf["teamFullName"] = playerdf["teamMarket"] + " " + playerdf["teamName"]
playerdf = playerdf.merge(statsdf, on="playerId", how="left", suffixes=("", "_stats"))

# -----------------------------
# FEATURES
# -----------------------------
# Shot-type features only — these form a single distribution (all FGA by type)
# Left/center/right and transition/halfcourt dropped: they're separate taxonomies,
# not one distribution, so the row-sum normalization was incorrect for them.
FINAL_FEATURES = [
    "rim_freq", "paint_freq", "midrange_freq",
    "corner3_freq", "atb3_freq", "deep3_freq"
]

# -----------------------------
# BUILD FEATURES
# -----------------------------
def build_features(df):
    df = df.copy()

    # layupDunkFgaFreq is the combined at-rim category; adding layup+dunk on top
    # would double-count. Use the combined field only.
    df["rim_freq"]      = df.get("layupDunkFgaFreq", 0)
    df["paint_freq"]    = df.get("paint2FgaFreq", 0)
    df["midrange_freq"] = df.get("mid2FgaFreqAllS01", 0) + df.get("mid2FgaFreqAllS12", 0) + df.get("mid2FgaFreqAllS23", 0)
    df["corner3_freq"]  = df.get("c3FgaFreq", 0)
    df["atb3_freq"]     = df.get("atb3FgaFreq", 0)
    df["deep3_freq"]    = df.get("lng3FgaFreq", 0) + df.get("nba3FgaFreq", 0)

    features = df[FINAL_FEATURES].fillna(0)
    row_sums = features.sum(axis=1).replace(0, 1)

    return features.div(row_sums, axis=0)

teamdf = pd.concat([teamdf, build_features(teamdf)], axis=1)
playerdf = pd.concat([playerdf, build_features(playerdf)], axis=1)
playerdf_all = pd.concat([playerdf_all, build_features(playerdf_all)], axis=1)

# Precompute eFG% normalization bounds across the portal player pool.
# eFG% = fg2Pct * (2PA share) + 1.5 * fg3Pct * (3PA share)
# where shot-type shares come from the player's own feature distribution.
def _player_efg(row):
    two_share  = row["rim_freq"] + row["paint_freq"] + row["midrange_freq"]
    three_share = row["corner3_freq"] + row["atb3_freq"] + row["deep3_freq"]
    return row.get("fg2Pct", 0) * two_share + 1.5 * row.get("fg3Pct", 0) * three_share

_efg_series = playerdf.apply(_player_efg, axis=1)
EFG_LO = float(_efg_series.quantile(0.05))
EFG_HI = float(_efg_series.quantile(0.95))

# -----------------------------
# GAP PROFILE (PORTAL + SENIORS)
# -----------------------------
def compute_team_gap_profile(team):

    in_portal = playerdf_all.get("inPortalAfterSeason", False)
    if isinstance(in_portal, bool):
        in_portal = pd.Series(False, index=playerdf_all.index)

    is_senior = playerdf_all["classYr"].astype(str).str.lower().str.contains("senior")

    lost = playerdf_all[
        (playerdf_all["teamId"] == team["teamId"]) &
        (in_portal | is_senior)
    ]

    if lost.empty:
        return np.zeros(len(FINAL_FEATURES))

    weights = lost["mpg"].fillna(0).values

    if weights.sum() > 0:
        lost_profile = np.average(
            lost[FINAL_FEATURES].values,
            axis=0,
            weights=weights
        )
    else:
        lost_profile = lost[FINAL_FEATURES].mean().values

    team_profile = team[FINAL_FEATURES].values

    gap = np.maximum(lost_profile - team_profile, 0)

    return gap

# -----------------------------
# EXPLANATION
# -----------------------------
def generate_explanation(player, gap, a):

    explanations = []
    names = FINAL_FEATURES

    top_idx = np.argsort(-a)[:2]
    for i in top_idx:
        explanations.append(f"Strong {names[i]} scoring")

    if np.sum(gap) > 0:
        explanations.append(f"Fills need in {names[np.argmax(gap)]}")

    if player.get("usagePct", 0) > 0.25:
        explanations.append("High-usage scorer")
    else:
        explanations.append("Fits as role player")

    return explanations[:3]

# -----------------------------
# MATCH SCORE
# -----------------------------
def compute_match_score(player, team, precomputed_gap=None):

    a = np.nan_to_num(np.array(player[FINAL_FEATURES], dtype=np.float64))
    b = np.nan_to_num(np.array(team[FINAL_FEATURES], dtype=np.float64))

    if np.sum(a) == 0 or np.sum(b) == 0:
        return {"FinalScore": 0}

    # Both vectors already sum to ~1 from build_features, but re-normalize
    # defensively in case of floating point drift.
    a_p = a / (np.sum(a) + 1e-9)
    b_p = b / (np.sum(b) + 1e-9)

    # Cosine similarity between shot distributions.
    shot_fit = float(cosine_similarity(a_p.reshape(1, -1), b_p.reshape(1, -1))[0][0])

    # Histogram intersection: fraction of the player's shot volume that lands in
    # zones the team already uses. Distinct from cosine — measures shared mass,
    # not angle. Naturally in [0, 1].
    opportunity_fit = float(np.sum(np.minimum(a_p, b_p)))

    # eFG% weighted by the player's own shot selection (no double-counting).
    two_share   = a_p[0] + a_p[1] + a_p[2]   # rim + paint + midrange
    three_share = a_p[3] + a_p[4] + a_p[5]   # corner3 + atb3 + deep3
    efg = player.get("fg2Pct", 0) * two_share + 1.5 * player.get("fg3Pct", 0) * three_share
    efficiency = float(np.clip((efg - EFG_LO) / (EFG_HI - EFG_LO + 1e-9), 0, 1))

    gap = precomputed_gap if precomputed_gap is not None else compute_team_gap_profile(team)

    if np.sum(gap) == 0:
        gap_fit = 0.5
    else:
        gap_p = gap / (np.sum(gap) + 1e-9)
        gap_fit = float(cosine_similarity(a_p.reshape(1, -1), gap_p.reshape(1, -1))[0][0])

    final = (
        0.35 * shot_fit +
        0.25 * opportunity_fit +
        0.20 * gap_fit +
        0.20 * efficiency
    )

    explanation = generate_explanation(player, gap, a_p)

    return {
        "FinalScore":      float(final),
        "ShotFit":         shot_fit,
        "OpportunityFit":  opportunity_fit,
        "GapFit":          gap_fit,
        "Efficiency":      efficiency,
        "Explanation":     explanation
    }

# -----------------------------
# TEAM FIT
# -----------------------------
@app.route("/get_team_fit/<team_id>")
def get_team_fit(team_id):

    # team = teamdf[teamdf["teamId"] == int(team_id)]
    team = teamdf[teamdf["fullName"] == team_id]
    if team.empty:
        return jsonify({"error":"Team not found"}),404

    team = team.iloc[0]
    precomputed_gap = compute_team_gap_profile(team)

    results = []
    for _, player in playerdf.iterrows():
        score = compute_match_score(player, team, precomputed_gap=precomputed_gap)

        results.append({
            "Player": player["fullName"],
            "PlayerId": player["playerId"],
            "PrevTeamId": player["teamId"],
            "Position": player["position"],
            "Year": player["classYr"],
            "PrevTeam": player["teamFullName"],
            **{c: round(float(player[c]), 3) if pd.notna(player.get(c)) else None for c in STAT_COLS},
            **score
        })

    df = pd.DataFrame(results).sort_values("FinalScore", ascending=False)
    return jsonify(df.head(50).to_dict(orient="records"))

# -----------------------------
# PLAYER FIT (BACK 🔥)
# -----------------------------
@app.route("/get_player_fit/<player_id>")
def get_player_fit(player_id):

    # player_row = playerdf[playerdf["playerId"] == int(player_id)]
    player_row = playerdf[playerdf["fullName"] == player_id]
    if player_row.empty:
        return jsonify({"error":"Player not found"}),404

    player = player_row.iloc[0]

    results = []
    for _, team in teamdf.iterrows():
        score = compute_match_score(player, team)

        results.append({
            "Team": team["fullName"],
            "TeamId": team["teamId"],
            "Conference": team.get("conferenceId"),
            **score
        })

    df = pd.DataFrame(results).sort_values("FinalScore", ascending=False)
    return jsonify(df.to_dict(orient="records"))

# -----------------------------
# TEAM NEEDS
# -----------------------------
@app.route("/get_team_needs/<team_id>")
def get_team_needs(team_id):

    team = teamdf[teamdf["teamId"] == int(team_id)]
    if team.empty:
        return jsonify({"error":"Team not found"}),404

    team = team.iloc[0]
    gap = compute_team_gap_profile(team)

    if np.sum(gap) == 0:
        return jsonify({"Team":team["fullName"],"Needs":[]})

    gap = gap / (np.sum(gap)+1e-9)

    idx = np.argsort(-gap)[:5]

    needs = [
        {"Feature": FINAL_FEATURES[i], "Importance": float(gap[i])}
        for i in idx
    ]

    return jsonify({"Team":team["fullName"],"Needs":needs})

# -----------------------------
# AI PLAYER OVERVIEW
# -----------------------------
@app.route("/get_player_overview/<player_name>")
def get_player_overview(player_name):
    if player_name in _overview_cache:
        return jsonify({"overview": _overview_cache[player_name]})

    row = playerdf[playerdf["fullName"] == player_name]
    if row.empty:
        return jsonify({"error": "Player not found"}), 404

    p = row.iloc[0]

    def fmt(val, pct=False):
        if pd.isna(val) or val is None:
            return "N/A"
        return f"{val*100:.1f}%" if pct else f"{val:.1f}"

    shot_profile = ", ".join(
        f"{f.replace('_freq','').replace('_',' ')}: {p[f]*100:.0f}%"
        for f in FINAL_FEATURES
    )

    prompt = f"""You are a college basketball analyst. Write a 3-4 sentence scouting report for this transfer portal player. Be specific, analytical, and direct. No fluff.

Player: {p['fullName']}
Position: {p.get('position','N/A')} | Year: {p.get('classYr','N/A')} | Previous team: {p.get('teamFullName','N/A')}

Per-game stats: {fmt(p.get('ptsScoredPg'))} pts, {fmt(p.get('rebPg'))} reb, {fmt(p.get('astPg'))} ast, {fmt(p.get('stlPg'))} stl, {fmt(p.get('blkPg'))} blk, {fmt(p.get('tovPg'))} tov
Shooting: FG {fmt(p.get('fgPct'), pct=True)}, 2P {fmt(p.get('fg2Pct'), pct=True)}, 3P {fmt(p.get('fg3Pct'), pct=True)}, FT {fmt(p.get('ftPct'), pct=True)}
Shot profile (% of FGA): {shot_profile}

Write the scouting report now:"""

    try:
        response = gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        overview = response.text.strip()
        _overview_cache[player_name] = overview
        return jsonify({"overview": overview})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# LISTS (for search dropdowns)
# -----------------------------
@app.route("/get_teams")
def get_teams():
    return jsonify(sorted(teamdf["fullName"].dropna().tolist()))

@app.route("/get_players")
def get_players():
    return jsonify(sorted(playerdf["fullName"].dropna().tolist()))

# -----------------------------
# AGENT TOOL FUNCTIONS
# -----------------------------
def _get_team_stats(team_name: str) -> str:
    matches = teamdf[teamdf["fullName"].str.lower().str.contains(team_name.lower(), na=False)]
    if matches.empty:
        return f"No team found matching '{team_name}'."
    t = matches.iloc[0]
    profile = {f.replace("_freq", ""): f"{t[f]*100:.1f}%" for f in FINAL_FEATURES if f in t}
    return json.dumps({"team": t["fullName"], "shot_profile": profile})

def _get_player_season_stats(player_name: str) -> str:
    matches = playerdf_all[playerdf_all["fullName"].str.lower().str.contains(player_name.lower(), na=False)]
    if matches.empty:
        return f"No player found matching '{player_name}'."
    p = matches.iloc[0]
    stats_row = statsdf[statsdf["playerId"] == p["playerId"]]
    result = {
        "name": p["fullName"],
        "team": str(p.get("teamMarket","")) + " " + str(p.get("teamName","")),
        "position": p.get("position","N/A"),
        "year": p.get("classYr","N/A"),
        "in_portal": bool(p.get("inPortalAfterSeason", False)),
    }
    if not stats_row.empty:
        s = stats_row.iloc[0]
        for col in STAT_COLS:
            if col in s and pd.notna(s[col]):
                result[col] = round(float(s[col]), 2)
    return json.dumps(result)

def _get_player_pbp_stats(player_name: str) -> str:
    matches = playerdf_all[playerdf_all["fullName"].str.lower().str.contains(player_name.lower(), na=False)]
    if matches.empty:
        return f"No player found matching '{player_name}'."
    p = matches.iloc[0]
    profile = {f.replace("_freq",""): f"{p[f]*100:.1f}%" for f in FINAL_FEATURES if f in p}
    return json.dumps({
        "name": p["fullName"],
        "shot_profile": profile,
        "fg2Pct": round(float(p["fg2Pct"]),3) if pd.notna(p.get("fg2Pct")) else None,
        "fg3Pct": round(float(p["fg3Pct"]),3) if pd.notna(p.get("fg3Pct")) else None,
        "fgPct":  round(float(p["fgPct"]), 3) if pd.notna(p.get("fgPct"))  else None,
        "mpg":    round(float(p["mpg"]),   1) if pd.notna(p.get("mpg"))    else None,
    })

def _get_team_game_log(conference_id: str = "53") -> str:
    url = f"https://api.cbbanalytics.com/api/gs/team-game-stats?competitionId=41097&conferenceId={conference_id}"
    try:
        data = requests.get(url, timeout=8).json()
        if not data:
            return "No game data found."
        df = pd.DataFrame(data)
        cols = [c for c in ["teamFullName","oppFullName","gameDate","ptScored","ptScoredOpp","win"] if c in df.columns]
        return df[cols].head(20).to_json(orient="records")
    except Exception as e:
        return f"Error: {e}"

def _get_player_game_log(player_name: str) -> str:
    matches = playerdf_all[playerdf_all["fullName"].str.lower().str.contains(player_name.lower(), na=False)]
    if matches.empty:
        return f"No player found matching '{player_name}'."
    p = matches.iloc[0]
    url = f"https://api.cbbanalytics.com/api/gs/player-game-stats?competitionId=41097&teamId={p['teamId']}&playerId={p['playerId']}"
    try:
        data = requests.get(url, timeout=8).json()
        if not data:
            return f"No game log found for {p['fullName']}."
        df = pd.DataFrame(data)
        cols = [c for c in ["gameDate","oppFullName","ptScored","ast","reb","fgPct","fg3Pct","min"] if c in df.columns]
        return json.dumps({"player": p["fullName"], "games": df[cols].head(20).to_dict(orient="records")})
    except Exception as e:
        return f"Error: {e}"

_TOOL_FNS = {
    "get_team_stats":          _get_team_stats,
    "get_player_season_stats": _get_player_season_stats,
    "get_player_pbp_stats":    _get_player_pbp_stats,
    "get_team_game_log":       _get_team_game_log,
    "get_player_game_log":     _get_player_game_log,
}

_AGENT_TOOLS = genai_types.Tool(function_declarations=[
    genai_types.FunctionDeclaration(
        name="get_team_stats",
        description="Get season shot profile for a team. Use partial name e.g. 'Maryland'.",
        parameters=genai_types.Schema(type=genai_types.Type.OBJECT,
            properties={"team_name": genai_types.Schema(type=genai_types.Type.STRING)},
            required=["team_name"]),
    ),
    genai_types.FunctionDeclaration(
        name="get_player_season_stats",
        description="Get per-game stats and shooting splits for a player. Use partial name.",
        parameters=genai_types.Schema(type=genai_types.Type.OBJECT,
            properties={"player_name": genai_types.Schema(type=genai_types.Type.STRING)},
            required=["player_name"]),
    ),
    genai_types.FunctionDeclaration(
        name="get_player_pbp_stats",
        description="Get shot zone frequencies and PBP-derived stats for a player.",
        parameters=genai_types.Schema(type=genai_types.Type.OBJECT,
            properties={"player_name": genai_types.Schema(type=genai_types.Type.STRING)},
            required=["player_name"]),
    ),
    genai_types.FunctionDeclaration(
        name="get_team_game_log",
        description="Get recent team game results for a conference. Default conference_id 53 = Big Ten.",
        parameters=genai_types.Schema(type=genai_types.Type.OBJECT,
            properties={"conference_id": genai_types.Schema(type=genai_types.Type.STRING)},
            required=[]),
    ),
    genai_types.FunctionDeclaration(
        name="get_player_game_log",
        description="Get individual game-by-game stats for a player by name.",
        parameters=genai_types.Schema(type=genai_types.Type.OBJECT,
            properties={"player_name": genai_types.Schema(type=genai_types.Type.STRING)},
            required=["player_name"]),
    ),
])

_AGENT_SYSTEM = (
    "You are an expert college basketball analyst with access to real current season data. "
    "Use tools to look up data before answering. Be concise, specific, and analytical."
)
AGENT_MODEL = GEMINI_MODEL  # reuse same model as overview

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json()
    user_message = body.get("message", "").strip()
    history_raw = body.get("history", [])
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Build conversation history for Gemini
    contents = []
    for msg in history_raw:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=msg["content"])]))
    contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)]))

    config = genai_types.GenerateContentConfig(
        system_instruction=_AGENT_SYSTEM,
        tools=[_AGENT_TOOLS],
        temperature=0.2,
    )

    try:
        # Agentic loop: up to 4 tool-call rounds
        for _ in range(4):
            response = gemini.models.generate_content(model=AGENT_MODEL, contents=contents, config=config)
            candidate = response.candidates[0].content

            # Check for tool calls
            tool_calls = [p for p in candidate.parts if p.function_call is not None]
            if not tool_calls:
                # Final text response
                text = "".join(p.text for p in candidate.parts if p.text)
                return jsonify({"response": text})

            # Execute all tool calls and append results
            contents.append(candidate)
            result_parts = []
            for part in tool_calls:
                fn = _TOOL_FNS.get(part.function_call.name)
                args = dict(part.function_call.args) if part.function_call.args else {}
                tool_result = fn(**args) if fn else f"Unknown tool: {part.function_call.name}"
                result_parts.append(genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=part.function_call.name,
                        response={"result": tool_result},
                    )
                ))
            contents.append(genai_types.Content(role="user", parts=result_parts))

        return jsonify({"response": "Max tool iterations reached."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5002)