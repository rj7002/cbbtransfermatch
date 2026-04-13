import os
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from google import genai
from google.genai import types as genai_types
import joblib
app = Flask(__name__)
CORS(app)

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "models/gemini-3.1-flash-lite-preview"
_overview_cache = {}

STAT_COLS = ["ptsScoredPg", "astPg", "rebPg", "blkPg", "stlPg", "tovPg",
             "fgPct", "fg2Pct", "fg3Pct", "ftPct"]

# Load gender-specific NIL models produced by nil_model.ipynb
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def _load_nil_bundle(gender_lower: str):
    with open(os.path.join(_MODELS_DIR, f"nil_meta_{gender_lower}.json")) as f:
        meta = json.load(f)
    return {
        "reg":       joblib.load(os.path.join(_MODELS_DIR, f"nil_regressor_{gender_lower}.pkl")),
        "features":  meta["features"],
        "low":       meta["nil_low_threshold"],
        "high":      meta["nil_high_threshold"],
        "top_confs": set(meta.get("conf_tier_top", [])),
        "mid_confs": set(meta.get("conf_tier_mid", [])),
    }

_NIL_BUNDLES = {
    "MALE":   _load_nil_bundle("male"),
    "FEMALE": _load_nil_bundle("female"),
}

# -----------------------------
# COMPETITION LOOKUP
# -----------------------------
def _current_season_start_year():
    now = datetime.now()
    # CBB season runs ~Nov-Apr; startYear is the fall year of the season
    return now.year if now.month >= 8 else now.year - 1

def _find_competition_id(gender: str) -> int:
    url = os.getenv("API_BASE_URL") + "/api/gs/competitions/"
    comps = requests.get(url).json()
    start_year = _current_season_start_year()
    matches = [c for c in comps if c.get("gender") == gender and c.get("startYear") == start_year]
    if not matches:
        raise RuntimeError(f"No {gender} competition found for startYear={start_year}")
    return matches[0]["competitionId"]

# -----------------------------
# LOAD DATA PER GENDER
# -----------------------------
_data: dict = {}

def load_gender_data(gender: str):
    nil = _NIL_BUNDLES[gender.upper()]
    nil_features = nil["features"]

    comp_id = _find_competition_id(gender)
    base = os.getenv("API_BASE_URL")

    conferences_url = f"{base}/api/gs/conferences/"
    team_url   = f"{base}/api/gs/team-agg-pbp-stats?competitionId={comp_id}&divisionId=1&scope=season"
    teamstats_url = f"{base}/api/gs/team-agg-stats/competition/{comp_id}/division/1/scope/season/"
    player_url = f"{base}/api/gs/player-agg-pbp-stats?competitionId={comp_id}&divisionId=1&scope=season"
    stats_url  = f"{base}/api/gs/player-agg-stats-public?competitionId={comp_id}&divisionId=1&scope=season"

    conferences_df = pd.DataFrame(requests.get(conferences_url).json())
    teamdf = pd.DataFrame(requests.get(team_url).json())
    teamstatsdf = pd.DataFrame(requests.get(teamstats_url).json())
    teamdf = teamdf.merge(conferences_df[["conferenceId", "conferenceLongName"]], on="conferenceId", how="left")
    teamdf = teamdf[teamdf["isOffense"] == True]
    playerdf_all = pd.DataFrame(requests.get(player_url).json())

    statsdf = pd.DataFrame(requests.get(stats_url).json())
    keep_cols = list(set(["playerId", "conferenceId"] + STAT_COLS + nil_features))
    statsdf = statsdf[[c for c in keep_cols if c in statsdf.columns]]

    playerdf_all["mpg"] = playerdf_all["minsPbp"] / playerdf_all["gpPbp"]
    playerdf_all["mpg"] = playerdf_all["mpg"].replace([np.inf, -np.inf], 0).fillna(0)

    playerdf = playerdf_all[
        (playerdf_all.get("inPortalAfterSeason", False) == True)
        &
        (playerdf_all["mpg"] >= playerdf_all["mpg"].mean()) &
        (playerdf_all["gpPbp"] >= playerdf_all["gpPbp"].mean())
    ].copy()

    teamdf["fullName"] = teamdf["teamMarket"] + " " + teamdf["teamName"]
    playerdf["teamFullName"] = playerdf["teamMarket"] + " " + playerdf["teamName"]
    playerdf = playerdf.merge(statsdf, on="playerId", how="left", suffixes=("", "_stats"))

    # Conference tier using the gender-specific mappings from training
    if "conferenceId" in playerdf.columns:
        playerdf["conf_tier"] = playerdf["conferenceId"].apply(
            lambda c: 3 if c in nil["top_confs"] else (2 if c in nil["mid_confs"] else 1)
        )
    else:
        playerdf["conf_tier"] = 1

    # NIL regression — log scale, back to dollars, tiers from gender-specific thresholds
    nil_X = playerdf.reindex(columns=nil_features).apply(pd.to_numeric, errors="coerce").fillna(0)
    raw_preds = np.expm1(nil["reg"].predict(nil_X))
    playerdf["nilValue"] = raw_preds.round(0).astype(int)
    playerdf["nilTier"]  = playerdf["nilValue"].apply(
        lambda v: "High Value" if v >= nil["high"] else ("Mid Value" if v >= nil["low"] else "Low Value")
    )

    # Build shot features
    teamdf_f     = pd.concat([teamdf,      build_features(teamdf)],      axis=1)
    playerdf_f   = pd.concat([playerdf,    build_features(playerdf)],    axis=1)
    playerdf_all_f = pd.concat([playerdf_all, build_features(playerdf_all)], axis=1)

    efg_series = playerdf_f.apply(_player_efg, axis=1)
    efg_lo = float(efg_series.quantile(0.05))
    efg_hi = float(efg_series.quantile(0.95))

    print(f"[{gender}] competitionId={comp_id} | {len(playerdf_f)} portal players")

    teamstatsdf["fullName"] = teamstatsdf["teamMarket"] + " " + teamstatsdf["teamName"]
    teamstatsdf_offense = teamstatsdf[teamstatsdf["isOffense"] == True]

    return {
        "teamdf":        teamdf_f,
        "teamstatsdf":   teamstatsdf_offense,
        "playerdf":      playerdf_f,
        "playerdf_all":  playerdf_all_f,
        "efg_lo":        efg_lo,
        "efg_hi":        efg_hi,
        "comp_id":       comp_id,
    }

def _get_data(gender: str) -> dict:
    g = gender.upper()
    if g not in _data:
        raise KeyError(f"Data for gender '{g}' not loaded")
    return _data[g]

# -----------------------------
# FEATURES
# -----------------------------
FINAL_FEATURES = [
    "rim_freq", "paint_freq", "midrange_freq",
    "corner3_freq", "atb3_freq", "deep3_freq"
]

def build_features(df):
    df = df.copy()
    df["rim_freq"]      = df.get("layupDunkFgaFreq", 0)
    df["paint_freq"]    = df.get("paint2FgaFreq", 0)
    df["midrange_freq"] = df.get("mid2FgaFreqAllS01", 0) + df.get("mid2FgaFreqAllS12", 0) + df.get("mid2FgaFreqAllS23", 0)
    df["corner3_freq"]  = df.get("c3FgaFreq", 0)
    df["atb3_freq"]     = df.get("atb3FgaFreq", 0)
    df["deep3_freq"]    = df.get("lng3FgaFreq", 0) + df.get("nba3FgaFreq", 0)
    features = df[FINAL_FEATURES].fillna(0)
    row_sums = features.sum(axis=1).replace(0, 1)
    return features.div(row_sums, axis=0)

def _player_efg(row):
    two_share   = row["rim_freq"] + row["paint_freq"] + row["midrange_freq"]
    three_share = row["corner3_freq"] + row["atb3_freq"] + row["deep3_freq"]
    return row.get("fg2Pct", 0) * two_share + 1.5 * row.get("fg3Pct", 0) * three_share

# Load both genders at startup
for _g in ("MALE", "FEMALE"):
    _data[_g] = load_gender_data(_g)

# -----------------------------
# GAP PROFILE (PORTAL + SENIORS)
# -----------------------------
def compute_team_gap_profile(team, playerdf_all):
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
        lost_profile = np.average(lost[FINAL_FEATURES].values, axis=0, weights=weights)
    else:
        lost_profile = lost[FINAL_FEATURES].mean().values

    return np.maximum(lost_profile - team[FINAL_FEATURES].values, 0)

# -----------------------------
# EXPLANATION
# -----------------------------
def generate_explanation(player, gap, a):
    explanations = []
    names = FINAL_FEATURES
    for i in np.argsort(-a)[:2]:
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
def compute_match_score(player, team, efg_lo, efg_hi, precomputed_gap=None, playerdf_all=None):
    a = np.nan_to_num(np.array(player[FINAL_FEATURES], dtype=np.float64))
    b = np.nan_to_num(np.array(team[FINAL_FEATURES], dtype=np.float64))

    if np.sum(a) == 0 or np.sum(b) == 0:
        return {"FinalScore": 0}

    a_p = a / (np.sum(a) + 1e-9)
    b_p = b / (np.sum(b) + 1e-9)

    shot_fit = float(cosine_similarity(a_p.reshape(1, -1), b_p.reshape(1, -1))[0][0])
    opportunity_fit = float(np.sum(np.minimum(a_p, b_p)))

    two_share   = a_p[0] + a_p[1] + a_p[2]
    three_share = a_p[3] + a_p[4] + a_p[5]
    efg = player.get("fg2Pct", 0) * two_share + 1.5 * player.get("fg3Pct", 0) * three_share
    efficiency = float(np.clip((efg - efg_lo) / (efg_hi - efg_lo + 1e-9), 0, 1))

    if precomputed_gap is not None:
        gap = precomputed_gap
    elif playerdf_all is not None:
        gap = compute_team_gap_profile(team, playerdf_all)
    else:
        gap = np.zeros(len(FINAL_FEATURES))

    if np.sum(gap) == 0:
        gap_fit = 0.5
    else:
        gap_p = gap / (np.sum(gap) + 1e-9)
        gap_fit = float(cosine_similarity(a_p.reshape(1, -1), gap_p.reshape(1, -1))[0][0])

    final = 0.35 * shot_fit + 0.25 * opportunity_fit + 0.20 * gap_fit + 0.20 * efficiency

    return {
        "FinalScore":      float(final),
        "ShotFit":         shot_fit,
        "OpportunityFit":  opportunity_fit,
        "GapFit":          gap_fit,
        "Efficiency":      efficiency,
        "Explanation":     generate_explanation(player, gap, a_p),
    }

# helper: parse ?gender= param, default MALE
def _gender_param():
    return request.args.get("gender", "MALE").upper()

# -----------------------------
# TEAM FIT
# -----------------------------
@app.route("/get_team_fit/<team_id>")
def get_team_fit(team_id):
    d = _get_data(_gender_param())
    teamdf, playerdf, playerdf_all = d["teamdf"], d["playerdf"], d["playerdf_all"]
    efg_lo, efg_hi = d["efg_lo"], d["efg_hi"]

    team = teamdf[teamdf["fullName"] == team_id]
    if team.empty:
        return jsonify({"error": "Team not found"}), 404

    team = team.iloc[0]
    precomputed_gap = compute_team_gap_profile(team, playerdf_all)

    results = []
    for _, player in playerdf.iterrows():
        score = compute_match_score(player, team, efg_lo, efg_hi, precomputed_gap=precomputed_gap)
        results.append({
            "Player": player["fullName"],
            "PlayerId": player["playerId"],
            "PrevTeamId": player["teamId"],
            "Position": player["position"],
            "Year": player["classYr"],
            "PrevTeam": player["teamFullName"],
            "NilTier": player.get("nilTier"),
            "NilValue": int(player["nilValue"]) if pd.notna(player.get("nilValue")) else None,
            **{c: round(float(player[c]), 3) if pd.notna(player.get(c)) else None for c in STAT_COLS},
            **score
        })

    df = pd.DataFrame(results).sort_values("FinalScore", ascending=False)
    return jsonify(df.head(50).to_dict(orient="records"))

# -----------------------------
# PLAYER FIT
# -----------------------------
TEAM_STAT_COLS = ["ptsScoredPg", "fgPct", "fg3Pct", "efgPct", "rebPg",
                  "astPg", "tovPg", "ortg", "drtg", "netRtg",
                  "pace", "overallWins", "overallLosses", "netRanking"]

@app.route("/get_player_fit/<player_id>")
def get_player_fit(player_id):
    d = _get_data(_gender_param())
    teamdf, playerdf, playerdf_all = d["teamdf"], d["playerdf"], d["playerdf_all"]
    teamstatsdf = d["teamstatsdf"]
    efg_lo, efg_hi = d["efg_lo"], d["efg_hi"]

    player_row = playerdf[playerdf["fullName"] == player_id]
    if player_row.empty:
        return jsonify({"error": "Player not found"}), 404

    player = player_row.iloc[0]

    # Index team stats by teamId for fast lookup
    ts_index = teamstatsdf.set_index("teamId")

    results = []
    for _, team in teamdf.iterrows():
        score = compute_match_score(player, team, efg_lo, efg_hi, playerdf_all=playerdf_all)

        # Attach team stats if available
        tid = team["teamId"]
        team_stats = {}
        if tid in ts_index.index:
            ts = ts_index.loc[tid]
            for col in TEAM_STAT_COLS:
                val = ts.get(col) if hasattr(ts, "get") else (ts[col] if col in ts.index else None)
                team_stats[col] = round(float(val), 3) if val is not None and pd.notna(val) else None

        results.append({
            "Team": team["fullName"],
            "TeamId": tid,
            "Conference": team.get("conferenceLongName") or team.get("conferenceId"),
            **team_stats,
            **score
        })

    df = pd.DataFrame(results).sort_values("FinalScore", ascending=False)
    return jsonify(df.to_dict(orient="records"))

# -----------------------------
# TEAM NEEDS
# -----------------------------
@app.route("/get_team_needs/<team_id>")
def get_team_needs(team_id):
    d = _get_data(_gender_param())
    teamdf, playerdf_all = d["teamdf"], d["playerdf_all"]

    team = teamdf[teamdf["teamId"] == int(team_id)]
    if team.empty:
        return jsonify({"error": "Team not found"}), 404

    team = team.iloc[0]
    gap = compute_team_gap_profile(team, playerdf_all)

    if np.sum(gap) == 0:
        return jsonify({"Team": team["fullName"], "Needs": []})

    gap = gap / (np.sum(gap) + 1e-9)
    idx = np.argsort(-gap)[:5]
    needs = [{"Feature": FINAL_FEATURES[i], "Importance": float(gap[i])} for i in idx]
    return jsonify({"Team": team["fullName"], "Needs": needs})

# -----------------------------
# AI PLAYER OVERVIEW
# -----------------------------
@app.route("/get_player_overview/<player_name>")
def get_player_overview(player_name):
    cache_key = f"{_gender_param()}:{player_name}"
    if cache_key in _overview_cache:
        return jsonify({"overview": _overview_cache[cache_key]})

    d = _get_data(_gender_param())
    row = d["playerdf"][d["playerdf"]["fullName"] == player_name]
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
        _overview_cache[cache_key] = overview
        return jsonify({"overview": overview})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# AI TEAM OVERVIEW
# -----------------------------
@app.route("/get_team_overview/<team_name>")
def get_team_overview(team_name):
    cache_key = f"team:{_gender_param()}:{team_name}"
    if cache_key in _overview_cache:
        return jsonify({"overview": _overview_cache[cache_key]})

    d = _get_data(_gender_param())
    row = d["teamstatsdf"][d["teamstatsdf"]["fullName"] == team_name]
    if row.empty:
        return jsonify({"error": "Team not found"}), 404

    t = row.iloc[0]

    def fmt(val, pct=False, dec=1):
        if pd.isna(val) or val is None:
            return "N/A"
        return f"{val*100:.{dec}f}%" if pct else f"{val:.{dec}f}"

    record    = f"{int(t.get('overallWins', 0))}-{int(t.get('overallLosses', 0))}"
    conf_rec  = f"{int(t.get('confWins', 0))}-{int(t.get('confLosses', 0))}"

    prompt = f"""You are a college basketball analyst. Write a 3-4 sentence program overview for a coaching staff evaluating transfer portal targets. Focus on the team's offensive identity, defensive profile, and what type of player would thrive here. Be specific and analytical. No fluff.

Team: {t['fullName']}
Record: {record} overall, {conf_rec} conference | NET Ranking: {t.get('netRanking', 'N/A')}
Pace: {fmt(t.get('pace'))} possessions/40 min
Offense: {fmt(t.get('ortg'))} ORtg | {fmt(t.get('ptsScoredPg'))} pts/g | {fmt(t.get('efgPct'), pct=True)} eFG% | {fmt(t.get('fga3Rate'), pct=True)} 3-pt rate | {fmt(t.get('astRatio'), pct=True)} ast ratio | {fmt(t.get('tovPct'), pct=True)} TOV%
Defense: {fmt(t.get('drtg'))} DRtg | {fmt(t.get('efgPctAgst'), pct=True)} opp eFG% | {fmt(t.get('orbPctAgst'), pct=True)} opp ORB% | {fmt(t.get('stlPct'), pct=True)} stl%
Rebounding: {fmt(t.get('orbPct'), pct=True)} ORB% | {fmt(t.get('drbPct'), pct=True)} DRB%
Adjusted: {fmt(t.get('ortgAdj'))} adj ORtg | {fmt(t.get('drtgAdj'))} adj DRtg | {fmt(t.get('netRtgAdj'))} adj net

Write the program overview now:"""

    try:
        response = gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        overview = response.text.strip()
        _overview_cache[cache_key] = overview
        return jsonify({"overview": overview})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# LISTS (for search dropdowns)
# -----------------------------
@app.route("/get_teams")
def get_teams():
    d = _get_data(_gender_param())
    return jsonify(sorted(d["teamdf"]["fullName"].dropna().tolist()))

@app.route("/get_players")
def get_players():
    d = _get_data(_gender_param())
    return jsonify(sorted(d["playerdf"]["fullName"].dropna().tolist()))

# -----------------------------
# AGENT TOOL FUNCTIONS
# (use gender passed in chat body, default MALE)
# -----------------------------
def _get_team_stats(team_name: str, gender: str = "MALE") -> str:
    d = _get_data(gender)
    matches = d["teamdf"][d["teamdf"]["fullName"].str.lower().str.contains(team_name.lower(), na=False)]
    if matches.empty:
        return f"No team found matching '{team_name}'."
    t = matches.iloc[0]
    profile = {f.replace("_freq", ""): f"{t[f]*100:.1f}%" for f in FINAL_FEATURES if f in t}
    return json.dumps({"team": t["fullName"], "shot_profile": profile})

def _get_player_season_stats(player_name: str, gender: str = "MALE") -> str:
    d = _get_data(gender)
    matches = d["playerdf_all"][d["playerdf_all"]["fullName"].str.lower().str.contains(player_name.lower(), na=False)]
    if matches.empty:
        return f"No player found matching '{player_name}'."
    p = matches.iloc[0]
    result = {
        "name": p["fullName"],
        "team": str(p.get("teamMarket","")) + " " + str(p.get("teamName","")),
        "position": p.get("position","N/A"),
        "year": p.get("classYr","N/A"),
        "in_portal": bool(p.get("inPortalAfterSeason", False)),
    }
    for col in STAT_COLS:
        if col in p and pd.notna(p[col]):
            result[col] = round(float(p[col]), 2)
    return json.dumps(result)

def _get_player_pbp_stats(player_name: str, gender: str = "MALE") -> str:
    d = _get_data(gender)
    matches = d["playerdf_all"][d["playerdf_all"]["fullName"].str.lower().str.contains(player_name.lower(), na=False)]
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

def _get_team_game_log(conference_id: str = "53", gender: str = "MALE") -> str:
    comp_id = _data.get(gender, {}).get("comp_id", 41097)
    url = f"https://api.cbbanalytics.com/api/gs/team-game-stats?competitionId={comp_id}&conferenceId={conference_id}"
    try:
        data = requests.get(url, timeout=8).json()
        if not data:
            return "No game data found."
        df = pd.DataFrame(data)
        cols = [c for c in ["teamFullName","oppFullName","gameDate","ptScored","ptScoredOpp","win"] if c in df.columns]
        return df[cols].head(20).to_json(orient="records")
    except Exception as e:
        return f"Error: {e}"

def _get_player_game_log(player_name: str, gender: str = "MALE") -> str:
    d = _get_data(gender)
    matches = d["playerdf_all"][d["playerdf_all"]["fullName"].str.lower().str.contains(player_name.lower(), na=False)]
    if matches.empty:
        return f"No player found matching '{player_name}'."
    p = matches.iloc[0]
    comp_id = d["comp_id"]
    url = f"https://api.cbbanalytics.com/api/gs/player-game-stats?competitionId={comp_id}&teamId={p['teamId']}&playerId={p['playerId']}"
    try:
        data = requests.get(url, timeout=8).json()
        if not data:
            return f"No game log found for {p['fullName']}."
        df = pd.DataFrame(data)
        cols = [c for c in ["gameDate","oppFullName","ptScored","ast","reb","fgPct","fg3Pct","min"] if c in df.columns]
        return json.dumps({"player": p["fullName"], "games": df[cols].head(20).to_dict(orient="records")})
    except Exception as e:
        return f"Error: {e}"

def _make_tool_fns(gender: str):
    return {
        "get_team_stats":          lambda **kw: _get_team_stats(gender=gender, **kw),
        "get_player_season_stats": lambda **kw: _get_player_season_stats(gender=gender, **kw),
        "get_player_pbp_stats":    lambda **kw: _get_player_pbp_stats(gender=gender, **kw),
        "get_team_game_log":       lambda **kw: _get_team_game_log(gender=gender, **kw),
        "get_player_game_log":     lambda **kw: _get_player_game_log(gender=gender, **kw),
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
    gender = body.get("gender", "MALE").upper()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    tool_fns = _make_tool_fns(gender)

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
                fn = tool_fns.get(part.function_call.name)
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
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
