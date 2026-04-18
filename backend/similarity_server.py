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
from mistralai.client import Mistral
import joblib
app = Flask(__name__)
CORS(app)

mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MISTRAL_MODEL = "mistral-small-2603"

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

    # Player environment index: percentile rank of tsPct * usagePct (no constants)
    player_eff_series = playerdf_all["tsPct"] * playerdf_all["usagePct"]
    playerdf_all["player_env"] = player_eff_series.rank(pct=True)
    playerdf["player_env"] = playerdf["playerId"].map(
        playerdf_all.set_index("playerId")["player_env"]
    )
    eff_lo = player_eff_series.quantile(0.05)
    eff_hi = player_eff_series.quantile(0.95)

    # Build shot features
    teamdf_f     = pd.concat([teamdf,      build_features(teamdf)],      axis=1)
    playerdf_f   = pd.concat([playerdf,    build_features(playerdf)],    axis=1)
    playerdf_all_f = pd.concat([playerdf_all, build_features(playerdf_all)], axis=1)

    teamstatsdf["fullName"] = teamstatsdf["teamMarket"] + " " + teamstatsdf["teamName"]
    teamstatsdf_offense = teamstatsdf[teamstatsdf["isOffense"] == True].copy()

    # Team environment index: percentile rank of ortg * efgPct (no constants)
    team_eff_raw = teamstatsdf_offense["ortg"] * teamstatsdf_offense["efgPct"]
    teamstatsdf_offense["team_env"] = team_eff_raw.rank(pct=True)

    # Merge team_env onto teamdf_f so match score functions can access it
    teamdf_f["team_env"] = teamdf_f["teamId"].map(
        teamstatsdf_offense.set_index("teamId")["team_env"]
    )

    print(f"[{gender}] competitionId={comp_id} | {len(playerdf_f)} portal players")



    return {
        "teamdf":        teamdf_f,
        "teamstatsdf":   teamstatsdf_offense,
        "playerdf":      playerdf_f,
        "playerdf_all":  playerdf_all_f,
        "eff_lo":        eff_lo,
        "eff_hi":          eff_hi,
        "comp_id":       comp_id,
        "eff_series" : player_eff_series
    }

def _get_data(gender: str) -> dict:
    g = gender.upper()
    if g not in _data:
        raise KeyError(f"Data for gender '{g}' not loaded")
    return _data[g]

# -----------------------------
# FEATURES
# -----------------------------

shot_features = ['nba3FgaFreq',
 'lane2FgaFreq',
 'atr2FgaFreq',
 'paint2FgaFreq',
 'mid2FgaFreq',
 'c3FgaFreq',
 'atb3FgaFreq',
 'lb2FgaFreq',
 'rb2FgaFreq',
 'le2FgaFreq',
 're2FgaFreq',
 'lc3FgaFreq',
 'rc3FgaFreq',
 'lw3FgaFreq',
 'rw3FgaFreq',
 'tok3FgaFreq',
 'med2FgaFreq',
 'lng2FgaFreq',
 'sht3FgaFreq',
 'lng3FgaFreq',
  'fgaFreqAllS01',
 'fgaFreqAllS12',
 'fgaFreqAllS23']

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

# def _player_efg(row):
#     two_share   = row["rim_freq"] + row["paint_freq"] + row["midrange_freq"]
#     three_share = row["corner3_freq"] + row["atb3_freq"] + row["deep3_freq"]
#     return row.get("fg2Pct", 0) * two_share + 1.5 * row.get("fg3Pct", 0) * three_share

# Load both genders at startup
for _g in ("MALE", "FEMALE"):
    _data[_g] = load_gender_data(_g)

# -----------------------------
# GAP PROFILE (PORTAL + SENIORS)
# -----------------------------
def compute_team_gap_profile(team, playerdf_all):
    # in_portal = playerdf_all.get("inPortalAfterSeason", False)
    # if isinstance(in_portal, bool):
    #     in_portal = pd.Series(False, index=playerdf_all.index)
    in_portal = playerdf_all.get("inPortalAfterSeason", pd.Series(False, index=playerdf_all.index)).fillna(False)

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
# def compute_match_score(player, team,precomputed_gap=None, playerdf_all=None):
#     a = np.nan_to_num(np.array(player[FINAL_FEATURES], dtype=np.float64))
#     b = np.nan_to_num(np.array(team[FINAL_FEATURES], dtype=np.float64))

#     a_shot = np.nan_to_num(np.array(player[shot_features], dtype=np.float64))
#     b_shot = np.nan_to_num(np.array(team[shot_features], dtype=np.float64)) 
    
#     if np.sum(a) == 0 or np.sum(b) == 0:
#         return {"FinalScore": 0}

#     a_p = a / (np.sum(a) + 1e-9)
#     b_p = b / (np.sum(b) + 1e-9)

#     shot_fit = float(cosine_similarity(a_shot.reshape(1, -1), b_shot.reshape(1, -1))[0][0])

#     efficiency = float(player.get("efficiency", 0))

#     if precomputed_gap is not None:
#         gap = precomputed_gap
#     elif playerdf_all is not None:
#         gap = compute_team_gap_profile(team, playerdf_all)
#     else:
#         gap = np.zeros(len(FINAL_FEATURES))

#     if np.sum(gap) == 0:
#         gap_fit = 0.5
#     else:
#         gap_p = gap / (np.sum(gap) + 1e-9)
#         gap_fit = float(cosine_similarity(a_p.reshape(1, -1), gap_p.reshape(1, -1))[0][0])

#     # final = 0.35 * shot_fit + 0.25 * opportunity_fit + 0.20 * gap_fit + 0.20 * efficiency
#     final = 0.45 * shot_fit + 0.25 * gap_fit + 0.30 * efficiency

#     return {
#         "FinalScore":      float(final),
#         "ShotFit":         shot_fit,
#         "GapFit":          gap_fit,
#         "Efficiency":      efficiency,
#         "Explanation":     generate_explanation(player, gap, a_p),
#     }

def compute_match_score_players_for_teams(player, team,precomputed_gap=None, playerdf_all=None):
    a = np.nan_to_num(np.array(player[FINAL_FEATURES], dtype=np.float64))
    b = np.nan_to_num(np.array(team[FINAL_FEATURES], dtype=np.float64))

    a_shot = np.nan_to_num(np.array(player[shot_features], dtype=np.float64))
    b_shot = np.nan_to_num(np.array(team[shot_features], dtype=np.float64))

    if np.sum(a) == 0 or np.sum(b) == 0:
        return {"FinalScore": 0}

    a_p = a / (np.sum(a) + 1e-9)
    b_p = b / (np.sum(b) + 1e-9)

    shot_fit = float(cosine_similarity(a_shot.reshape(1, -1), b_shot.reshape(1, -1))[0][0])

    # Context efficiency: sigmoid over the percentile-rank difference — natural [0,1], no hard clips
    player_env = float(player.get("player_env", 0.5))
    team_env   = float(team.get("team_env", 0.5))
    context_eff = float(1.0 / (1.0 + np.exp(-7.0 * (player_env - team_env))))

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

    final = 0.45 * shot_fit + 0.25 * gap_fit + 0.30 * context_eff

    return {
        "FinalScore":   float(final),
        "ShotFit":      shot_fit,
        "GapFit":       gap_fit,
        "ContextEff":   round(context_eff, 4),
        "PlayerEnv":    round(player_env, 4),
        "TeamEnv":      round(team_env, 4),
        "Explanation":  generate_explanation(player, gap, a_p),
    }

def compute_match_score_teams_for_players(player, team,precomputed_gap=None, playerdf_all=None):
    a = np.nan_to_num(np.array(player[FINAL_FEATURES], dtype=np.float64))
    b = np.nan_to_num(np.array(team[FINAL_FEATURES], dtype=np.float64))

    a_shot = np.nan_to_num(np.array(player[shot_features], dtype=np.float64))
    b_shot = np.nan_to_num(np.array(team[shot_features], dtype=np.float64))
    if np.sum(a) == 0 or np.sum(b) == 0:
        return {"FinalScore": 0}

    a_p = a / (np.sum(a) + 1e-9)
    b_p = b / (np.sum(b) + 1e-9)

    shot_fit = float(cosine_similarity(a_shot.reshape(1, -1), b_shot.reshape(1, -1))[0][0])

    # Context efficiency: sigmoid over the percentile-rank difference — natural [0,1], no hard clips
    player_env = float(player.get("player_env", 0.5))
    team_env   = float(team.get("team_env", 0.5))
    context_eff = float(1.0 / (1.0 + np.exp(-7.0 * (player_env - team_env))))

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

    final = 0.45 * shot_fit + 0.25 * gap_fit + 0.30 * context_eff

    return {
        "FinalScore":   float(final),
        "ShotFit":      shot_fit,
        "GapFit":       gap_fit,
        "ContextEff":   round(context_eff, 4),
        "PlayerEnv":    round(player_env, 4),
        "TeamEnv":      round(team_env, 4),
        "Explanation":  generate_explanation(player, gap, a_p),
    }

# helper: parse ?gender= param, default MALE
def _gender_param():
    return request.args.get("gender", "MALE").upper()

# -----------------------------
# TEAM FIT
# -----------------------------
@app.route("/get_team_fit/<team_id>", methods=["GET", "POST"])
def get_team_fit(team_id):
    body = request.get_json(silent=True) or {}
    gender = body.get("gender", request.args.get("gender", "MALE")).upper()
    d = _get_data(gender)
    teamdf, playerdf, playerdf_all = d["teamdf"], d["playerdf"], d["playerdf_all"]

    team = teamdf[teamdf["fullName"] == team_id]
    if team.empty:
        return jsonify({"error": "Team not found"}), 404

    team = team.iloc[0]
    precomputed_gap = compute_team_gap_profile(team, playerdf_all)

    # Read filters from POST body OR query params
    nil_min = body.get("nil_min") or (request.args.get("nil_min") and int(request.args.get("nil_min")))
    nil_max = body.get("nil_max") or (request.args.get("nil_max") and int(request.args.get("nil_max")))

    years_raw = body.get("years") or request.args.get("years", "")
    years = years_raw if isinstance(years_raw, list) else [y for y in years_raw.split(",") if y]

    pos_raw = body.get("positions") or request.args.get("positions", "")
    positions = pos_raw if isinstance(pos_raw, list) else [p for p in pos_raw.split(",") if p]

    pool = playerdf
    if nil_min is not None:
        pool = pool[pool["nilValue"] >= nil_min]
    if nil_max is not None:
        pool = pool[pool["nilValue"] <= nil_max]
    if years:
        pool = pool[pool["classYr"].isin(years)]
    if positions:
        pool = pool[pool["position"].isin(positions)]

    print(f"[team_fit] gender={gender} years={years} pos={positions} nil={nil_min}-{nil_max} | pool {len(playerdf)}→{len(pool)}")

    results = []
    for _, player in pool.iterrows():
        score = compute_match_score_players_for_teams(player, team, precomputed_gap=precomputed_gap)
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

    if not results:
        return jsonify([])
    df = pd.DataFrame(results).sort_values("FinalScore", ascending=False)
    # return jsonify(df.head(50).to_dict(orient="records"))
    top = df.head(25)
    return jsonify(top.where(top.notna(), other=None).to_dict(orient="records"))

# -----------------------------
# PLAYER FIT
# -----------------------------
TEAM_STAT_COLS = ["ptsScoredPg", "fgPct", "fg3Pct", "efgPct", "rebPg",
                  "astPg", "tovPg", "ortg", "drtg", "netRtg",
                  "pace", "overallWins", "overallLosses", "netRanking"]

@app.route("/get_player_fit/<player_id>", methods=["GET", "POST"])
def get_player_fit(player_id):
    body = request.get_json(silent=True) or {}
    gender = body.get("gender", request.args.get("gender", "MALE")).upper()
    d = _get_data(gender)
    teamdf, teamstatsdf, playerdf, playerdf_all = d["teamdf"], d['teamstatsdf'], d["playerdf"], d["playerdf_all"]
    teamstatsdf = d["teamstatsdf"]

    player_row = playerdf[playerdf["fullName"] == player_id]
    if player_row.empty:
        return jsonify({"error": "Player not found"}), 404

    player = player_row.iloc[0]

    # Read conferences filter from POST body OR query params
    conf_raw = body.get("conferences") or request.args.get("conferences", "")
    conferences = conf_raw if isinstance(conf_raw, list) else [c for c in conf_raw.split(",") if c]

    ts_index = teamstatsdf.set_index("teamId")
    total_teams = len(ts_index)

    # Look up the player's current team net ranking from teamstatsdf
    old_tid = player.get("teamId")
    old_net = None
    if old_tid is not None and old_tid in ts_index.index:
        v = ts_index.loc[old_tid].get("netRanking")
        old_net = float(v) if v is not None and pd.notna(v) else None

    # teamdf has FINAL_FEATURES (shot profile) + efficiency; use it as the pool
    pool = teamdf
    if conferences:
        pool = pool[pool["conferenceLongName"].isin(conferences)]

    results = []
    for _, team in pool.iterrows():
        score = compute_match_score_teams_for_players(player, team, playerdf_all=playerdf_all)

        tid = team["teamId"]
        team_stats = {}
        new_net = None
        if tid in ts_index.index:
            ts = ts_index.loc[tid]
            for col in TEAM_STAT_COLS:
                val = ts.get(col) if hasattr(ts, "get") else (ts[col] if col in ts.index else None)
                team_stats[col] = round(float(val), 3) if val is not None and pd.notna(val) else None
            v = ts.get("netRanking")
            new_net = float(v) if v is not None and pd.notna(v) else None

        # Ranking jump: informational only — not factored into FinalScore
        if old_net is not None and new_net is not None and total_teams > 0:
            ranking_jump = float(np.clip((old_net - new_net) / total_teams, 0.0, 1.0))
        else:
            ranking_jump = 0.0

        results.append({
            "Team": team["fullName"],
            "TeamId": tid,
            "Conference": team.get("conferenceLongName") or team.get("conferenceId"),
            **team_stats,
            **score,
            "RankingJump": round(ranking_jump, 4),
        })

    df = pd.DataFrame(results).sort_values("FinalScore", ascending=False)
    return jsonify(df.where(df.notna(), other=None).to_dict(orient="records"))

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

    prompt = f"""You are an expert college basketball analyst. Write a 3-4 sentence scouting report for this transfer portal player. Be specific, analytical, and direct. No fluff.

Player: {p['fullName']}
Position: {p.get('position','N/A')} | Height (inches): {p.get('height','N/A')} | Year: {p.get('classYr','N/A')} | Previous team: {p.get('teamFullName','N/A')}

Per-game stats: {fmt(p.get('ptsScoredPg'))} pts, {fmt(p.get('rebPg'))} reb, {fmt(p.get('astPg'))} ast, {fmt(p.get('stlPg'))} stl, {fmt(p.get('blkPg'))} blk, {fmt(p.get('tovPg'))} tov
Shooting: FG {fmt(p.get('fgPct'), pct=True)}, 2P {fmt(p.get('fg2Pct'), pct=True)}, 3P {fmt(p.get('fg3Pct'), pct=True)}, FT {fmt(p.get('ftPct'), pct=True)}
Shot profile (% of FGA): {shot_profile}

Do not add a title just give the scouting report. DO NOT MAKE UP ANY INFORMATION. ONLY USE INFORMATION YOU ARE GIVEN TO CREATE THE REPORT.
Write the scouting report now:"""

    try:
        response = mistral.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        overview = response.choices[0].message.content.strip()
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

    prompt = f"""You are an expert college basketball analyst. Write a 3-4 sentence program overview for a player who is considering transferring to the program. Focus on the team's offensive identity, defensive profile, and overall playstyle. Be specific and analytical. No fluff.

Team: {t['fullName']}
Record: {record} overall, {conf_rec} conference | NET Ranking: {t.get('netRanking', 'N/A')}
Pace: {fmt(t.get('pace'))} possessions/40 min
Offense: {fmt(t.get('ortg'))} ORtg | {fmt(t.get('ptsScoredPg'))} pts/g | {fmt(t.get('efgPct'), pct=True)} eFG% | {fmt(t.get('fga3Rate'), pct=True)} 3-pt rate | {fmt(t.get('astRatio'), pct=True)} ast ratio | {fmt(t.get('tovPct'), pct=True)} TOV%
Defense: {fmt(t.get('drtg'))} DRtg | {fmt(t.get('efgPctAgst'), pct=True)} opp eFG% | {fmt(t.get('orbPctAgst'), pct=True)} opp ORB% | {fmt(t.get('stlPct'), pct=True)} stl%
Rebounding: {fmt(t.get('orbPct'), pct=True)} ORB% | {fmt(t.get('drbPct'), pct=True)} DRB%
Adjusted: {fmt(t.get('ortgAdj'))} adj ORtg | {fmt(t.get('drtgAdj'))} adj DRtg | {fmt(t.get('netRtgAdj'))} adj net

Do not add a title just give the overview. DO NOT MAKE UP ANY INFORMATION. ONLY USE INFORMATION YOU ARE GIVEN TO CREATE THE OVERVIEW.
Write the program overview now:"""

    try:
        response = mistral.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        overview = response.choices[0].message.content.strip()
        _overview_cache[cache_key] = overview
        return jsonify({"overview": overview})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# MATCH SCORE (player × team)
# -----------------------------
@app.route("/get_match_score/<player_name>/<team_name>")
def get_match_score(player_name, team_name):
    gender = request.args.get("gender", "MALE").upper()
    d = _get_data(gender)
    teamdf, playerdf, playerdf_all = d["teamdf"], d["playerdf"], d["playerdf_all"]
    teamstatsdf = d["teamstatsdf"]
    eff_lo, eff_hi = d["eff_lo"], d["eff_hi"]
    eff_series = d["eff_series"]

    player_row = playerdf[playerdf["fullName"] == player_name]
    if player_row.empty:
        return jsonify({"error": "Player not found"}), 404

    team_row = teamdf[teamdf["fullName"] == team_name]
    if team_row.empty:
        return jsonify({"error": "Team not found"}), 404

    player = player_row.iloc[0]
    team   = team_row.iloc[0]

    score = compute_match_score_players_for_teams(player, team, playerdf_all=playerdf_all)

    # Pull team stats for the prompt
    eff_index = teamstatsdf.set_index("teamId")
    tid = team["teamId"]
    t = eff_index.loc[tid] if tid in eff_index.index else None

    def fmt(val, pct=False):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val*100:.1f}%" if pct else f"{val:.1f}"

    shot_profile = ", ".join(
        f"{f.replace('_freq','').replace('_',' ')}: {player[f]*100:.0f}%"
        for f in FINAL_FEATURES if f in player
    )

    team_offense = ""
    if t is not None:
        team_offense = (
            f"ORtg {fmt(t.get('ortg'))}, {fmt(t.get('ptsScoredPg'))} pts/g, "
            f"eFG% {fmt(t.get('efgPct'), pct=True)}, 3pt rate {fmt(t.get('fga3Rate'), pct=True)}, "
            f"pace {fmt(t.get('pace'))}"
        )

    gap = compute_team_gap_profile(team, playerdf_all)
    if np.sum(gap) > 0:
        top_gap_idx = int(np.argmax(gap))
        top_gap = FINAL_FEATURES[top_gap_idx].replace("_freq","").replace("_"," ")
    else:
        top_gap = "none identified"

    record = ""
    if t is not None:
        wins   = t.get("overallWins")
        losses = t.get("overallLosses")
        net    = t.get("netRanking")
        if wins is not None and losses is not None:
            record = f"{int(wins)}-{int(losses)}"
            if net is not None:
                record += f", NET #{int(net)}"

    prompt = f"""You are an expert college basketball analyst. Analyze this player-team transfer fit in 3-4 sentences. Be specific and analytical — reference the actual numbers. No fluff.   

Player: {player['fullName']} | {player.get('position','N/A')} | {player.get('classYr','N/A')} | From: {player.get('teamFullName','N/A')}
Stats: {fmt(player.get('ptsScoredPg'))} pts, {fmt(player.get('rebPg'))} reb, {fmt(player.get('astPg'))} ast | FG {fmt(player.get('fgPct'), pct=True)}, 3P {fmt(player.get('fg3Pct'), pct=True)}, TS% {fmt(player.get('tsPct'), pct=True)}
Shot profile: {shot_profile}

Team: {team['fullName']} | {record}
Offensive profile: {team_offense}
Biggest roster gap: {top_gap}

Match scores: Overall {score['FinalScore']*100:.1f}/100 | Shot Fit {score['ShotFit']*100:.1f}/100 | Gap Fill {score['GapFit']*100:.1f}/100 | Efficiency {score['ContextEff']*100:.1f}/100

Explain specifically why this player does or doesn't fit this team — connect the shot profile to the team's style, the gap fill score to the roster need, and the efficiency score to how the player would contribute.
DO NOT MAKE UP ANY INFORMATION. ONLY USE INFORMATION AND STATS YOU ARE PROVIDED TO CREATE THE REPORT. FOR EXAMPLE, DO NOT MENTION ANY PLAYERS ON THE TEAM OR ANYTHING ABOUT THE COACH IF YOU ARE NOT GIVEN THAT INFORMATION. 
Write the analysis now:"""

    try:
        response = mistral.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        justification = response.choices[0].message.content.strip()
    except Exception:
        justification = None

    def _to_py(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
        return v

    result = {
        "Player":       player["fullName"],
        "PlayerId":     _to_py(player.get("playerId")),
        "PrevTeamId":   _to_py(player.get("teamId")),
        "Position":     player.get("position"),
        "Year":         player.get("classYr"),
        "PrevTeam":     player.get("teamFullName"),
        "NilValue":     int(player["nilValue"]) if pd.notna(player.get("nilValue")) else None,
        "NilTier":      player.get("nilTier"),
        "Team":         team["fullName"],
        "TeamId":       _to_py(tid),
        "Conference":   team.get("conferenceLongName") or team.get("conferenceId"),
        **{c: round(float(player[c]), 3) if pd.notna(player.get(c)) else None for c in STAT_COLS},
        **{k: _to_py(v) for k, v in score.items()},
        "Justification": justification,
    }
    return jsonify(result)

# -----------------------------
# NATURAL LANGUAGE SEARCH
# -----------------------------
PLAYER_STAT_FIELDS = {
    "ptsScoredPg": "points per game",
    "astPg": "assists per game",
    "rebPg": "rebounds per game",
    "blkPg": "blocks per game",
    "stlPg": "steals per game",
    "tovPg": "turnovers per game",
    "fgPct": "field goal percentage (0-1)",
    "fg2Pct": "two-point percentage (0-1)",
    "fg3Pct": "three-point percentage (0-1)",
    "ftPct": "free throw percentage (0-1)",
    "tsPct": "true shooting percentage (0-1)",
    "mpg": "minutes per game",
    "rim_freq": "frequency of shots at the rim (0-1)",
    "paint_freq": "frequency of paint/close shots (0-1)",
    "midrange_freq": "frequency of mid-range shots (0-1)",
    "corner3_freq": "frequency of corner three shots (0-1)",
    "atb3_freq": "frequency of above-the-break three shots (0-1)",
    "deep3_freq": "frequency of deep three shots (0-1)",
}

TEAM_STAT_FIELDS = {
    "ptsScoredPg": "points scored per game",
    "fgPct": "field goal percentage (0-1)",
    "fg3Pct": "three-point percentage (0-1)",
    "efgPct": "effective FG percentage (0-1)",
    "rebPg": "rebounds per game",
    "astPg": "assists per game",
    "tovPg": "turnovers per game",
    "ortg": "offensive rating",
    "drtg": "defensive rating",
    "netRtg": "net rating",
    "pace": "pace (possessions per 40 min)",
    "rim_freq": "frequency of rim/at-basket shots (0-1)",
    "paint_freq": "frequency of paint shots (0-1)",
    "midrange_freq": "frequency of mid-range shots (0-1)",
    "corner3_freq": "frequency of corner three shots (0-1)",
    "atb3_freq": "frequency of above-the-break threes (0-1)",
    "deep3_freq": "frequency of deep threes (0-1)",
}

CONFERENCE_ALIASES = {
    "SEC": "Southeastern Conference",
    "ACC": "Atlantic Coast Conference",
    "Big Ten": "Big Ten Conference",
    "Big 12": "Big 12 Conference",
    "Pac-12": "Pac-12 Conference",
    "American": "American Athletic Conference",
    "Mountain West": "Mountain West Conference",
    "MAC": "Mid-American Conference",
    "Sun Belt": "Sun Belt Conference",
    "CUSA": "Conference USA",
    "WCC": "West Coast Conference",
    "A-10": "Atlantic 10 Conference",
    "MVC": "Missouri Valley Conference",
    "Big East": "Big East Conference",
    "WAC": "Western Athletic Conference",
    "Big West": "Big West Conference",
    "CAA": "Coastal Athletic Association",
    "OVC": "Ohio Valley Conference",
    "SoCon": "Southern Conference",
    "SWAC": "Southwestern Athletic Conference",
    "MEAC": "Mid-Eastern Athletic Conference",
    "MAAC": "Metro Atlantic Athletic Conference",
    "Ivy": "Ivy League",
    "Patriot": "Patriot League",
    "NEC": "Northeast Conference",
    "AEC": "America East Conference",
    "Big South": "Big South Conference",
    "Horizon": "Horizon League",
    "Summit": "Summit League",
    "ASUN": "ASUN Conference",
}

@app.route("/natural_search", methods=["POST"])
def natural_search():
    body = request.get_json(silent=True) or {}
    query   = body.get("query", "").strip()
    gender  = body.get("gender", "MALE").upper()
    target  = body.get("target", "players")  # "players" or "teams"

    if not query:
        return jsonify({"error": "No query provided"}), 400

    d = _get_data(gender)

    if target == "players":
        stat_schema = "\n".join(f"  - {k}: {v}" for k, v in PLAYER_STAT_FIELDS.items())
        system_prompt = f"""You are a college basketball data analyst. Convert a natural language player search query into a JSON filter object.

Available filter fields:
  - conferences: list of full conference names (e.g. ["Southeastern Conference", "Big Ten Conference"])
  - positions: list of position strings — use substrings like ["G"] for guards, ["F"] for forwards, ["C"] for centers. Can combine: ["G", "F"]
  - height_min: minimum height in inches (e.g. 76 = 6'4")
  - height_max: maximum height in inches
  - class_years: list of class year strings, e.g. ["Junior", "Senior", "Graduate"]
  - stat_filters: dict of field → {{"min": X, "max": Y}} using these fields:
{stat_schema}
  - sort_by: list of field names to sort descending (best matches first), max 2 fields
  - description: one-sentence plain English summary of what you understood

Conference aliases: {json.dumps(CONFERENCE_ALIASES)}

Rules:
- Only include fields that are clearly implied by the query. Omit everything else.
- For percentage fields (fgPct, fg3Pct, tsPct etc.) use 0-1 scale, not 0-100.
- For shot profile freqs (rim_freq, corner3_freq etc.) "a lot" ≈ min 0.15, "very often" ≈ min 0.20
- For fg3Pct: "efficient from three" ≈ min 0.36, "very efficient" ≈ min 0.39
- For tsPct: "efficient scorer" ≈ min 0.56, "very efficient" ≈ min 0.60
- "tall guard" = position G, height_min ~75 (6'3")
- Return ONLY valid JSON, no markdown, no explanation.

Example output:
{{"conferences": ["Southeastern Conference"], "positions": ["G"], "height_min": 75, "stat_filters": {{"fg3Pct": {{"min": 0.37}}, "corner3_freq": {{"min": 0.12}}}}, "sort_by": ["fg3Pct", "tsPct"], "description": "Tall guards from the SEC who shoot efficiently from three"}}"""

    else:
        stat_schema = "\n".join(f"  - {k}: {v}" for k, v in TEAM_STAT_FIELDS.items())
        system_prompt = f"""You are a college basketball data analyst. Convert a natural language team search query into a JSON filter object.

Available filter fields:
  - conferences: list of full conference names (e.g. ["Southeastern Conference"])
  - stat_filters: dict of field → {{"min": X, "max": Y}} using these fields:
{stat_schema}
  - sort_by: list of field names to sort descending (best matches first), max 2 fields
  - description: one-sentence plain English summary of what you understood

Conference aliases: {json.dumps(CONFERENCE_ALIASES)}

Rules:
- Only include fields that are clearly implied by the query.
- For percentage fields use 0-1 scale.
- For shot freqs: "a lot" ≈ min 0.15, "very often" ≈ min 0.20
- "attack the rim" → rim_freq min ~0.20, "shoot corner threes" → corner3_freq min ~0.12
- "fast pace" → pace min ~72, "slow" → pace max ~68
- "elite offense" → ortg min ~115, "good defense" → drtg max ~105
- Return ONLY valid JSON, no markdown, no explanation.

Example output:
{{"conferences": ["Southeastern Conference"], "stat_filters": {{"corner3_freq": {{"min": 0.12}}, "rim_freq": {{"min": 0.20}}}}, "sort_by": ["corner3_freq", "rim_freq"], "description": "SEC teams that attack the rim and shoot corner threes"}}"""

    try:
        response = mistral.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        criteria = json.loads(raw)
    except Exception as e:
        return jsonify({"error": f"Could not parse query: {e}"}), 500

    description = criteria.get("description", query)

    if target == "players":
        pool = d["playerdf"].copy()

        # playerdf only has conferenceId — build name mapping from teamdf
        conf_map = d["teamdf"].drop_duplicates("conferenceId").set_index("conferenceId")["conferenceLongName"].to_dict()
        pool["conferenceLongName"] = pool["conferenceId"].map(conf_map)

        conferences = criteria.get("conferences", [])
        if conferences:
            pool = pool[pool["conferenceLongName"].isin(conferences)]

        positions = criteria.get("positions", [])
        if positions:
            mask = pool["position"].astype(str).str.contains("|".join(positions), case=False, na=False)
            pool = pool[mask]

        h_min = criteria.get("height_min")
        h_max = criteria.get("height_max")
        if h_min is not None and "height" in pool.columns:
            pool = pool[pd.to_numeric(pool["height"], errors="coerce").fillna(0) >= h_min]
        if h_max is not None and "height" in pool.columns:
            pool = pool[pd.to_numeric(pool["height"], errors="coerce").fillna(0) <= h_max]

        class_years = criteria.get("class_years", [])
        if class_years:
            mask = pool["classYr"].astype(str).str.lower().str.contains(
                "|".join(y.lower() for y in class_years), na=False
            )
            pool = pool[mask]

        for field, bounds in criteria.get("stat_filters", {}).items():
            if field not in pool.columns:
                continue
            col = pd.to_numeric(pool[field], errors="coerce")
            if "min" in bounds:
                pool = pool[col >= bounds["min"]]
            if "max" in bounds:
                pool = pool[col <= bounds["max"]]

        sort_fields = [f for f in criteria.get("sort_by", []) if f in pool.columns]
        if sort_fields:
            pool = pool.sort_values(sort_fields, ascending=False)

        records = []
        for _, p in pool.head(25).iterrows():
            records.append({
                "Player":     p["fullName"],
                "PlayerId":   int(p["playerId"]) if pd.notna(p.get("playerId")) else None,
                "PrevTeamId": int(p["teamId"])   if pd.notna(p.get("teamId"))   else None,
                "Position":   p.get("position"),
                "Year":       p.get("classYr"),
                "PrevTeam":   p.get("teamFullName"),
                "Conference": p.get("conferenceLongName"),
                "NilTier":    p.get("nilTier"),
                "NilValue":   int(p["nilValue"]) if pd.notna(p.get("nilValue")) else None,
                **{c: round(float(p[c]), 3) if pd.notna(p.get(c)) else None for c in STAT_COLS},
                "FinalScore": 0,
            })

        return jsonify({"description": description, "results": records, "target": "players"})

    else:
        # Teams
        teamdf   = d["teamdf"].copy()
        teamstatsdf = d["teamstatsdf"].copy()
        ts_index = teamstatsdf.set_index("teamId")

        pool = teamdf.copy()

        conferences = criteria.get("conferences", [])
        if conferences:
            pool = pool[pool["conferenceLongName"].isin(conferences)]

        # Merge shot profile into pool for shot freq filters
        for field, bounds in criteria.get("stat_filters", {}).items():
            if field in pool.columns:
                col = pd.to_numeric(pool[field], errors="coerce")
                if "min" in bounds:
                    pool = pool[col >= bounds["min"]]
                if "max" in bounds:
                    pool = pool[col <= bounds["max"]]
            elif field in TEAM_STAT_COLS:
                # Filter on teamstatsdf then restrict pool
                valid_tids = set()
                for tid, row in ts_index.iterrows():
                    val = row.get(field)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    ok = True
                    if "min" in bounds and float(val) < bounds["min"]:
                        ok = False
                    if "max" in bounds and float(val) > bounds["max"]:
                        ok = False
                    if ok:
                        valid_tids.add(tid)
                pool = pool[pool["teamId"].isin(valid_tids)]

        sort_fields = [f for f in criteria.get("sort_by", []) if f in pool.columns]
        if sort_fields:
            pool = pool.sort_values(sort_fields, ascending=False)
        elif not sort_fields:
            # sort by first stat-filter field in teamstatsdf if available
            sort_candidates = [f for f in criteria.get("sort_by", []) if f in ts_index.columns]
            if sort_candidates:
                pool = pool.copy()
                pool["_sort"] = pool["teamId"].map(
                    lambda tid: float(ts_index.loc[tid, sort_candidates[0]])
                    if tid in ts_index.index and pd.notna(ts_index.loc[tid, sort_candidates[0]])
                    else 0
                )
                pool = pool.sort_values("_sort", ascending=False)

        records = []
        for _, team in pool.head(25).iterrows():
            tid = team["teamId"]
            team_stats = {}
            if tid in ts_index.index:
                ts = ts_index.loc[tid]
                for col in TEAM_STAT_COLS:
                    val = ts.get(col) if hasattr(ts, "get") else (ts[col] if col in ts.index else None)
                    team_stats[col] = round(float(val), 3) if val is not None and pd.notna(val) else None
            records.append({
                "Team":       team["fullName"],
                "TeamId":     int(tid) if pd.notna(tid) else None,
                "Conference": team.get("conferenceLongName") or team.get("conferenceId"),
                **team_stats,
                "FinalScore": 0,
            })

        return jsonify({"description": description, "results": records, "target": "teams"})

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

_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_team_stats",
            "description": "Get season shot profile for a team. Use partial name e.g. 'Maryland'.",
            "parameters": {
                "type": "object",
                "properties": {"team_name": {"type": "string"}},
                "required": ["team_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_season_stats",
            "description": "Get per-game stats and shooting splits for a player. Use partial name.",
            "parameters": {
                "type": "object",
                "properties": {"player_name": {"type": "string"}},
                "required": ["player_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_pbp_stats",
            "description": "Get shot zone frequencies and PBP-derived stats for a player.",
            "parameters": {
                "type": "object",
                "properties": {"player_name": {"type": "string"}},
                "required": ["player_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_game_log",
            "description": "Get recent team game results for a conference. Default conference_id 53 = Big Ten.",
            "parameters": {
                "type": "object",
                "properties": {"conference_id": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_game_log",
            "description": "Get individual game-by-game stats for a player by name.",
            "parameters": {
                "type": "object",
                "properties": {"player_name": {"type": "string"}},
                "required": ["player_name"],
            },
        },
    },
]

_AGENT_SYSTEM = (
    "You are an expert college basketball analyst with access to real current season data. "
    "Use tools to look up data before answering. Be concise, specific, and analytical."
)

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json()
    user_message = body.get("message", "").strip()
    history_raw = body.get("history", [])
    gender = body.get("gender", "MALE").upper()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    tool_fns = _make_tool_fns(gender)

    # Build conversation history for Mistral
    messages = [{"role": "system", "content": _AGENT_SYSTEM}]
    for msg in history_raw:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        # Agentic loop: up to 4 tool-call rounds
        for _ in range(4):
            response = mistral.chat.complete(
                model=MISTRAL_MODEL,
                messages=messages,
                tools=_AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
            assistant_msg = response.choices[0].message

            if not assistant_msg.tool_calls:
                return jsonify({"response": assistant_msg.content})

            # Append assistant message with tool calls
            messages.append({"role": "assistant", "content": assistant_msg.content, "tool_calls": assistant_msg.tool_calls})

            # Execute each tool call and append results
            for tc in assistant_msg.tool_calls:
                fn = tool_fns.get(tc.function.name)
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                tool_result = fn(**args) if fn else f"Unknown tool: {tc.function.name}"
                messages.append({
                    "role": "tool",
                    "name": tc.function.name,
                    "content": tool_result,
                    "tool_call_id": tc.id,
                })

        return jsonify({"response": "Max tool iterations reached."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5002)