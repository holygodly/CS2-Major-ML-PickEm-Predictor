"""
Quick playoff simulation baseline.

This runner uses only ELO plus mild map-to-map randomness. It is intentionally
fast, but it now preserves BO5 finals and reports matchup-conditional win rates.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _is_auto_count(value):
    if value in (None, 0, ""):
        return True
    return isinstance(value, str) and value.strip().lower() in {"0", "auto", "none", "null"}


def _resolve_num_simulations(simulation_cfg, default=50000):
    value = simulation_cfg.get("playoff_num_simulations")
    if _is_auto_count(value):
        value = simulation_cfg.get("num_simulations")
    if _is_auto_count(value):
        return default
    return int(value)


def calculate_elo_from_data(config, base_dir=Path(".")):
    df = pd.read_csv(base_dir / "data" / "map_level_dataset_with_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_url", "map_index"])

    base_elo = 1500
    time_decay_days = 50
    teams = set(df["team1"].unique()) | set(df["team2"].unique())

    initial_ratings = {team: base_elo for team in teams}
    source_data_dir = Path(config["data"]["data_dir"])
    if not source_data_dir.is_absolute():
        source_data_dir = (base_dir / source_data_dir).resolve()
    player_stats_path = source_data_dir / config["data"]["player_stats_pattern"]

    try:
        player_stats = pd.read_csv(player_stats_path, on_bad_lines="skip")
        both_side = player_stats[player_stats["side"] == "Both"]
        for team in teams:
            team_players = both_side[both_side["team"] == team].tail(50)
            if team_players.empty:
                continue
            avg_rating = team_players["rating"].mean()
            adjustment = max(-200, min(200, (avg_rating - 1.0) * 500))
            initial_ratings[team] = base_elo + adjustment
    except Exception:
        pass

    elo = initial_ratings.copy()
    match_results = (
        df.groupby("match_url")
        .agg(
            team1=("team1", "first"),
            team2=("team2", "first"),
            winner=("winner", "sum"),
            date=("date", "first"),
            match_type=("match_type", "first"),
        )
        .reset_index()
        .sort_values("date")
    )
    latest_date = match_results["date"].max()

    team_match_count = defaultdict(int)

    for _, row in match_results.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        match_type = row["match_type"]

        maps_to_win = {"BO1": 1, "BO3": 2, "BO5": 3}.get(match_type, 2)
        team1_match_winner = int(int(row["winner"]) >= maps_to_win)

        count1 = team_match_count[team1]
        count2 = team_match_count[team2]
        k1 = 50 if count1 < 15 else (40 if count1 < 30 else 30)
        k2 = 50 if count2 < 15 else (40 if count2 < 30 else 30)
        adaptive_k = (k1 + k2) / 2

        days_ago = (latest_date - row["date"]).days
        time_weight = math.exp(-days_ago / time_decay_days)
        format_weight = {"BO1": 1.0, "BO3": 1.2, "BO5": 1.5}.get(match_type, 1.0)
        k_factor = adaptive_k * format_weight * time_weight

        expected_team1 = 1 / (1 + math.pow(10, (elo[team2] - elo[team1]) / 400))
        elo[team1] += k_factor * (team1_match_winner - expected_team1)
        elo[team2] += k_factor * ((1 - team1_match_winner) - (1 - expected_team1))

        team_match_count[team1] += 1
        team_match_count[team2] += 1

    return elo


def simulate_series_simple(elo1, elo2, match_type="BO3"):
    base_prob = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    maps_to_win = 2 if match_type == "BO3" else 3
    max_maps = 3 if match_type == "BO3" else 5

    score = [0, 0]
    for _ in range(max_maps):
        if score[0] == maps_to_win or score[1] == maps_to_win:
            break
        map_prob = float(np.clip(base_prob + np.random.normal(0, 0.05), 0.10, 0.90))
        if np.random.random() < map_prob:
            score[0] += 1
        else:
            score[1] += 1

    return 1 if score[0] > score[1] else 0


def run_quick_prediction(config_path="config.yaml"):
    config_path_obj = Path(config_path).resolve()
    base_dir = config_path_obj.parent
    config = load_config(config_path)

    tournament_cfg = config.get("tournament", {})
    simulation_cfg = config.get("simulation", {})
    quarterfinal_pairs = tournament_cfg.get("quarterfinals", [])
    semifinal_pairs = tournament_cfg.get("semifinal_pairs", [[0, 3], [1, 2]])
    num_simulations = _resolve_num_simulations(simulation_cfg)
    tournament_name = tournament_cfg.get("name", "Playoff Prediction")

    print("=" * 70)
    print(tournament_name)
    print("=" * 70)

    print("[1/3] Calculating team ELO...")
    team_elo = calculate_elo_from_data(config, base_dir=base_dir)

    quarterfinals = []
    playoff_teams = []
    for team1, team2 in quarterfinal_pairs:
        quarterfinals.append(
            {
                "team1": team1,
                "team2": team2,
                "elo1": team_elo.get(team1, 1500),
                "elo2": team_elo.get(team2, 1500),
            }
        )
        if team1 not in playoff_teams:
            playoff_teams.append(team1)
        if team2 not in playoff_teams:
            playoff_teams.append(team2)

    print()
    for team in playoff_teams:
        print(f"  {team:<18} {team_elo.get(team, 1500):.0f}")

    print()
    print(f"[2/3] Running Monte Carlo simulation ({num_simulations:,})...")

    qf_wins = {idx: {qf['team1']: 0, qf['team2']: 0} for idx, qf in enumerate(quarterfinals)}
    sf_wins = {team: 0 for team in playoff_teams}
    final_wins = {team: 0 for team in playoff_teams}
    champion_count = {team: 0 for team in playoff_teams}

    sf_matchups = {idx: defaultdict(int) for idx in range(len(semifinal_pairs))}
    sf_match_wins = {idx: defaultdict(lambda: defaultdict(int)) for idx in range(len(semifinal_pairs))}
    final_matchups = defaultdict(int)
    final_match_wins = defaultdict(lambda: defaultdict(int))

    for sim_idx in range(num_simulations):
        if (sim_idx + 1) % 10000 == 0:
            print(f"  Progress: {sim_idx + 1:,}/{num_simulations:,}")

        qf_results = []
        for qf_idx, qf in enumerate(quarterfinals):
            winner = simulate_series_simple(qf["elo1"], qf["elo2"], match_type="BO3")
            winner_team = qf["team1"] if winner == 1 else qf["team2"]
            winner_elo = qf["elo1"] if winner == 1 else qf["elo2"]
            qf_wins[qf_idx][winner_team] += 1
            qf_results.append({"team": winner_team, "elo": winner_elo})

        sf_results = []
        for sf_idx, (left_idx, right_idx) in enumerate(semifinal_pairs):
            team1 = qf_results[left_idx]["team"]
            team2 = qf_results[right_idx]["team"]
            elo1 = qf_results[left_idx]["elo"]
            elo2 = qf_results[right_idx]["elo"]
            matchup_key = tuple(sorted([team1, team2]))
            sf_matchups[sf_idx][matchup_key] += 1

            winner = simulate_series_simple(elo1, elo2, match_type="BO3")
            winner_team = team1 if winner == 1 else team2
            winner_elo = elo1 if winner == 1 else elo2

            sf_wins[winner_team] += 1
            sf_match_wins[sf_idx][matchup_key][winner_team] += 1
            sf_results.append({"team": winner_team, "elo": winner_elo})

        final_team1 = sf_results[0]["team"]
        final_team2 = sf_results[1]["team"]
        final_elo1 = sf_results[0]["elo"]
        final_elo2 = sf_results[1]["elo"]
        final_key = tuple(sorted([final_team1, final_team2]))
        final_matchups[final_key] += 1

        winner = simulate_series_simple(final_elo1, final_elo2, match_type="BO5")
        winner_team = final_team1 if winner == 1 else final_team2
        final_wins[winner_team] += 1
        champion_count[winner_team] += 1
        final_match_wins[final_key][winner_team] += 1

    print("[3/3] Results")
    print()

    print("Quarterfinals")
    print("-" * 70)
    for qf_idx, qf in enumerate(quarterfinals):
        team1 = qf["team1"]
        team2 = qf["team2"]
        prob1 = qf_wins[qf_idx][team1] / num_simulations * 100
        prob2 = qf_wins[qf_idx][team2] / num_simulations * 100
        print(f"QF{qf_idx + 1}: {team1} {prob1:.1f}% vs {team2} {prob2:.1f}%")

    print()
    print("Semifinals")
    print("-" * 70)
    for sf_idx in range(len(semifinal_pairs)):
        sorted_matchups = sorted(sf_matchups[sf_idx].items(), key=lambda item: item[1], reverse=True)
        if not sorted_matchups:
            continue
        matchup_key, matchup_count = sorted_matchups[0]
        team1, team2 = matchup_key
        team1_wins = sf_match_wins[sf_idx][matchup_key].get(team1, 0)
        team2_wins = sf_match_wins[sf_idx][matchup_key].get(team2, 0)
        total = max(matchup_count, 1)
        print(f"SF{sf_idx + 1}: {team1} vs {team2}")
        print(f"  Matchup Probability: {matchup_count / num_simulations * 100:.1f}%")
        print(f"  Conditional Win Probability: {team1_wins / total * 100:.1f}% vs {team2_wins / total * 100:.1f}%")

    print()
    print("Grand Final")
    print("-" * 70)
    sorted_finals = sorted(final_matchups.items(), key=lambda item: item[1], reverse=True)
    if sorted_finals:
        final_key, final_count = sorted_finals[0]
        team1, team2 = final_key
        team1_wins = final_match_wins[final_key].get(team1, 0)
        team2_wins = final_match_wins[final_key].get(team2, 0)
        total = max(final_count, 1)
        print(f"Most Likely Final: {team1} vs {team2}")
        print(f"  Matchup Probability: {final_count / num_simulations * 100:.1f}%")
        print(f"  Conditional Win Probability: {team1_wins / total * 100:.1f}% vs {team2_wins / total * 100:.1f}%")

    print()
    print("Championship Ranking")
    print("-" * 70)
    sorted_champions = sorted(champion_count.items(), key=lambda item: item[1], reverse=True)
    for rank, (team, count) in enumerate(sorted_champions, start=1):
        probability = count / num_simulations * 100
        print(f"{rank:>2}. {team:<18} {probability:6.2f}% [ELO: {team_elo.get(team, 1500):.0f}]")

    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_simulations": num_simulations,
        "tournament_name": tournament_name,
        "championship_probabilities": {
            team: count / num_simulations for team, count in sorted_champions
        },
        "predicted_champion": sorted_champions[0][0],
    }

    output_path = base_dir / "output" / "playoff_predictions_quick.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    run_quick_prediction()
