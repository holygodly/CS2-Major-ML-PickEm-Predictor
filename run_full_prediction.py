"""
Full playoff simulation entrypoint.

Reads config.yaml, runs Monte Carlo playoff simulations, and saves a structured
summary to output/playoff_predictions_full.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from hybrid_playoff_predictor import HybridPlayoffPredictor


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _normalize_matchup_key(team1, team2):
    return tuple(sorted([team1, team2]))


def _combo_from_veto(veto_info):
    if not veto_info:
        return None
    return tuple(map_info["map"] for map_info in veto_info["maps"])


def _is_auto_count(value):
    if value in (None, 0, ""):
        return True
    return isinstance(value, str) and value.strip().lower() in {"0", "auto", "none", "null"}


def _resolve_num_simulations(simulation_cfg, default=5000):
    value = simulation_cfg.get("playoff_num_simulations")
    if _is_auto_count(value):
        value = simulation_cfg.get("num_simulations")
    if _is_auto_count(value):
        return default
    return int(value)


def run_full_prediction(config_path="config.yaml"):
    config = load_config(config_path)
    tournament_cfg = config.get("tournament", {})
    simulation_cfg = config.get("simulation", {})

    quarterfinal_pairs = tournament_cfg.get("quarterfinals", [])
    semifinal_pairs = tournament_cfg.get("semifinal_pairs", [[0, 3], [1, 2]])
    num_simulations = _resolve_num_simulations(simulation_cfg)
    tournament_name = tournament_cfg.get("name", "Playoff Prediction")

    print(f"Simulations: {num_simulations:,}")
    print(f"Tournament: {tournament_name}")

    predictor = HybridPlayoffPredictor(config_path)

    quarterfinals = []
    playoff_teams = []
    for team1, team2 in quarterfinal_pairs:
        quarterfinals.append(
            {
                "team1": team1,
                "team2": team2,
                "elo1": predictor.team_elo.get(team1, 1500),
                "elo2": predictor.team_elo.get(team2, 1500),
            }
        )
        if team1 not in playoff_teams:
            playoff_teams.append(team1)
        if team2 not in playoff_teams:
            playoff_teams.append(team2)

    qf_wins = {team: 0 for team in playoff_teams}
    sf_wins = {team: 0 for team in playoff_teams}
    final_wins = {team: 0 for team in playoff_teams}
    champion_count = {team: 0 for team in playoff_teams}

    qf_map_combos = {idx: defaultdict(int) for idx in range(len(quarterfinals))}
    sf_map_combos = {idx: defaultdict(int) for idx in range(len(semifinal_pairs))}
    final_map_combos = defaultdict(int)

    sf_matchups = {idx: defaultdict(int) for idx in range(len(semifinal_pairs))}
    sf_match_wins = {idx: defaultdict(lambda: defaultdict(int)) for idx in range(len(semifinal_pairs))}
    final_matchups = defaultdict(int)
    final_match_wins = defaultdict(lambda: defaultdict(int))

    for sim_idx in range(num_simulations):
        if (sim_idx + 1) % 2000 == 0:
            print(f"  Completed {sim_idx + 1:,}/{num_simulations:,}")

        qf_results = []
        for qf_idx, qf in enumerate(quarterfinals):
            winner, _, veto_info = predictor.simulate_bo3(
                qf["team1"],
                qf["team2"],
                elo_team1=qf["elo1"],
                elo_team2=qf["elo2"],
                use_veto=True,
            )
            winner_team = qf["team1"] if winner == 1 else qf["team2"]
            winner_elo = qf["elo1"] if winner == 1 else qf["elo2"]
            qf_wins[winner_team] += 1

            combo = _combo_from_veto(veto_info)
            if combo:
                qf_map_combos[qf_idx][combo] += 1

            qf_results.append({"team": winner_team, "elo": winner_elo})

        sf_results = []
        for sf_idx, (left_idx, right_idx) in enumerate(semifinal_pairs):
            team1 = qf_results[left_idx]["team"]
            team2 = qf_results[right_idx]["team"]
            elo1 = qf_results[left_idx]["elo"]
            elo2 = qf_results[right_idx]["elo"]

            matchup_key = _normalize_matchup_key(team1, team2)
            sf_matchups[sf_idx][matchup_key] += 1

            winner, _, veto_info = predictor.simulate_bo3(
                team1,
                team2,
                elo_team1=elo1,
                elo_team2=elo2,
                use_veto=True,
            )
            winner_team = team1 if winner == 1 else team2
            winner_elo = elo1 if winner == 1 else elo2
            sf_wins[winner_team] += 1
            sf_match_wins[sf_idx][matchup_key][winner_team] += 1

            combo = _combo_from_veto(veto_info)
            if combo:
                sf_map_combos[sf_idx][combo] += 1

            sf_results.append({"team": winner_team, "elo": winner_elo})

        final_team1 = sf_results[0]["team"]
        final_team2 = sf_results[1]["team"]
        final_elo1 = sf_results[0]["elo"]
        final_elo2 = sf_results[1]["elo"]

        final_key = _normalize_matchup_key(final_team1, final_team2)
        final_matchups[final_key] += 1

        winner, _, veto_info = predictor.simulate_bo5(
            final_team1,
            final_team2,
            elo_team1=final_elo1,
            elo_team2=final_elo2,
            use_veto=True,
        )
        winner_team = final_team1 if winner == 1 else final_team2
        final_wins[winner_team] += 1
        champion_count[winner_team] += 1
        final_match_wins[final_key][winner_team] += 1

        combo = _combo_from_veto(veto_info)
        if combo:
            final_map_combos[combo] += 1

    print("Simulation complete")
    print()

    print("Quarterfinals")
    print("-" * 70)
    for qf_idx, qf in enumerate(quarterfinals):
        team1 = qf["team1"]
        team2 = qf["team2"]
        team1_prob = qf_wins[team1] / num_simulations * 100
        team2_prob = qf_wins[team2] / num_simulations * 100
        print(f"QF{qf_idx + 1}: {team1} {team1_prob:.1f}% vs {team2} {team2_prob:.1f}%")

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
        total_matchup_games = max(matchup_count, 1)

        print(f"SF{sf_idx + 1}: {team1} vs {team2}")
        print(f"  Matchup Probability: {matchup_count / num_simulations * 100:.1f}%")
        print(f"  Conditional Win Probability: {team1_wins / total_matchup_games * 100:.1f}% vs {team2_wins / total_matchup_games * 100:.1f}%")

    print()
    print("Grand Final")
    print("-" * 70)
    sorted_finals = sorted(final_matchups.items(), key=lambda item: item[1], reverse=True)
    if sorted_finals:
        final_key, final_count = sorted_finals[0]
        team1, team2 = final_key
        team1_wins = final_match_wins[final_key].get(team1, 0)
        team2_wins = final_match_wins[final_key].get(team2, 0)
        total_finals = max(final_count, 1)

        print(f"Most Likely Final: {team1} vs {team2}")
        print(f"  Matchup Probability: {final_count / num_simulations * 100:.1f}%")
        print(f"  Conditional Win Probability: {team1_wins / total_finals * 100:.1f}% vs {team2_wins / total_finals * 100:.1f}%")

    print()
    print("Championship Ranking")
    print("-" * 70)
    sorted_champions = sorted(champion_count.items(), key=lambda item: item[1], reverse=True)
    for rank, (team, count) in enumerate(sorted_champions, start=1):
        probability = count / num_simulations * 100
        elo = predictor.team_elo.get(team, 1500)
        print(f"{rank:>2}. {team:<18} {probability:6.2f}% [ELO: {elo:.0f}]")

    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_simulations": num_simulations,
        "tournament_name": tournament_name,
        "quarterfinals": quarterfinals,
        "championship_probabilities": {
            team: count / num_simulations for team, count in sorted_champions
        },
        "predicted_champion": sorted_champions[0][0],
        "predicted_final": list(sorted_finals[0][0]) if sorted_finals else [],
        "quarterfinal_map_combos": {
            str(idx): {", ".join(combo): count / num_simulations for combo, count in combos.items()}
            for idx, combos in qf_map_combos.items()
        },
        "semifinal_map_combos": {
            str(idx): {", ".join(combo): count / num_simulations for combo, count in combos.items()}
            for idx, combos in sf_map_combos.items()
        },
        "final_map_combos": {
            ", ".join(combo): count / num_simulations for combo, count in final_map_combos.items()
        },
    }

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "playoff_predictions_full.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    run_full_prediction()
