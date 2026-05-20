"""
Swiss-stage ML prediction runner.

This script keeps the existing pure-code Swiss system untouched and reuses:
- the same seeded teams by default
- the same Buchholz pairing rules

Round 1 matchups can be overridden manually in swiss_stage_config.yaml.

Only the match-outcome engine changes: it uses the trained ML model plus veto,
side bias, and ELO fallback from the hybrid predictor.
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import yaml

from hybrid_playoff_predictor import HybridPlayoffPredictor
from pickem_optimizer import PickemOptimizer


def load_yaml_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


PAIRING_PRIORITY = [
    [(0, 5), (1, 4), (2, 3)],
    [(0, 5), (1, 3), (2, 4)],
    [(0, 4), (1, 5), (2, 3)],
    [(0, 4), (1, 3), (2, 5)],
    [(0, 3), (1, 5), (2, 4)],
    [(0, 3), (1, 4), (2, 5)],
    [(0, 5), (1, 2), (3, 4)],
    [(0, 4), (1, 2), (3, 5)],
    [(0, 2), (1, 5), (3, 4)],
    [(0, 2), (1, 4), (3, 5)],
    [(0, 3), (1, 2), (4, 5)],
    [(0, 2), (1, 3), (4, 5)],
    [(0, 1), (2, 5), (3, 4)],
    [(0, 1), (2, 4), (3, 5)],
    [(0, 1), (2, 3), (4, 5)],
]


def load_swiss_reference():
    root_dir = Path(__file__).resolve().parents[1]
    reference_path = root_dir / "cs2_major_prediction_system" / "cs2_swiss_predictor_cpu.py"

    if not reference_path.exists():
        return {
            "root_dir": root_dir,
            "seeded_teams": [],
            "teams": [],
            "round1_matchups": [],
            "team_seeds": {},
            "source": "config.yaml (reference file not found)",
        }

    spec = importlib.util.spec_from_file_location("swiss_reference", reference_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    seeded_teams = list(module.SEEDED_TEAMS)
    round1_matchups = list(module.ROUND1_MATCHUPS)
    return {
        "root_dir": root_dir,
        "seeded_teams": seeded_teams,
        "teams": list(module.TEAMS),
        "round1_matchups": round1_matchups,
        "team_seeds": {team: idx + 1 for idx, team in enumerate(seeded_teams)},
        "source": str(reference_path),
    }


def validate_round1_matchups(seeded_teams, round1_matchups):
    if len(round1_matchups) != len(seeded_teams) // 2:
        raise ValueError("round1_matchups 数量不对，16 支队伍应该正好有 8 场。")

    flattened = []
    for matchup in round1_matchups:
        if len(matchup) != 2:
            raise ValueError("每一场 round1_matchups 都必须正好有两支队伍。")
        flattened.extend(matchup)

    if len(flattened) != len(set(flattened)):
        raise ValueError("round1_matchups 里有队伍重复出现。")

    missing = sorted(set(seeded_teams) - set(flattened))
    extra = sorted(set(flattened) - set(seeded_teams))
    if missing or extra:
        raise ValueError(
            f"round1_matchups 队伍集合不完整。缺少: {missing or '无'}; 额外: {extra or '无'}"
        )


def derive_seeds_from_round1(round1_matchups):
    """从首轮对阵反推种子顺序。

    官方首轮固定是 1v9, 2v10, ..., 8v16:每场第一个队是高种子(1-8),
    第二个队是对应的低种子(9-16)。所以种子顺序 = 所有高种子(按场次顺序)
    接上所有低种子。要求 round1_matchups 按场次 1-8 的顺序排,每场把高种子写在前面。
    """
    high_seeds = [match[0] for match in round1_matchups]   # seeds 1..8
    low_seeds = [match[1] for match in round1_matchups]    # seeds 9..16
    return high_seeds + low_seeds


def load_swiss_stage_setup(reference, setup_path=None, main_config=None, main_config_path=None):
    root_dir = reference["root_dir"]
    main_config = main_config or {}
    if main_config.get("swiss_stage"):
        config = main_config.get("swiss_stage") or {}
        config_path = Path(main_config_path) if main_config_path else Path("config.yaml")
        source = f"{config_path}:swiss_stage"
    else:
        config_path = Path(setup_path) if setup_path else (root_dir / "playoffs_ml_enhanced" / "swiss_stage_config.yaml")

        if not config_path.exists():
            return {
                "seeded_teams": list(reference["seeded_teams"]),
                "teams": list(reference["teams"]),
                "round1_matchups": list(reference["round1_matchups"]),
                "team_seeds": dict(reference["team_seeds"]),
                "source": f"{reference['source']} (fallback for round 1)",
            }

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        source = str(config_path)

    round1_matchups = [tuple(match) for match in config.get("round1_matchups") or reference["round1_matchups"]]
    seeded_teams = list(config.get("seeded_teams") or reference["seeded_teams"])

    # 没填 seeded_teams 就从首轮对阵反推(官方 1v9, 2v10... 规则)
    if not seeded_teams and round1_matchups:
        seeded_teams = derive_seeds_from_round1(round1_matchups)
        print(f"  [info] 未填 seeded_teams,已从首轮对阵反推种子顺序(1v9 规则): {seeded_teams}")

    if len(seeded_teams) != len(set(seeded_teams)):
        raise ValueError("seeded_teams 里有重复队伍。")

    validate_round1_matchups(seeded_teams, round1_matchups)

    return {
        "seeded_teams": seeded_teams,
        "teams": list(seeded_teams),
        "round1_matchups": round1_matchups,
        "team_seeds": {team: idx + 1 for idx, team in enumerate(seeded_teams)},
        "source": source,
    }


def load_num_simulations(root_dir, main_config=None):
    sim_config = (main_config or {}).get("simulation", {}) or {}
    for key in ("swiss_num_simulations", "num_simulations"):
        value = sim_config.get(key)
        if value is not None:
            try:
                int_val = int(value)
                if int_val > 0:
                    return int_val
            except (ValueError, TypeError):
                pass  # 'auto' or invalid → fall through

    batch_config_path = root_dir / "cs2_major_prediction_system" / "batchsize.yaml"
    if not batch_config_path.exists():
        return 500

    try:
        with batch_config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        val = int(config.get("simulation", {}).get("num_simulations", 500))
        return val if val > 0 else 500
    except Exception:
        return 500


def choose_round_2_3_pairs(teams, match_history):
    remaining = teams.copy()
    pairs = []

    while len(remaining) >= 2:
        team1 = remaining.pop(0)
        matched = False

        for idx in range(len(remaining) - 1, -1, -1):
            team2 = remaining[idx]
            if team2 not in match_history[team1]:
                remaining.pop(idx)
                pairs.append((team1, team2))
                matched = True
                break

        if not matched:
            team2 = remaining.pop()
            pairs.append((team1, team2))

    return pairs


def choose_round_4_5_pairs(teams, match_history):
    for priority_pattern in PAIRING_PRIORITY:
        valid = True
        test_pairs = []

        for idx1, idx2 in priority_pattern:
            if idx1 >= len(teams) or idx2 >= len(teams):
                valid = False
                break
            team1 = teams[idx1]
            team2 = teams[idx2]
            if team2 in match_history[team1]:
                valid = False
                break
            test_pairs.append((team1, team2))

        if valid:
            return test_pairs

    fallback_pairs = []
    for idx1, idx2 in PAIRING_PRIORITY[0]:
        if idx1 < len(teams) and idx2 < len(teams):
            fallback_pairs.append((teams[idx1], teams[idx2]))
    return fallback_pairs


def get_buchholz_sorted_groups(records, match_history, active_teams, team_seeds):
    difficulty = {}
    for team in active_teams:
        total = 0
        for opponent in match_history[team]:
            opp_wins, opp_losses = records[opponent]
            total += opp_wins - opp_losses
        difficulty[team] = total

    return sorted(active_teams, key=lambda team: (-difficulty[team], team_seeds.get(team, 999)))


def evaluate_pickem(prediction, all_simulations):
    success_count = 0

    for simulation in all_simulations:
        correct = 0
        correct += len(set(prediction["3-0"]) & simulation["3-0"])
        correct += len(set(prediction["0-3"]) & simulation["0-3"])

        qualified_non_30 = simulation["qualified"] - simulation["3-0"]
        correct += len(set(prediction["advances"]) & qualified_non_30)

        if correct >= 5:
            success_count += 1

    return success_count / len(all_simulations) if all_simulations else 0.0


def recommend_pickem(probabilities, all_simulations):
    ranked_30 = [team for team, _ in sorted(probabilities.items(), key=lambda item: item[1]["3-0"], reverse=True)]
    ranked_03 = [team for team, _ in sorted(probabilities.items(), key=lambda item: item[1]["0-3"], reverse=True)]
    ranked_q = [team for team, _ in sorted(probabilities.items(), key=lambda item: item[1]["qualified"], reverse=True)]

    candidate_30 = ranked_30[:5]
    candidate_03 = ranked_03[:5]

    best_prediction = None
    best_success_rate = -1.0

    for teams_30 in combinations(candidate_30, 2):
        for teams_03 in combinations([team for team in candidate_03 if team not in teams_30], 2):
            candidate_adv_pool = [team for team in ranked_q if team not in teams_30 and team not in teams_03][:10]
            for advances in combinations(candidate_adv_pool, 6):
                prediction = {
                    "3-0": list(teams_30),
                    "advances": list(advances),
                    "0-3": list(teams_03),
                }
                success_rate = evaluate_pickem(prediction, all_simulations)
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_prediction = prediction

    return best_prediction, best_success_rate


class SwissMLRunner:
    def __init__(self, config_path="config.yaml", swiss_stage_setup_path=None):
        self.config_path = Path(config_path).resolve()
        self.config = load_yaml_config(self.config_path)
        self.predictor = HybridPlayoffPredictor(config_path)
        self.reference = load_swiss_reference()
        self.setup = load_swiss_stage_setup(
            self.reference,
            setup_path=swiss_stage_setup_path,
            main_config=self.config,
            main_config_path=self.config_path,
        )
        self.teams = list(self.setup["teams"])
        self.round1_matchups = list(self.setup["round1_matchups"])
        self.team_seeds = dict(self.setup["team_seeds"])
        self.num_simulations = load_num_simulations(self.reference["root_dir"], main_config=self.config)

    def _simulate_match(self, team1, team2, bo_format):
        elo1 = self.predictor.team_elo.get(team1, 1500)
        elo2 = self.predictor.team_elo.get(team2, 1500)

        if bo_format == "BO1":
            winner, map_result, veto_info = self.predictor.simulate_bo1(
                team1,
                team2,
                elo_team1=elo1,
                elo_team2=elo2,
                use_veto=True,
            )
            map_names = [map_result["map_name"]]
        else:
            winner, map_results, veto_info = self.predictor.simulate_bo3(
                team1,
                team2,
                elo_team1=elo1,
                elo_team2=elo2,
                use_veto=True,
            )
            map_names = [item["map_name"] for item in map_results]

        return winner, veto_info, map_names

    def simulate_one_swiss(self):
        records = {team: (0, 0) for team in self.teams}
        match_history = {team: [] for team in self.teams}
        round_logs = []

        for team1, team2 in self.round1_matchups:
            winner, veto_info, map_names = self._simulate_match(team1, team2, "BO1")
            winner_team = team1 if winner == 1 else team2
            loser_team = team2 if winner == 1 else team1

            wins, losses = records[winner_team]
            records[winner_team] = (wins + 1, losses)
            wins, losses = records[loser_team]
            records[loser_team] = (wins, losses + 1)

            match_history[team1].append(team2)
            match_history[team2].append(team1)
            round_logs.append(
                {
                    "round": 1,
                    "group": "0-0",
                    "team1": team1,
                    "team2": team2,
                    "bo_format": "BO1",
                    "winner": winner_team,
                    "maps": map_names,
                    "veto": veto_info,
                }
            )

        for round_num in range(2, 6):
            grouped_teams = defaultdict(list)
            for team, (wins, losses) in records.items():
                if wins < 3 and losses < 3:
                    grouped_teams[(wins, losses)].append(team)

            if not grouped_teams:
                break

            for record_key, teams in grouped_teams.items():
                ordered_teams = get_buchholz_sorted_groups(records, match_history, teams, self.team_seeds)

                if round_num in (2, 3):
                    pairs = choose_round_2_3_pairs(ordered_teams, match_history)
                else:
                    pairs = choose_round_4_5_pairs(ordered_teams, match_history)

                for team1, team2 in pairs:
                    wins1, losses1 = records[team1]
                    wins2, losses2 = records[team2]
                    is_elimination_or_advancement = (
                        wins1 == 2 or losses1 == 2 or wins2 == 2 or losses2 == 2
                    )
                    bo_format = "BO3" if is_elimination_or_advancement else "BO1"

                    winner, veto_info, map_names = self._simulate_match(team1, team2, bo_format)
                    winner_team = team1 if winner == 1 else team2
                    loser_team = team2 if winner == 1 else team1

                    wins, losses = records[winner_team]
                    records[winner_team] = (wins + 1, losses)
                    wins, losses = records[loser_team]
                    records[loser_team] = (wins, losses + 1)

                    match_history[team1].append(team2)
                    match_history[team2].append(team1)
                    round_logs.append(
                        {
                            "round": round_num,
                            "group": f"{record_key[0]}-{record_key[1]}",
                            "team1": team1,
                            "team2": team2,
                            "bo_format": bo_format,
                            "winner": winner_team,
                            "maps": map_names,
                            "veto": veto_info,
                        }
                    )

        summary = {"3-0": set(), "qualified": set(), "0-3": set()}
        for team, (wins, losses) in records.items():
            if wins == 3 and losses == 0:
                summary["3-0"].add(team)
                summary["qualified"].add(team)
            elif wins == 3:
                summary["qualified"].add(team)
            elif losses == 3 and wins == 0:
                summary["0-3"].add(team)

        return summary, round_logs

    def run(self, num_simulations=None):
        total_simulations = num_simulations or self.num_simulations
        print("=" * 72)
        print("CS2 Major Swiss Stage ML Prediction")
        print("=" * 72)
        print(f"Simulations: {total_simulations:,}")
        print(f"Rules source: {self.reference['source']}")
        print(f"Round 1 source: {self.setup['source']}")
        print()

        print("Seeded Teams")
        print("-" * 72)
        for idx, team in enumerate(self.setup["seeded_teams"], start=1):
            print(f"  Seed {idx:>2}: {team}")

        print()
        print("Round 1 Matchups")
        print("-" * 72)
        for idx, (team1, team2) in enumerate(self.round1_matchups, start=1):
            print(f"  Match {idx}: {team1} vs {team2}")

        probabilities = defaultdict(lambda: {"3-0": 0, "qualified": 0, "0-3": 0, "total": 0})
        all_simulations = []
        round_matchup_counts = {round_num: defaultdict(int) for round_num in range(1, 6)}
        round_bo_counts = {round_num: defaultdict(int) for round_num in range(1, 6)}
        round_map_counts = {round_num: defaultdict(int) for round_num in range(1, 6)}

        for sim_idx in range(total_simulations):
            if (sim_idx + 1) % 100 == 0:
                print(f"  Completed {sim_idx + 1:,}/{total_simulations:,}")

            summary, round_logs = self.simulate_one_swiss()
            all_simulations.append(summary)

            for team in self.teams:
                probabilities[team]["total"] += 1
                if team in summary["3-0"]:
                    probabilities[team]["3-0"] += 1
                    probabilities[team]["qualified"] += 1
                elif team in summary["qualified"]:
                    probabilities[team]["qualified"] += 1
                elif team in summary["0-3"]:
                    probabilities[team]["0-3"] += 1

            for entry in round_logs:
                matchup_key = tuple(sorted([entry["team1"], entry["team2"]]))
                round_matchup_counts[entry["round"]][matchup_key] += 1
                round_bo_counts[entry["round"]][entry["bo_format"]] += 1
                for map_name in entry["maps"]:
                    round_map_counts[entry["round"]][map_name] += 1

        final_probabilities = {}
        for team, stats in probabilities.items():
            total = stats["total"] or 1
            final_probabilities[team] = {
                "3-0": stats["3-0"] / total,
                "qualified": stats["qualified"] / total,
                "0-3": stats["0-3"] / total,
                "3-1-or-3-2": (stats["qualified"] - stats["3-0"]) / total,
            }

        if not all_simulations:
            print("  [!] No simulations ran — cannot compute Pick'em")
            recommendation = {"3-0": [], "advances": [], "0-3": []}
            success_rate = 0.0
        else:
            try:
                optimizer = PickemOptimizer(
                    teams=self.teams,
                    all_simulations=all_simulations,
                    use_gpu=True,
                    batch_size=20000,
                )
                recommendation, success_rate = optimizer.run()
                if recommendation is None:
                    raise RuntimeError("brute-force returned None")
            except Exception as exc:
                print(f"  [!] Brute-force failed ({exc}), falling back to heuristic")
                recommendation, success_rate = recommend_pickem(final_probabilities, all_simulations)
            if recommendation is None:
                recommendation = {"3-0": [], "advances": [], "0-3": []}
                success_rate = 0.0

        print()
        print("Swiss ML Probabilities")
        print("-" * 72)
        sorted_teams = sorted(final_probabilities.items(), key=lambda item: item[1]["qualified"], reverse=True)
        print(f"{'Team':<20} {'3-0':>8} {'Qualified':>10} {'0-3':>8} {'3-1/3-2':>10}")
        print("-" * 72)
        for team, stats in sorted_teams:
            print(
                f"{team:<20} "
                f"{stats['3-0'] * 100:7.2f}% "
                f"{stats['qualified'] * 100:9.2f}% "
                f"{stats['0-3'] * 100:7.2f}% "
                f"{stats['3-1-or-3-2'] * 100:9.2f}%"
            )

        print()
        print("Most Common Matchups By Round")
        print("-" * 72)
        for round_num in range(1, 6):
            print(f"Round {round_num}:")
            top_matchups = sorted(round_matchup_counts[round_num].items(), key=lambda item: item[1], reverse=True)[:8]
            for matchup, count in top_matchups:
                probability = count / total_simulations * 100
                print(f"  {matchup[0]} vs {matchup[1]}: {probability:.1f}%")
            if not top_matchups:
                print("  No active matchups recorded")

        print()
        print("Pick'Em Recommendation (ML Swiss)")
        print("-" * 72)
        print(f"  3-0 : {', '.join(recommendation['3-0'])}")
        print(f"  Adv : {', '.join(recommendation['advances'])}")
        print(f"  0-3 : {', '.join(recommendation['0-3'])}")
        print(f"  Estimated success rate (>=5 hits): {success_rate * 100:.2f}%")

        output = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_simulations": total_simulations,
            "seeded_teams": self.setup["seeded_teams"],
            "round1_matchups": self.setup["round1_matchups"],
            "rules_source": self.reference["source"],
            "round1_source": self.setup["source"],
            "probabilities": final_probabilities,
            "pickem_recommendation": recommendation,
            "pickem_success_rate": success_rate,
            "round_matchup_frequency": {
                str(round_num): {
                    f"{matchup[0]} vs {matchup[1]}": count / total_simulations
                    for matchup, count in sorted(
                        round_matchup_counts[round_num].items(), key=lambda item: item[1], reverse=True
                    )
                }
                for round_num in range(1, 6)
            },
            "round_bo_frequency": {
                str(round_num): {
                    bo_format: count / max(sum(counter.values()), 1)
                    for bo_format, count in counter.items()
                }
                for round_num, counter in round_bo_counts.items()
            },
            "round_map_frequency": {
                str(round_num): {
                    map_name: count / max(sum(counter.values()), 1)
                    for map_name, count in sorted(counter.items(), key=lambda item: item[1], reverse=True)
                }
                for round_num, counter in round_map_counts.items()
            },
        }

        output_path = Path("output") / "swiss_ml_predictions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)

        print()
        print(f"Saved to: {output_path.resolve()}")

        # 生成图表；失败不影响预测结果。
        try:
            from generate_charts import generate_all as _gen_charts
            charts_dir = output_path.parent / "charts"
            print()
            print("Generating visualization charts...")
            _gen_charts(output_path, charts_dir)
        except Exception as exc:
            print(f"[WARN] Chart generation failed (non-fatal): {exc}")

        return output


if __name__ == "__main__":
    runner = SwissMLRunner("config.yaml")
    runner.run()
