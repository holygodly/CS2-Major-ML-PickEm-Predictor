"""
Ban/pick simulation utilities for BO3 and BO5 playoff matches.

The veto model is intentionally simple and interpretable:
- bans target the opponent's strong maps and the team's own weak maps
- picks favor the team's strong maps and the opponent's weak maps
- side choice uses map-level T/CT bias
"""

from typing import Dict, List

import numpy as np


class VetoSimulator:
    DEFAULT_MAP_POOL = ["Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass"]
    DEFAULT_T_SIDE_WIN_RATE = {
        "Ancient": 0.51,
        "Anubis": 0.47,
        "Dust2": 0.50,
        "Inferno": 0.46,
        "Mirage": 0.49,
        "Nuke": 0.42,
        "Overpass": 0.45,
        "Vertigo": 0.48,
        "Train": 0.45,
    }

    def __init__(self, map_stats_df=None, map_pool=None):
        self.map_stats_df = map_stats_df

        if map_pool is not None:
            self.map_pool = list(map_pool)
        elif map_stats_df is not None and "map_name" in map_stats_df.columns:
            self.map_pool = sorted(map_stats_df["map_name"].dropna().unique().tolist())
        else:
            self.map_pool = list(self.DEFAULT_MAP_POOL)

        self.map_t_side_win_rate = {
            map_name: self.DEFAULT_T_SIDE_WIN_RATE.get(map_name, 0.5)
            for map_name in self.map_pool
        }
        self.team_map_preferences = {}
        self._preference_cache = {}

    def calculate_team_map_preference(self, team: str, map_name: str, recent_matches: int = 10) -> float:
        cache_key = f"{team}_{map_name}_{recent_matches}"
        if cache_key in self._preference_cache:
            return self._preference_cache[cache_key]

        if self.map_stats_df is None:
            return 0.5

        team_map_data = self.map_stats_df[
            ((self.map_stats_df["team1"] == team) | (self.map_stats_df["team2"] == team))
            & (self.map_stats_df["map_name"] == map_name)
        ].tail(recent_matches)

        if team_map_data.empty:
            preference = 0.5
        else:
            wins = (
                ((team_map_data["team1"] == team) & (team_map_data["winner"] == 1)).sum()
                + ((team_map_data["team2"] == team) & (team_map_data["winner"] == 0)).sum()
            )
            preference = wins / len(team_map_data)

        self._preference_cache[cache_key] = preference
        return preference

    def get_team_map_pool(self, team: str) -> Dict[str, float]:
        if team in self.team_map_preferences:
            return self.team_map_preferences[team]

        preferences = {
            map_name: self.calculate_team_map_preference(team, map_name)
            for map_name in self.map_pool
        }
        self.team_map_preferences[team] = preferences
        return preferences

    def _choose_ban(
        self,
        banning_team: str,
        opponent: str,
        available_maps: List[str],
        own_prefs: Dict[str, float],
        opp_prefs: Dict[str, float],
        temperature: float = 0.35,
    ) -> str:
        scores = {}
        for map_name in available_maps:
            own_score = own_prefs.get(map_name, 0.5)
            opp_score = opp_prefs.get(map_name, 0.5)
            scores[map_name] = opp_score * 0.6 + (1 - own_score) * 0.4
        maps = list(scores.keys())
        values = np.array([scores[m] for m in maps])
        exp_values = np.exp(values / temperature)
        probabilities = exp_values / exp_values.sum()
        return np.random.choice(maps, p=probabilities)

    def _choose_pick(
        self,
        picking_team: str,
        opponent: str,
        available_maps: List[str],
        own_prefs: Dict[str, float],
        opp_prefs: Dict[str, float],
        temperature: float = 0.35,
    ) -> str:
        scores = {}
        for map_name in available_maps:
            own_score = own_prefs.get(map_name, 0.5)
            opp_score = opp_prefs.get(map_name, 0.5)
            scores[map_name] = own_score * 0.7 + (1 - opp_score) * 0.3
        maps = list(scores.keys())
        values = np.array([scores[m] for m in maps])
        exp_values = np.exp(values / temperature)
        probabilities = exp_values / exp_values.sum()
        return np.random.choice(maps, p=probabilities)

    def _choose_starting_side(self, choosing_team: str, map_name: str, team_prefs: Dict[str, float]) -> bool:
        t_win_rate = self.map_t_side_win_rate.get(map_name, 0.5)
        ct_strength = 1 - t_win_rate

        if ct_strength > 0.53:
            return False
        if t_win_rate > 0.53:
            return True

        team_preference = team_prefs.get(map_name, 0.5)
        if team_preference >= 0.55:
            return t_win_rate > 0.5
        if team_preference <= 0.45:
            return t_win_rate >= 0.5
        return np.random.random() < 0.5

    def _append_pick_map(
        self,
        final_maps: List[Dict],
        map_index: int,
        picked_map: str,
        picker: str,
        side_chooser: str,
        team1: str,
        team2: str,
        chooser_prefs: Dict[str, float],
    ):
        chooser_opponent_starts_ct = self._choose_starting_side(side_chooser, picked_map, chooser_prefs)
        if side_chooser == team1:
            team1_starts_ct = chooser_opponent_starts_ct
        else:
            team1_starts_ct = not chooser_opponent_starts_ct

        final_maps.append(
            {
                "map": picked_map,
                "picker": picker,
                "side_chooser": side_chooser,
                "team1_starts_ct": team1_starts_ct,
                "map_index": map_index,
            }
        )

    def simulate_bo3_veto(self, team1: str, team2: str) -> Dict:
        available_maps = self.map_pool.copy()
        veto_sequence = []
        final_maps = []

        team1_prefs = self.get_team_map_pool(team1)
        team2_prefs = self.get_team_map_pool(team2)

        team1_ban = self._choose_ban(team1, team2, available_maps, team1_prefs, team2_prefs)
        available_maps.remove(team1_ban)
        veto_sequence.append({"action": "ban", "team": team1, "map": team1_ban})

        team2_ban = self._choose_ban(team2, team1, available_maps, team2_prefs, team1_prefs)
        available_maps.remove(team2_ban)
        veto_sequence.append({"action": "ban", "team": team2, "map": team2_ban})

        team1_pick = self._choose_pick(team1, team2, available_maps, team1_prefs, team2_prefs)
        available_maps.remove(team1_pick)
        veto_sequence.append({"action": "pick", "team": team1, "map": team1_pick})
        self._append_pick_map(final_maps, 0, team1_pick, team1, team2, team1, team2, team2_prefs)

        team2_pick = self._choose_pick(team2, team1, available_maps, team2_prefs, team1_prefs)
        available_maps.remove(team2_pick)
        veto_sequence.append({"action": "pick", "team": team2, "map": team2_pick})
        self._append_pick_map(final_maps, 1, team2_pick, team2, team1, team1, team2, team1_prefs)

        team1_ban2 = self._choose_ban(team1, team2, available_maps, team1_prefs, team2_prefs)
        available_maps.remove(team1_ban2)
        veto_sequence.append({"action": "ban", "team": team1, "map": team1_ban2})

        team2_ban2 = self._choose_ban(team2, team1, available_maps, team2_prefs, team1_prefs)
        available_maps.remove(team2_ban2)
        veto_sequence.append({"action": "ban", "team": team2, "map": team2_ban2})

        decider = available_maps[0]
        final_maps.append(
            {
                "map": decider,
                "picker": "decider",
                "side_chooser": "knife_round",
                "team1_starts_ct": np.random.random() < 0.5,
                "map_index": 2,
            }
        )

        return {
            "veto_sequence": veto_sequence,
            "maps": final_maps,
            "reasoning": {
                "team1_bans": [team1_ban, team1_ban2],
                "team2_bans": [team2_ban, team2_ban2],
                "team1_pick": team1_pick,
                "team2_pick": team2_pick,
                "decider": decider,
            },
        }

    def simulate_bo1_veto(self, team1: str, team2: str) -> Dict:
        available_maps = self.map_pool.copy()
        veto_sequence = []

        team1_prefs = self.get_team_map_pool(team1)
        team2_prefs = self.get_team_map_pool(team2)

        for banning_team, own_prefs, opp_prefs in (
            (team1, team1_prefs, team2_prefs),
            (team2, team2_prefs, team1_prefs),
            (team1, team1_prefs, team2_prefs),
            (team2, team2_prefs, team1_prefs),
            (team1, team1_prefs, team2_prefs),
            (team2, team2_prefs, team1_prefs),
        ):
            opponent = team2 if banning_team == team1 else team1
            banned_map = self._choose_ban(banning_team, opponent, available_maps, own_prefs, opp_prefs)
            available_maps.remove(banned_map)
            veto_sequence.append({"action": "ban", "team": banning_team, "map": banned_map})

        decider = available_maps[0]
        final_maps = [
            {
                "map": decider,
                "picker": "decider",
                "side_chooser": "knife_round",
                "team1_starts_ct": np.random.random() < 0.5,
                "map_index": 0,
            }
        ]

        return {
            "veto_sequence": veto_sequence,
            "maps": final_maps,
            "reasoning": {
                "decider": decider,
                "team1_bans": [entry["map"] for entry in veto_sequence if entry["team"] == team1],
                "team2_bans": [entry["map"] for entry in veto_sequence if entry["team"] == team2],
            },
        }

    def simulate_bo5_veto(self, team1: str, team2: str) -> Dict:
        available_maps = self.map_pool.copy()
        veto_sequence = []
        final_maps = []

        team1_prefs = self.get_team_map_pool(team1)
        team2_prefs = self.get_team_map_pool(team2)

        team1_ban = self._choose_ban(team1, team2, available_maps, team1_prefs, team2_prefs)
        available_maps.remove(team1_ban)
        veto_sequence.append({"action": "ban", "team": team1, "map": team1_ban})

        team2_ban = self._choose_ban(team2, team1, available_maps, team2_prefs, team1_prefs)
        available_maps.remove(team2_ban)
        veto_sequence.append({"action": "ban", "team": team2, "map": team2_ban})

        pick_order = [
            (team1, team2, team1_prefs, team2_prefs),
            (team2, team1, team2_prefs, team1_prefs),
            (team1, team2, team1_prefs, team2_prefs),
            (team2, team1, team2_prefs, team1_prefs),
        ]

        for map_index, (picker, side_chooser, picker_prefs, opponent_prefs) in enumerate(pick_order):
            picked_map = self._choose_pick(picker, side_chooser, available_maps, picker_prefs, opponent_prefs)
            available_maps.remove(picked_map)
            veto_sequence.append({"action": "pick", "team": picker, "map": picked_map})
            chooser_prefs = team1_prefs if side_chooser == team1 else team2_prefs
            self._append_pick_map(final_maps, map_index, picked_map, picker, side_chooser, team1, team2, chooser_prefs)

        decider = available_maps[0]
        final_maps.append(
            {
                "map": decider,
                "picker": "decider",
                "side_chooser": "knife_round",
                "team1_starts_ct": np.random.random() < 0.5,
                "map_index": 4,
            }
        )

        return {
            "veto_sequence": veto_sequence,
            "maps": final_maps,
            "reasoning": {
                "team1_ban": team1_ban,
                "team2_ban": team2_ban,
                "team1_picks": [final_maps[0]["map"], final_maps[2]["map"]],
                "team2_picks": [final_maps[1]["map"], final_maps[3]["map"]],
                "decider": decider,
            },
        }

    def calculate_map_advantage(self, team1: str, team2: str, map_name: str, team1_starts_ct: bool) -> float:
        adjustment = 0.0

        team1_map_wr = self.calculate_team_map_preference(team1, map_name)
        team2_map_wr = self.calculate_team_map_preference(team2, map_name)
        adjustment += (team1_map_wr - team2_map_wr) * 0.3

        t_win_rate = self.map_t_side_win_rate.get(map_name, 0.5)
        if team1_starts_ct:
            if t_win_rate < 0.47:
                adjustment += 0.03
            elif t_win_rate > 0.53:
                adjustment -= 0.03
        else:
            if t_win_rate > 0.53:
                adjustment += 0.03
            elif t_win_rate < 0.47:
                adjustment -= 0.03

        return float(np.clip(adjustment, -0.15, 0.15))

    def get_side_swap_adjustment(self, map_name: str, current_half: int) -> float:
        t_win_rate = self.map_t_side_win_rate.get(map_name, 0.5)
        if current_half == 1:
            return 16 * t_win_rate
        return 16 * (1 - t_win_rate)


class TacticalAnalyzer:
    def __init__(self, veto_sim: VetoSimulator):
        self.veto_sim = veto_sim

    def analyze_series_tactical_advantage(self, team1: str, team2: str, match_type: str = "BO3") -> Dict:
        if match_type == "BO5":
            veto_result = self.veto_sim.simulate_bo5_veto(team1, team2)
        else:
            veto_result = self.veto_sim.simulate_bo3_veto(team1, team2)

        map_advantages = []
        total_advantage = 0.0

        for map_info in veto_result["maps"]:
            map_name = map_info["map"]
            advantage = self.veto_sim.calculate_map_advantage(
                team1, team2, map_name, map_info["team1_starts_ct"]
            )
            map_advantages.append(
                {
                    "map": map_name,
                    "advantage": advantage,
                    "team1_starts_ct": map_info["team1_starts_ct"],
                    "picker": map_info["picker"],
                }
            )
            total_advantage += advantage

        overall_advantage = total_advantage / len(map_advantages) if map_advantages else 0.0
        baseline_maps = 1.5 if match_type == "BO3" else 2.5

        return {
            "veto_result": veto_result,
            "map_advantages": map_advantages,
            "overall_advantage": overall_advantage,
            "team1_expected_maps_won": baseline_maps + overall_advantage * 3,
        }

    def analyze_bo3_tactical_advantage(self, team1: str, team2: str) -> Dict:
        return self.analyze_series_tactical_advantage(team1, team2, match_type="BO3")

    def analyze_bo5_tactical_advantage(self, team1: str, team2: str) -> Dict:
        return self.analyze_series_tactical_advantage(team1, team2, match_type="BO5")
