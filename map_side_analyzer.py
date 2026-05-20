"""
Map side analyzer for CT/T advantage estimation.

When detailed half-side data is available, this module derives map-specific
T-side win rates from the scraped historical dataset. Otherwise it falls back
to conservative defaults.
"""


class MapSideAnalyzer:
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

    def __init__(self, map_data_df=None, map_pool=None):
        if map_pool is not None:
            self.map_pool = list(map_pool)
        elif map_data_df is not None and "map_name" in map_data_df.columns:
            self.map_pool = sorted(map_data_df["map_name"].dropna().unique().tolist())
        else:
            self.map_pool = list(self.DEFAULT_T_SIDE_WIN_RATE.keys())

        if map_data_df is not None:
            self.map_t_side_win_rate = self._calculate_from_data(map_data_df)
        else:
            self.map_t_side_win_rate = {
                map_name: self.DEFAULT_T_SIDE_WIN_RATE.get(map_name, 0.5)
                for map_name in self.map_pool
            }

    @staticmethod
    def _parse_half_pair(value):
        if value is None:
            return None

        text = str(value).strip()
        if not text or text.lower() == "unknown":
            return None

        text = text.replace(" ", "")
        parts = text.split("-")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            return None

        return int(parts[0]), int(parts[1])

    def _extract_t_rounds(self, row):
        t_rounds_won = 0
        total_rounds = 0

        for half_idx in (1, 2):
            rounds = self._parse_half_pair(row.get(f"half{half_idx}_t_ct"))
            if not rounds:
                continue

            team1_rounds, team2_rounds = rounds
            team1_side = str(row.get(f"team1_half{half_idx}_side", "unknown")).upper()
            team2_side = str(row.get(f"team2_half{half_idx}_side", "unknown")).upper()

            if team1_side == "T":
                t_rounds_won += team1_rounds
                total_rounds += team1_rounds + team2_rounds
            elif team2_side == "T":
                t_rounds_won += team2_rounds
                total_rounds += team1_rounds + team2_rounds

        return t_rounds_won, total_rounds

    def _calculate_from_data(self, df):
        calculated = {}

        for map_name in self.map_pool:
            map_data = df[df["map_name"] == map_name]
            if len(map_data) < 50:
                calculated[map_name] = self.DEFAULT_T_SIDE_WIN_RATE.get(map_name, 0.5)
                continue

            t_rounds_won = 0
            total_rounds = 0

            for _, row in map_data.iterrows():
                map_t_rounds_won, map_total_rounds = self._extract_t_rounds(row)
                t_rounds_won += map_t_rounds_won
                total_rounds += map_total_rounds

            if total_rounds < 500:
                calculated[map_name] = self.DEFAULT_T_SIDE_WIN_RATE.get(map_name, 0.5)
            else:
                calculated[map_name] = t_rounds_won / total_rounds

        return calculated

    def get_side_advantage(self, map_name, team1_starts_ct):
        t_win_rate = self.map_t_side_win_rate.get(map_name, 0.5)

        if team1_starts_ct:
            if t_win_rate < 0.47:
                return 0.03
            if t_win_rate > 0.53:
                return -0.03
        else:
            if t_win_rate > 0.53:
                return 0.03
            if t_win_rate < 0.47:
                return -0.03

        return 0.0
