"""
Feature engineering for the playoff ML pipeline.

This step converts the preprocessed map-level dataset into a fixed feature
matrix and label vector for model training.

Vectorized implementation — avoids iterrows loops for O(N²) → O(N·G) speedup.
"""

import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class FeatureEngineering:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.df = None
        self.map_names = []

    def load_data(self):
        """Load the preprocessed map-level dataset."""
        print("\n[1/7] 加载地图级别数据...")
        self.df = pd.read_csv("data/map_level_dataset.csv")
        self.df["date"] = pd.to_datetime(self.df["date"])
        if "is_augmented" in self.df.columns:
            self.df = self.df.sort_values(
                ["date", "match_url", "map_index", "is_augmented"]
            ).reset_index(drop=True)
        else:
            self.df = self.df.sort_values(["date", "match_url", "map_index"]).reset_index(drop=True)
        self.map_names = sorted(self.df["map_name"].dropna().unique().tolist())
        print(f"  ✓ 数据维度: {self.df.shape}")
        print(f"  ✓ 地图池: {', '.join(self.map_names)}")

    def build_map_features(self):
        """Build map and team-history features (vectorized).

        Uses cumulative exponential-weighted sums per (team, map) group
        instead of per-row DataFrame filtering. Complexity: O(N·G) where
        G = number of unique (team, map) groups, vs original O(N²).
        """
        print("\n[2/7] 构建地图特征...")

        features = []

        TIME_DECAY_HALF_LIFE_DAYS = 30.0
        TIME_DECAY_SCALE = TIME_DECAY_HALF_LIFE_DAYS / np.log(2)
        # WIN_RATE: 弱先验 (1, 2) — 让真实地图胜率差异能显现，不再被强行拉向 0.5。
        # 每个 (team, map) 通常有 10+ 样本，先验 weight=2 即可。
        WIN_RATE_PRIOR_WINS = 1.0
        WIN_RATE_PRIOR_TOTAL = 2.0
        # H2H: 保留 (2, 4) 强先验 — H2H 每对队伍每张图常 0-2 次交手,
        # 弱先验会让单场结果占主导，放大噪声。
        H2H_PRIOR_WINS = 2.0
        H2H_PRIOR_TOTAL = 4.0

        # ── 1) Build long-format view: one row per (team, map, match) ──
        df = self.df
        ref_date = df["date"].min()
        df_days = (df["date"] - ref_date).dt.total_seconds().values / 86400.0

        rows1 = pd.DataFrame({
            "orig_idx": df.index,
            "date": df["date"].values,
            "days": df_days,
            "team": df["team1"].values,
            "map_name": df["map_name"].values,
            "won": (df["winner"] == 1).values.astype(np.float32),
        })
        rows2 = pd.DataFrame({
            "orig_idx": df.index,
            "date": df["date"].values,
            "days": df_days,
            "team": df["team2"].values,
            "map_name": df["map_name"].values,
            "won": (df["winner"] == 0).values.astype(np.float32),
        })
        long = pd.concat([rows1, rows2], ignore_index=True)
        long.sort_values(["team", "map_name", "days"], inplace=True)

        # ── 2) Build date-indexed lookup for cross-map queries ──
        # For each (team, map) group, store cumulative (cum_wins_f, cum_total_f)
        # at each date boundary. For any query date, binary-search to find the
        # state strictly before that date.
        team_map_state = {}  # (team, map) → (sorted_dates, cum_wins_f_array, cum_total_f_array)

        for (team, map_name), grp in long.groupby(["team", "map_name"], sort=False):
            days = grp["days"].values
            won = grp["won"].values
            f = np.exp(days / TIME_DECAY_SCALE)

            # Build cumulative state at each date boundary (AFTER processing that date)
            boundary_days = []
            boundary_cum_wins = []
            boundary_cum_total = []
            cum_w = 0.0
            cum_t = 0.0
            i = 0
            n = len(days)
            while i < n:
                j = i + 1
                while j < n and days[j] == days[i]:
                    j += 1
                for k in range(i, j):
                    cum_w += won[k] * f[k]
                    cum_t += f[k]
                boundary_days.append(days[i])
                boundary_cum_wins.append(cum_w)
                boundary_cum_total.append(cum_t)
                i = j

            team_map_state[(team, map_name)] = (
                np.array(boundary_days),
                np.array(boundary_cum_wins),
                np.array(boundary_cum_total),
            )

        default_rate = WIN_RATE_PRIOR_WINS / WIN_RATE_PRIOR_TOTAL

        def _lookup_rate(team, map_name, query_days):
            """Get team's win rate on map at query_days (exclusive of same-date)."""
            key = (team, map_name)
            if key not in team_map_state:
                return default_rate
            b_days, b_cum_w, b_cum_t = team_map_state[key]
            # Find last boundary with days < query_days (strict)
            pos = np.searchsorted(b_days, query_days, side="left") - 1
            if pos < 0:
                return default_rate
            cum_w = b_cum_w[pos]
            cum_t = b_cum_t[pos]
            f_query = np.exp(query_days / TIME_DECAY_SCALE)
            w_wins = cum_w / f_query
            w_total = cum_t / f_query
            return (w_wins + WIN_RATE_PRIOR_WINS) / (w_total + WIN_RATE_PRIOR_TOTAL)

        # ── 4) Fill feature columns ──
        teams1 = df["team1"].values
        teams2 = df["team2"].values

        for map_name in self.map_names:
            col1 = f"team1_{map_name}_win_rate"
            col2 = f"team2_{map_name}_win_rate"
            rates1 = np.empty(len(df), dtype=np.float32)
            rates2 = np.empty(len(df), dtype=np.float32)
            for i in range(len(df)):
                rates1[i] = _lookup_rate(teams1[i], map_name, df_days[i])
                rates2[i] = _lookup_rate(teams2[i], map_name, df_days[i])
            self.df[col1] = rates1
            self.df[col2] = rates2
            features.extend([col1, col2])

        self.df["team1_pick_advantage"] = (df["picker"] == df["team1"]).astype(int)
        self.df["team2_pick_advantage"] = (df["picker"] == df["team2"]).astype(int)
        features.extend(["team1_pick_advantage", "team2_pick_advantage"])

        # ── 5) H2H win rate (vectorized per matchup pair) ──
        # Use raw (team1, team2) order — NOT canonical — to preserve direction.
        # Augmented rows (B vs A) naturally land in a separate bucket from (A vs B).
        pair_key = df["team1"].values.astype(str) + "|||" + df["team2"].values.astype(str)
        h2h_long = pd.DataFrame({
            "orig_idx": df.index,
            "days": df_days,
            "pair_key": pair_key,
            "map_name": df["map_name"].values,
            # "won" from team1's perspective
            "won": (df["winner"] == 1).values.astype(np.float32),
        })
        h2h_long.sort_values(["pair_key", "map_name", "days"], inplace=True)

        h2h_results = np.full(len(df), H2H_PRIOR_WINS / H2H_PRIOR_TOTAL, dtype=np.float32)

        for _, grp in h2h_long.groupby(["pair_key", "map_name"], sort=False):
            if len(grp) == 0:
                continue
            idx = grp["orig_idx"].values
            days = grp["days"].values
            won = grp["won"].values
            f = np.exp(days / TIME_DECAY_SCALE)

            cum_wins_f = 0.0
            cum_total_f = 0.0
            i = 0
            n = len(days)
            while i < n:
                j = i + 1
                while j < n and days[j] == days[i]:
                    j += 1
                for k in range(i, j):
                    w_wins = cum_wins_f / f[k]
                    w_total = cum_total_f / f[k]
                    h2h_results[idx[k]] = (w_wins + H2H_PRIOR_WINS) / (w_total + H2H_PRIOR_TOTAL)
                for k in range(i, j):
                    cum_wins_f += won[k] * f[k]
                    cum_total_f += f[k]
                i = j

        self.df["team1_h2h_map_win_rate"] = h2h_results
        features.append("team1_h2h_map_win_rate")

        features.extend(
            [
                "team1_recent_win_rate",
                "team2_recent_win_rate",
                "team1_avg_rating",
                "team1_rating_std",
                "team1_top_rating",
                "team2_avg_rating",
                "team2_rating_std",
                "team2_top_rating",
            ]
        )

        print(f"  ✓ 构建了 {len(features)} 个地图相关特征")
        return features

    def build_momentum_features(self):
        """Build within-series momentum features (vectorized)."""
        print("\n[3/7] 构建Momentum特征...")

        features = []
        self.df["won_previous_map"] = (self.df["previous_map_winner"] == 1).astype(int)
        self.df["lost_previous_map"] = (self.df["previous_map_winner"] == 0).astype(int)
        features.extend(["won_previous_map", "lost_previous_map"])

        self.df["series_leading"] = (
            self.df["series_score_team1"] > self.df["series_score_team2"]
        ).astype(int)
        self.df["series_trailing"] = (
            self.df["series_score_team1"] < self.df["series_score_team2"]
        ).astype(int)
        self.df["series_tied"] = (
            self.df["series_score_team1"] == self.df["series_score_team2"]
        ).astype(int)
        features.extend(["series_leading", "series_trailing", "series_tied"])

        self.df["is_decider_int"] = self.df["is_decider"].astype(int)
        features.append("is_decider_int")

        # Vectorized win streak: group by (match_url, is_augmented), sort by map_index,
        # compute consecutive wins from team1's perspective going backwards.
        has_aug = "is_augmented" in self.df.columns
        aug_col = self.df["is_augmented"].values if has_aug else np.zeros(len(self.df), dtype=int)

        # team1 won this map?
        team1_won = (self.df["winner"] == 1).values.astype(int)

        # Group key
        group_keys = self.df["match_url"].values.astype(str) + "||" + aug_col.astype(str)
        map_indices = self.df["map_index"].values

        win_streaks = np.zeros(len(self.df), dtype=int)

        # Process per series group
        for gk, grp_idx in self.df.groupby(pd.Series(group_keys), sort=False).groups.items():
            grp_idx_arr = grp_idx.values if hasattr(grp_idx, 'values') else np.array(grp_idx)
            if len(grp_idx_arr) <= 1:
                continue
            # Sort within group by map_index
            order = np.argsort(map_indices[grp_idx_arr])
            sorted_idx = grp_idx_arr[order]
            won_arr = team1_won[sorted_idx]
            # For each position, count consecutive wins looking backwards
            for pos in range(1, len(sorted_idx)):
                streak = 0
                for prev in range(pos - 1, -1, -1):
                    if won_arr[prev]:
                        streak += 1
                    else:
                        break
                win_streaks[sorted_idx[pos]] = streak

        self.df["team1_win_streak"] = win_streaks
        features.append("team1_win_streak")

        print(f"  ✓ 构建了 {len(features)} 个Momentum特征")
        return features

    def build_advanced_features(self):
        """Build advanced features: roster stability, upset rate, event stage weight (vectorized)."""
        print("\n[4/7] 构建高级特征（阵容变动 / 冷门率 / 赛事阶段）...")

        features = []

        # ------------------------------------------------------------------
        # 1) 阵容稳定性 — vectorized via pre-built lookup
        # ------------------------------------------------------------------
        player_stats_path = self.config["data"].get("player_stats_pattern", "player_stats.csv")
        data_dir = self.config["data"]["data_dir"]
        ps_path = Path(data_dir) / player_stats_path

        if ps_path.exists():
            ps = pd.read_csv(ps_path, on_bad_lines="skip")
            ps["match_date"] = pd.to_datetime(ps.get("match_date", ""), errors="coerce")
            if "match_url" in ps.columns and self.df is not None:
                url_dates = self.df.groupby("match_url")["date"].first().to_dict()
                missing_dates = ps["match_date"].isna()
                ps.loc[missing_dates, "match_date"] = ps.loc[missing_dates, "match_url"].map(url_dates)
            data_end_date = self.config.get("data", {}).get("data_end_date")
            if data_end_date:
                ps = ps[ps["match_date"].notna() & (ps["match_date"] <= pd.to_datetime(data_end_date))]
            both_ps = ps[ps["side"] == "Both"].copy() if "side" in ps.columns else ps.copy()

            roster_per_match = (
                both_ps.groupby(["match_url", "team"])["player_name"]
                .apply(set)
                .reset_index()
                .rename(columns={"player_name": "roster"})
            )
            roster_per_match = roster_per_match.merge(
                both_ps[["match_url", "match_date"]].drop_duplicates(),
                on="match_url", how="left"
            )
            roster_per_match = roster_per_match.sort_values("match_date").reset_index(drop=True)

            # Pre-build per-team sorted roster history for O(1) lookup
            team_roster_hist = defaultdict(list)  # team → [(date, roster_set), ...]
            for _, r in roster_per_match.iterrows():
                if r["roster"] and pd.notna(r["match_date"]):
                    team_roster_hist[r["team"]].append((r["match_date"], r["roster"]))

            def _roster_stability_fast(team, current_date, n_recent=10):
                hist = team_roster_hist.get(team, [])
                # Binary search for entries before current_date
                # hist is sorted by date
                end = len(hist)
                # Find rightmost index with date < current_date
                lo, hi = 0, end
                while lo < hi:
                    mid = (lo + hi) // 2
                    if hist[mid][0] < current_date:
                        lo = mid + 1
                    else:
                        hi = mid
                relevant = hist[max(0, lo - n_recent):lo]
                if len(relevant) < 2:
                    return 1.0
                latest_roster = relevant[-1][1]
                if not latest_roster:
                    return 1.0
                overlaps = []
                for _, roster in relevant[:-1]:
                    if roster:
                        overlap = len(latest_roster & roster) / max(len(latest_roster), 1)
                        overlaps.append(overlap)
                return float(np.mean(overlaps)) if overlaps else 1.0

            teams1 = self.df["team1"].values
            teams2 = self.df["team2"].values
            dates = self.df["date"].values
            stab1 = np.empty(len(self.df), dtype=np.float32)
            stab2 = np.empty(len(self.df), dtype=np.float32)
            for i in range(len(self.df)):
                stab1[i] = _roster_stability_fast(teams1[i], dates[i])
                stab2[i] = _roster_stability_fast(teams2[i], dates[i])
            self.df["team1_roster_stability"] = stab1
            self.df["team2_roster_stability"] = stab2
        else:
            self.df["team1_roster_stability"] = 1.0
            self.df["team2_roster_stability"] = 1.0

        features.extend(["team1_roster_stability", "team2_roster_stability"])

        # ------------------------------------------------------------------
        # 2) 冷门率 — vectorized via per-team sorted history
        # ------------------------------------------------------------------
        _orig_mask = (self.df["is_augmented"] == 0) if "is_augmented" in self.df.columns else pd.Series(True, index=self.df.index)
        orig_df = self.df[_orig_mask].copy()

        # Pre-build per-team history: list of (date, is_underdog, won)
        team_upset_hist = defaultdict(list)  # team → [(date, is_underdog, won), ...]

        t1_rates = orig_df.get("team1_recent_win_rate", pd.Series(0.5, index=orig_df.index)).values
        t2_rates = orig_df.get("team2_recent_win_rate", pd.Series(0.5, index=orig_df.index)).values
        orig_teams1 = orig_df["team1"].values
        orig_teams2 = orig_df["team2"].values
        orig_dates = orig_df["date"].values
        orig_winners = orig_df["winner"].values

        for i in range(len(orig_df)):
            d = orig_dates[i]
            # team1 perspective
            is_underdog1 = t1_rates[i] < t2_rates[i]
            won1 = orig_winners[i] == 1
            team_upset_hist[orig_teams1[i]].append((d, is_underdog1, won1))
            # team2 perspective
            is_underdog2 = t2_rates[i] < t1_rates[i]
            won2 = orig_winners[i] == 0
            team_upset_hist[orig_teams2[i]].append((d, is_underdog2, won2))

        def _upset_rate_fast(team, current_date, n_recent=30):
            hist = team_upset_hist.get(team, [])
            # hist is in chronological order; find entries before current_date
            end = len(hist)
            lo, hi = 0, end
            while lo < hi:
                mid = (lo + hi) // 2
                if hist[mid][0] < current_date:
                    lo = mid + 1
                else:
                    hi = mid
            relevant = hist[max(0, lo - n_recent):lo]
            if len(relevant) < 5:
                return 0.5
            underdog_count = 0
            upsets = 0
            for _, is_underdog, won in relevant:
                if is_underdog:
                    underdog_count += 1
                    if won:
                        upsets += 1
            if underdog_count < 3:
                return 0.5
            return upsets / underdog_count

        teams1 = self.df["team1"].values
        teams2 = self.df["team2"].values
        dates = self.df["date"].values
        upset1 = np.empty(len(self.df), dtype=np.float32)
        upset2 = np.empty(len(self.df), dtype=np.float32)
        for i in range(len(self.df)):
            upset1[i] = _upset_rate_fast(teams1[i], dates[i])
            upset2[i] = _upset_rate_fast(teams2[i], dates[i])

        self.df["team1_upset_rate"] = upset1
        self.df["team2_upset_rate"] = upset2
        features.extend(["team1_upset_rate", "team2_upset_rate"])

        # ------------------------------------------------------------------
        # 3) 大赛胜率 — vectorized via per-team BO3/BO5 history with cum sums
        # ------------------------------------------------------------------
        TIME_DECAY_SCALE = 30.0 / np.log(2)

        big_mask = _orig_mask & self.df["match_type"].isin(["BO3", "BO5"])
        big_df = self.df[big_mask].copy()
        ref_date = self.df["date"].min()

        # Build per-team sorted big-match history with cumulative exp sums
        team_big_state = {}  # team → (boundary_days, cum_wins_f, cum_total_f)

        if len(big_df) > 0:
            big_teams1 = big_df["team1"].values
            big_teams2 = big_df["team2"].values
            big_days = (big_df["date"] - ref_date).dt.total_seconds().values / 86400.0
            big_winners = big_df["winner"].values

            team_big_entries = defaultdict(list)  # team → [(days, won)]
            for i in range(len(big_df)):
                d = big_days[i]
                team_big_entries[big_teams1[i]].append((d, float(big_winners[i] == 1)))
                team_big_entries[big_teams2[i]].append((d, float(big_winners[i] == 0)))

            for team, entries in team_big_entries.items():
                entries.sort(key=lambda x: x[0])
                days_arr = np.array([e[0] for e in entries])
                won_arr = np.array([e[1] for e in entries])
                f = np.exp(days_arr / TIME_DECAY_SCALE)

                boundary_days = []
                boundary_cum_wins = []
                boundary_cum_total = []
                cum_w = 0.0
                cum_t = 0.0
                i = 0
                n = len(days_arr)
                while i < n:
                    j = i + 1
                    while j < n and days_arr[j] == days_arr[i]:
                        j += 1
                    for k in range(i, j):
                        cum_w += won_arr[k] * f[k]
                        cum_t += f[k]
                    boundary_days.append(days_arr[i])
                    boundary_cum_wins.append(cum_w)
                    boundary_cum_total.append(cum_t)
                    i = j

                team_big_state[team] = (
                    np.array(boundary_days),
                    np.array(boundary_cum_wins),
                    np.array(boundary_cum_total),
                )

        df_days = (self.df["date"] - ref_date).dt.total_seconds().values / 86400.0

        def _big_match_rate_fast(team, query_days):
            if team not in team_big_state:
                return 0.5
            b_days, b_cum_w, b_cum_t = team_big_state[team]
            pos = np.searchsorted(b_days, query_days, side="left") - 1
            if pos < 0:
                return 0.5
            f_query = np.exp(query_days / TIME_DECAY_SCALE)
            w_wins = b_cum_w[pos] / f_query
            w_total = b_cum_t[pos] / f_query
            if w_total < 2.5:  # fewer than ~3 effective matches
                return 0.5
            return (w_wins + 2.0) / (w_total + 4.0)

        big1 = np.empty(len(self.df), dtype=np.float32)
        big2 = np.empty(len(self.df), dtype=np.float32)
        for i in range(len(self.df)):
            big1[i] = _big_match_rate_fast(teams1[i], df_days[i])
            big2[i] = _big_match_rate_fast(teams2[i], df_days[i])

        self.df["team1_big_match_win_rate"] = big1
        self.df["team2_big_match_win_rate"] = big2
        features.extend(["team1_big_match_win_rate", "team2_big_match_win_rate"])

        print(f"  ✓ 构建了 {len(features)} 个高级特征")
        print(f"    - 阵容稳定性: team1/2_roster_stability")
        print(f"    - 冷门率: team1/2_upset_rate")
        print(f"    - 大赛胜率: team1/2_big_match_win_rate")
        return features

    def build_elo_features(self):
        """Build ELO-based opponent-strength features.

        Computes incremental ELO ratings from historical match results so that
        the model can distinguish between teams that beat strong opponents vs
        teams that only beat weak ones.

        K = base × format × venue × opponent_reliability:
          - Format: BO1 → 0.8×, BO3 → 1.0× (baseline), BO5 → 1.2×
          - Venue:  LAN → 1.3×, Online → 1.0×
          - Reliability: both teams ≥15 matches → 1.0×, ≥5 → 0.75×, <5 → 0.5×

        Side-effects: stores ``self._match_elo_snapshot`` and
        ``self._team_match_history`` for downstream SoS / quality features.
        """
        print("\n[ELO] 构建对手强度特征 (incremental ELO, variable K)...")

        BASE_K = 32.0
        INIT_ELO = 1000.0
        # BO3 is now baseline (1.0), BO1 penalized, BO5 slightly boosted.
        # This prevents local BO3s vs weak teams from being over-weighted.
        BO_MULT = {"BO1": 0.8, "BO3": 1.0, "BO5": 1.2}
        # Minimum match count thresholds for opponent reliability
        RELIABLE_MATCHES = 15
        SEMI_RELIABLE = 5

        features = []

        # Work on original (non-augmented) rows only to avoid double-counting
        has_aug = "is_augmented" in self.df.columns
        if has_aug:
            orig_mask = self.df["is_augmented"] == 0
        else:
            orig_mask = pd.Series(True, index=self.df.index)

        orig_df = self.df[orig_mask].copy()
        # Deduplicate to series level: one ELO update per series, not per map.
        # Use the first map of each series (map_index == 0) as the anchor row.
        series_df = orig_df[orig_df["map_index"] == 0].sort_values("date").reset_index()

        elo = {}  # team → current ELO
        match_count = {}  # team → number of series played so far
        # match_url → (team1_elo_before, team2_elo_before)
        match_elo_snapshot = {}
        # team → [(date, opponent_elo_pre, won_bool, match_type, is_lan)]
        team_match_history = {}

        for _, row in series_df.iterrows():
            t1, t2 = row["team1"], row["team2"]
            e1 = elo.get(t1, INIT_ELO)
            e2 = elo.get(t2, INIT_ELO)
            match_elo_snapshot[row["match_url"]] = (e1, e2)

            # ── K = base × format × venue × opponent_reliability ──
            match_type = str(row.get("match_type", "BO1"))
            is_lan = bool(row.get("is_lan", False)) if pd.notna(row.get("is_lan")) else False

            k_format = BO_MULT.get(match_type, 1.0)
            k_venue = 1.3 if is_lan else 1.0

            # Opponent reliability: dampen K when either team's ELO is uncertain.
            # Use BOTH match count AND ELO deviation from INIT — a team with few
            # dataset matches but a diverged ELO is still somewhat "known".
            mc1, mc2 = match_count.get(t1, 0), match_count.get(t2, 0)
            dev1 = abs(e1 - INIT_ELO)
            dev2 = abs(e2 - INIT_ELO)
            known1 = mc1 >= RELIABLE_MATCHES or (mc1 >= SEMI_RELIABLE and dev1 > 50)
            known2 = mc2 >= RELIABLE_MATCHES or (mc2 >= SEMI_RELIABLE and dev2 > 50)
            if known1 and known2:
                k_reliability = 1.0
            elif known1 or known2:
                k_reliability = 0.75
            else:
                k_reliability = 0.5

            # VRS match quality weight
            k_vrs = 1.0
            vrs_lookup = self._load_valve_rankings_lookup()
            if vrs_lookup is not None:
                date_str = str(row["date"])[:10]
                v1 = vrs_lookup.get_points(t1, date_str)
                v2 = vrs_lookup.get_points(t2, date_str)
                if v1 and v2:
                    avg_vrs = (v1 + v2) / 2
                    k_vrs = 0.8 + 0.4 * min(1.0, max(0, (avg_vrs - 1000)) / 800)
                elif v1 or v2:
                    k_vrs = 0.9
                else:
                    k_vrs = 0.7

            K = BASE_K * k_format * k_venue * k_reliability * k_vrs

            # Record history BEFORE updating ELO (strictly causal)
            match_date = row["date"]
            won = row["winner"]  # 1 = team1 won this map; approximate series winner
            team_match_history.setdefault(t1, []).append(
                (match_date, e2, won == 1, match_type, is_lan)
            )
            team_match_history.setdefault(t2, []).append(
                (match_date, e1, won == 0, match_type, is_lan)
            )

            # ELO update
            score1 = 1.0 if won == 1 else 0.0
            expected1 = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))
            elo[t1] = e1 + K * (score1 - expected1)
            elo[t2] = e2 + K * ((1.0 - score1) - (1.0 - expected1))

            # Update match counts
            match_count[t1] = match_count.get(t1, 0) + 1
            match_count[t2] = match_count.get(t2, 0) + 1

        # Store for downstream use by build_opponent_strength_features()
        self._match_elo_snapshot = match_elo_snapshot
        self._team_match_history = team_match_history
        self._elo_ratings = elo
        self._match_count = match_count
        self._INIT_ELO = INIT_ELO

        # Map back to full dataframe via match_url
        urls = self.df["match_url"].values
        elo1_arr = np.full(len(self.df), INIT_ELO, dtype=np.float32)
        elo2_arr = np.full(len(self.df), INIT_ELO, dtype=np.float32)

        for i in range(len(self.df)):
            snapshot = match_elo_snapshot.get(urls[i])
            if snapshot is not None:
                elo1_arr[i] = snapshot[0]
                elo2_arr[i] = snapshot[1]

        # For augmented rows (team1↔team2 swapped), flip the ELO
        if has_aug:
            aug_mask = self.df["is_augmented"] == 1
            elo1_arr[aug_mask], elo2_arr[aug_mask] = (
                elo2_arr[aug_mask].copy(),
                elo1_arr[aug_mask].copy(),
            )

        self.df["team1_elo"] = elo1_arr
        self.df["team2_elo"] = elo2_arr
        self.df["elo_delta"] = elo1_arr - elo2_arr
        features.extend(["team1_elo", "team2_elo", "elo_delta"])

        # Stats
        n_teams = len(elo)
        elo_vals = sorted(elo.values(), reverse=True)
        print(f"  ✓ 为 {n_teams} 支队伍计算了增量 ELO (variable K)")
        print(f"    Top ELO: {elo_vals[0]:.0f}, Median: {elo_vals[n_teams//2]:.0f}, Bottom: {elo_vals[-1]:.0f}")
        print(f"  ✓ 构建了 {len(features)} 个 ELO 特征")
        return features

    def _load_valve_rankings_lookup(self):
        """Load the Valve official rankings CSV via the shared lookup utility.

        Tries multiple paths relative to this script's location:
          1. ./data/valve_rankings_global.csv  (bundled sample data)
          2. ../HLTV/data/valve_rankings_global.csv  (local dev layout)
          3. Configured path from config.yaml
        Returns a ValveRankingsLookup instance or None.
        """
        if hasattr(self, "_vrs_lookup"):
            return self._vrs_lookup

        # Candidate paths
        base = Path(__file__).resolve().parent
        candidates = [
            base / "data" / "valve_rankings_global.csv",
            base.parent / "HLTV" / "data" / "valve_rankings_global.csv",
        ]
        cfg_path = self.config.get("valve_rankings_csv")
        if cfg_path:
            candidates.insert(0, Path(cfg_path))

        csv_path = None
        for p in candidates:
            if p.exists():
                csv_path = p
                break

        if csv_path is None:
            self._vrs_lookup = None
            return None

        # Import the lookup utility
        scripts_dir = base.parent / "HLTV" / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        try:
            from valve_rankings_lookup import ValveRankingsLookup
            self._vrs_lookup = ValveRankingsLookup(str(csv_path))
            return self._vrs_lookup
        except Exception as e:
            print(f"  ⚠ 加载 ValveRankingsLookup 失败: {e}")
            self._vrs_lookup = None
            return None

    def build_vrs_features(self):
        """Build VRS (Valve Ranking System) features.

        Data sources (in priority order):
          1. Valve official rankings CSV (valve_rankings_global.csv)
             — comprehensive: 700+ teams, 42 snapshots, 2024-08~2026-05
          2. HLTV match-page VRS (team1_vrs_points_raw columns)
             — sparse: only Valve-ranked events have these

        For each match, we first try the Valve CSV (nearest prior snapshot).
        If missing, fall back to HLTV match-page VRS with forward-fill.
        This is strictly causal — only uses data available before match date.

        Features produced:
          team1_vrs_points, team2_vrs_points, vrs_points_delta
          team1_vrs_rank, team2_vrs_rank, vrs_rank_delta
        """
        print("\n[VRS] 构建 Valve 排名特征 (official CSV + match-page fallback)...")

        features = []
        df = self.df

        # ── Source 1: Valve official rankings CSV ──
        vrs_lookup = self._load_valve_rankings_lookup()
        csv_available = vrs_lookup is not None
        if csv_available:
            print(f"  ✓ Valve 官方排名: {len(vrs_lookup.snapshot_dates)} 期快照, {vrs_lookup.num_teams} 支队伍")
        else:
            print("  ⚠ 未找到 Valve 官方排名 CSV，仅使用 HLTV 比赛页 VRS")

        # ── Source 2: HLTV match-page VRS (forward-fill) ──
        has_raw = "team1_vrs_points_raw" in df.columns
        has_aug = "is_augmented" in df.columns
        if has_aug:
            orig_mask = df["is_augmented"] == 0
        else:
            orig_mask = pd.Series(True, index=df.index)

        # Build forward-fill lookup from match-page VRS
        match_page_vrs = {}  # match_url → (t1_pts, t1_rank, t2_pts, t2_rank)
        if has_raw:
            orig_df = df[orig_mask].copy()
            series_df = orig_df.drop_duplicates(subset="match_url").sort_values("date").reset_index()
            team_vrs_ffill = {}  # team → {'points': float, 'rank': float}

            for _, row in series_df.iterrows():
                t1, t2 = row["team1"], row["team2"]
                if pd.notna(row.get("team1_vrs_points_raw")):
                    team_vrs_ffill[t1] = {
                        "points": float(row["team1_vrs_points_raw"]),
                        "rank": float(row["team1_vrs_rank_raw"]) if pd.notna(row.get("team1_vrs_rank_raw")) else np.nan,
                    }
                if pd.notna(row.get("team2_vrs_points_raw")):
                    team_vrs_ffill[t2] = {
                        "points": float(row["team2_vrs_points_raw"]),
                        "rank": float(row["team2_vrs_rank_raw"]) if pd.notna(row.get("team2_vrs_rank_raw")) else np.nan,
                    }
                v1 = team_vrs_ffill.get(t1, {})
                v2 = team_vrs_ffill.get(t2, {})
                match_page_vrs[row["match_url"]] = (
                    v1.get("points", np.nan), v1.get("rank", np.nan),
                    v2.get("points", np.nan), v2.get("rank", np.nan),
                )

        # ── Merge: CSV primary, match-page fallback ──
        urls = df["match_url"].values
        t1s = df["team1"].values
        t2s = df["team2"].values
        dates = df["date"].values

        t1_pts = np.full(len(df), np.nan, dtype=np.float32)
        t1_rank = np.full(len(df), np.nan, dtype=np.float32)
        t2_pts = np.full(len(df), np.nan, dtype=np.float32)
        t2_rank = np.full(len(df), np.nan, dtype=np.float32)

        csv_hits = 0
        ffill_hits = 0

        for i in range(len(df)):
            filled = False

            # Try CSV first
            if csv_available:
                date_str = str(dates[i])[:10]
                info1 = vrs_lookup.get(str(t1s[i]), date_str)
                info2 = vrs_lookup.get(str(t2s[i]), date_str)
                if info1 is not None:
                    t1_pts[i] = info1["points"]
                    t1_rank[i] = info1["standing"]
                if info2 is not None:
                    t2_pts[i] = info2["points"]
                    t2_rank[i] = info2["standing"]
                if info1 is not None and info2 is not None:
                    csv_hits += 1
                    filled = True

            # Fall back to match-page VRS if CSV didn't cover both teams
            if not filled and has_raw:
                snap = match_page_vrs.get(urls[i])
                if snap is not None:
                    if np.isnan(t1_pts[i]) and not np.isnan(snap[0]):
                        t1_pts[i] = snap[0]
                    if np.isnan(t1_rank[i]) and not np.isnan(snap[1]):
                        t1_rank[i] = snap[1]
                    if np.isnan(t2_pts[i]) and not np.isnan(snap[2]):
                        t2_pts[i] = snap[2]
                    if np.isnan(t2_rank[i]) and not np.isnan(snap[3]):
                        t2_rank[i] = snap[3]
                    if np.isfinite(t1_pts[i]) and np.isfinite(t2_pts[i]):
                        ffill_hits += 1

        # For augmented rows, swap team1↔team2
        if has_aug:
            aug_mask = df["is_augmented"] == 1
            t1_pts[aug_mask], t2_pts[aug_mask] = t2_pts[aug_mask].copy(), t1_pts[aug_mask].copy()
            t1_rank[aug_mask], t2_rank[aug_mask] = t2_rank[aug_mask].copy(), t1_rank[aug_mask].copy()

        df["team1_vrs_points"] = t1_pts
        df["team2_vrs_points"] = t2_pts
        df["vrs_points_delta"] = t1_pts - t2_pts
        df["team1_vrs_rank"] = t1_rank
        df["team2_vrs_rank"] = t2_rank
        df["vrs_rank_delta"] = t1_rank - t2_rank

        features.extend([
            "team1_vrs_points", "team2_vrs_points", "vrs_points_delta",
            "team1_vrs_rank", "team2_vrs_rank", "vrs_rank_delta",
        ])

        # Stats
        has_vrs = np.isfinite(t1_pts) & np.isfinite(t2_pts)
        n_with = int(has_vrs.sum())
        n_total = len(df)
        pct = 100.0 * n_with / n_total if n_total > 0 else 0
        print(f"  ✓ {n_with}/{n_total} 行有 VRS 特征 ({pct:.1f}%)")
        if csv_available:
            print(f"    - Valve CSV 命中: {csv_hits} 行")
        if has_raw:
            print(f"    - HLTV 比赛页补充: {ffill_hits} 行")
        print(f"  ✓ 构建了 {len(features)} 个 VRS 特征")
        return features

    def build_opponent_strength_features(self):
        """Build Strength-of-Schedule and quality-adjusted features.

        Requires ``build_elo_features()`` to have run first so that
        ``self._team_match_history`` and ``self._match_elo_snapshot`` exist.

        For each match, looks back 180 days per team and computes:
          - sos_6m:           avg opponent ELO (Strength of Schedule)
          - qa_winrate_6m:    win rate weighted by opponent ELO
          - strong_winrate_6m: win rate vs opponents with ELO > median
          - weak_farm_ratio_6m: fraction of matches vs low-ELO opponents
          - best_win_elo_6m:  highest opponent ELO among wins
          - worst_loss_elo_6m: lowest opponent ELO among losses
          - elo_uncertainty:  1 / sqrt(quality_match_count) — fewer quality
                               matches → higher uncertainty
        All features are strictly causal (only use pre-match data).
        """
        print("\n[SoS] 构建对手强度 / 赛程质量特征...")

        if not hasattr(self, "_team_match_history"):
            print("  ⚠ 未找到 ELO 历史数据（请先运行 build_elo_features），跳过")
            return []

        features = []
        INIT_ELO = self._INIT_ELO
        WINDOW_DAYS = 180

        # Fixed thresholds based on ELO scale — no leakage from future data.
        # "Strong" = consistently above average; "Weak" = well below average.
        STRONG_THRESHOLD = INIT_ELO + 50   # 1050
        WEAK_THRESHOLD = INIT_ELO - 100    # 900
        # "Known" opponent = ELO has diverged enough OR has enough matches.
        # Many opponents have few dataset matches (scraping focused on Major teams)
        # but their ELO may still be meaningful if it moved away from INIT.
        KNOWN_MIN_MATCHES = 5
        KNOWN_ELO_DEVIATION = 50  # |elo - 1000| > 50 → system has a signal

        has_aug = "is_augmented" in self.df.columns
        if has_aug:
            orig_mask = self.df["is_augmented"] == 0
        else:
            orig_mask = pd.Series(True, index=self.df.index)

        orig_df = self.df[orig_mask].copy()
        series_df = orig_df[orig_df["map_index"] == 0].sort_values("date").reset_index()

        # Rebuild walking history + running match count per team
        team_hist_walk = {}   # team → [(date, opp_elo, won, mt, lan, opp_mc)]

        # match_url → tuple of 10 stats per team (20 total)
        match_sos_snapshot = {}

        N_STATS = 10  # features per team

        def _compute_sos_for_team(team, match_date):
            """Compute SoS features using matches strictly before match_date."""
            hist = team_hist_walk.get(team, [])
            if not hist:
                return (np.nan,) * N_STATS

            cutoff = match_date - pd.Timedelta(days=WINDOW_DAYS)
            window = [r for r in hist if r[0] >= cutoff and r[0] < match_date]

            if not window:
                return (np.nan,) * N_STATS

            opp_elos = np.array([r[1] for r in window], dtype=np.float64)
            wins = np.array([r[2] for r in window], dtype=np.float64)
            opp_mcs = np.array([r[5] for r in window], dtype=np.float64)

            # 1) Strength of Schedule — mean opponent ELO
            sos = float(np.mean(opp_elos))

            # 2) Quality-adjusted win rate (wins weighted by opponent ELO)
            total_weight = float(np.sum(opp_elos))
            qa_winrate = float(np.sum(wins * opp_elos)) / total_weight if total_weight > 0 else 0.5

            # 3) Win rate vs strong opponents (ELO > fixed threshold)
            strong_mask = opp_elos > STRONG_THRESHOLD
            strong_wr = float(wins[strong_mask].mean()) if strong_mask.sum() > 0 else np.nan

            # 4) Weak opponent farm ratio
            weak_mask = opp_elos < WEAK_THRESHOLD
            weak_farm = float(weak_mask.sum()) / len(window)

            # 5) Best win ELO
            win_elos = opp_elos[wins.astype(bool)]
            best_win = float(np.max(win_elos)) if len(win_elos) > 0 else np.nan

            # 6) Worst loss ELO
            loss_elos = opp_elos[~wins.astype(bool)]
            worst_loss = float(np.min(loss_elos)) if len(loss_elos) > 0 else np.nan

            # 7) Uncertainty: 1 / sqrt(quality_match_count)
            n_quality = int((~weak_mask).sum())
            uncertainty = 1.0 / np.sqrt(max(n_quality, 1))

            # 8) Effective match count (quality-weighted)
            #    "Known" = opponent has enough matches OR their ELO diverged from INIT.
            #    This handles opponents with few dataset matches but meaningful ELO.
            opp_elo_dev = np.abs(opp_elos - INIT_ELO)
            is_known = (opp_mcs >= KNOWN_MIN_MATCHES) | (opp_elo_dev > KNOWN_ELO_DEVIATION)
            eff_count = float(np.sum(np.where(is_known, 1.0, 0.25)))

            # 9) Known opponent ratio
            known_ratio = float(is_known.sum()) / len(window)

            # 10) LAN match ratio in window
            lan_flags = np.array([r[4] for r in window], dtype=np.float64)
            lan_ratio = float(np.mean(lan_flags))

            return (sos, qa_winrate, strong_wr, weak_farm, best_win,
                    worst_loss, uncertainty, eff_count, known_ratio, lan_ratio)

        # Walk through series chronologically
        running_mc = {}  # team → match count so far
        for _, row in series_df.iterrows():
            t1, t2 = row["team1"], row["team2"]
            e1_pre, e2_pre = self._match_elo_snapshot.get(row["match_url"], (INIT_ELO, INIT_ELO))
            match_date = row["date"]

            # Compute BEFORE adding this match (strictly causal)
            t1_stats = _compute_sos_for_team(t1, match_date)
            t2_stats = _compute_sos_for_team(t2, match_date)
            match_sos_snapshot[row["match_url"]] = t1_stats + t2_stats

            # Add this match to walking history
            won = row["winner"]
            match_type = str(row.get("match_type", "BO1"))
            is_lan = bool(row.get("is_lan", False)) if pd.notna(row.get("is_lan")) else False
            mc_t1 = running_mc.get(t1, 0)
            mc_t2 = running_mc.get(t2, 0)
            # Record opponent's match count at time of match (measures how "known" the opp is)
            team_hist_walk.setdefault(t1, []).append(
                (match_date, e2_pre, won == 1, match_type, is_lan, mc_t2)
            )
            team_hist_walk.setdefault(t2, []).append(
                (match_date, e1_pre, won == 0, match_type, is_lan, mc_t1)
            )
            running_mc[t1] = mc_t1 + 1
            running_mc[t2] = mc_t2 + 1

        # Map to full dataframe
        N = len(self.df)
        FEAT_NAMES = [
            "sos_6m", "qa_winrate_6m", "strong_winrate_6m",
            "weak_farm_ratio_6m", "best_win_elo_6m", "worst_loss_elo_6m",
            "elo_uncertainty", "eff_match_count_6m", "known_opp_ratio_6m",
            "lan_match_ratio_6m",
        ]
        arrs = {}
        for prefix in ["team1", "team2"]:
            for fname in FEAT_NAMES:
                arrs[f"{prefix}_{fname}"] = np.full(N, np.nan, dtype=np.float32)

        urls = self.df["match_url"].values
        for i in range(N):
            snap = match_sos_snapshot.get(urls[i])
            if snap is not None:
                for j, fname in enumerate(FEAT_NAMES):
                    arrs[f"team1_{fname}"][i] = snap[j]
                    arrs[f"team2_{fname}"][i] = snap[N_STATS + j]

        # Swap for augmented rows
        if has_aug:
            aug_mask = self.df["is_augmented"] == 1
            for fname in FEAT_NAMES:
                k1, k2 = f"team1_{fname}", f"team2_{fname}"
                arrs[k1][aug_mask], arrs[k2][aug_mask] = (
                    arrs[k2][aug_mask].copy(), arrs[k1][aug_mask].copy()
                )

        # Write to dataframe + create diff features
        for fname in FEAT_NAMES:
            k1, k2 = f"team1_{fname}", f"team2_{fname}"
            self.df[k1] = arrs[k1]
            self.df[k2] = arrs[k2]
            diff_col = f"{fname}_diff"
            self.df[diff_col] = arrs[k1] - arrs[k2]
            features.extend([k1, k2, diff_col])

        # Stats
        has_sos = np.isfinite(arrs["team1_sos_6m"]) & np.isfinite(arrs["team2_sos_6m"])
        n_with = int(has_sos.sum())
        print(f"  ✓ {n_with}/{N} 行有 SoS 特征 ({100.0*n_with/N:.1f}%)")
        print(f"  ✓ 强队阈值: {STRONG_THRESHOLD:.0f}, 弱队阈值: {WEAK_THRESHOLD:.0f} (固定, 无泄漏)")
        print(f"  ✓ 构建了 {len(features)} 个对手强度特征 (含 diff)")
        return features

    def build_roster_reliability_features(self):
        """Build roster age, core sample size, and team-shell risk features."""
        print("\n[R] 构建阵容可靠性特征 (core age / core samples / shell risk)...")

        features = []
        player_stats_path = self.config["data"].get("player_stats_pattern", "player_stats.csv")
        data_dir = Path(self.config["data"]["data_dir"])
        ps_path = data_dir / player_stats_path
        if not ps_path.exists():
            print("  ! 未找到 player_stats.csv，跳过阵容可靠性特征")
            return []

        ps = pd.read_csv(ps_path, on_bad_lines="skip")
        ps["match_date"] = pd.to_datetime(ps.get("match_date", ""), errors="coerce")
        if "match_url" in ps.columns and self.df is not None:
            url_dates = self.df.groupby("match_url")["date"].first().to_dict()
            missing_dates = ps["match_date"].isna()
            ps.loc[missing_dates, "match_date"] = ps.loc[missing_dates, "match_url"].map(url_dates)
        data_end_date = self.config.get("data", {}).get("data_end_date")
        if data_end_date:
            ps = ps[ps["match_date"].notna() & (ps["match_date"] <= pd.to_datetime(data_end_date))]
        both_ps = ps[ps["side"] == "Both"].copy() if "side" in ps.columns else ps.copy()
        if both_ps.empty:
            print("  ! player_stats 中没有 Both side 数据，跳过")
            return []

        both_ps["rating"] = pd.to_numeric(both_ps.get("rating", np.nan), errors="coerce")
        roster_per_match = (
            both_ps.groupby(["match_url", "team"])["player_name"]
            .apply(lambda values: set(str(v) for v in values if pd.notna(v)))
            .reset_index()
            .rename(columns={"player_name": "roster"})
        )
        roster_per_match = roster_per_match.merge(
            both_ps[["match_url", "match_date"]].drop_duplicates(),
            on="match_url",
            how="left",
        )
        roster_per_match = roster_per_match.dropna(subset=["match_date"]).sort_values("match_date")

        team_roster_hist = defaultdict(list)
        team_player_first_seen = defaultdict(dict)
        for _, row in roster_per_match.iterrows():
            team = row["team"]
            match_date = row["match_date"]
            roster = row["roster"]
            if not roster:
                continue
            team_roster_hist[team].append((match_date, roster))
            first_seen = team_player_first_seen[team]
            for player in roster:
                if player not in first_seen or match_date < first_seen[player]:
                    first_seen[player] = match_date

        player_hist = defaultdict(list)
        for _, row in both_ps.dropna(subset=["match_date"]).iterrows():
            player = str(row.get("player_name", ""))
            if not player:
                continue
            player_hist[player].append((row["match_date"], row.get("rating", np.nan)))
        for player in list(player_hist):
            player_hist[player].sort(key=lambda item: item[0])

        stat_names = [
            "core_age_days",
            "core_matches_90d",
            "core_matches_180d",
            "core_match_share_180d",
            "new_player_count_60d",
            "new_player_count_120d",
            "player_experience_matches_180d",
            "player_experience_matches_365d",
            "player_experience_rating_180d",
            "team_history_reliability",
            "shell_change_risk",
        ]

        def _player_experience(roster, current_date, days):
            cutoff = current_date - pd.Timedelta(days=days)
            counts = []
            ratings = []
            for player in roster:
                window = [
                    (date, rating)
                    for date, rating in player_hist.get(player, [])
                    if cutoff <= date < current_date
                ]
                counts.append(len(window))
                ratings.extend(float(rating) for _, rating in window if pd.notna(rating))
            avg_count = float(np.mean(counts)) if counts else 0.0
            avg_rating = float(np.mean(ratings)) if ratings else 1.0
            return avg_count, avg_rating

        def _team_roster_features(team, current_date):
            hist = [(date, roster) for date, roster in team_roster_hist.get(team, []) if date < current_date]
            if not hist:
                return {
                    "core_age_days": 0.0,
                    "core_matches_90d": 0.0,
                    "core_matches_180d": 0.0,
                    "core_match_share_180d": 0.0,
                    "new_player_count_60d": 0.0,
                    "new_player_count_120d": 0.0,
                    "player_experience_matches_180d": 0.0,
                    "player_experience_matches_365d": 0.0,
                    "player_experience_rating_180d": 1.0,
                    "team_history_reliability": 0.0,
                    "shell_change_risk": 0.0,
                }

            latest_roster = hist[-1][1]
            threshold = min(3, max(len(latest_roster), 1))
            core_hist = [(date, roster) for date, roster in hist if len(latest_roster & roster) >= threshold]
            first_core_date = core_hist[0][0] if core_hist else hist[-1][0]
            core_age_days = float(max((current_date - first_core_date).days, 0))

            cutoff_90 = current_date - pd.Timedelta(days=90)
            cutoff_180 = current_date - pd.Timedelta(days=180)
            total_180 = sum(1 for date, _ in hist if cutoff_180 <= date < current_date)
            core_90 = sum(1 for date, _ in core_hist if cutoff_90 <= date < current_date)
            core_180 = sum(1 for date, _ in core_hist if cutoff_180 <= date < current_date)
            core_share_180 = float(core_180 / total_180) if total_180 else 0.0

            first_seen = team_player_first_seen.get(team, {})
            new_60 = 0
            new_120 = 0
            for player in latest_roster:
                seen_date = first_seen.get(player)
                if seen_date is None:
                    continue
                age = (current_date - seen_date).days
                if age <= 60:
                    new_60 += 1
                if age <= 120:
                    new_120 += 1

            exp_180, rating_180 = _player_experience(latest_roster, current_date, 180)
            exp_365, _ = _player_experience(latest_roster, current_date, 365)
            reliability = float(min(core_180 / 30.0, 1.0) * core_share_180)
            shell_risk = float(
                (1.0 - core_share_180)
                * min(total_180 / 30.0, 1.0)
                * (1.0 - min(core_180 / 20.0, 1.0))
            )

            return {
                "core_age_days": min(core_age_days, 730.0),
                "core_matches_90d": float(core_90),
                "core_matches_180d": float(core_180),
                "core_match_share_180d": core_share_180,
                "new_player_count_60d": float(new_60),
                "new_player_count_120d": float(new_120),
                "player_experience_matches_180d": exp_180,
                "player_experience_matches_365d": exp_365,
                "player_experience_rating_180d": rating_180,
                "team_history_reliability": reliability,
                "shell_change_risk": shell_risk,
            }

        N = len(self.df)
        values = {
            f"{side}_{name}": np.zeros(N, dtype=np.float32)
            for side in ["team1", "team2"]
            for name in stat_names
        }

        teams1 = self.df["team1"].values
        teams2 = self.df["team2"].values
        dates = pd.to_datetime(self.df["date"]).values
        for i in range(N):
            current_date = pd.Timestamp(dates[i])
            stats1 = _team_roster_features(teams1[i], current_date)
            stats2 = _team_roster_features(teams2[i], current_date)
            for name in stat_names:
                values[f"team1_{name}"][i] = stats1[name]
                values[f"team2_{name}"][i] = stats2[name]

        for name in stat_names:
            k1 = f"team1_{name}"
            k2 = f"team2_{name}"
            self.df[k1] = values[k1]
            self.df[k2] = values[k2]
            diff = f"{name}_diff"
            self.df[diff] = values[k1] - values[k2]
            features.extend([k1, k2, diff])

        for side in ["team1", "team2"]:
            weak_col = f"{side}_weak_farm_ratio_6m"
            rel_col = f"{side}_team_history_reliability"
            risk_col = f"{side}_farm_core_risk"
            if weak_col in self.df.columns:
                self.df[risk_col] = self.df[weak_col].astype(float).fillna(0.0) * (
                    1.0 - self.df[rel_col].astype(float).fillna(0.0)
                )
            else:
                self.df[risk_col] = 0.0
            features.append(risk_col)
        self.df["farm_core_risk_diff"] = self.df["team1_farm_core_risk"] - self.df["team2_farm_core_risk"]
        features.append("farm_core_risk_diff")

        print(f"  ✓ 构建了 {len(features)} 个阵容可靠性特征")
        return features

    def build_metadata_features(self):
        """从比赛元数据构建特征: 世界排名、赛事质量、阵容稳定性。"""
        print("\n[M] 构建元数据特征 (world_rank / event / lineup)...")

        features = []

        # ── 世界排名特征 ──
        if "team1_world_rank" in self.df.columns:
            # 先强制转数字（处理字符串/非法值）
            for side in ["team1", "team2"]:
                self.df[f"{side}_world_rank"] = pd.to_numeric(
                    self.df[f"{side}_world_rank"], errors="coerce"
                )

            # 排名归一化: rank 1-100 → 1.0-0.0 (rank越低=越强=值越大)
            for side in ["team1", "team2"]:
                col = f"{side}_world_rank"
                norm_col = f"{side}_world_rank_norm"
                self.df[norm_col] = self.df[col].apply(
                    lambda x: max(0.0, 1.0 - (x - 1) / 99.0) if pd.notna(x) and x > 0 else 0.5
                )
                features.append(norm_col)

            # 排名差
            self.df["world_rank_diff"] = (
                self.df["team2_world_rank"].fillna(50).astype(float)
                - self.df["team1_world_rank"].fillna(50).astype(float)
            )
            features.append("world_rank_diff")

            # 是否 Top10 / Top30 对决
            r1 = self.df["team1_world_rank"].fillna(999)
            r2 = self.df["team2_world_rank"].fillna(999)
            self.df["is_top10_matchup"] = ((r1 <= 10) & (r2 <= 10)).astype(int)
            self.df["is_top30_matchup"] = ((r1 <= 30) & (r2 <= 30)).astype(int)
            features.extend(["is_top10_matchup", "is_top30_matchup"])

        # ── 赛事质量特征 ──
        if "event_name" in self.df.columns:
            # 从赛事名称判断级别
            def event_tier(name):
                if pd.isna(name):
                    return 0.5
                name_lower = str(name).lower()
                if "major" in name_lower:
                    return 1.0
                elif any(kw in name_lower for kw in ["blast", "esl pro", "iem", "pgl"]):
                    return 0.85
                elif any(kw in name_lower for kw in ["dreamhack", "rmc", "betboom", "yalla"]):
                    return 0.7
                elif any(kw in name_lower for kw in ["qualifier", "rmr", "open qualifier"]):
                    return 0.6
                else:
                    return 0.5
            self.df["event_tier"] = self.df["event_name"].apply(event_tier)
            features.append("event_tier")

        # ── 阶段特征 ──
        if "stage" in self.df.columns:
            def stage_importance(stage):
                if pd.isna(stage):
                    return 0.5
                stage_lower = str(stage).lower()
                if any(kw in stage_lower for kw in ["final", "grand final"]):
                    return 1.0
                elif any(kw in stage_lower for kw in ["semifinal", "semi-final"]):
                    return 0.9
                elif any(kw in stage_lower for kw in ["quarterfinal", "quarter-final", "playoff"]):
                    return 0.8
                elif any(kw in stage_lower for kw in ["elimination", "decider"]):
                    return 0.7
                elif any(kw in stage_lower for kw in ["group", "swiss", "opening"]):
                    return 0.5
                else:
                    return 0.5
            self.df["stage_importance"] = self.df["stage"].apply(stage_importance)
            features.append("stage_importance")

        # ── 阵容匹配度特征（当前阵容 vs 比赛时阵容）──
        # 计算每场比赛中，队伍当时的 lineup 有多少人在最近比赛中也出现过（阵容稳定性指标）
        if "team1_lineup" in self.df.columns:
            for side in ["team1", "team2"]:
                col = f"{side}_lineup"
                team_col = side  # "team1" or "team2"
                stability_col = f"{side}_lineup_size"
                # 简单特征: lineup 人数（正常=5, <5可能有替补或数据缺失）
                self.df[stability_col] = self.df[col].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                )
                features.append(stability_col)

        filled = self.df[[f for f in features if f in self.df.columns]].notna().mean()
        print(f"  ✓ 构建了 {len(features)} 个元数据特征")
        print(f"  填充率: {filled.mean():.1%}")
        return features

    def build_diff_features(self):
        """构建 team1 - team2 的 diff 特征。

        现有特征大量是 (team1_X, team2_X) 成对存在,模型只能通过树深度
        自己学相减。显式加 diff 让"相对优劣"信号一步到位,腾出树深度
        学其他关系。小样本量 (4500) 下尤其值得。

        所有 diff 都是 team1 - team2 视角:
          - 正值 = team1 有优势
          - 负值 = team1 处于劣势
          - 0 = 势均力敌
        对称增强行(team1↔team2 互换)会自动让 diff 翻号,与 winner 翻号
        匹配,模型学到的对称性自然正确。
        """
        print("\n[6/7] 构建 diff 特征 (team1 - team2)...")

        features = []

        # 选手属性 diffs(模型最看重的两个特征都是 top_rating)
        for stat in ["top_rating", "avg_rating", "rating_std"]:
            col1, col2 = f"team1_{stat}", f"team2_{stat}"
            if col1 in self.df.columns and col2 in self.df.columns:
                diff_col = f"{stat}_diff"
                self.df[diff_col] = self.df[col1].astype(float) - self.df[col2].astype(float)
                features.append(diff_col)

        # 状态 diff
        if "team1_recent_win_rate" in self.df.columns and "team2_recent_win_rate" in self.df.columns:
            self.df["recent_winrate_diff"] = (
                self.df["team1_recent_win_rate"].astype(float)
                - self.df["team2_recent_win_rate"].astype(float)
            )
            features.append("recent_winrate_diff")

        # 每张图的 win rate diff
        for map_name in self.map_names:
            col1 = f"team1_{map_name}_win_rate"
            col2 = f"team2_{map_name}_win_rate"
            if col1 in self.df.columns and col2 in self.df.columns:
                diff_col = f"{map_name}_winrate_diff"
                self.df[diff_col] = self.df[col1].astype(float) - self.df[col2].astype(float)
                features.append(diff_col)

        # 高级特征 diffs
        for stat in ["roster_stability", "upset_rate", "big_match_win_rate"]:
            col1, col2 = f"team1_{stat}", f"team2_{stat}"
            if col1 in self.df.columns and col2 in self.df.columns:
                diff_col = f"{stat}_diff"
                self.df[diff_col] = self.df[col1].astype(float) - self.df[col2].astype(float)
                features.append(diff_col)

        # 休息天数 diff(正值 = team1 比 team2 休息得久)
        if "team1_days_since_last" in self.df.columns and "team2_days_since_last" in self.df.columns:
            self.df["days_since_last_diff"] = (
                self.df["team1_days_since_last"].astype(float)
                - self.df["team2_days_since_last"].astype(float)
            )
            features.append("days_since_last_diff")

        # 世界排名 diff（由 build_metadata_features 生成的归一化排名）
        if "team1_world_rank_norm" in self.df.columns and "team2_world_rank_norm" in self.df.columns:
            self.df["world_rank_norm_diff"] = (
                self.df["team1_world_rank_norm"].astype(float)
                - self.df["team2_world_rank_norm"].astype(float)
            )
            features.append("world_rank_norm_diff")

        print(f"  ✓ 构建了 {len(features)} 个 diff 特征")
        return features

    def build_context_features(self):
        """Build match-context features (vectorized)."""
        print("\n[5/7] 构建上下文特征...")

        features = []
        format_map = {"BO1": 1, "BO3": 3, "BO5": 5}
        self.df["bo_format_encoded"] = self.df["match_type"].map(format_map)
        features.append("bo_format_encoded")

        self.df["map_index_normalized"] = self.df["map_index"] / 5.0
        features.append("map_index_normalized")

        if "is_lan" in self.df.columns:
            self.df["is_lan_int"] = self.df["is_lan"].fillna(False).astype(int)
        else:
            self.df["is_lan_int"] = 0
        features.append("is_lan_int")

        # ── days_since_last_match (vectorized) ──
        DEFAULT_DAYS = 14
        team_last_dates = {}
        match_team_days = {}

        if "is_augmented" in self.df.columns:
            series_first = self.df[
                (self.df["is_augmented"] == 0) & (self.df["map_index"] == 0)
            ]
        else:
            series_first = self.df[self.df["map_index"] == 0]
        series_first = series_first.sort_values(["date", "match_url"])

        # This loop is over series_first only (~1/3 of rows), not full df
        # 用 pd.Timestamp 保证跨 numpy 1.x / 2.x 的 timedelta 行为一致
        # (numpy 2.x 的 .item() 在 timedelta64 上返回 datetime.timedelta 而非 int)
        sf_dates = pd.DatetimeIndex(series_first["date"].values)
        sf_t1 = series_first["team1"].values
        sf_t2 = series_first["team2"].values
        sf_urls = series_first["match_url"].values

        for i in range(len(series_first)):
            d = sf_dates[i]
            t1, t2 = sf_t1[i], sf_t2[i]
            m = sf_urls[i]
            d1 = (d - team_last_dates[t1]).total_seconds() / 86400.0 if t1 in team_last_dates else float(DEFAULT_DAYS)
            d2 = (d - team_last_dates[t2]).total_seconds() / 86400.0 if t2 in team_last_dates else float(DEFAULT_DAYS)
            match_team_days[(m, t1)] = min(max(d1, 0.0), 60.0) / 60.0
            match_team_days[(m, t2)] = min(max(d2, 0.0), 60.0) / 60.0
            team_last_dates[t1] = d
            team_last_dates[t2] = d

        # Vectorized lookup via merge instead of iterrows
        fallback = DEFAULT_DAYS / 60.0
        urls = self.df["match_url"].values
        t1s = self.df["team1"].values
        t2s = self.df["team2"].values
        days1 = np.full(len(self.df), fallback, dtype=np.float32)
        days2 = np.full(len(self.df), fallback, dtype=np.float32)
        for i in range(len(self.df)):
            days1[i] = match_team_days.get((urls[i], t1s[i]), fallback)
            days2[i] = match_team_days.get((urls[i], t2s[i]), fallback)
        self.df["team1_days_since_last"] = days1
        self.df["team2_days_since_last"] = days2
        features.extend(["team1_days_since_last", "team2_days_since_last"])

        map_dummies = pd.get_dummies(self.df["map_name"], prefix="map")
        self.df = pd.concat([self.df, map_dummies], axis=1)
        features.extend(map_dummies.columns.tolist())

        print(f"  ✓ 构建了 {len(features)} 个上下文特征")
        return features

    def _matches_feature_filter(self, feature, patterns):
        lower = feature.lower()
        return any(str(pattern).lower() in lower for pattern in patterns)

    def apply_feature_filter(self, features):
        cfg = ((self.config.get("features") or {}).get("feature_filter") or {})
        if not cfg or not cfg.get("enabled", False):
            return features

        include_patterns = cfg.get("include_patterns") or []
        exclude_patterns = cfg.get("exclude_patterns") or []
        kept = list(features)

        if include_patterns:
            kept = [
                feature for feature in kept
                if self._matches_feature_filter(feature, include_patterns)
            ]
        if exclude_patterns:
            kept = [
                feature for feature in kept
                if not self._matches_feature_filter(feature, exclude_patterns)
            ]

        print(f"  Feature filter: {len(features)} -> {len(kept)}")
        return kept

    def finalize_features(self, all_features):
        """Create and save the final feature matrix."""
        print("\n[7/7] 最终化特征集...")

        available_features = [feature for feature in all_features if feature in self.df.columns]
        missing_features = sorted(set(all_features) - set(available_features))
        if missing_features:
            print(f"  ! 缺失特征: {missing_features}")

        available_features = self.apply_feature_filter(available_features)

        X = self.df[available_features].copy().fillna(0)
        y = self.df["winner"].copy()

        Path("data").mkdir(exist_ok=True)
        X.to_csv("data/X_features.csv", index=False)
        y.to_csv("data/y_labels.csv", index=False)

        with open("data/feature_names.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(available_features))

        self.df.to_csv("data/map_level_dataset_with_features.csv", index=False)

        print(f"  ✓ 特征矩阵: {X.shape}")
        print(f"  ✓ 标签向量: {y.shape}")
        print("  ✓ 已保存到 data/ 目录")

        # diff 特征单独成类(避免被分到 map/player 等其他类里重复计数)
        diff_features = [feature for feature in available_features if feature.endswith("_diff")]
        non_diff = [feature for feature in available_features if feature not in diff_features]
        map_features = [
            feature
            for feature in non_diff
            if "map" in feature.lower() or "pick" in feature.lower() or "h2h" in feature.lower()
        ]
        player_features = [
            feature for feature in non_diff if "rating" in feature or "adr" in feature or "kast" in feature
        ]
        momentum_features = [
            feature
            for feature in non_diff
            if any(key in feature for key in ["previous", "series", "decider", "streak"])
        ]
        advanced_features = [
            feature
            for feature in non_diff
            if any(key in feature for key in ["roster_stability", "upset_rate", "big_match"])
        ]
        context_features = [
            feature
            for feature in non_diff
            if feature not in map_features + player_features + momentum_features + advanced_features
        ]

        print("\n特征分类汇总:")
        print(f"  - 地图特征: {len(map_features)} 维")
        print(f"  - 选手特征: {len(player_features)} 维")
        print(f"  - Momentum特征: {len(momentum_features)} 维")
        print(f"  - 高级特征: {len(advanced_features)} 维")
        print(f"  - 上下文特征: {len(context_features)} 维")
        print(f"  - Diff 特征: {len(diff_features)} 维")
        print(f"  - 总计: {len(available_features)} 维")

    def run(self):
        print("\n" + "=" * 80)
        print("特征工程")
        print("=" * 80)

        self.load_data()
        all_features = []
        all_features.extend(self.build_map_features())
        all_features.extend(self.build_momentum_features())
        all_features.extend(self.build_advanced_features())
        # diff 特征必须在 context 之后,因为它依赖 days_since_last
        all_features.extend(self.build_context_features())
        all_features.extend(self.build_elo_features())
        all_features.extend(self.build_vrs_features())
        all_features.extend(self.build_opponent_strength_features())
        all_features.extend(self.build_roster_reliability_features())
        all_features.extend(self.build_metadata_features())
        all_features.extend(self.build_diff_features())
        self.finalize_features(all_features)

        print("\n" + "=" * 80)
        print("✓ 特征工程完成！")
        print("=" * 80)
        print("\n下一步: python model_training.py")


if __name__ == "__main__":
    FeatureEngineering().run()
