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
            data_end_date = self.config.get("data", {}).get("data_end_date")
            if data_end_date:
                ps = ps[ps["match_date"].isna() | (ps["match_date"] <= pd.to_datetime(data_end_date))]
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

    def finalize_features(self, all_features):
        """Create and save the final feature matrix."""
        print("\n[7/7] 最终化特征集...")

        available_features = [feature for feature in all_features if feature in self.df.columns]
        missing_features = sorted(set(all_features) - set(available_features))
        if missing_features:
            print(f"  ! 缺失特征: {missing_features}")

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
        all_features.extend(self.build_diff_features())
        self.finalize_features(all_features)

        print("\n" + "=" * 80)
        print("✓ 特征工程完成！")
        print("=" * 80)
        print("\n下一步: python model_training.py")


if __name__ == "__main__":
    FeatureEngineering().run()
