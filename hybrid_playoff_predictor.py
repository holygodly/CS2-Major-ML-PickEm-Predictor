"""
Hybrid playoff predictor:
- XGBoost map model
- ELO fallback
- veto / side simulation for BO3 and BO5

This version is designed to stay aligned with the training pipeline:
- dynamic map pool support
- dynamic map feature support
- side-bias estimation from scraped half data
- current-tournament veto pool inferred from recent data unless configured
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from gpu_accelerator import DeviceManager
from map_side_analyzer import MapSideAnalyzer
from veto_simulator import TacticalAnalyzer, VetoSimulator


def _is_auto_count(value):
    if value in (None, 0, ""):
        return True
    return isinstance(value, str) and value.strip().lower() in {"0", "auto", "none", "null"}


def calculate_elo_from_data(map_data_df, player_stats_path=None):
    """Build team ELO from historical MAP-LEVEL results.

    每张地图单独更新 ELO，而非按比赛聚合。这样 ELO 的语义是"地图胜率"，
    与 XGBoost 预测的 P(team1 wins this map) 同一维度，
    避免在 hybrid 公式里把 match-level 概率当 map-level 用的系统性偏差。
    """
    import math

    df = map_data_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_url", "map_index"])

    base_elo = 1500
    time_decay_days = 50
    teams = set(df["team1"].unique()) | set(df["team2"].unique())

    initial_ratings = {team: base_elo for team in teams}
    if player_stats_path:
        try:
            player_stats = pd.read_csv(player_stats_path, on_bad_lines="skip")
            both_side = player_stats[player_stats["side"] == "Both"]
            for team in teams:
                team_players = both_side[both_side["team"] == team].tail(50)
                if team_players.empty:
                    continue
                avg_rating = team_players["rating"].mean()
                adjustment = float(np.clip((avg_rating - 1.0) * 500, -200, 200))
                initial_ratings[team] = base_elo + adjustment
        except Exception:
            pass

    elo = initial_ratings.copy()
    latest_date = df["date"].max()
    team_map_count = defaultdict(int)

    # 逐地图更新 ELO（map-level granularity）
    for _, row in df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        team1_won_map = int(row["winner"] == 1)

        count1 = team_map_count[team1]
        count2 = team_map_count[team2]
        k1 = 40 if count1 < 30 else (30 if count1 < 60 else 20)
        k2 = 40 if count2 < 30 else (30 if count2 < 60 else 20)
        adaptive_k = (k1 + k2) / 2

        days_ago = (latest_date - row["date"]).days
        time_weight = math.exp(-days_ago / time_decay_days)
        k_factor = adaptive_k * time_weight

        expected_team1 = 1 / (1 + math.pow(10, (elo[team2] - elo[team1]) / 400))
        elo[team1] += k_factor * (team1_won_map - expected_team1)
        elo[team2] += k_factor * ((1 - team1_won_map) - (1 - expected_team1))

        team_map_count[team1] += 1
        team_map_count[team2] += 1

    return elo


class HybridPlayoffPredictor:
    # 这三个常量必须与 feature_engineering.build_map_features 保持一致，
    # 否则推理时算出来的 win-rate 特征分布和训练样本对不上。
    # WIN_RATE 用弱先验 (1, 2);H2H 在 build_features_for_map 里显式传 (2, 4) 强先验。
    TIME_DECAY_HALF_LIFE_DAYS = 30.0
    WIN_RATE_PRIOR_WINS = 1.0
    WIN_RATE_PRIOR_TOTAL = 2.0

    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path).resolve()
        self.base_dir = self.config_path.parent

        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"

        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(self.models_dir / "xgboost_map_predictor.json")
        # 推理也用 GPU（DeviceManager 初始化在下方，此处先用环境变量判断）
        # set_param 在 DeviceManager 初始化后统一设置
        with (self.models_dir / "feature_names.json").open("r", encoding="utf-8") as file:
            self.feature_names = json.load(file)

        self.map_data = pd.read_csv(self.data_dir / "map_level_dataset_with_features.csv")
        self.map_data["date"] = pd.to_datetime(self.map_data["date"])
        # 增强行仅用于训练对称性，推理时只保留原始样本，
        # 否则 ELO groupby / tail(N) 等操作会受到重复行干扰。
        if "is_augmented" in self.map_data.columns:
            self.map_data = self.map_data[self.map_data["is_augmented"] == 0].copy()
        self.map_data = self.map_data.sort_values(["date", "match_url", "map_index"]).reset_index(drop=True)
        # 推理基准日期：用数据集中最新一场比赛的日期。
        # 含义是"以最新数据为今天，按距离今天的天数做时间衰减"。
        self._inference_reference_date = self.map_data["date"].max()
        self._time_decay_scale = self.TIME_DECAY_HALF_LIFE_DAYS / float(np.log(2))

        self.available_maps = sorted(self.map_data["map_name"].dropna().unique().tolist())
        self.current_map_pool = self._resolve_map_pool()
        self.feature_map_names = self._extract_feature_map_names()
        self.map_one_hot_features = [name for name in self.feature_names if name.startswith("map_")]

        self.veto_simulator = VetoSimulator(self.map_data, map_pool=self.current_map_pool)
        self.tactical_analyzer = TacticalAnalyzer(self.veto_simulator)
        self.side_analyzer = MapSideAnalyzer(self.map_data, map_pool=self.current_map_pool)
        self.device_mgr = DeviceManager()
        # 将 XGBoost 推理模型也放到 GPU 上
        self.xgb_model.set_param({"device": self.device_mgr.device_str})

        # 加载概率校准器（如果存在）
        self.calibrator = None
        calibrator_path = self.base_dir / "models" / "calibrator.pkl"
        if calibrator_path.exists():
            import pickle
            try:
                with calibrator_path.open("rb") as f:
                    self.calibrator = pickle.load(f)
                print(f"  [OK] Calibrator loaded from {calibrator_path.name}")
            except Exception as e:
                print(f"  [WARN] Calibrator load failed ({e}), using raw probs")
                self.calibrator = None

        hybrid_cfg = self.config["models"]["hybrid"]
        self.xgb_weight = hybrid_cfg["xgboost_weight"]
        self.momentum_weight = hybrid_cfg["lstm_weight"]
        self.elo_weight = hybrid_cfg["elo_weight"]

        source_data_dir = Path(self.config["data"]["data_dir"])
        if not source_data_dir.is_absolute():
            source_data_dir = (self.base_dir / source_data_dir).resolve()
        player_stats_path = source_data_dir / self.config["data"]["player_stats_pattern"]
        self.team_match_stats, self.default_rating_stats = self._load_team_match_stats(player_stats_path)
        self.team_elo = calculate_elo_from_data(self.map_data, player_stats_path=player_stats_path)

        print("[OK] Hybrid predictor initialized")
        print(f"  [OK] Current veto map pool: {', '.join(self.current_map_pool)}")
        print(f"  [OK] ELO ratings built for {len(self.team_elo)} teams")
        if self.device_mgr.cuda_available:
            print(f"  [OK] Prediction device: GPU ({self.device_mgr.cuda_device_name})")
        else:
            print(f"  [OK] Prediction device: CPU ({self.device_mgr.cpu_cores} cores)")

    def _resolve_map_pool(self):
        configured_pool = (self.config.get("tournament", {}) or {}).get("map_pool")
        if configured_pool:
            return list(configured_pool)

        recent_days = self.config.get("data", {}).get("recent_days_window", 60)
        latest_date = self.map_data["date"].max()
        cutoff_date = latest_date - pd.Timedelta(days=recent_days)

        recent_data = self.map_data[self.map_data["date"] >= cutoff_date]
        if recent_data.empty:
            recent_data = self.map_data

        usage = (
            recent_data.groupby("map_name")
            .agg(last_seen=("date", "max"), samples=("map_name", "size"))
            .sort_values(["last_seen", "samples"], ascending=[False, False])
        )

        resolved_pool = usage.head(7).index.tolist()
        if resolved_pool:
            return resolved_pool
        return list(self.available_maps)

    def _extract_feature_map_names(self):
        available_map_set = set(self.available_maps)
        feature_maps = []

        for name in self.feature_names:
            if not (name.startswith("team1_") and name.endswith("_win_rate")):
                continue
            candidate = name[len("team1_") : -len("_win_rate")]
            if candidate not in available_map_set:
                continue
            if f"team2_{candidate}_win_rate" in self.feature_names:
                feature_maps.append(candidate)

        return sorted(set(feature_maps))

    def _load_team_match_stats(self, player_stats_path):
        empty_stats = pd.DataFrame(columns=["match_url", "team", "rating", "adr", "kast", "match_date"])
        default_stats = {
            "avg_rating": 1.0,
            "rating_std": 0.15,
            "top_rating": 1.0,
            "avg_adr": 70.0,
            "avg_kast": 70.0,
        }

        if not player_stats_path.exists():
            return empty_stats, default_stats

        player_stats = pd.read_csv(player_stats_path, on_bad_lines="skip")
        for col in ["rating", "adr"]:
            if col in player_stats.columns:
                player_stats[col] = pd.to_numeric(player_stats[col], errors="coerce")
        if "kast" in player_stats.columns:
            player_stats["kast"] = (
                player_stats["kast"].astype(str).str.replace("%", "", regex=False)
            )
            player_stats["kast"] = pd.to_numeric(player_stats["kast"], errors="coerce")

        if "match_date" not in player_stats.columns:
            url_to_date = self.map_data.groupby("match_url")["date"].first().to_dict()
            player_stats["match_date"] = player_stats["match_url"].map(url_to_date)

        player_stats["match_date"] = pd.to_datetime(player_stats["match_date"], errors="coerce")
        data_end_date = self.config.get("data", {}).get("data_end_date")
        if data_end_date:
            player_stats = player_stats[player_stats["match_date"].isna() | (player_stats["match_date"] <= pd.to_datetime(data_end_date))]
        both_side_stats = player_stats[player_stats["side"] == "Both"].copy()
        if both_side_stats.empty:
            return empty_stats, default_stats

        match_team_stats = (
            both_side_stats.groupby(["match_url", "team"])
            .agg({"rating": "mean", "adr": "mean", "kast": "mean"})
            .reset_index()
        )
        match_team_stats = match_team_stats.merge(
            both_side_stats[["match_url", "match_date"]].drop_duplicates(),
            on="match_url",
            how="left",
        )
        match_team_stats = match_team_stats.sort_values(["match_date", "match_url"]).reset_index(drop=True)

        default_stats = {
            "avg_rating": float(both_side_stats["rating"].mean()) if both_side_stats["rating"].notna().any() else 1.0,
            "rating_std": float(both_side_stats["rating"].std()) if both_side_stats["rating"].notna().any() else 0.15,
            "top_rating": float(both_side_stats["rating"].mean()) if both_side_stats["rating"].notna().any() else 1.0,
            "avg_adr": float(both_side_stats["adr"].mean()) if both_side_stats["adr"].notna().any() else 70.0,
            "avg_kast": float(both_side_stats["kast"].mean()) if both_side_stats["kast"].notna().any() else 70.0,
        }
        return match_team_stats, default_stats

    def _collect_team_rating_stats(self, team_match_history):
        if team_match_history.empty:
            return dict(self.default_rating_stats)

        return {
            "avg_rating": float(team_match_history["rating"].mean()),
            "rating_std": float(team_match_history["rating"].std())
            if team_match_history["rating"].notna().sum() > 1
            else self.default_rating_stats["rating_std"],
            "top_rating": float(team_match_history["rating"].max()),
            "avg_adr": float(team_match_history["adr"].mean())
            if team_match_history["adr"].notna().any()
            else self.default_rating_stats["avg_adr"],
            "avg_kast": float(team_match_history["kast"].mean())
            if team_match_history["kast"].notna().any()
            else self.default_rating_stats["avg_kast"],
        }

    def _smoothed_weighted_win_rate(self, history, team, prior_wins=None, prior_total=None):
        """与 feature_engineering 一致：指数时间衰减 + Beta 先验平滑。

        基准日期使用 self._inference_reference_date（数据集中最新比赛日期），
        训练时基准是每行的 current_date，这是 leak-free 的；推理时所有特征都对
        着同一个"今天"算就行，等价于按距今天数加权。
        """
        prior_wins = self.WIN_RATE_PRIOR_WINS if prior_wins is None else prior_wins
        prior_total = self.WIN_RATE_PRIOR_TOTAL if prior_total is None else prior_total
        if history.empty:
            return float(prior_wins / prior_total)
        days_ago = (self._inference_reference_date - history["date"]).dt.days.clip(lower=0).to_numpy()
        time_weights = np.exp(-days_ago / self._time_decay_scale)
        won = (
            ((history["team1"] == team) & (history["winner"] == 1))
            | ((history["team2"] == team) & (history["winner"] == 0))
        ).to_numpy().astype(float)
        weighted_wins = float((won * time_weights).sum())
        weighted_total = float(time_weights.sum())
        return float((weighted_wins + prior_wins) / (weighted_total + prior_total))

    def _precompute_team_stats(self):
        if hasattr(self, "_team_stats_cache"):
            return

        history_data = self.map_data.copy()
        self._team_stats_cache = {}
        self._h2h_cache = {}

        # Pre-build roster lookup for stability computation
        roster_per_match = self._build_roster_lookup()

        all_teams = sorted(set(self.map_data["team1"].unique()) | set(self.map_data["team2"].unique()))
        ref_date = self._inference_reference_date

        # ── Pass 1: 基础统计（recent_win_rate, rating, map_win_rate, roster_stability）──
        # 这些不依赖其他队伍的 cache，可以一遍算完
        team_histories = {}
        for team in all_teams:
            self._team_stats_cache[team] = {}

            team_history = history_data[
                (history_data["team1"] == team) | (history_data["team2"] == team)
            ]
            team_histories[team] = team_history

            team_recent = team_history.tail(5)
            if team_recent.empty:
                self._team_stats_cache[team]["recent_win_rate"] = 0.5
            else:
                wins = (
                    ((team_recent["team1"] == team) & (team_recent["winner"] == 1)).sum()
                    + ((team_recent["team2"] == team) & (team_recent["winner"] == 0)).sum()
                )
                self._team_stats_cache[team]["recent_win_rate"] = float(wins / len(team_recent))

            team_match_history = self.team_match_stats[self.team_match_stats["team"] == team].tail(20)
            rating_stats = self._collect_team_rating_stats(team_match_history)
            self._team_stats_cache[team]["avg_rating"] = rating_stats["avg_rating"]
            self._team_stats_cache[team]["rating_std"] = rating_stats["rating_std"]
            self._team_stats_cache[team]["top_rating"] = rating_stats["top_rating"]

            for map_name in self.feature_map_names:
                team_map_history = history_data[
                    (
                        (history_data["team1"] == team) | (history_data["team2"] == team)
                    )
                    & (history_data["map_name"] == map_name)
                ]
                # 时间衰减 + Beta 平滑，与 feature_engineering 完全一致
                self._team_stats_cache[team][f"{map_name}_win_rate"] = (
                    self._smoothed_weighted_win_rate(team_map_history, team)
                )

            # 阵容稳定性不依赖其他队伍
            self._team_stats_cache[team]["roster_stability"] = self._compute_roster_stability(
                team, ref_date, roster_per_match
            )

        # ── Pass 2: 依赖其他队伍 recent_win_rate 的特征 ──
        # 此时所有队伍的 recent_win_rate 已经在 cache 里
        for team in all_teams:
            self._team_stats_cache[team]["upset_rate"] = self._compute_upset_rate(
                team, team_histories[team]
            )
            self._team_stats_cache[team]["big_match_win_rate"] = self._compute_big_match_win_rate(
                team, team_histories[team]
            )

            # days_since_last_match: 距离最新比赛日期的天数
            th = team_histories[team]
            if not th.empty:
                last_match_date = th["date"].max()
                days = max((ref_date - last_match_date).days, 0)
                self._team_stats_cache[team]["days_since_last"] = min(days, 60) / 60.0
            else:
                self._team_stats_cache[team]["days_since_last"] = 14 / 60.0

        print(f"  [OK] Precomputed stats for {len(all_teams)} teams")

    def _build_roster_lookup(self):
        """Build roster-per-match DataFrame from player_stats for stability calc."""
        source_data_dir = Path(self.config["data"]["data_dir"])
        if not source_data_dir.is_absolute():
            source_data_dir = (self.base_dir / source_data_dir).resolve()
        ps_path = source_data_dir / self.config["data"]["player_stats_pattern"]

        if not ps_path.exists():
            return None

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
        return roster_per_match.sort_values("match_date")

    def _compute_roster_stability(self, team, current_date, roster_per_match, n_recent=10):
        """Fraction of latest roster that appeared in prior N matches."""
        if roster_per_match is None:
            return 1.0
        team_matches = roster_per_match[
            (roster_per_match["team"] == team) & (roster_per_match["match_date"] < current_date)
        ].tail(n_recent)
        if len(team_matches) < 2:
            return 1.0
        latest_roster = team_matches.iloc[-1]["roster"]
        if not latest_roster:
            return 1.0
        overlaps = []
        for _, row in team_matches.iloc[:-1].iterrows():
            if row["roster"]:
                overlap = len(latest_roster & row["roster"]) / max(len(latest_roster), 1)
                overlaps.append(overlap)
        return float(np.mean(overlaps)) if overlaps else 1.0

    def _compute_upset_rate(self, team, team_history, n_recent=30):
        """Fraction of matches won when team was the underdog (by recent form)."""
        recent = team_history.tail(n_recent)
        if len(recent) < 5:
            return 0.5
        upsets = 0
        underdog_count = 0
        for _, row in recent.iterrows():
            if row["team1"] == team:
                # Use cached recent_win_rate if available, else 0.5
                my_rate = self._team_stats_cache.get(team, {}).get("recent_win_rate", 0.5)
                opp = row["team2"]
                opp_rate = self._team_stats_cache.get(opp, {}).get("recent_win_rate", 0.5)
                won = row["winner"] == 1
            else:
                my_rate = self._team_stats_cache.get(team, {}).get("recent_win_rate", 0.5)
                opp = row["team1"]
                opp_rate = self._team_stats_cache.get(opp, {}).get("recent_win_rate", 0.5)
                won = row["winner"] == 0
            if my_rate < opp_rate:
                underdog_count += 1
                if won:
                    upsets += 1
        if underdog_count < 3:
            return 0.5
        return float(upsets / underdog_count)

    def _compute_big_match_win_rate(self, team, team_history, n_recent=20):
        """Win rate in BO3/BO5 matches with time decay (proxy for LAN/important)."""
        big_matches = team_history[team_history["match_type"].isin(["BO3", "BO5"])].tail(n_recent)
        if len(big_matches) < 3:
            return 0.5
        days_ago = (self._inference_reference_date - big_matches["date"]).dt.days.clip(lower=0).to_numpy()
        time_weights = np.exp(-days_ago / self._time_decay_scale)
        won = (
            ((big_matches["team1"] == team) & (big_matches["winner"] == 1))
            | ((big_matches["team2"] == team) & (big_matches["winner"] == 0))
        ).to_numpy().astype(float)
        weighted_wins = float((won * time_weights).sum())
        weighted_total = float(time_weights.sum())
        return (weighted_wins + self.WIN_RATE_PRIOR_WINS) / (weighted_total + self.WIN_RATE_PRIOR_TOTAL)

    def build_features_for_map(self, team1, team2, map_name, series_state):
        self._precompute_team_stats()

        features = {}
        team1_stats = self._team_stats_cache.get(team1, {})
        team2_stats = self._team_stats_cache.get(team2, {})

        for feature_map in self.feature_map_names:
            features[f"team1_{feature_map}_win_rate"] = team1_stats.get(f"{feature_map}_win_rate", 0.5)
            features[f"team2_{feature_map}_win_rate"] = team2_stats.get(f"{feature_map}_win_rate", 0.5)

        picker = series_state.get("picker")
        features["team1_pick_advantage"] = int(picker == team1)
        features["team2_pick_advantage"] = int(picker == team2)

        # H2H 双向过滤(包含反向 orientation 的历史) + Beta(2, 4) 强先验。
        # 推理 map_data 已经过滤掉 augmented 行,所以必须 OR 两个方向才能补齐
        # 训练侧 raw pair_key "A|||B" 通过 augmented 行隐含包含的反向历史。
        # _smoothed_weighted_win_rate(h2h, team1) 内部会按 team1 视角算 won,所以方向正确。
        # 必须显式传 prior_wins=2, prior_total=4,否则会用类默认 WIN_RATE_PRIOR(1, 2)。
        h2h_key = f"{team1}__{team2}__{map_name}"
        if h2h_key not in self._h2h_cache:
            h2h = self.map_data[
                (
                    ((self.map_data["team1"] == team1) & (self.map_data["team2"] == team2))
                    | ((self.map_data["team1"] == team2) & (self.map_data["team2"] == team1))
                )
                & (self.map_data["map_name"] == map_name)
            ]
            self._h2h_cache[h2h_key] = self._smoothed_weighted_win_rate(
                h2h, team1, prior_wins=2.0, prior_total=4.0
            )
        features["team1_h2h_map_win_rate"] = self._h2h_cache[h2h_key]

        features["team1_recent_win_rate"] = team1_stats.get("recent_win_rate", 0.5)
        features["team2_recent_win_rate"] = team2_stats.get("recent_win_rate", 0.5)

        features["team1_avg_rating"] = team1_stats.get("avg_rating", self.default_rating_stats["avg_rating"])
        features["team1_rating_std"] = team1_stats.get("rating_std", self.default_rating_stats["rating_std"])
        features["team1_top_rating"] = team1_stats.get("top_rating", self.default_rating_stats["top_rating"])
        features["team2_avg_rating"] = team2_stats.get("avg_rating", self.default_rating_stats["avg_rating"])
        features["team2_rating_std"] = team2_stats.get("rating_std", self.default_rating_stats["rating_std"])
        features["team2_top_rating"] = team2_stats.get("top_rating", self.default_rating_stats["top_rating"])

        # 高级特征: 阵容稳定性 / 冷门率 / 大赛胜率
        features["team1_roster_stability"] = team1_stats.get("roster_stability", 1.0)
        features["team2_roster_stability"] = team2_stats.get("roster_stability", 1.0)
        features["team1_upset_rate"] = team1_stats.get("upset_rate", 0.5)
        features["team2_upset_rate"] = team2_stats.get("upset_rate", 0.5)
        features["team1_big_match_win_rate"] = team1_stats.get("big_match_win_rate", 0.5)
        features["team2_big_match_win_rate"] = team2_stats.get("big_match_win_rate", 0.5)

        previous_winner = series_state.get("previous_winner")
        features["won_previous_map"] = int(previous_winner == 1)
        features["lost_previous_map"] = int(previous_winner == 0)

        current_score = list(series_state.get("current_score", [0, 0]))
        match_type = series_state.get("match_type", "BO3")
        if match_type == "BO1":
            maps_to_win = 1
            bo_format_encoded = 1
        elif match_type == "BO3":
            maps_to_win = 2
            bo_format_encoded = 3
        else:
            maps_to_win = 3
            bo_format_encoded = 5

        features["series_leading"] = int(current_score[0] > current_score[1])
        features["series_trailing"] = int(current_score[0] < current_score[1])
        features["series_tied"] = int(current_score[0] == current_score[1])
        features["is_decider_int"] = int(match_type == "BO1" or current_score == [maps_to_win - 1, maps_to_win - 1])
        features["team1_win_streak"] = series_state.get("team1_win_streak", 0)

        features["bo_format_encoded"] = bo_format_encoded
        features["map_index_normalized"] = series_state.get("map_index", 0) / 5.0

        # is_lan: Major 线下赛默认 True
        features["is_lan_int"] = 1

        # days_since_last_match: 推理时用预计算值
        features["team1_days_since_last"] = team1_stats.get("days_since_last", 14 / 60.0)
        features["team2_days_since_last"] = team2_stats.get("days_since_last", 14 / 60.0)

        for feature_name in self.map_one_hot_features:
            features[feature_name] = int(feature_name == f"map_{map_name}")

        # ── diff 特征 (team1 - team2),与 feature_engineering.build_diff_features 完全一致 ──
        # 选手 diffs
        for stat in ("top_rating", "avg_rating", "rating_std"):
            features[f"{stat}_diff"] = float(features.get(f"team1_{stat}", 0.0)) - float(features.get(f"team2_{stat}", 0.0))
        # 状态 diff
        features["recent_winrate_diff"] = float(features.get("team1_recent_win_rate", 0.5)) - float(features.get("team2_recent_win_rate", 0.5))
        # 每张图 diff(注意:feature_map_names 是 self.df 训练时见过的图)
        for fm in self.feature_map_names:
            features[f"{fm}_winrate_diff"] = float(features.get(f"team1_{fm}_win_rate", 0.5)) - float(features.get(f"team2_{fm}_win_rate", 0.5))
        # 高级特征 diffs
        for stat in ("roster_stability", "upset_rate", "big_match_win_rate"):
            features[f"{stat}_diff"] = float(features.get(f"team1_{stat}", 0.5)) - float(features.get(f"team2_{stat}", 0.5))
        # 休息天数 diff
        features["days_since_last_diff"] = float(features.get("team1_days_since_last", 14/60.0)) - float(features.get("team2_days_since_last", 14/60.0))

        return pd.DataFrame([{feature_name: features.get(feature_name, 0.0) for feature_name in self.feature_names}])

    def predict_map_ml(self, team1, team2, map_name, series_state):
        feature_df = self.build_features_for_map(team1, team2, map_name, series_state)
        dmatrix = xgb.DMatrix(feature_df)
        raw_prob = float(self.xgb_model.predict(dmatrix)[0])
        if self.calibrator is not None:
            return float(self.calibrator.transform([raw_prob])[0])
        return raw_prob

    def predict_map_hybrid(self, team1, team2, map_name, series_state, elo_team1=1500, elo_team2=1500):
        ml_prob = self.predict_map_ml(team1, team2, map_name, series_state)
        return float(np.clip(ml_prob, 0.05, 0.95))

    @staticmethod
    def _calculate_team1_win_streak(map_results):
        streak = 0
        for result in reversed(map_results):
            if result["winner"] == 1:
                streak += 1
            else:
                break
        return streak

    def _simulate_series(self, team1, team2, match_type, maps=None, elo_team1=1500, elo_team2=1500, use_veto=True):
        maps_to_win = 2 if match_type == "BO3" else 3

        if use_veto and maps is None:
            if match_type == "BO5":
                tactical_result = self.tactical_analyzer.analyze_bo5_tactical_advantage(team1, team2)
            else:
                tactical_result = self.tactical_analyzer.analyze_bo3_tactical_advantage(team1, team2)
            veto_info = tactical_result["veto_result"]
            side_info = [dict(map_info) for map_info in veto_info["maps"]]
            map_list = [map_info["map"] for map_info in side_info]
        else:
            veto_info = None
            if maps is None:
                map_list = list(self.current_map_pool[: max(3, maps_to_win + 1)])
            else:
                map_list = list(maps)
            side_info = [
                {"map": map_name, "team1_starts_ct": np.random.random() < 0.5, "picker": "manual"}
                for map_name in map_list
            ]

        score = [0, 0]
        map_results = []

        for map_idx, map_name in enumerate(map_list):
            if score[0] == maps_to_win or score[1] == maps_to_win:
                break

            map_side_info = side_info[map_idx]
            team1_starts_ct = map_side_info.get("team1_starts_ct", np.random.random() < 0.5)
            series_state = {
                "current_score": score.copy(),
                "previous_winner": map_results[-1]["winner"] if map_results else None,
                "previous_was_comeback": map_results[-1].get("comeback", False) if map_results else False,
                "previous_was_close": map_results[-1].get("is_close", False) if map_results else False,
                "match_type": match_type,
                "map_index": map_idx,
                "team1_win_streak": self._calculate_team1_win_streak(map_results),
                "picker": map_side_info.get("picker", "unknown"),
            }

            prob_team1 = self.predict_map_hybrid(
                team1,
                team2,
                map_name,
                series_state,
                elo_team1=elo_team1,
                elo_team2=elo_team2,
            )
            side_advantage = self.side_analyzer.get_side_advantage(map_name, team1_starts_ct)
            prob_team1 = float(np.clip(prob_team1 + side_advantage, 0.05, 0.95))

            winner = 1 if np.random.random() < prob_team1 else 0
            if winner == 1:
                score[0] += 1
            else:
                score[1] += 1

            map_results.append(
                {
                    "map_name": map_name,
                    "winner": winner,
                    "prob_team1": prob_team1,
                    "score": score.copy(),
                    "team1_starts_ct": team1_starts_ct,
                    "side_advantage": side_advantage,
                    "picker": map_side_info.get("picker", "unknown"),
                }
            )

        final_winner = 1 if score[0] > score[1] else 0
        return final_winner, map_results, veto_info

    def simulate_bo3(self, team1, team2, maps=None, elo_team1=1500, elo_team2=1500, use_veto=True):
        return self._simulate_series(
            team1,
            team2,
            "BO3",
            maps=maps,
            elo_team1=elo_team1,
            elo_team2=elo_team2,
            use_veto=use_veto,
        )

    def simulate_bo1(self, team1, team2, map_name=None, elo_team1=1500, elo_team2=1500, use_veto=True):
        if use_veto and map_name is None:
            veto_info = self.veto_simulator.simulate_bo1_veto(team1, team2)
            map_info = dict(veto_info["maps"][0])
            selected_map = map_info["map"]
        else:
            veto_info = None
            selected_map = map_name or np.random.choice(self.current_map_pool)
            # 注:这里只用 team1_starts_ct / picker;map 名用 selected_map,不再放进 dict(避免 dead write)
            map_info = {
                "team1_starts_ct": np.random.random() < 0.5,
                "picker": "manual",
            }

        series_state = {
            "current_score": [0, 0],
            "previous_winner": None,
            "previous_was_comeback": False,
            "previous_was_close": False,
            "match_type": "BO1",
            "map_index": 0,
            "team1_win_streak": 0,
            "picker": map_info.get("picker", "unknown"),
        }

        team1_starts_ct = map_info.get("team1_starts_ct", np.random.random() < 0.5)
        prob_team1 = self.predict_map_hybrid(
            team1,
            team2,
            selected_map,
            series_state,
            elo_team1=elo_team1,
            elo_team2=elo_team2,
        )
        side_advantage = self.side_analyzer.get_side_advantage(selected_map, team1_starts_ct)
        prob_team1 = float(np.clip(prob_team1 + side_advantage, 0.05, 0.95))

        winner = 1 if np.random.random() < prob_team1 else 0
        map_result = {
            "map_name": selected_map,
            "winner": winner,
            "prob_team1": prob_team1,
            "score": [1, 0] if winner == 1 else [0, 1],
            "team1_starts_ct": team1_starts_ct,
            "side_advantage": side_advantage,
            "picker": map_info.get("picker", "unknown"),
        }
        return winner, map_result, veto_info

    def simulate_bo5(self, team1, team2, maps=None, elo_team1=1500, elo_team2=1500, use_veto=True):
        return self._simulate_series(
            team1,
            team2,
            "BO5",
            maps=maps,
            elo_team1=elo_team1,
            elo_team2=elo_team2,
            use_veto=use_veto,
        )

    # ==================================================================
    # GPU 向量化淘汰赛模拟
    # ==================================================================

    def simulate_playoffs_bracket_gpu(self, quarterfinals, num_simulations=500000):
        """GPU 向量化淘汰赛模拟：预计算概率 + 批量 tensor 采样。"""
        import torch
        import time

        device = self.device_mgr.get_torch_device()
        N = num_simulations
        t0 = time.time()

        print(f"\n  [GPU-Vec] 向量化模拟 {N:,} 次淘汰赛 (device={device})...")

        # ── 1. 收集队伍信息 ──
        qf_teams = [(qf["team1"], qf["team2"]) for qf in quarterfinals]
        semifinal_pairs_cfg = self.config.get("tournament", {}).get("semifinal_pairs", [[0, 3], [1, 2]])

        # ── 2. 枚举所有可能的对阵 ──
        all_matchups_bo3 = set()
        all_matchups_bo5 = set()

        # QF (BO3)
        for t1, t2 in qf_teams:
            all_matchups_bo3.add((t1, t2))

        # SF (BO3): left_qf_winner vs right_qf_winner
        for left_qf, right_qf in semifinal_pairs_cfg:
            for lt in qf_teams[left_qf]:
                for rt in qf_teams[right_qf]:
                    all_matchups_bo3.add((lt, rt))

        # Final (BO5): any SF0 winner vs any SF1 winner
        l0, r0 = semifinal_pairs_cfg[0]
        l1, r1 = semifinal_pairs_cfg[1]
        sf0_possible = set(qf_teams[l0]) | set(qf_teams[r0])
        sf1_possible = set(qf_teams[l1]) | set(qf_teams[r1])
        for t1 in sf0_possible:
            for t2 in sf1_possible:
                all_matchups_bo5.add((t1, t2))

        # ── 3. 预计算所有对阵的每张地图概率 ──
        total_matchups = len(all_matchups_bo3) + len(all_matchups_bo5)
        print(f"  [GPU-Vec] 预计算 {total_matchups} 个对阵概率...")

        # prob_cache[(t1,t2,mt)] = [(p_first, p_won_prev, p_lost_prev), ...] per map
        prob_cache = {}

        def _precompute_matchup(t1, t2, match_type):
            elo1 = self.team_elo.get(t1, 1500)
            elo2 = self.team_elo.get(t2, 1500)
            if match_type == "BO5":
                tactical = self.tactical_analyzer.analyze_bo5_tactical_advantage(t1, t2)
            else:
                tactical = self.tactical_analyzer.analyze_bo3_tactical_advantage(t1, t2)
            veto = tactical["veto_result"]
            side_info = [dict(m) for m in veto["maps"]]

            probs = []
            for mi, map_info in enumerate(side_info):
                map_name = map_info["map"]
                t1_ct = map_info.get("team1_starts_ct", True)
                side_adv = self.side_analyzer.get_side_advantage(map_name, t1_ct)
                map_probs = []
                for prev_winner in [None, 1, 0]:  # first / won_prev / lost_prev
                    ss = {
                        "current_score": [0, 0],
                        "previous_winner": prev_winner,
                        "previous_was_comeback": False,
                        "previous_was_close": False,
                        "match_type": match_type,
                        "map_index": mi,
                        "team1_win_streak": 0,
                        "picker": map_info.get("picker", "unknown"),
                    }
                    p = self.predict_map_hybrid(t1, t2, map_name, ss, elo1, elo2)
                    p = float(np.clip(p + side_adv, 0.05, 0.95))
                    map_probs.append(p)
                probs.append(tuple(map_probs))
            return probs

        for t1, t2 in all_matchups_bo3:
            prob_cache[(t1, t2, "BO3")] = _precompute_matchup(t1, t2, "BO3")
        for t1, t2 in all_matchups_bo5:
            prob_cache[(t1, t2, "BO5")] = _precompute_matchup(t1, t2, "BO5")

        precomp_time = time.time() - t0
        print(f"  [GPU-Vec] 预计算完成 ({precomp_time:.1f}s)，开始 tensor 模拟...")

        # ── 4. 向量化系列赛 helper ──
        _one = torch.ones(N, device=device, dtype=torch.long)
        _two = torch.full((N,), 2, device=device, dtype=torch.long)

        def _sim_series(prob_list, rand_cols, maps_to_win):
            """单一对阵的向量化 BO 模拟。"""
            t1_score = torch.zeros(N, device=device)
            t2_score = torch.zeros(N, device=device)
            prev = torch.zeros(N, device=device, dtype=torch.long)  # 0=first,1=won,2=lost
            for mi, (pf, pw, pl) in enumerate(prob_list):
                active = (t1_score < maps_to_win) & (t2_score < maps_to_win)
                if not active.any():
                    break
                pf_t = torch.full((N,), pf, device=device)
                pw_t = torch.full((N,), pw, device=device)
                pl_t = torch.full((N,), pl, device=device)
                p = torch.where(prev == 0, pf_t,
                        torch.where(prev == 1, pw_t, pl_t))
                t1_win = (rand_cols[:, mi] < p) & active
                t1_score += t1_win.float()
                t2_score += (~t1_win & active).float()
                prev = torch.where(active, torch.where(t1_win, _one, _two), prev)
            return t1_score >= maps_to_win

        def _sim_multi(matchup_keys, masks, rand_cols, maps_to_win):
            """变对阵（SF/Final）的向量化模拟。"""
            max_maps = max(len(prob_cache[k]) for k in matchup_keys)
            t1_score = torch.zeros(N, device=device)
            t2_score = torch.zeros(N, device=device)
            prev = torch.zeros(N, device=device, dtype=torch.long)
            for mi in range(max_maps):
                active = (t1_score < maps_to_win) & (t2_score < maps_to_win)
                if not active.any():
                    break
                p = torch.zeros(N, device=device)
                for key, mask in zip(matchup_keys, masks):
                    if mi >= len(prob_cache[key]):
                        continue
                    pf, pw, pl = prob_cache[key][mi]
                    pf_t = torch.full((N,), pf, device=device)
                    pw_t = torch.full((N,), pw, device=device)
                    pl_t = torch.full((N,), pl, device=device)
                    p_val = torch.where(prev == 0, pf_t,
                                torch.where(prev == 1, pw_t, pl_t))
                    p = torch.where(mask, p_val, p)
                t1_win = (rand_cols[:, mi] < p) & active
                t1_score += t1_win.float()
                t2_score += (~t1_win & active).float()
                prev = torch.where(active, torch.where(t1_win, _one, _two), prev)
            return t1_score >= maps_to_win

        # ── 5. 生成随机数 (QF:12 + SF:6 + Final:5 = 23 列) ──
        rand = torch.rand(N, 25, device=device)
        col = 0

        # ── 6. 四分之一决赛 (QF - BO3) ──
        qf_t1_wins = []
        for qf_idx in range(len(qf_teams)):
            t1, t2 = qf_teams[qf_idx]
            key = (t1, t2, "BO3")
            n_maps = len(prob_cache[key])
            result = _sim_series(prob_cache[key], rand[:, col:col + n_maps], 2)
            qf_t1_wins.append(result)
            col += n_maps

        # ── 7. 半决赛 (SF - BO3) ──
        sf_left_wins = []
        for sf_idx, (left_qf, right_qf) in enumerate(semifinal_pairs_cfg):
            lt1, lt2 = qf_teams[left_qf]
            rt1, rt2 = qf_teams[right_qf]
            left_t1_won = qf_t1_wins[left_qf]
            right_t1_won = qf_t1_wins[right_qf]
            variants = [
                ((lt1, rt1, "BO3"), left_t1_won & right_t1_won),
                ((lt1, rt2, "BO3"), left_t1_won & ~right_t1_won),
                ((lt2, rt1, "BO3"), ~left_t1_won & right_t1_won),
                ((lt2, rt2, "BO3"), ~left_t1_won & ~right_t1_won),
            ]
            keys = [v[0] for v in variants]
            masks = [v[1] for v in variants]
            max_maps = max(len(prob_cache[k]) for k in keys)
            result = _sim_multi(keys, masks, rand[:, col:col + max_maps], 2)
            sf_left_wins.append(result)
            col += max_maps

        # ── 8. 决赛 (Final - BO5) ──
        sf0_finalists = [
            (qf_teams[l0][0], qf_t1_wins[l0] & sf_left_wins[0]),
            (qf_teams[l0][1], ~qf_t1_wins[l0] & sf_left_wins[0]),
            (qf_teams[r0][0], qf_t1_wins[r0] & ~sf_left_wins[0]),
            (qf_teams[r0][1], ~qf_t1_wins[r0] & ~sf_left_wins[0]),
        ]
        sf1_finalists = [
            (qf_teams[l1][0], qf_t1_wins[l1] & sf_left_wins[1]),
            (qf_teams[l1][1], ~qf_t1_wins[l1] & sf_left_wins[1]),
            (qf_teams[r1][0], qf_t1_wins[r1] & ~sf_left_wins[1]),
            (qf_teams[r1][1], ~qf_t1_wins[r1] & ~sf_left_wins[1]),
        ]
        final_keys = []
        final_masks = []
        for sf0_team, sf0_mask in sf0_finalists:
            for sf1_team, sf1_mask in sf1_finalists:
                combined = sf0_mask & sf1_mask
                if combined.any():
                    final_keys.append((sf0_team, sf1_team, "BO5"))
                    final_masks.append(combined)

        max_final_maps = max(len(prob_cache[k]) for k in final_keys)
        final_t1_wins = _sim_multi(final_keys, final_masks, rand[:, col:col + max_final_maps], 3)

        # ── 9. 统计结果 ──
        qf_wins = {}
        qf_matchup_results = {}
        for qf_idx in range(len(qf_teams)):
            t1, t2 = qf_teams[qf_idx]
            t1_count = int(qf_t1_wins[qf_idx].sum().item())
            qf_wins[t1] = t1_count
            qf_wins[t2] = N - t1_count
            qf_matchup_results[qf_idx] = {t1: t1_count / N, t2: (N - t1_count) / N}

        # 为每个队伍追踪晋级路径
        sf_appearances = {}
        sf_wins_dict = {}
        final_appearances = {}
        champion_count = {}

        for qf_idx in range(len(qf_teams)):
            t1, t2 = qf_teams[qf_idx]
            # 找到此 QF 对应的 SF
            sf_idx = is_left = None
            for si, (lq, rq) in enumerate(semifinal_pairs_cfg):
                if qf_idx == lq:
                    sf_idx, is_left = si, True
                    break
                elif qf_idx == rq:
                    sf_idx, is_left = si, False
                    break
            is_sf0 = (sf_idx == 0)

            for team, is_t1 in [(t1, True), (t2, False)]:
                won_qf = qf_t1_wins[qf_idx] if is_t1 else ~qf_t1_wins[qf_idx]
                won_sf = sf_left_wins[sf_idx] if is_left else ~sf_left_wins[sf_idx]
                won_final = final_t1_wins if is_sf0 else ~final_t1_wins

                sf_app = won_qf
                sf_win = won_qf & won_sf
                final_app = sf_win
                champ = sf_win & won_final

                sf_appearances[team] = int(sf_app.sum().item())
                sf_wins_dict[team] = int(sf_win.sum().item())
                final_appearances[team] = int(final_app.sum().item())
                champion_count[team] = int(champ.sum().item())

        # SF 对阵频率
        sf_matchup_count = defaultdict(int)
        for sf_idx, (left_qf, right_qf) in enumerate(semifinal_pairs_cfg):
            lt1, lt2 = qf_teams[left_qf]
            rt1, rt2 = qf_teams[right_qf]
            for lteam, lmask in [(lt1, qf_t1_wins[left_qf]), (lt2, ~qf_t1_wins[left_qf])]:
                for rteam, rmask in [(rt1, qf_t1_wins[right_qf]), (rt2, ~qf_t1_wins[right_qf])]:
                    c = int((lmask & rmask).sum().item())
                    if c > 0:
                        sf_matchup_count[tuple(sorted([lteam, rteam]))] += c

        # Final 对阵频率
        final_matchup_count = defaultdict(int)
        for key, mask in zip(final_keys, final_masks):
            c = int(mask.sum().item())
            if c > 0:
                final_matchup_count[tuple(sorted([key[0], key[1]]))] += c

        elapsed = time.time() - t0
        print(f"  [GPU-Vec] 完成! 耗时 {elapsed:.2f}s (预计算 {precomp_time:.1f}s + 模拟 {elapsed - precomp_time:.1f}s)")

        return {
            "num_simulations": N,
            "quarterfinal_win_prob": {t: c / N for t, c in qf_wins.items()},
            "semifinal_appearance_prob": {t: c / N for t, c in sf_appearances.items()},
            "semifinal_win_prob": {t: c / N for t, c in sf_wins_dict.items()},
            "final_appearance_prob": {t: c / N for t, c in final_appearances.items()},
            "champion_prob": {t: c / N for t, c in champion_count.items()},
            "qf_matchup_results": qf_matchup_results,
            "sf_matchup_frequency": {
                f"{m[0]} vs {m[1]}": c / N
                for m, c in sorted(sf_matchup_count.items(), key=lambda x: x[1], reverse=True)[:10]
            },
            "final_matchup_frequency": {
                f"{m[0]} vs {m[1]}": c / N
                for m, c in sorted(final_matchup_count.items(), key=lambda x: x[1], reverse=True)[:10]
            },
        }

    def simulate_playoffs_bracket(self, quarterfinals, num_simulations=50000):
        semifinal_pairs = self.config.get("tournament", {}).get("semifinal_pairs", [[0, 3], [1, 2]])

        qf_wins = {}
        sf_appearances = {}
        sf_wins = {}
        final_appearances = {}
        champion_count = {}

        all_teams = []
        for qf in quarterfinals:
            all_teams.extend([qf["team1"], qf["team2"]])
        for team in all_teams:
            qf_wins[team] = 0
            sf_appearances[team] = 0
            sf_wins[team] = 0
            final_appearances[team] = 0
            champion_count[team] = 0

        qf_matchup_wins = {
            idx: {quarterfinals[idx]["team1"]: 0, quarterfinals[idx]["team2"]: 0}
            for idx in range(len(quarterfinals))
        }
        sf_matchup_count = defaultdict(int)
        final_matchup_count = defaultdict(int)

        for sim_idx in range(num_simulations):
            if (sim_idx + 1) % 10000 == 0:
                print(f"  Progress: {sim_idx + 1}/{num_simulations}")

            semifinalists = []
            for qf_idx, qf in enumerate(quarterfinals):
                winner, _, _ = self.simulate_bo3(
                    qf["team1"],
                    qf["team2"],
                    elo_team1=qf["elo1"],
                    elo_team2=qf["elo2"],
                    use_veto=True,
                )
                winner_team = qf["team1"] if winner == 1 else qf["team2"]
                winner_elo = qf["elo1"] if winner == 1 else qf["elo2"]

                qf_wins[winner_team] += 1
                qf_matchup_wins[qf_idx][winner_team] += 1
                semifinalists.append({"team": winner_team, "elo": winner_elo})

            for semifinalist in semifinalists:
                sf_appearances[semifinalist["team"]] += 1

            finalists = []
            for left_idx, right_idx in semifinal_pairs:
                left_team = semifinalists[left_idx]["team"]
                right_team = semifinalists[right_idx]["team"]
                matchup_key = tuple(sorted([left_team, right_team]))
                sf_matchup_count[matchup_key] += 1

                winner, _, _ = self.simulate_bo3(
                    left_team,
                    right_team,
                    elo_team1=semifinalists[left_idx]["elo"],
                    elo_team2=semifinalists[right_idx]["elo"],
                    use_veto=True,
                )
                winner_team = left_team if winner == 1 else right_team
                winner_elo = semifinalists[left_idx]["elo"] if winner == 1 else semifinalists[right_idx]["elo"]

                sf_wins[winner_team] += 1
                finalists.append({"team": winner_team, "elo": winner_elo})

            for finalist in finalists:
                final_appearances[finalist["team"]] += 1

            final_key = tuple(sorted([finalists[0]["team"], finalists[1]["team"]]))
            final_matchup_count[final_key] += 1

            winner, _, _ = self.simulate_bo5(
                finalists[0]["team"],
                finalists[1]["team"],
                elo_team1=finalists[0]["elo"],
                elo_team2=finalists[1]["elo"],
                use_veto=True,
            )
            champion = finalists[0]["team"] if winner == 1 else finalists[1]["team"]
            champion_count[champion] += 1

        return {
            "num_simulations": num_simulations,
            "quarterfinal_win_prob": {team: count / num_simulations for team, count in qf_wins.items()},
            "semifinal_appearance_prob": {
                team: count / num_simulations for team, count in sf_appearances.items()
            },
            "semifinal_win_prob": {team: count / num_simulations for team, count in sf_wins.items()},
            "final_appearance_prob": {
                team: count / num_simulations for team, count in final_appearances.items()
            },
            "champion_prob": {team: count / num_simulations for team, count in champion_count.items()},
            "qf_matchup_results": {
                idx: {team: count / num_simulations for team, count in wins.items()}
                for idx, wins in qf_matchup_wins.items()
            },
            "sf_matchup_frequency": {
                f"{matchup[0]} vs {matchup[1]}": count / num_simulations
                for matchup, count in sorted(sf_matchup_count.items(), key=lambda item: item[1], reverse=True)[:10]
            },
            "final_matchup_frequency": {
                f"{matchup[0]} vs {matchup[1]}": count / num_simulations
                for matchup, count in sorted(final_matchup_count.items(), key=lambda item: item[1], reverse=True)[:10]
            },
        }

    def run_prediction(self):
        tournament_cfg = self.config.get("tournament", {})
        quarterfinal_pairs = tournament_cfg.get("quarterfinals", [])
        if not quarterfinal_pairs:
            raise ValueError("config.yaml 中还没有 quarterfinals，对阵出来后再运行完整预测。")

        quarterfinals = []
        for team1, team2 in quarterfinal_pairs:
            quarterfinals.append(
                {
                    "team1": team1,
                    "team2": team2,
                    "elo1": self.team_elo.get(team1, 1500),
                    "elo2": self.team_elo.get(team2, 1500),
                }
            )

        simulation_cfg = self.config.get("simulation", {})
        num_simulations = simulation_cfg.get("playoff_num_simulations", 0)
        if _is_auto_count(num_simulations):
            num_simulations = simulation_cfg.get("num_simulations", 0)

        # GPU 向量化 vs CPU 逐次模拟
        if self.device_mgr.cuda_available:
            if _is_auto_count(num_simulations):
                num_simulations = self.device_mgr.recommend_playoff_sim_count()
            else:
                num_simulations = int(num_simulations)
            print(f"  Playoff 模拟次数: {num_simulations:,}")
            results = self.simulate_playoffs_bracket_gpu(quarterfinals, num_simulations=num_simulations)
        else:
            if _is_auto_count(num_simulations):
                num_simulations = min(self.device_mgr.recommend_sim_count(), 50000)
            else:
                num_simulations = int(num_simulations)
            print(f"  Playoff 模拟次数: {num_simulations:,} (CPU)")
            results = self.simulate_playoffs_bracket(quarterfinals, num_simulations=num_simulations)

        champion_prob = results["champion_prob"]
        sorted_champions = sorted(champion_prob.items(), key=lambda item: item[1], reverse=True)
        predicted_champion = sorted_champions[0][0]

        output = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_simulations": num_simulations,
            "quarterfinals": quarterfinals,
            "predictions": results,
            "predicted_champion": predicted_champion,
        }

        output_path = self.base_dir / "playoffs_ml_prediction.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)

        print(f"[OK] Prediction saved to {output_path.name}")
        print("Top championship probabilities:")
        for team, probability in sorted_champions[:8]:
            print(f"  {team:<20} {probability * 100:6.2f}%")


if __name__ == "__main__":
    predictor = HybridPlayoffPredictor()
    predictor.run_prediction()
