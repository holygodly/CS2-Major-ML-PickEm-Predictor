"""
Map-level dataset preparation for playoff ML models.

This script reads the latest HLTV crawler outputs and converts them into a
map-level training dataset used by feature engineering and model training.
"""

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class MapLevelDataPreparation:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.matches_df = None
        self.match_details = None
        self.player_stats = None
        self.map_level_data = []

    def load_data(self):
        """Load all source data files."""
        print("\n[1/6] 加载数据...")

        data_dir = Path(self.config["data"]["data_dir"])

        match_files = list(data_dir.glob(self.config["data"]["match_details_pattern"]))
        if not match_files:
            raise FileNotFoundError("未找到 match_details_lite.json")

        self.match_details = {}
        for match_file in match_files:
            with open(match_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.match_details.update(data)
            print(f"  ✓ 加载 {match_file.name}: {len(data)} 场")
        print(f"  ✓ 总比赛详情: {len(self.match_details)} 场")

        results_files = list(data_dir.glob(self.config["data"].get("results_pattern", "results_*.csv")))
        if results_files:
            dfs = [pd.read_csv(path) for path in results_files]
            self.matches_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            print(f"  ✓ 比赛列表: {len(self.matches_df)} 场")
        else:
            self.matches_df = None
            print("  ! 未找到比赛列表文件，跳过")

        player_files = list(data_dir.glob(self.config["data"]["player_stats_pattern"]))
        if not player_files:
            raise FileNotFoundError("未找到 player_stats.csv")

        dfs = []
        for player_file in player_files:
            df = pd.read_csv(player_file, on_bad_lines="skip")
            dfs.append(df)
            print(f"  ✓ 加载 {player_file.name}: {len(df)} 条")

        self.player_stats = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        # 按 data_end_date 过滤选手数据，防止未来数据泄露。
        # 日期无法确认的行不能用于回测；prior stage 会由 runner 先补日期。
        data_end_date = self.config.get("data", {}).get("data_end_date")
        if data_end_date:
            self.player_stats["match_date"] = pd.to_datetime(
                self.player_stats.get("match_date", ""), errors="coerce"
            )
            url_dates = {}
            for url, item in (self.match_details or {}).items():
                parsed = pd.to_datetime(item.get("date"), errors="coerce")
                if pd.notna(parsed):
                    url_dates[url] = parsed
            if "match_url" in self.player_stats.columns and url_dates:
                missing_dates = self.player_stats["match_date"].isna()
                self.player_stats.loc[missing_dates, "match_date"] = (
                    self.player_stats.loc[missing_dates, "match_url"].map(url_dates)
                )
            before = len(self.player_stats)
            self.player_stats = self.player_stats[
                self.player_stats["match_date"].notna()
                & (self.player_stats["match_date"] <= pd.to_datetime(data_end_date))
            ].reset_index(drop=True)
            print(f"  ✓ 总选手数据: {len(self.player_stats)} 条（排除截止日后 {before - len(self.player_stats)} 条）")
        else:
            print(f"  ✓ 总选手数据: {len(self.player_stats)} 条")

    def parse_half_score(self, half_score_str):
        """Parse '(3:9;1:4)' and return the first-half score tuple."""
        if not half_score_str:
            return None, None

        cleaned = str(half_score_str).strip()
        if cleaned in {"unknown", "---"}:
            return None, None

        try:
            cleaned = cleaned.strip("()")
            first_half = cleaned.split(";")[0].split(":")
            if len(first_half) == 2:
                return int(first_half[0]), int(first_half[1])
        except Exception:
            return None, None

        return None, None

    def parse_map_score(self, score_str):
        """Parse map scores like '9-13' or '9 - 13'."""
        if not score_str:
            return None, None

        cleaned = str(score_str).strip()
        if cleaned in {"---", "- - -", "unknown"}:
            return None, None

        cleaned = cleaned.replace(" ", "")
        parts = cleaned.split("-")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            return None, None

        return int(parts[0]), int(parts[1])

    def get_match_maps(self, match_data):
        """Support both legacy top-level maps and newer details.maps layout."""
        details = match_data.get("details") or {}
        return match_data.get("maps") or details.get("maps") or []

    def extract_map_samples(self):
        """Extract one training sample per played map."""
        print("\n[2/6] 提取地图级别样本...")

        valid_maps = {
            "Ancient",
            "Anubis",
            "Dust2",
            "Inferno",
            "Mirage",
            "Nuke",
            "Overpass",
            "Train",
        }

        # 数据截止日期：排除之后的所有比赛
        data_end_date = self.config.get("data", {}).get("data_end_date")
        if data_end_date:
            data_end_date = pd.to_datetime(data_end_date)
            print(f"  数据截止日期: {data_end_date.date()}（之后的比赛全部排除）")

        # 过滤无选手数据的比赛（表演赛/弃权/forfeit）
        ps_urls = set(self.player_stats['match_url'].dropna().unique()) if self.player_stats is not None else set()
        excluded_no_stats = 0

        map_count = 0
        skipped_count = 0
        excluded_by_date = 0

        for match_url, match_data in self.match_details.items():
            if ps_urls and match_url not in ps_urls:
                excluded_no_stats += 1
                continue
            try:
                match_type = match_data["type"]
                team1 = match_data["team1"]
                team2 = match_data["team2"]
                match_date = match_data["date"]
                details = match_data.get("details") or {}

                try:
                    parsed_date = pd.to_datetime(match_date)
                except Exception:
                    print(f"  警告: 日期解析失败 '{match_date}'")
                    parsed_date = datetime.now()

                # 排除截止日期之后的比赛
                if data_end_date and parsed_date > data_end_date:
                    excluded_by_date += 1
                    continue

                maps_data = self.get_match_maps(match_data)
                current_score = [0, 0]
                previous_map_winner = -1

                for map_idx, map_info in enumerate(maps_data):
                    map_name = map_info.get("map_name")
                    if map_name not in valid_maps:
                        continue

                    score1, score2 = self.parse_map_score(map_info.get("score", ""))
                    if score1 is None or score2 is None:
                        skipped_count += 1
                        continue

                    winner = 1 if score1 > score2 else 0
                    half1_team1, half1_team2 = self.parse_half_score(map_info.get("half_score", ""))

                    comeback = 0
                    if half1_team1 is not None and half1_team2 is not None:
                        if winner == 1 and half1_team1 < half1_team2:
                            comeback = 1
                        elif winner == 0 and half1_team2 < half1_team1:
                            comeback = 1

                    series_score_team1 = current_score[0]
                    series_score_team2 = current_score[1]
                    is_decider = (
                        match_type == "BO1"
                        or (
                            current_score[0] == current_score[1]
                            and (
                                (match_type == "BO3" and map_idx == 2)
                                or (match_type == "BO5" and map_idx == 4)
                            )
                        )
                    )
                    is_close = abs(score1 - score2) <= 3 or max(score1, score2) >= 16

                    sample = {
                        "match_url": match_url,
                        "date": parsed_date,
                        "team1": team1,
                        "team2": team2,
                        "map_name": map_name,
                        "map_index": map_idx,
                        "match_type": match_type,
                        "picker": map_info.get("picker", "Unknown"),
                        "score_team1": score1,
                        "score_team2": score2,
                        "winner": winner,
                        "series_score_team1": series_score_team1,
                        "series_score_team2": series_score_team2,
                        "previous_map_winner": previous_map_winner,
                        "is_decider": is_decider,
                        "is_close": is_close,
                        "comeback": comeback,
                        "half_score": map_info.get("half_score", ""),
                        "half1_t_ct": map_info.get("half1_t_ct", ""),
                        "half2_t_ct": map_info.get("half2_t_ct", ""),
                        "overtime_t_ct": map_info.get("overtime_t_ct", ""),
                        "team1_half1_side": map_info.get("team1_half1_side", "unknown"),
                        "team2_half1_side": map_info.get("team2_half1_side", "unknown"),
                        "team1_half2_side": map_info.get("team1_half2_side", "unknown"),
                        "team2_half2_side": map_info.get("team2_half2_side", "unknown"),
                        "team1_rounds": map_info.get("team1_rounds", score1),
                        "team2_rounds": map_info.get("team2_rounds", score2),
                        "is_lan": details.get("is_lan"),
                        "team1_vrs_points_raw": (details.get("vrs_data") or {}).get("team1_vrs_points"),
                        "team2_vrs_points_raw": (details.get("vrs_data") or {}).get("team2_vrs_points"),
                        "team1_vrs_rank_raw": (details.get("vrs_data") or {}).get("team1_vrs_rank"),
                        "team2_vrs_rank_raw": (details.get("vrs_data") or {}).get("team2_vrs_rank"),
                        "team1_vrs_change_raw": (details.get("vrs_data") or {}).get("team1_vrs_change"),
                        "team2_vrs_change_raw": (details.get("vrs_data") or {}).get("team2_vrs_change"),
                        "team1_world_rank": details.get("team1_world_rank"),
                        "team2_world_rank": details.get("team2_world_rank"),
                        "event_name": details.get("event_name"),
                        "stage": details.get("stage"),
                        "team1_lineup": details.get("team1_lineup"),
                        "team2_lineup": details.get("team2_lineup"),
                        "is_augmented": 0,
                    }

                    self.map_level_data.append(sample)
                    map_count += 1

                    if winner == 1:
                        current_score[0] += 1
                    else:
                        current_score[1] += 1
                    previous_map_winner = winner

            except Exception as e:
                print(f"  错误处理比赛 {match_url}: {e}")

        print(f"  ✓ 提取了 {map_count} 张地图样本")
        if excluded_no_stats > 0:
            print(f"  排除了 {excluded_no_stats} 场无选手数据的比赛（表演赛/弃权）")
        if excluded_by_date > 0:
            print(f"  排除了 {excluded_by_date} 场截止日期之后的比赛")
        if skipped_count > 0:
            print(f"  跳过了 {skipped_count} 张无效地图")
        return map_count

    def add_player_aggregations(self):
        """Add historical team-level aggregates from player stats."""
        print("\n[3/6] 添加选手聚合特征...")

        if not self.map_level_data:
            raise ValueError("没有地图级样本，无法继续计算选手聚合特征。")

        player_stats_sorted = self.player_stats.copy()
        for col in ["rating", "adr"]:
            if col in player_stats_sorted.columns:
                player_stats_sorted[col] = pd.to_numeric(player_stats_sorted[col], errors="coerce")
        if "kast" in player_stats_sorted.columns:
            player_stats_sorted["kast"] = (
                player_stats_sorted["kast"].astype(str).str.replace("%", "", regex=False)
            )
            player_stats_sorted["kast"] = pd.to_numeric(player_stats_sorted["kast"], errors="coerce")

        if "match_date" not in player_stats_sorted.columns:
            url_to_date = {row["match_url"]: row["date"] for row in self.map_level_data}
            player_stats_sorted["match_date"] = player_stats_sorted["match_url"].map(url_to_date)

        player_stats_sorted["match_date"] = pd.to_datetime(player_stats_sorted["match_date"], errors="coerce")

        both_side_stats = player_stats_sorted[player_stats_sorted["side"] == "Both"].copy()
        global_avg_rating = both_side_stats["rating"].mean() if len(both_side_stats) > 0 else 1.0
        global_std_rating = both_side_stats["rating"].std() if len(both_side_stats) > 0 else 0.15
        global_avg_adr = both_side_stats["adr"].mean() if len(both_side_stats) > 0 else 70.0
        global_avg_kast = both_side_stats["kast"].mean() if len(both_side_stats) > 0 else 70.0

        print(
            f"  全局默认值: rating={global_avg_rating:.3f}, "
            f"adr={global_avg_adr:.1f}, kast={global_avg_kast:.1f}"
        )

        match_team_stats = (
            both_side_stats.groupby(["match_url", "team"])
            .agg({"rating": "mean", "adr": "mean", "kast": "mean"})
            .reset_index()
        )
        match_team_stats = match_team_stats.merge(
            player_stats_sorted[["match_url", "match_date"]].drop_duplicates(),
            on="match_url",
            how="left",
        )

        for i, sample in enumerate(self.map_level_data):
            current_date = sample["date"]
            team1 = sample["team1"]
            team2 = sample["team2"]

            team1_history = match_team_stats[
                (match_team_stats["team"] == team1) & (match_team_stats["match_date"] < current_date)
            ].sort_values("match_date")
            if len(team1_history) >= 3:
                recent_matches = team1_history.tail(20)
                self.map_level_data[i]["team1_avg_rating"] = recent_matches["rating"].mean()
                self.map_level_data[i]["team1_rating_std"] = recent_matches["rating"].std()
                self.map_level_data[i]["team1_top_rating"] = recent_matches["rating"].max()
                self.map_level_data[i]["team1_avg_adr"] = recent_matches["adr"].mean()
                self.map_level_data[i]["team1_avg_kast"] = recent_matches["kast"].mean()
            else:
                self.map_level_data[i]["team1_avg_rating"] = global_avg_rating
                self.map_level_data[i]["team1_rating_std"] = global_std_rating
                self.map_level_data[i]["team1_top_rating"] = global_avg_rating
                self.map_level_data[i]["team1_avg_adr"] = global_avg_adr
                self.map_level_data[i]["team1_avg_kast"] = global_avg_kast

            team2_history = match_team_stats[
                (match_team_stats["team"] == team2) & (match_team_stats["match_date"] < current_date)
            ].sort_values("match_date")
            if len(team2_history) >= 3:
                recent_matches = team2_history.tail(20)
                self.map_level_data[i]["team2_avg_rating"] = recent_matches["rating"].mean()
                self.map_level_data[i]["team2_rating_std"] = recent_matches["rating"].std()
                self.map_level_data[i]["team2_top_rating"] = recent_matches["rating"].max()
                self.map_level_data[i]["team2_avg_adr"] = recent_matches["adr"].mean()
                self.map_level_data[i]["team2_avg_kast"] = recent_matches["kast"].mean()
            else:
                self.map_level_data[i]["team2_avg_rating"] = global_avg_rating
                self.map_level_data[i]["team2_rating_std"] = global_std_rating
                self.map_level_data[i]["team2_top_rating"] = global_avg_rating
                self.map_level_data[i]["team2_avg_adr"] = global_avg_adr
                self.map_level_data[i]["team2_avg_kast"] = global_avg_kast

            if (i + 1) % 100 == 0:
                print(f"  处理进度: {i + 1}/{len(self.map_level_data)}")

        print("  ✓ 已添加选手聚合特征")

    @staticmethod
    def _swap_pair_string(value, separator):
        """Swap the two numbers/tokens in a string like 'A SEP B' to 'B SEP A'.

        Returns the original value if it cannot be safely parsed (e.g. 'unknown',
        empty string, or unexpected format). Used by data augmentation so that
        team1/team2 swaps stay consistent across half-score derived fields.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or text.lower() == "unknown":
            return value
        if separator not in text:
            return value
        parts = text.split(separator)
        if len(parts) != 2:
            return value
        return f"{parts[1].strip()}{separator}{parts[0].strip()}"

    @staticmethod
    def _swap_half_score_string(value):
        """Swap team1/team2 numbers in a string like '(3:9;1:4)' or '(3:9;1:4;2:0)'.

        Handles either parenthesized or bare strings, with 2 or 3 halves
        (regulation + optional overtime). Returns the value unchanged when it
        cannot be parsed.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or text.lower() == "unknown":
            return value
        has_paren = text.startswith("(") and text.endswith(")")
        inner = text[1:-1] if has_paren else text
        swapped = []
        for chunk in inner.split(";"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                swapped.append(chunk)
                continue
            a, b = chunk.split(":", 1)
            swapped.append(f"{b.strip()}:{a.strip()}")
        result = ";".join(swapped)
        return f"({result})" if has_paren else result

    def augment_with_flipped_samples(self):
        """Double the dataset by swapping team1/team2 for every sample.

        This eliminates positional bias: the model cannot learn that the team
        labelled 'team1' in the raw data tends to win more often.
        The is_augmented flag lets downstream steps (e.g. win-streak lookup)
        correctly isolate each orientation's within-series history.

        IMPORTANT: every team1/team2-dependent field must be swapped, including
        half-score derived strings (half1_t_ct, half2_t_ct, overtime_t_ct,
        half_score). Otherwise downstream consumers like MapSideAnalyzer would
        read inconsistent T/CT rounds for flipped rows and the side-bias
        estimates collapse toward 50%.
        """
        print("\n[+] 对称性数据增强（team1/team2 互换）...")
        original_count = len(self.map_level_data)
        flipped = []
        for sample in self.map_level_data:
            f = sample.copy()
            f["team1"] = sample["team2"]
            f["team2"] = sample["team1"]
            f["winner"] = 1 - sample["winner"]
            f["series_score_team1"] = sample["series_score_team2"]
            f["series_score_team2"] = sample["series_score_team1"]
            prev = sample["previous_map_winner"]
            if prev == 1:
                f["previous_map_winner"] = 0
            elif prev == 0:
                f["previous_map_winner"] = 1
            f["score_team1"] = sample["score_team2"]
            f["score_team2"] = sample["score_team1"]
            f["team1_rounds"] = sample["team2_rounds"]
            f["team2_rounds"] = sample["team1_rounds"]
            f["team1_half1_side"] = sample["team2_half1_side"]
            f["team2_half1_side"] = sample["team1_half1_side"]
            f["team1_half2_side"] = sample["team2_half2_side"]
            f["team2_half2_side"] = sample["team1_half2_side"]
            # half_score 形如 "(7:5;2:8)" 或 "(7:5;2:8;3:2)"，
            # half{1,2}_t_ct / overtime_t_ct 形如 "7 - 5"。
            # 这些字段都是 "team1_rounds : team2_rounds" 的语义，
            # team1/team2 互换后必须同步翻转，否则 MapSideAnalyzer 会
            # 在 augmented 行上读到与 team{1,2}_half_side 不一致的回合数。
            if "half_score" in sample:
                f["half_score"] = self._swap_half_score_string(sample["half_score"])
            for key in ("half1_t_ct", "half2_t_ct", "overtime_t_ct"):
                if key in sample:
                    f[key] = self._swap_pair_string(sample[key], " - ")
            for stat in ["avg_rating", "rating_std", "top_rating", "avg_adr", "avg_kast"]:
                k1, k2 = f"team1_{stat}", f"team2_{stat}"
                if k1 in sample and k2 in sample:
                    f[k1] = sample[k2]
                    f[k2] = sample[k1]
            if "team1_recent_win_rate" in sample and "team2_recent_win_rate" in sample:
                f["team1_recent_win_rate"] = sample["team2_recent_win_rate"]
                f["team2_recent_win_rate"] = sample["team1_recent_win_rate"]
            for vrs_key in ["vrs_points_raw", "vrs_rank_raw", "vrs_change_raw"]:
                k1, k2 = f"team1_{vrs_key}", f"team2_{vrs_key}"
                if k1 in sample and k2 in sample:
                    f[k1] = sample[k2]
                    f[k2] = sample[k1]
            f["is_augmented"] = 1
            flipped.append(f)
        self.map_level_data.extend(flipped)
        print(f"  ✓ 增强后样本数: {original_count} → {len(self.map_level_data)}")

    def calculate_rolling_stats(self):
        """Calculate rolling recent win rate features."""
        print("\n[4/6] 计算滚动统计...")

        df = (
            pd.DataFrame(self.map_level_data)
            .sort_values(["date", "match_url", "map_index"])
            .reset_index(drop=True)
        )
        unique_teams = set(df["team1"].unique()) | set(df["team2"].unique())

        for team in unique_teams:
            team_matches = df[(df["team1"] == team) | (df["team2"] == team)].copy()
            team_matches["team_won"] = (
                ((team_matches["team1"] == team) & (team_matches["winner"] == 1))
                | ((team_matches["team2"] == team) & (team_matches["winner"] == 0))
            ).astype(int)
            team_matches["rolling_win_rate"] = (
                team_matches["team_won"]
                .shift(1)
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(0.5)
            )

            for idx, row in team_matches.iterrows():
                if row["team1"] == team:
                    df.at[idx, "team1_recent_win_rate"] = row["rolling_win_rate"]
                else:
                    df.at[idx, "team2_recent_win_rate"] = row["rolling_win_rate"]

        df["team1_recent_win_rate"] = df["team1_recent_win_rate"].fillna(0.5)
        df["team2_recent_win_rate"] = df["team2_recent_win_rate"].fillna(0.5)
        self.map_level_data = df.to_dict("records")
        print("  ✓ 已计算滚动统计")

    def save_dataset(self):
        """Save the generated dataset."""
        print("\n[5/6] 保存数据集...")

        df = pd.DataFrame(self.map_level_data)
        Path("data").mkdir(exist_ok=True)

        output_file = "data/map_level_dataset.csv"
        df.to_csv(output_file, index=False)

        print(f"  ✓ 已保存到: {output_file}")
        print(f"  ✓ 数据维度: {df.shape}")
        print("\n数据统计:")
        print(f"  总地图数: {len(df)}")
        print(f"  BO1地图: {len(df[df['match_type'] == 'BO1'])}")
        print(f"  BO3地图: {len(df[df['match_type'] == 'BO3'])}")
        print(f"  BO5地图: {len(df[df['match_type'] == 'BO5'])}")
        print(f"  时间跨度: {df['date'].min()} ~ {df['date'].max()}")
        all_teams = set(df["team1"].unique()) | set(df["team2"].unique())
        print(f"  参赛队伍: {len(all_teams)} 支")

        return output_file

    def run(self):
        """Run the full preprocessing pipeline."""
        print("\n" + "=" * 80)
        print("数据预处理：扩展到地图级别")
        print("=" * 80)

        self.load_data()
        map_count = self.extract_map_samples()
        if map_count == 0:
            raise ValueError("未提取到任何地图样本，请检查 match_details_lite.json 的地图明细结构。")
        self.add_player_aggregations()
        self.calculate_rolling_stats()
        self.augment_with_flipped_samples()
        output_file = self.save_dataset()

        print("\n" + "=" * 80)
        print("✓ 数据预处理完成！")
        print("=" * 80)
        print("\n下一步: python feature_engineering.py")

        return output_file


if __name__ == "__main__":
    prep = MapLevelDataPreparation()
    prep.run()
