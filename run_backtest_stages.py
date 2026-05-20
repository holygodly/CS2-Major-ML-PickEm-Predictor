"""
Austin Major 2025 回测：自动跑 Stage 1 + Stage 2 Swiss 预测，对比真实结果。

Stage 1: 训练数据 2025-01-10 ~ 2025-11-23（Major 前），预测 Opening Stage Swiss
Stage 2: 训练数据 2025-01-10 ~ 2025-11-29（含 Stage 1），预测 Elimination Stage Swiss
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

# ============================================================
# 真实对阵和结果
# ============================================================

STAGE1_CONFIG = {
    "name": "Stage 1 (Opening Stage)",
    "data_end_date": "2025-11-23",
    "train_cutoff_date": "2025-11-08",
    "seeded_teams": [
        # 从 Round 1 matchups 推断种子顺序 (1v9, 2v10, ... 8v16)
        "Legacy", "FaZe", "B8", "GamerLegion",
        "Fnatic", "PARIVISION", "NiP", "Imperial",
        "FlyQuest", "Lynn Vision", "M80", "Fluxo",
        "RED Canids", "The Huns", "NRG", "Rare Atom",
    ],
    "round1_matchups": [
        ["Legacy", "FlyQuest"],
        ["FaZe", "Lynn Vision"],
        ["B8", "M80"],
        ["GamerLegion", "Fluxo"],
        ["Fnatic", "RED Canids"],
        ["PARIVISION", "The Huns"],
        ["NiP", "NRG"],
        ["Imperial", "Rare Atom"],
    ],
    "actual_results": {
        "3-0": {"M80", "FlyQuest"},
        "qualified": {"M80", "FlyQuest", "B8", "Fnatic", "NiP", "PARIVISION", "Imperial", "FaZe"},
        "0-3": {"Lynn Vision", "Rare Atom"},
    },
}

STAGE2_CONFIG = {
    "name": "Stage 2 (Elimination Stage)",
    "data_end_date": "2025-11-29",
    "train_cutoff_date": "2025-11-24",
    "seeded_teams": [
        # 从 Round 1 matchups 推断种子顺序
        "Aurora", "Natus Vincere", "Liquid", "3DMAX",
        "Astralis", "TYLOO", "MIBR", "Passion UA",
        "M80", "FlyQuest", "B8", "Fnatic",
        "NiP", "PARIVISION", "Imperial", "FaZe",
    ],
    "round1_matchups": [
        ["Aurora", "M80"],
        ["Natus Vincere", "FlyQuest"],
        ["Liquid", "B8"],
        ["3DMAX", "Fnatic"],
        ["Astralis", "NiP"],
        ["TYLOO", "PARIVISION"],
        ["MIBR", "Imperial"],
        ["Passion UA", "FaZe"],
    ],
    "actual_results": {
        "3-0": {"Natus Vincere", "FaZe"},
        "qualified": {"Natus Vincere", "FaZe", "B8", "Imperial", "PARIVISION", "Liquid", "Passion UA", "3DMAX"},
        "0-3": {"MIBR", "FlyQuest"},
    },
}


def update_config(stage_cfg):
    """修改 config.yaml 的 data_end_date, train_cutoff_date, swiss_stage"""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["data"]["data_end_date"] = stage_cfg["data_end_date"]
    config["data"]["train_cutoff_date"] = stage_cfg["train_cutoff_date"]
    config["swiss_stage"] = {
        "seeded_teams": stage_cfg["seeded_teams"],
        "round1_matchups": stage_cfg["round1_matchups"],
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"  [OK] Config updated: data_end_date={stage_cfg['data_end_date']}, "
          f"train_cutoff={stage_cfg['train_cutoff_date']}")


def run_step(script_name, description):
    """运行一个 pipeline 步骤"""
    print(f"\n  >>> {description}: python {script_name}")
    result = subprocess.run(
        [sys.executable, "-u", script_name],
        cwd=str(BASE_DIR),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  [FAIL] {script_name} 退出码 {result.returncode}")
        return False
    return True


def evaluate_predictions(stage_cfg, predictions_path):
    """对比预测结果和真实结果"""
    actual = stage_cfg["actual_results"]
    stage_name = stage_cfg["name"]

    if not predictions_path.exists():
        print(f"  [!] 预测文件不存在: {predictions_path}")
        return None

    with predictions_path.open("r", encoding="utf-8") as f:
        output = json.load(f)

    probs = output.get("probabilities", {})
    recommendation = output.get("pickem_recommendation", {})

    print(f"\n{'=' * 72}")
    print(f"  {stage_name} 回测结果")
    print(f"{'=' * 72}")

    # 1. 概率排名 vs 实际结果
    print("\n  晋级概率排名 vs 实际:")
    sorted_by_qual = sorted(probs.items(), key=lambda x: x[1]["qualified"], reverse=True)
    actual_qualified = actual["qualified"]
    actual_30 = actual["3-0"]
    actual_03 = actual["0-3"]

    correct_qual_top8 = 0
    for i, (team, stats) in enumerate(sorted_by_qual[:8], 1):
        mark = "✓" if team in actual_qualified else "✗"
        print(f"    {i:>2}. {team:<20} {stats['qualified']*100:5.1f}%  {mark}")
        if team in actual_qualified:
            correct_qual_top8 += 1

    print(f"\n  Top 8 晋级命中: {correct_qual_top8}/8")

    # 2. 3-0 预测
    sorted_by_30 = sorted(probs.items(), key=lambda x: x[1]["3-0"], reverse=True)
    print(f"\n  3-0 概率 Top 5 vs 实际 ({', '.join(actual_30)}):")
    for i, (team, stats) in enumerate(sorted_by_30[:5], 1):
        mark = "✓" if team in actual_30 else ""
        print(f"    {i}. {team:<20} {stats['3-0']*100:5.1f}%  {mark}")

    # 3. 0-3 预测
    sorted_by_03 = sorted(probs.items(), key=lambda x: x[1]["0-3"], reverse=True)
    print(f"\n  0-3 概率 Top 5 vs 实际 ({', '.join(actual_03)}):")
    for i, (team, stats) in enumerate(sorted_by_03[:5], 1):
        mark = "✓" if team in actual_03 else ""
        print(f"    {i}. {team:<20} {stats['0-3']*100:5.1f}%  {mark}")

    # 4. Pick'em 评估
    if recommendation:
        print(f"\n  Pick'em 推荐:")
        print(f"    3-0: {recommendation.get('3-0', [])}")
        print(f"    Adv: {recommendation.get('advances', [])}")
        print(f"    0-3: {recommendation.get('0-3', [])}")

        # 计算 Pick'em 命中数
        hits = 0
        pred_30 = set(recommendation.get("3-0", []))
        pred_adv = set(recommendation.get("advances", []))
        pred_03 = set(recommendation.get("0-3", []))

        hits += len(pred_30 & actual_30)
        hits += len(pred_03 & actual_03)
        hits += len(pred_adv & actual_qualified)

        total_picks = len(pred_30) + len(pred_adv) + len(pred_03)
        print(f"\n  Pick'em 命中: {hits}/{total_picks} ({'PASS (≥5)' if hits >= 5 else 'FAIL (<5)'})")
        print(f"  预估成功率: {output.get('pickem_success_rate', 0)*100:.2f}%")

        return {"stage": stage_name, "qual_top8_hits": correct_qual_top8, "pickem_hits": hits}

    return {"stage": stage_name, "qual_top8_hits": correct_qual_top8, "pickem_hits": 0}


def run_stage(stage_cfg):
    """运行一个阶段的完整流水线"""
    stage_name = stage_cfg["name"]
    print(f"\n{'#' * 72}")
    print(f"# {stage_name}")
    print(f"# data_end_date: {stage_cfg['data_end_date']}")
    print(f"# train_cutoff: {stage_cfg['train_cutoff_date']}")
    print(f"{'#' * 72}")

    # 1. 更新配置
    update_config(stage_cfg)

    # 2. 数据预处理
    if not run_step("data_preparation.py", "数据预处理"):
        return None

    # 3. 特征工程
    if not run_step("feature_engineering.py", "特征工程"):
        return None

    # 4. 模型训练
    if not run_step("model_training.py", "模型训练"):
        return None

    # 5. Swiss 预测
    if not run_step("run_swiss_ml_prediction.py", "Swiss 预测 + Pick'em"):
        return None

    # 6. 评估
    predictions_path = BASE_DIR / "output" / "swiss_ml_predictions.json"
    result = evaluate_predictions(stage_cfg, predictions_path)

    # 保存该阶段结果副本
    stage_output = BASE_DIR / "output" / f"swiss_backtest_{stage_cfg['data_end_date']}.json"
    if predictions_path.exists():
        shutil.copy2(predictions_path, stage_output)
        print(f"\n  [OK] 结果已保存: {stage_output.name}")

    return result


def main():
    print("=" * 72)
    print("Austin Major 2025 — Swiss 阶段回测")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # 备份 config.yaml，跑完恢复
    config_backup = CONFIG_PATH.with_suffix(".yaml.bak")
    shutil.copy2(CONFIG_PATH, config_backup)
    print(f"  [OK] Config backed up to {config_backup.name}")

    t0 = time.time()
    results = []

    try:
        for stage_cfg in [STAGE1_CONFIG, STAGE2_CONFIG]:
            result = run_stage(stage_cfg)
            if result:
                results.append(result)
    finally:
        # 恢复原始 config
        shutil.copy2(config_backup, CONFIG_PATH)
        config_backup.unlink()
        print(f"\n  [OK] Config restored from backup")

    # 汇总
    print(f"\n\n{'=' * 72}")
    print("回测汇总")
    print(f"{'=' * 72}")
    for r in results:
        print(f"  {r['stage']}:")
        print(f"    晋级 Top8 命中: {r['qual_top8_hits']}/8")
        print(f"    Pick'em 命中: {r['pickem_hits']}")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}s")

    # 保存汇总
    summary_path = BASE_DIR / "output" / "backtest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"汇总保存: {summary_path}")


if __name__ == "__main__":
    main()
