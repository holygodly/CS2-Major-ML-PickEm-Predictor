"""
预测结果图表。

跑完 run_swiss_ml_prediction.py 后调用本脚本(也可由预测脚本调用),
读取 output/swiss_ml_predictions.json,在 output/charts/ 下生成几张
PNG:

  1. qualification_ranking.png  — 16 队晋级概率排名(主图)
  2. outcome_breakdown.png      — 3-0 / 晋级 / 0-3 概率分解堆叠图
  3. pickem_card.png            — Pick'em 推荐 + 估计成功率卡片
  4. matchup_heatmap.png        — Round 2-5 各队相遇频率热图

单独运行:
    python generate_charts.py
    python generate_charts.py path/to/predictions.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# 中文字体兼容(Windows 用 Microsoft YaHei,Linux 服务器一般有 DejaVu Sans)
# 如果服务器没有中文字体,会回退到 DejaVu Sans 显示拉丁字符,中文部分变方框。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 图表配色
# ============================================================
COLOR_GOLD = "#FFB300"      # 3-0
COLOR_GREEN = "#4CAF50"     # 3-1 / 3-2 晋级
COLOR_GRAY = "#9E9E9E"      # 未晋级(非 0-3)
COLOR_RED = "#E53935"       # 0-3
COLOR_TOP8_BG = "#E8F5E9"   # 顶部 8 队浅绿底色
COLOR_BOT8_BG = "#FFEBEE"   # 底部 8 队浅红底色
COLOR_AXIS = "#37474F"
COLOR_GRID = "#CFD8DC"


def _format_meta_line(output: dict) -> str:
    ts = output.get("timestamp", "")
    n_sim = output.get("num_simulations", "?")
    return f"Generated: {ts} | {n_sim:,} simulations | Pure ML + Isotonic Calibration"


# ============================================================
# Chart 1: 晋级概率排名
# ============================================================
def chart_qualification_ranking(output: dict, save_path: Path):
    probs = output["probabilities"]
    sorted_teams = sorted(probs.items(), key=lambda kv: kv[1]["qualified"], reverse=True)
    n = len(sorted_teams)

    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(n)[::-1]   # 倒序 → top 在最上

    qual_vals = [t[1]["qualified"] * 100 for t in sorted_teams]
    p30_vals = [t[1].get("3-0", 0) * 100 for t in sorted_teams]
    p31_or_32 = [(t[1].get("3-1-or-3-2", t[1]["qualified"] - t[1].get("3-0", 0))) * 100 for t in sorted_teams]

    # 主体:晋级率(灰色背景,显示总长度)
    ax.barh(y_pos, qual_vals, color=COLOR_GRAY, alpha=0.25, height=0.68, label="P(eliminated 1-3 / 2-3)")
    # 3-0 部分(金色,左起)
    ax.barh(y_pos, p30_vals, color=COLOR_GOLD, alpha=0.95, height=0.68, label="P(3-0)")
    # 3-1 / 3-2 部分(绿色,接在 3-0 后面)
    ax.barh(y_pos, p31_or_32, left=p30_vals, color=COLOR_GREEN, alpha=0.85, height=0.68, label="P(3-1 / 3-2)")

    # 队名 + 总晋级率标签
    for i, (team, stats) in enumerate(sorted_teams):
        yi = y_pos[i]
        q = stats["qualified"] * 100
        # 队名(左侧)
        ax.text(-2, yi, team, va="center", ha="right", fontsize=11, color=COLOR_AXIS,
                fontweight="bold")
        # 晋级率数字(条形尾部)
        ax.text(q + 1, yi, f"{q:.1f}%", va="center", ha="left", fontsize=10, color=COLOR_AXIS)

    # Top 8 / Bot 8 分割线(8 支晋级 = 第 8 名后画线)
    if n >= 9:
        ax.axhline(y_pos[8] + 0.5, color="#FF6F00", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(102, y_pos[8] + 0.5, "Qualify cutoff", va="center", ha="left",
                fontsize=9, color="#FF6F00", fontweight="bold")

    ax.set_yticks([])
    ax.set_xlim(0, 110)
    ax.set_xlabel("Probability (%)", fontsize=11, color=COLOR_AXIS)
    ax.set_title("CS2 Major Swiss — Qualification Probability per Team",
                 fontsize=14, fontweight="bold", color=COLOR_AXIS, pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", colors=COLOR_AXIS)
    ax.grid(axis="x", color=COLOR_GRID, linestyle="-", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95)

    # 注脚
    fig.text(0.5, 0.01, _format_meta_line(output), ha="center", fontsize=8,
             color="#78909C", style="italic")

    plt.subplots_adjust(left=0.16, right=0.93, top=0.93, bottom=0.07)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {save_path.name}")


# ============================================================
# Chart 2: 三类结果(3-0 / 晋级 / 0-3)分解
# ============================================================
def chart_outcome_breakdown(output: dict, save_path: Path):
    probs = output["probabilities"]
    sorted_teams = sorted(probs.items(), key=lambda kv: kv[1]["qualified"], reverse=True)
    n = len(sorted_teams)
    teams = [t[0] for t in sorted_teams]

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(n)
    width = 0.27

    p30 = [t[1].get("3-0", 0) * 100 for t in sorted_teams]
    p_adv = [(t[1].get("3-1-or-3-2", t[1]["qualified"] - t[1].get("3-0", 0))) * 100 for t in sorted_teams]
    p03 = [t[1].get("0-3", 0) * 100 for t in sorted_teams]

    ax.bar(x - width, p30, width, color=COLOR_GOLD, label="P(3-0)", edgecolor="white", linewidth=0.5)
    ax.bar(x, p_adv, width, color=COLOR_GREEN, label="P(3-1 / 3-2)", edgecolor="white", linewidth=0.5)
    ax.bar(x + width, p03, width, color=COLOR_RED, label="P(0-3)", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=42, ha="right", fontsize=9, color=COLOR_AXIS)
    ax.set_ylabel("Probability (%)", fontsize=11, color=COLOR_AXIS)
    ax.set_title("Outcome Probability Breakdown — 3-0 / Qualified 3-X / 0-3",
                 fontsize=13, fontweight="bold", color=COLOR_AXIS, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=COLOR_AXIS)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="-", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95)

    fig.text(0.5, 0.01, _format_meta_line(output), ha="center", fontsize=8,
             color="#78909C", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {save_path.name}")


# ============================================================
# Chart 3: Pick'em 推荐卡片
# ============================================================
def chart_pickem_card(output: dict, save_path: Path):
    rec = output.get("pickem_recommendation", {})
    probs = output["probabilities"]
    success_rate = output.get("pickem_success_rate", 0) * 100

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    fig.suptitle("Pick'em Recommendation — Optimal Combo (Brute-Force)",
                 fontsize=16, fontweight="bold", color=COLOR_AXIS, y=0.96)

    # 估计成功率 badge
    badge_color = COLOR_GREEN if success_rate >= 50 else (COLOR_GOLD if success_rate >= 35 else COLOR_RED)
    fig.text(0.5, 0.89,
             f"Estimated Success Rate (≥5/10 hits): {success_rate:.1f}%",
             ha="center", fontsize=12, color="white",
             bbox=dict(boxstyle="round,pad=0.6", facecolor=badge_color, edgecolor="none"))

    # 三栏: 3-0 / Adv / 0-3
    sections = [
        ("3-0", rec.get("3-0", []), COLOR_GOLD, 0.15),
        ("Qualified (3-1/3-2)", rec.get("advances", []), COLOR_GREEN, 0.42),
        ("0-3", rec.get("0-3", []), COLOR_RED, 0.85),
    ]

    # 三个区域横向并排
    x_positions = [0.08, 0.34, 0.71]
    widths = [0.22, 0.32, 0.22]

    for (title, teams, color, _), x0, w in zip(sections, x_positions, widths):
        ax = fig.add_axes([x0, 0.12, w, 0.68])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        # 标题 chip
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, 0.92), 1.0, 0.08, boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color, edgecolor="none", transform=ax.transAxes,
        ))
        ax.text(0.5, 0.96, title, ha="center", va="center",
                fontsize=12, color="white", fontweight="bold", transform=ax.transAxes)

        if not teams:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                    fontsize=11, color="#B0BEC5", transform=ax.transAxes)
            continue

        # 每个队一行
        row_height = 0.85 / max(len(teams), 1)
        for i, team in enumerate(teams):
            y = 0.88 - (i + 0.5) * row_height
            # 取对应的概率字段
            if title == "3-0":
                pct = probs.get(team, {}).get("3-0", 0) * 100
                label = f"P(3-0)={pct:.0f}%"
            elif title == "0-3":
                pct = probs.get(team, {}).get("0-3", 0) * 100
                label = f"P(0-3)={pct:.0f}%"
            else:
                pct = probs.get(team, {}).get("3-1-or-3-2",
                       probs.get(team, {}).get("qualified", 0) - probs.get(team, {}).get("3-0", 0)) * 100
                label = f"P(3-1/3-2)={pct:.0f}%"
            ax.text(0.5, y + row_height * 0.18, team, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=COLOR_AXIS, transform=ax.transAxes)
            ax.text(0.5, y - row_height * 0.18, label, ha="center", va="center",
                    fontsize=9.5, color="#607D8B", transform=ax.transAxes)
            # 分隔线(axhline 自动用 axes 坐标系)
            if i < len(teams) - 1:
                ax.axhline(y - row_height * 0.5, xmin=0.1, xmax=0.9,
                           color="#ECEFF1", linewidth=1)

    fig.text(0.5, 0.04, _format_meta_line(output), ha="center", fontsize=8,
             color="#78909C", style="italic")

    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {save_path.name}")


# ============================================================
# Chart 4: Round 2-5 各队相遇频率热图
# ============================================================
def chart_matchup_heatmap(output: dict, save_path: Path):
    teams = list(output["seeded_teams"])
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # 累加 Round 2-5 的相遇频率
    matrix = np.zeros((n, n))
    rmf = output.get("round_matchup_frequency", {})
    for rnd_key, matchups in rmf.items():
        if rnd_key == "1":
            continue  # Round 1 是 100% 固定,看不出信息
        for matchup_str, freq in matchups.items():
            try:
                t1, t2 = matchup_str.split(" vs ")
            except ValueError:
                continue
            i, j = idx.get(t1), idx.get(t2)
            if i is None or j is None:
                continue
            matrix[i, j] += freq
            matrix[j, i] += freq

    # 主对角填 NaN 以便区分
    for i in range(n):
        matrix[i, i] = np.nan

    fig, ax = plt.subplots(figsize=(10, 9))
    # 自定义颜色映射:浅 → 深(类似热力图)
    cmap = LinearSegmentedColormap.from_list(
        "heat", ["#FFFFFF", "#FFE0B2", "#FF8A65", "#D84315"], N=256,
    )
    im = ax.imshow(matrix, cmap=cmap, aspect="equal", origin="upper")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(teams, rotation=45, ha="right", fontsize=9, color=COLOR_AXIS)
    ax.set_yticklabels(teams, fontsize=9, color=COLOR_AXIS)

    # 单元格里写数字(超过阈值的)
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if np.isnan(v) or v < 0.05:  # 阈值:总相遇概率 < 5% 不写
                continue
            txt_color = "white" if v > 0.4 else COLOR_AXIS
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                    fontsize=7.5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Cumulative meet probability (Rounds 2-5)", fontsize=10, color=COLOR_AXIS)
    cbar.ax.tick_params(colors=COLOR_AXIS)

    ax.set_title("Matchup Frequency — Cumulative Across Rounds 2-5",
                 fontsize=13, fontweight="bold", color=COLOR_AXIS, pad=14)
    ax.spines[:].set_visible(False)

    fig.text(0.5, 0.01, _format_meta_line(output), ha="center", fontsize=8,
             color="#78909C", style="italic")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {save_path.name}")


# ============================================================
# Driver
# ============================================================
def generate_all(predictions_path: Path, output_dir: Path):
    print(f"[Charts] 读取: {predictions_path}")
    with predictions_path.open("r", encoding="utf-8") as f:
        output = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Charts] 输出目录: {output_dir}")

    try:
        chart_qualification_ranking(output, output_dir / "qualification_ranking.png")
    except Exception as exc:
        print(f"  [WARN] qualification_ranking 失败: {exc}")
    try:
        chart_outcome_breakdown(output, output_dir / "outcome_breakdown.png")
    except Exception as exc:
        print(f"  [WARN] outcome_breakdown 失败: {exc}")
    try:
        chart_pickem_card(output, output_dir / "pickem_card.png")
    except Exception as exc:
        print(f"  [WARN] pickem_card 失败: {exc}")
    try:
        chart_matchup_heatmap(output, output_dir / "matchup_heatmap.png")
    except Exception as exc:
        print(f"  [WARN] matchup_heatmap 失败: {exc}")

    print(f"[Charts] 完成,共 4 张图保存到 {output_dir}/")


def main():
    if len(sys.argv) > 1:
        predictions_path = Path(sys.argv[1])
    else:
        predictions_path = Path("output/swiss_ml_predictions.json")

    output_dir = predictions_path.parent / "charts"
    generate_all(predictions_path, output_dir)


if __name__ == "__main__":
    main()
