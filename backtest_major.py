"""
Historical Backtest: PGL Major Copenhagen 2025 Swiss Stages
============================================================
1. Filter raw data to pre-Major only (before Nov 25 2025)
2. Run full ML pipeline: data prep → feature eng → model training
3. Simulate Swiss rounds for Stage 1 and Stage 2
4. Compare predictions vs actual results, score accuracy
"""

import io
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Force UTF-8 stdout once; pipeline modules re-wrap stdout at import time
# which can close our buffer.  We save and restore after each step.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
_safe_stdout = sys.stdout


def _restore_stdout():
    """Restore stdout if a pipeline module re-wrapped it."""
    global _safe_stdout
    if sys.stdout is not _safe_stdout:
        try:
            # Detach so old wrapper doesn't close our buffer
            if hasattr(sys.stdout, "detach"):
                sys.stdout.detach()
        except Exception:
            pass
        sys.stdout = _safe_stdout

# ── Actual results (extracted from HLTV data) ─────────────────────────
STAGE1_ACTUAL = {
    "3-0": {"FlyQuest", "M80"},
    "qualified": {"FlyQuest", "FaZe", "B8", "M80", "fnatic",
                  "PARIVISION", "Ninjas in Pyjamas", "Imperial"},
    "0-3": {"Lynn Vision", "Rare Atom"},
}

STAGE2_ACTUAL = {
    "3-0": {"Natus Vincere", "FaZe"},
    "qualified": {"Natus Vincere", "Liquid", "B8", "3DMAX",
                  "PARIVISION", "Imperial", "Passion UA", "FaZe"},
    "0-3": {"FlyQuest", "MIBR"},
}

# ── Stage 1 config ────────────────────────────────────────────────────
STAGE1_SEEDED_TEAMS = [
    "Legacy", "FlyQuest", "FaZe", "Lynn Vision",
    "B8", "M80", "GamerLegion", "Fluxo",
    "fnatic", "RED Canids", "PARIVISION", "The Huns",
    "Ninjas in Pyjamas", "NRG", "Imperial", "Rare Atom",
]
STAGE1_ROUND1 = [
    ("Legacy", "FlyQuest"),
    ("FaZe", "Lynn Vision"),
    ("B8", "M80"),
    ("GamerLegion", "Fluxo"),
    ("fnatic", "RED Canids"),
    ("PARIVISION", "The Huns"),
    ("Ninjas in Pyjamas", "NRG"),
    ("Imperial", "Rare Atom"),
]

# ── Stage 2 config ────────────────────────────────────────────────────
STAGE2_SEEDED_TEAMS = [
    "Aurora", "M80", "Natus Vincere", "FlyQuest",
    "Liquid", "B8", "3DMAX", "fnatic",
    "Astralis", "Ninjas in Pyjamas", "TYLOO", "PARIVISION",
    "MIBR", "Imperial", "Passion UA", "FaZe",
]
STAGE2_ROUND1 = [
    ("Aurora", "M80"),
    ("Natus Vincere", "FlyQuest"),
    ("Liquid", "B8"),
    ("3DMAX", "fnatic"),
    ("Astralis", "Ninjas in Pyjamas"),
    ("TYLOO", "PARIVISION"),
    ("MIBR", "Imperial"),
    ("Passion UA", "FaZe"),
]

MAJOR_CUTOFF_DATE = "2025-11-25"

# =====================================================================
# Step 1: Filter data and run pipeline
# =====================================================================

def filter_json_by_date(src_json, dst_json, cutoff):
    """Copy match_details_lite.json keeping only matches before cutoff."""
    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = {url: m for url, m in data.items() if m.get("date", "9999") < cutoff}
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"  Filtered JSON: {len(data)} → {len(filtered)} matches (before {cutoff})")
    return len(filtered)


def filter_csv_by_date(src_csv, dst_csv, cutoff, date_col="date"):
    """Copy CSV keeping only rows before cutoff."""
    df = pd.read_csv(src_csv, on_bad_lines="skip")
    before = len(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col] < cutoff]
    df.to_csv(dst_csv, index=False)
    print(f"  Filtered CSV ({Path(src_csv).name}): {before} → {len(df)} rows")
    return len(df)


def run_pipeline(work_dir):
    """Run data_preparation → feature_engineering → model_training in work_dir."""
    original_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        # Data preparation
        from data_preparation import MapLevelDataPreparation
        prep = MapLevelDataPreparation()
        prep.run()

        # Feature engineering
        from feature_engineering import FeatureEngineering
        fe = FeatureEngineering()
        fe.run()

        # Model training
        from model_training import ModelTrainer
        trainer = ModelTrainer()
        trainer.run()
    finally:
        os.chdir(original_dir)


# =====================================================================
# Step 2: Swiss simulation (reuse existing logic)
# =====================================================================

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


def get_buchholz_sorted(records, match_history, active_teams, team_seeds):
    difficulty = {}
    for team in active_teams:
        total = 0
        for opp in match_history[team]:
            ow, ol = records[opp]
            total += ow - ol
        difficulty[team] = total
    return sorted(active_teams, key=lambda t: (-difficulty[t], team_seeds.get(t, 999)))


def choose_r23_pairs(teams, match_history):
    remaining = teams.copy()
    pairs = []
    while len(remaining) >= 2:
        t1 = remaining.pop(0)
        matched = False
        for i in range(len(remaining) - 1, -1, -1):
            if remaining[i] not in match_history[t1]:
                pairs.append((t1, remaining.pop(i)))
                matched = True
                break
        if not matched:
            pairs.append((t1, remaining.pop()))
    return pairs


def choose_r45_pairs(teams, match_history):
    for pattern in PAIRING_PRIORITY:
        valid = True
        test = []
        for i1, i2 in pattern:
            if i1 >= len(teams) or i2 >= len(teams):
                valid = False
                break
            if teams[i2] in match_history[teams[i1]]:
                valid = False
                break
            test.append((teams[i1], teams[i2]))
        if valid:
            return test
    return [(teams[i1], teams[i2]) for i1, i2 in PAIRING_PRIORITY[0]
            if i1 < len(teams) and i2 < len(teams)]


def simulate_swiss(predictor, seeded_teams, round1_matchups, num_sims=3000):
    """Run Monte Carlo Swiss simulation, return probabilities and raw sims."""
    team_seeds = {t: i + 1 for i, t in enumerate(seeded_teams)}
    teams = list(seeded_teams)

    prob_counts = defaultdict(lambda: {"3-0": 0, "qualified": 0, "0-3": 0, "total": 0})
    all_sims = []

    for sim_idx in range(num_sims):
        if (sim_idx + 1) % max(num_sims // 10, 1) == 0:
            print(f"    sim {sim_idx + 1}/{num_sims}")

        records = {t: (0, 0) for t in teams}
        history = {t: [] for t in teams}

        # Round 1
        for t1, t2 in round1_matchups:
            winner = _sim_match(predictor, t1, t2, records)
            _update(records, history, t1, t2, winner)

        # Rounds 2-5
        for rnd in range(2, 6):
            groups = defaultdict(list)
            for t, (w, l) in records.items():
                if w < 3 and l < 3:
                    groups[(w, l)].append(t)
            if not groups:
                break
            for key, grp_teams in groups.items():
                ordered = get_buchholz_sorted(records, history, grp_teams, team_seeds)
                pairs = choose_r23_pairs(ordered, history) if rnd in (2, 3) \
                    else choose_r45_pairs(ordered, history)
                for t1, t2 in pairs:
                    w1, l1 = records[t1]
                    w2, l2 = records[t2]
                    is_elim_adv = w1 == 2 or l1 == 2 or w2 == 2 or l2 == 2
                    winner = _sim_match(predictor, t1, t2, records, bo3=is_elim_adv)
                    _update(records, history, t1, t2, winner)

        # Collect
        sim_result = {"3-0": set(), "qualified": set(), "0-3": set()}
        for t, (w, l) in records.items():
            prob_counts[t]["total"] += 1
            if w == 3 and l == 0:
                prob_counts[t]["3-0"] += 1
                prob_counts[t]["qualified"] += 1
                sim_result["3-0"].add(t)
                sim_result["qualified"].add(t)
            elif w == 3:
                prob_counts[t]["qualified"] += 1
                sim_result["qualified"].add(t)
            elif l == 3 and w == 0:
                prob_counts[t]["0-3"] += 1
                sim_result["0-3"].add(t)
        all_sims.append(sim_result)

    # Convert to probabilities
    probs = {}
    for t, c in prob_counts.items():
        n = c["total"] or 1
        probs[t] = {
            "3-0": c["3-0"] / n,
            "qualified": c["qualified"] / n,
            "0-3": c["0-3"] / n,
        }
    return probs, all_sims


def _sim_match(predictor, t1, t2, records, bo3=False):
    """Simulate a single match using the ML predictor."""
    elo1 = predictor.team_elo.get(t1, 1500)
    elo2 = predictor.team_elo.get(t2, 1500)
    try:
        if bo3:
            winner_flag, _, _ = predictor.simulate_bo3(t1, t2, elo_team1=elo1, elo_team2=elo2, use_veto=True)
        else:
            winner_flag, _, _ = predictor.simulate_bo1(t1, t2, elo_team1=elo1, elo_team2=elo2, use_veto=True)
        return t1 if winner_flag == 1 else t2
    except Exception:
        # Fallback: ELO coin flip
        elo_prob = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        import random
        return t1 if random.random() < elo_prob else t2


def _update(records, history, t1, t2, winner):
    loser = t2 if winner == t1 else t1
    w, l = records[winner]
    records[winner] = (w + 1, l)
    w, l = records[loser]
    records[loser] = (w, l + 1)
    history[t1].append(t2)
    history[t2].append(t1)


# =====================================================================
# Step 3: Score predictions
# =====================================================================

def pickem_score_strict(pick, actual):
    """Strict Pick'em scoring: 3-0 must be exactly 3-0, 0-3 exactly 0-3."""
    score = 0
    detail = []
    for t in pick["3-0"]:
        hit = t in actual["3-0"]
        score += int(hit)
        detail.append((t, "3-0", hit))
    for t in pick["advances"]:
        hit = t in actual["qualified"]
        score += int(hit)
        detail.append((t, "Adv", hit))
    for t in pick["0-3"]:
        hit = t in actual["0-3"]
        score += int(hit)
        detail.append((t, "0-3", hit))
    return score, detail


def naive_pickem(probs):
    """Naive Pick'em: top-2 by P(3-0), top-2 by P(0-3), top-6 remaining by P(qual)."""
    ranked_30 = sorted(probs, key=lambda t: probs[t]["3-0"], reverse=True)
    ranked_03 = sorted(probs, key=lambda t: probs[t]["0-3"], reverse=True)
    ranked_q = sorted(probs, key=lambda t: probs[t]["qualified"], reverse=True)

    pick_30 = ranked_30[:2]
    pick_03 = [t for t in ranked_03 if t not in pick_30][:2]
    used = set(pick_30) | set(pick_03)
    pick_adv = [t for t in ranked_q if t not in used][:6]
    return {"3-0": pick_30, "advances": pick_adv, "0-3": pick_03}


def score_stage(name, probs, actual, all_sims, device_mgr=None):
    """Score ML predictions against actual results, with naive + optimizer."""
    print(f"\n{'=' * 72}")
    print(f"  {name} — Prediction vs Reality")
    print(f"{'=' * 72}")

    # Print probability table
    sorted_teams = sorted(probs.items(), key=lambda x: x[1]["qualified"], reverse=True)
    print(f"\n{'Team':<22} {'P(3-0)':>8} {'P(Qual)':>9} {'P(0-3)':>8}  Actual")
    print("-" * 72)

    correct_qual = 0
    correct_elim = 0
    for team, p in sorted_teams:
        actual_status = ""
        if team in actual["3-0"]:
            actual_status = "✅ 3-0"
        elif team in actual["qualified"]:
            actual_status = "✅ Qualified"
        elif team in actual["0-3"]:
            actual_status = "❌ 0-3"
        else:
            actual_status = "❌ Eliminated"

        is_qual = team in actual["qualified"]
        predicted_qual = p["qualified"] > 0.5

        if is_qual and predicted_qual:
            correct_qual += 1
        elif not is_qual and not predicted_qual:
            correct_elim += 1

        marker = "→" if predicted_qual == is_qual else "✗"
        print(f"{team:<22} {p['3-0']*100:7.1f}% {p['qualified']*100:8.1f}% {p['0-3']*100:7.1f}%  {actual_status}  {marker}")

    total_correct = correct_qual + correct_elim
    print(f"\nQualified/Eliminated classification: {total_correct}/16 correct")
    print(f"  Correctly predicted qualifiers: {correct_qual}/8")
    print(f"  Correctly predicted eliminations: {correct_elim}/8")

    # ── Naive Pick'em ──────────────────────────────────────────────
    naive = naive_pickem(probs)
    naive_pts, naive_detail = pickem_score_strict(naive, actual)
    print(f"\n  --- Naive Pick'em (top-by-probability) ---")
    print(f"  3-0: {naive['3-0']}")
    print(f"  Adv: {naive['advances']}")
    print(f"  0-3: {naive['0-3']}")
    print(f"  Score: {naive_pts}/10  {'✅ PASS' if naive_pts >= 5 else '❌ FAIL'}")
    for t, slot, hit in naive_detail:
        print(f"    {t:<22} [{slot}]  {'✅' if hit else '❌'}")

    # ── Optimizer Pick'em ──────────────────────────────────────────
    print(f"\n  --- Optimized Pick'em (brute-force) ---")
    from pickem_optimizer import PickemOptimizer
    _restore_stdout()
    _num_sims = len(all_sims)
    _batch = device_mgr.recommend_batch_size(num_sims=_num_sims) if device_mgr else 5000
    _dev_id = device_mgr.cuda_device_id if (device_mgr and device_mgr.cuda_available) else None
    optimizer = PickemOptimizer(
        teams=list(probs.keys()),
        all_simulations=all_sims,
        use_gpu=True,
        batch_size=_batch,
        device_id=_dev_id,
    )
    opt_pick, opt_rate = optimizer.run()
    _restore_stdout()

    if opt_pick:
        opt_pts, opt_detail = pickem_score_strict(opt_pick, actual)
        print(f"  3-0: {opt_pick['3-0']}")
        print(f"  Adv: {opt_pick['advances']}")
        print(f"  0-3: {opt_pick['0-3']}")
        print(f"  Estimated success rate: {opt_rate*100:.2f}%")
        print(f"  Actual score: {opt_pts}/10  {'✅ PASS' if opt_pts >= 5 else '❌ FAIL'}")
        for t, slot, hit in opt_detail:
            print(f"    {t:<22} [{slot}]  {'✅' if hit else '❌'}")
    else:
        opt_pts = naive_pts
        print(f"  Optimizer returned None, using naive result.")

    return {
        "qual_accuracy": total_correct / 16,
        "naive_pts": naive_pts,
        "opt_pts": opt_pts,
        "opt_rate": opt_rate if opt_pick else 0,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 72)
    print("  BACKTEST: PGL Major 2025 Swiss Stages")
    print("=" * 72)

    work_dir = Path(__file__).parent.resolve()
    data_src = (work_dir.parent / "HLTV" / "data").resolve()
    backtest_data = work_dir / "backtest_data"

    # ── 1. Prepare filtered data ──────────────────────────────────
    print("\n[Step 1] Filtering data to pre-Major (before 2025-11-25)...")
    backtest_data.mkdir(exist_ok=True)

    filter_json_by_date(
        data_src / "match_details_lite.json",
        backtest_data / "match_details_lite.json",
        MAJOR_CUTOFF_DATE,
    )
    filter_csv_by_date(
        data_src / "results_all_matches.csv",
        backtest_data / "results_all_matches.csv",
        MAJOR_CUTOFF_DATE,
    )
    filter_csv_by_date(
        data_src / "player_stats.csv",
        backtest_data / "player_stats.csv",
        MAJOR_CUTOFF_DATE,
        date_col="match_date",
    )

    # ── 2. Patch config to use backtest data ──────────────────────
    print("\n[Step 2] Running ML pipeline on pre-Major data...")
    import yaml
    config_path = work_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["data"]["data_dir"] = str(backtest_data)

    backtest_config = work_dir / "config_backtest.yaml"
    with open(backtest_config, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # Run pipeline
    os.chdir(work_dir)

    from data_preparation import MapLevelDataPreparation
    _restore_stdout()
    prep = MapLevelDataPreparation(config_path=str(backtest_config))
    prep.run()
    _restore_stdout()

    from feature_engineering import FeatureEngineering
    _restore_stdout()
    fe = FeatureEngineering()
    fe.run()
    _restore_stdout()

    from model_training import ModelTrainer
    _restore_stdout()
    trainer = ModelTrainer(config_path=str(backtest_config))
    trainer.run()
    _restore_stdout()

    # ── 3. Load predictor and simulate ────────────────────────────
    print("\n[Step 3] Loading trained model and simulating Swiss stages...")
    from hybrid_playoff_predictor import HybridPlayoffPredictor
    _restore_stdout()
    predictor = HybridPlayoffPredictor(str(backtest_config))

    # 选择设备并估算模拟参数
    from gpu_accelerator import DeviceManager
    _restore_stdout()
    device_mgr = DeviceManager()
    device_mgr.print_config_summary()
    optimal = device_mgr.get_optimal_params()
    NUM_SIMS = optimal["num_simulations"]

    # GPU-accelerated Swiss simulation
    from swiss_simulator_gpu import SwissSimulatorGPU
    _restore_stdout()

    print(f"\n--- Stage 1 Simulation ({NUM_SIMS:,} sims, GPU-accelerated) ---")
    s1_sim = SwissSimulatorGPU(predictor, STAGE1_SEEDED_TEAMS, device_mgr)
    s1_probs, s1_sims = s1_sim.simulate(STAGE1_ROUND1, NUM_SIMS)
    s1_scores = score_stage("Stage 1: Opening Stage", s1_probs, STAGE1_ACTUAL, s1_sims, device_mgr)

    print(f"\n--- Stage 2 Simulation ({NUM_SIMS:,} sims, GPU-accelerated) ---")
    s2_sim = SwissSimulatorGPU(predictor, STAGE2_SEEDED_TEAMS, device_mgr)
    s2_probs, s2_sims = s2_sim.simulate(STAGE2_ROUND1, NUM_SIMS)
    s2_scores = score_stage("Stage 2: Elimination Stage", s2_probs, STAGE2_ACTUAL, s2_sims, device_mgr)

    # ── 4. Summary ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  BACKTEST SUMMARY")
    print("=" * 72)
    print(f"  {'':30} {'Stage 1':>10} {'Stage 2':>10}")
    print(f"  {'Qual/Elim accuracy':<30} {s1_scores['qual_accuracy']*100:9.0f}% {s2_scores['qual_accuracy']*100:9.0f}%")
    lbl_naive = "Naive Pick'em score"
    lbl_opt = "Optimized Pick'em score"
    print(f"  {lbl_naive:<30} {s1_scores['naive_pts']:>7}/10 {s2_scores['naive_pts']:>7}/10")
    print(f"  {lbl_opt:<30} {s1_scores['opt_pts']:>7}/10 {s2_scores['opt_pts']:>7}/10")
    print(f"  {'Optimizer est. success rate':<30} {s1_scores['opt_rate']*100:8.1f}% {s2_scores['opt_rate']*100:8.1f}%")
    print()
    s1_pass = '✅' if s1_scores['opt_pts'] >= 5 else '❌'
    s2_pass = '✅' if s2_scores['opt_pts'] >= 5 else '❌'
    print(f"  Pick'em pass (≥5):  Stage 1 {s1_pass}  Stage 2 {s2_pass}")

    # Cleanup
    _cleanup_backtest(backtest_config, backtest_data)
    print("\n✓ Backtest complete.")


def _cleanup_backtest(backtest_config, backtest_data):
    """清理 backtest 产生的临时文件。可独立调用。"""
    if backtest_config and backtest_config.exists():
        backtest_config.unlink()
    # backtest_data 目录保留（可能想检查中间数据），但下次会覆盖


def cleanup_stale():
    """清理上次中断残留的临时文件，供重跑前手动调用。"""
    work_dir = Path(__file__).parent.resolve()
    backtest_config = work_dir / "config_backtest.yaml"

    cleaned = []
    if backtest_config.exists():
        backtest_config.unlink()
        cleaned.append(str(backtest_config.name))

    # __pycache__ 不清理（加速后续 import）
    # models/ 和 data/ 也保留（pipeline 会覆盖）

    if cleaned:
        print(f"已清理: {', '.join(cleaned)}")
    else:
        print("无残留文件需要清理，可以直接重新运行。")


if __name__ == "__main__":
    main()
