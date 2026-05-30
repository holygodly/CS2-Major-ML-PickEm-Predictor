"""
GPU-accelerated Swiss stage Monte Carlo simulator.

Strategy:
1. Pre-compute ALL pairwise win probabilities via ONE batch XGBoost GPU call
   (16×16×7_maps ≈ 1792 rows → single DMatrix predict).
2. Derive BO1/BO3 win probability matrices from map-level predictions.
3. Run the entire Swiss simulation using PyTorch tensors on GPU:
   - All N simulations run in parallel as a batch.
   - Random draws are vectorized.
   - Only the pairing logic (Buchholz sort) remains on CPU per-round,
     but outcomes are resolved on GPU in one shot.

This eliminates ~2M individual XGBoost calls → 1 batch call + tensor ops.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


_MP_SIMULATOR = None
_MP_RAND_ALL = None


def _run_swiss_chunk_worker(args):
    start, end, r1_pairs, team_seeds = args
    rand_chunk = _MP_RAND_ALL[start:end]
    records, matchup_counts, bo_counts = _MP_SIMULATOR._run_all_sims(
        r1_pairs,
        rand_chunk,
        end - start,
        team_seeds,
    )
    return (
        start,
        records,
        {round_num: dict(counter) for round_num, counter in matchup_counts.items()},
        {round_num: dict(counter) for round_num, counter in bo_counts.items()},
    )


class SwissSimulatorGPU:
    """GPU-accelerated Swiss Monte Carlo simulator."""

    def __init__(self, predictor, teams: List[str], device_mgr=None):
        """
        Args:
            predictor: HybridPlayoffPredictor instance (already initialized)
            teams: List of team names in seed order
            device_mgr: DeviceManager instance for GPU selection
        """
        self.predictor = predictor
        self.teams = list(teams)
        self.num_teams = len(self.teams)
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        self.device_mgr = device_mgr or getattr(predictor, "device_mgr", None)

        # Swiss simulation uses PyTorch; do not depend on XGBoost's GPU probe.
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_id = getattr(self.device_mgr, "cuda_device_id", 0) if self.device_mgr else 0
            self.device = torch.device(f"cuda:{device_id}")
            torch.empty((1,), device=self.device)
            self.backend = "gpu"
        elif TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            self.backend = "torch_cpu"
        else:
            self.device = None
            self.backend = "numpy"

        # Pre-compute probability matrices
        self._precompute_probabilities()

    # ------------------------------------------------------------------
    # Step 1: Batch probability computation
    # ------------------------------------------------------------------

    def _precompute_probabilities(self):
        """Compute all pairwise win probabilities in one batch GPU call.

        Uses 3 different series contexts to avoid the BO1-for-BO3 bias:
          - BO1 context: match_type=BO1 (for Swiss BO1 rounds)
          - BO3 normal: match_type=BO3, score=[0,0] (for maps 1 & 2)
          - BO3 decider: match_type=BO3, score=[1,1] (for map 3)
        """
        t0 = time.time()
        print(f"  [Swiss GPU] Pre-computing {self.num_teams}×{self.num_teams} matchup matrix...")

        map_pool = self.predictor.current_map_pool
        n = self.num_teams
        num_maps = len(map_pool)

        # Three series contexts: BO1, BO3 normal map, BO3 decider
        contexts = [
            {  # BO1
                "current_score": [0, 0], "previous_winner": None,
                "previous_was_comeback": False, "previous_was_close": False,
                "match_type": "BO1", "map_index": 0,
                "team1_win_streak": 0, "picker": "unknown",
            },
            {  # BO3 normal (maps 1-2)
                "current_score": [0, 0], "previous_winner": None,
                "previous_was_comeback": False, "previous_was_close": False,
                "match_type": "BO3", "map_index": 0,
                "team1_win_streak": 0, "picker": "unknown",
            },
            {  # BO3 decider (map 3)
                "current_score": [1, 1], "previous_winner": None,
                "previous_was_comeback": False, "previous_was_close": False,
                "match_type": "BO3", "map_index": 2,
                "team1_win_streak": 0, "picker": "unknown",
            },
        ]

        all_features = []
        pair_map_index = []  # (context_idx, i, j, map_idx) for each row

        for ctx_idx, ctx in enumerate(contexts):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    t1, t2 = self.teams[i], self.teams[j]
                    for m_idx, map_name in enumerate(map_pool):
                        features = self.predictor.build_features_for_map(t1, t2, map_name, ctx)
                        all_features.append(features)
                        pair_map_index.append((ctx_idx, i, j, m_idx))

        # Concatenate and predict in ONE batch
        all_features_df = pd.concat(all_features, ignore_index=True)
        dmatrix = xgb.DMatrix(all_features_df)
        ml_probs_raw = self.predictor.xgb_model.predict(dmatrix)

        # 有校准器就校准概率。
        if self.predictor.calibrator is not None:
            ml_probs = self.predictor.calibrator.transform(ml_probs_raw)
        else:
            ml_probs = ml_probs_raw

        def _mix_ml_elo_probability(ml_prob, team_i, team_j):
            elo1 = self.predictor.team_elo.get(self.teams[team_i], 1500)
            elo2 = self.predictor.team_elo.get(self.teams[team_j], 1500)
            elo_prob = self.predictor._elo_probability(elo1, elo2)
            ml_weight = self.predictor.xgb_weight + self.predictor.momentum_weight
            total_weight = ml_weight + self.predictor.elo_weight
            if total_weight <= 0:
                return float(ml_prob)
            return float((ml_prob * ml_weight + elo_prob * self.predictor.elo_weight) / total_weight)

        # Build 3 map-level probability tensors
        # prob_maps[ctx_idx][i][j][m] = P(team_i beats team_j on map_m, given context)
        prob_maps = [np.full((n, n, num_maps), 0.5, dtype=np.float32) for _ in range(3)]

        for idx, (ctx_idx, i, j, m_idx) in enumerate(pair_map_index):
            mixed_prob = _mix_ml_elo_probability(float(ml_probs[idx]), i, j)
            prob_maps[ctx_idx][i, j, m_idx] = np.clip(mixed_prob, 0.05, 0.95)

        prob_map_bo1 = prob_maps[0]
        prob_map_bo3_normal = prob_maps[1]
        prob_map_bo3_decider = prob_maps[2]

        # Compute BO1/BO3 match probabilities via veto
        self.prob_bo1 = np.full((n, n), 0.5, dtype=np.float32)
        self.prob_bo3 = np.full((n, n), 0.5, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                t1, t2 = self.teams[i], self.teams[j]

                # BO1: veto → pick map → lookup from BO1 context
                try:
                    veto = self.predictor.veto_simulator.simulate_bo1_veto(t1, t2)
                    bo1_map = veto["maps"][0]["map"]
                    m_idx = map_pool.index(bo1_map) if bo1_map in map_pool else 0
                    side_adv = self.predictor.side_analyzer.get_side_advantage(
                        bo1_map, veto["maps"][0].get("team1_starts_ct", True)
                    )
                    self.prob_bo1[i, j] = np.clip(prob_map_bo1[i, j, m_idx] + side_adv, 0.05, 0.95)
                except Exception:
                    elo1 = self.predictor.team_elo.get(t1, 1500)
                    elo2 = self.predictor.team_elo.get(t2, 1500)
                    self.prob_bo1[i, j] = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

                # BO3: veto → 3 maps → use BO3-context probabilities
                try:
                    tactical = self.predictor.tactical_analyzer.analyze_bo3_tactical_advantage(t1, t2)
                    veto_maps = tactical["veto_result"]["maps"]
                    map_probs = []
                    for map_pos, map_info in enumerate(veto_maps[:3]):
                        m_name = map_info["map"]
                        m_idx = map_pool.index(m_name) if m_name in map_pool else 0
                        side_adv = self.predictor.side_analyzer.get_side_advantage(
                            m_name, map_info.get("team1_starts_ct", True)
                        )
                        # Map 3 (decider) uses decider context; maps 1-2 use normal BO3 context
                        if map_pos < 2:
                            p = np.clip(prob_map_bo3_normal[i, j, m_idx] + side_adv, 0.05, 0.95)
                        else:
                            p = np.clip(prob_map_bo3_decider[i, j, m_idx] + side_adv, 0.05, 0.95)
                        map_probs.append(p)

                    while len(map_probs) < 3:
                        map_probs.append(0.5)
                    p1, p2, p3 = map_probs[0], map_probs[1], map_probs[2]
                    p_bo3 = p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
                    self.prob_bo3[i, j] = np.clip(p_bo3, 0.05, 0.95)
                except Exception:
                    self.prob_bo3[i, j] = self.prob_bo1[i, j]

        self._enforce_matchup_symmetry(self.prob_bo1)
        self._enforce_matchup_symmetry(self.prob_bo3)

        elapsed = time.time() - t0
        n_predictions = len(all_features_df)
        print(f"  [Swiss GPU] Probability matrix computed in {elapsed:.1f}s "
              f"({n_predictions} predictions batched, 3 contexts)")
        print(f"  [Swiss GPU] Backend: {self.backend} | Device: {self.device}")

    def _enforce_matchup_symmetry(self, matrix):
        """Make P(A beats B) exactly complement P(B beats A).

        The ML feature builder can produce slightly different probabilities for
        A-vs-B and B-vs-A. Swiss pairing order should not decide match strength,
        so we average both directions into one consistent matchup probability.
        """
        n = matrix.shape[0]
        for i in range(n):
            matrix[i, i] = 0.5
            for j in range(i + 1, n):
                p_ij = float(matrix[i, j])
                p_ji = float(matrix[j, i])
                symmetric = float(np.clip((p_ij + (1.0 - p_ji)) / 2.0, 0.05, 0.95))
                matrix[i, j] = symmetric
                matrix[j, i] = 1.0 - symmetric

    def _apply_match_format_variance(self, probability, match_type, size=None):
        prob = float(np.clip(probability, 0.01, 0.99))
        if match_type == "BO1":
            prob = 0.5 + (prob - 0.5) * self.predictor.bo1_probability_shrink

        if not self.predictor.enable_form_variance:
            if size is None:
                return prob
            return np.full(size, prob, dtype=np.float32)

        variance = self.predictor.form_variance_by_format.get(
            match_type,
            self.predictor.form_variance_by_format["BO3"],
        )
        if variance <= 0:
            if size is None:
                return prob
            return np.full(size, prob, dtype=np.float32)

        elo_delta = self.predictor._probability_to_elo_delta(prob)
        noise_scale = np.sqrt(2) * variance
        if size is None:
            noisy_delta = elo_delta + float(np.random.normal(0, noise_scale))
            return float(self.predictor._elo_delta_to_probability(noisy_delta))

        noisy_delta = elo_delta + np.random.normal(0, noise_scale, size=size)
        return np.clip(
            1 / (1 + np.power(10, -noisy_delta / 400)),
            0.05,
            0.95,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 2: GPU-vectorized Swiss simulation
    # ------------------------------------------------------------------

    def simulate(self, round1_matchups: List[Tuple[str, str]], num_sims: int = 50000):
        """Run full Swiss simulation with GPU-accelerated random draws + multiprocessing.

        Returns:
            probs: dict[team] -> {"3-0": float, "qualified": float, "0-3": float}
            all_sims: list of dicts (for Pick'em optimizer compatibility)
        """
        t0 = time.time()

        import multiprocessing as mp

        # Pre-generate all random numbers on GPU (much faster than CPU)
        max_matches_per_sim = 40
        if self.backend == "gpu" and TORCH_AVAILABLE:
            rand_tensor = torch.rand(num_sims, max_matches_per_sim, device=self.device)
            rand_all = rand_tensor.cpu().numpy()
        else:
            rand_all = np.random.rand(num_sims, max_matches_per_sim).astype(np.float32)

        # 首轮对阵转成队伍下标。
        r1_pairs = [(self.team_to_idx[t1], self.team_to_idx[t2]) for t1, t2 in round1_matchups]
        team_seeds = {t: i for i, t in enumerate(self.teams)}

        worker_count = int(os.environ.get("SWISS_CPU_WORKERS", "0") or "0")
        if worker_count <= 0:
            worker_count = min(os.cpu_count() or 1, 16)
        worker_count = max(1, min(worker_count, num_sims))

        round_matchup_counts = {round_num: defaultdict(int) for round_num in range(1, 6)}
        round_bo_counts = {round_num: defaultdict(int) for round_num in range(1, 6)}

        if worker_count > 1 and os.name != "nt":
            chunk_size = (num_sims + worker_count - 1) // worker_count
            chunks = [
                (start, min(start + chunk_size, num_sims), r1_pairs, team_seeds)
                for start in range(0, num_sims, chunk_size)
            ]
            print(
                f"  [Swiss GPU] Swiss pairing workers: {worker_count}, chunks: {len(chunks)}",
                flush=True,
            )

            global _MP_SIMULATOR, _MP_RAND_ALL
            _MP_SIMULATOR = self
            _MP_RAND_ALL = rand_all

            records_all = np.empty((num_sims, self.num_teams, 2), dtype=np.int8)
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=worker_count) as pool:
                for done, (start, records_chunk, chunk_matchups, chunk_bos) in enumerate(
                    pool.imap_unordered(_run_swiss_chunk_worker, chunks),
                    1,
                ):
                    end = start + records_chunk.shape[0]
                    records_all[start:end] = records_chunk
                    for round_num, counter in chunk_matchups.items():
                        for matchup, count in counter.items():
                            round_matchup_counts[round_num][matchup] += count
                    for round_num, counter in chunk_bos.items():
                        for bo_format, count in counter.items():
                            round_bo_counts[round_num][bo_format] += count
                    print(
                        f"  [Swiss GPU] pairing chunk {done}/{len(chunks)} completed",
                        flush=True,
                    )

            _MP_SIMULATOR = None
            _MP_RAND_ALL = None
        else:
            records_all, round_matchup_counts, round_bo_counts = self._run_all_sims(
                r1_pairs, rand_all, num_sims, team_seeds
            )

        self.round_matchup_counts = round_matchup_counts
        self.round_bo_counts = round_bo_counts

        # ── Collect results (vectorized) ──
        n = self.num_teams
        wins = records_all[:, :, 0]   # (num_sims, n)
        losses = records_all[:, :, 1]  # (num_sims, n)

        # Compute probabilities using vectorized numpy
        all_sims_list = []

        # Vectorized counting
        is_30 = (wins == 3) & (losses == 0)      # (num_sims, n)
        is_qual = wins >= 3                        # (num_sims, n)
        is_03 = (losses >= 3) & (wins == 0)       # (num_sims, n)

        probs = {}
        for t_idx in range(n):
            team = self.teams[t_idx]
            probs[team] = {
                "3-0": float(is_30[:, t_idx].sum()) / num_sims,
                "qualified": float(is_qual[:, t_idx].sum()) / num_sims,
                "0-3": float(is_03[:, t_idx].sum()) / num_sims,
            }

        # Build all_sims_list for Pick'em optimizer
        for sim_idx in range(num_sims):
            sim_result = {"3-0": set(), "qualified": set(), "0-3": set()}
            for t_idx in range(n):
                team = self.teams[t_idx]
                if is_30[sim_idx, t_idx]:
                    sim_result["3-0"].add(team)
                    sim_result["qualified"].add(team)
                elif is_qual[sim_idx, t_idx]:
                    sim_result["qualified"].add(team)
                elif is_03[sim_idx, t_idx]:
                    sim_result["0-3"].add(team)
            all_sims_list.append(sim_result)

        elapsed = time.time() - t0
        print(f"  [Swiss GPU] {num_sims:,} simulations completed in {elapsed:.1f}s")
        return probs, all_sims_list

    def _run_all_sims(self, r1_pairs, rand_all, num_sims, team_seeds):
        """Core simulation loop - optimized with numpy operations."""
        n = self.num_teams

        round_matchup_counts = {round_num: defaultdict(int) for round_num in range(1, 6)}
        round_bo_counts = {round_num: defaultdict(int) for round_num in range(1, 6)}

        # W-L records: shape (num_sims, num_teams, 2)
        records = np.zeros((num_sims, n, 2), dtype=np.int8)
        # Match history bit matrix
        history = np.zeros((num_sims, n, n), dtype=np.bool_)
        match_counter = np.zeros(num_sims, dtype=np.int32)

        # Round 1: 固定对阵，向量化处理。
        for t1_idx, t2_idx in r1_pairs:
            matchup_key = tuple(sorted((self.teams[t1_idx], self.teams[t2_idx])))
            round_matchup_counts[1][matchup_key] += num_sims
            round_bo_counts[1]["BO1"] += num_sims

            prob = self._apply_match_format_variance(
                self.prob_bo1[t1_idx, t2_idx],
                "BO1",
                size=num_sims,
            )
            rand_vals = rand_all[np.arange(num_sims), match_counter]
            match_counter += 1

            t1_wins = rand_vals < prob
            records[t1_wins, t1_idx, 0] += 1
            records[~t1_wins, t2_idx, 0] += 1
            records[t1_wins, t2_idx, 1] += 1
            records[~t1_wins, t1_idx, 1] += 1
            history[:, t1_idx, t2_idx] = True
            history[:, t2_idx, t1_idx] = True

        # ── Rounds 2-5 ──
        for rnd in range(2, 6):
            active_mask = (records[:, :, 0] < 3) & (records[:, :, 1] < 3)
            self._simulate_round_batched(
                rnd,
                records,
                history,
                active_mask,
                rand_all,
                match_counter,
                team_seeds,
                round_matchup_counts,
                round_bo_counts,
            )

        return records, round_matchup_counts, round_bo_counts

    def _simulate_round_batched(
        self,
        rnd,
        records,
        history,
        active_mask,
        rand_all,
        match_counter,
        team_seeds,
        round_matchup_counts,
        round_bo_counts,
    ):
        """Simulate one round across all sims.

        Optimization: Group sims by their record-state signature to amortize
        the pairing computation. Many sims share identical record patterns
        (especially in early rounds), so we compute pairings once per group.
        """
        num_sims = records.shape[0]
        n = self.num_teams
        round_records = records.copy()
        round_history = history.copy()

        # Compute a compact state key per sim: tuple of (w, l) for each active team
        # Use record bytes as hash key for grouping
        # Each team's state is w*4+l (fits in 4 bits since max w=3, l=3)
        state_keys = (round_records[:, :, 0].astype(np.int16) * 4 + round_records[:, :, 1].astype(np.int16))
        # Convert to bytes for hashing
        state_bytes = state_keys.tobytes()
        row_size = n * 2  # int16 = 2 bytes per team

        # Group sims by identical record state
        state_groups = defaultdict(list)  # state_hash -> list of sim indices
        for sim_idx in range(num_sims):
            key = state_bytes[sim_idx * row_size: (sim_idx + 1) * row_size]
            state_groups[key].append(sim_idx)

        # 同一组战绩只算一次配对，再应用到该组所有模拟。
        for state_key, sim_indices in state_groups.items():
            representative = sim_indices[0]

            # Get active teams for this state
            active_indices = np.where(active_mask[representative])[0]
            if len(active_indices) < 2:
                continue

            # Group active teams by (W, L)
            groups = defaultdict(list)
            for t_idx in active_indices:
                w = int(round_records[representative, t_idx, 0])
                l = int(round_records[representative, t_idx, 1])
                groups[(w, l)].append(int(t_idx))

            # For each W-L group, compute Buchholz ordering
            # NOTE: Buchholz depends on opponent records AND match history,
            # which may differ across sims in this group even with same records.
            # For speed, we use the representative's history for ordering
            # (acceptable approximation - Buchholz ordering is a tiebreaker).
            all_pairs_for_state = []
            for key, grp_indices in groups.items():
                if len(grp_indices) < 2:
                    continue

                w_count, l_count = key
                is_elim_adv = w_count == 2 or l_count == 2
                prob_matrix = self.prob_bo3 if is_elim_adv else self.prob_bo1
                match_type = "BO3" if is_elim_adv else "BO1"

                # For each sim in group, compute individual pairings
                # (history differs so rematches differ)
                all_pairs_for_state.append((grp_indices, prob_matrix, rnd, match_type))

            # Resolve matches for each sim in this group
            for grp_indices, prob_matrix, round_num, match_type in all_pairs_for_state:
                for sim_idx in sim_indices:
                    # Compute Buchholz sort for this specific sim
                    buchholz_scores = np.zeros(len(grp_indices), dtype=np.float32)
                    for k, t_idx in enumerate(grp_indices):
                        opps = np.where(round_history[sim_idx, t_idx])[0]
                        buchholz_scores[k] = -sum(
                            int(round_records[sim_idx, opp, 0]) - int(round_records[sim_idx, opp, 1])
                            for opp in opps
                        )

                    # Sort by Buchholz then seed
                    sort_keys = [(buchholz_scores[k], team_seeds.get(self.teams[grp_indices[k]], 999), k)
                                 for k in range(len(grp_indices))]
                    sort_keys.sort()
                    ordered = [grp_indices[k] for _, _, k in sort_keys]

                    # 避免重复交手。
                    pairs = self._pair_teams(ordered, round_history[sim_idx], round_num)

                    # Resolve each match
                    for t1_idx, t2_idx in pairs:
                        matchup_key = tuple(sorted((self.teams[t1_idx], self.teams[t2_idx])))
                        round_matchup_counts[rnd][matchup_key] += 1
                        round_bo_counts[rnd][match_type] += 1

                        mc = int(match_counter[sim_idx])
                        if mc >= rand_all.shape[1]:
                            rand_val = np.random.rand()
                        else:
                            rand_val = rand_all[sim_idx, mc]
                        match_counter[sim_idx] += 1

                        prob = self._apply_match_format_variance(
                            prob_matrix[t1_idx, t2_idx],
                            match_type,
                        )
                        if rand_val < prob:
                            records[sim_idx, t1_idx, 0] += 1
                            records[sim_idx, t2_idx, 1] += 1
                        else:
                            records[sim_idx, t2_idx, 0] += 1
                            records[sim_idx, t1_idx, 1] += 1
                        history[sim_idx, t1_idx, t2_idx] = True
                        history[sim_idx, t2_idx, t1_idx] = True

    def _pair_teams(self, ordered: List[int], history_sim, rnd: int) -> List[Tuple[int, int]]:
        """Pair teams avoiding rematches. Uses top-vs-bottom for rounds 2-3,
        priority patterns for rounds 4-5."""
        if rnd <= 3:
            return self._pair_r23(ordered, history_sim)
        else:
            return self._pair_r45(ordered, history_sim)

    def _pair_r23(self, ordered, history_sim):
        remaining = list(ordered)
        pairs = []
        while len(remaining) >= 2:
            t1 = remaining.pop(0)
            matched = False
            for i in range(len(remaining) - 1, -1, -1):
                if not history_sim[t1, remaining[i]]:
                    pairs.append((t1, remaining.pop(i)))
                    matched = True
                    break
            if not matched and remaining:
                pairs.append((t1, remaining.pop()))
        return pairs

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

    def _pair_r45(self, ordered, history_sim):
        for pattern in self.PAIRING_PRIORITY:
            valid = True
            test = []
            for i1, i2 in pattern:
                if i1 >= len(ordered) or i2 >= len(ordered):
                    valid = False
                    break
                if history_sim[ordered[i1], ordered[i2]]:
                    valid = False
                    break
                test.append((ordered[i1], ordered[i2]))
            if valid:
                return test
        # Fallback
        return [(ordered[i1], ordered[i2]) for i1, i2 in self.PAIRING_PRIORITY[0]
                if i1 < len(ordered) and i2 < len(ordered)]
