"""
Pick'em brute-force optimizer with GPU/CPU tensor acceleration.

Exhaustively evaluates all C(N,6)·C(N-6,2)·C(N-8,2) Pick'em combinations
against Monte Carlo simulation results using batched matrix multiplication.

GPU path uses PyTorch; falls back to NumPy on CPU when torch is unavailable.
"""

from __future__ import annotations

import itertools
import time
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class PickemOptimizer:
    """Tensor-accelerated Pick'em brute-force search."""

    def __init__(
        self,
        teams: List[str],
        all_simulations: list,
        use_gpu: bool = True,
        batch_size: int = 20000,
        device_id: Optional[int] = None,
    ):
        self.teams = list(teams)
        self.num_teams = len(self.teams)
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        self.all_simulations = all_simulations
        self.num_sims = len(all_simulations)
        self.batch_size = batch_size

        if _torch_available():
            import torch

            if use_gpu and torch.cuda.is_available():
                gpu_id = device_id if device_id is not None else 0
                self.device = torch.device(f"cuda:{gpu_id}")
                self.backend = "torch_gpu"
            else:
                self.device = torch.device("cpu")
                self.backend = "torch_cpu"
        else:
            self.backend = "numpy"

        self._prepare_matrices()

    # ------------------------------------------------------------------
    # Tensor preparation
    # ------------------------------------------------------------------

    def _prepare_matrices(self):
        """Convert simulation list into (num_teams, num_sims) float matrices."""
        ns = self.num_sims
        nt = self.num_teams

        mat_30 = np.zeros((ns, nt), dtype=np.float32)
        mat_adv = np.zeros((ns, nt), dtype=np.float32)
        mat_03 = np.zeros((ns, nt), dtype=np.float32)

        for si, sim in enumerate(self.all_simulations):
            for team in sim["3-0"]:
                idx = self.team_to_idx.get(team)
                if idx is not None:
                    mat_30[si, idx] = 1.0
            for team in sim["0-3"]:
                idx = self.team_to_idx.get(team)
                if idx is not None:
                    mat_03[si, idx] = 1.0
            qualified_non_30 = sim["qualified"] - sim["3-0"]
            for team in qualified_non_30:
                idx = self.team_to_idx.get(team)
                if idx is not None:
                    mat_adv[si, idx] = 1.0

        # Transpose → (num_teams, num_sims) for matmul: one-hot @ matrix
        if self.backend.startswith("torch"):
            import torch

            self.mat_30 = torch.from_numpy(mat_30.T).to(self.device)
            self.mat_adv = torch.from_numpy(mat_adv.T).to(self.device)
            self.mat_03 = torch.from_numpy(mat_03.T).to(self.device)
        else:
            self.mat_30 = mat_30.T.copy()
            self.mat_adv = mat_adv.T.copy()
            self.mat_03 = mat_03.T.copy()

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def _evaluate_batch_torch(self, buf_adv, buf_30, buf_03):
        import torch

        bs = len(buf_adv)
        nt = self.num_teams

        t_adv = torch.tensor(buf_adv, dtype=torch.long, device=self.device)
        t_30 = torch.tensor(buf_30, dtype=torch.long, device=self.device)
        t_03 = torch.tensor(buf_03, dtype=torch.long, device=self.device)

        oh_adv = torch.zeros((bs, nt), device=self.device)
        oh_adv.scatter_(1, t_adv, 1.0)
        oh_30 = torch.zeros((bs, nt), device=self.device)
        oh_30.scatter_(1, t_30, 1.0)
        oh_03 = torch.zeros((bs, nt), device=self.device)
        oh_03.scatter_(1, t_03, 1.0)

        scores = (
            torch.mm(oh_adv, self.mat_adv)
            + torch.mm(oh_30, self.mat_30)
            + torch.mm(oh_03, self.mat_03)
        )
        pass_rates = (scores >= 5.0).float().mean(dim=1)
        max_rate, max_idx = torch.max(pass_rates, dim=0)
        return max_rate.item(), max_idx.item()

    def _evaluate_batch_numpy(self, buf_adv, buf_30, buf_03):
        bs = len(buf_adv)
        nt = self.num_teams

        oh_adv = np.zeros((bs, nt), dtype=np.float32)
        oh_30 = np.zeros((bs, nt), dtype=np.float32)
        oh_03 = np.zeros((bs, nt), dtype=np.float32)

        for i, indices in enumerate(buf_adv):
            for j in indices:
                oh_adv[i, j] = 1.0
        for i, indices in enumerate(buf_30):
            for j in indices:
                oh_30[i, j] = 1.0
        for i, indices in enumerate(buf_03):
            for j in indices:
                oh_03[i, j] = 1.0

        scores = oh_adv @ self.mat_adv + oh_30 @ self.mat_30 + oh_03 @ self.mat_03
        pass_rates = (scores >= 5.0).astype(np.float32).mean(axis=1)
        max_idx = int(np.argmax(pass_rates))
        return float(pass_rates[max_idx]), max_idx

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def run(self) -> Tuple[Optional[Dict], float]:
        total = comb(self.num_teams, 6) * comb(self.num_teams - 6, 2) * comb(self.num_teams - 8, 2)
        print(f"\n[Pick'em Brute-Force] backend={self.backend}")
        print(f"  Teams: {self.num_teams}, Simulations: {self.num_sims:,}")
        print(f"  Total combinations: {total:,}")

        evaluate = (
            self._evaluate_batch_torch
            if self.backend.startswith("torch")
            else self._evaluate_batch_numpy
        )

        all_idx = list(range(self.num_teams))
        best_rate = -1.0
        best_pred: Optional[Dict] = None

        buf_adv: list = []
        buf_30: list = []
        buf_03: list = []
        processed = 0
        t0 = time.time()
        last_print = 0

        for adv_combo in itertools.combinations(all_idx, 6):
            adv_set = set(adv_combo)
            remaining = [i for i in all_idx if i not in adv_set]

            for t30_combo in itertools.combinations(remaining, 2):
                t30_set = set(t30_combo)
                remaining2 = [i for i in remaining if i not in t30_set]

                for t03_combo in itertools.combinations(remaining2, 2):
                    buf_adv.append(adv_combo)
                    buf_30.append(t30_combo)
                    buf_03.append(t03_combo)

                    if len(buf_adv) >= self.batch_size:
                        batch_rate, batch_idx = evaluate(buf_adv, buf_30, buf_03)
                        if batch_rate > best_rate:
                            best_rate = batch_rate
                            best_pred = {
                                "3-0": [self.teams[i] for i in buf_30[batch_idx]],
                                "advances": [self.teams[i] for i in buf_adv[batch_idx]],
                                "0-3": [self.teams[i] for i in buf_03[batch_idx]],
                            }
                        processed += len(buf_adv)
                        if processed - last_print >= 500_000:
                            last_print = processed
                            elapsed = time.time() - t0
                            speed = processed / max(elapsed, 0.001)
                            eta = (total - processed) / max(speed, 1)
                            print(
                                f"  {processed:,}/{total:,} ({processed / total * 100:.1f}%) "
                                f"best={best_rate:.4%}  ETA={eta:.0f}s"
                            )
                        buf_adv, buf_30, buf_03 = [], [], []

        if buf_adv:
            batch_rate, batch_idx = evaluate(buf_adv, buf_30, buf_03)
            if batch_rate > best_rate:
                best_rate = batch_rate
                best_pred = {
                    "3-0": [self.teams[i] for i in buf_30[batch_idx]],
                    "advances": [self.teams[i] for i in buf_adv[batch_idx]],
                    "0-3": [self.teams[i] for i in buf_03[batch_idx]],
                }
            processed += len(buf_adv)

        elapsed = time.time() - t0
        print(f"  Done: {processed:,} combos in {elapsed:.1f}s ({processed / max(elapsed, 0.001):,.0f} combo/s)")
        print(f"  Best success rate: {best_rate:.4%}")

        return best_pred, best_rate
