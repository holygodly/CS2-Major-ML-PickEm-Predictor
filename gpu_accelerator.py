"""
GPU 设备选择和批量参数估算

功能：
1. 检测所有 GPU，选择可用显存最大的那块
2. 按可用显存估算 batch_size / 模拟次数
3. 多 GPU 环境下跳过被占用的卡
4. 无 GPU 时回退 CPU
"""
import multiprocessing
import os
import subprocess
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GPUInfo:
    """单块 GPU 的状态信息"""
    index: int
    name: str
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    utilization_pct: float  # GPU 利用率 %
    temperature: int = 0

    @property
    def free_ratio(self) -> float:
        return self.free_memory_mb / self.total_memory_mb if self.total_memory_mb > 0 else 0


def query_all_gpus() -> List[GPUInfo]:
    """通过 nvidia-smi 查询所有 GPU 状态，失败返回空列表。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            gpus.append(GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                total_memory_mb=float(parts[2]),
                used_memory_mb=float(parts[3]),
                free_memory_mb=float(parts[4]),
                utilization_pct=float(parts[5]),
                temperature=int(parts[6]),
            ))
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def select_best_gpu(gpus: List[GPUInfo], min_free_mb: float = 2000) -> Optional[GPUInfo]:
    """选择可用显存最大且空闲率 >20% 的 GPU。
    
    Args:
        gpus: GPU 列表
        min_free_mb: 最低可用显存阈值 (MB)，低于此值的 GPU 视为不可用
    """
    candidates = [g for g in gpus if g.free_memory_mb >= min_free_mb and g.free_ratio > 0.2]
    if not candidates:
        return None
    # 按可用显存降序，利用率升序
    candidates.sort(key=lambda g: (-g.free_memory_mb, g.utilization_pct))
    return candidates[0]


class DeviceManager:
    """选择运行设备，并估算模拟参数。"""

    def __init__(self, config=None):
        self.config = config
        self.cuda_available = False
        self.cuda_device_id = 0
        self.cuda_device_name = "N/A"
        self.gpu_info: Optional[GPUInfo] = None
        self.cpu_cores = multiprocessing.cpu_count()

        # 探测 GPU，尊重 CUDA_VISIBLE_DEVICES
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpus = query_all_gpus()
        if gpus and visible_devices is not None:
            # 只保留 CUDA_VISIBLE_DEVICES 指定的物理 GPU
            visible_ids = [int(x.strip()) for x in visible_devices.split(",") if x.strip().isdigit()]
            gpus = [g for g in gpus if g.index in visible_ids]

        if gpus:
            best = select_best_gpu(gpus)
            if best:
                self.gpu_info = best
                # CUDA_VISIBLE_DEVICES 下，逻辑 ID 是在可见列表中的位置
                if visible_devices is not None:
                    visible_ids = [int(x.strip()) for x in visible_devices.split(",") if x.strip().isdigit()]
                    self.cuda_device_id = visible_ids.index(best.index) if best.index in visible_ids else 0
                else:
                    self.cuda_device_id = best.index
                self.cuda_device_name = best.name
                # 验证 XGBoost 能否实际使用该 GPU
                self.cuda_available = self._verify_xgboost_cuda(self.cuda_device_id)

        if self.cuda_available:
            print(f"  🚀 已选择 GPU {self.cuda_device_id}: {self.cuda_device_name}")
            print(f"     显存: {self.gpu_info.free_memory_mb:.0f}/{self.gpu_info.total_memory_mb:.0f} MB 可用"
                  f" | 利用率: {self.gpu_info.utilization_pct:.0f}%")
            if len(gpus) > 1:
                print(f"     (共检测到 {len(gpus)} 块GPU，已选择可用显存最大的那块)")
        else:
            print(f"  💻 使用 CPU ({self.cpu_cores} 核)")
            if gpus:
                print(f"     (检测到 {len(gpus)} 块GPU，但均不可用或显存不足)")

    def _verify_xgboost_cuda(self, device_id: int) -> bool:
        """验证 XGBoost 能否在指定 GPU 上运行。"""
        try:
            import xgboost as xgb
            import numpy as np
            test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            test_label = np.array([0, 1], dtype=np.float32)
            params = {"device": f"cuda:{device_id}", "objective": "binary:logistic"}
            dtrain = xgb.DMatrix(test_data, label=test_label)
            xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
            return True
        except Exception:
            return False

    @property
    def device_str(self) -> str:
        """返回 'cuda:N' 或 'cpu'"""
        if self.cuda_available:
            return f"cuda:{self.cuda_device_id}"
        return "cpu"

    def get_xgboost_params(self) -> dict:
        """获取 XGBoost 设备参数"""
        return {"device": self.device_str}

    def get_torch_device(self):
        """获取 PyTorch device 对象"""
        import torch
        if self.cuda_available and torch.cuda.is_available():
            return torch.device(f"cuda:{self.cuda_device_id}")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # 参数估算
    # ------------------------------------------------------------------

    def recommend_sim_count(self, base: int = 5000, max_cap: int = 500000) -> int:
        """按 GPU 显存估算 Swiss 模拟次数。
        
        模拟本身在 CPU，但结果矩阵在 GPU 上做 tensor 计算。
        16 队 × N_sims × float32 ≈ N_sims × 192 bytes
        保守地用可用显存的 30% 来估算能放多少 sims。
        """
        if not self.gpu_info:
            return base

        free_bytes = self.gpu_info.free_memory_mb * 1024 * 1024
        # 3 个矩阵 (mat_30, mat_adv, mat_03): 各 N×16 float32 = N×64 bytes
        bytes_per_sim = 16 * 4 * 3  # 192 bytes per sim
        budget = free_bytes * 0.3  # 用 30% 显存
        max_sims = int(budget / bytes_per_sim)
        recommended = min(max(max_sims, base), max_cap)
        return recommended

    def recommend_playoff_sim_count(self, base: int = 500000, max_cap: int = 50000000) -> int:
        """按 GPU 显存估算 Playoff bracket 向量化模拟次数。

        Playoff GPU 模拟的显存布局：
        - rand tensor: N × 25 × float32 = 100N bytes
        - 常量 _one/_two: 2 × N × int64 = 16N bytes
        - 临时 score/prev/active: ~50N bytes (峰值)
        - QF/SF/Final 结果 masks: ~10 × N × bool = 10N bytes
        总峰值约 200N bytes。用可用显存的 40% 来估算。
        """
        if not self.gpu_info:
            return base

        free_bytes = self.gpu_info.free_memory_mb * 1024 * 1024
        bytes_per_sim = 200  # 保守估计
        budget = free_bytes * 0.4
        max_sims = int(budget / bytes_per_sim)
        recommended = min(max(max_sims, base), max_cap)
        return recommended

    def recommend_batch_size(self, num_sims: int = 50000, max_cap: int = 500000) -> int:
        """按 GPU 显存估算 Pick'em 优化器 batch_size。
        
        主要运算: torch.mm(one_hot[batch_size, 16], sim_matrix[16, num_sims])
        输出: (batch_size, num_sims) float32，共3个矩阵同时存在
        内存 = batch_size × num_sims × 4 bytes × 3
        用可用显存的 60% 来估算（留余量给 PyTorch 开销）。
        """
        if not self.gpu_info:
            return 5000  # CPU fallback

        free_bytes = self.gpu_info.free_memory_mb * 1024 * 1024
        # 3 个输出矩阵 + one_hot 输入 ≈ batch × num_sims × 4 × 3.5
        bytes_per_combo = num_sims * 4 * 3.5
        budget = free_bytes * 0.6
        max_batch = int(budget / bytes_per_combo)
        recommended = min(max(max_batch, 1000), max_cap)
        return recommended

    def get_optimal_params(self) -> dict:
        """返回设备和批量参数。"""
        sim_count = self.recommend_sim_count()
        batch_size = self.recommend_batch_size(num_sims=sim_count)
        return {
            "device": self.device_str,
            "num_simulations": sim_count,
            "batch_size": batch_size,
            "gpu_name": self.cuda_device_name if self.cuda_available else "CPU",
            "free_memory_mb": self.gpu_info.free_memory_mb if self.gpu_info else 0,
        }

    def print_config_summary(self):
        """打印最终使用的配置摘要。"""
        params = self.get_optimal_params()
        print(f"\n  {'─' * 50}")
        print(f"  设备: {params['gpu_name']} ({params['device']})")
        if params['free_memory_mb']:
            print(f"  可用显存: {params['free_memory_mb']:.0f} MB")
        print(f"  模拟次数: {params['num_simulations']:,}")
        print(f"  batch_size: {params['batch_size']:,}")
        print(f"  {'─' * 50}\n")
