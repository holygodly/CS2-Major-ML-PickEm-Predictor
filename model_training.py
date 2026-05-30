"""
模型训练：XGBoost 地图胜率预测器

流程：
1. 加载特征数据
2. 按时间顺序划分训练集和验证集
3. 训练 XGBoost
4. 评估并保存模型
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from gpu_accelerator import DeviceManager

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class ModelTrainer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.xgb_model = None
        self.cutoff_date = None

        self.device_mgr = DeviceManager(config=self.config)
        Path("models").mkdir(exist_ok=True)

    def _resolve_time_split(self, map_data):
        data_config = self.config.get("data", {})
        explicit_cutoff = data_config.get("train_cutoff_date")

        if explicit_cutoff:
            return pd.Timestamp(explicit_cutoff)

        test_ratio = float(data_config.get("train_test_split", 0.15))
        if not 0 < test_ratio < 1:
            raise ValueError(f"train_test_split 必须在 0 和 1 之间，当前为: {test_ratio}")

        unique_dates = sorted(map_data["date"].dt.normalize().unique())
        if len(unique_dates) < 2:
            raise ValueError("可用日期不足 2 个，无法按时间切分训练集和验证集")

        cutoff_index = int(len(unique_dates) * (1 - test_ratio))
        cutoff_index = min(max(cutoff_index, 1), len(unique_dates) - 1)
        return pd.Timestamp(unique_dates[cutoff_index])

    def load_data(self):
        """加载特征数据，并按时间顺序切分训练集和验证集。"""
        print("\n[1/6] 加载特征数据...")

        map_data = pd.read_csv("data/map_level_dataset.csv")
        self.X = pd.read_csv("data/X_features.csv")
        self.y = pd.read_csv("data/y_labels.csv")["winner"]

        if not (len(map_data) == len(self.X) == len(self.y)):
            raise ValueError(
                "数据长度不一致："
                f"map_level_dataset={len(map_data)}, X={len(self.X)}, y={len(self.y)}"
            )

        print(f"  ✓ 特征矩阵: {self.X.shape}")
        print(f"  ✓ 标签分布: {self.y.value_counts().to_dict()}")

        map_data["date"] = pd.to_datetime(map_data["date"])
        self.cutoff_date = self._resolve_time_split(map_data)

        train_mask = map_data["date"] < self.cutoff_date
        test_mask = map_data["date"] >= self.cutoff_date

        # 测试集只保留原始行（非增强），避免同一比赛被评估两次导致指标虚高
        is_augmented = map_data.get("is_augmented", pd.Series(0, index=map_data.index))
        test_mask = test_mask & (is_augmented == 0)

        self.X_train = self.X.loc[train_mask].reset_index(drop=True)
        self.X_test = self.X.loc[test_mask].reset_index(drop=True)
        self.y_train = self.y.loc[train_mask].reset_index(drop=True)
        self.y_test = self.y.loc[test_mask].reset_index(drop=True)

        # 训练样本时间权重：近期比赛权重高，老比赛衰减
        half_life = self.config.get("data", {}).get("sample_weight_half_life_days", 60)
        train_dates_series = map_data.loc[train_mask, "date"]
        ref_date = train_dates_series.max()
        days_ago = (ref_date - train_dates_series).dt.total_seconds() / 86400.0
        self.train_weights = np.exp(-days_ago.values * np.log(2) / half_life).astype(np.float32)
        print(f"  ✓ 样本时间权重: half_life={half_life}d, "
              f"min={self.train_weights.min():.4f}, max={self.train_weights.max():.4f}")

        # BO1 样本上采样:数据集里 BO1 只占 ~2.6%,但 Swiss Round 1 全是 BO1。
        # 给 BO1 样本 3× 权重,让模型在 BO1 分布上学到的东西更接近 BO3。
        bo1_multiplier = float(self.config.get("data", {}).get("bo1_sample_weight_multiplier", 3.0))
        if bo1_multiplier != 1.0:
            train_match_types = map_data.loc[train_mask, "match_type"].values
            bo1_mask = (train_match_types == "BO1")
            self.train_weights[bo1_mask] *= bo1_multiplier
            n_bo1 = int(bo1_mask.sum())
            print(f"  ✓ BO1 权重 ×{bo1_multiplier}: {n_bo1} 个 BO1 训练样本被上采样")

        if self.X_train.empty or self.X_test.empty:
            raise ValueError(
                f"时间切分后训练集或验证集为空，cutoff={self.cutoff_date.date()}，"
                f"train={len(self.X_train)}，test={len(self.X_test)}"
            )

        train_dates = map_data.loc[train_mask, "date"]
        test_dates = map_data.loc[test_mask, "date"]

        print(f"\n  按日期划分（分界点: {self.cutoff_date.date()}）:")
        print(
            f"  ✓ 训练集: {self.X_train.shape[0]} 样本 "
            f"({train_dates.min().date()} 到 {train_dates.max().date()})"
        )
        print(
            f"  ✓ 验证集: {self.X_test.shape[0]} 样本 "
            f"({test_dates.min().date()} 到 {test_dates.max().date()})"
        )
        print(f"  ✓ 训练集标签: {self.y_train.value_counts().to_dict()}")
        print(f"  ✓ 验证集标签: {self.y_test.value_counts().to_dict()}")

    def select_features(self):
        """用初步模型做特征选择，去掉低重要性噪声特征。"""
        print("\n[2/6] 自动特征选择...")

        device_params = self.device_mgr.get_xgboost_params()

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train, weight=self.train_weights)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        # 快速训练一个初步模型（少量迭代）
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.2,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "seed": 42,
            **device_params,
        }
        prelim_model = xgb.train(
            params, dtrain, num_boost_round=100,
            evals=[(dtest, "test")],
            early_stopping_rounds=10,
            verbose_eval=0,
        )

        # 获取特征重要性并排序
        importance = prelim_model.get_score(importance_type="gain")
        feature_names = self.X.columns.tolist()

        scored = {}
        for key, value in importance.items():
            if key in feature_names:
                scored[key] = value
            elif key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if 0 <= idx < len(feature_names):
                    scored[feature_names[idx]] = value

        if not scored:
            print("  ! 初步模型无特征重要性，跳过选择")
            return

        # 按重要性排序
        sorted_features = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        total_importance = sum(v for _, v in sorted_features)

        # 保留累积重要性达到 95% 的特征，或 top-30，取多的那个
        cumulative = 0.0
        keep_by_importance = []
        for feat, imp in sorted_features:
            keep_by_importance.append(feat)
            cumulative += imp
            if cumulative >= total_importance * 0.95:
                break

        # 取 max(累积95%的特征数, 30, 15) 作为保留数
        min_keep = max(len(keep_by_importance), min(30, len(sorted_features)))
        keep_features = [f for f, _ in sorted_features[:min_keep]]

        removed = set(feature_names) - set(keep_features)
        if removed:
            print(f"  原始特征: {len(feature_names)} 维")
            print(f"  保留特征: {len(keep_features)} 维 (累积 95% 重要性)")
            print(f"  移除特征: {sorted(removed)}")

            self.X = self.X[keep_features]
            self.X_train = self.X_train[keep_features]
            self.X_test = self.X_test[keep_features]
        else:
            print(f"  所有 {len(feature_names)} 个特征均有效，无需移除")

    def train_xgboost(self):
        """训练 XGBoost 模型（参数针对小数据集优化）。"""
        print("\n[3/6] 训练 XGBoost 模型...")

        device_params = self.device_mgr.get_xgboost_params()
        xgb_params = self.config["models"]["xgboost"]

        # 根据样本量自动调整防过拟合参数
        n_samples = len(self.X_train)
        n_features = self.X_train.shape[1]
        auto_max_depth = min(xgb_params["max_depth"], max(3, int(np.log2(n_samples / n_features))))
        auto_min_child = max(xgb_params["min_child_weight"], min(15, int(n_samples * 0.003)))

        if auto_max_depth != xgb_params["max_depth"] or auto_min_child != xgb_params["min_child_weight"]:
            print(f"  [自动调参] max_depth: {xgb_params['max_depth']} → {auto_max_depth} "
                  f"(样本/特征比={n_samples/n_features:.0f})")
            print(f"  [自动调参] min_child_weight: {xgb_params['min_child_weight']} → {auto_min_child}")

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train, weight=self.train_weights)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": auto_max_depth,
            "learning_rate": xgb_params["learning_rate"],
            "subsample": xgb_params["subsample"],
            "colsample_bytree": xgb_params["colsample_bytree"],
            "min_child_weight": auto_min_child,
            "gamma": xgb_params["gamma"],
            "reg_alpha": xgb_params["reg_alpha"],
            "reg_lambda": xgb_params["reg_lambda"],
            "seed": 42,
            **device_params,
        }

        evals = [(dtrain, "train"), (dtest, "test")]
        # 捕获每轮 train/test logloss + AUC,后面画 learning curve
        self._evals_result = {}
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=xgb_params["n_estimators"],
            evals=evals,
            evals_result=self._evals_result,
            early_stopping_rounds=20,
            verbose_eval=50,
        )

        print(f"  ✓ 选中迭代: {self.xgb_model.best_iteration}")
        print(f"  ✓ 测试 AUC: {self.xgb_model.best_score:.4f}")

        # 保存实际使用的训练参数，供校准阶段复用
        self._train_params = dict(params)

    def calibrate_model(self):
        """用训练集交叉验证 OOF 预测拟合 isotonic 校准器。"""
        print("\n[3.5/6] 概率校准 (Isotonic)...")
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import KFold
        import pickle

        oof_preds = np.zeros(len(self.X_train))

        # 复用主模型的实际训练参数
        base_params = {k: v for k, v in self._train_params.items()
                       if k != "eval_metric"}
        base_params["eval_metric"] = "logloss"

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(self.X_train)):
            X_tr = self.X_train.iloc[tr_idx]
            y_tr = self.y_train.iloc[tr_idx]
            w_tr = self.train_weights[tr_idx] if self.train_weights is not None else None
            X_val = self.X_train.iloc[val_idx]

            dtrain_fold = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
            dval_fold = xgb.DMatrix(X_val)

            fold_params = dict(base_params)
            fold_params["seed"] = 42 + fold_idx
            fold_model = xgb.train(
                fold_params, dtrain_fold,
                num_boost_round=self.xgb_model.best_iteration + 1,
                verbose_eval=0,
            )
            oof_preds[val_idx] = fold_model.predict(dval_fold)

        self.calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
        self.calibrator.fit(oof_preds, self.y_train.values)

        calibrated_oof = self.calibrator.transform(oof_preds)
        raw_brier = brier_score_loss(self.y_train, oof_preds)
        cal_brier = brier_score_loss(self.y_train, calibrated_oof)
        print(f"  OOF Brier (raw):        {raw_brier:.4f}")
        print(f"  OOF Brier (calibrated): {cal_brier:.4f}")

        cal_path = Path("models/calibrator.pkl")
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with cal_path.open("wb") as f:
            pickle.dump(self.calibrator, f)
        print(f"  ✓ 校准器已保存: {cal_path}")

    def evaluate_model(self):
        """评估模型。"""
        print("\n[4/6] 评估模型...")

        dtest = xgb.DMatrix(self.X_test)
        y_pred_proba = self.xgb_model.predict(dtest)

        # 如果有校准器，同时显示校准后指标
        y_pred_calibrated = None
        if hasattr(self, 'calibrator') and self.calibrator is not None:
            y_pred_calibrated = self.calibrator.transform(y_pred_proba)

        y_pred = (y_pred_proba > 0.5).astype(int)

        accuracy = accuracy_score(self.y_test, y_pred)
        logloss = log_loss(self.y_test, y_pred_proba)
        brier = brier_score_loss(self.y_test, y_pred_proba)

        try:
            auc = roc_auc_score(self.y_test, y_pred_proba)
        except ValueError:
            auc = float("nan")

        print("\n测试集指标:")
        print(f"  准确率:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  对数损失:  {logloss:.4f}")
        print(f"  AUC:       {auc:.4f}" if pd.notna(auc) else "  AUC:       N/A")
        print(f"  Brier分数: {brier:.4f}")

        if y_pred_calibrated is not None:
            cal_acc = accuracy_score(self.y_test, (y_pred_calibrated > 0.5).astype(int))
            cal_brier = brier_score_loss(self.y_test, y_pred_calibrated)
            print(f"\n  [校准后] 准确率: {cal_acc:.4f} ({cal_acc * 100:.2f}%)")
            print(f"  [校准后] Brier:  {cal_brier:.4f}")

        # 概率校准检查
        try:
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(self.y_test, y_pred_proba, n_bins=8)
            print("\n  概率校准 (predicted → actual):")
            miscal = []
            for pred, true in zip(prob_pred, prob_true):
                arrow = "✓" if abs(pred - true) < 0.05 else "✗"
                print(f"    {pred:.2f} → {true:.2f}  {arrow}")
                miscal.append(abs(pred - true))
            avg_miscal = sum(miscal) / len(miscal) if miscal else 0
            print(f"  平均校准误差: {avg_miscal:.3f}")
            if avg_miscal > 0.05:
                print("  ⚠ 校准偏差较大，建议后接 isotonic 校准")
        except Exception as e:
            print(f"  ! 校准检查失败: {e}")

        results = {
            "accuracy": float(accuracy),
            "log_loss": float(logloss),
            "roc_auc": None if pd.isna(auc) else float(auc),
            "brier_score": float(brier),
            "test_samples": int(len(self.y_test)),
            "best_iteration": int(self.xgb_model.best_iteration),
            "cutoff_date": self.cutoff_date.strftime("%Y-%m-%d"),
        }

        with open("models/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def plot_evaluation_charts(self):
        """生成特征重要性、校准曲线和 ROC 曲线。"""
        print("\n[可视化] 生成评估图表...")

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        dtest = xgb.DMatrix(self.X_test)
        y_pred_proba = self.xgb_model.predict(dtest)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ── 1. Feature Importance (Top 15) ──
        ax = axes[0]
        importance = self.xgb_model.get_score(importance_type="gain")
        feature_names = self.X.columns.tolist()
        imp_items = []
        for key, value in importance.items():
            if key in feature_names:
                imp_items.append((key, value))
            elif key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if 0 <= idx < len(feature_names):
                    imp_items.append((feature_names[idx], value))
        imp_items.sort(key=lambda x: x[1], reverse=True)
        top_n = imp_items[:15]
        if top_n:
            names, vals = zip(*reversed(top_n))
            ax.barh(range(len(names)), vals, color="#2196F3")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("Gain")
            ax.set_title("Top 15 Feature Importance")

        # ── 2. Calibration Curve ──
        ax = axes[1]
        try:
            n_bins = min(10, len(self.y_test) // 5)  # 每 bin 至少 5 个样本
            n_bins = max(n_bins, 3)
            prob_true, prob_pred = calibration_curve(self.y_test, y_pred_proba, n_bins=n_bins)
            ax.plot(prob_pred, prob_true, "s-", color="#FF5722", label="Model")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.legend(loc="lower right")
        except Exception as e:
            ax.text(0.5, 0.5, f"N/A\n({e})", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title("Calibration Curve")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # ── 3. ROC Curve ──
        ax = axes[2]
        try:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_val = roc_auc_score(self.y_test, y_pred_proba)
            ax.plot(fpr, tpr, color="#4CAF50", lw=2, label=f"AUC = {auc_val:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.legend(loc="lower right")
        except Exception as e:
            ax.text(0.5, 0.5, f"N/A\n({e})", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = "models/evaluation_charts.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 已保存: {out_path}")

    def plot_training_diagnostics(self):
        """训练健康度诊断图(2×2):学习曲线 + CV 折分布 + 预测分布 + 测试集混淆矩阵。

        看这张图时主要判断:
          - 学习曲线 (train vs test) 间距大 → 过拟合
          - test logloss 早早就开始上升 → 严重过拟合
          - CV 各折方差大 → 模型不稳定/数据不一致
          - 预测概率分布扁平 → 模型分不清类别
        """
        print("\n[可视化] 生成训练诊断图...")

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ── 1) Learning curve: AUC 随 boosting round ──
        ax = axes[0, 0]
        if hasattr(self, "_evals_result") and self._evals_result:
            train_auc = self._evals_result.get("train", {}).get("auc", [])
            test_auc = self._evals_result.get("test", {}).get("auc", [])
            rounds = list(range(1, len(train_auc) + 1))
            ax.plot(rounds, train_auc, color="#2196F3", linewidth=2, label="Train AUC")
            ax.plot(rounds, test_auc, color="#FF5722", linewidth=2, label="Test AUC")
            # 标注 early stopping 选中的迭代轮数
            best_iter = int(self.xgb_model.best_iteration) + 1
            if 1 <= best_iter <= len(test_auc):
                ax.axvline(best_iter, color="#4CAF50", linestyle="--", alpha=0.6,
                           label=f"Best iter = {best_iter}")
                ax.scatter([best_iter], [test_auc[best_iter - 1]], color="#4CAF50",
                           s=80, zorder=5)
            # gap 标注:最后一轮的差距越大越过拟合
            if train_auc and test_auc:
                gap = train_auc[-1] - test_auc[-1]
                ax.text(0.02, 0.02, f"Final gap (train-test): {gap:+.3f}",
                        transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="#FFF3E0" if abs(gap) > 0.1 else "#E8F5E9",
                                  edgecolor="none"))
            ax.legend(loc="lower right", fontsize=10)
        else:
            ax.text(0.5, 0.5, "(no eval history)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("AUC")
        ax.set_title("Learning Curve — AUC (gap = overfitting)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # ── 2) Learning curve: Logloss 随 boosting round ──
        ax = axes[0, 1]
        if hasattr(self, "_evals_result") and self._evals_result:
            train_ll = self._evals_result.get("train", {}).get("logloss", [])
            test_ll = self._evals_result.get("test", {}).get("logloss", [])
            rounds = list(range(1, len(train_ll) + 1))
            ax.plot(rounds, train_ll, color="#2196F3", linewidth=2, label="Train logloss")
            ax.plot(rounds, test_ll, color="#FF5722", linewidth=2, label="Test logloss")
            # 找 test logloss 开始上升的 round(过拟合起点)
            if len(test_ll) > 3:
                min_idx = int(np.argmin(test_ll))
                if min_idx < len(test_ll) - 1:
                    ax.axvline(min_idx + 1, color="#F44336", linestyle=":", alpha=0.7,
                               label=f"Test loss min @ round {min_idx + 1}")
                    ax.scatter([min_idx + 1], [test_ll[min_idx]], color="#F44336",
                               s=80, zorder=5)
            ax.legend(loc="upper right", fontsize=10)
        else:
            ax.text(0.5, 0.5, "(no eval history)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Log loss")
        ax.set_title("Learning Curve — Log loss (test rises = overfitting)",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # ── 3) CV 各折准确率(模型稳定性) ──
        ax = axes[1, 0]
        cv_scores = getattr(self, "_cv_scores", [])
        if cv_scores:
            x = np.arange(1, len(cv_scores) + 1)
            mean = np.mean(cv_scores)
            std = np.std(cv_scores)
            colors = ["#4CAF50" if s >= mean else "#FFA726" for s in cv_scores]
            ax.bar(x, cv_scores, color=colors, edgecolor="white", linewidth=1.5, width=0.65)
            ax.axhline(mean, color="#37474F", linestyle="--", linewidth=1.5,
                       label=f"Mean = {mean:.4f}")
            ax.axhline(mean + std, color="#90A4AE", linestyle=":", linewidth=1,
                       label=f"±1 std = {std:.4f}")
            ax.axhline(mean - std, color="#90A4AE", linestyle=":", linewidth=1)
            ax.axhline(0.5, color="#E53935", linestyle="-.", linewidth=1, alpha=0.6,
                       label="Random baseline (0.5)")
            # 数值标在柱子顶
            for xi, score in zip(x, cv_scores):
                ax.text(xi, score + 0.005, f"{score:.3f}", ha="center", va="bottom",
                        fontsize=9, color="#37474F")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Fold {i}" for i in x])
            ax.set_ylim(0.45, max(max(cv_scores) + 0.05, 0.65))
            ax.legend(loc="lower right", fontsize=9)
        else:
            ax.text(0.5, 0.5, "(no CV scores)", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Accuracy")
        ax.set_title("Cross-Validation Folds (TimeSeriesSplit)",
                     fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        # ── 4) 预测概率分布 by 真实类别(分得开 = 模型有区分力) ──
        ax = axes[1, 1]
        try:
            dtest = xgb.DMatrix(self.X_test)
            y_pred = self.xgb_model.predict(dtest)
            y_true = self.y_test.values
            pos_preds = y_pred[y_true == 1]
            neg_preds = y_pred[y_true == 0]
            bins = np.linspace(0, 1, 21)
            ax.hist(neg_preds, bins=bins, alpha=0.65, color="#E53935",
                    label=f"True class = 0 (n={len(neg_preds)})", edgecolor="white")
            ax.hist(pos_preds, bins=bins, alpha=0.65, color="#4CAF50",
                    label=f"True class = 1 (n={len(pos_preds)})", edgecolor="white")
            ax.axvline(0.5, color="#37474F", linestyle="--", alpha=0.6, label="Threshold = 0.5")
            # 计算两个分布的重叠区(用直方图近似)
            overlap = np.minimum(
                np.histogram(neg_preds, bins=bins, density=True)[0],
                np.histogram(pos_preds, bins=bins, density=True)[0],
            ).sum() * (bins[1] - bins[0])
            ax.text(0.02, 0.97,
                    f"Distribution overlap: {overlap:.2f}\n(lower = more separable)",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="#FFF3E0" if overlap > 0.55 else "#E8F5E9",
                              edgecolor="none"))
            ax.legend(loc="upper center", fontsize=9)
        except Exception as exc:
            ax.text(0.5, 0.5, f"(error: {exc})", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Predicted probability of class 1")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Distribution by True Class",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Model Training Diagnostics", fontsize=15, fontweight="bold", y=1.00)
        plt.tight_layout()
        out_path = "models/training_diagnostics.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  ✓ 已保存: {out_path}")

    def analyze_feature_importance(self):
        """分析特征重要性。"""
        print("\n[5/6] 分析特征重要性...")

        importance = self.xgb_model.get_score(importance_type="gain")
        feature_names = self.X.columns.tolist()

        rows = []
        for key, value in importance.items():
            if key in feature_names:
                rows.append({"feature": key, "importance": value})
                continue

            if key.startswith("f") and key[1:].isdigit():
                feature_index = int(key[1:])
                if 0 <= feature_index < len(feature_names):
                    rows.append({"feature": feature_names[feature_index], "importance": value})

        if not rows:
            pd.DataFrame(columns=["feature", "importance"]).to_csv(
                "models/feature_importance.csv", index=False
            )
            print("  ! 当前模型没有产生可用的特征重要性，已保存空表。")
            return

        importance_df = pd.DataFrame(rows).sort_values("importance", ascending=False)
        importance_df.to_csv("models/feature_importance.csv", index=False)

        print("\nTop 20 重要特征:")
        for _, row in importance_df.head(20).iterrows():
            print(f"  {row['feature']:40s} {row['importance']:10.2f}")

        map_features = importance_df[
            importance_df["feature"].str.contains("map|pick|h2h", case=False, na=False)
        ]
        player_features = importance_df[
            importance_df["feature"].str.contains("rating|adr|kast", case=False, na=False)
        ]
        momentum_features = importance_df[
            importance_df["feature"].str.contains(
                "previous|series|decider|comeback|streak", case=False, na=False
            )
        ]

        print("\n特征重要性汇总:")
        print(f"  地图特征:     {map_features['importance'].sum():.2f} ({len(map_features)} 维)")
        print(f"  选手特征:     {player_features['importance'].sum():.2f} ({len(player_features)} 维)")
        print(
            f"  Momentum特征: {momentum_features['importance'].sum():.2f} "
            f"({len(momentum_features)} 维)"
        )

    def save_models(self):
        """保存模型与元数据。"""
        print("\n[6/6] 保存模型...")

        self.xgb_model.save_model("models/xgboost_map_predictor.json")
        print("  ✓ XGBoost模型: models/xgboost_map_predictor.json")

        with open("models/feature_names.json", "w", encoding="utf-8") as f:
            json.dump(self.X.columns.tolist(), f, ensure_ascii=False, indent=2)
        print("  ✓ 特征名称: models/feature_names.json")

    def cross_validate(self):
        """仅在训练集上执行交叉验证。"""
        print("\n[额外] 交叉验证（仅训练集）...")

        requested_folds = int(self.config.get("evaluation", {}).get("cross_validation_folds", 5))
        cv_folds = min(requested_folds, len(self.X_train) - 1)

        if cv_folds < 2:
            print("  ! 训练集样本不足，跳过交叉验证。")
            return

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        classifier_params = dict(self.config["models"]["xgboost"])
        # sklearn wrapper 在交叉验证时强制走 CPU，避免设备不匹配警告。
        classifier_params["device"] = "cpu"

        clf = xgb.XGBClassifier(
            **classifier_params,
            eval_metric="logloss",
            random_state=42,
        )

        scores = cross_val_score(clf, self.X_train, self.y_train, cv=tscv, scoring="accuracy")
        self._cv_scores = list(scores)   # 存下来给诊断图用
        print(f"  交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        print(f"  各折得分: {[f'{score:.4f}' for score in scores]}")

    def run(self):
        """执行完整训练流程。"""
        print("\n" + "=" * 80)
        print("模型训练")
        print("=" * 80)

        self.load_data()
        training_cfg = self.config.get("training", {})
        if training_cfg.get("enable_feature_selection", True):
            self.select_features()
        else:
            print("\n[2/6] 自动特征选择已关闭")
        self.train_xgboost()
        if training_cfg.get("enable_calibration", True):
            self.calibrate_model()
        else:
            self.calibrator = None
            print("\n[3.5/6] 概率校准已关闭")
        results = self.evaluate_model()
        self.plot_evaluation_charts()
        self.analyze_feature_importance()
        self.cross_validate()
        # 诊断图需要 evals_result(train_xgboost 留下)和 _cv_scores(cross_validate 留下),
        # 所以放在 cross_validate 后面。
        self.plot_training_diagnostics()
        self.save_models()

        print("\n" + "=" * 80)
        print("✓ 模型训练完成！")
        print("=" * 80)
        print("\n模型性能:")
        print(f"  测试集准确率: {results['accuracy'] * 100:.2f}%")
        if results["roc_auc"] is None:
            print("  AUC: N/A")
        else:
            print(f"  AUC: {results['roc_auc']:.4f}")
        print("\n下一步: python hybrid_playoff_predictor.py")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
