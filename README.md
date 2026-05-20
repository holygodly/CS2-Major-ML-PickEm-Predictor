# CS2 Major 预测系统 (ML 版)

基于机器学习的CS2 Major预测系统。核心是 XGBoost 地图级胜率模型+Isotonic概率校准,配合v社瑞士轮规则和蒙特卡洛模拟,预测瑞士轮每队 3-0 / 晋级 / 0-3 概率并暴搜最优作业,也能模拟淘汰赛冠军概率。

[上一版](https://github.com/holygodly/CS2_Major_Swiss_System_Predictor)用纯ELO算实力,这一版把ELO换成了XGBoost学的地图级模型,能区分某队在不同地图上的强弱。哪个版本更好见仁见智吧，也欢迎ML大佬们来优化模型。

## 快速开始

### 1. 配置

瑞士轮预测,编辑 `config.yaml` 的 `swiss_stage`:

```yaml
swiss_stage:
  seeded_teams:          # 可不填。不填就从下面的首轮对阵自动反推(每场前者=种子1-8,后者=种子9-16)
    - 队伍1              # 想手动指定就按官方种子1-16顺序填满16支
    - ...
  round1_matchups:       # 8场首轮BO1,按场次顺序填,每场高种子写前面、低种子写后面
    - [队伍1, 队伍9]      # 第1场:队伍1=种子1,队伍9=种子9
    - ...
```

种子顺序只影响 R2 之后 Buchholz 配对的tiebreaker,ELO和特征都是按队名从数据自动提取的,跟种子无关。

淘汰赛预测,编辑 `config.yaml` 的 `tournament`:

```yaml
tournament:
  name: "赛事名称"       # 写不写无所谓
  map_pool: null         # 自动从数据提取,或手动指定 ['Ancient', 'Inferno', ...]
  quarterfinals:         # 根据赛程自己填,系统自动从数据提取 ELO 和特征,不用管 V 社积分
    - [队伍A, 队伍B]      # QF1
    - [队伍C, 队伍D]      # QF2
    - [队伍E, 队伍F]      # QF3
    - [队伍G, 队伍H]      # QF4
  semifinal_pairs: [[0, 3], [1, 2]]   # QF1胜者vsQF4胜者, QF2胜者vsQF3胜者
```

### 2. 运行预测

```bash
# 瑞士轮预测(主功能,顺带生成图表)
python run_swiss_ml_prediction.py

# 淘汰赛快速预测(基于ELO,速度快)
python run_quick_prediction.py

# 淘汰赛完整预测(用ML模型,较慢)
python run_full_prediction.py
```

### 3. 查看结果

结果保存在 `output/`:

- `swiss_ml_predictions.json` - 瑞士轮概率 + 作业推荐
- `charts/` - 可视化图表(晋级概率排名、结果分解、作业卡片、对阵热图)
- `playoff_predictions_quick.json` / `playoff_predictions_full.json` - 淘汰赛结果

## 文件结构

```
cs2-major-ml-predictor/
├── config.yaml                  # 配置(赛事、队伍、模型参数)
├── swiss_stage_config.yaml      # 瑞士轮种子/首轮赛程备用配置
├── requirements.txt
├── data_preparation.py          # 数据准备(原始数据 → 地图级数据集)
├── feature_engineering.py       # 特征工程
├── model_training.py            # 模型训练 + Isotonic 校准 + 诊断图
├── hybrid_playoff_predictor.py  # 核心预测器
├── veto_simulator.py            # Ban/Pick 模拟器
├── map_side_analyzer.py         # 地图 T/CT 侧分析
├── pickem_optimizer.py          # 作业暴搜(GPU/CPU)
├── swiss_simulator_gpu.py       # GPU 瑞士轮模拟
├── gpu_accelerator.py           # GPU 加速
├── run_swiss_ml_prediction.py   # 瑞士轮预测入口
├── run_quick_prediction.py      # 淘汰赛快速预测
├── run_full_prediction.py       # 淘汰赛完整预测
├── run_backtest_stages.py       # 历史 Major 回测
├── generate_charts.py           # 生成图表
├── data/                        # 数据集(流水线生成)
├── models/                      # 训练产物（model_training.py 生成，含 XGBoost + Isotonic 校准器）
├── output/                      # 预测输出
└── docs/major-rulebook.md       # Valve 官方规则手册
```

## 重新训练模型

爬好数据(见下面「数据来源」)后,依次跑:

```bash
# 1. 准备数据
python data_preparation.py

# 2. 特征工程
python feature_engineering.py

# 3. 训练模型
python model_training.py
```

## 数据来源

训练数据是我自己写的 HLTV 爬虫爬的,爬虫不方便公开,**大家可自行爬取**。数据全部来自 HLTV 比赛页(`hltv.org/matches/...`)。爬好的数据路径可在 `config.yaml` 的 `data.data_dir` 改,需要下面三个文件,字段格式如下,照着抓就行。

### 1. `results_all_matches.csv` — 比赛列表

一行一场比赛,从 HLTV results 列表页抓。

| 字段 | 说明 | 例 |
|---|---|---|
| `match_id` | HLTV 比赛 ID(URL 里那串数字) | `2380799` |
| `index` | 序号,自己生成即可 | `1` |
| `type` | 赛制 | `BO1` / `BO3` / `BO5` |
| `team1` / `team2` | 对阵双方队名 | `FaZe` |
| `team1_score` / `team2_score` | 大比分(赢了几张图) | `2` |
| `date` | 比赛日期 | `2026-05-17` |
| `url` | 比赛页完整链接 | `https://www.hltv.org/matches/...` |

### 2. `player_stats.csv` — 选手数据(模型主要靠这个)

一行 = 一个选手在某场某张图某个阵营的数据,一场比赛有很多行(选手 × 地图 × Both/T/CT),从比赛页 stats 表抓。

| 字段 | 说明 | 例 |
|---|---|---|
| `match_id` / `match_index` / `match_date` / `match_type` | 比赛基本信息 | |
| `team1` / `team2` | 对阵双方 | |
| `team` | 这名选手所属队 | `FaZe` |
| `player_id` / `player_name` | 选手 ID 和昵称 | `karrigan` |
| `map_name` | `All maps` 或具体地图 | `Dust2` |
| `side` | 阵营 | `Both` / `T` / `CT` |
| `kills` / `deaths` | 击杀 / 死亡 | |
| `plus_minus` | 净胜(+/-) | |
| `adr` | 场均伤害 | `77.3` |
| `kast` | KAST 百分比 | `83.3` |
| `rating` | HLTV Rating(**模型最看重的特征**) | `1.18` |
| `match_url` | 比赛页链接 | |

### 3. `match_details_lite.json` — 比赛详情(地图 / veto / 比分)

一个 dict,**key 是比赛 URL**,value 结构如下:

```jsonc
{
  "https://www.hltv.org/matches/2380799/...": {
    "match_id": "2380799", "index": "1", "type": "BO1",
    "team1": "-72c", "team2": "Lynn Vision",
    "team1_score": "0", "team2_score": "1", "date": "2026-05-17",
    "details": {
      "veto_steps": [                       // ban/pick 流程
        {"step": 1, "team": "-72c", "action": "removed", "map": "Inferno"}
        // action: removed / picked / leftover
      ],
      "maps": [                             // 每张实际打的地图
        {
          "map_name": "Dust2",
          "team1": "-72c", "team2": "Lynn Vision",
          "picker": "Unknown",              // 谁选的这张图
          "score": "6 - 13",                // 该图最终比分
          "winner": "team2",
          "half_score": "(4:8;2:5)",        // 上下半场
          "half1_t_ct": "4:8", "half2_t_ct": "2:5",
          "team1_half1_side": "T",  "team2_half1_side": "CT",   // 半场阵营,用于 T/CT 侧分析
          "team1_half2_side": "CT", "team2_half2_side": "T"
        }
      ],
      "total_maps": 1
    }
  }
}
```

模型用得最多的:`player_stats.csv` 的 `rating` / `adr` / `kast`,以及 `match_details` 里每张图的 `score` / `winner` / `picker` / 半场阵营(算 T/CT 侧优势)。其它字段没爬到给默认值也能跑,但 rating 和地图比分这两块最好齐全。

## 预测原理

1. **ELO 评分**:从历史比赛动态计算,含时间衰减和自适应 K 因子(用于淘汰赛快速 baseline 和特征初始化)
2. **特征工程**:每张地图构 ~40 维特征 —— 地图胜率、H2H、选手 Rating、近期状态、阵容稳定性及其差值,全部带 30 天半衰期时间衰减 + Beta 平滑
3. **XGBoost 模型**:预测单张地图胜率(核心),自动调正则 + early stopping + 时间加权 + BO1 上采样
4. **Isotonic 校准**:5 折 OOF 拟合,修正概率偏差
5. **Ban/Pick 模拟**:Softmax 概率化模拟真实 Veto + T/CT 侧优势
6. **蒙特卡洛模拟**:跑 50 万次瑞士轮,统计战绩概率分布
7. **作业 暴搜**:在模拟结果上暴搜约 1000 万种组合,挑命中率最高的

## 回测结果

目前的作业规则是:要选 10 个队,分三组、不能重复 

- **2 个 3-0**(三胜零负)
- **6 个晋级**(3-1 或 3-2 出线)
- **2 个 0-3**(三连败出局)

命中 ≥5 个算过线。系统在几十万次模拟结果上暴搜全部约1000万种合法组合(`C(16,6)·C(10,2)·C(8,2)`),挑"命中 ≥5"概率最高的那套,**这个概率就是预期通过率**。

用 2025 年底那届 Major 回测(只用赛前数据训练),最优作业的预期通过率:

- 第一阶段:约 54%(本算法最后命中 2/10,没过)
- 第二阶段:约 42%(本算法最后命中 5/10,过了)

预期通过率是模型自己估的概率;真实一届就是一次抽样,方差很大,所以会出现"预期高的那次反而没过"。第一阶段是开赛阶段,爆冷最多,哪怕最优解也就五成出头。反观我当时的ELO算法，第一阶段和第二阶段都过了，哪种办法好大家见仁见智吧。。。训练集 5 折交叉验证准确率约 0.56,地图级预测做到这感觉差不多到顶了，欢迎ML大佬的建议。

## 注意事项

- 队伍名称必须与数据中的名称完全一致,否则查不到该队特征,默认按 0.5 算
- 地图池自动从数据中提取,换图只改 config
- 支持 GPU 加速(需要 CUDA),无显卡自动回退 CPU
- 瑞士轮前几轮 BO1,生死局/晋级局 BO3,系统自动判定
- Windows 中文环境若报 GBK 编码错,加 `PYTHONUTF8=1` 运行
- 项目纯整活，千万不要当真，预测结果仅供娱乐参考，别拿去赌钱啥的！！祝大家作业都必过！
