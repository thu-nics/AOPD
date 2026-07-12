# VPR 游戏环境（vpr_tictactoe / vpr_sudoku / vpr_minesweeper / vpr_sokoban）

本目录提供把 VPR 论文的四个推理游戏环境集成进 verl-agent 的训练/评测脚本。
环境遵循 VPR 的 **Markovian 单步训练范式**：每一步模型只看到“当前棋盘状态”（不拼接历史），
并对每个动作给出 **稠密 oracle 奖励**。训练用标准 GRPO，本地模型为
`/mnt/project_rlinf/yuanhuining/models/Qwen3-4B`。

---

## 一、实现了什么

### 1. 四个环境

| 环境名 | 来源 | 默认配置 | Oracle（最优动作）判定 |
|--------|------|----------|------------------------|
| `vpr_tictactoe` | 本仓库直接实现 | 3×3，对手 `random`，`max_steps=9` | 精确 minimax（带 α-β 剪枝）算出最优动作集合 |
| `vpr_sudoku` | 封装 `gem` 库 | 9×9 / **40 个空格**，`terminate_on_wrong_digit=True` | 对照唯一解 O(1) 查表：`solution[r][c]==digit` |
| `vpr_minesweeper` | 封装 `gem` 库 | 5×5 / 5 雷，`max_steps=25` | 后验概率 oracle（见下） |
| `vpr_sokoban` | 封装现有 `gym_sokoban` 环境 | 6×6 / 1 箱子，`max_steps=15` | BFS 最短解路径的首步动作集合 |

- **坐标统一 1-indexed**；动作格式统一 `<think>可选推理</think><action>...</action>`。
- **稠密奖励约定**：oracle 最优动作 `+1.0`，合法但非最优 `0.0`，非法/无法解析/越界 `-1.0`（可配 `invalid_penalty`）。
- 每个 `step()` 返回标准 info（含 `env_name/step/max_steps/raw_action/parsed_action/parse_ok/illegal_action/available_actions/vpr_reward/terminal_success/terminal_reason` 等），并由 manager 注入 `is_action_valid`。

**各环境关键语义：**

- **TicTacToe**：非法/已占格动作 → 终止 + 惩罚；胜负平局终止。额外 info：`game_result/oracle_valid_actions/opponent_action`。
- **Sudoku**：`clues=40` 被当作“**必须正好 40 个空格**”的硬约束——GEM 在保持唯一解时常常挖不满 40（实测约 58% 的种子原生不足），适配器用**确定性派生种子**重试直到正好 40 格，凑不到才 `ValueError`（绝不静默返回错的格数），并保证同种子 / GRPO 组内副本棋盘一致。错填默认终止（`terminate_on_wrong_digit=True`）。额外 info：`num_blanks_remaining/completion_rate`。
- **Minesweeper**：后验 oracle = 对“与当前已揭示信息一致 + 满足全局雷数”的所有布雷构型枚举（前沿连通分量分解 + 词典序枚举，预算 `N_max=50000`，超预算退化为局部推断并置 `oracle_degraded`）。`reveal` 后验雷概率最小的格 = oracle；`flag` 仅当后验恰好为 1（整数判定）= oracle；取消插旗 = 合法非 oracle（0.0）。**成功标准被改为 reveal-only**：揭开所有非雷格即通关（**不要求插旗所有雷**，覆盖 GEM 原生 flag-all），踩雷判失败（reward 0、reason `mine_hit`）。GEM 首次点击安全保留，首步 reveal 恒 `+1.0`。额外 info：`posterior_min_prob/posterior_prob_for_action/oracle_valid_actions/completion_rate/oracle_degraded`。

### 2. 共享基础设施

- `VPRBaseEnvironmentManager`：初始化时强制 `history_length=0`（非 0 直接 `ValueError`）；返回 `{"text","image","anchor"}`；`success_evaluator()` 读 `terminal_success`（不是 GEM 的 `won`）。
- 共享解析器 `vpr_games/common/parser.py`：取最后一个 `<action>` 块、别名展开（`open/click→reveal`、`mark→flag`）、空/缺标签哨兵、永不崩溃；非法动作在进入 GEM 前被拦截。
- 在 `make_envs()` 注册四个环境；训练池 `env_num=train_batch_size, group_n=rollout.n`，验证池 `group_n=1, seed+1000`。

### 3. VPR 训练管线（`algorithm.adv_estimator=vpr`）

- **逐 turn 归一化的 advantage**：对每个 turn 位置 t，用该 batch 中到达过 t 步的所有样本的奖励做 `(r_t − mean_t)/(std_t+ε)`；当同位置样本 < 4 时回退为整 batch 归一化。
- **可整除 padding 不污染统计**：`adjust_batch(mode="copy")` 为凑 DP 整除会复制行，这些行被标 `is_padding`，在归一化统计、日志 metric、证据、loss 中全部排除（padded 行 advantage/response_mask 置 0）。
- **标准 outcome 奖励**：终止步按 `outcome_reward_scale`（默认 +1.0 成功、0 否则）叠加，仅在终止步出现；可设 0 关闭。

### 4. 配置文件

`verl/trainer/config/vpr_{tictactoe,sudoku,minesweeper,sokoban}.yaml`，均 `defaults: [ppo_trainer, _self_]` 继承基础配置，只覆盖 `env`（`env_name/history_length=0/max_steps/各游戏参数`）和 `algorithm`（`adv_estimator: vpr`、`vpr.outcome_reward_scale`）。

---

## 二、如何训练

### 前置

- 模型：`/mnt/project_rlinf/yuanhuining/models/Qwen3-4B`
- Python（含 ray/torch/vllm/gem 的环境）：`/opt/venv/verl-agent/bin/python`
- 安装 gem：`/opt/venv/verl-agent/bin/python -m pip install 'git+https://github.com/axon-rl/gem.git'`

### 步骤 1：准备数据（每个环境一次）

```bash
PY=/opt/venv/verl-agent/bin/python
$PY examples/vpr_games/prepare_data.py --env-name vpr_sudoku --train-size 8 --val-size 1
# 输出到 examples/vpr_games/data/vpr_sudoku/{train,test}.parquet
# 数据本身只是触发样本（prompt 由环境在 rollout 时动态生成），train-size/val-size 控制条数
```

`--env-name` 取 `vpr_tictactoe | vpr_sudoku | vpr_minesweeper | vpr_sokoban`。

### 步骤 2：训练

**最简方式——直接跑 smoke 脚本**（2 步训练、2×H100、自带证据校验）：

```bash
bash examples/vpr_games/smoke/grpo_tictactoe_smoke.sh
bash examples/vpr_games/smoke/grpo_sudoku_smoke.sh
bash examples/vpr_games/smoke/grpo_minesweeper_smoke.sh
bash examples/vpr_games/smoke/grpo_sokoban_smoke.sh
# 模型/解释器路径可覆盖：MODEL_PATH=... PYTHON=... bash examples/vpr_games/smoke/grpo_sudoku_smoke.sh
```

**通用方式——直接调 main_ppo**（按需改超参）：

```bash
/opt/venv/verl-agent/bin/python -m verl.trainer.main_ppo \
    --config-name vpr_tictactoe \
    data.train_files=examples/vpr_games/data/vpr_tictactoe/train.parquet \
    data.val_files=examples/vpr_games/data/vpr_tictactoe/test.parquet \
    data.train_batch_size=2 data.val_batch_size=1 \
    data.max_response_length=64 \
    actor_rollout_ref.model.path=/mnt/project_rlinf/yuanhuining/models/Qwen3-4B \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    env.seed=0 env.rollout.n=2 env.max_steps=9 \
    algorithm.adv_estimator=vpr \
    trainer.total_training_steps=2 trainer.n_gpus_per_node=2 trainer.nnodes=1 \
    trainer.logger=[console]
```

常用旋钮：
- `--config-name vpr_<env>`：选环境（已含该环境默认值）。
- `data.train_batch_size` × `env.rollout.n` = 训练并行的 Ray 环境/actor 数（GRPO 组大小 = `rollout.n`）。
- `env.max_steps`：每个 episode 最大步数（TicTacToe 9 / Minesweeper 25 / Sudoku 视需要）。
- `algorithm.vpr.outcome_reward_scale`：终止 outcome 奖励权重（默认 1.0，设 0 关闭）。
- Sudoku 若想让未训练模型也能跑出多步轨迹：`+env.sudoku.terminate_on_invalid_parse=false` 且把 `data.max_response_length` 提到 ≥256（让模型有空间输出 `<action>` 标签）。
- 正式训练把 `trainer.total_training_steps`、`data.train_batch_size`、`env.rollout.n` 调大，`trainer.save_freq` 设为正数保存 checkpoint。

> 注意：所有 VPR 配置强制 `env.history_length=0`；若误传非 0 会在 manager 初始化时直接 `ValueError`（这是设计的防呆）。

---

## 三、如何评测（eval / validation）

训练期间的 validation 内建在 PPO 循环中；完整 checkpoint 评测使用本页第九节的 `eval_in_domain_all.sh`：

- `data.val_files`：验证集（由 `prepare_data.py --val-size` 生成）。
- `trainer.test_freq=N`：每 N 个训练步运行一次；`trainer.val_before_train` 控制训练前验证。
- 验证使用独立环境池：`group_n=1`、种子 `env.seed+1000`。
- 正式脚本使用采样评测：`temperature=1.0`、`top_p=1.0`、`top_k=-1`。
- 重点指标包括各环境的 `success_rate`、`completion_rate`、`valid_action_rate` 及算法指标。

---

## 四、自测与证据校验

- 单元测试：
  ```bash
  /opt/venv/verl-agent/bin/python -m pytest tests/vpr_games/ -q
  # 含 ray/torch/gem 的环境约 210 passed；纯系统 python3（无 ray）约 181 passed / 29 skipped
  ```
- smoke 证据校验：smoke 脚本结尾会用 `smoke_verify.py` 严格校验本次训练发出的证据
  （`VPR_SMOKE_EVIDENCE` 指定的 evidence JSON + 训练日志），核对每步奖励一致性、逐 turn
  advantage 重算、多步轨迹、prompt 局部性（无历史泄漏）、padding 已排除等。`rc=0` 即通过。
  其输入是项目自身管线产物（可信输入）。

---

## 五、目录速览

```
examples/vpr_games/
|-- prepare_data.py             # 所有训练和评测入口共享的数据生成器
|-- eval_in_domain_all.sh       # 标准 in-domain 评测入口
|-- summarize_in_domain_eval.py # 原始评测结果聚合
|-- smoke/                      # 四个 GRPO smoke 脚本及 smoke_verify.py
|-- grpo/                       # outcome-reward GRPO 基线
|-- vpr/                        # VPR 正式训练脚本
|-- vineppo/                    # VinePPO 正式训练脚本
|-- turn_level_ppo/             # Turn-level PPO 正式训练脚本
`-- data/<env>/                 # 生成的数据（gitignored）
```

各训练脚本默认值以对应文件为准，并可通过同名环境变量覆盖。训练产物默认写入仓库根目录的
`runs/<timestamp>/`；smoke 日志写入 `examples/vpr_games/smoke/smoke_logs/`。
另外 `grpo_<env>_outcome.sh`（见下）是“标准 GRPO + outcome 奖励”基线脚本，
`tictactoe/mcts_opponent.py` 是 OpenSpiel MCTS 对手封装，
`common/rewards.py` 是共享的胜负 outcome 奖励函数。

---

## 六、两种训练范式：VPR vs. outcome-reward + 标准 GRPO

`reward_mode`（环境发什么奖励）与 `algorithm.adv_estimator`（怎么算 advantage）是**正交**的两个旋钮：

| 维度 | VPR（过程监督，默认） | outcome + 标准 GRPO（结果监督基线） |
|------|----------------------|------------------------------------|
| `algorithm.adv_estimator` | `vpr` | `grpo`（verl 原生 `compute_grpo_outcome_advantage`，组内归一化） |
| `env.<game>.reward_mode` | `oracle`（稠密逐步 oracle 奖励：最优 +1 / 合法 0 / 非法 -1） | `outcome`（仅终局给分：赢 +1 / 输 -1 / 平局·超时 0） |
| advantage | 每个 turn 位置单独归一化，同位置 <4 时回退整 batch | 整条轨迹 episode 级标量在同 prompt 组内 `(r-mean)/std` |
| 终止 outcome bonus | VPR 估计器内按 `outcome_reward_scale` 叠加（VPR 专属） | 不适用（结果本身就是奖励） |
| 可整除 padding 行 | 从统计/metric/evidence/loss 中排除（VPR 专属） | 维持 verl 原生行为（未排除） |

要点：
- 切到 `adv_estimator=grpo` 后，**完全不走** VPR 的任何代码（`compute_vpr_turn_level_advantage`、证据采集、padding 排除、loss_mask 置零都 gated 在 `adv_estimator=='vpr'`）。
- `reward_mode` 默认全部为 `oracle`，所以默认行为 = 原 VPR，不受影响。
- 正式训练脚本（VPR 四个环境；outcome 基线覆盖四个环境，超参完全对齐，只差 adv_estimator 与 reward_mode）：
  ```bash
  # VPR（过程监督，默认范式）
  bash examples/vpr_games/vpr/vpr_tictactoe.sh
  bash examples/vpr_games/vpr/vpr_sudoku.sh
  bash examples/vpr_games/vpr/vpr_minesweeper.sh
  bash examples/vpr_games/vpr/vpr_sokoban.sh
  # outcome + 标准 GRPO（结果监督基线）
  bash examples/vpr_games/grpo/grpo_tictactoe_outcome.sh
  bash examples/vpr_games/grpo/grpo_sudoku_outcome.sh
  bash examples/vpr_games/grpo/grpo_minesweeper_outcome.sh
  # 同名环境变量可覆盖：TRAIN_STEPS / TRAIN_BATCH / ROLLOUT_N(=组大小) / VAL_BATCH /
  #   MAX_RESP / ENABLE_THINKING / USE_KL / KL_COEF / SAVE_FREQ / GPU_MEM_UTIL / RUN_DIR ...
  ```
  各 outcome 胜负映射（`common/rewards.py:outcome_reward`）：非终止步 0；`terminal_success` 终止 +1；
  中性终止（超时/步数耗尽/`already_done`）0；其余终止（踩雷/错填/非法动作…）-1。

---

## 七、TicTacToe 对手与角色（random / MCTS、X/O 可配）

`vpr_tictactoe` 的 `env.tictactoe` 配置：

- `opponent: random | mcts`（默认 `random`）。
  - `mcts` 用 **OpenSpiel 的 C++ MCTS**（`MCTSBot` + `RandomRolloutEvaluator`）作对手，需安装：
    ```bash
    /opt/venv/verl-agent/bin/python -m pip install open_spiel
    ```
    OpenSpiel 仅在 `opponent=mcts` 时**惰性导入**（软依赖）。参数在 `env.tictactoe.mcts`：
    `max_simulations`（默认 1000）、`uct_c`（2.0）、`rollout_count`（1）。
  - oracle **始终是 minimax**（不随对手类型改变）；MCTS 只用于对手落子。
- `agent_player: X | O`（默认 `X`）：策略模型控制的棋子。设 `O` 时对手（X）在 `reset()` 先手落子；
  minimax oracle、胜负判定都按所选棋子计算。
- 实现：对手落子（random/mcts）与胜负在 `tictactoe/game.py`；MCTS 通过同步一份 pyspiel `tic_tac_toe`
  状态来落子（每步双写棋盘+pyspiel，避免重建走子顺序），见 `tictactoe/mcts_opponent.py`。

示例（MCTS 对手、模型执 O）：
```bash
bash examples/vpr_games/grpo/grpo_tictactoe_outcome.sh  # 末尾追加 Hydra 覆盖：
#   env.tictactoe.opponent=mcts env.tictactoe.agent_player=O env.tictactoe.mcts.max_simulations=1000
```

---

## 八、环境 Prompt 模板

模板定义于 `agent_system/environments/prompts/vpr_games.py`，由各 manager 的 `build_text_obs` 填充
`{board}/{grid}/{blank_cells}/{unrevealed_cells}/{flagged_cells}` 等占位符（其中棋盘由各 worker 的
`_render*` 渲染并经 `info["observation"]` 传入）。

**TicTacToe**（`{mark}`/`{opp}` 反映所选棋子）：
```
You are playing Tic-Tac-Toe. You play as {mark}, your opponent plays as {opp}.

{board}

Choose one of the legal cells listed above.
Respond with your chosen cell number inside <action> tags.
You may reason briefly in <think>...</think> before your answer.
Your final answer must be: <action>CELL_NUMBER</action>
```
`{board}` 示例：
```
TicTacToe board (you=X, opponent=O):
 1 | 2 | 3
---+---+---
 4 | X | 6
---+---+---
 7 | 8 | 9
Legal cells: 1, 2, 3, 4, 6, 7, 8, 9
```

**Sudoku**：
```
You are solving a Sudoku puzzle. Fill in the blank cells (shown as .) using digits 1-9.
Each row, column, and 3×3 box must contain digits 1-9 exactly once.

Current grid:
{grid}

Blank cells: {blank_cells}

Choose one blank cell and one digit to fill it.
Format: <action>ROW COL DIGIT</action>  (rows and columns are 1-indexed, e.g. <action>3 5 7</action>)
You may reason briefly in <think>...</think> before your answer.
```
`{blank_cells}` 为空白格坐标列表（截断到 20 个）；`{grid}` 为带 C1–C9/R1–R9 表头、空格显示为 `.` 的 9×9 网格。

**Minesweeper**：
```
You are playing Minesweeper on a {rows}×{cols} board with {mines} mines.
Legend: . = hidden, F = flagged, numbers = revealed (count of adjacent mines)

Current board:
{board}

Unrevealed cells: {unrevealed_cells}
Flagged cells: {flagged_cells}

Choose one action:
  reveal ROW COL  — reveal a hidden cell
  flag ROW COL    — toggle flag on a hidden cell
Rows and columns are 1-indexed.

Format: <action>ACTION ROW COL</action>  (e.g. <action>reveal 2 3</action> or <action>flag 1 4</action>)
Aliases: open/click → reveal, mark → flag
You may reason briefly in <think>...</think> before your answer.
```
`{unrevealed_cells}`/`{flagged_cells}` 各截断到 15 个。

---

## 九、标准 In-domain 评测

统一入口保留在 `examples/vpr_games/eval_in_domain_all.sh`。默认依次评测 Sokoban、Sudoku 和
Minesweeper 的 Base/VPR/GRPO/VinePPO checkpoint，并对每个模型运行 5 个环境 seed、每个 seed
100 局：

```bash
bash examples/vpr_games/eval_in_domain_all.sh
```

每次未指定 `RUN_DIR` 时会创建 `runs/eval_<UTC timestamp>/`。目录包含：

- `protocol.env` / `protocol.sha256`：完整采样、环境和 checkpoint 协议；恢复时不一致会直接报错。
- `source.env` / `source_resume_*.env`：首次及恢复启动的 Git revision、dirty 状态和命令。
- `<task>/<model>/seed_<seed>/raw/*.jsonl`：逐 turn 原始生成。
- `<task>/<model>/seed_<seed>/raw/*.metrics.json`：单次评测指标。
- `summary.json` / `summary.csv` / `summary.md`：完成结果的逐 run 表和模型聚合表。

恢复同一评测时显式复用目录，已完成且协议一致的 `.done` 项会跳过：

```bash
RUN_DIR=$(pwd)/runs/eval_20260712T120000 bash examples/vpr_games/eval_in_domain_all.sh
```

主要覆盖项：

```bash
TASK_FILTER=sokoban MODEL_FILTER=vpr_sokoban ENV_SEEDS="0 100 200 300 400" VAL_GAMES=100 TEMPERATURE=1.0 TOP_P=1.0 TOP_K=-1 bash examples/vpr_games/eval_in_domain_all.sh
```

`ENV_SEEDS` 设置 `env.seed`；validation worker 实际使用该值加 1000 作为起始地图 seed。`GENERATION_SEED` 默认为空（使用推理引擎 RNG），设为 `env` 时逐
run 使用对应环境 seed，也可以设为固定非负整数。`MIN_P` 默认为空。模型和 checkpoint 路径可通过
`MODEL_PATH`、`VPR_SOKOBAN_CKPT`、`GRPO_SUDOKU_CKPT` 等同名环境变量覆盖。

`DRY_RUN=1` 只生成数据、协议和命令，不启动模型；`FORCE=1` 在协议一致的前提下重跑已有结果。
