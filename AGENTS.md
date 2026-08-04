# AGENTS.md

These instructions apply to the entire repository.

## Project Overview

This is a research fork of `verl-agent` whose main local work is VPR for
multi-step LLM reinforcement learning. The implemented VPR stack includes:

- dense, oracle-guided rewards for TicTacToe, Sudoku, Minesweeper, and Sokoban;
- Markovian per-step agent training;
- vanilla and state-group rollouts;
- VPR turn/state-group advantage estimation;
- outcome-GRPO, Turn-level PPO, and VinePPO baselines;
- mixed math-and-game DAPO training;
- Tau Bench VPR and outcome integrations; and
- in-domain and agentic OOD evaluation pipelines.

Preserve upstream `verl` behavior outside the requested scope. Prefer focused
changes over broad refactors.

## Repository Map

- `agent_system/environments/env_package/vpr_games/`
  - Game workers, managers, parsers, rewards, oracles, and mixed environments.
- `agent_system/environments/prompts/vpr_games.py`
  - VPR prompt templates and action-format variants.
- `agent_system/environments/env_package/tau_bench/`
  - Tau action validation, environment adapter, oracle, cache, and manager.
- `agent_system/environments/env_manager.py`
  - Environment registration and construction.
- `agent_system/multi_turn_rollout/rollout_loop.py`
  - Vanilla rollout and state-group candidate rollout.
- `agent_system/multi_turn_rollout/utils.py`
  - Batch adjustment and the `is_padding` contract.
- `gigpo/core_gigpo.py`
  - VPR, Turn-level PPO, and VinePPO advantage estimators.
- `verl/trainer/ppo/ray_trainer.py`
  - Estimator dispatch, masks, metrics, and evidence integration.
- `verl/trainer/config/vpr_*.yaml`
  - Executable defaults for the four VPR games.
- `verl/trainer/config/{dapo_vpr_mixed,tau_vpr,tau_outcome}.yaml`
  - Mixed and Tau experiment configs.
- `examples/vpr_games/`
  - Data preparation, training scripts, smoke tests, evaluation, and detailed
    documentation.
- `examples/dapo_trainer/`
  - Mixed math/game preparation and launch scripts.
- `examples/tau_bench/`
  - Tau installation, official-split data preparation, training, and evaluation.
- `tests/vpr_games/` and `tests/tau_bench/`
  - Primary regression suites.

When prose and executable behavior disagree, inspect the source, Hydra config,
launch script, and tests together. Do not assume README defaults are current.

## VPR Invariants

### Markovian game prompts

VPR game environments expose the current state, not the trajectory history.
`VPRBaseEnvironmentManager` enforces:

```text
env.history_length=0
```

Do not add prior turns to VPR game prompts or relax this check unless the user
explicitly requests a different experimental protocol.

### Action handling

- Game actions normally use `<action>...</action>`, with optional reasoning in
  `<think>...</think>`.
- The shared parser uses the final action block and handles documented aliases.
- Malformed, missing, out-of-range, and illegal actions must become controlled
  invalid actions; model output must not crash an environment worker.
- Grid coordinates are 1-indexed.
- Preserve native tool-call validation for Tau.

### Reward and estimator are independent

Keep these controls separate:

- `env.<game>.reward_mode` decides which rewards the environment emits.
- `algorithm.adv_estimator` decides how rewards become advantages.

The main VPR game path uses dense `oracle` rewards with
`adv_estimator: vpr`. The outcome baseline uses sparse `outcome` rewards with
`adv_estimator: grpo`. Estimator-specific snapshots, evidence, metrics, padding
rules, and loss masks must remain gated by the selected estimator.

Current reward scales differ by environment. Read the corresponding YAML,
worker, launch script, and tests before changing a constant.

### State-group rollout

State-group rollout must preserve the following:

1. All candidates in a group are generated from the same environment state.
2. Every candidate receives its own parsed action, reward, and metadata.
3. All eligible candidates may train the policy.
4. Exactly one candidate is committed to advance the environment.
5. The next group starts from the selected successor state.

`env.rollout.n` is the environment-level candidate count. Do not confuse it
with an inference engine's independent sequence count.

Preserve explicit group and row metadata, including `state_group_uid`,
candidate rank, selected status, turn index, trajectory ID, selection type, and
terminal fields. Use IDs rather than reconstructing groups from row order.

Equal-reward state groups contain no preference signal and are skipped. Any
change to this behavior must update advantage calculation, loss masking,
metrics, evidence, and tests together.

### Divisibility padding

`adjust_batch(mode="copy")` may duplicate rows only to satisfy distributed
batch divisibility. Duplicates are marked with `is_padding=true`.

For VPR, padding rows must not affect:

- normalization statistics;
- logged metrics;
- evidence;
- advantages; or
- gradients.

Do not infer padding from duplicate content or row position.

### Structured environment metadata

Metrics and training logic consume structured `info` fields. Preserve the
meaning and type of fields such as:

- `vpr_game`, `raw_action`, `parsed_action`, `parse_ok`, `illegal_action`;
- `is_action_valid`, `available_actions`, `vpr_reward`;
- `move_optimal`, `legal_non_oracle`, `oracle_valid_actions`, `oracle_tier`;
- `is_terminal`, `terminal_success`, `terminal_reason`; and
- environment-specific completion and oracle diagnostics.

`terminal_success` is the canonical success signal. Do not substitute a
third-party environment's native success flag without adapting its semantics.

## Environment Notes

- TicTacToe uses exact minimax for oracle actions. Random or OpenSpiel MCTS
  changes the opponent, not the oracle.
- Sudoku generation must remain deterministic and satisfy the configured blank
  count. Its oracle distinguishes forced/MRV, legal non-oracle, wrong-digit,
  and invalid-cell actions.
- Minesweeper uses posterior reasoning with the global mine count and reports
  degraded oracle inference explicitly. Success is reveal-based and does not
  require flagging every mine.
- Sokoban uses a search-based oracle. Keep snapshot/restore behavior, legal
  action semantics, and completion metrics consistent.
- Mixed training combines math, Sokoban, Sudoku, and Minesweeper. Configured
  trajectory counts must sum to the batch size, and per-domain metrics must
  remain separable.

## Tau Protocol

Tau experiments are protocol-sensitive. Preserve:

- the pinned Tau source commit and compatibility-patch checksum;
- the complete official Airline/Retail `train` split for training;
- deterministic fixed-domain, complete validation batches drawn from official
  `base` domains, with all dropped tail rows recorded;
- native tool-call/action validation and tool-call ID linkage;
- the DB/communicate terminal reward protocol;
- disabled user-simulator reasoning where required;
- cache-first exact-state expert action sets with single-flight generation;
- AWM multiset frequency semantics and Tau's existing deduplicated-set semantics;
- cache versioning and strict resume compatibility; and
- separate training and evaluation step limits.

`tau_vpr` uses state-group rollout. `tau_outcome` uses vanilla rollout and
trajectory-level outcome normalization. Do not route the outcome baseline
through the oracle/state-group path.

Tau training has no expert-success qualification gate. Protocol mismatches
should fail loudly. Do not silently accept source/split drift or reuse cache
records from a different protocol. Final reported evaluation should use Tau's
native runner; periodic in-process validation may reuse the training vLLM.

## Development Workflow

Before editing:

1. Run `git status --short`.
2. Preserve unrelated modified, deleted, and untracked files.
3. Read the nearest implementation, Hydra config, launch script, and tests.
4. Trace cross-cutting changes through:

```text
prompt -> parser -> worker -> manager -> rollout metadata
       -> reward manager -> advantage estimator -> loss mask -> metrics/evidence
```

For an environment change, update its worker/manager, config, prompt if
applicable, and focused tests. For an estimator or rollout change, verify
padding, skipped-row masks, trainer dispatch, metrics, and evidence.

Do not weaken validation or deterministic behavior merely to make a test pass.

## Tests

Run the smallest relevant test first:

```bash
python -m pytest tests/vpr_games/test_parser.py -q
python -m pytest tests/vpr_games/test_vpr_advantage.py -q
python -m pytest tests/vpr_games/test_mixed_vpr.py -q
python -m pytest tests/tau_bench/test_actions.py -q
python -m pytest tests/tau_bench/test_protocol.py -q
```

Then run the affected suites:

```bash
python -m pytest tests/vpr_games/ -q
python -m pytest tests/tau_bench/ -q
python -m pytest \
  tests/test_dapo_turn_reward.py \
  tests/test_math_reasoning_env.py \
  tests/test_agentic_eval_envs.py -q
```

Optional packages such as Ray, Torch, GEM, `gym_sokoban`, OpenSpiel, Tau, and
GPU inference runtimes may be unavailable locally. Report skips or missing
dependencies explicitly instead of hiding them with broad mocks.

For changed Python files:

```bash
ruff check <changed-files>
ruff format --check <changed-files>
```

Do not format the entire fork unless requested.

## Training and Evaluation

- Use `examples/vpr_games/prepare_data.py` to create game trigger datasets.
- Use scripts in `examples/vpr_games/vpr/`, `grpo/`, `turn_level_ppo/`, and
  `vineppo/` for matched training variants.
- Use `examples/vpr_games/smoke/` after changes to rewards, rollout, advantage,
  trainer, or distributed batching when GPUs are available.
- A successful training exit is insufficient if `smoke_verify.py` rejects the
  emitted evidence.
- Use evaluation entry points under `examples/vpr_games/eval/`.
- Preserve protocol manifests, source/model identity, resume checks, raw
  generations, and per-seed results.
- Prefer `DRY_RUN=1` when checking an evaluation command without launching a
  model.
- Never stop unrelated GPU processes.

Model, interpreter, dataset, and checkpoint paths in scripts are cluster
defaults. Make them overridable rather than replacing them with machine-specific
hard-coded paths.

## Generated Artifacts

Datasets, qualification caches, checkpoints, logs, smoke evidence, evaluation
outputs, and `runs/` contents are experiment artifacts. Do not commit, delete,
or rewrite them unless the user explicitly asks for artifact management.

## Definition of Done

Before handing off a VPR-related change:

- reward and invalid-action semantics are explicit and tested;
- seeds and reset behavior remain deterministic where promised;
- Markovian game prompts do not leak history;
- state-group candidates share one source state and exactly one advances it;
- padding and skipped equal-reward rows cannot produce gradients;
- reward mode and advantage estimator remain orthogonal;
- metrics still consume canonical structured fields;
- Tau protocol checks remain strict if Tau code changed;
- configs, launch scripts, and relevant docs agree; and
- focused tests were run, with unavailable GPU tests or dependency skips noted.
