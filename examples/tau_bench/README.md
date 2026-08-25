# Tau Bench Agentic OPD

This directory contains Tau Airline/Retail training plus in-process and native
evaluation for this research fork.

## Protocol

- Source: `/mnt/public2/yuanhuining/repos/tau2-bench`, pinned to commit
  `17e07b1da2bbc0cadfddeea36412686e0604127b` plus the checked-in optional-voice
  compatibility patch.
- Training tasks: the complete official `train` split, Airline 30 and Retail
  74. There is no expert-success qualification gate.
- Periodic validation draws from the official `base` split with one trial per
  task, but intentionally materializes only complete, fixed-composition batches.
  With `VAL_BATCH=16`, Airline-only uses 16 Airline slots and evaluates 48 of
  50 tasks; Airline plus Retail uses 5/11 slots and evaluates 50 plus 110 tasks.
  The manifest records the fixed quota and every dropped tail row.
- Student prompt: the exact instruction template from the pinned Tau2 native
  `LLMAgent`, rendered through Qwen's native ChatML function-calling format with
  the actual Tau tool schemas. Periodic validation uses the same prompt; final
  evaluation always uses Tau's native `LLMAgent` implementation.
- User simulator: `openrouter/qwen/qwen3.6-27b`, temperature 1, reasoning
  disabled.
- Expert: `deepseek/deepseek-v4-flash`, three independent requests per exact
  state. Concurrent requests for the same state use single-flight; the ordered
  K=3 multiset is reused from the run-local v6 cache for the remainder of that
  run. Legacy v5 deduplicated-set records are ignored.
- Duplicate expert actions are retained. Tool calls use exact canonical-count
  matching; messages are judged independently against all three teacher
  samples. Tau defaults to `TEACHER_REWARD_MODE=appearance` for its historical
  any-match reward. Set `frequency_weighted` to use the same soft consensus
  bonus as AWM/EnvScaler; `FREQUENCY_BONUS_SCALE` defaults to 0.5.
- If the expert emits parallel tool calls, Tau executes only the first. A
  student multi-call output remains invalid under the single-action protocol.
- Training/evaluation caps are 20/30 agent decisions.
- Terminal reward is Tau's deterministic DB component multiplied by
  COMMUNICATE when applicable; LLM-judged NL assertions are excluded.

## Setup

Tau requires Python 3.12 or newer.

```bash
PYTHON=/opt/venvs/verl-agent/bin/python \
bash examples/tau_bench/install_tau2.sh
export TAU2_DATA_DIR=/mnt/public2/yuanhuining/repos/tau2-bench/data
export OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
```

The installer keeps both source and Tau's dataset cache under the shared
`tau2-bench` checkout and installs it editable into the selected environment.

## Training

```bash
PYTHON=/opt/venvs/verl-agent/bin/python \
MODEL_PATH=<LOCAL_QWEN_MODEL> \
bash examples/tau_bench/run_tau_agentic_opd.sh

PYTHON=/opt/venvs/verl-agent/bin/python \
MODEL_PATH=<LOCAL_QWEN_MODEL> \
bash examples/tau_bench/run_tau_outcome.sh
```

Both launchers materialize a deterministic cyclic schedule from the official
train tasks. Generated data, TensorBoard logs, caches, and checkpoints live
under `runs/<UTC timestamp>/`. The expert cache defaults to
`$RUN_DIR/cache/teacher.jsonl`; pass `ORACLE_CACHE` only when deliberate
cross-run reuse is desired. Set `SMOKE=1` for a one-step, two-decision smoke.

Agentic OPD uses four student candidates per visited state, commits exactly one
uniformly among the highest-reward candidates, and masks equal-reward groups.
`algorithm.state_group` controls normalization, the absolute minimum number of
effective groups, and compact policy rows across Tau, AWM, EnvScaler, and VPR games.
Outcome uses four complete rollouts per task and trajectory-level GRPO.

## AWM periodic validation

The formal AWM Agentic OPD launcher calls the same in-process Tau adapter at step
0 and every 20 optimizer steps, using the training vLLM instance and sampling
parameters. Every worker keeps one domain for its lifetime, and every validation
batch uses the fixed domain quota recorded in the data manifest. A tail that
cannot fill that exact template is omitted.

```bash
# Default: 48 of 50 Airline base tasks (three complete 16-task batches)
bash examples/awm/train/run_agentic_opd.sh

# Airline 50 + Retail 110 (ten complete 5+11 batches)
TAU_VAL_DOMAINS=airline,retail \
bash examples/awm/train/run_agentic_opd.sh
```

On steps divisible by both save and validation frequency, the checkpoint is
written before validation. Formal AWM defaults are 200 steps, 64 tasks per
step, four candidate actions per state, save every 10, validate every 20, and
retain all checkpoints. The standalone smoke entry point is:

```bash
bash examples/awm/train/run_agentic_opd_smoke.sh
```

## Evaluation

The lightweight in-process evaluator defaults to fixed 5-Airline/11-Retail
complete batches from the official `base` pools and is configurable through
`VALIDATION_DOMAINS`. It is intended for periodic or diagnostic comparison, not
final complete-split reporting:

```bash
PYTHON=/opt/venvs/verl-agent/bin/python \
MODEL_SPECS_FILE=<MODEL_REGISTRY_TSV> \
bash examples/tau_bench/run_tau_eval.sh
```

Training does not launch an automatic full evaluation at its final step. For
full-split final reporting, manually run the separate native Tau runner. By
default it reserves physical GPU 0 for a local Qwen3.5-9B user simulator and
serves the evaluated model on every remaining GPU:

```bash
MODEL_SPECS_FILE=<MODEL_REGISTRY_TSV> \
DOMAINS="airline retail telecom-workflow" \
RUN_DIR=runs/tau_native_eval_final \
bash examples/tau_bench/run_tau_native_eval.sh
```

The local user model defaults to
`/mnt/public2/yuanhuining/models/Qwen3.5-9B` and is served by
`/opt/venvs/vllm-nightly-cu129/bin/vllm`. Thinking is enabled with Qwen3.5's
general-task sampling parameters: temperature 1.0, top-p 0.95, top-k 20,
min-p 0, presence penalty 1.5, and repetition penalty 1.0. Tau replays only its final text and structured tool calls to the model. Raw
response metadata, including reasoning content, remains in result artifacts but
is not replayed in later prompts. At least
two GPUs are required. `CUDA_VISIBLE_DEVICES` selects only agent GPUs and must
not include GPU 0; when omitted, all physical GPUs except GPU 0 are selected and
DP is derived automatically.

The local-user service uses a 65,536-token context and an 8,192-token output
budget. Truncated or empty generations are retried twice with deterministic
alternate seeds without retaining the rejected turn. The Qwen3 agent service
uses its native 40,960-token context. During terminal replay, an unknown tool
call is skipped only when its original tool result was explicitly marked as an
error, so the failed call cannot have changed environment state. Tau's native
checkpoint resume excludes infrastructure-error placeholders and reruns those
trials. Set `ALLOW_INFRASTRUCTURE_PROTOCOL_UPGRADE=1` once to resume a compatible
protocol-v4 run under this repair-only protocol-v6 migration.

The remote compatibility path remains available explicitly:

```bash
export DEEPSEEK_API_KEY=<DEEPSEEK_API_KEY>
MODEL_SPECS_FILE=<MODEL_REGISTRY_TSV> \
USER_SIMULATOR_MODE=remote \
USER_MODEL=deepseek/deepseek-v4-flash \
RUN_DIR=runs/tau_native_eval_remote_user \
bash examples/tau_bench/run_tau_native_eval.sh
```

The runner has one agent protocol: Tau's pinned native `LLMAgent` with
structured function calling. Native results are checkpointed in task shards and
resume completed trials. `NUM_TASKS=1 DOMAINS=airline` is
the smallest native smoke. Supported native domains are `airline`, `retail`,
`telecom`, and Tau2's workflow-policy variant `telecom-workflow`. Remote
DeepSeek models use `DEEPSEEK_API_KEY` with provider-native thinking disabled;
OpenRouter models retain their existing key and arguments.

## Metrics

- `episode/env/protocol_reward` and `episode/env/success_rate`: deterministic
  terminal task result.
- `episode/env/valid_action_rate`: schema-valid tool call or non-empty user
  message.
- `episode/env/transfer_tool_call_rate`: trajectories that invoked
  `transfer_to_human_agents`.
- `episode/env/transfer_handoff_rate`: trajectories that emitted Tau's exact
  fixed handoff message; `transfer_handoff_count` is its additive denominator.
- `episode/env/transfer_acknowledged_rate`: trajectories whose simulated user
  returned `###TRANSFER###`; `transfer_ack_failure_rate` counts a handoff without
  that marker, and `transfer_ack_success_rate_given_handoff` conditions only on
  trajectories that emitted the fixed handoff.
- `episode/env/decision_limit_rate`: trajectories force-closed at the configured
  agent-decision limit.
- `episode/env/oracle_hit_rate`: process-action match rate for VPR.
- `episode/env/oracle_cache_*`: cache lookups, hits, misses, single-flight
  waits, generated sets, and hit rate.
- `dapo/skipped_oracle_rate` and per-domain variants: oracle-candidate share
  among rows in fully skipped equal-reward groups.

The same environment metrics are logged under `val/` during periodic
validation. `val/env/trajectory_count` and per-domain trajectory counts are
full-validation totals summed across complete batches, not the mean batch size.
